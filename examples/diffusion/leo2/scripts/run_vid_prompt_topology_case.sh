#!/usr/bin/env bash
# Run one reproducible Leo2 native-inference topology case on one 8-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
KEEPALIVE_DIR=/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/.cursor/skills/taiji-node-env-bootstrap/scripts

: "${LEO2_CASE_NAME:?expected non-empty LEO2_CASE_NAME}"
: "${LEO2_CASE_CSV:?expected an absolute CSV path in LEO2_CASE_CSV}"
: "${LEO2_CASE_OUTPUT_ROOT:?expected an absolute output path in LEO2_CASE_OUTPUT_ROOT}"
: "${LEO2_CKPT_DIR:?expected LEO2_CKPT_DIR}"
: "${LEO2_ASSETS_BASE:?expected LEO2_ASSETS_BASE}"

PYTHON_BIN="${LEO2_RUNTIME_PYTHON:-/root/.local/leo2-runtime-py312-torch271-cu129/bin/python}"
EXPECTED="${LEO2_CASE_EXPECTED:-4}"
INFER_STEPS="${LEO2_INFER_STEPS:-50}"
DP_SHARD="${LEO2_CASE_DP_SHARD:-8}"
CP="${LEO2_CASE_CP:-8}"
EP="${LEO2_CASE_EP:-1}"
TP="${LEO2_CASE_TP:-1}"
KEEPALIVE_DURING_LOAD="${LEO2_CASE_KEEPALIVE_DURING_LOAD:-1}"
OUTPUT_DIR="${LEO2_CASE_OUTPUT_ROOT}/${LEO2_CASE_NAME}"
RUN_LOG="${OUTPUT_DIR}/run.log"
GPU_CSV="${OUTPUT_DIR}/gpu_metrics.csv"
CASE_ENV="${OUTPUT_DIR}/case.env"

for pair in \
  "LEO2_CASE_EXPECTED:${EXPECTED}" \
  "LEO2_INFER_STEPS:${INFER_STEPS}" \
  "LEO2_CASE_DP_SHARD:${DP_SHARD}" \
  "LEO2_CASE_CP:${CP}" \
  "LEO2_CASE_EP:${EP}" \
  "LEO2_CASE_TP:${TP}"; do
  name=${pair%%:*}
  value=${pair#*:}
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || { echo "expected a positive integer for ${name}, got ${value@Q}" >&2; exit 2; }
done
[[ "${LEO2_CASE_CSV}" == /* && -f "${LEO2_CASE_CSV}" ]] \
  || { echo "expected an existing absolute CSV path, got ${LEO2_CASE_CSV@Q}" >&2; exit 2; }
[[ "${LEO2_CASE_OUTPUT_ROOT}" == /* ]] \
  || { echo "expected an absolute LEO2_CASE_OUTPUT_ROOT, got ${LEO2_CASE_OUTPUT_ROOT@Q}" >&2; exit 2; }
[[ "${KEEPALIVE_DURING_LOAD}" == 0 || "${KEEPALIVE_DURING_LOAD}" == 1 ]] \
  || {
    echo "expected LEO2_CASE_KEEPALIVE_DURING_LOAD to be 0 or 1, got ${KEEPALIVE_DURING_LOAD@Q}" >&2
    exit 2
  }
[[ -x "${PYTHON_BIN}" ]] \
  || { echo "expected executable Leo2 runtime python, got ${PYTHON_BIN@Q}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] \
  || { echo "refusing to reuse topology output directory: ${OUTPUT_DIR}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}"

stop_watchdog() {
  local pid
  if [[ -f /root/keepalive_watchdog.pid ]]; then
    pid=$(cat /root/keepalive_watchdog.pid)
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "watchdog pid ${pid} did not stop before benchmark ${LEO2_CASE_NAME}" >&2
      return 1
    fi
  fi
}

stop_burner() {
  local pids=()
  mapfile -t pids < <(pgrep -f '/root/gpu_keepalive.py$' || true)
  ((${#pids[@]} == 0)) || kill -TERM "${pids[@]}" 2>/dev/null || true
}

start_burner() {
  KEEPALIVE_PYTHON=/opt/conda/envs/ff/bin/python \
    bash "${KEEPALIVE_DIR}/keepalive_launch.sh" >/root/leo2_case_keepalive.log 2>&1
}

restore_burner() {
  stop_burner
  start_burner || {
    echo "failed to restore keepalive after ${LEO2_CASE_NAME}; see /root/leo2_case_keepalive.log" >&2
    return 1
  }
}

monitor_gpu() {
  echo 'epoch,gpu,utilization_pct,memory_used_mib'
  while true; do
    epoch=$(date +%s)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits \
      | awk -F, -v epoch="${epoch}" \
          '{gsub(/ /, "", $0); print epoch "," $1 "," $2 "," $3}'
    sleep 1
  done
}

stop_watchdog
if [[ "${KEEPALIVE_DURING_LOAD}" == 1 ]]; then
  start_burner
else
  stop_burner
fi
monitor_gpu >"${GPU_CSV}" &
MONITOR_PID=$!

cleanup() {
  local original_status=$?
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  restore_burner || original_status=1
  exit "${original_status}"
}
trap cleanup EXIT INT TERM

{
  echo "case_name=${LEO2_CASE_NAME}"
  echo "created_epoch=$(date +%s)"
  echo "hostname=$(hostname)"
  echo "git_head=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "csv=${LEO2_CASE_CSV}"
  echo "csv_sha256=$(sha256sum "${LEO2_CASE_CSV}" | awk '{print $1}')"
  echo "expected=${EXPECTED}"
  echo "image_size=464x848"
  echo "num_frames=121"
  echo "infer_steps=${INFER_STEPS}"
  echo "dp_shard=${DP_SHARD}"
  echo "cp=${CP}"
  echo "ep=${EP}"
  echo "tp_requested=${TP}"
  echo "keepalive_during_load=${KEEPALIVE_DURING_LOAD}"
} >"${CASE_ENV}"

export ASSETS_BASE="${LEO2_ASSETS_BASE}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HOST_GPU_NUM=8
export HOST_NUM=1
export PYTHONPATH="${REPO_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset LD_LIBRARY_PATH

started_epoch=$(date +%s)
echo "started_epoch=${started_epoch}" >>"${CASE_ENV}"
set +e
bash "${SCRIPT_DIR}/native_t2v.sh" \
  --testsets "${LEO2_CASE_CSV}@@_first_n_=${EXPECTED}" \
  --sample-save-base "${OUTPUT_DIR}/samples" \
  --max-sample-batches "${EXPECTED}" \
  --dp-shard "${DP_SHARD}" \
  --context-parallel-size "${CP}" \
  --expert-model-parallel-size "${EP}" \
  --tensor-model-parallel-size "${TP}" \
  >"${RUN_LOG}" 2>&1 &
INFERENCE_PID=$!

burner_stopped=$((1 - KEEPALIVE_DURING_LOAD))
while kill -0 "${INFERENCE_PID}" 2>/dev/null; do
  if (( burner_stopped == 0 )); then
    memory_peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk '{if ($1 > peak) peak=$1} END {print peak + 0}')
    if grep -q 'Finished loading checkpoint in' "${RUN_LOG}" || (( memory_peak >= 80000 )); then
      stop_burner
      burner_stopped=1
      {
        echo "burner_stopped_epoch=$(date +%s)"
        echo "burner_stop_memory_peak_mib=${memory_peak}"
      } >>"${CASE_ENV}"
    fi
  fi
  sleep 2
done
wait "${INFERENCE_PID}"
status=$?
set -e
ended_epoch=$(date +%s)
{
  echo "ended_epoch=${ended_epoch}"
  echo "process_wall_seconds=$((ended_epoch - started_epoch))"
  echo "exit_code=${status}"
} >>"${CASE_ENV}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_vid_prompt_topology.py" \
  --case-dir "${OUTPUT_DIR}" --expected "${EXPECTED}"
if (( status != 0 )); then
  echo "Leo2 topology case ${LEO2_CASE_NAME} failed with exit code ${status}; see ${RUN_LOG}" >&2
  exit "${status}"
fi
