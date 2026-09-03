#!/usr/bin/env bash
# Run paired Leo2 first-block-cache benchmarks on one 8-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENDOR_ROOT="${REPO_ROOT}/unirl/models/leo2/vendor/gen_ar"
MODEL_CONFIG="${VENDOR_ROOT}/hymm/configs/leo2/leo2_moe_v1_1_a12b_muon_wzd_480p_stage3.yaml"
GENERATION_CONFIG="${REPO_ROOT}/unirl/models/leo2/resources/generation_config_rl_video.json"
PROMPTS_CSV="${LEO2_CACHE_BENCH_PROMPTS:-${REPO_ROOT}/examples/diffusion/leo2/data/cache_benchmark_16.csv}"
CASES_CSV="${LEO2_CACHE_BENCH_CASES:-${REPO_ROOT}/examples/diffusion/leo2/data/cache_benchmark_cases.csv}"
PYTHON_BIN="${LEO2_RUNTIME_PYTHON:-python}"
DRY_RUN="${LEO2_CACHE_BENCH_DRY_RUN:-0}"
PILOT="${LEO2_CACHE_BENCH_PILOT:-0}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
if [[ "${PILOT}" == "1" ]]; then
  MODE=pilot
  EXPECTED_VIDEOS=1
  case_rows=$'pilot_off_shift9,off,9.0\npilot_t000_shift9,0,9.0\npilot_t005_shift9,0.05,9.0'
else
  MODE=full
  EXPECTED_VIDEOS=16
  case_rows=""
fi
OUTPUT_ROOT="${LEO2_CACHE_BENCH_OUTPUT:-${REPO_ROOT}/outputs/leo2/cache-${MODE}-${TIMESTAMP}}"
INFER_STEPS="${LEO2_INFER_STEPS:-50}"
ARTIFACT_MANIFEST="${REPO_ROOT}/unirl/models/leo2/resources/artifacts.yaml"

if [[ ! -f "${PROMPTS_CSV}" ]]; then
  echo "Prompt CSV does not exist: ${PROMPTS_CSV}" >&2
  exit 2
fi
if [[ ! -f "${CASES_CSV}" ]]; then
  echo "Case CSV does not exist: ${CASES_CSV}" >&2
  exit 2
fi
if [[ "${PILOT}" != "0" && "${PILOT}" != "1" ]]; then
  echo "LEO2_CACHE_BENCH_PILOT must be 0 or 1, got: ${PILOT}" >&2
  exit 2
fi
if [[ "${PILOT}" == "0" ]]; then
  case_rows="$(tail -n +2 "${CASES_CSV}")"
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  : "${LEO2_CKPT_DIR:?Set LEO2_CKPT_DIR to the native Torch DCP weights directory}"
  : "${LEO2_ASSETS_BASE:?Set LEO2_ASSETS_BASE to hymm_ar_assets}"
  if [[ ! -f "${LEO2_CKPT_DIR}/.metadata" ]]; then
    echo "Checkpoint metadata does not exist: ${LEO2_CKPT_DIR}/.metadata" >&2
    exit 2
  fi
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse benchmark output root: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

export ASSETS_BASE="${LEO2_ASSETS_BASE:-}"
export HF_HOME="${LEO2_CACHE_BENCH_HF_HOME:-${OUTPUT_ROOT}/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HOST_GPU_NUM=8
export HOST_NUM=1
export NCCL_IB_DISABLE="${LEO2_CACHE_BENCH_NCCL_IB_DISABLE:-0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export PYTHONPATH="${REPO_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "${HF_HOME}"

prompt_count="$(awk -F, 'NR > 1 {count += 1} END {print count + 0}' "${PROMPTS_CSV}")"
if [[ "${prompt_count}" -ne 16 ]]; then
  echo "Expected 16 prompt rows, found ${prompt_count}: ${PROMPTS_CSV}" >&2
  exit 2
fi

git_head="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git_diff_sha256="$(git -C "${REPO_ROOT}" diff --binary HEAD -- | sha256sum | awk '{print $1}')"
harness_sha256="$(
  cd "${REPO_ROOT}"
  sha256sum \
    examples/diffusion/leo2/CACHE_BENCHMARK.md \
    examples/diffusion/leo2/data/cache_benchmark_16.csv \
    examples/diffusion/leo2/data/cache_benchmark_cases.csv \
    examples/diffusion/leo2/scripts/cache_benchmark.sh \
    examples/diffusion/leo2/scripts/cache_benchmark_entry.py \
    examples/diffusion/leo2/scripts/summarize_cache_benchmark.py \
    | sha256sum | awk '{print $1}'
)"
checkpoint_metadata_sha256=DRY_RUN
checkpoint_shard_inventory_sha256=DRY_RUN
if [[ "${DRY_RUN}" != "1" ]]; then
  checkpoint_metadata_sha256="$(sha256sum "${LEO2_CKPT_DIR}/.metadata" | awk '{print $1}')"
  checkpoint_shard_inventory_sha256="$(find "${LEO2_CKPT_DIR}" -maxdepth 1 -type f -name '*.distcp' \
    -printf '%f %s\n' | sort | sha256sum | awk '{print $1}')"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader >"${OUTPUT_ROOT}/gpu_inventory.csv"
fi

{
  echo "created_at=${TIMESTAMP}"
  echo "mode=${MODE}"
  echo "repo_root=${REPO_ROOT}"
  echo "git_head=${git_head}"
  echo "git_diff_sha256=${git_diff_sha256}"
  echo "benchmark_harness_sha256=${harness_sha256}"
  echo "prompts_csv=${PROMPTS_CSV}"
  echo "prompts_sha256=$(sha256sum "${PROMPTS_CSV}" | awk '{print $1}')"
  echo "cases_csv=${CASES_CSV}"
  echo "cases_sha256=$(sha256sum "${CASES_CSV}" | awk '{print $1}')"
  echo "artifact_manifest_sha256=$(sha256sum "${ARTIFACT_MANIFEST}" | awk '{print $1}')"
  echo "model_config_sha256=$(sha256sum "${MODEL_CONFIG}" | awk '{print $1}')"
  echo "generation_config_sha256=$(sha256sum "${GENERATION_CONFIG}" | awk '{print $1}')"
  echo "checkpoint_dir=${LEO2_CKPT_DIR:-DRY_RUN}"
  echo "assets_base=${LEO2_ASSETS_BASE:-DRY_RUN}"
  echo "checkpoint_metadata_sha256=${checkpoint_metadata_sha256}"
  echo "checkpoint_shard_inventory_sha256=${checkpoint_shard_inventory_sha256}"
  echo "runtime_python=${PYTHON_BIN}"
  echo "runtime_python_version=$("${PYTHON_BIN}" --version 2>&1)"
  echo "image_size=464x848"
  echo "num_frames=121"
  echo "diff_infer_steps=${INFER_STEPS}"
  echo "expected_videos_per_case=${EXPECTED_VIDEOS}"
} >"${OUTPUT_ROOT}/benchmark.env"

case_count=0
while IFS=, read -r case_name cache_threshold flow_shift_video; do
  if [[ "${case_name}" == "name" || -z "${case_name}" ]]; then
    continue
  fi
  if [[ ! "${case_name}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Invalid case name: ${case_name}" >&2
    exit 2
  fi
  if [[ ! "${cache_threshold}" =~ ^(off|[0-9]+([.][0-9]+)?)$ ]]; then
    echo "Invalid cache threshold for ${case_name}: ${cache_threshold}" >&2
    exit 2
  fi
  if [[ ! "${flow_shift_video}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid flow shift for ${case_name}: ${flow_shift_video}" >&2
    exit 2
  fi

  case_count=$((case_count + 1))
  case_dir="${OUTPUT_ROOT}/${case_name}"
  sample_dir="${case_dir}/samples"
  latent_dir="${case_dir}/latents"
  mkdir -p "${sample_dir}" "${latent_dir}"
  export OUTPUT_PATH="${case_dir}/runtime"
  command=(
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=8
    "${SCRIPT_DIR}/cache_benchmark_entry.py"
    --leo2-cache-threshold "${cache_threshold}"
    --leo2-latent-dir "${latent_dir}"
    --leo2-prompt-csv "${PROMPTS_CSV}"
    --config-path "${MODEL_CONFIG}"
    --sampler leo2_sampler.Leo2Sampler
    --framework fsdp
    --ckpt "${LEO2_CKPT_DIR:-DRY_RUN_CKPT}"
    --generation-config "${GENERATION_CONFIG}"
    --testsets "${PROMPTS_CSV}@@_first_n_=${EXPECTED_VIDEOS}"
    --sample-save-base "${sample_dir}"
    --task-id "leo2_cache_bench_${case_name}"
    --bot-task video
    --use-system-prompt li-dit-encode-visual-qwen-3.5
    --gate-impl deepseek
    --vae-type 16x16x4-48c-hy-v3_3-release2
    --image-size 464x848
    --num-frames 121
    --video-fps 24
    --diff-infer-steps "${INFER_STEPS}"
    --diff-guidance-scale 1.0
    --flow-shift-video "${flow_shift_video}"
    --sample-batch-size 1
    --max-sample-batches "${EXPECTED_VIDEOS}"
    --dp-shard 8
    --context-parallel-size 8
    --expert-model-parallel-size 1
  )
  if [[ "${PILOT}" == "1" && "${cache_threshold}" != "off" && "${cache_threshold}" != "0" ]]; then
    command+=(--leo2-require-cache-hit)
  fi
  {
    printf '#!/usr/bin/env bash\n'
    printf 'export ASSETS_BASE=%q\n' "${ASSETS_BASE}"
    printf 'export HF_HOME=%q\n' "${HF_HOME}"
    printf 'export HF_HUB_OFFLINE=%q\n' "${HF_HUB_OFFLINE}"
    printf 'export HOST_GPU_NUM=%q\n' "${HOST_GPU_NUM}"
    printf 'export HOST_NUM=%q\n' "${HOST_NUM}"
    printf 'export NCCL_IB_DISABLE=%q\n' "${NCCL_IB_DISABLE}"
    printf 'export NCCL_IB_GID_INDEX=%q\n' "${NCCL_IB_GID_INDEX}"
    printf 'export OUTPUT_PATH=%q\n' "${OUTPUT_PATH}"
    printf 'export PYTHONPATH=%q\n' "${PYTHONPATH}"
    printf 'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n'
    printf 'exec '
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"${case_dir}/command.sh"
  chmod +x "${case_dir}/command.sh"
  {
    echo "case=${case_name}"
    echo "cache_threshold=${cache_threshold}"
    echo "flow_shift_video=${flow_shift_video}"
    echo "expected_videos=${EXPECTED_VIDEOS}"
  } >"${case_dir}/case.env"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${command[@]}"
    printf '\n'
    echo 0 >"${case_dir}/exit_code.txt"
    continue
  fi

  echo "Starting ${case_name}: cache=${cache_threshold}, flow_shift_video=${flow_shift_video}"
  started_epoch="$(date +%s)"
  echo "started_epoch=${started_epoch}" >>"${case_dir}/case.env"
  set +e
  "${command[@]}" 2>&1 | tee "${case_dir}/run.log"
  status=${PIPESTATUS[0]}
  set -e
  ended_epoch="$(date +%s)"
  {
    echo "ended_epoch=${ended_epoch}"
    echo "process_wall_seconds=$((ended_epoch - started_epoch))"
  } >>"${case_dir}/case.env"
  echo "${status}" >"${case_dir}/exit_code.txt"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_cache_benchmark.py" \
    --root "${OUTPUT_ROOT}" --expected-videos "${EXPECTED_VIDEOS}" >/dev/null
  if [[ "${status}" -ne 0 ]]; then
    echo "Benchmark case failed (${status}): ${case_name}" >&2
    exit "${status}"
  fi
  "${PYTHON_BIN}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["complete"] else 1)' \
    "${case_dir}/summary.json" \
    || { echo "Benchmark validation failed: ${case_name}" >&2; exit 1; }
done <<<"${case_rows}"

if [[ "${case_count}" -eq 0 ]]; then
  echo "No benchmark cases found in ${CASES_CSV}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_cache_benchmark.py" \
    --root "${OUTPUT_ROOT}" --expected-videos "${EXPECTED_VIDEOS}"
fi
echo "Benchmark root: ${OUTPUT_ROOT}"
