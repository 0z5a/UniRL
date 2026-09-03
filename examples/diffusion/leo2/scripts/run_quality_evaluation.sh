#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/leo2-output/cache-full-final-848x464x121-20260903-1039}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LEO2_DIR="${REPO_ROOT}/examples/diffusion/leo2"
CASES_CSV="${LEO2_CACHE_BENCH_CASES:-${LEO2_DIR}/data/cache_benchmark_cases.csv}"
PROMPTS_CSV="${LEO2_CACHE_BENCH_PROMPTS:-${LEO2_DIR}/data/cache_benchmark_16.csv}"
OUTPUT="${ROOT}/quality_eval"
VBENCH_SOURCE="${VBENCH_SOURCE:-/root/leo2-eval/sources/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/root/leo2-eval/envs/vbench/bin/python}"
VIDEOSCORE_PYTHON="${VIDEOSCORE_PYTHON:-/root/leo2-eval/envs/videoscore2/bin/python}"
GPU_OFFSET="${LEO2_QUALITY_GPU_OFFSET:-0}"

if [[ ! "${GPU_OFFSET}" =~ ^[0-7]$ ]]; then
  echo "LEO2_QUALITY_GPU_OFFSET must be an integer in 0..7, got: ${GPU_OFFSET}" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/vbench" "${OUTPUT}/videoscore2" "${OUTPUT}/logs"
mapfile -t CASES < <(
  python "${LEO2_DIR}/scripts/cache_benchmark_cases.py" --emit-records "${CASES_CSV}" \
    | cut -d $'\x1f' -f1
)
if ((GPU_OFFSET + ${#CASES[@]} > 8)); then
  echo "Quality cases exceed GPUs 0..7 after offset ${GPU_OFFSET}" >&2
  exit 2
fi

run_vbench_case() {
  local case="$1" gpu="$2"
  local case_output="${OUTPUT}/vbench/${case}"
  local videos
  local -a video_dirs
  mapfile -t video_dirs < <(find "${ROOT}/${case}/samples" -type d -name videos)
  if [[ "${#video_dirs[@]}" -ne 1 ]]; then
    echo "Expected one videos directory for ${case}, found ${#video_dirs[@]}" >&2
    return 2
  fi
  videos="${video_dirs[0]}"
  local prompt_file="${case_output}/prompts.json"
  mkdir -p "${case_output}"
  if compgen -G "${case_output}/*_eval_results.json" >/dev/null; then
    return
  fi
  PROMPTS_CSV="${PROMPTS_CSV}" VIDEOS="${videos}" PROMPT_FILE="${prompt_file}" \
    python - <<'PY'
import csv
import json
import os

with open(os.environ["PROMPTS_CSV"], newline="") as handle:
    rows = list(csv.DictReader(handle))
prompts = {f'{row["index"]}_0.mp4': row["prompt"] for row in rows}
with open(os.environ["PROMPT_FILE"], "w") as handle:
    json.dump(prompts, handle, indent=2)
PY
  CUDA_VISIBLE_DEVICES="${gpu}" MASTER_ADDR=127.0.0.1 MASTER_PORT="$((29800 + gpu))" \
    VBENCH_CACHE_DIR=/root/leo2-eval/cache/vbench \
    TORCH_HOME=/root/leo2-eval/cache/torch \
    HF_HOME=/root/leo2-eval/cache/huggingface \
    "${VBENCH_PYTHON}" "${VBENCH_SOURCE}/evaluate.py" \
      --videos_path "${videos}" \
      --output_path "${case_output}" \
      --prompt_file "${prompt_file}" \
      --mode custom_input \
      --load_ckpt_from_local True \
      --dimension subject_consistency background_consistency motion_smoothness \
        dynamic_degree aesthetic_quality imaging_quality \
      >"${OUTPUT}/logs/vbench_${case}.log" 2>&1
}

run_videoscore_case() {
  local case="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME=/root/leo2-eval/cache/huggingface \
    "${VIDEOSCORE_PYTHON}" "${LEO2_DIR}/scripts/evaluate_videoscore2.py" \
      --benchmark-root "${ROOT}" \
      --case "${case}" \
      --prompts-csv "${PROMPTS_CSV}" \
      --output "${OUTPUT}/videoscore2/${case}.jsonl" \
      --resume \
      >"${OUTPUT}/logs/videoscore2_${case}.log" 2>&1
}

pids=()
for index in "${!CASES[@]}"; do
  run_vbench_case "${CASES[$index]}" "$((GPU_OFFSET + index))" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
((status == 0)) || exit "${status}"

pids=()
for index in "${!CASES[@]}"; do
  run_videoscore_case "${CASES[$index]}" "$((GPU_OFFSET + index))" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
((status == 0)) || exit "${status}"

echo "Quality evaluation completed: ${OUTPUT}"
