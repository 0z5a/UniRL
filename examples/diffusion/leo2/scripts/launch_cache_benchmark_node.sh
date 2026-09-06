#!/usr/bin/env bash
# Run one single-node Leo2 benchmark while preserving GPU keepalive before/after.
set -euo pipefail

OUTPUT_ROOT="${1:?expected OUTPUT_ROOT as argument 1}"
CASES_CSV="${2:?expected CASES_CSV as argument 2}"
PROMPTS_CSV="${3:?expected PROMPTS_CSV as argument 3}"
INFER_STEPS="${4:?expected INFER_STEPS as argument 4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEEPALIVE_SCRIPT="${LEO2_KEEPALIVE_SCRIPT:-/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/.cursor/skills/taiji-node-env-bootstrap/scripts/keepalive_launch.sh}"
KEEPALIVE_PYTHON="${KEEPALIVE_PYTHON:-/opt/conda/envs/ff/bin/python}"

for path in "${CASES_CSV}" "${PROMPTS_CSV}" "${KEEPALIVE_SCRIPT}"; do
  [[ -f "${path}" ]] || { echo "expected input file, got ${path}" >&2; exit 2; }
done
[[ "${INFER_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "expected positive integer INFER_STEPS, got ${INFER_STEPS@Q}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to overwrite existing output root: ${OUTPUT_ROOT}" >&2
  exit 2
}

restore_keepalive() {
  KEEPALIVE_PYTHON="${KEEPALIVE_PYTHON}" bash "${KEEPALIVE_SCRIPT}" || {
    echo "failed to restore GPU keepalive after benchmark" >&2
    return 1
  }
}
trap restore_keepalive EXIT

pkill -TERM -f 'gpu_keepalive[.]py' 2>/dev/null || true
for _ in $(seq 1 60); do
  pgrep -f 'gpu_keepalive[.]py' >/dev/null || break
  sleep 1
done
if pgrep -f 'gpu_keepalive[.]py' >/dev/null; then
  echo "GPU keepalive did not stop within 60 seconds" >&2
  exit 1
fi

env \
  LEO2_CKPT_DIR="${LEO2_CKPT_DIR:-/root/leo2-heavy/checkpoint}" \
  LEO2_ASSETS_BASE="${LEO2_ASSETS_BASE:-/root/leo2-heavy/assets}" \
  LEO2_RUNTIME_PYTHON="${LEO2_RUNTIME_PYTHON:-/root/.local/leo2-runtime-py312-torch271-cu129/bin/python}" \
  LEO2_CACHE_BENCH_OUTPUT="${OUTPUT_ROOT}" \
  LEO2_CACHE_BENCH_PROMPTS="${PROMPTS_CSV}" \
  LEO2_CACHE_BENCH_CASES="${CASES_CSV}" \
  LEO2_CACHE_BENCH_PILOT=0 \
  LEO2_INFER_STEPS="${INFER_STEPS}" \
  LEO2_VIDEO_FPS=24 \
  bash "${SCRIPT_DIR}/cache_benchmark.sh"
