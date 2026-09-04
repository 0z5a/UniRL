#!/usr/bin/env bash
# Leo2 x UniRL trainside DiffusionNFT — single-node long run with W&B media.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${LEO2_RUN_ROOT:-${REPO_ROOT}/outputs/leo2}"
RUN_NAME="${RUN_NAME:-leo2_t2v_nft_lora64_$(date +%m%d_%H%M)}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-300}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

mkdir -p "${RUN_ROOT}/wandb" "${RUN_ROOT}/ckpts/${RUN_NAME}" "${RUN_ROOT}/logs"
if [[ -n "${WANDB_ENV_FILE:-}" ]]; then
    [[ -f "${WANDB_ENV_FILE}" ]] || {
        echo "WANDB_ENV_FILE does not exist: ${WANDB_ENV_FILE}" >&2
        exit 2
    }
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
fi
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-leo2-rl}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export REPORT_TO_WANDB=true

LOG="${RUN_ROOT}/logs/nft_longrun_${RUN_NAME}.log"
printf '%s\n' "${LOG}" >"${RUN_ROOT}/logs/nft_longrun_latest.path"
set -o pipefail
bash "${SCRIPT_DIR}/unirl_nft_smoke.sh" \
    "num_rollouts=${NUM_ROLLOUTS}" \
    "+save_interval=${SAVE_INTERVAL}" \
    "+save_dir=${RUN_ROOT}/ckpts/${RUN_NAME}" \
    "+save_mode=auto" \
    "logging.report_to_wandb=true" \
    "logging.log_media=true" \
    "logging.run_name=${WANDB_RUN_NAME}" \
    "logging.project_name=${WANDB_PROJECT}" \
    "$@" 2>&1 | tee "${LOG}"
