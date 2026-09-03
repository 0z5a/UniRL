#!/usr/bin/env bash
# Sampler A/B probe (node1): one rollout at ~ODE (eta 0.02) with frame dumps of
#   (a) hymm's own pipeline on the same wrapped weights  -> docs/frames_probe/hymm_*.png
#   (b) the UniRL Leo2DiffusionStage rollout              -> docs/frames_probe/decode*.png
# No wandb; writes comparison frames below LEO2_RUN_ROOT.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${LEO2_RUN_ROOT:-${REPO_ROOT}/outputs/leo2}"
export LEO2_DUMP_VIDEOS="${RUN_ROOT}/frames_probe_unirl"
export LEO2_DEBUG_HYMM_SAMPLE=1
mkdir -p "$LEO2_DUMP_VIDEOS"
bash "${SCRIPT_DIR}/unirl_smoke.sh" \
  num_rollouts=1 \
  sampling.eta=0.02 \
  logging.report_to_wandb=false \
  "$@"
ls -la "$LEO2_DUMP_VIDEOS"
