#!/usr/bin/env bash
# Sampler A/B probe (node1): one rollout at ~ODE (eta 0.02) with frame dumps of
#   (a) hymm's own pipeline on the same wrapped weights  -> docs/frames_probe/hymm_*.png
#   (b) the UniRL Leo2DiffusionStage rollout              -> docs/frames_probe/decode*.png
# No wandb. Restores keepalive on exit via unirl_smoke.sh's trap-less flow +
# explicit restart below.
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
export LEO2_DUMP_VIDEOS=$X/docs/frames_probe_unirl
export LEO2_DEBUG_HYMM_SAMPLE=1
rm -rf "$LEO2_DUMP_VIDEOS"; mkdir -p "$LEO2_DUMP_VIDEOS"
bash $H/jobs/remote/unirl_smoke.sh \
  num_rollouts=1 \
  sampling.eta=0.02 \
  logging.report_to_wandb=false \
  "$@"
ls -la "$LEO2_DUMP_VIDEOS"
bash $H/jobs/remote/restart_keepalive.sh
