#!/usr/bin/env bash
# Sampler probe #2 (node1): one rollout with the CPS kernel (coefficient-
# preserving SDE: no variance blow-up at sigma=1, zero noise into the final
# latent) at eta 0.6 on steps [0,2,4,6] -- the candidate fix for the garbage
# rollouts produced by FlowSDE eta 0.7 on [0,3,6,9]. Dumps frames of both the
# UniRL rollout and hymm's own pipeline (A/B) to docs/frames_probe_cps.
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
export LEO2_DUMP_VIDEOS=$X/docs/frames_probe_cps
# hymm A/B disabled: generate_video's CPU-side allocations OOM'd the host
# (8 ranks x model staging) and took the whole probe down on the first try.
unset LEO2_DEBUG_HYMM_SAMPLE
mkdir -p "$LEO2_DUMP_VIDEOS"
bash $H/jobs/remote/unirl_smoke.sh \
  num_rollouts=1 \
  pipeline.strategy._target_=unirl.sde.kernels.CPSSDEStrategy \
  sampling.eta=0.6 \
  "sampling.sde_indices=[0,2,4,6]" \
  logging.report_to_wandb=false \
  "$@"
ls -la "$LEO2_DUMP_VIDEOS"
bash $H/jobs/remote/restart_keepalive.sh
