#!/usr/bin/env bash
# Long run v3b (node1, control for v3): identical native-mirror recipe plus the
# two verified speed-ups (old_logp from rollout, resident text encoder) and a
# 4x higher LoRA lr (1e-4) -- the one free knob, to see whether the reward
# moves faster than v3's 2.5e-5 without drifting (v1's drift was the broken
# sampler, not the lr).
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
export RUN_NAME=leo2_t2v_v3b_nativemirror_lr1e-4_fast_$(date +%m%d_%H%M)
export NUM_ROLLOUTS=${NUM_ROLLOUTS:-200}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
export LEO2_DUMP_VIDEOS=$X/docs/frames_$RUN_NAME
mkdir -p "$LEO2_DUMP_VIDEOS"
exec bash $H/jobs/remote/unirl_longrun.sh \
  sampling.num_inference_steps=30 \
  sampling.eta=0.5 \
  "sampling.sde_indices=[0,1,2,3,4]" \
  sampling.samples_per_prompt=8 \
  sampling.init_same_noise=true \
  stack.num_updates_per_batch=2 \
  backend.optimizer_cfg.learning_rate=1.0e-4 \
  adv_use_global_std=false \
  +reward.backend.config.frame_selection=uniform \
  +reward.backend.config.num_score_frames=4 \
  algorithm.old_logp_source=rollout \
  bundle.config.text_encoder_gpu_transient=false \
  "$@"
