#!/usr/bin/env bash
# Perf probe (node1): the v3 native-mirror recipe for 2 rollouts with two
# cheap speed-ups, to measure their effect and the rollout/replay parity:
#   * old_logp_source=rollout  (skip the no_grad replay: 5 x 8 forwards/rank)
#   * text encoder resident on GPU (no CPU<->GPU shuttle per rollout)
# Compare lifecycle rollout.generate / train_track against v3's log.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${LEO2_RUN_ROOT:-${REPO_ROOT}/outputs/leo2}"
export LEO2_DUMP_VIDEOS="${RUN_ROOT}/frames_probe_perf"
mkdir -p "$LEO2_DUMP_VIDEOS"
bash "${SCRIPT_DIR}/unirl_smoke.sh" \
  num_rollouts=2 \
  sampling.num_inference_steps=30 \
  sampling.eta=0.5 \
  "sampling.sde_indices=[0,1,2,3,4]" \
  sampling.samples_per_prompt=8 \
  sampling.init_same_noise=true \
  stack.num_updates_per_batch=2 \
  backend.optimizer_cfg.learning_rate=2.5e-5 \
  adv_use_global_std=false \
  +reward.backend.config.frame_selection=uniform \
  +reward.backend.config.num_score_frames=4 \
  algorithm.old_logp_source=rollout \
  bundle.config.text_encoder_gpu_transient=false \
  logging.report_to_wandb=false \
  "$@"
