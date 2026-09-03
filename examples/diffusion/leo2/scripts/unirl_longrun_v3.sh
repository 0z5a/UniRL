#!/usr/bin/env bash
# Long run v3 = mirror of the native pure-torch run that learns
# (experiments/2026-08-28_native-rl-perf/configs/leo2_grpo_a12b_256p_kl30_rb8_liger.yaml):
#   sde_type flow_grpo (= FlowSDEStrategy)      grpo_sampling_steps 30
#   eta 0.5                                     progressive/timesteps_group_size 5
#     -> SDE noise + training ONLY on the first 5 (highest-sigma) transitions,
#        the other 25 are plain ODE (that is what killed v1/v2: 10 steps with
#        noise on [0,3,6,9] blew the marginal at sigma=1 and dumped noise into
#        the final latent -- see DESIGN.md R17)
#   num_generations 8, use_same_noise true      -> samples_per_prompt 8, shared x_T
#   rollout_cfg_scale 1.0, shift 3.0, 192x336x49 (already the yaml defaults)
#   clip_range 1e-4, per-group advantage std    reward: PickScore mean over 2 fps
#   frames (~4 for 49 frames)                   -> uniform 4-frame mean
# Not mirrored (inherent): full-param muon lr 1e-5 -> LoRA r64 AdamW 2.5e-5;
# KL 0.001 ref model / grpo_guard -> none. 8 prompts x 8 = 64 samples/step
# (DP_SCATTER needs batch_size % 8 == 0), ~2x their 32.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${LEO2_RUN_ROOT:-${REPO_ROOT}/outputs/leo2}"
export RUN_NAME=leo2_t2v_v3_nativemirror_30s_eta0.5_sde0-4_k8_$(date +%m%d_%H%M)
export NUM_ROLLOUTS=${NUM_ROLLOUTS:-200}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
# frame dumps of the first decodes so the first rollout can be eyeballed
export LEO2_DUMP_VIDEOS="${RUN_ROOT}/frames_${RUN_NAME}"
mkdir -p "$LEO2_DUMP_VIDEOS"
exec bash "${SCRIPT_DIR}/unirl_longrun.sh" \
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
  "$@"
