#!/usr/bin/env bash
# Long run v3b (node1, control for v3): identical native-mirror recipe plus the
# Long run v4: v3b (lr 1e-4 + speed-ups) plus the missing regularizer -- KL
# against the LoRA-disabled base policy (UniRL FlowGRPO beta>0: Gaussian KL on
# per-step means, zero extra memory, one extra no_grad replay per update).
# v3b collapsed into stripe textures by rollout ~50 (reward hacking, R26); the
# native line survives lr-equivalent updates only with kl_weight 0.001.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${LEO2_RUN_ROOT:-${REPO_ROOT}/outputs/leo2}"
export RUN_NAME=leo2_t2v_v4_kl_lr1e-4_$(date +%m%d_%H%M)
export NUM_ROLLOUTS=${NUM_ROLLOUTS:-200}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
export LEO2_DUMP_VIDEOS="${RUN_ROOT}/frames_${RUN_NAME}"
mkdir -p "$LEO2_DUMP_VIDEOS"
exec bash "${SCRIPT_DIR}/unirl_longrun.sh" \
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
  +algorithm.beta=0.001 \
  "$@"
