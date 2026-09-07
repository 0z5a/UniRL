#!/usr/bin/env bash
# Run the vendored native Leo2 T2V sampler on one 8-GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENDOR_ROOT="${REPO_ROOT}/unirl/models/leo2/vendor/gen_ar"
MODEL_CONFIG="${VENDOR_ROOT}/hymm/configs/leo2/leo2_moe_v1_1_a12b_muon_wzd_480p_stage3.yaml"
GENERATION_CONFIG="${REPO_ROOT}/unirl/models/leo2/resources/generation_config_rl_video.json"

: "${LEO2_CKPT_DIR:?Set LEO2_CKPT_DIR to a native Torch DCP weights directory}"
: "${LEO2_ASSETS_BASE:?Set LEO2_ASSETS_BASE to the hymm_ar_assets directory}"

PYTHON_BIN="${LEO2_RUNTIME_PYTHON:-python}"
TESTSETS="${LEO2_TESTSETS:-${REPO_ROOT}/examples/diffusion/leo2/data/native_smoke.csv@@_first_n_=1}"
OUTPUT_DIR="${LEO2_OUTPUT_DIR:-${REPO_ROOT}/outputs/leo2/native_t2v}"
NUM_FRAMES="${LEO2_NUM_FRAMES:-121}"
IMAGE_SIZE="${LEO2_IMAGE_SIZE:-464x848}"
INFER_STEPS="${LEO2_INFER_STEPS:-50}"

export ASSETS_BASE="${LEO2_ASSETS_BASE}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
export HOST_NUM="${HOST_NUM:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/runtime}"
export PYTHONPATH="${REPO_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=8 \
  -m unirl.models.leo2.native_entry \
  --config-path "${MODEL_CONFIG}" \
  --sampler leo2_sampler.Leo2Sampler \
  --framework fsdp \
  --ckpt "${LEO2_CKPT_DIR}" \
  --generation-config "${GENERATION_CONFIG}" \
  --testsets "${TESTSETS}" \
  --sample-save-base "${OUTPUT_DIR}" \
  --task-id leo2_native_t2v \
  --bot-task av \
  --use-system-prompt li-dit-encode-visual-qwen-3.5 \
  --gate-impl deepseek \
  --vae-type 16x16x4-48c-hy-v3_3-release2 \
  --use-audio-vae \
  --audio-vae-type dual_channel_48k \
  --audio-vae-latent-dim 96 \
  --image-size "${IMAGE_SIZE}" \
  --num-frames "${NUM_FRAMES}" \
  --video-fps 24 \
  --diff-infer-steps "${INFER_STEPS}" \
  --diff-guidance-scale 1.0 \
  --flow-shift-video 9.0 \
  --sample-batch-size 1 \
  --max-sample-batches 1 \
  --dp-shard 8 \
  --context-parallel-size 8 \
  --expert-model-parallel-size 8 \
  "$@"
