#!/usr/bin/env bash
# Leo2 x UniRL trainside FlowGRPO -- single-node 8-GPU smoke.
# Run only when this node's GPUs are free (kills keepalive itself).
set -x
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
U=$X/code/UniRL-leo2
source $H/env/leo2-venv/bin/activate
# hymm runtime on PYTHONPATH for the *driver* and reward actors too: the rollout
# Sample carries Leo2Conditions.hymm blobs whose pickles reference hymm classes
# (R13: driver died unpickling them with "No module named 'hymm'").
C=$U/unirl/models/leo2/vendor/gen_ar
export PYTHONPATH=$H/env/leo2-venv/lib/python3.12/site-packages:$U:$C:$C/deps/hy_parallelism:$C/deps/IndexKits:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ASSETS_BASE=$H/assets/hymm_ar_assets
export HF_HOME=$H/hf_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# wandb off for smoke; NCCL env matches the verified inference runs
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# stage weights to node-local disk (12T, one-time ~3min; 8 ranks then read RAM-cached local)
LOCAL_W=/root/leo2_weights/iter_0063300_torch/weights
if [ ! -f "$LOCAL_W/.staged" ]; then
  mkdir -p "$LOCAL_W"
  cp $H/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights/__0_0.distcp "$LOCAL_W/" && \
  cp $H/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights/.metadata "$LOCAL_W/" && \
  touch "$LOCAL_W/.staged"
fi
export LEO2_CKPT_DIR=$LOCAL_W

fuser -v /dev/nvidia* -k 2>/dev/null; sleep 3
cd $U
ray stop --force 2>/dev/null; sleep 2
ray start --head --port=6501 --num-gpus=8 --disable-usage-stats 2>&1 | tail -2
export RAY_ADDRESS=127.0.0.1:6501
set -o pipefail
python -m unirl.train_diffusion --config-name diffusion/leo2/leo2_t2v_trainside \
  num_devices=8 "$@" 2>&1 | tee $X/logs/smoke_$(date +%H%M%S).log
echo SMOKE_RC=$?
ray stop --force 2>/dev/null
