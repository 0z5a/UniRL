#!/usr/bin/env bash
# Leo2 x UniRL trainside FlowGRPO -- single-node 8-GPU LONG RUN (wandb on).
# Same recipe as unirl_smoke.sh (R15-validated) plus:
#   * wandb: online if $H/env/wandb.env exports WANDB_API_KEY, else offline
#     (sync later with `wandb sync <dir>`); run dir under $X/wandb/
#   * LoRA-adapter checkpoints every SAVE_INTERVAL rollouts under $X/ckpts/<run>
#   * NUM_ROLLOUTS rollouts (default 300 ~ 16 h at ~196 s/step)
# Run only when this node's GPUs are free (kills keepalive itself); restore the
# keepalive afterwards (trap below covers normal exit and crashes).
set -x
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
X=$H/experiments/2026-08-27_leo2-unirl-flowgrpo
U=$X/code/UniRL-leo2
C=$U/unirl/models/leo2/vendor/gen_ar
source $H/env/leo2-venv/bin/activate
export PYTHONPATH=$H/env/leo2-venv/lib/python3.12/site-packages:$U:$C:$C/deps/hy_parallelism:$C/deps/IndexKits:$PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ASSETS_BASE=$H/assets/hymm_ar_assets
export HF_HOME=$H/hf_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME=${RUN_NAME:-leo2_t2v_lora64_lr1e-4_$(date +%m%d_%H%M)}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-300}
SAVE_INTERVAL=${SAVE_INTERVAL:-25}
mkdir -p $X/wandb $X/ckpts/$RUN_NAME $X/logs

# wandb credentials live outside the repo; the file just exports WANDB_API_KEY
# (and optionally WANDB_ENTITY / WANDB_PROJECT / WANDB_BASE_URL).
[ -r $H/env/wandb.env ] && source $H/env/wandb.env
export WANDB_DIR=$X/wandb
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=${WANDB_MODE:-online}; else export WANDB_MODE=offline; fi
export WANDB_PROJECT=${WANDB_PROJECT:-unirl-leo2-t2v}
echo "wandb: mode=$WANDB_MODE project=$WANDB_PROJECT entity=${WANDB_ENTITY:-<default>} run=$RUN_NAME dir=$WANDB_DIR"

LOCAL_W=/root/leo2_weights/iter_0063300_torch/weights
if [ ! -f "$LOCAL_W/.staged" ]; then
  mkdir -p "$LOCAL_W"
  cp $H/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights/__0_0.distcp "$LOCAL_W/" && \
  cp $H/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights/.metadata "$LOCAL_W/" && \
  touch "$LOCAL_W/.staged"
fi
export LEO2_CKPT_DIR=$LOCAL_W

restore_keepalive() { ray stop --force 2>/dev/null; bash $H/jobs/remote/restart_keepalive.sh; }
trap restore_keepalive EXIT

fuser -v /dev/nvidia* -k 2>/dev/null; sleep 3
cd $U
ray stop --force 2>/dev/null; sleep 2
ray start --head --port=6501 --num-gpus=8 --disable-usage-stats 2>&1 | tail -2
export RAY_ADDRESS=127.0.0.1:6501
set -o pipefail
LOG=$X/logs/longrun_${RUN_NAME}.log
echo "$LOG" > $X/logs/longrun_latest.path
# Control channel: the queue runner does not accept new jobs while this one
# runs, so a long run can only be stopped from the outside by touching
# $X/logs/STOP_<run> (or STOP_ALL) -- polled every 60 s below.
STOP_FILE=$X/logs/STOP_$RUN_NAME
rm -f "$STOP_FILE"
python -m unirl.train_diffusion --config-name diffusion/leo2/leo2_t2v_trainside \
  num_devices=8 \
  num_rollouts=$NUM_ROLLOUTS +save_interval=$SAVE_INTERVAL +save_dir=$X/ckpts/$RUN_NAME +save_mode=auto \
  logging.report_to_wandb=true logging.run_name=$RUN_NAME logging.project_name=$WANDB_PROJECT \
  ${WANDB_ENTITY:+logging.entity=$WANDB_ENTITY} \
  "$@" > "$LOG" 2>&1 &
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
  if [ -f "$STOP_FILE" ] || [ -f "$X/logs/STOP_ALL" ]; then
    echo "STOP file seen, terminating run $RUN_NAME" | tee -a "$LOG"
    pkill -TERM -f "unirl[.]train_diffusion" 2>/dev/null; sleep 20
    pkill -KILL -f "unirl[.]train_diffusion" 2>/dev/null
    rm -f "$STOP_FILE"
    break
  fi
  sleep 60
done
wait $TRAIN_PID 2>/dev/null; rc=$?
echo "LONGRUN_RC=$rc" | tee -a "$LOG"
