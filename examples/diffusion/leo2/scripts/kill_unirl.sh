#!/usr/bin/env bash
# Stop the UniRL trainside smoke on THIS node (hydra driver + ray workers),
# free the GPUs, then restore the keepalive. Node0/1 only (UniRL line).
# Bracketed patterns so pkill never matches this script's own command line.
set -x
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
source $H/env/leo2-venv/bin/activate
pkill -f "unirl[.]train_diffusion" 2>/dev/null; sleep 3
ray stop --force 2>/dev/null; sleep 2
pkill -f "ray[:][:]" 2>/dev/null
fuser -v /dev/nvidia* -k 2>/dev/null; sleep 3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
bash $H/jobs/remote/restart_keepalive.sh
