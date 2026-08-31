#!/usr/bin/env bash
# Stop the UniRL smoke on node0 and restore its keepalive, dispatched from
# another node's queue (node0's queue is blocked by the smoke itself).
# The kill logic lives in a file on the remote side so this ssh command line
# never contains the pkill pattern (self-match pitfall).
H=/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 29.162.240.239 \
  "bash $H/jobs/remote/kill_unirl.sh"
echo "ssh rc=$?"
