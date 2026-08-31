#!/usr/bin/env bash
# Sample nvidia-smi for DUR seconds (default 120) every 2 s; print per-GPU max
# memory.used and mean utilisation. Read-only; safe to run alongside a job.
DUR=${GPU_SNAPSHOT_DUR:-120}
end=$((SECONDS + DUR))
tmp=$(mktemp)
while [ $SECONDS -lt $end ]; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$tmp"
  sleep 2
done
awk -F', *' '{ if ($2 > mx[$1]) mx[$1] = $2; ut[$1] += $3; n[$1]++ }
  END { for (i in mx) printf "gpu%s max_mem_used=%d MiB mean_util=%.0f%% samples=%d\n", i, mx[i], ut[i]/n[i], n[i] }' "$tmp" | sort
rm -f "$tmp"
