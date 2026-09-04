# Leo2 vid_prompt topology benchmark

## Pilot topology results

| Case | Status | DP shard | CP | EP | TP requested/effective | Videos | Wall (s) | Videos/h | Peak GiB | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cp1_shard8_ep1 | PASS | 8 | 1 | 1 | 1/1 | 4 | 1444 | 9.97 | 88.40 | PASS |
| cp2_shard8_ep1 | PASS | 8 | 2 | 1 | 1/1 | 4 | 840 | 17.14 | 83.79 | PASS |
| cp4_ep1 | PASS | 8 | 4 | 1 | 1/1 | 4 | 888 | 16.22 | 83.56 | PASS |
| cp4_shard4_ep1 | FAIL | 4 | 4 | 1 | 1/1 | 4 | 224 | 0.00 | 94.42 | CUDA OOM |
| cp8_ep1 | PASS | 8 | 8 | 1 | 1/1 | 4 | 967 | 14.89 | 83.14 | PASS |
| cp8_ep2_no_load_burner | FAIL | 8 | 8 | 2 | 1/1 | 1 | 46 | 0.00 | 93.75 | CUDA OOM |
| cp8_ep4_no_load_burner | FAIL | 8 | 8 | 4 | 1/1 | 1 | 80 | 0.00 | 95.00 | CUDA OOM |
| cp8_ep8 | PASS | 8 | 8 | 8 | 1/1 | 4 | 989 | 14.56 | 94.52 | PASS |
| cp8_tp8_requested | PASS | 8 | 8 | 1 | 8/1 | 1 | 388 | 9.28 | 83.10 | PASS |

## Selected topology

Best successful four-video pilot: `cp2_shard8_ep1` at 17.14 videos/hour including model load.

TP=8 is not supported by the native FSDP sampler and was reset to TP=1. EP=2 and EP=4 failed with CUDA OOM even when the loading keepalive was disabled.

## Full 256-prompt run

- Complete: True
- Output: 256 non-empty videos, 848x464, 121 frames, 50 steps
- Cluster: 8 nodes / 64 GPUs
- Topology: dp_shard=8, cp=2, ep=1, tp=1; 8 independent replicas over round-robin prompt shards
- Wall time including model load: 5589 s
- Actual cluster throughput including load: 164.90 videos/hour
- Estimated steady cluster throughput: 170.34 videos/hour
- Mean checkpoint load: 100.89 s
- Max peak GPU memory: 77.78 GiB
- Mean sampled GPU utilization: 97.43%
- GPU observations above 90%: 97.38%
