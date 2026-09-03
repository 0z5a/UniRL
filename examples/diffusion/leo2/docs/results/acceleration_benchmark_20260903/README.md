# Leo2 shift-9 acceleration results

This directory stores compact, reviewable snapshots for the acceleration
experiments. Generated MP4s and latents remain in the output roots listed
below and are intentionally not committed.

All evaluation cases use the same 16 prompt/seed pairs, 848x464 output, 121
frames, 24 FPS, 50 Euler FlowMatch steps, guidance 1.0, `flow_shift_video=9`,
CP=8, FSDP shard=8, and EP=ETP=TP=PP=1. The immutable exact and static-cache
reference root is
`/root/leo2-output/cache-full-final-848x464x121-20260903-1039`.

## Taylor pilot

The first pilot used prompt indices 0, 8, 10 and 14. A follow-up added the
more conservative 0.25 cap. Pixel RMSE below is the mean per-video RMSE.

| Max extrapolation | Paired speedup | Reused steps | Latent rel L1 | Latent rel L2 | Pixel MAE | Pixel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| static first-block | 1.9430x | 55.5% | 0.23058 | 0.23979 | 0.04641 | 0.08091 |
| 0.25 | 1.9667x | 55.5% | 0.23521 | 0.24705 | 0.04818 | 0.08430 |
| 0.50 | 1.9609x | 55.5% | 0.22546 | 0.23851 | 0.04606 | 0.08105 |
| 1.00 | 1.9453x | 55.0% | 0.24516 | 0.25929 | 0.05086 | 0.08721 |
| 2.00 | 1.9589x | 55.5% | 0.24388 | 0.25508 | 0.04997 | 0.08457 |

`max_extrapolation=0.50` was the only pilot setting retained for the full
suite. The speed differences in this four-video pilot are not interpreted as
significant because Taylor changes only the reused residual arithmetic, not
the expensive block count.

## Taylor full-16 result

| Metric | Exact off | Static 0.10 | Taylor 0.10 / 0.50 |
|---|---:|---:|---:|
| Generation mean (s) | 198.118 | 100.179 | 100.342 |
| Paired speedup | 1.0000x | 1.9800x | 1.9756x |
| Reused denoise steps | 0.00% | 55.50% | 55.25% |
| Latent relative L1 | 0 | 0.21221 | 0.19018 |
| Latent relative L2 | 0 | 0.22196 | 0.20029 |
| Latent cosine | 1 | 0.97293 | 0.97676 |
| Latent max-abs worst case | 0 | 1.38048 | 1.54487 |
| Pixel MAE | 0 | 0.03796 | 0.03363 |
| Pixel mean RMSE | 0 | 0.06329 | 0.05726 |
| Pixel relative L1 | 0 | 0.10185 | 0.09046 |
| Pixel relative L2 | 0 | 0.14909 | 0.13463 |

Taylor preserves the static cache's approximately 1.98x speedup while
reducing mean latent and decoded-pixel drift by roughly 10%. It does not
strictly dominate: the worst latent max-absolute error is larger, and the
paired-speed confidence intervals overlap. VBench and VideoScore2 are run only
for full-16 survivors and will be added to the consolidated report after the
MagCache and FasterCache DFR funnels finish.

## Artifact roots

- Taylor 0.5/1.0/2.0 pilot:
  `/root/leo2-output/accel-taylor-pilot-shift9-20260903-223454`
- Taylor 0.25 pilot:
  `/root/leo2-output/accel-taylor-pilot-m025-shift9-20260903-2306`
- Taylor 0.5 full-16:
  `/root/leo2-output/accel-taylor-full-m050-shift9-20260903-2316`

Each snapshot subdirectory retains the cases CSV, `benchmark.env`, case/root
summary, paired latent metrics and paired/aggregate pixel metrics.
