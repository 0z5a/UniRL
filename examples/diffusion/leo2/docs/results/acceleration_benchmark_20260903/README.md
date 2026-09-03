# Leo2 shift-9 acceleration results

This directory stores compact, reviewable snapshots for the acceleration
experiments. Generated MP4s and latents remain in the output roots listed
below and are intentionally not committed.

The committed report manifest resolves summary and evaluator inputs from this
repository. Its separate `source_root` fields preserve the original media
locations: regenerating with raw latent/video verification therefore requires
those output roots, while the published tables and their checksums remain
portable review artifacts.

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
paired-speed confidence intervals overlap.

Taylor cache residency is reported as N/A. This run predates residency
instrumentation, and the historical summarizer replaced its missing field
with zero; zero is not treated as a measurement.

## MagCache calibration and pilot

The shift-9 profile was calibrated with four prompts disjoint from the
benchmark prompts. Calibration ran all 50 denoise steps exactly for every
prompt. The resulting profile has SHA-256
`1fab27b88c5fd74a96de85c4e9f4cd29809289c2cd7df2cb0e143b1b3bef64b7`.

The pilot reused prompt indices 0, 8, 10 and 14. Pixel RMSE below is the mean
per-video RMSE against exact inference.

| Threshold / max consecutive reuse | Paired speedup | Reused steps | Latent rel L1 | Latent rel L2 | Pixel MAE | Pixel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.06 / 2 | 1.9476x | 54.0% | 0.17384 | 0.18129 | 0.03448 | 0.06325 |
| 0.12 / 4 | 2.3560x | 64.0% | 0.23511 | 0.23958 | 0.04592 | 0.08021 |

The 0.12 / 4 case was selected for the full-16 suite. On the pilot prompts it
matched the static 0.10 cache's quality envelope while increasing paired
throughput by about 21%.

## MagCache full-16 result

| Metric | Exact off | Static 0.10 | MagCache 0.12 / 4 |
|---|---:|---:|---:|
| Generation mean (s) | 198.118 | 100.179 | 82.623 |
| Paired speedup | 1.0000x | 1.9800x | 2.3996x |
| Reused denoise steps | 0.00% | 55.50% | 64.00% |
| Latent relative L1 | 0 | 0.21221 | 0.20629 |
| Latent relative L2 | 0 | 0.22196 | 0.20911 |
| Latent cosine | 1 | 0.97293 | 0.97642 |
| Latent max-abs worst case | 0 | 1.38048 | 1.01727 |
| Pixel MAE | 0 | 0.03796 | 0.03569 |
| Pixel mean RMSE | 0 | 0.06329 | 0.05905 |
| Pixel relative L1 | 0 | 0.10185 | 0.09655 |
| Pixel relative L2 | 0 | 0.14909 | 0.14043 |

Within this 16-prompt suite, MagCache increased paired throughput by about
21% over static 0.10 while improving every reported aggregate latent and pixel
drift metric. Its paired-speedup 95% confidence interval was 2.3776x--2.4217x.
The rank-max cache residency was 97,607,680 bytes and rank-max peak allocated
GPU memory was 69,513,028,096 bytes.

## FasterCache DFR smoke and pilot

The one-prompt smoke test exercised all 48 Leo attention blocks over the
half-open denoise window `[20, 45)` with interval 2. It reported the expected
38 exact and 12 reuse decisions, or 1,824 exact and 576 reused attention calls,
and passed exact latent pairing against the immutable reference root.

The four-prompt pilot compared the smoke window with the wider `[10, 49)`
window. Pixel RMSE below is the mean per-video RMSE against exact inference.
The reuse percentage counts managed-attention calls, not skipped full
transformer steps.

| DFR window / interval | Paired speedup | Attention reuse | Latent rel L1 | Latent rel L2 | Pixel MAE | Pixel RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| `[20, 45)` / 2 | 1.1467x | 24.0% | 0.04351 | 0.04731 | 0.01458 | 0.02365 |
| `[10, 49)` / 2 | 1.2586x | 38.0% | 0.13212 | 0.14222 | 0.02812 | 0.05290 |

The wider window was selected for the full-16 suite. It increased paired
throughput by about 9.8% over the narrower window while remaining inside the
static 0.10 cache's pilot latent and pixel drift envelope. Both pilot settings
used 4,686,348,288 bytes of rank-max DFR cache residency; the selected case's
rank-max peak allocated GPU memory was 74,107,523,072 bytes.

## FasterCache DFR full-16 result

| Metric | Exact off | Static 0.10 | DFR `[10, 49)` / 2 |
|---|---:|---:|---:|
| Generation mean (s) | 198.118 | 100.179 | 157.306 |
| Paired speedup | 1.0000x | 1.9800x | 1.2595x |
| Managed-attention reuse | 0.00% | 0.00% | 38.00% |
| Latent relative L1 | 0 | 0.21221 | 0.11203 |
| Latent relative L2 | 0 | 0.22196 | 0.12026 |
| Latent cosine | 1 | 0.97293 | 0.99202 |
| Latent max-abs worst case | 0 | 1.38048 | 0.88852 |
| Pixel MAE | 0 | 0.03796 | 0.02100 |
| Pixel mean RMSE | 0 | 0.06329 | 0.03639 |
| Pixel relative L1 | 0 | 0.10185 | 0.05743 |
| Pixel relative L2 | 0 | 0.14909 | 0.08665 |

Every full-suite request made 1,488 exact and 912 reused managed-attention
calls, matching the configured 31 exact and 19 reuse denoise decisions across
48 blocks. DFR had the smallest latent and decoded-pixel drift of the tested
accelerators, but only reached 1.2595x and retained 4,686,348,288 bytes
(4,469.25 MiB) of rank-maximum activation cache. Its peak allocated memory was
74,108,845,568 bytes (69.019 GiB), about 4.37 GiB above exact inference.

## Consolidated quality and performance

The complete settings-as-columns, metrics-as-rows result is available as
[`consolidated/shift9_guidance1.md`](consolidated/shift9_guidance1.md), with
CSV and JSON equivalents beside it. The compact view below uses global RMSE,
defined as `sqrt(MSE)` over all equal-size samples; this is intentionally
different from the mean per-video RMSE used in the method-specific tables.
The reporter recomputes timing and counters from request rows, latent metrics
from digest-checked tensors, pixel aggregates from paired rows, and VBench and
VideoScore2 aggregates from their raw per-video outputs.

| Metric | Exact off | Static 0.10 | Taylor 0.10 / 0.50 | MagCache 0.12 / 4 | DFR `[10, 49)` / 2 |
|---|---:|---:|---:|---:|---:|
| Paired speedup | 1.0000x | 1.9800x | 1.9756x | **2.3996x** | 1.2595x |
| Global latent RMSE | 0 | 0.03102 | 0.02928 | 0.02940 | **0.01753** |
| Global pixel RMSE | 0 | 0.06663 | 0.06221 | 0.06185 | **0.03935** |
| VBench subject consistency | 0.89380 | **0.89703** | 0.89433 | 0.89475 | 0.89688 |
| VBench background consistency | 0.92790 | 0.93022 | **0.93541** | 0.93451 | 0.92873 |
| VBench motion smoothness | 0.97972 | 0.98251 | 0.98049 | **0.98418** | 0.97956 |
| VBench dynamic degree (descriptive) | 0.8125 | 0.6875 | 0.7500 | 0.7500 | 0.8125 |
| VBench aesthetic quality | 0.51231 | 0.50729 | **0.51357** | 0.49626 | 0.50920 |
| VBench imaging quality | 0.57294 | 0.54648 | **0.57362** | 0.52751 | 0.57086 |
| VideoScore2 visual quality `[1,5]` | 3.1250 | 3.1875 | 3.0625 | **3.3750** | 3.0625 |
| VideoScore2 text alignment `[1,5]` | 3.3125 | 3.1875 | 3.3750 | **3.5000** | 3.2500 |
| VideoScore2 physical consistency `[1,5]` | 3.3750 | 3.3125 | 3.0625 | **3.4375** | 3.1250 |

For throughput, MagCache is the strongest candidate: it is about 21% faster
than static 0.10 and has slightly lower aggregate latent and pixel drift.
However, its lower VBench aesthetic/imaging scores conflict with its leading
VideoScore2 scores, so it requires human paired review before replacing the
preferred static setting. Taylor keeps essentially the same throughput as
static cache while reducing aggregate drift and retaining strong VBench
aesthetic/imaging scores, but it has the largest worst-pair latent max-absolute
error. DFR is the fidelity-oriented point on the frontier, with the smallest
drift and exact-level dynamic-degree rate, at the cost of lower speed and
roughly 4.37 GiB extra peak allocation.

These quality results are VBench custom-input scores over 16 prompts, not
leaderboard scores. Dynamic degree is the fraction of clips classified as
sufficiently dynamic and is descriptive rather than a monotonic quality
measure. VideoScore2 reports discrete hard scores, so differences of 0.0625
represent one point over 16 videos. The sample is too small for fine-grained
ranking; metric disagreement and worst-pair videos should be resolved by a
blinded human A/B. CFG-cache at guidance 5.0 remains a separate deferred
experiment and is not included here. Evaluator versions and script hashes are
recorded in [`EVALUATION_PROVENANCE.md`](EVALUATION_PROVENANCE.md).

From the repository root, regenerate and deeply validate the report with:

```bash
python examples/diffusion/leo2/scripts/summarize_acceleration_results.py \
  --spec examples/diffusion/leo2/docs/results/acceleration_benchmark_20260903/consolidated/report_spec.json \
  --output-dir examples/diffusion/leo2/docs/results/acceleration_benchmark_20260903/consolidated \
  --name shift9_guidance1 \
  --force
```

## Artifact roots

- Taylor 0.5/1.0/2.0 pilot:
  `/root/leo2-output/accel-taylor-pilot-shift9-20260903-223454`
- Taylor 0.25 pilot:
  `/root/leo2-output/accel-taylor-pilot-m025-shift9-20260903-2306`
- Taylor 0.5 full-16:
  `/root/leo2-output/accel-taylor-full-m050-shift9-20260903-2316`
- MagCache calibration:
  `/root/leo2-output/accel-magcache-calibration-shift9-20260904-0001`
- MagCache pilot:
  `/root/leo2-output/accel-magcache-pilot-shift9-20260904-0018`
- MagCache 0.12 / 4 full-16:
  `/root/leo2-output/accel-magcache-full-t012-k4-shift9-20260904-0038`
- FasterCache DFR `[20, 45)` smoke:
  `/root/leo2-output/accel-dfr-smoke-w20-45-shift9-20260904-0107`
- FasterCache DFR pilot:
  `/root/leo2-output/accel-dfr-pilot-shift9-20260904-0114`
- FasterCache DFR `[10, 49)` / 2 full-16:
  `/root/leo2-output/accel-dfr-full-w10-49-shift9-20260904-0142`
- Consolidated report generated from the five full-16 cases:
  `/root/leo2-output/accel-consolidated-shift9-g1-20260904`

Each snapshot subdirectory retains the cases CSV, `benchmark.env`, case/root
summary, paired latent metrics and paired/aggregate pixel metrics. The three
selected accelerator directories additionally retain VBench and VideoScore2
results; exact and static quality artifacts remain in the sibling
`cache_benchmark_20260903` snapshot and the immutable reference root.
