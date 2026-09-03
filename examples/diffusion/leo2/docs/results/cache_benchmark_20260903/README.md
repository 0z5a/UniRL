# Leo2 cache benchmark aggregate snapshot

This directory is the compact, machine-readable snapshot accompanying
[`CACHE_BENCHMARK_REPORT_20260903.md`](../../CACHE_BENCHMARK_REPORT_20260903.md).
It contains the final case summaries, paired latent/pixel metrics, VBench
per-video results, VideoScore2 scores/raw responses, and unified quality
summary. It does not contain the 1.3 GiB of MP4 and latent payloads.

The source root was:

```text
/root/leo2-output/cache-full-final-848x464x121-20260903-1039
```

Validate the snapshot from this directory with:

```bash
sha256sum -c SHA256SUMS
```

`quality_eval/summary/quality_metrics_cases.csv` is the machine-readable source
for the three per-flow-shift tables in the report. `benchmark.env` deliberately
retains the source run's absolute paths and launch-time fingerprints as
provenance. The complete videos, latents, per-case logs and replay commands
remain in the source root above.
