# Leo2 first-block cache benchmark

This harness compares Diffusers-compatible first-block caching with an exact
cache-off baseline. Every case uses the same 16 prompts and seeds, one sample at
a time, on the same eight-rank FSDP/CP topology. The native media size is
`848x464`, expressed to hymm as `--image-size 464x848`, and every video contains
121 frames.

The benchmark pins EP/ETP/TP/PP to one, CP to eight and the FSDP shard group to
eight. Leo2 cache decisions support CP plus the actual FSDP shard group, but
conservatively fall back when EP, ETP, TP or PP is greater than one. The entry
point resolves the real communication plan after wrapping and exits before the
first video if caching would be ineffective. The pilot's positive-threshold case
also requires at least one skipped step; threshold zero must skip no steps and
must reproduce the cache-off final latent exactly. Full cases retain zero-hit
requests as valid measurements.

Every request starts with a distributed barrier before its timer. Rank zero
encodes the preceding MP4 outside `generate_video`; without this barrier, the
other ranks would start the next request early and charge that encoding delay to
the next sample. MP4 encoding and filesystem writes are therefore excluded from
the measured generation time on every rank.

The launcher uses an output-local Hugging Face cache and explicitly keeps the
node's validated IB/RDMA path enabled (`NCCL_IB_DISABLE=0`) by default, so
unrelated parent-shell cache and communication settings cannot silently change
a run. Override these only with
`LEO2_CACHE_BENCH_HF_HOME` or `LEO2_CACHE_BENCH_NCCL_IB_DISABLE`.

`flow_shift_video` is the scheduler/latent-trajectory shift referred to as
“latent shift” in benchmark discussions. It is not the VAE latent normalization
offset. The committed matrix holds shift 9 constant for the main threshold
sweep, then repeats threshold 0.05 at shifts 7 and 3.

## Run

Set the same portable artifact variables as native Leo2 inference:

```bash
export LEO2_RUNTIME_PYTHON=/path/to/leo2/bin/python
export LEO2_CKPT_DIR=/path/to/iter_xxxxxxx/weights
export LEO2_ASSETS_BASE=/path/to/hymm_ar_assets
bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

The default eight-case matrix generates 128 videos and reloads the model for
every case. This makes logs and process-level timings independent, but it is
expensive. To run a smaller pilot, copy
`data/cache_benchmark_cases.csv`, retain the desired rows, and set
`LEO2_CACHE_BENCH_CASES` to that file. `LEO2_INFER_STEPS` defaults to 50 and
must stay identical across paired cases. Set `LEO2_CACHE_BENCH_OUTPUT` to choose
an explicit new output root; the launcher refuses to reuse an existing root.

Before the full matrix, run the built-in three-case, one-prompt pilot. It checks
cache off, threshold-zero correctness and threshold 0.05 effectiveness at shift
9 without editing the cases CSV:

```bash
LEO2_CACHE_BENCH_PILOT=1 \
  bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

Use a dry run to inspect all commands without accessing a GPU or model asset:

```bash
LEO2_CACHE_BENCH_DRY_RUN=1 \
  LEO2_CACHE_BENCH_OUTPUT=/tmp/leo2-cache-dry-run \
  bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

## Outputs

Each case has its own `samples/`, final `latents/`, `run.log`, replayable
`command.sh`, settings, exit code and `summary.json`. The benchmark root contains
Git/diff/artifact fingerprints in `benchmark.env`, `gpu_inventory.csv`,
`summary.csv`, `paired_metrics.csv` and `summary.json`. Each request records its
prompt index/hash, seed, effective topology/runtime and latent digest. Request
latency is CUDA-synchronized and reduced with `MAX` across all eight ranks;
cache counters must have identical `MIN` and `MAX`. The summary reports
all-request latency and a steady mean that excludes request one as warm-up,
basic 95% confidence intervals, full/skip step counts, skip ratio and peak CUDA
allocation/reservation.

The case summary also reports the complete torchrun process wall time, including
model load and media encoding. Every MP4 is decoded by the packaged
`imageio-ffmpeg` binary and must be exactly 848×464, 121 frames and 24 FPS before
the case is marked complete.

Speedup and final-latent drift are paired by prompt hash against the complete
cache-off case with the same flow shift. The committed matrix includes matched
off/0.05 cases at shifts 3 and 7, plus off/0.02/0.05/0.10 at shift 9. Drift is
reported as relative L1, relative L2, cosine similarity and maximum absolute
difference. Saved latents are the denormalized tensors passed to VAE decode.

After the final summary is complete, decoded RGB pixel-space metrics can be
computed without loading Torch or a GPU. The tool strictly validates every
candidate/baseline pair as 848×464, 121 frames and 24 FPS, then writes per-pair
and per-case MAE, RMSE, relative L1/L2 and maximum absolute error:

```bash
python examples/diffusion/leo2/scripts/compute_pixel_metrics.py \
  --root /path/to/completed-benchmark \
  --output-dir /path/to/new-output-directory
```

The measured `generate_video` interval includes prompt preparation performed
inside that call, denoising and VAE decode. Model load and MP4 encoding remain
visible in `run.log` and process wall time but are outside per-request latency.
Inspect paired videos as well as speed: a higher skip ratio is not by itself a
quality result.
