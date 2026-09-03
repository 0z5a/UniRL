# Leo2 cache benchmark

This harness compares request-scoped Leo2 acceleration methods with an explicit
reference case. Full runs use the same 16 prompts and seeds, one sample at a
time, on eight ranks with FSDP shard size 8, CP size 8, and EP/ETP/TP/PP size 1.
The native media size is 848x464 (`--image-size 464x848`), every video contains
121 frames at 24 FPS, and the committed matrix fixes 50 denoising steps, flow
shift 9, and guidance scale 1.0.

Each request starts with a distributed barrier before timing. Rank zero encodes
the preceding MP4 outside `generate_video`; the barrier prevents other ranks
from charging that work to the next sample. The measured interval includes
model-input preparation, text conditioning, denoising and VAE decode, but not
model load, MP4 encoding or filesystem writes.

## Case schema

Schema v2 adds `method`, `guidance_scale`, `baseline_case` and
`reference_root`. Supported methods and their specific columns are:

- `off`: no method-specific values.
- `first_block`: `cache_threshold`.
- `taylor`: `cache_threshold` and `taylor_max_extrapolation` (default 1.0).
- `magcache`: `magcache_profile`, `magcache_threshold`,
  `magcache_max_skip_steps` (default 4), and
  `magcache_retention_ratio` (default 0.2).
- `magcache_calibrate`: the MagCache fields, with an optional input profile.
- `fastercache_dfr`: `dfr_start_step`, `dfr_end_step`, `dfr_interval`, and
  optional `dfr_layers`, using syntax such as `0-7;16;31-47`.

Unused method-specific fields must be empty. Original three-column
`name,cache_threshold,flow_shift_video` matrices remain accepted: they infer
`off`/`first_block`, guidance 1.0 and a same-shift local cache-off baseline.

An explicit `reference_root` lets a new root contain only new cases.
`baseline_case` is loaded read-only from that root. Prompt hash, seed, index,
flow shift, guidance, step count, latent digest, video path and video geometry
are validated before pairing. A new four-prompt case may use a completed
16-prompt case as its reference.

## Run

Set the portable artifact variables used by native Leo2 inference:

```bash
export LEO2_RUNTIME_PYTHON=/path/to/leo2/bin/python
export LEO2_CKPT_DIR=/path/to/iter_xxxxxxx/weights
export LEO2_ASSETS_BASE=/path/to/hymm_ar_assets
bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

The default matrix reruns cache-off and the preferred threshold-0.10
first-block case at shift 9. Set `LEO2_CACHE_BENCH_CASES` to run another matrix
and `LEO2_CACHE_BENCH_OUTPUT` to select a new output root. Existing output roots
are always rejected. `LEO2_INFER_STEPS` defaults to 50 and must match the
reference.

The expected case size defaults to the prompt CSV row count. For a four-prompt
method pilot without rerunning a baseline, make the case row reference the old
root and run the normal mode with the committed diverse subset:

```bash
LEO2_CACHE_BENCH_PROMPTS=examples/diffusion/leo2/data/cache_benchmark_pilot_4.csv \
LEO2_CACHE_BENCH_CASES=/path/to/new-method-cases.csv \
LEO2_CACHE_BENCH_OUTPUT=/path/to/new-output \
  bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

`LEO2_CACHE_BENCH_EXPECTED_VIDEOS` can select the first N rows explicitly. The
legacy built-in one-prompt correctness pilot remains available:

```bash
LEO2_CACHE_BENCH_PILOT=1 \
  bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

Inspect commands without loading a model or using a GPU:

```bash
LEO2_CACHE_BENCH_DRY_RUN=1 \
LEO2_CACHE_BENCH_OUTPUT=/tmp/leo2-cache-dry-run \
  bash examples/diffusion/leo2/scripts/cache_benchmark.sh
```

The launcher uses an output-local Hugging Face cache and leaves the validated
IB/RDMA path enabled by default. Override those settings only through
`LEO2_CACHE_BENCH_HF_HOME` or `LEO2_CACHE_BENCH_NCCL_IB_DISABLE`.

## Validation and outputs

Each case contains `samples/`, final `latents/`, `run.log`, replayable
`command.sh`, `case.env`, `exit_code.txt` and `summary.json`. The root contains
Git, diff, input and artifact fingerprints in `benchmark.env`, plus
`summary.csv`, `paired_metrics.csv` and `summary.json`.

Request latency is CUDA-synchronized and reduced with MAX across ranks. Cache
decision counters must have identical MIN and MAX. `cache_bytes` is rank-local
under CP because padding can differ, so it follows peak CUDA memory and reports
MAX. Whole-tail methods validate exactly 50 total tail compute/reuse steps.
FasterCache DFR uses attention counters and is not forced into that invariant.
Reports retain legacy full/skip counters and add tail compute/reuse, Taylor
prediction/fallback, attention compute/reuse, CFG compute/reuse, cache bytes,
and peak CUDA allocation/reservation.

Taylor requests also record prediction warm-up count and the mean/maximum
extrapolation coefficient. A `magcache_calibrate` request records its complete
50-value magnitude-ratio and scheduler-timestep arrays. Scalar counters and
fixed-size diagnostics are checked across all ranks independently; a rank count,
array length, or value mismatch aborts the request before rank zero emits it.

Every MP4 must decode as exactly 848x464, 121 frames and 24 FPS. Speed and final
latent drift are paired with the explicit baseline by prompt hash and seed.
Drift includes relative L1/L2, cosine similarity and maximum absolute error.

Compute decoded RGB metrics after a successful run:

```bash
python examples/diffusion/leo2/scripts/compute_pixel_metrics.py \
  --root /path/to/completed-benchmark \
  --output-dir /path/to/new-output-directory
```

Pixel metrics follow the same local or external baseline and record SHA-256 for
both encoded videos. `run_quality_evaluation.sh` evaluates only cases from the
selected cases CSV. The quality summarizer can merge an external baseline's
existing `quality_eval/summary/quality_metrics_cases.json` while keeping new
case results under the new root.

## MagCache calibration profile

Run `magcache_calibrate` without a replay profile using the four disjoint
prompts in `data/magcache_calibration_4.csv`, then build the portable JSON
consumed by a `magcache` case:

```bash
python examples/diffusion/leo2/scripts/build_magcache_profile.py \
  --root /path/to/completed-calibration-root \
  --case magcache_calibrate_shift9 \
  --output /path/to/new/magcache_shift9_profile.json
```

The builder refuses to replace an existing output. Before writing, it verifies
the successful case summary and exit code, the 50-step request accounting,
unique prompt identities, identical timesteps, positive finite ratios with
`ratio[0] == 1`, and the recorded Git, checkpoint, prompt, cases, model,
generation-config and artifact-manifest fingerprints. It aggregates ratios
independently at every step. Top-level `ratios` is the arithmetic mean and is
directly loader-compatible; `ratio_statistics` additionally records sample
standard deviation and nearest-rank p95 together with their definitions.
