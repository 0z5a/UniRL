# Leo2 cache benchmark

This harness compares request-scoped Leo2 acceleration methods with an explicit
reference case. Current runs use paired English/Chinese Context-IR prompts, one
sample at a time, on eight ranks with FSDP shard size 8, CP size 8, and
EP/ETP/TP/PP size 1.
The native media size is 848x464 (`--image-size 464x848`), every video contains
193 frames at Leo2's normal 24 FPS (`(193 - 1) / 24 = 8` seconds). The
committed matrices fix flow shift 9 and independently cover 6, 28 and 50
denoising steps at guidance scales 1.0 and 5.0.

Each request starts with a distributed barrier before timing. Rank zero encodes
the preceding MP4 outside `generate_video`; the barrier prevents other ranks
from charging that work to the next sample. The measured interval includes
model-input preparation, text conditioning, denoising and VAE decode, but not
model load, MP4 encoding or filesystem writes.

## Case schema

Schema v2 adds `method`, `guidance_scale`, `baseline_case` and
`reference_root`. New matrices also bind `diff_infer_steps`; legacy generated
rows infer 6/28/50 from the `_sN_` case name, and the launcher rejects a
mismatched `LEO2_INFER_STEPS`. Supported methods and their specific columns are:

- `off`: no method-specific values.
- `first_block`: `cache_threshold`.
- `taylor`: `cache_threshold` and `taylor_max_extrapolation` (default 1.0).
- `magcache`: `magcache_profile`, `magcache_threshold`,
  `magcache_max_skip_steps` (default 4), and
  `magcache_retention_ratio` (default 0.2).
- `magcache_calibrate`: the MagCache fields, with an optional input profile.
- `fastercache_dfr`: `dfr_start_step`, `dfr_end_step`, `dfr_interval`, and
  optional `dfr_layers`, using syntax such as `0-7;16;31-47`.
- `cfg_cache`: the `cfg_*` window, interval, and low/high-frequency weights.
- `fastercache_dfr+cfg_cache`: both sets of DFR and CFG fields.
  Formal 193-frame composite cases use layers 24–47 to keep peak memory within
  8×H20; an omitted layer list is normalized to that bound.

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
Formal runs require 24 FPS and batch size one. The CFG-output controller rejects
larger base batches rather than guessing the cond/uncond split.

## Validation and outputs

Each case contains `samples/`, final `latents/`, `run.log`, replayable
`command.sh`, `case.env`, `exit_code.txt` and `summary.json`. The root contains
Git, diff, input and artifact fingerprints in `benchmark.env`, plus
`summary.csv`, `paired_metrics.csv` and `summary.json`.

Request latency is CUDA-synchronized and reduced with MAX across ranks. Cache
decision counters must have identical MIN and MAX. `cache_bytes` is rank-local
under CP because padding can differ, so it follows peak CUDA memory and reports
MAX. Whole-tail methods validate exactly the requested total tail
compute/reuse steps. FasterCache DFR validates the requested total exact/reuse
steps against its window and
interval, including the first-candidate warm-up fallback when `start_step=0`.
Its managed-attention compute/reuse calls must equal those step counts times the
selected-layer count. Reports retain legacy full/skip counters and add tail
compute/reuse, Taylor prediction/fallback, attention compute/reuse, CFG
compute/reuse, cache bytes, and peak CUDA allocation/reservation.

Taylor requests also record prediction warm-up count and the mean/maximum
extrapolation coefficient. A `magcache_calibrate` request records its complete
step-indexed magnitude-ratio and scheduler-timestep arrays. Scalar counters and
fixed-size diagnostics are checked across all ranks independently; a rank count,
array length, or value mismatch aborts the request before rank zero emits it.

Every MP4 must decode as exactly 848x464, 193 frames and 24 FPS. Speed and final
latent drift are paired with the explicit baseline by prompt hash, seed, steps,
guidance, language and pair ID.
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

## Consolidated acceleration report

After quality evaluation, compare completed cases from independent roots with
an explicit JSON manifest:

```json
{
  "schema_version": 1,
  "expected": {
    "flow_shift_video": 9.0,
    "guidance_scale": 1.0,
    "sample_count": 16,
    "baseline_label": "exact"
  },
  "cases": [
    {
      "label": "exact",
      "case": "cache_off_shift9",
      "artifact_root": "/path/to/exact-snapshot",
      "source_root": "/path/to/exact-run"
    },
    {
      "label": "static-0.10",
      "case": "cache_t010_shift9",
      "artifact_root": "/path/to/static-root"
    }
  ]
}
```

Paths may be absolute or relative to the manifest. Each artifact root must
contain `benchmark.env`, `summary.json`, `paired_metrics.csv`, and
`quality_eval/summary/quality_metrics_cases.json`. Accelerated cases also need
`pixel_metrics_pairs.csv`; the exact baseline has zero pixel drift by
definition. A case entry may override any result path with `summary_json`,
`paired_metrics_csv`, `pixel_metrics_pairs_csv`, or `quality_metrics_json`.
`source_root` defaults to `artifact_root`; set it explicitly when compact
metrics were copied away from the original run. Recorded baseline and artifact
paths must agree with these declared source roots. Paired source videos must
remain readable; schema-v2 video digests are rechecked while building the
report.

```bash
python examples/diffusion/leo2/scripts/summarize_acceleration_results.py \
  --spec /path/to/shift9-guidance1.json \
  --output-dir /path/to/report
```

The command writes CSV, Markdown, and JSON matrices with settings as columns
and metrics as rows. Use one manifest per flow shift so each shift produces a
separate table. It rejects incomplete cases, mixed prompts, sampling
settings, topology, checkpoint/config fingerprints, guidance, flow shift,
sample count, or baseline identity. It also recomputes decoded-pixel global
MSE from the per-video pair records before accepting a quality summary.

## Gotchas

- `pixel_metrics_cases.csv` reports the mean of per-video RMSE values. The
  consolidated report instead uses global MSE over all decoded RGB samples and
  its square root, so the two RMSE values need not be equal.
- Schema-v1 benchmark artifacts did not serialize guidance. Their fixed
  guidance-1.0 contract is accepted only when the report manifest also requires
  guidance 1.0; schema-v2 artifacts must record guidance explicitly.

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
