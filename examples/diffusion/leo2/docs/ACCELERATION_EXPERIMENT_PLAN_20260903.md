# Leo2 shift-9 acceleration experiment plan

## Fixed comparison contract

All main experiments reuse the 16 prompts and seeds in
`examples/diffusion/leo2/data/cache_benchmark_16.csv` (SHA-256
`9f7a596d1e7dd37a2f1a9ddffe566591e04f232be9fb4c4258d1231d5d18bfcc`).
Sampling stays at 848x464, 121 frames, 24 FPS, 50 Euler FlowMatch steps,
`flow_shift_video=9`, `guidance_scale=1.0`, batch size one, CP=8, FSDP shard=8,
and EP=ETP=TP=PP=1. Cache methods are mutually exclusive and native-inference
only.

The immutable reference run is
`/root/leo2-output/cache-full-final-848x464x121-20260903-1039`:

- `cache_off_shift9`: exact baseline, 16/16 complete.
- `cache_t010_shift9`: selected first-block baseline, 16/16 complete.

New runs must record the reference root, baseline case, source Git commit,
method configuration digest, prompt digest, checkpoint inventory digest and
runtime inventory. A new output root is mandatory; launchers must refuse to
overwrite an existing run.

## Experiment funnel

### Taylor tail prediction

1. Run four prompts (indices 0, 8, 10 and 14) at threshold 0.10 with
   `max_extrapolation` 0.5, 1.0 and 2.0.
2. Compare all three against `cache_off_shift9`; decision hits should remain
   close to `cache_t010_shift9`, while prediction should reduce paired drift.
3. Run the best one or two settings on all 16 prompts.

### MagCache

1. Collect full-forward residual ratios on at least four prompts disjoint from
   the evaluation set, then aggregate per-step mean, standard deviation and
   p95 into a versioned profile.
2. Keep retention ratio at 0.2 and pilot `(threshold, max_skip_steps)` settings
   `(0.06, 2)`, `(0.12, 4)` and `(0.18, 4)`.
3. Run the Pareto candidate closest to or better than the selected first-block
   baseline's skip ratio on all 16 evaluation prompts.

### FasterCache dynamic feature reuse

1. Run a one-prompt all-layer memory and correctness smoke with interval two.
2. Pilot windows `[20, 45)` and `[10, 49)` on the same four evaluation prompts.
3. If all-layer history is too costly, record and test an explicit layer subset;
   never silently change the selected layers.
4. Run the surviving setting on all 16 prompts.

### CFG-Cache follow-up

CFG-Cache is a separate `guidance_scale=5.0` experiment. It requires its own
cache-off shift-9 baseline and must not compare against the guidance-1.0
reference. Leo's native batch order is conditional then unconditional, which
must be preserved explicitly in the implementation.

## Measurements and gates

Every pilot records end-to-end request latency, denoise/cache counters, process
wall time, peak allocated/reserved memory and rank-wise counter consensus.
Every full case additionally records paired latent and decoded-RGB L1/L2/MSE,
RMSE, cosine similarity, VBench dimensions and VideoScore2 dimensions.

No method becomes a default solely from aggregate VBench. In particular,
`dynamic_degree` is a coarse per-video dynamic/static rate on a 16-video suite.
Select methods from the speed/quality Pareto frontier, inspect the worst paired
videos and retain all method-specific counters rather than converting every
method to one generic skip ratio.

## Commit sequence

1. `feat(leo2-cache): add sigma-aware Taylor tail prediction`
2. `bench(leo2): generalize acceleration benchmark cases`
3. `bench(leo2-cache): record Taylor shift-9 results`
4. `feat(leo2-cache): add MagCache calibration and replay`
5. `bench(leo2-cache): record MagCache shift-9 results`
6. `feat(leo2-cache): add FasterCache dynamic attention reuse`
7. `bench(leo2-cache): record FasterCache shift-9 results`
8. `feat(leo2-cache): add CFG output reuse`
9. `bench(leo2-cache): record guidance-5 CFG-cache results`

Implementation and benchmark-result commits remain separate. Generated MP4s,
latents, weights and runtime archives stay outside Git; only compact manifests,
metrics and reports are committed.
