# Leo2 DiffusionNFT GPU-bubble optimization plan

## Scope and baseline

This plan targets the UniRL trainside Leo2 DiffusionNFT recipe only. Flow-Factory
remains an algorithm reference and is not modified.

The 2026-09-05 baseline used 8×H20, CP=2, EP=8 with DeepEP, 48 prompt groups,
16 samples per group, 17 frames, and 10 rollout steps. Before termination it
completed 364 DiT forwards on each CP pair. The instrumented forwards consumed
1909.52 seconds; direct `nvidia-smi` sampling measured 81.76% mean utilization,
or 18.24% utilization-derived GPU bubble. Low-utilization windows occurred at
video boundaries rather than during steady DiT forwards.

## Implemented P0 changes

1. **Make per-forward profiling opt-in.**
   `Leo2DiffusionStage.predict_noise` previously synchronized CUDA before and
   after every forward even in production. `profile_forward=false` now avoids
   these benchmark-only synchronization points.
2. **Select the available grouped-GEMM path directly.**
   The launcher sets `HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM=0`. This avoids a
   failed CUTLASS attempt and one repeated log emission per MoE layer per
   forward while retaining the already validated `EPGroupGemm` implementation.
3. **Reuse frozen sibling conditions.**
   A bounded prompt/geometry/seed LRU in `Leo2CondStage` shares immutable tensor
   storage while cloning mutable containers. Prompt-major groups therefore
   encode once and reuse conditions for the remaining siblings. The production
   group size of 16 has a theoretical 15/16 (93.75%) cache-hit rate.
4. **Keep the frozen text encoder resident.**
   The recipe disables transient CPU/GPU shuttling. This adds about 15.6 GiB per
   GPU (24.5 GiB to 40.1 GiB pre-rollout allocation in the benchmark) but the
   full rollout and random-two-step training smoke completed without OOM.
5. **Activate request-scoped Diffusers first-block cache in UniRL rollout.**
   Rollout now enters `model.cache_context`; replay/training remains exact.
   Cache decisions are reduced across CP, EP, and actual FSDP groups, so every
   rank participating in DeepEP takes the same full/reuse branch.

The restarted 48×16 production rollout was sampled for 640 GPU observations
over 78.9 seconds after multiple trajectories had completed. Mean utilization
was 97.92%, corresponding to 2.08% utilization-derived bubble; only 0.94% of
samples were below 50% utilization. Relative to the 18.24% baseline bubble,
this is a 16.16 percentage-point or 88.60% reduction. The threshold-0.1 cache
continued to record zero reuse during this probe, so the utilization gain is
from the P0 scheduling/condition/logging changes rather than skipped DiT work.

## Validation

- `tests/test_leo2_diffusionnft.py`: 20 passed.
- Added coverage for the threshold-0.1 Diffusers cache config, request-scoped
  rollout cache context, sibling-condition reuse/container isolation, and the
  complete requested recipe.
- Real 8-GPU CP2/EP8/DeepEP A/B completed both rollout and training, validating
  that the synchronized cache branch cannot deadlock DeepEP and that training
  does not use inference cache.

## P1/P2 follow-up work

1. **P1: batch or defer VAE decoding.** The current batch-one loop switches from
   DiT to VAE and copies each decoded video to CPU after every trajectory.
   Accumulating a bounded latent batch and decoding it together should reduce
   the dominant remaining video-boundary bubble. It requires a memory sweep and
   output-order tests.
2. **P1: make group-level conditioning explicit.** The P0 LRU is safe and local,
   but a first-class prompt-group condition object would remove reliance on
   prompt-major ordering and expose hit/miss metrics to W&B.
3. **P2: overlap reward scoring with later rollout chunks.** This requires
   stream/process isolation because reward and Leo2 currently share the same
   GPUs. It should be evaluated only after VAE batching.
4. **P2: investigate batch>1 packed Leo2 metadata.** The stage currently rejects
   batch>1 because hymm metadata lacks a model batch axis. Enabling it could
   increase arithmetic intensity, but it is substantially higher risk than
   bounded VAE batching.

## Decision rule

Performance changes must use the same checkpoint, prompts/seeds, topology,
geometry, step count, guidance, and cache setting. Report rollout lifecycle
latency separately from model-load and training latency. A cache candidate is
an acceleration only when it records non-zero reuse and has speedup greater
than 1.0×; an enabled zero-hit cache is reported as overhead, not acceleration.
