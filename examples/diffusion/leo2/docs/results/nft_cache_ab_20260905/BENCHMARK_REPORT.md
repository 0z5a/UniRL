# Leo2 DiffusionNFT first-block cache A/B (2026-09-05)

## Result

Diffusers first-block cache at `threshold=0.1` is functional and distributed-safe
under CP=2, EP=8, and DeepEP, but it does not accelerate the current 10-step,
shift-3 rollout. It recorded zero cache hits and added 2.68% rollout latency.

- Rollout lifecycle: cache off 251.179 s; first-block 0.1 257.921 s;
  0.974× speedup, or 2.68% slower.
- Rollout throughput: 3.822 versus 3.722 videos/min, a 2.61% decrease.
- Training lifecycle: 177.6 versus 181.5 s. Cache is deliberately inactive in
  training, so this 2.20% difference is run-to-run variation.
- Mean reward: 0.7018 in both cases.
- Cache decisions: 160 full and zero reused steps in both cases.
- Rank-local cache storage: zero versus 327,421,952 bytes (+312.25 MiB).

The cache counter printed by rank 0 was `full_steps=10, skipped_steps=0` for each
of its four trajectories. The EP-wide decision is shared by all four DP lanes,
so the complete 16-video run executed 160 full and zero reused denoising steps.
With no approximation applied, matching reward means are expected; no quality
claim beyond this exact zero-hit behavior is needed.

## Production GPU-bubble probe

After applying the P0 changes, the restarted 48-group × 16-sample run was
sampled for 640 GPU observations over 78.9 seconds. It had already completed
multiple trajectories, and cache telemetry still showed zero reuse.

- Baseline mean utilization/bubble: 81.76% / 18.24%.
- Optimized mean utilization/bubble: 97.92% / 2.08%.
- Change: bubble decreased by 16.16 percentage points, or 88.60% relative.
- Per-GPU optimized mean utilization: 96.83% to 98.51%.
- Samples below 50% utilization: 0.94%.

This probe isolates full-compute behavior because threshold 0.1 did not skip
any DiT step. It indicates that removing forced synchronization, repeated
failed-kernel logging, condition re-encoding, and text-encoder transfers
materially reduced the observed rollout bubble.

## Protocol

- Hardware: one node, 8×NVIDIA H20.
- Topology: CP=2, DP=4, EP=8, DeepEP enabled.
- Model and data: the same Leo2 checkpoint and `vid_prompt` source as the long
  run.
- Per case: 4 prompt groups × 4 sibling videos = 16 videos.
- Sampling: 848×464, 17 frames, 10 inference steps, guidance 1.0, flow shift 3.
- Training: DiffusionNFT random 2-step, beta 1.0, one update.
- Both cases used the same P0 bubble changes: resident text encoder,
  one-entry sibling condition cache, production profiling disabled, and direct
  `EPGroupGemm` fallback.
- Model load is excluded from lifecycle latency.

## Interpretation

The earlier standalone 50-step, shift-9 cache benchmark reached 55.5% reuse and
1.98× speedup at threshold 0.1. That result does not transfer to this 10-step,
shift-3 training rollout: adjacent denoising states are farther apart, and the
EP-safe policy uses the maximum decision score across concurrently processed
samples. Threshold 0.1 therefore never accepts a tail reuse.

The requested cache setting can still be launched for contract validation, but
it should not be described as an acceleration for this recipe. For useful cache
speedup, the next experiment must change either the denoising schedule/step
count or the cache method/threshold, followed by a new quality A/B.

## Artifacts and tests

- Raw A/B logs:
  `/root/UniRL/outputs/leo2/nft_cache_ab_20260905_1045/{none,first_block}.log`
- Unit tests: `20 passed` in `tests/test_leo2_diffusionnft.py`.
- Both 8-GPU cases completed rollout, reward, random-two-step training, and
  optimizer update without deadlock or OOM.
