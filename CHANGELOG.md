# Changelog

## 2026-09-08

### Fixed

- Made Leo2 joint audio-video SDE sampling use the request-scoped timestep generator for both modalities. This keeps audio trajectories identical across context-parallel replicas and prevents later transformer forwards from mixing divergent audio states.
- Preserved the stored trajectory dtype at Leo2 scheduler boundaries so rollout and replay score the same BF16 transition instead of comparing an FP32 rollout transition with its stored BF16 value.
- Limited the FlowGRPO rollout/replay parity threshold to the first optimizer update, while retaining drift metrics on later updates and failing closed on non-finite values.

### Validation

- A checkpoint-40 replay on 64 GPUs with context parallelism 2 produced bit-identical video and audio predictions and log-probabilities at both sampled timesteps: ratio `1.0000 ± 0.0000` and maximum `|Δlogp| = 0`.
- The existing per-timestep forward, backward, and gradient accumulation order is unchanged.
