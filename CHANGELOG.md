# Changelog

## 2026-09-08

### Added

- Added a `velocity_mse` reference regularizer to FlowGRPO and FlowDPPO. Leo2 reuses the policy and LoRA-disabled reference transformer velocities from each replay and combines video and audio errors by latent element count.

### Fixed

- Made Leo2 joint audio-video SDE sampling use the request-scoped timestep generator for both modalities. This keeps audio trajectories identical across context-parallel replicas and prevents later transformer forwards from mixing divergent audio states.
- Preserved the stored trajectory dtype at Leo2 scheduler boundaries so rollout and replay score the same BF16 transition instead of comparing an FP32 rollout transition with its stored BF16 value.
- Limited the FlowGRPO rollout/replay parity threshold to the first optimizer update, while retaining drift metrics on later updates and failing closed on non-finite values.
- Aligned FlowDPPO anchor replay and training with FlowGRPO's per-timestep forward and backward order while retaining the KL-new-old advantage mask.
- Added FlowDPPO rollout/replay drift diagnostics and a non-finite-safe parity gate that is active only on the first optimizer update.

### Changed

- Set the Leo2 motion-bilingual FlowGRPO recipe's SDE noise coefficient `eta` to `0.9` for subsequent runs.

### Validation

- A checkpoint-40 replay on 64 GPUs with context parallelism 2 produced bit-identical video and audio predictions and log-probabilities at both sampled timesteps: ratio `1.0000 ± 0.0000` and maximum `|Δlogp| = 0`.
- The existing per-timestep forward, backward, and gradient accumulation order is unchanged.
- FlowDPPO's per-timestep implementation matched the former all-step objective's loss and gradients, including the reference-policy KL term; 74 focused algorithm, Leo2, and SDE tests passed.
- The new reference-loss mode keeps `transition_kl` as the default; selecting `velocity_mse` changes the timestep weighting and requires retuning `beta`.
- An analytical CPU harness matched the joint audio-video velocity loss and gradients for FlowGRPO and FlowDPPO and verified that Leo2 replay returns the transformer outputs consumed by both schedulers.
- An analytical CPU harness matched joint video-audio velocity-MSE values and gradients for FlowGRPO and FlowDPPO, and verified that Leo2 replay exposes the exact scheduler inputs; all 74 related regression tests passed.
