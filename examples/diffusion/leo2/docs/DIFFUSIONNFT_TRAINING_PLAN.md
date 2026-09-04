# Leo2 FlowGRPO and DiffusionNFT training plan

Scope: UniRL only. Flow-Factory is an algorithm/configuration reference and is
not a dependency or modification target.

## Current support

- FlowGRPO: supported on one 8-GPU node through
  `leo2_t2v_trainside.yaml`; rollout, reward, replay, backward and LoRA
  checkpointing have completed long runs.
- DiffusionNFT: the generic algorithm and FSDP EMA-LoRA backend exist, but Leo2
  previously lacked the single-timestep prediction interface and a recipe.
- Training parallelism: Leo2 training currently initializes hymm with CP=1 and
  EP=1. Native-inference CP/EP results do not establish training support.

## Implementation and validation gates

1. Add Leo2 `predict_noise_at_step` and a DiffusionNFT EMA-LoRA recipe.
2. Unit-test the forward-process contract and resolve the Hydra config without
   loading weights.
3. Run a one-rollout reduced-geometry GPU smoke with W&B disabled. Require:
   terminal latent generation, reward, new/EMA-old forwards, finite backward,
   optimizer step and EMA update.
4. Align UniRL `DP_SCATTER` model-parallel grouping with hymm CP. CP ranks must
   receive identical conditions before any CP collective. Validate CP=2 with
   the same reduced smoke, then with 848x464x121.
5. Test the memory matrix at fixed LoRA r64/alpha128 and lr=3e-4. Reject a
   topology on OOM, non-finite loss, missing gradients, collective mismatch or
   EMA divergence.
6. Treat EP as a separate gate: enable hymm fused EP model construction,
   EP-aware DCP loading, and DeepEP. Before training, require a real cross-node
   DeepEP round trip for the intended EP group. Compare EP=8 and EP=16 where
   world size and 64 experts are divisible.
7. Start the long run only after one complete full-resolution rollout/update
   and a multi-rollout stability smoke pass. Enable W&B project `leo2-rl` and
   media upload only for this accepted configuration.

## Fixed long-run contract

- Rollout: 10 steps, guidance 1.0, no inference cache, CP=2,
  848x464 output, 121 frames.
- Training: generation-branch-only EMA LoRA, rank 64, alpha 128, AdamW
  learning rate 3e-4.
- Logging: online W&B, project `leo2-rl`, generated video media enabled.
- Assets and code are local on every node; checkpoints/logs use an explicit run
  root. Keepalive is handed off before jobs and restored on every exit path.
