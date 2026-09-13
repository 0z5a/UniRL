# Leo2 UniRL/native bitwise alignment report

Date: 2026-09-12

## Scope

Compared UniRL's Leo2 trainside RL implementation against
`/data/home/aimicahchen/work/hunyuan_multimoda_gen_ar-dev_video` at commit
`e8937335fe8b4711232c0cda818a5af07c4afe8f`.

Covered:

- layer forward tensors;
- layer backward tensors, latent gradients, parameter gradients and optimizer update;
- 10-step and 30-step FlowGRPO transitions (prediction, mean, log-prob, next latent);
- batched replay with different timesteps;
- a minimal complete rollout -> replay -> GRPO-Guard loss -> backward -> AdamW update;
- real-checkpoint forward/backward, selected parameter gradients and native
  Muon/AdamW update;
- FA3 and deterministic FA2 timing on the full 48-layer A12B architecture.

## Fixed differences

- trajectories changed from BF16 to FP32;
- first `sigma == 1` transition uses the full schedule's `sigmas[1]` limit;
- Leo2-native transition/log-prob floating-point operation order;
- per-row replay for batched distinct timesteps;
- B=1 timestep modulation retains native `repeat_interleave` accumulation;
- native per-row initial video RNG and cloned audio-generator state;
- native process CUDA RNG stream for stochastic denoise steps;
- audio rollout remains stochastic independently of whether audio log-prob is in the policy;
- joint video/audio policy log-prob uses the native sum of modality means;
- native GRPO-Guard ratio normalization, loss coefficient and advantage clipping;
- native AV joint-policy log-prob combines video/audio modality means by direct sum;
- FA3 Python API compatibility and deterministic attention/reproducibility switches.

## Bitwise results

Evidence is under
`/apdcephfs_gz44/share_305110755/aimicahchen/leo2-bitwise-align-20260911-v2/results/`.

| Check | Evidence | Result |
|---|---|---|
| Tiny layer forward/backward/parameter-grad/AdamW update | `tiny_compare_actual.json` | 172/172 exact |
| 10-step layer trajectory | `trajectory_compare_v3.json` | 61/61 exact |
| 30-step standalone SDE | `sde_rollout_exact.json` | 90/90 exact |
| Batched distinct-timestep replay | `sde_batch_parity_v4.json` | 3/3 exact |
| Full 48-layer A12B architecture, FA2 deterministic, CP4/32 GPUs | `full_random_fa2det_compare_v1.json` | 434/434 exact |
| Native/native FA2 reproducibility | `full_random_native_fa2det_repro_v1_v2.json` | 434/434 exact |
| Minimal RL rollout/replay/GRPO-Guard/backward/update | `tiny_rl_compare_v4.json` | 202/202 common tensors exact; no missing tensors |
| Real DCP, full 48-layer A12B, FA2 deterministic, CP4/4 GPUs | `realckpt-fa2det-compare-v2.json` | 678/678 dumped tensors exact; no missing tensors |
| Native Muon/AdamW backend composition, local, 2 updates | `tiny_native_muon_backend_parity.json` | Parameters and optimizer state exact across all 3 child optimizers |
| Native Muon/AdamW DCP optimizer roundtrip | `tiny_native_muon_dcp_roundtrip.json` | 52/52 state entries exact |
| Native Muon/AdamW backend composition, FSDP2/4 GPUs, 2 updates | `tiny_native_muon_fsdp_parity_fp32master_v1.json` | Exact on all 4 ranks |
| Native Muon/AdamW backend composition, FSDP2/8 GPUs, 2 updates | `tiny_native_muon_fsdp_parity_current_8gpu.json` | Exact on all 8 ranks |

The minimal RL check includes 21 rollout tensors, 35 replay/layer forward/backward
tensors, 69 parameter gradients and 74 post-AdamW parameter tensors. The second
run uses UniRL's actual `FlowGRPO(use_grpo_guard=true, adv_clip_max=5)` path; all
202 compared tensors are exact.

The real-DCP deterministic run compares 144 inference layer outputs, 144
train-side layer outputs, 143 layer-output gradients, the latent gradient,
80 selected parameter gradients, 80 pre-step parameter shards, 84 post-step
parameter shards, and both final predictions. The selected set exercises both
native Muon matrix updates and native AdamW special-parameter updates. All 678
files are bitwise exact.

FA3 forward is exact, but 96 backward records are non-bitwise. The same 96
records differ in real-DCP native/native and UniRL/UniRL reruns
(`realckpt-fa3-nondeterminism-summary-v1.json`), as well as in the earlier random
weight reruns. This is FA3 backward nondeterminism rather than an implementation
split. Strict backward bitwise validation therefore uses `flash_packed` plus
`reproduce=true`.

## Performance

Full 48-layer architecture, random but identical weights, CP4/32 H20:

- FA3 steady forward:
  - native mean across two runs: 0.8687 s;
  - UniRL: 0.8739 s;
  - UniRL delta: +0.6%.
- deterministic FA2 steady forward:
  - native mean across two runs: 1.3499 s;
  - UniRL: 1.3708 s;
  - UniRL delta: +1.5%.

These are three-iteration short runs; the observed differences are within run
noise and do not establish an implementation-level regression.

Real DCP, CP4/4 H20, sequence length 512, two runs per
implementation/attention mode:

- deterministic FA2:
  - native mean of run means: 1.2892 s;
  - UniRL vendor-path mean of run means: 1.2732 s;
  - UniRL delta: -1.2%.
- FA3:
  - native mean of run means: 1.0293 s;
  - UniRL vendor-path mean of run means: 1.0500 s;
  - UniRL delta: +2.0%;
  - mean best-iteration delta: +0.3%.

The benchmark harness swaps only the native versus UniRL Leo model source inside
the same native trainer/runtime. It measures implementation-level model-forward
overhead without conflating the surrounding UniRL orchestration. The full
breakdown is in `realckpt-benchmark-summary-v4.json`; the observed differences
remain small enough that this short benchmark does not establish a regression.

### Frozen encoders and decoders

The production-geometry auxiliary benchmark used `480x848x121`, deterministic
inputs, two warmups and five timed iterations. Native and UniRL output hashes
were identical for all measured modules:

| Module | Native mean | UniRL mean | Delta | Incremental peak |
|---|---:|---:|---:|---:|
| Text tokenize | 1.164 ms | 0.967 ms | -17.0% | 0 GiB / 0 GiB |
| Text encode | 134.062 ms | 158.464 ms | +18.2% | 0.425 GiB / 0.425 GiB |
| Video VAE encode | 4.813 s | 4.796 s | -0.35% | 3.795 GiB / 3.795 GiB |
| Video VAE decode | 10.798 s | 10.831 s | +0.31% | 8.405 GiB / 8.404 GiB |
| Audio VAE encode | 61.488 ms | 61.483 ms | -0.01% | 0.871 GiB / 0.871 GiB |
| Audio VAE decode | 27.697 ms | 28.548 ms | +3.07% | 0.522 GiB / 0.522 GiB |

The text-encode mean includes one UniRL outlier; the best times differ by only
0.48%. Evidence: `aux-prod-native-v1/summary.json`,
`aux-prod-unirl-v1/summary.json`, and `aux-prod-compare-v1.json`.

### Condition preparation and rollout

The native RL configuration's `480x848x121` condition preparation produced
12/12 common tensors exactly, including masks, text states, media indices and
training audio noise. Cold single-pass time was 9.080 s native versus 8.934 s
UniRL (-1.61%). UniRL additionally exposes `root.text.embeds`, so this wrapper
field is not treated as a missing native tensor. Evidence:
`condition-rl-compare-v2.json`.

A real-checkpoint AV rollout smoke at `192x336x49`, CP4 and 16 H20s confirmed
exact video/audio latent trajectories, final latents, video log-probabilities,
video previous means and decoded audio. The only residual non-bitwise output is
decoded video postprocessing: maximum FP16 difference `0.0051` and mean
absolute difference `1.63e-4`. In the same short in-process run, native took
4.223 s and UniRL took 5.945 s; peak allocated memory was 47.96 GiB versus
56.21 GiB. This two-step timing is diagnostic, not a stable throughput result.
Evidence: `rollout-smoke-16gpu-prefetch-clean-v2/summary.json`.

Repeated real-checkpoint 30-step attempts on 8 and 16 H20s reached model step
11-13 and then failed in the native denoising path at about 90 GiB allocated per
GPU. The failure reproduced with `PYTORCH_ALLOC_CONF=expandable_segments:True`,
explicit FSDP prefetch disabled, activation checkpointing disabled, and forced
per-step reshard/pending-all-gather cleanup. This rules out allocator
fragmentation and the UniRL rollout implementation as the primary cause; it is
an upstream/runtime FSDP residency limitation at these topologies. The intended
32/64-GPU allocation did not start before cleanup because it remained queued.

### Training-side timing

For the real-DCP deterministic CP4/4-GPU comparison, the latest paired run
measured inference at 8303.98 ms native versus 8402.32 ms UniRL (+1.18%), and
training forward+backward at 8335.53 ms native versus 8331.07 ms UniRL
(-0.05%). The same run's 678 dumped tensors, including selected native
Muon/AdamW updates, are exact. Full optimizer-state parity is separately exact
for two updates locally and on FSDP2 with four and eight GPUs.

## Native-parity recipe

`examples/diffusion/leo2/leo2_t2v_native_parity.yaml` pins the native rollout
geometry and schedule: 32 H20, CP4, 480x848, 121 frames, video/audio shifts 7/1,
30 steps, eta 0.5, SDE steps 0..4, eight samples per prompt, shared initial
noise, guidance 1.0, deterministic FA2. It also selects full-model training,
native Muon/AdamW parameter routing, FP32 router submodules, and DCP optimizer
checkpoints.

## Real checkpoint

The real DCP checkpoint is
`/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights`.
Direct concurrent DCP reads over the GZ-to-ZW mount repeatedly stalled or failed
with `OSError: [Errno 107] Transport endpoint is not connected`. A resumable,
per-chunk checksummed staging pass copied all 150,508,043,106 bytes to local NVMe,
after which native and UniRL completed from the same local file. The resulting
deterministic comparison is `realckpt-fa2det-compare-v2.json`.

A persistent GZ copy is available at
`/apdcephfs_gz44/share_305110755/aimicahchen/leo2-bitwise-align-20260911-v2/checkpoint-staged-v2/weights`.
Its byte size matches the source, and 32 sampled 4 MiB ranges spanning the file
match the ZW source byte-for-byte (`persistent_checkpoint_sample_verify.json`).
An attempted full `480x848x121` forward on only four H20 GPUs exhausted memory
at about 94.9 GiB per GPU. The completed real-DCP model benchmark therefore uses
the same implementation and attention settings at sequence length 512.

## Scope boundary

The real-checkpoint comparison exercises the native mixed Muon/AdamW update on
selected parameters so model-side optimizer inputs and updates can be compared
without allocating full-model optimizer state. UniRL's native-parity recipe now
uses the same full-model optimizer partition. Its backend construction was
validated against the native optimizer container for two FSDP2 updates on four
and eight GPUs, including parameter and optimizer-state equality. DCP save/load
also preserves all 52 checked optimizer-state entries exactly. A complete
production-geometry rollout and full-model optimizer step still requires the
intended larger topology. Real 30-step rollout attempts at reduced geometry
also exposed the native-path FSDP residency issue described above.

## Verification

- focused CPU tests: 68 passed;
- the additional timestep-repeat test module could not import the vendored
  checkpoint helper under the submit host's older PyTorch because
  `torch.utils.checkpoint._NoopSaveInputs` is unavailable; these tests had
  already passed in the native PyTorch 2.10 image;
- focused native-image GPU tests: 70 passed;
- native Muon/AdamW FSDP2 parity: exact on all four- and eight-GPU ranks after
  two updates;
- native Muon/AdamW DCP optimizer roundtrip: 52/52 state entries exact;
- Ruff checks passed for all changed first-party Python files; the two modified
  vendored files retain exactly the same 24 pre-existing diagnostics by code;
- recipe-target and one-line-docstring guards passed;
- `git diff --check` passed.

The full repository test collection still requires optional environments not
installed on the submit host (`vllm_omni`, `vllm`, and `imageio_ffmpeg`).

## Resource restoration

All temporary 8/16/32-GPU Leo2 tasks were stopped. The original
`occupy-qwen3p8-64h20-gz-0909` definition was restored to eight nodes × eight
H20 GPUs, the Qwen runtime image, and `bash start.sh`, then submitted again. At
the final check it was `TRAINING_RESOURCE_WAITING`; no Leo2 temporary task was
active. The scheduler, rather than an active experiment, was the only remaining
reason the occupy allocation was not yet consuming GPUs.

## 64-H20 completion addendum — 2026-09-13

The requested persistent instance
`8b1d818da08afca801a097b6c46b163f` became available with eight nodes and eight
H20 GPUs per node. The 150.5 GB DCP was staged to every node's
`/tmp/leo2-real-ckpt/weights`. Runtime-only compatibility work was required
because the retained Qwen image uses Python 3.12 and Torch 2.13 rather than the
original Leo2 image: missing audio packages and IndexKits were supplied, the
checkpoint helper was adapted, non-differentiable routing falls back to
`torch.topk`, and the incompatible fused-unpermute Triton kernel was disabled
for both compared paths.

### 64-GPU layer/train-side parity

Deterministic FA2, CP4, real DCP, sequence length 512:

- native: `realckpt-native-fa2det-cp4-64gpu-v4`;
- UniRL vendor path: `realckpt-unirl-fa2det-cp4-64gpu-v1`;
- comparison: `realckpt-fa2det-cp4-64gpu-compare-v1.json`;
- result: **678/678 tensors bitwise exact**, with no missing tensors.

The dump covers inference and train-side layer outputs, backward outputs, input
gradient, selected parameter gradients, pre-update shards, post-update shards,
and predictions. Steady forward timing was 2.1227 s native versus 2.1586 s
UniRL (+1.69%); best iteration was 2.1112 s versus 2.1404 s (+1.38%). The
single instrumented forward was 13.703 s versus 13.675 s, while the
forward+backward capture was 17.936 s versus 17.080 s; these one-shot values
include dump-hook overhead and should not be used as throughput estimates.

### 64-GPU 30-step rollout

At `192x336x49`, the native and UniRL 30-step runs completed in 71.466 s and
72.336 s respectively, a UniRL delta of +1.22%. Peak allocated memory was
17.020/17.046 GiB. All compared trajectory, log-probability, previous-mean,
final-latent, and decoded-audio tensors were bitwise exact.

At the actual requested `480x848x121` geometry, both paths completed all 30
steps. Native took 407.510 s and UniRL took 418.404 s (+2.67%). Peak allocated
memory was 20.701/20.713 GiB. All compared RL-state tensors and decoded audio
were bitwise exact. Decoded FP16 video remained the sole exception:
max absolute difference `0.008544921875`, mean absolute difference
`1.2699065e-4`.

The production rollout GPU monitor collected 13,768 samples. During samples
with more than 8 GiB allocated, mean utilization was 87.13%, median 99%, and
maximum observed memory was 31,531 MiB/GPU. During the denoising loops all 64
GPUs were repeatedly observed at approximately 99–100% utilization.

### Frozen encoder/decoder rerun

The production-geometry auxiliary benchmark was rerun in the same retained
Torch 2.13 image (`aux-prod-native-v2`, `aux-prod-unirl-v2`,
`aux-prod-compare-v2.json`). Text tokenization/encoding, video VAE
encoding/decoding, and audio VAE encoding/decoding are all bitwise exact. Both
video decoders also reproduced their own output bitwise on an additional decode
of the same latent. Video VAE encode was 5.3856/5.3839 s and decode was
9.4857/9.4857 s (native/UniRL); audio encode median was 62.218/62.129 ms and
audio decode mean was 38.751/38.996 ms.

The isolated decoded-video discrepancy in the sequential rollout A/B is
therefore not a source or weight mismatch: the VAE source hashes are identical,
standalone outputs are exact, and the RL terminal latent is exact. It is an
execution-context-sensitive FP16 decoder artifact. The strict bitwise claim is
therefore made for training and rollout state, not for the final decoded FP16
video tensor.

### Final resource state

All experiment launchers exited. The original occupy process was restarted on
all eight nodes. Final verification showed `TRAINING_RUNNING`, 64/64 GPUs at
100% utilization, and approximately 1199 MiB allocated per GPU.
