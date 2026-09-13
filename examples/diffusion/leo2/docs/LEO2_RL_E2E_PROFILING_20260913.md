# Leo 2.0 UniRL RL End-to-End Profiling and Memory Analysis

Date: 2026-09-13

## Scope

This report profiles the trainside FlowGRPO path from
`personal/bowen/leo2.0/dev` on Taiji instance
`8b1d818da08afca801a097b6c46b163f` (8 nodes × 8 NVIDIA H20, 64 GPUs).
The measurements below use the same model, CP2/EP1/HSDP8 topology, 464×848×121
video geometry, cached conditioning, and VideoPickScore reward as the production
recipe. To make repeated A/B tests finish in a bounded window, the measured
smoke workload uses 4 rollout steps, SDE steps `[0, 1]`, and 64 videos per
iteration. Historical full-workload measurements are included separately and
must not be numerically compared with the reduced smoke workload.

The retained allocation image contains Python 3.12, Torch 2.13.0+cu130, Ray
2.58.0, and transformers 5.12.1 rather than the repository's pinned Leo2
Torch-2.10 runtime. The branch therefore includes three compatibility fixes
needed by this instance: no hard import of the removed private
`_NoopSaveInputs`, a non-differentiable `torch.topk` fallback when
`colt5_attention` is absent, and an environment gate disabling the incompatible
fused unpermute kernel. Performance numbers are valid for this retained image.

Raw logs, 2-second GPU telemetry, and per-rank JSONL phase records are under:

```text
/apdcephfs_gz44/share_305110755/aimicahchen/leo2-rl-e2e-profile-20260913-v2
```

The compact machine-readable result is
[`results/leo2_rl_e2e_profile_20260913.json`](results/leo2_rl_e2e_profile_20260913.json).

## Instrumentation

The branch adds opt-in `UNIRL_PHASE_PROFILE_DIR` JSONL records and matching
`torch.profiler.record_function` regions for:

- condition loading, initial-noise construction, denoising, video decode, and
  audio decode;
- every rollout model forward and SDE transition;
- trajectory and decoded-output tensor byte accounting;
- train input alignment, anchor preparation, update, replay forward, backward,
  gradient clipping, optimizer, scheduler, and EMA;
- local model, trainable parameter, gradient, and optimizer-state bytes.

`examples/diffusion/leo2/scripts/summarize_e2e_profile.py` converts those
records and node GPU telemetry into one machine-readable summary.

## 64-GPU Baseline

Successful run: `baseline_smoke_v11`.

| Phase | Wall time |
|---|---:|
| Rollout generation | 159.4 s |
| Reward | 6.2 s |
| Score finalization | 0.14 s |
| Training | 233.0 s |
| Approximate measured iteration | 398.7 s |

The rank-local critical path inside the distributed calls was:

| Worker phase | Mean | Max |
|---|---:|---:|
| Rollout denoise | 115.4 s | 117.7 s |
| Video VAE decode | 38.9 s | 39.4 s |
| Audio VAE decode | 0.085 s | 0.098 s |
| Replay forward, per SDE step | 30.6 s | 32.0 s |
| Backward, per SDE step | 79.4 s | 82.0 s |
| Optimizer update total | 230.9 s | 231.0 s |
| Gradient clipping | 4.49 s | 5.03 s |
| Optimizer step | 0.39 s | 1.17 s |

The first completed update had reward `0.6948`, finite loss/gradients, and
maximum rollout/replay log-probability drift `6.39e-6`, below the configured
`1e-2` gate.

The node monitor observed a maximum of `97,283 MiB/GPU`. This is effectively
the 95 GiB H20 capacity ceiling. During samples above 8 GiB, mean utilization
was `84.9%`; long initialization periods lower all-run utilization to `39.2%`.

## Memory Ledger

Per training rank after model/FSDP/optimizer construction:

| Component | Local bytes |
|---|---:|
| Model parameters | 18,885,895,972 B = 17.59 GiB |
| Trainable LoRA parameters | 79,798,272 B = 76.1 MiB |
| Optimizer state before first step | 0 B, lazily initialized |

Per rollout pack of two videos:

| Component | Bytes |
|---|---:|
| Frozen conditions | 4,703,688 B = 4.49 MiB |
| Sparse video latents | 73,185,792 B = 69.8 MiB |
| Initial latents duplicate | 18,296,448 B = 17.4 MiB |
| Rollout transition means | 36,592,896 B = 34.9 MiB |
| Audio latent trajectory | 390,144 B = 0.37 MiB |
| Total trajectory payload | 128,465,364 B = 122.5 MiB |
| Decoded video/audio payload | 1,146,544,176 B = 1.07 GiB |

At production group size 16, the measured payloads scale to approximately:

- trajectory: `0.96 GiB` per DP group and `30.6 GiB` cluster-wide;
- decoded media: `8.54 GiB` per DP group and `273.4 GiB` cluster-wide.

The decoded FP32 121-frame videos are therefore the dominant persistent
cross-stage payload. Historical 512-video execution reported roughly 498 GiB of
Ray object-store spill, consistent with this payload plus serialization copies.

## A/B Results

### Forward batch size

`fbs1_v1` completed successfully with the same total 64 videos.

| Case | Rollout | Train | Peak GPU memory | Result |
|---|---:|---:|---:|---|
| FBS2/MBS2 baseline | 159.4 s | 233.0 s | 97,283 MiB | pass |
| FBS1/MBS1 | 163.6 s | 225.4 s | 70,449 MiB | pass |

FBS1 reduced peak memory by `26,834 MiB` (`27.6%`) while increasing rollout
time by `2.7%`. Training time was `3.3%` lower in this single-run comparison,
which is within the noise expected from one cold-started update. The reliable
conclusion is that FBS1 is the available low-memory fallback; FBS2 remains the
throughput baseline.

FBS4 was started with twice the total work because the group must preserve the
recorded forward pack. It did not finish within the bounded test window and
showed a 92,681 MiB peak during rollout. No throughput claim is made; a same-workload
FBS4 comparison requires a longer reservation.

### Video VAE residency

`vae_resident` completed successfully.

| Case | Rollout | Video decode/rank | Train | Peak |
|---|---:|---:|---:|---:|
| VAE transient baseline | 159.4 s | 38.86 s | 233.0 s | 97,283 MiB |
| VAE resident | 154.7 s | 36.14 s | 243.0 s | 97,283 MiB |

VAE residency reduced rollout by `3.0%` and video decode by `7.0%`, but did not
reduce the observed peak and the one-run end-to-end total was slightly worse
because training varied upward. Treat it as a rollout latency optimization, not
as a demonstrated end-to-end win.

### Trajectory field trimming

The new `store_sde_means=false` and `store_initial_latents=false` switches
remove data not consumed by ordinary `FlowGRPO(beta=0, use_grpo_guard=false)`.
At FBS1, the per-pack trajectory payload fell from:

```text
64,232,716 B → 36,788,044 B
```

This is a `42.7%` reduction, or `27.44 MiB` per one-video pack. The A/B update
completed with reward `0.6948`, finite gradients, and max log-probability drift
`8.31e-6`. Runtime was unchanged within noise (`223.4 s → 223.2 s` training).
The nvidia-smi peak remained 70,449 MiB because activation/FSDP peaks dominate
this reduced workload.

The FBS2 trimming run reached training but OOMed at the existing near-capacity
peak. The removal is valid and useful for transport/object-store pressure, but
does not solve the training activation peak.

A full 30-step FBS1+trim run was also exercised for about 17 minutes. It
recorded 3,272 of the expected 3,840 per-rank rollout-forward events before
manual termination, with a 50,073 MiB node-level peak and 95.5% mean active GPU
utilization. This confirms full-schedule rollout stability and headroom, but it
is not an end-to-end completion result.

### FSDP forward prefetch

`forward_prefetch=true` reached training and failed with CUDA/NCCL OOM at the
95 GiB boundary. It provided no proven speedup before failure. Reject for the
current CP2 topology.

### Activation checkpointing

Both tested forms failed:

- CP2 + AC off: OOM while allocating 748 MiB with 94.28 GiB already used.
- CP4 + AC off: OOM while allocating 1.46 GiB with 93.65 GiB already used.

Historical CP16 data showed training-time savings from disabling checkpointing,
but the current 64-GPU CP2/CP4 configuration has insufficient headroom.

## Historical Production-Workload Evidence

The earlier 64-GPU topology report measured the full 30-step, 512-video
workload. Its rollout completed in `6819.6 s`; the run then spilled about
`498 GiB` from the Ray object store before training completion. Separate
8-GPU-per-case topology sweeps found:

| CP / EP / AC | Iteration | Peak |
|---|---:|---:|
| CP2 / EP1 / on | 1365.6 s | 62.48 GiB |
| CP4 / EP1 / on | 1441.4 s | 50.62 GiB |
| CP8 / EP1 / on | 1602.7 s | 51.32 GiB |
| CP16 / EP16 / off | 1848.9 s | 82.84 GiB |

The production-scale rollout microbenchmark on this same retained instance
completed 480×848×121×30 denoising in `418.4 s` for UniRL, with all training
state tensors bitwise aligned to native and about `20.7 GiB` allocator peak.
That isolates model rollout from decode/reward/train orchestration.

## Optimization Checklist

### Recommended

- [x] Keep cached conditions.
- [x] Keep `old_logp_source=rollout`.
- [x] Keep CP2/EP1/HSDP8 and root wrapping as the throughput baseline.
- [x] Add low-overhead phase and tensor-byte instrumentation.
- [x] Add optional omission of `sde_means` and duplicate `initial_latents`.
- [ ] Implement reward-only sparse-frame transport. Expected payload reduction:
  about 30× before optional uint8 conversion.
- [ ] Decode only on the CP collect head; non-head ranks currently perform
  redundant VAE decode work before their outputs are discarded.
- [ ] Hoist transient VAE onload/offload around the whole rollout rather than
  every packed call, or keep it resident when workload-specific headroom permits.
- [ ] Batch all four VideoPickScore frames and reuse each prompt's text
  embedding.
- [ ] Add a readonly preprocessing-cache LRU keyed by prompt and geometry.
- [ ] Profile and remove the EP1 per-layer host synchronization around expert
  token counts if the CUDA trace confirms it as a material bubble.
- [ ] Add async-trainer queue/version instrumentation before claiming overlap
  gains from separate rollout/training slabs.

### Rejected for the current topology

- [x] FSDP forward prefetch: OOM.
- [x] Activation checkpointing off at CP2: OOM.
- [x] Activation checkpointing off at CP4: OOM.
- [x] EP4/EP8 as a general optimization: historical runs were slower and
  reached the memory limit.
- [x] FBS4 as a default: high memory and no completed same-workload result.
- [x] Inference-cache methods as transparent FlowGRPO acceleration: they change
  the rollout policy while replay remains exact.
- [x] BF16 trajectory as a memory-only change: it violates current native
  numerical-parity requirements.

### Optional, workload-dependent

- [ ] VAE resident: keep only when rollout latency matters more than memory
  headroom; measured rollout improvement was 3.0%.
- [ ] FBS1/MBS1: use as the low-memory fallback; measured peak reduction was
  27.6% for a 2.7% rollout slowdown.
- [ ] Async DCP checkpointing: useful only on checkpoint iterations.
- [ ] Separate/async topology: retest after payload streaming and async
  launched/used/discarded accounting are implemented.

## Recommended Production Configuration

Keep the current CP2/EP1/HSDP8/AC-on topology and add:

```yaml
bundle:
  config:
    store_sde_means: false
    store_initial_latents: false
```

Use this only for ordinary FlowGRPO with:

```yaml
algorithm:
  beta: 0.0
  use_grpo_guard: false
  old_logp_source: rollout
```

For maximum headroom, set rollout and training batch sizes to one. For the
default throughput setting, keep batch size two and prioritize sparse decoded
frame transport before attempting larger batches or disabling checkpointing.

## Validation

- Final focused test set: `73 passed`.
- Ruff checks passed for all modified first-party files.
- `git diff --check` passed.
- 64-GPU successful cases:
  - baseline FBS2;
  - VAE resident FBS2;
  - FBS1;
  - FBS1 plus trajectory trimming.
- 64-GPU capacity failures recorded:
  - CP2 forward prefetch;
  - CP2 AC off;
  - CP4 AC off.

## Resource Restoration

After every run, Ray and training Python processes were stopped and the
original `gpu_occupy_force.py` process was relaunched on all eight nodes. Final
verification must show 64/64 H20s at approximately 1199 MiB and 100%
utilization before handoff.
