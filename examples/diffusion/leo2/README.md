# Leo2 (HunyuanVideo 2.0, MoE-A12B) × UniRL trainside FlowGRPO

Offline prompt/target preprocessing, cached SFT, and frozen-component offload are documented in
[the model package README](../../../unirl/models/leo2/README.md).
The [64-GPU preprocessing and topology report](docs/results/preprocessing_topology_20260906/BENCHMARK_REPORT.md)
records the current cluster validation and FlowGRPO launch settings.

Text-to-video GRPO on the 75B three-stream MMDiT (`hunyuan_multimoda_gen_ar`,
branch `dev_video`) with UniRL's in-process (trainside) rollout engine: one
FSDP2-sharded copy of the model per DP rank does rollout, log-prob replay and
the LoRA update — no separate inference engine, no weight sync.

The cached 121-frame recipe is [leo2_t2v_flowgrpo_cached.yaml](leo2_t2v_flowgrpo_cached.yaml).
[leo2_t2v_flowgrpo_separate.yaml](leo2_t2v_flowgrpo_separate.yaml) instead assigns independent GPU slabs to native
Leo2 rollout and FSDP training, with separate CP/EP settings and verified LoRA synchronization.
See the topology report for its current GPU validation status and measured communication costs.
With `python -m unirl.train_async_diffusion`, the separate trainer launches the
next rollout before scoring the completed one, so rollout and reward overlap.
Its `eval_reward_async: true` setting similarly pipelines each eval chunk's
reward RPCs with generation of the following chunk; failures remain ordered and
are propagated when the pending call is resolved.

## Layout

| Path | What |
|---|---|
| `unirl/models/leo2/` | model package and vendored gen-ar runtime: `bundle.py` (hymm bootstrap, DCP load, FSDP-ready placement), `sde.py`, `optimizer.py`, `vendor/`, `text_embed.py`, `diffusion.py`, `vae.py`, `pipeline.py`, `conditions.py`, `config.py` |
| `examples/diffusion/leo2/leo2_t2v_trainside.yaml` | base recipe (FSDP2 + LoRA r64, FlowSDE, PickScore, FlowGRPO) |
| `examples/diffusion/leo2/leo2_t2v_flowgrpo_profile.yaml` | bounded 64-GPU end-to-end profiling workload |
| `examples/diffusion/leo2/leo2_t2v_native_parity.yaml` | deterministic 32×H20 CP4 profile matching the native MoE GRPO rollout geometry and schedule |
| `examples/diffusion/leo2/scripts/` | portable launchers (`unirl_longrun*.sh`, `unirl_smoke.sh`), probes, artifact validation and performance helpers |
| `examples/diffusion/leo2/docs/DESIGN.md` | decision log R1–R27 (every pitfall and its fix) |
| `examples/diffusion/leo2/docs/R15_perf_report.html` | timing / memory / deployment measurements (with erratum) |
| `examples/diffusion/leo2/docs/LEO2_RL_E2E_PROFILING_20260913.md` | 64-H20 end-to-end timing, memory ledger, and optimization A/B report |
| `examples/diffusion/leo2/docs/CACHE_BENCHMARK_REPORT_20260903.md` | 8×H20、848×464×121 first-block cache 正式实验报告 |
| `examples/diffusion/leo2/docs/QUALITY_EVALUATION_PLAN.md` | VBench 系列调研、已安装环境、评估协议与复现命令 |
| `examples/diffusion/leo2/docs/results/acceleration_benchmark_20260903/README.md` | shift-9 Taylor、MagCache、FasterCache DFR 的速度、漂移和质量实测汇总 |
| `examples/diffusion/leo2/data/` | the 32 training prompts + 8 held-out prompts |

## Portable setup

All Leo2 source dependencies are vendored under `unirl/models/leo2/vendor`;
there are no Git submodules or external source checkouts at runtime. The default
model YAML is the vendored 480p stage-3 configuration and the T2V generation
JSON is a package resource. The recipe uses the committed prompt files.

Weights and model assets remain external because they are about 178 GB. Set
`LEO2_CKPT_DIR` and `LEO2_ASSETS_BASE` explicitly, then validate them before
launching:

```bash
python -m unirl.models.leo2.verify_artifacts --require-checksums
bash examples/diffusion/leo2/scripts/unirl_smoke.sh
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the Torch 2.10 ABI contract, compiled
FA2/FA3/DeepEP requirements and the 480p×121 launch override. The versioned
artifact layout and checksum interface is in the packaged
[artifacts.yaml](../../../unirl/models/leo2/resources/artifacts.yaml).

For a source-isolation check that directly produces one 480p, 121-frame video
with the retained native sampler, run:

```bash
LEO2_RUNTIME_PYTHON=/path/to/python \
  bash examples/diffusion/leo2/scripts/native_t2v.sh
```

The default is 50 denoising steps. Set `LEO2_INFER_STEPS=1` for a quick load,
forward and decode smoke; output goes to `outputs/leo2/native_t2v` unless
`LEO2_OUTPUT_DIR` is set.

## Native inference caches

Leo2's vendored hymm pipeline supports the same user-facing lifecycle as
Diffusers `FirstBlockCacheConfig`: disabled by default, enabled on the model,
scoped to one denoising request, and reset even when generation raises.

```python
from unirl.models.leo2 import Leo2Bundle, Leo2PipelineConfig

config = Leo2PipelineConfig(
    inference_cache_method="first_block",
    inference_cache_threshold=0.05,
)
bundle = Leo2Bundle.from_config(config)
output = bundle.model.generate_video(prompt="A red panda walking in the snow")
print(bundle.model.cache_stats())
bundle.model.disable_cache()
```

For an already loaded model, the direct Diffusers-style API is also available:

```python
from diffusers import FirstBlockCacheConfig

bundle.model.enable_cache(FirstBlockCacheConfig(threshold=0.05))
```

The cache always runs Leo block 0 and compares its residual with the last full
step. When the normalized change is below the threshold, it reuses the residual
of blocks 1..N. Higher thresholds usually skip more work but can reduce quality.

For pure-video, guidance-1 inference, `fastercache_dfr` instead caches each
selected layer's raw self-attention outputs after FA3/CP and `o_proj`, before
the layer's gate and residual. It keeps two exact outputs and linearly
extrapolates non-anchor steps inside a half-open denoising-step window:

```python
config = Leo2PipelineConfig(
    inference_cache_method="fastercache_dfr",
    inference_cache_fastercache_start_step=4,
    inference_cache_fastercache_end_step=46,
    inference_cache_fastercache_interval=2,
    inference_cache_fastercache_layers=None,  # all Leo layers
)
```

The fixed `linear_window` schedule uses
`w=(step-start_step)/(end_step-start_step)` in
`latest + (latest - previous) * w`. Window-external and interval-anchor steps
run exact attention, skipped steps do not advance exact history, and all
selected layers share one decision per denoising step. `cache_stats()` reports
`attention_compute_calls`, `attention_reuse_calls`, `selected_layers`, and the
peak request-local `cache_bytes`. Audio is explicitly unsupported. First-block,
Taylor, FasterCache DFR, and any additional installed controller are mutually
exclusive through the model's single `enable_cache()` slot.

Only hymm's native `generate_image` / `generate_video` path opens a cache
context; UniRL rollout and gradient replay remain exact. CP statistics are
combined before the decision, and the maximum score is synchronized over the
block's actual FSDP shard group for the supported EP=ETP=TP=PP=1 topology.
DFR synchronizes its step, history validity, and final exact/reuse decision over
the same CP and actual FSDP groups.
Other parallel topologies, uninitialized distributed parallel state, or an
unresolvable shard group conservatively fall back to a full forward. The
decision remains eager code; whole-model `torch.compile(fullgraph=True)` is not
supported.

## Recipe that learns (v3, mirrors the native pure-torch GRPO run)

```
Leo2FlowSDEStrategy · 30 inference steps · eta 0.5 · SDE noise + training only on
transitions [0..4] (highest sigma; the rest are ODE)
samples_per_prompt 8 · shared x_T per prompt group · batch 8 prompts (64/step)
guidance 1.0 · video_shift 3.0 · 192x336 · 49 frames · seed 42
LoRA r64 a256 on attention (video+text streams), shared MLP, text MLP · AdamW 2.5e-5
FlowGRPO clip 1e-4 · old_logp replay · 2 updates/batch · per-group advantage std
PickScore mean over 4 uniformly spaced frames
```

For strict implementation comparison, use `leo2_t2v_native_parity.yaml`. It
pins the native rollout settings (`480×848×121`, shift `7/1`, 30 steps,
`eta=0.5`, SDE transitions `0..4`, eight samples per prompt, shared initial
noise, CP4) and selects deterministic FA2 (`flash_packed` + `reproduce=true`).
It also disables LoRA, retains the checkpoint's mixed BF16/FP32 layout, and
selects the native full-model Muon/AdamW parameter split. FA3 remains the
production-speed comparison mode because its backward kernel is not bitwise
reproducible even across two native reruns.

`scripts/unirl_longrun_v3.sh` is the exact launcher. Measured on 8×H20: ~630 s/step
(rollout 400 s + train 225 s); with `old_logp_source=rollout` and a GPU-resident
text encoder (`scripts/unirl_longrun_v3b.sh`) ~495 s/step. Per-GPU peak ~52 GB.

## Things that bit us (see DESIGN.md for the full story)

* FSDP2 root units never reshard after forward → `root_wrap: true` (R8–R10, 92.9 GB OOM).
* hy_parallelism's global parallel state must be initialised in the process
  that runs the forward (`ensure_hy_parallel_state()`), otherwise every
  `get_parallel_state()` call builds a new device mesh → thousands of leaked
  NCCL process groups (R11–R13).
* `LeoModel.forward` in train mode runs the SFT loss path; `predict_noise`
  pins eval mode so rollout and replay traverse the same graph (R15).
* FlowSDE with 10 steps, eta 0.7 and noise on the last transition produces
  structureless videos while PickScore still reports ~0.73 — always eyeball
  rollout media next to the reward curve (R17, R26).
* LoRA lr 1e-4 without KL reward-hacks into stripe textures by rollout ~50;
  2.5e-5 stays clean. `algorithm.beta>0` regularizes against the adapter-disabled
  base model. `algorithm.reference_loss_type` selects the existing
  `transition_kl` or direct joint audio-video `velocity_mse`; retune `beta` when
  switching because their scales differ (R26–R27, `scripts/unirl_longrun_v4.sh`).
