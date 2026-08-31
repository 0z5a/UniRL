# Leo2 (HunyuanVideo 2.0, MoE-A12B) × UniRL trainside FlowGRPO

Text-to-video GRPO on the 75B three-stream MMDiT (`hunyuan_multimoda_gen_ar`,
branch `dev_video`) with UniRL's in-process (trainside) rollout engine: one
FSDP2-sharded copy of the model per DP rank does rollout, log-prob replay and
the LoRA update — no separate inference engine, no weight sync.

## Layout

| Path | What |
|---|---|
| `unirl/models/leo2/` | model package: `bundle.py` (hymm bootstrap, DCP load, FSDP-ready placement), `text_embed.py` (capture-based conditioning through hymm's own `generate_video` input pipeline), `diffusion.py` (`Leo2DiffusionStage`: predict_noise / generate / replay), `vae.py` (3D-VAE decode → `Videos`), `pipeline.py`, `conditions.py`, `config.py` |
| `examples/diffusion/leo2/leo2_t2v_trainside.yaml` | base recipe (FSDP2 + LoRA r64, FlowSDE, PickScore, FlowGRPO) |
| `examples/diffusion/leo2/scripts/` | cluster launchers (`unirl_longrun*.sh`, `unirl_smoke.sh`), one-rollout probes, ops helpers, `perf_table.py` |
| `examples/diffusion/leo2/docs/DESIGN.md` | decision log R1–R27 (every pitfall and its fix) |
| `examples/diffusion/leo2/docs/R15_perf_report.html` | timing / memory / deployment measurements (with erratum) |
| `examples/diffusion/leo2/data/` | the 32 training prompts + 8 held-out prompts |

## Recipe that learns (v3, mirrors the native pure-torch GRPO run)

```
FlowSDEStrategy · 30 inference steps · eta 0.5 · SDE noise + training only on
transitions [0..4] (highest sigma; the rest are ODE)
samples_per_prompt 8 · shared x_T per prompt group · batch 8 prompts (64/step)
guidance 1.0 · video_shift 3.0 · 192x336 · 49 frames · seed 42
LoRA r64 a256 on attention (video+text streams), shared MLP, text MLP · AdamW 2.5e-5
FlowGRPO clip 1e-4 · old_logp replay · 2 updates/batch · per-group advantage std
PickScore mean over 4 uniformly spaced frames
```

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
  2.5e-5 stays clean. `algorithm.beta>0` (KL against the adapter-disabled base
  model) is the intended fix (R26–R27, `scripts/unirl_longrun_v4.sh`).
