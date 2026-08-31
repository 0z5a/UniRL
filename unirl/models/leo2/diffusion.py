"""Leo2 diffusion stage -- t2v only, unpacked latents, single-branch (cfg=1).

The per-step forward mirrors ``pipeline_leo.__call__``'s loop body
(lines ~824-860) verbatim: optional zero channel-cond concat, sigma*1000
timesteps, ``prepare_inputs_for_generation`` + model call, fp32 cast of the
velocity. Sign convention: Leo2's FlowMatchDiscreteScheduler Euler step is
``x_next = x + v * (sigma_next - sigma)`` which is exactly FlowSDEStrategy's
eta=0 mean -- the prediction feeds ``strategy.denoise`` UN-negated (do NOT
copy H3's minus sign).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .config import LEO2_TIMESTEP_SCALE
from .conditions import Leo2Conditions


class Leo2DiffusionStage(DiffusionStage[Leo2Conditions]):
    """Rollout-level Leo2 stage: conditions -> ``LatentSegment``."""

    _no_split_modules: ClassVar[List[str]] = ["LeoLayer", "LeoDualLayer", "LeoTripleLayer"]

    def __init__(
        self,
        bundle,
        strategy: StepStrategy,
        *,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.bundle = bundle
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        return self.bundle.trainable_module()

    def _autocast(self):
        if self.autocast_dtype == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    # ------------------------------------------------------------------ step
    def _prep_channel_cond(self, blob, latents: torch.Tensor):
        """Zero cond latents + mask once per trajectory (t2v: no cond image).

        Mirrors pipeline_leo: with ``extend_latent_channels`` the patch_embed
        consumes 48+48+1 channels even for plain t2v, the extra ones all-zero.
        """
        pipeline = self.bundle.model.diffusion_pipeline
        args = self.bundle.hymm_args
        if not getattr(args, "extend_latent_channels", False):
            return None, None
        cond_latents, cond_mask, _task_type = pipeline.prepare_channel_cond_latents(None, latents)
        return cond_latents, cond_mask

    def predict_noise(
        self,
        blob: dict,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        channel_cond: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """One forward -> unpacked velocity ``[B, C, T, H, W]``."""
        from .bundle import ensure_hy_parallel_state

        # Cheap dict check after the first call; see ensure_hy_parallel_state.
        require(ensure_hy_parallel_state(),
                "Leo2DiffusionStage: torch.distributed not initialised before the first forward")
        model = self.bundle.model
        # LeoModel.forward is a predictor only when `not self.training`; in train
        # mode it runs the SFT loss path (`diffusion_loss_fn(...)` -> None) and
        # also takes different text de-padding branches. TrainStack flips the
        # module to train mode before each update, so pin eval here: gradients
        # still flow, and rollout/replay traverse the identical graph (parity).
        if model.training:
            model.eval()
        device = sample.device
        # Mirror the reference RL (leo2_grpo_puretorch_dit): move tensor kwargs
        # to the compute device, forward None entries AS-IS (never drop keys --
        # prepare_inputs_for_generation dereferences them with kwargs[...]).
        if blob.get("_device") != str(device):
            blob["input_ids"] = blob["input_ids"].to(device)
            blob["model_kwargs"] = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in blob["model_kwargs"].items()
            }
            blob["_device"] = str(device)
        cond_latents, cond_mask = channel_cond
        if cond_latents is not None:
            latent_model_input = torch.cat([sample, cond_latents, cond_mask], dim=1)
        else:
            latent_model_input = sample
        t_expand = (sigma.to(sample.device) * LEO2_TIMESTEP_SCALE).reshape(1).repeat(latent_model_input.shape[0])

        model_inputs = model.prepare_inputs_for_generation(
            blob["input_ids"],
            latents=latent_model_input,
            timesteps=t_expand,
            audio_latents=None,
            audio_timesteps=None,
            **blob["model_kwargs"],
        )
        _dbg = getattr(self, "_mem_calls", 0)
        self._mem_calls = _dbg + 1
        _prof = torch.cuda.is_available()
        if _prof:
            torch.cuda.synchronize()
            # Peak since the previous forward's reset covers whatever ran in
            # between (backward + optimizer step in the update phase).
            _gap_peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            _before = torch.cuda.memory_allocated() / 2**30
            _t0 = time.perf_counter()
        model_output = model(**model_inputs)
        if _prof:
            torch.cuda.synchronize()
            print(f"[leo2 perf] fwd#{_dbg} grad={torch.is_grad_enabled()} tokens={blob['input_ids'].shape[-1]} "
                  f"dt={time.perf_counter() - _t0:.2f}s before={_before:.1f}GB "
                  f"after={torch.cuda.memory_allocated()/2**30:.1f}GB "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB "
                  f"gap_peak={_gap_peak:.1f}GB", flush=True)
        pred = model_output.get("diffusion_prediction", None)
        require(pred is not None, "Leo2DiffusionStage: model returned no diffusion_prediction")
        pred = pred.to(dtype=torch.float32)
        if pred.ndim == 5 and pred.size(2) == 1 and sample.ndim == 4:
            pred = pred.squeeze(2)
        return pred

    # -------------------------------------------------------------- rollout
    def generate(
        self,
        conditions: Leo2Conditions,
        *,
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        initial_latents: torch.Tensor,
        sde_indices: Optional[List[int]] = None,
        denoise_seed_keys: Optional[List[str]] = None,
        denoise_base_seed: int = 0,
    ) -> LatentSegment:
        require(
            conditions.hymm is not None and len(conditions.hymm) == 1,
            f"Leo2DiffusionStage: batch-1 only (packed hymm metadata carries no batch axis); "
            f"got {0 if conditions.hymm is None else len(conditions.hymm)} blobs. "
            f"Set rollout.forward_batch_size=1 and stack.micro_batch_size=1.",
        )
        blob = conditions.hymm[0]

        num_steps = int(sigmas.shape[0]) - 1
        require(
            num_steps == int(params.num_inference_steps),
            f"Leo2DiffusionStage: schedule has {num_steps} transitions, params want {params.num_inference_steps}",
        )

        device = self.bundle.device
        if not getattr(self, "_mem_reported", False) and torch.cuda.is_available():
            self._mem_reported = True
            m = self.bundle.model
            from torch.distributed.tensor import DTensor
            n_dt = sum(p.numel() for p in m.parameters() if isinstance(p, DTensor))
            n_plain_gpu = sum(p.numel() for p in m.parameters() if not isinstance(p, DTensor) and p.is_cuda)
            n_plain_cpu = sum(p.numel() for p in m.parameters() if not isinstance(p, DTensor) and not p.is_cuda)
            local_bytes = sum(
                (p.to_local().numel() if isinstance(p, DTensor) else p.numel()) * p.element_size()
                for p in m.parameters() if isinstance(p, DTensor) or p.is_cuda
            )
            for key in ("text_encoder", "vae"):
                aux = m.model_dict.get(key) if hasattr(m, "model_dict") else None
                if aux is not None and hasattr(aux, "parameters"):
                    ps = list(aux.parameters())
                    gb = sum(q.numel() * q.element_size() for q in ps if q.is_cuda) / 2**30
                    devs = sorted({str(q.device) for q in ps})[:3]
                    dts = sorted({str(q.dtype) for q in ps})
                    print(f"[leo2 mem] aux {key}: devices={devs} dtypes={dts} gpu_bytes={gb:.1f}GB", flush=True)
            print(f"[leo2 mem] pre-rollout alloc={torch.cuda.memory_allocated()/2**30:.1f}GB "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GB | params: dtensor={n_dt/1e9:.2f}B "
                  f"plain_gpu={n_plain_gpu/1e9:.2f}B plain_cpu={n_plain_cpu/1e9:.2f}B | local GPU param bytes={local_bytes/2**30:.1f}GB",
                  flush=True)
        x = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        channel_cond = self._prep_channel_cond(blob, x)

        sde_sorted = sorted(sde_indices) if sde_indices is not None else list(range(num_steps))
        sde_set = set(sde_sorted)
        # The terminal latent is always needed (pipeline decodes latents_at(num_steps));
        # compute_trajectory_positions only covers max(sde)+1, which falls short
        # when the last transitions are ODE-only (e.g. sde_indices [2,4,6] of 10).
        needed = set(compute_trajectory_positions(sde_sorted, num_steps)) | {num_steps}

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        sde_logp_list: List[torch.Tensor] = []
        if 0 in needed:
            stored_pairs.append((0, x.detach().clone()))

        with self._autocast():
            for step_idx in range(num_steps):
                step_eta = float(params.eta) if step_idx in sde_set else 0.0
                step_generators = (
                    make_denoise_step_generators(
                        base_seed=int(denoise_base_seed),
                        step_index=step_idx,
                        sample_ids=[str(key) for key in denoise_seed_keys],
                    )
                    if step_eta > 0.0 and denoise_seed_keys is not None
                    else None
                )
                pred = self.predict_noise(blob, sample=x, sigma=sigmas[step_idx], channel_cond=channel_cond)
                x_next, log_prob, _ = self.strategy.denoise(
                    noise_pred=pred,
                    sample=x.to(torch.float32),
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    eta=step_eta,
                    generator=step_generators,
                    step_index=step_idx,
                )
                x = x_next.to(dtype=self.trajectory_dtype)
                if (step_idx + 1) in needed:
                    stored_pairs.append((step_idx + 1, x.detach().clone()))
                if log_prob is not None:
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions = [p for p, _ in stored_pairs]
        return make_video_segment(
            latents=torch.stack([t for _, t in stored_pairs], dim=1),
            indices=torch.tensor(positions, dtype=torch.long, device=device),
            sigmas=sigmas.detach().clone(),
            sde_logp=torch.stack(sde_logp_list, dim=1) if sde_logp_list else None,
            sde_indices=(torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None),
            initial_latents=initial_latents.detach().clone(),
        )

    # ---------------------------------------------------------------- train
    def replay(
        self,
        conditions: Leo2Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        require(
            segment.sde_indices is not None and segment.latents is not None and segment.sigmas is not None,
            "Leo2DiffusionStage.replay: segment.sde_indices / latents / sigmas missing",
        )
        require(
            conditions.hymm is not None and len(conditions.hymm) == 1,
            "Leo2DiffusionStage.replay: batch-1 only",
        )
        blob = conditions.hymm[0]

        sigmas = segment.sigmas.to(self.bundle.device)
        stored = [int(i) for i in segment.sde_indices.tolist()]
        targets = [int(i) for i in (step_indices if step_indices is not None else stored)]

        channel_cond = None
        log_probs: List[torch.Tensor] = []
        means: List[torch.Tensor] = []
        with self._autocast():
            for step_idx in targets:
                x = segment.latents_at(step_idx).to(self.bundle.device)
                prev_x = segment.latents_at(step_idx + 1).to(self.bundle.device)
                if channel_cond is None:
                    channel_cond = self._prep_channel_cond(blob, x)
                pred = self.predict_noise(blob, sample=x, sigma=sigmas[step_idx], channel_cond=channel_cond)
                _, log_prob, mean = self.strategy.denoise(
                    noise_pred=pred,
                    sample=x.to(torch.float32),
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    eta=float(params.eta),
                    prev_sample=prev_x.to(torch.float32),
                    step_index=step_idx,
                )
                log_probs.append(log_prob.to(dtype=self.logprob_dtype))
                means.append(mean)

        return ReplayResult(
            log_probs=torch.stack(log_probs, dim=1),
            prev_sample_means=torch.stack(means, dim=1) if means else None,
        )


__all__ = ["Leo2DiffusionStage"]
