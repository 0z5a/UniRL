"""Run Leo2 T2V diffusion through one packed path for every logical batch size."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import ClassVar, List, Optional, Sequence, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .conditions import Leo2Conditions
from .config import LEO2_TIMESTEP_SCALE
from .packing import (
    as_packed_media,
    pack_hymm_conditions,
    unpack_packed_prediction,
)

MediaBatch = torch.Tensor | Sequence[torch.Tensor]


def _media_batch_size(batch: MediaBatch, *, name: str) -> int:
    """Return the logical batch size of dense or ragged media."""
    if isinstance(batch, torch.Tensor):
        if batch.ndim < 3 or batch.shape[0] < 1:
            raise ValueError(f"Leo2 {name} requires a non-empty batched Tensor")
        return int(batch.shape[0])
    if not isinstance(batch, (list, tuple)) or not batch:
        raise ValueError(f"Leo2 {name} requires a non-empty sequence of per-sample Tensors")
    if any(not isinstance(row, torch.Tensor) or row.ndim < 3 or row.shape[0] != 1 for row in batch):
        raise ValueError(f"Leo2 ragged {name} requires per-sample Tensors with leading dimension one")
    return len(batch)


def _media_device(batch: MediaBatch, *, name: str) -> torch.device:
    """Return the shared device of dense or ragged media."""
    if isinstance(batch, torch.Tensor):
        return batch.device
    _media_batch_size(batch, name=name)
    device = batch[0].device
    if any(row.device != device for row in batch[1:]):
        raise ValueError(f"Leo2 ragged {name} samples must share one device")
    return device


def _media_cat_channels(
    sample: MediaBatch,
    cond_latents: MediaBatch | None,
    cond_mask: MediaBatch | None,
) -> MediaBatch:
    """Append channel conditioning without forcing ragged samples dense."""
    if cond_latents is None:
        if cond_mask is not None:
            raise ValueError("Leo2 channel_cond mask requires conditioning latents")
        return sample
    if cond_mask is None:
        raise ValueError("Leo2 channel_cond latents require a mask")
    if isinstance(sample, torch.Tensor):
        if not isinstance(cond_latents, torch.Tensor) or not isinstance(cond_mask, torch.Tensor):
            raise TypeError("Leo2 dense media requires dense channel conditioning")
        return torch.cat([sample, cond_latents, cond_mask], dim=1)
    if not isinstance(cond_latents, (list, tuple)) or not isinstance(cond_mask, (list, tuple)):
        raise TypeError("Leo2 ragged media requires ragged channel conditioning")
    if len(sample) != len(cond_latents) or len(sample) != len(cond_mask):
        raise ValueError("Leo2 ragged media and channel conditioning batch sizes differ")
    return [torch.cat([row, cond, mask], dim=1) for row, cond, mask in zip(sample, cond_latents, cond_mask)]


def _media_float(batch: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    """Cast packed predictions while retaining ragged structure."""
    if isinstance(batch, torch.Tensor):
        return batch.to(dtype=torch.float32)
    return [row.to(dtype=torch.float32) for row in batch]


def _combine_modality_logp(
    video_logp: torch.Tensor,
    audio_logp: torch.Tensor,
    *,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Combine per-modality means by generated scalar degrees of freedom."""
    for name, value in (("n_video", n_video), ("n_audio", n_audio)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Leo2 joint log-prob expected positive int {name}, got {type(value).__name__}: {value!r}")
    if not isinstance(video_logp, torch.Tensor) or not isinstance(audio_logp, torch.Tensor):
        raise TypeError(
            "Leo2 joint log-prob expected Tensor video_logp/audio_logp, "
            f"got {type(video_logp).__name__}/{type(audio_logp).__name__}"
        )
    if video_logp.shape != audio_logp.shape or video_logp.ndim != 1:
        raise ValueError(
            "Leo2 joint log-prob expected matching per-sample tensors shaped [B], "
            f"got video={tuple(video_logp.shape)}, audio={tuple(audio_logp.shape)}"
        )
    total = n_video + n_audio
    return (video_logp * n_video + audio_logp * n_audio) / total


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
        profile_forward: bool = False,
        enable_audio: bool = True,
        video_shift: float = 3.0,
        audio_shift: float = 3.0,
        audio_joint_sde: bool = False,
    ) -> None:
        if type(profile_forward) is not bool:
            raise TypeError(
                "Leo2DiffusionStage expected bool for profile_forward, "
                f"got {type(profile_forward).__name__}: {profile_forward!r}"
            )
        if type(enable_audio) is not bool or type(audio_joint_sde) is not bool:
            raise TypeError(
                "Leo2DiffusionStage enable_audio/audio_joint_sde must be bool, "
                f"got {enable_audio!r}/{audio_joint_sde!r}"
            )
        if (
            not isinstance(video_shift, (int, float))
            or float(video_shift) <= 0
            or not isinstance(audio_shift, (int, float))
            or float(audio_shift) <= 0
        ):
            raise ValueError(
                "Leo2DiffusionStage video_shift/audio_shift must be positive numeric, "
                f"got {video_shift!r}/{audio_shift!r}"
            )
        self.bundle = bundle
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.profile_forward = profile_forward
        self.enable_audio = bool(enable_audio)
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        self.audio_joint_sde = bool(audio_joint_sde)

    def audio_schedule(self, video_schedule: torch.Tensor) -> torch.Tensor:
        """Build Leo2's independent audio sigma grid."""
        return get_sigma_schedule(
            num_steps=int(video_schedule.shape[0]) - 1,
            shift=self.audio_shift,
            device=video_schedule.device,
        ).to(dtype=video_schedule.dtype)

    def audio_sigma_from_video(self, video_sigma: torch.Tensor) -> torch.Tensor:
        """Map one shifted video sigma to the corresponding audio sigma."""
        denominator = self.video_shift - (self.video_shift - 1.0) * video_sigma
        base_sigma = video_sigma / denominator
        return (self.audio_shift * base_sigma) / (1.0 + (self.audio_shift - 1.0) * base_sigma)

    def trainable_module(self) -> torch.nn.Module:
        return self.bundle.trainable_module()

    def _autocast(self):
        if self.autocast_dtype == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    # ------------------------------------------------------------------ step
    def _prep_channel_cond(self, blob, latents: MediaBatch):
        """Build zero conditioning latents and mask once per T2V trajectory."""
        args = self.bundle.hymm_args
        if not getattr(args, "extend_latent_channels", False):
            return None, None
        # Native prepare_channel_cond_latents(None, latents), specialized to the supported T2V path.
        if not isinstance(latents, torch.Tensor):
            return (
                [torch.zeros_like(row) for row in latents],
                [torch.zeros_like(row[:, :1]) for row in latents],
            )
        return torch.zeros_like(latents), torch.zeros_like(latents[:, :1])

    @staticmethod
    def _scaled_timesteps(
        sigma: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        """Scale timesteps while preserving a shared scalar for model-side broadcast."""
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(f"Leo2 {name} must be a Tensor, got {type(sigma).__name__}")
        values = sigma.to(device=device).reshape(-1)
        if values.numel() not in (1, batch_size):
            raise ValueError(f"Leo2 {name} expected one value or batch {batch_size}, got shape={tuple(sigma.shape)}")
        return values * LEO2_TIMESTEP_SCALE

    def _model_conditions(
        self,
        blobs: List[dict],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Move conditions once and cache the last batch's static packed tensors."""
        from .preprocessing_cache import map_tensors

        for blob in blobs:
            if blob.get("_device") != str(device):
                blob["input_ids"] = blob["input_ids"].to(device)
                blob["model_kwargs"] = map_tensors(blob["model_kwargs"], device)
                blob["_device"] = str(device)

        sequence_parallel_size = int(getattr(getattr(self.bundle, "config", None), "context_parallel_size", 1))
        cache_key = (
            str(device),
            sequence_parallel_size,
            tuple((id(blob), blob["input_ids"].data_ptr()) for blob in blobs),
        )
        if getattr(self, "_packed_condition_cache_key", None) == cache_key:
            packed = self._packed_condition_cache
        else:
            packed = pack_hymm_conditions(
                blobs,
                sequence_parallel_size=sequence_parallel_size,
            )
            self._packed_condition_cache_key = cache_key
            self._packed_condition_cache = packed
        return packed.input_ids, packed.model_kwargs, packed.packing_kwargs

    def predict_joint_noise(
        self,
        blobs: dict | List[dict],
        *,
        sample: MediaBatch,
        sigma: torch.Tensor,
        channel_cond: Tuple[Optional[MediaBatch], Optional[MediaBatch]],
        audio_sample: Optional[MediaBatch] = None,
        audio_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor | list[torch.Tensor], Optional[torch.Tensor | list[torch.Tensor]]]:
        """Run one dense or shape-ragged packed model forward."""
        from .bundle import ensure_hy_parallel_state

        # Cheap dict check after the first call; see ensure_hy_parallel_state.
        require(
            ensure_hy_parallel_state(), "Leo2DiffusionStage: torch.distributed not initialised before the first forward"
        )
        model = self.bundle.model
        # LeoModel.forward is a predictor only when `not self.training`; in train
        # mode it runs the SFT loss path (`diffusion_loss_fn(...)` -> None) and
        # also takes different text de-padding branches. TrainStack flips the
        # module to train mode before each update, so pin eval here: gradients
        # still flow, and rollout/replay traverse the identical graph (parity).
        if model.training:
            model.eval()
        batch_size = _media_batch_size(sample, name="video sample")
        device = _media_device(sample, name="video sample")
        if isinstance(blobs, dict):
            blobs = [blobs]
        elif isinstance(blobs, list):
            blobs = list(blobs)
        else:
            raise TypeError(f"Leo2 predict_joint_noise expected a blob or list of blobs, got {type(blobs).__name__}")
        if not blobs or len(blobs) != batch_size:
            raise ValueError(
                "Leo2 predict_joint_noise requires one condition blob per latent sample; "
                f"got blobs={len(blobs)}, sample_batch={batch_size}"
            )

        # Preserve None entries required by prepare_inputs_for_generation.
        input_ids, condition_kwargs, packing_kwargs = self._model_conditions(blobs, device=device)
        model_config = getattr(model, "_config", None)
        text_proj_type = getattr(model_config, "text_proj_type", None)
        if batch_size > 1 and text_proj_type not in (None, "linear"):
            raise ValueError(f"Leo2 packed B>1 currently requires text_proj_type='linear'; got {text_proj_type!r}")
        attn_impl = getattr(model_config, "attn_impl", None)
        if not torch.is_grad_enabled() and not model.training:
            attn_impl = getattr(model_config, "inference_attn_impl", None) or attn_impl
        if attn_impl is not None and attn_impl not in ("flash_packed", "flash3_packed"):
            raise ValueError(
                f"Leo2 logical packed forward requires flash_packed or flash3_packed attention, got {attn_impl!r}"
            )
        latent_model_input = _media_cat_channels(sample, *channel_cond)
        t_expand = self._scaled_timesteps(
            sigma,
            batch_size=batch_size,
            device=device,
            name="video sigma",
        )
        if (audio_sample is None) != (audio_sigma is None):
            raise ValueError(
                "Leo2 joint prediction requires audio_sample and audio_sigma together, "
                f"got audio_sample={audio_sample is not None}, audio_sigma={audio_sigma is not None}"
            )
        if audio_sample is not None:
            audio_batch_size = _media_batch_size(audio_sample, name="audio sample")
            if audio_batch_size != batch_size:
                raise ValueError(f"Leo2 audio batch {audio_batch_size} does not match video batch {batch_size}")
            if _media_device(audio_sample, name="audio sample") != device:
                raise ValueError("Leo2 video and audio samples must share one device")
        audio_t_expand = (
            None
            if audio_sigma is None
            else self._scaled_timesteps(
                audio_sigma,
                batch_size=batch_size,
                device=device,
                name="audio sigma",
            )
        )

        # Shared timesteps stay scalar; distinct per-sample timesteps retain logical batch cardinality.
        model_latents = as_packed_media(latent_model_input)
        model_timesteps = [t_expand]
        model_audio_latents = as_packed_media(audio_sample) if audio_sample is not None else None
        model_audio_timesteps = [audio_t_expand] if audio_t_expand is not None else None

        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            latents=model_latents,
            timesteps=model_timesteps,
            audio_latents=model_audio_latents,
            audio_timesteps=model_audio_timesteps,
            **condition_kwargs,
        )
        # Restore native packing metadata omitted by the generation adapter.
        model_inputs.update(packing_kwargs)
        _dbg = getattr(self, "_mem_calls", 0)
        self._mem_calls = _dbg + 1
        _prof = self.profile_forward and torch.cuda.is_available()
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
            print(
                f"[leo2 perf] fwd#{_dbg} grad={torch.is_grad_enabled()} bsz={batch_size} "
                f"packed=True tokens={input_ids.shape[-1]} "
                f"dt={time.perf_counter() - _t0:.2f}s before={_before:.1f}GB "
                f"after={torch.cuda.memory_allocated() / 2**30:.1f}GB "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                f"gap_peak={_gap_peak:.1f}GB",
                flush=True,
            )
        pred = model_output.get("diffusion_prediction", None)
        require(pred is not None, "Leo2DiffusionStage: model returned no diffusion_prediction")
        pred = _media_float(unpack_packed_prediction(pred, batch_size=batch_size, name="video"))
        if isinstance(pred, torch.Tensor):
            if isinstance(sample, torch.Tensor) and pred.ndim == 5 and pred.size(2) == 1 and sample.ndim == 4:
                pred = pred.squeeze(2)
        else:
            sample_rows = list(sample) if not isinstance(sample, torch.Tensor) else list(sample.split(1))
            pred = [
                row.squeeze(2) if row.ndim == 5 and row.size(2) == 1 and source.ndim == 4 else row
                for row, source in zip(pred, sample_rows)
            ]
        audio_pred = model_output.get("audio_diffusion_prediction", None)
        if audio_sample is not None:
            require(
                audio_pred is not None,
                "Leo2DiffusionStage: AV forward returned no audio_diffusion_prediction",
            )
            audio_pred = _media_float(unpack_packed_prediction(audio_pred, batch_size=batch_size, name="audio"))
        return pred, audio_pred

    def predict_noise(
        self,
        blobs: dict | List[dict],
        *,
        sample: MediaBatch,
        sigma: torch.Tensor,
        channel_cond: Tuple[Optional[MediaBatch], Optional[MediaBatch]],
    ) -> torch.Tensor | list[torch.Tensor]:
        """Compatibility wrapper for video-only single-step objectives."""
        pred, _ = self.predict_joint_noise(
            blobs,
            sample=sample,
            sigma=sigma,
            channel_cond=channel_cond,
        )
        return pred

    # -------------------------------------------------------------- rollout
    def generate(
        self,
        conditions: Leo2Conditions,
        *,
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        initial_latents: torch.Tensor,
        initial_audio_latents: Optional[torch.Tensor] = None,
        sde_indices: Optional[List[int]] = None,
        denoise_seed_keys: Optional[List[str]] = None,
        denoise_base_seed: int = 0,
    ) -> LatentSegment:
        require(
            conditions.hymm is not None and len(conditions.hymm) > 0,
            "Leo2DiffusionStage.generate requires at least one hymm condition blob",
        )
        blobs = conditions.hymm

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
                for p in m.parameters()
                if isinstance(p, DTensor) or p.is_cuda
            )
            for key in ("text_encoder", "vae", "audio_vae"):
                aux = m.model_dict.get(key) if hasattr(m, "model_dict") else None
                if aux is not None and hasattr(aux, "parameters"):
                    ps = list(aux.parameters())
                    gb = sum(q.numel() * q.element_size() for q in ps if q.is_cuda) / 2**30
                    devs = sorted({str(q.device) for q in ps})[:3]
                    dts = sorted({str(q.dtype) for q in ps})
                    print(f"[leo2 mem] aux {key}: devices={devs} dtypes={dts} gpu_bytes={gb:.1f}GB", flush=True)
            print(
                f"[leo2 mem] pre-rollout alloc={torch.cuda.memory_allocated() / 2**30:.1f}GB "
                f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GB | params: dtensor={n_dt / 1e9:.2f}B "
                f"plain_gpu={n_plain_gpu / 1e9:.2f}B plain_cpu={n_plain_cpu / 1e9:.2f}B | local GPU param bytes={local_bytes / 2**30:.1f}GB",
                flush=True,
            )
        x = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        require(
            x.ndim in (4, 5) and x.shape[0] == len(blobs),
            "Leo2DiffusionStage.generate requires one video latent per condition; "
            f"got latent shape={tuple(x.shape)}, conditions={len(blobs)}",
        )
        if denoise_seed_keys is not None:
            require(
                len(denoise_seed_keys) == len(blobs),
                "Leo2DiffusionStage.generate requires one denoise seed key per condition; "
                f"got keys={len(denoise_seed_keys)}, conditions={len(blobs)}",
            )
        audio_sigmas = self.audio_schedule(sigmas) if self.enable_audio else None
        if self.enable_audio:
            require(
                initial_audio_latents is not None,
                "Leo2DiffusionStage AV generation requires initial_audio_latents",
            )
            a = initial_audio_latents.to(device=device, dtype=self.trajectory_dtype)
            require(
                a.shape[0] == len(blobs),
                "Leo2DiffusionStage.generate requires one audio latent per condition; "
                f"got audio batch={a.shape[0]}, conditions={len(blobs)}",
            )
        else:
            a = None
        channel_cond = self._prep_channel_cond(blobs[0], x)

        sde_sorted = sorted(sde_indices) if sde_indices is not None else list(range(num_steps))
        sde_set = set(sde_sorted)
        # The terminal latent is always needed (pipeline decodes latents_at(num_steps));
        # compute_trajectory_positions only covers max(sde)+1, which falls short
        # when the last transitions are ODE-only (e.g. sde_indices [2,4,6] of 10).
        needed = set(compute_trajectory_positions(sde_sorted, num_steps)) | {num_steps}

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        stored_audio: List[torch.Tensor] = []
        sde_logp_list: List[torch.Tensor] = []
        sde_means_list: List[torch.Tensor] = []
        if 0 in needed:
            stored_pairs.append((0, x.detach().clone()))
            if a is not None:
                stored_audio.append(a.detach().clone())

        model = self.bundle.model
        cache_context_factory = getattr(model, "cache_context", None)
        cache_context = cache_context_factory("unirl_rollout") if callable(cache_context_factory) else nullcontext()
        with self._autocast(), cache_context:
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
                if self.enable_audio:
                    pred, audio_pred = self.predict_joint_noise(
                        blobs,
                        sample=x,
                        sigma=sigmas[step_idx],
                        channel_cond=channel_cond,
                        audio_sample=a,
                        audio_sigma=(audio_sigmas[step_idx] if audio_sigmas is not None else None),
                    )
                else:
                    pred = self.predict_noise(
                        blobs,
                        sample=x,
                        sigma=sigmas[step_idx],
                        channel_cond=channel_cond,
                    )
                    audio_pred = None
                x_next, log_prob, prev_mean = self.strategy.denoise(
                    noise_pred=pred,
                    sample=x,
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    eta=step_eta,
                    generator=step_generators,
                    step_index=step_idx,
                )
                x = x_next.to(dtype=self.trajectory_dtype)
                audio_log_prob = None
                if a is not None:
                    require(
                        audio_pred is not None and audio_sigmas is not None,
                        "Leo2DiffusionStage AV step requires audio prediction and schedule",
                    )
                    a_next, audio_log_prob, _ = self.strategy.denoise(
                        noise_pred=audio_pred,
                        sample=a,
                        sigma=audio_sigmas[step_idx],
                        sigma_next=audio_sigmas[step_idx + 1],
                        eta=step_eta if self.audio_joint_sde else 0.0,
                        # Reuse the request stream so CP replicas sample the same audio transition.
                        generator=step_generators,
                        step_index=step_idx,
                    )
                    a = a_next.to(dtype=self.trajectory_dtype)
                if (step_idx + 1) in needed:
                    stored_pairs.append((step_idx + 1, x.detach().clone()))
                    if a is not None:
                        stored_audio.append(a.detach().clone())
                if log_prob is not None:
                    require(prev_mean is not None, "Leo2 SDE rollout produced log_prob without prev_sample_mean")
                    if self.audio_joint_sde and audio_log_prob is not None and a is not None:
                        log_prob = _combine_modality_logp(
                            log_prob,
                            audio_log_prob,
                            n_video=x[0].numel(),
                            n_audio=a[0].numel(),
                        )
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))
                    sde_means_list.append(prev_mean.detach())

        cache_stats_factory = getattr(model, "cache_stats", None)
        if callable(cache_stats_factory):
            cache_stats = cache_stats_factory()
            if cache_stats.get("method", "none") != "none" and int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[leo2 cache] "
                    f"method={cache_stats.get('method')} "
                    f"threshold={cache_stats.get('threshold')} "
                    f"full_steps={cache_stats.get('full_steps')} "
                    f"skipped_steps={cache_stats.get('skipped_steps')} "
                    f"cache_bytes={cache_stats.get('cache_bytes', 0)}",
                    flush=True,
                )

        positions = [p for p, _ in stored_pairs]
        return make_video_segment(
            latents=torch.stack([t for _, t in stored_pairs], dim=1),
            indices=torch.tensor(positions, dtype=torch.long, device=device),
            sigmas=sigmas.detach().clone(),
            sde_logp=torch.stack(sde_logp_list, dim=1) if sde_logp_list else None,
            sde_means=torch.stack(sde_means_list, dim=1) if sde_means_list else None,
            sde_indices=(torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None),
            initial_latents=initial_latents.detach().clone(),
            aux_latents=(torch.stack(stored_audio, dim=1) if stored_audio else None),
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
            conditions.hymm is not None and len(conditions.hymm) > 0,
            "Leo2DiffusionStage.replay requires at least one hymm condition blob",
        )
        blobs = conditions.hymm
        require(
            segment.latents.shape[0] == len(blobs),
            "Leo2DiffusionStage.replay requires one condition per trajectory; "
            f"got trajectory batch={segment.latents.shape[0]}, conditions={len(blobs)}",
        )

        sigmas = segment.sigmas.to(self.bundle.device)
        audio_sigmas = self.audio_schedule(sigmas) if self.enable_audio else None
        if self.enable_audio:
            require(
                segment.aux_latents is not None,
                "Leo2DiffusionStage AV replay requires segment.aux_latents",
            )
        stored = [int(i) for i in segment.sde_indices.tolist()]
        targets = [int(i) for i in (step_indices if step_indices is not None else stored)]

        channel_cond = None
        log_probs: List[torch.Tensor] = []
        means: List[torch.Tensor] = []
        model_outputs: List[torch.Tensor] = []
        aux_model_outputs: List[torch.Tensor] = []
        with self._autocast():
            for step_idx in targets:
                x = segment.latents_at(step_idx).to(self.bundle.device)
                prev_x = segment.latents_at(step_idx + 1).to(self.bundle.device)
                a = segment.aux_latents_at(step_idx).to(self.bundle.device) if self.enable_audio else None
                if channel_cond is None:
                    channel_cond = self._prep_channel_cond(blobs[0], x)
                pred, audio_pred = self.predict_joint_noise(
                    blobs,
                    sample=x,
                    sigma=sigmas[step_idx],
                    channel_cond=channel_cond,
                    audio_sample=a,
                    audio_sigma=audio_sigmas[step_idx] if audio_sigmas is not None else None,
                )
                model_outputs.append(pred)
                if audio_pred is not None:
                    aux_model_outputs.append(audio_pred)
                _, log_prob, mean = self.strategy.denoise(
                    noise_pred=pred,
                    sample=x,
                    sigma=sigmas[step_idx],
                    sigma_next=sigmas[step_idx + 1],
                    eta=float(params.eta),
                    prev_sample=prev_x,
                    step_index=step_idx,
                )
                if self.audio_joint_sde:
                    require(
                        a is not None and audio_pred is not None and audio_sigmas is not None,
                        "Leo2DiffusionStage joint audio replay is missing audio state",
                    )
                    prev_a = segment.aux_latents_at(step_idx + 1).to(self.bundle.device)
                    _, audio_log_prob, _ = self.strategy.denoise(
                        noise_pred=audio_pred,
                        sample=a,
                        sigma=audio_sigmas[step_idx],
                        sigma_next=audio_sigmas[step_idx + 1],
                        eta=float(params.eta),
                        prev_sample=prev_a,
                        step_index=step_idx,
                    )
                    log_prob = _combine_modality_logp(
                        log_prob,
                        audio_log_prob,
                        n_video=x[0].numel(),
                        n_audio=a[0].numel(),
                    )
                log_probs.append(log_prob.to(dtype=self.logprob_dtype))
                means.append(mean)

        return ReplayResult(
            log_probs=torch.stack(log_probs, dim=1),
            prev_sample_means=torch.stack(means, dim=1) if means else None,
            model_outputs=torch.stack(model_outputs, dim=1) if model_outputs else None,
            aux_model_outputs=torch.stack(aux_model_outputs, dim=1) if aux_model_outputs else None,
        )

    def predict_noise_at_step(
        self,
        conditions: Leo2Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Run one Leo2 velocity prediction for forward-process objectives."""
        require(
            conditions.hymm is not None and len(conditions.hymm) > 0,
            "Leo2DiffusionStage.predict_noise_at_step requires at least one hymm condition blob",
        )
        blobs = conditions.hymm
        require(
            sample.ndim == 5 and sample.shape[0] == len(blobs),
            "Leo2DiffusionStage.predict_noise_at_step expected video latents [B,C,T,H,W] "
            f"with one condition per sample, got shape={tuple(sample.shape)}, conditions={len(blobs)}",
        )
        sigma_values = sigma.reshape(-1)
        require(
            sigma_values.numel() in (1, sample.shape[0]),
            "Leo2DiffusionStage.predict_noise_at_step expected one shared sigma or one per sample; "
            f"got shape={tuple(sigma.shape)}, batch={sample.shape[0]}",
        )
        require(
            float(params.guidance_scale) == 1.0,
            "Leo2DiffusionStage.predict_noise_at_step: Leo2 training supports "
            f"guidance_scale=1.0, got {params.guidance_scale}",
        )

        sample = sample.to(device=self.bundle.device, dtype=self.trajectory_dtype)
        channel_cond = self._prep_channel_cond(blobs[0], sample)
        with self._autocast():
            if self.enable_audio:
                audio_noise_rows = [blob.get("training_audio_noise") for blob in blobs]
                require(
                    all(isinstance(noise, torch.Tensor) for noise in audio_noise_rows),
                    "Leo2 AV single-step training requires training_audio_noise in every condition",
                )
                audio_shapes = {tuple(noise.shape[1:]) for noise in audio_noise_rows}
                require(
                    all(noise.ndim == 3 and noise.shape[0] == 1 for noise in audio_noise_rows)
                    and len(audio_shapes) == 1,
                    "Leo2 packed AV training requires same-shape audio noise rows [1,C,L]; "
                    f"got {[tuple(noise.shape) for noise in audio_noise_rows]}",
                )
                audio_noise = torch.cat(audio_noise_rows, dim=0)
                video_sigma = sigma_values.expand(sample.shape[0]) if sigma_values.numel() == 1 else sigma_values
                audio_sigma = self.audio_sigma_from_video(video_sigma)
                audio_sample = audio_noise.to(
                    device=self.bundle.device,
                    dtype=self.trajectory_dtype,
                ) * audio_sigma.to(self.bundle.device).reshape(-1, 1, 1)
                pred, _ = self.predict_joint_noise(
                    blobs,
                    sample=sample,
                    sigma=sigma_values,
                    channel_cond=channel_cond,
                    audio_sample=audio_sample,
                    audio_sigma=audio_sigma,
                )
                return pred
            return self.predict_noise(
                blobs,
                sample=sample,
                sigma=sigma_values,
                channel_cond=channel_cond,
            )


__all__ = ["Leo2DiffusionStage"]
