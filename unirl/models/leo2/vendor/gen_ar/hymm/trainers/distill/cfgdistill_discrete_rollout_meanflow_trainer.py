# Discrete teacher-rollout CFG distillation for the MeanFlow student.
#
# This file is additive only. It does not modify existing trainers, models, configs,
# schedulers, or loss implementations.
#
# Algorithm:
#   1. Sample shared video/audio base indices t_idx=x and r_idx=y on a 960 grid.
#   2. Require y-x >= MIN_INDEX_GAP and y-x divisible by TEACHER_STEPS.
#   3. Roll the ordinary diffusion teacher for exactly TEACHER_STEPS Euler steps:
#        index(k)=x+k*s, k=0..TEACHER_STEPS, s=(y-x)/TEACHER_STEPS.
#   4. Video/audio use the same indices but their own shifted time tables.
#   5. Regress student(x_t, t, r, cond) to the rollout's average velocity.
#
# Teacher guidance defaults to 1.0 (one conditional forward; useful when the teacher
# is already CFG-distilled). For standard CFG scale w != 1.0, an unconditional teacher
# forward is added at every rollout step and pred=uncond+w*(cond-uncond) is used.
#
# Usage:
#   ... run_pure_torch_leo_cfgd_meanflow.sh <config> \
#     --trainer distill.cfgdistill_discrete_rollout_meanflow_trainer.MultimodalTrainer
#
# Optional environment overrides:
#   CFGD_ROLLOUT_TOTAL_STEPS=960
#   CFGD_ROLLOUT_MIN_INDEX_GAP=16
#   CFGD_ROLLOUT_TEACHER_STEPS=4
#   CFGD_ROLLOUT_VIDEO_SHIFT=9
#   CFGD_ROLLOUT_AUDIO_SHIFT=1
#   CFGD_ROLLOUT_GUIDANCE_SCALE=1
#   CFGD_ROLLOUT_CP_CHECK_STEPS=3

import os

import torch

from .cfgdistill_discrete_meanflow_trainer import MultimodalTrainer as DiscreteCFGDTrainer
from ...diffusion.flow.imf_loss import perturb_latents, time_binary, time_unary


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else float(default)


class MultimodalTrainer(DiscreteCFGDTrainer):
    """MeanFlow student distilled from a configurable discrete diffusion-teacher rollout."""

    TOTAL_STEPS = _env_int("CFGD_ROLLOUT_TOTAL_STEPS", 960)
    MIN_INDEX_GAP = _env_int("CFGD_ROLLOUT_MIN_INDEX_GAP", 16)
    TEACHER_STEPS = _env_int("CFGD_ROLLOUT_TEACHER_STEPS", 4)
    VIDEO_SHIFT = _env_float("CFGD_ROLLOUT_VIDEO_SHIFT", 9.0)
    AUDIO_SHIFT = _env_float("CFGD_ROLLOUT_AUDIO_SHIFT", 1.0)
    TEACHER_GUIDANCE_SCALE = _env_float("CFGD_ROLLOUT_GUIDANCE_SCALE", 1.0)
    CP_CHECK_STEPS = _env_int("CFGD_ROLLOUT_CP_CHECK_STEPS", 3)

    def after_initialize(self):
        super().after_initialize()
        if self.TEACHER_STEPS <= 0:
            raise ValueError("CFGD_ROLLOUT_TEACHER_STEPS must be positive.")
        if self.MIN_INDEX_GAP < self.TEACHER_STEPS:
            raise ValueError("CFGD_ROLLOUT_MIN_INDEX_GAP must be >= CFGD_ROLLOUT_TEACHER_STEPS.")
        if self.MIN_INDEX_GAP > self.TOTAL_STEPS:
            raise ValueError("CFGD_ROLLOUT_MIN_INDEX_GAP must not exceed the discrete grid size.")
        effective_min_gap = (
            (self.MIN_INDEX_GAP + self.TEACHER_STEPS - 1) // self.TEACHER_STEPS
        ) * self.TEACHER_STEPS
        if getattr(self, "rank", 0) == 0:
            self.logger.info(
                "[CFGD-ROLLOUT] enabled: "
                f"grid={self.TOTAL_STEPS}, requested_min_gap={self.MIN_INDEX_GAP}, "
                f"effective_min_gap={effective_min_gap}, teacher_steps={self.TEACHER_STEPS}, "
                f"video_shift={self.VIDEO_SHIFT}, audio_shift={self.AUDIO_SHIFT}, "
                f"teacher_guidance={self.TEACHER_GUIDANCE_SCALE}."
            )

    # The parent calls this method while injecting the student's discrete (t, r).
    # Save the sampled root pair so prepare_cfgdistill_target can construct the exact
    # same teacher rollout path.
    def _sample_index_pair_like(self, ref):
        is_root = not getattr(self, "_rollout_sampling_indices", False)
        if is_root:
            self._rollout_sampling_indices = True
        try:
            pair = self._sample_rollout_index_pair_like(ref)
        finally:
            if is_root:
                self._rollout_sampling_indices = False
        if is_root:
            self._rollout_index_pair = pair
        return pair

    def _sample_rollout_index_pair_like(self, ref):
        if self._is_seq(ref):
            pairs = [self._sample_rollout_index_pair_like(x) for x in ref]
            return [p[0] for p in pairs], [p[1] for p in pairs]

        # A configurable N-step rollout with all endpoints on the integer grid requires
        # r_idx-t_idx to be divisible by TEACHER_STEPS. Sample the per-step integer
        # stride uniformly from all valid strides, with total gap >= MIN_INDEX_GAP.
        min_stride = (self.MIN_INDEX_GAP + self.TEACHER_STEPS - 1) // self.TEACHER_STEPS
        min_gap = min_stride * self.TEACHER_STEPS
        max_t_index = self.TOTAL_STEPS - min_gap
        if max_t_index < 0:
            raise ValueError("No valid t/r pair exists for the configured rollout span.")

        t_idx = torch.randint(
            max_t_index + 1,
            ref.shape,
            device=ref.device,
            generator=self.vae.generator,
            dtype=torch.long,
        )
        max_stride = (self.TOTAL_STEPS - t_idx) // self.TEACHER_STEPS
        num_choices = max_stride - min_stride + 1
        u = torch.rand(ref.shape, device=ref.device, generator=self.vae.generator)
        stride = min_stride + (u * num_choices.float()).floor().long()
        r_idx = t_idx + self.TEACHER_STEPS * stride
        return t_idx, r_idx

    def _check_ordered_index_pair(self, t_indices, r_indices):
        super()._check_ordered_index_pair(t_indices, r_indices)
        device = torch.device("cuda", self.args.local_rank)
        t_flat = self._index_vector(device, t_indices)
        r_flat = self._index_vector(device, r_indices)
        gaps = r_flat - t_flat
        if gaps.numel() == 0:
            return
        if not torch.all(gaps >= self.MIN_INDEX_GAP):
            raise RuntimeError(
                f"[CFGD-ROLLOUT] expected r_idx-t_idx >= {self.MIN_INDEX_GAP}."
            )
        if not torch.all(torch.remainder(gaps, self.TEACHER_STEPS) == 0):
            raise RuntimeError(
                "[CFGD-ROLLOUT] t/r index gap must be divisible by the number of teacher steps."
            )

    def _rollout_indices(self, t_indices, r_indices):
        return [
            time_binary(
                t_indices,
                r_indices,
                lambda t, r, k=k: t + ((r - t) // self.TEACHER_STEPS) * k,
            )
            for k in range(self.TEACHER_STEPS + 1)
        ]

    def _tree_scale(self, value, scale):
        if isinstance(value, (list, tuple)):
            if isinstance(scale, (list, tuple)):
                return [self._tree_scale(v, s) for v, s in zip(value, scale)]
            if torch.is_tensor(scale) and scale.ndim > 0 and scale.shape[0] == len(value):
                return [self._tree_scale(v, scale[i]) for i, v in enumerate(value)]
            return [self._tree_scale(v, scale) for v in value]
        scale_b = self._broadcast_t(scale, value)
        return value.float() * scale_b

    def _tree_add(self, left, right):
        if left is None:
            return right
        if isinstance(left, (list, tuple)):
            return [self._tree_add(a, b) for a, b in zip(left, right)]
        return left + right

    def _tree_average_velocity(self, displacement, total_dt):
        if isinstance(displacement, (list, tuple)):
            if isinstance(total_dt, (list, tuple)):
                return [self._tree_average_velocity(x, dt) for x, dt in zip(displacement, total_dt)]
            if torch.is_tensor(total_dt) and total_dt.ndim > 0 and total_dt.shape[0] == len(displacement):
                return [self._tree_average_velocity(x, total_dt[i]) for i, x in enumerate(displacement)]
            return [self._tree_average_velocity(x, total_dt) for x in displacement]
        dt_b = self._broadcast_t(total_dt, displacement)
        if torch.any(dt_b.abs() < 1e-8):
            raise RuntimeError("[CFGD-ROLLOUT] zero teacher-rollout time span.")
        return (displacement / dt_b).to(dtype=torch.bfloat16).detach().clone()

    def _teacher_prediction(self, cond_kwargs, uncond_model_input_kwargs, need_uncond):
        out_cond = self.teacher_engine(**cond_kwargs)
        v_cond = self._get_pred(out_cond, "diffusion_prediction")
        a_cond = self._get_pred(out_cond, "audio_diffusion_prediction")

        if not need_uncond:
            return v_cond, a_cond

        uncond_kwargs = dict(cond_kwargs)
        uncond_kwargs.update(uncond_model_input_kwargs)
        out_uncond = self.teacher_engine(**uncond_kwargs)
        v_uncond = self._get_pred(out_uncond, "diffusion_prediction")
        a_uncond = self._get_pred(out_uncond, "audio_diffusion_prediction")
        guidance = torch.tensor(
            self.TEACHER_GUIDANCE_SCALE,
            device=cond_kwargs["timesteps"][0].device
            if isinstance(cond_kwargs["timesteps"], list)
            else cond_kwargs["timesteps"].device,
        )
        v_pred = self._calc_cfg_target(v_cond, v_uncond, guidance)
        a_pred = None
        if a_cond is not None or a_uncond is not None:
            if a_cond is None or a_uncond is None:
                raise RuntimeError("[CFGD-ROLLOUT] conditional/unconditional audio predictions do not align.")
            a_pred = self._calc_cfg_target(a_cond, a_uncond, guidance)
        return v_pred, a_pred

    @torch.no_grad()
    def prepare_cfgdistill_target(self, model_input_kwargs, uncond_model_input_kwargs, device):
        pair = getattr(self, "_rollout_index_pair", None)
        if pair is None:
            raise RuntimeError("[CFGD-ROLLOUT] missing t/r indices from prepare_model_inputs.")
        t_indices, r_indices = pair
        index_path = self._rollout_indices(t_indices, r_indices)
        tables = self._build_timestep_tables(device)

        video_loss_fn = model_input_kwargs.get("diffusion_loss_fn")
        audio_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn")
        v_t_ref = video_loss_fn.keywords["t"]
        has_audio = (
            model_input_kwargs.get("audio_latents") is not None
            and audio_loss_fn is not None
            and audio_loss_fn.keywords.get("t") is not None
            and audio_loss_fn.keywords.get("ut") is not None
        )
        a_t_ref = audio_loss_fn.keywords["t"] if has_audio else None

        current_v_latents = model_input_kwargs["latents"]
        current_a_latents = model_input_kwargs.get("audio_latents") if has_audio else None
        video_displacement = None
        audio_displacement = None
        need_uncond = self.TEACHER_GUIDANCE_SCALE != 1.0

        teacher_base = dict(model_input_kwargs)
        teacher_base.pop("timestep_r", None)
        teacher_base.pop("audio_timestep_r", None)
        for key in ("guidance_index", "guidance"):
            teacher_base.pop(key, None)

        for step in range(self.TEACHER_STEPS):
            idx_now = index_path[step]
            idx_next = index_path[step + 1]

            v_now = self._gather_like(idx_now, tables["video"], v_t_ref)
            v_next = self._gather_like(idx_next, tables["video"], v_t_ref)
            v_dt = time_binary(v_next, v_now, lambda nxt, cur: nxt.float() - cur.float())

            cond_kwargs = dict(teacher_base)
            cond_kwargs["latents"] = current_v_latents
            cond_kwargs["timesteps"] = time_unary(v_now, self.video_denoiser.get_model_t)

            if has_audio:
                a_now = self._gather_like(idx_now, tables["audio"], a_t_ref)
                a_next = self._gather_like(idx_next, tables["audio"], a_t_ref)
                a_dt = time_binary(a_next, a_now, lambda nxt, cur: nxt.float() - cur.float())
                cond_kwargs["audio_latents"] = current_a_latents
                cond_kwargs["audio_timesteps"] = time_unary(a_now, self.audio_denoiser.get_model_t)

            v_pred, a_pred = self._teacher_prediction(
                cond_kwargs, uncond_model_input_kwargs, need_uncond
            )
            if v_pred is None:
                raise RuntimeError("[CFGD-ROLLOUT] teacher video prediction is unavailable.")

            video_displacement = self._tree_add(
                video_displacement, self._tree_scale(v_pred, v_dt)
            )
            current_v_latents = perturb_latents(current_v_latents, v_pred, v_dt)

            if has_audio:
                if a_pred is None:
                    raise RuntimeError("[CFGD-ROLLOUT] teacher audio prediction is unavailable.")
                audio_displacement = self._tree_add(
                    audio_displacement, self._tree_scale(a_pred, a_dt)
                )
                current_a_latents = perturb_latents(current_a_latents, a_pred, a_dt)

        v_start = self._gather_like(index_path[0], tables["video"], v_t_ref)
        v_end = self._gather_like(index_path[-1], tables["video"], v_t_ref)
        v_total_dt = time_binary(v_end, v_start, lambda end, start: end.float() - start.float())
        video_target = self._tree_average_velocity(video_displacement, v_total_dt)

        audio_target = None
        if has_audio:
            a_start = self._gather_like(index_path[0], tables["audio"], a_t_ref)
            a_end = self._gather_like(index_path[-1], tables["audio"], a_t_ref)
            a_total_dt = time_binary(a_end, a_start, lambda end, start: end.float() - start.float())
            audio_target = self._tree_average_velocity(audio_displacement, a_total_dt)

        if getattr(self, "_cfgd_rollout_dbg_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgd_rollout_dbg_left = getattr(self, "_cfgd_rollout_dbg_left", 3) - 1
            t_flat = self._index_vector(device, t_indices)
            r_flat = self._index_vector(device, r_indices)
            gap = r_flat - t_flat
            flat_paths = [self._index_vector(device, idx).tolist() for idx in index_path]
            sample_paths = list(zip(*flat_paths))[:8]
            self.logger.info(
                f"[CFGD-ROLLOUT] {self.TEACHER_STEPS}-step teacher rollout: "
                f"paths={sample_paths}, pairs={list(zip(t_flat.tolist(), r_flat.tolist()))[:8]}, "
                f"gap_min={gap.min().item()}, gap_max={gap.max().item()}, "
                f"guidance={self.TEACHER_GUIDANCE_SCALE}, uncond_forward={need_uncond}."
            )

        guidance_out = torch.tensor(self.TEACHER_GUIDANCE_SCALE, device=device)
        return video_target, audio_target, guidance_out
