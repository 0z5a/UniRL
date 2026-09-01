# Discrete-timestep improved-MeanFlow (iMF) trainer.
#
# This file is additive only. It does not modify the existing iMF trainer.
# It keeps the original iMF loss construction, but replaces continuous t and
# r~U(0,t) sampling with shared video/audio discrete inference-index sampling.
#
# Usage:
#   ... run_pure_torch_leo_imf.sh <config> \
#     --trainer distill.imf_discrete_trainer.IMFDiscreteTrainer
#
# Optional env overrides:
#   IMF_DISCRETE_TOTAL_STEPS=960
#   IMF_DISCRETE_VIDEO_SHIFT=9
#   IMF_DISCRETE_AUDIO_SHIFT=1
#   IMF_DISCRETE_CP_CHECK_STEPS=3   # 0=disable, -1=always

import functools
import os

import torch
import torch.distributed as dist

from .imf_trainer import IMFTrainer
from ...diffusion.flow.imf_loss import (
    broadcast_time_,
    compute_cfg_target,
    compute_dudt,
    make_imf_loss_fn,
    perturb_latents,
    time_binary,
    time_unary,
)


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else float(default)


class IMFDiscreteTrainer(IMFTrainer):
    """iMF trainer with discrete inference-index t/r sampling.

    The original iMF objective is unchanged. Only the time sampler changes:
    draw a base t index x and an r index y>x from a 960-step grid. Video/audio
    share the same x/y indices, then map them to shifted flow times using their
    branch-specific inference shifts. Thus actual video/audio timestep values may
    differ, but their discrete inference positions are one-to-one.
    """

    TOTAL_STEPS = _env_int("IMF_DISCRETE_TOTAL_STEPS", 960)
    VIDEO_SHIFT = _env_float("IMF_DISCRETE_VIDEO_SHIFT", 9.0)
    AUDIO_SHIFT = _env_float("IMF_DISCRETE_AUDIO_SHIFT", 1.0)
    CP_CHECK_STEPS = _env_int("IMF_DISCRETE_CP_CHECK_STEPS", 3)

    def after_initialize(self):
        super().after_initialize()
        self._check_inference_time_semantics()
        if getattr(self, "rank", 0) == 0:
            self.logger.info(
                "[iMF-DISCRETE] enabled: "
                f"discrete_grid_steps={self.TOTAL_STEPS}, "
                f"video_shift={self.VIDEO_SHIFT}, audio_shift={self.AUDIO_SHIFT}, "
                f"video_model_time_scale={getattr(self.video_denoiser, 'training_timesteps', None)}, "
                f"audio_model_time_scale={getattr(getattr(self, 'audio_denoiser', None), 'training_timesteps', None)}."
            )

    def _check_inference_time_semantics(self):
        for name, denoiser in (("video", self.video_denoiser), ("audio", getattr(self, "audio_denoiser", None))):
            if denoiser is None:
                continue
            if not getattr(denoiser.path_sampler, "reverse", False):
                raise ValueError(
                    f"[iMF-DISCRETE] {name} denoiser is not reverse-time. "
                    "Expected flow_reverse=True: index 0/sigma=1 is noise, final index/sigma=0 is clean."
                )
        if getattr(self, "rank", 0) == 0:
            v0 = self._time_shift(torch.tensor(1.0), self.VIDEO_SHIFT).item()
            v1 = self._time_shift(torch.tensor(0.0), self.VIDEO_SHIFT).item()
            a0 = self._time_shift(torch.tensor(1.0), self.AUDIO_SHIFT).item()
            a1 = self._time_shift(torch.tensor(0.0), self.AUDIO_SHIFT).item()
            self.logger.info(
                "[iMF-DISCRETE] time semantics checked: "
                f"base index 0 -> sigma=1/noise, base index {self.TOTAL_STEPS} -> sigma=0/clean; "
                f"video_shift endpoints=({v0:.4f}->{v1:.4f}), audio_shift endpoints=({a0:.4f}->{a1:.4f})."
            )

    @staticmethod
    def _is_seq(x):
        return isinstance(x, (list, tuple))

    @staticmethod
    def _time_shift(t, shift):
        if float(shift) == 1.0:
            return t
        return (shift * t) / (1 + (shift - 1) * t)

    def _build_timestep_tables(self, device):
        key = (self.TOTAL_STEPS, self.VIDEO_SHIFT, self.AUDIO_SHIFT)
        if getattr(self, "_imf_discrete_time_cache_key", None) != key:
            base = torch.arange(self.TOTAL_STEPS, -1, -1, dtype=torch.float32) / float(self.TOTAL_STEPS)
            self._imf_discrete_time_cache = {
                "video": self._time_shift(base, self.VIDEO_SHIFT),
                "audio": self._time_shift(base, self.AUDIO_SHIFT),
            }
            self._imf_discrete_time_cache_key = key
            if getattr(self, "rank", 0) == 0:
                self.logger.info(
                    f"[iMF-DISCRETE] built {base.numel()} shifted timestep values "
                    f"for base indices [0, {self.TOTAL_STEPS}]."
                )
        return {k: v.to(device=device) for k, v in self._imf_discrete_time_cache.items()}

    def _sample_index_pair_like(self, ref):
        if self._is_seq(ref):
            pairs = [self._sample_index_pair_like(x) for x in ref]
            return [p[0] for p in pairs], [p[1] for p in pairs]

        t_idx = torch.randint(
            self.TOTAL_STEPS, ref.shape, device=ref.device, generator=self.vae.generator,
            dtype=torch.long,
        )
        max_delta = (self.TOTAL_STEPS - t_idx).clamp_min(1)
        u = torch.rand(ref.shape, device=ref.device, generator=self.vae.generator)
        r_idx = t_idx + (u * max_delta.float()).floor().long() + 1
        r_idx = r_idx.clamp_max(self.TOTAL_STEPS)
        return t_idx, r_idx

    def _gather_like(self, indices, table, ref):
        if self._is_seq(indices):
            return [self._gather_like(i, table, r) for i, r in zip(indices, ref)]
        return table.to(indices.device)[indices].to(dtype=ref.dtype)

    @staticmethod
    def _flatten_indices(x, out):
        if torch.is_tensor(x):
            out.append(x.detach().long().reshape(-1))
        elif isinstance(x, (list, tuple)):
            for y in x:
                IMFDiscreteTrainer._flatten_indices(y, out)

    def _index_vector(self, device, *items):
        parts = []
        for item in items:
            self._flatten_indices(item, parts)
        if parts:
            return torch.cat([p.to(device=device) for p in parts])
        return torch.empty(0, dtype=torch.long, device=device)

    def _check_ordered_index_pair(self, t_indices, r_indices):
        device = torch.device("cuda", self.args.local_rank)
        t_flat = self._index_vector(device, t_indices)
        r_flat = self._index_vector(device, r_indices)
        if t_flat.numel() != r_flat.numel() or not torch.all(r_flat > t_flat):
            raise RuntimeError("[iMF-DISCRETE] expected ordered timestep indices r_idx > t_idx.")
        if t_flat.numel() > 0 and (int(t_flat.min().item()) < 0 or int(r_flat.max().item()) > self.TOTAL_STEPS):
            raise RuntimeError("[iMF-DISCRETE] timestep indices are out of the discrete grid range.")

    def _check_cp_group_index_consistency(self, t_indices, r_indices):
        if self.CP_CHECK_STEPS == 0:
            return
        checked = getattr(self, "_imf_cp_check_count", 0)
        if self.CP_CHECK_STEPS > 0 and checked >= self.CP_CHECK_STEPS:
            return
        self._imf_cp_check_count = checked + 1

        if not (dist.is_available() and dist.is_initialized()):
            return

        device = torch.device("cuda", self.args.local_rank)
        flat = self._index_vector(device, t_indices, r_indices)
        local_bad = torch.zeros((), dtype=torch.long, device=device)

        cp_group = getattr(self.p_state, "cp_group", None)
        cp_size = getattr(self.p_state, "cp_size", 1)
        if cp_group is not None and cp_size > 1:
            local_n = torch.tensor([flat.numel()], dtype=torch.long, device=device)
            sizes = [torch.zeros_like(local_n) for _ in range(cp_size)]
            dist.all_gather(sizes, local_n, group=cp_group)
            same_size = all(int(s.item()) == int(local_n.item()) for s in sizes)
            if not same_size:
                local_bad.fill_(1)
            else:
                ref = flat.clone()
                src = dist.get_global_rank(cp_group, 0)
                dist.broadcast(ref, src=src, group=cp_group)
                if not torch.equal(flat, ref):
                    local_bad.fill_(1)

        global_bad = local_bad.clone()
        dist.all_reduce(global_bad, op=dist.ReduceOp.SUM)
        if int(global_bad.item()) != 0:
            raise RuntimeError("[iMF-DISCRETE] t/r index consistency check failed in at least one CP group.")
        if getattr(self, "rank", 0) == 0:
            self.logger.info(
                f"[iMF-DISCRETE] CP t/r index consistency check passed "
                f"({self._imf_cp_check_count}/{self.CP_CHECK_STEPS if self.CP_CHECK_STEPS > 0 else 'always'})."
            )

    def _assert_same_structure(self, indices, ref, name):
        if self._is_seq(indices):
            if not self._is_seq(ref) or len(indices) != len(ref):
                raise ValueError(f"{name} timestep structure differs from video timestep structure.")
            for i, r in zip(indices, ref):
                self._assert_same_structure(i, r, name)
            return
        if not torch.is_tensor(ref) or indices.shape != ref.shape:
            raise ValueError(
                f"{name} timestep shape differs from video: {tuple(indices.shape)} vs "
                f"{None if not torch.is_tensor(ref) else tuple(ref.shape)}"
            )

    @staticmethod
    def _broadcast_t(t, ref):
        t = t.float()
        if t.ndim == 0 or t.numel() == 1:
            return t.reshape(())
        if ref.ndim >= 1 and t.shape[0] == ref.shape[0]:
            return t.view(t.shape[0], *([1] * (ref.ndim - 1)))
        return t.view(t.shape[0], *([1] * max(ref.ndim - 1, 0)))

    def _recompute_xt(self, x0, ut, t, denoiser):
        if self._is_seq(x0):
            return [self._recompute_xt(x, u, tt, denoiser) for x, u, tt in zip(x0, ut, t)]

        tb = self._broadcast_t(t, x0)
        x0f = x0.float()
        utf = ut.float()
        x1 = x0f - utf if denoiser.path_sampler.reverse else x0f + utf
        if denoiser.path_sampler.reverse:
            xt = (1.0 - tb) * x1 + tb * x0f
        else:
            xt = tb * x1 + (1.0 - tb) * x0f
        return xt.to(dtype=x0.dtype)

    def _replace_prefix(self, latents, xt):
        if self._is_seq(latents):
            return [self._replace_prefix(x, y) for x, y in zip(latents, xt)]
        if latents.shape == xt.shape:
            return xt.to(dtype=latents.dtype)

        diff_dims = [i for i, (a, b) in enumerate(zip(latents.shape, xt.shape)) if a != b]
        if len(diff_dims) != 1:
            raise ValueError(f"Cannot replace latent prefix: latents={tuple(latents.shape)}, xt={tuple(xt.shape)}")
        dim = diff_dims[0]
        if latents.shape[dim] < xt.shape[dim]:
            raise ValueError(f"Latent prefix is smaller than xt: latents={tuple(latents.shape)}, xt={tuple(xt.shape)}")

        out = latents.clone()
        sl = [slice(None)] * out.ndim
        sl[dim] = slice(0, xt.shape[dim])
        out[tuple(sl)] = xt.to(dtype=out.dtype)
        return out

    @staticmethod
    def _flatten_time(x, out):
        if torch.is_tensor(x):
            out.append(x.detach().float().reshape(-1))
        elif isinstance(x, (list, tuple)):
            for y in x:
                IMFDiscreteTrainer._flatten_time(y, out)

    def _log_time_stats(self, name, t, r):
        ts, rs = [], []
        self._flatten_time(t, ts)
        self._flatten_time(r, rs)
        if not ts or not rs:
            return
        tv = torch.cat(ts)
        rv = torch.cat(rs)
        self.logger.info(
            f"[iMF-DISCRETE][{name}] "
            f"t_mean={tv.mean().item():.4f}, r_mean={rv.mean().item():.4f}, "
            f"t_minmax=({tv.min().item():.4f},{tv.max().item():.4f}), "
            f"r_minmax=({rv.min().item():.4f},{rv.max().item():.4f}), n={tv.numel()}"
        )

    def _sample_discrete_times(self, model_input_kwargs, video_loss_fn, audio_loss_fn, has_audio):
        v_t_ref = video_loss_fn.keywords["t"]
        tables = self._build_timestep_tables(torch.device("cuda", self.args.local_rank))
        t_indices, r_indices = self._sample_index_pair_like(v_t_ref)

        cp_group = getattr(self.p_state, "cp_group", None)
        cp_size = getattr(self.p_state, "cp_size", 1)
        if cp_group is not None and cp_size > 1:
            src = dist.get_global_rank(cp_group, 0)
            broadcast_time_(t_indices, src, cp_group)
            broadcast_time_(r_indices, src, cp_group)
        self._check_ordered_index_pair(t_indices, r_indices)
        self._check_cp_group_index_consistency(t_indices, r_indices)

        v_t = self._gather_like(t_indices, tables["video"], v_t_ref)
        v_r = self._gather_like(r_indices, tables["video"], v_t_ref)
        v_x0 = video_loss_fn.keywords["x0"]
        v_ut = video_loss_fn.keywords["ut"]
        v_xt = self._recompute_xt(v_x0, v_ut, v_t, self.video_denoiser)

        model_input_kwargs["latents"] = self._replace_prefix(model_input_kwargs["latents"], v_xt)
        model_input_kwargs["timesteps"] = time_unary(v_t, self.video_denoiser.get_model_t)
        model_input_kwargs["ut"] = v_ut
        video_loss_fn.keywords.update(t=v_t, xt=v_xt, ut=v_ut)

        if has_audio:
            a_t_ref = audio_loss_fn.keywords["t"]
            self._assert_same_structure(t_indices, a_t_ref, "audio")
            a_t = self._gather_like(t_indices, tables["audio"], a_t_ref)
            a_r = self._gather_like(r_indices, tables["audio"], a_t_ref)
            a_x0 = audio_loss_fn.keywords["x0"]
            a_ut = audio_loss_fn.keywords["ut"]
            a_xt = self._recompute_xt(a_x0, a_ut, a_t, self.audio_denoiser)

            model_input_kwargs["audio_latents"] = self._replace_prefix(model_input_kwargs["audio_latents"], a_xt)
            model_input_kwargs["audio_timesteps"] = time_unary(a_t, self.audio_denoiser.get_model_t)
            model_input_kwargs["aut"] = a_ut
            audio_loss_fn.keywords.update(t=a_t, xt=a_xt, ut=a_ut)
        else:
            a_t, a_r = None, None

        if getattr(self, "_imf_discrete_dbg_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._imf_discrete_dbg_left = getattr(self, "_imf_discrete_dbg_left", 3) - 1
            self._log_time_stats("video", v_t, v_r)
            if a_t is not None:
                self._log_time_stats("audio", a_t, a_r)

        return v_t, v_r, a_t, a_r

    def prepare_model_inputs(self, batch, device):
        cfg = self._imf_cfg()
        guidance = cfg["guidance"]
        anchor_weight = cfg["anchor_weight"]
        dt = cfg["fd_dt"]

        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = super(IMFTrainer, self).prepare_model_inputs(batch, device)

        video_loss_fn = model_input_kwargs.get("diffusion_loss_fn", None)
        audio_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn", None)

        has_video = ("latents" in model_input_kwargs) and isinstance(video_loss_fn, functools.partial)
        has_audio = (
            ("audio_latents" in model_input_kwargs)
            and isinstance(audio_loss_fn, functools.partial)
            and audio_loss_fn.keywords.get("ut", None) is not None
        )

        if not has_video:
            return model_input_kwargs, bsz, seqlen

        v_t, v_r, a_t, a_r = self._sample_discrete_times(
            model_input_kwargs, video_loss_fn, audio_loss_fn, has_audio
        )
        v_ut = video_loss_fn.keywords["ut"]
        a_ut = audio_loss_fn.keywords["ut"] if has_audio else None

        with torch.autocast(device_type="cuda", enabled=False):
            uncond_text_states, uncond_text_mask = self._build_uncond_text(batch, device)

        is_mf = getattr(self.args, "model_structure", "") == "LeoModelMeanFlow"
        if is_mf:
            v_r_mt = time_unary(v_r, self.video_denoiser.get_model_t)
            a_r_mt = time_unary(a_r, self.audio_denoiser.get_model_t) if has_audio else None

        v_t_plus = time_unary(v_t, lambda x: (x.float() + dt).clamp(max=1.0))
        v_t_minus = time_unary(v_t, lambda x: (x.float() - dt).clamp(min=0.0))
        v_span = time_binary(v_t_plus, v_t_minus, lambda a, b: (a - b).clamp(min=1e-6))
        v_step_plus = time_binary(v_t_plus, v_t, lambda a, b: a - b.float())
        v_step_minus = time_binary(v_t_minus, v_t, lambda a, b: a - b.float())
        v_ts_plus = time_unary(v_t_plus, self.video_denoiser.get_model_t)
        v_ts_minus = time_unary(v_t_minus, self.video_denoiser.get_model_t)
        if has_audio:
            a_t_plus = time_unary(a_t, lambda x: (x.float() + dt).clamp(max=1.0))
            a_t_minus = time_unary(a_t, lambda x: (x.float() - dt).clamp(min=0.0))
            a_span = time_binary(a_t_plus, a_t_minus, lambda a, b: (a - b).clamp(min=1e-6))
            a_step_plus = time_binary(a_t_plus, a_t, lambda a, b: a - b.float())
            a_step_minus = time_binary(a_t_minus, a_t, lambda a, b: a - b.float())
            a_ts_plus = time_unary(a_t_plus, self.audio_denoiser.get_model_t)
            a_ts_minus = time_unary(a_t_minus, self.audio_denoiser.get_model_t)

        self.model_engine.eval()
        for m in self.model_engine.fsdp_models:
            m.train()

        try:
            with torch.no_grad():
                v_cond_v, v_cond_a = self._student_predict(model_input_kwargs, has_audio)

                uncond_kwargs = dict(model_input_kwargs)
                uncond_kwargs["cond_text_states"] = uncond_text_states
                uncond_kwargs["cond_text_mask"] = uncond_text_mask
                v_unc_v, v_unc_a = self._student_predict(uncond_kwargs, has_audio)

                distill_audio = has_audio and v_cond_a is not None and v_unc_a is not None
                if has_audio and not distill_audio and getattr(self, "rank", 0) == 0:
                    self.logger.warning(
                        "[iMF-DISCRETE] audio_diffusion_prediction is unavailable; "
                        "skip audio iMF distillation and keep the original audio FM loss."
                    )

                plus_kwargs = dict(model_input_kwargs)
                plus_kwargs["latents"] = perturb_latents(model_input_kwargs["latents"], v_cond_v, v_step_plus)
                plus_kwargs["timesteps"] = v_ts_plus
                if distill_audio:
                    plus_kwargs["audio_latents"] = perturb_latents(
                        model_input_kwargs["audio_latents"], v_cond_a, a_step_plus
                    )
                    plus_kwargs["audio_timesteps"] = a_ts_plus
                if is_mf:
                    plus_kwargs["timestep_r"] = v_r_mt
                    if distill_audio:
                        plus_kwargs["audio_timestep_r"] = a_r_mt
                v_plus_v, v_plus_a = self._student_predict(plus_kwargs, distill_audio)

                minus_kwargs = dict(model_input_kwargs)
                minus_kwargs["latents"] = perturb_latents(model_input_kwargs["latents"], v_cond_v, v_step_minus)
                minus_kwargs["timesteps"] = v_ts_minus
                if distill_audio:
                    minus_kwargs["audio_latents"] = perturb_latents(
                        model_input_kwargs["audio_latents"], v_cond_a, a_step_minus
                    )
                    minus_kwargs["audio_timesteps"] = a_ts_minus
                if is_mf:
                    minus_kwargs["timestep_r"] = v_r_mt
                    if distill_audio:
                        minus_kwargs["audio_timestep_r"] = a_r_mt
                v_minus_v, v_minus_a = self._student_predict(minus_kwargs, distill_audio)

                v_tgt_v = compute_cfg_target(v_ut, v_cond_v, v_unc_v, guidance)
                dudt_v = compute_dudt(v_plus_v, v_minus_v, v_span)
                if distill_audio:
                    v_tgt_a = compute_cfg_target(a_ut, v_cond_a, v_unc_a, guidance)
                    dudt_a = compute_dudt(v_plus_a, v_minus_a, a_span)
        finally:
            self.model_engine.train()

        time_diff_v = time_binary(v_t, v_r, lambda a, b: a.float() - b.float())
        model_input_kwargs["diffusion_loss_fn"] = make_imf_loss_fn(time_diff_v, v_tgt_v, dudt_v, anchor_weight)
        if distill_audio:
            time_diff_a = time_binary(a_t, a_r, lambda a, b: a.float() - b.float())
            model_input_kwargs["audio_diffusion_loss_fn"] = make_imf_loss_fn(
                time_diff_a, v_tgt_a, dudt_a, anchor_weight
            )

        if is_mf:
            model_input_kwargs["timestep_r"] = v_r_mt
            if distill_audio:
                model_input_kwargs["audio_timestep_r"] = a_r_mt

        return model_input_kwargs, bsz, seqlen


MultimodalTrainer = IMFDiscreteTrainer
