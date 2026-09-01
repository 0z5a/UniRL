# Discrete-inference-timestep CFGD MeanFlow trainer.
#
# This file is additive only. It does not modify existing trainer / config / scheduler code.
#
# Usage:
#   ... run_pure_torch_leo_cfgd_meanflow.sh <config> \
#     --trainer distill.cfgdistill_discrete_meanflow_trainer.MultimodalTrainer
#
# Optional env overrides:
#   CFGD_DISCRETE_TOTAL_STEPS=960
#   CFGD_DISCRETE_VIDEO_SHIFT=9
#   CFGD_DISCRETE_AUDIO_SHIFT=1
#   CFGD_DISCRETE_CP_CHECK_STEPS=3   # 0=disable, -1=always

import os

import torch
import torch.distributed as dist

from .cfgdistill_meanflow_trainer import MultimodalTrainer as CFGDMeanFlowTrainer
from ...diffusion.flow.imf_loss import broadcast_time_, time_unary


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in (None, "") else int(default)


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else float(default)


class MultimodalTrainer(CFGDMeanFlowTrainer):
    """CFGD-MF trainer whose t/r are sampled by shared discrete timestep indices.

    We build the 960-step inference index grid once: index 0 is sigma=1 (pure
    noise under flow_reverse=True), index 960 is sigma=0 (clean sample). For every
    sample, draw a t index x and an r index y>x on this base grid. Video and audio
    use the same x/y indices, then apply their own inference time shift (video=9,
    audio=1), so numerical times may differ while discrete inference positions stay
    exactly one-to-one. Model timesteps are scaled by the existing denoiser scale,
    matching the current Leo2 inference scheduler unless inference is changed too.
    """

    TOTAL_STEPS = _env_int("CFGD_DISCRETE_TOTAL_STEPS", 960)
    VIDEO_SHIFT = _env_float("CFGD_DISCRETE_VIDEO_SHIFT", 9.0)
    AUDIO_SHIFT = _env_float("CFGD_DISCRETE_AUDIO_SHIFT", 1.0)
    CP_CHECK_STEPS = _env_int("CFGD_DISCRETE_CP_CHECK_STEPS", 3)

    def build_teacher_model(self):
        orig_model_structure = getattr(self.args, "model_structure", None)
        self.args.model_structure = "LeoModel"
        try:
            return super().build_teacher_model()
        finally:
            self.args.model_structure = orig_model_structure

    def after_initialize(self):
        super().after_initialize()
        self._check_inference_time_semantics()
        self.logger.info(
            "[CFGD-MF-DISCRETE] enabled: "
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
                    f"[CFGD-MF-DISCRETE] {name} denoiser is not reverse-time. "
                    "The provided Leo2 inference command uses flow_reverse=True, where index 0/sigma=1 is noise "
                    "and the last index/sigma=0 is clean. Refuse to run to avoid inverted shift semantics."
                )
        if getattr(self, "rank", 0) == 0:
            v0 = self._time_shift(torch.tensor(1.0), self.VIDEO_SHIFT).item()
            v1 = self._time_shift(torch.tensor(0.0), self.VIDEO_SHIFT).item()
            a0 = self._time_shift(torch.tensor(1.0), self.AUDIO_SHIFT).item()
            a1 = self._time_shift(torch.tensor(0.0), self.AUDIO_SHIFT).item()
            self.logger.info(
                "[CFGD-MF-DISCRETE] time semantics checked against FlowMatchDiscreteScheduler(reverse=True): "
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
        if getattr(self, "_discrete_time_cache_key", None) != key:
            base = torch.arange(self.TOTAL_STEPS, -1, -1, dtype=torch.float32) / float(self.TOTAL_STEPS)
            self._discrete_time_cache = {
                "video": self._time_shift(base, self.VIDEO_SHIFT),
                "audio": self._time_shift(base, self.AUDIO_SHIFT),
            }
            self._discrete_time_cache_key = key
            if getattr(self, "rank", 0) == 0:
                self.logger.info(
                    f"[CFGD-MF-DISCRETE] built {base.numel()} shifted timestep values "
                    f"for base indices [0, {self.TOTAL_STEPS}]."
                )
        return {k: v.to(device=device) for k, v in self._discrete_time_cache.items()}

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
                MultimodalTrainer._flatten_indices(y, out)

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
            raise RuntimeError("[CFGD-MF-DISCRETE] expected ordered timestep indices r_idx > t_idx.")
        if t_flat.numel() > 0 and (int(t_flat.min().item()) < 0 or int(r_flat.max().item()) > self.TOTAL_STEPS):
            raise RuntimeError("[CFGD-MF-DISCRETE] timestep indices are out of the 960-step grid range.")

    def _check_cp_group_index_consistency(self, t_indices, r_indices):
        if self.CP_CHECK_STEPS == 0:
            return
        checked = getattr(self, "_cfgd_cp_check_count", 0)
        if self.CP_CHECK_STEPS > 0 and checked >= self.CP_CHECK_STEPS:
            return
        self._cfgd_cp_check_count = checked + 1

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
            raise RuntimeError(
                "[CFGD-MF-DISCRETE] t/r index consistency check failed in at least one CP group."
            )
        if getattr(self, "rank", 0) == 0:
            self.logger.info(
                f"[CFGD-MF-DISCRETE] CP t/r index consistency check passed "
                f"({self._cfgd_cp_check_count}/{self.CP_CHECK_STEPS if self.CP_CHECK_STEPS > 0 else 'always'})."
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
                MultimodalTrainer._flatten_time(y, out)

    def _log_time_stats(self, name, t, r):
        ts, rs = [], []
        self._flatten_time(t, ts)
        self._flatten_time(r, rs)
        if not ts or not rs:
            return
        tv = torch.cat(ts)
        rv = torch.cat(rs)
        self.logger.info(
            f"[CFGD-MF-DISCRETE][{name}] "
            f"t_mean={tv.mean().item():.4f}, r_mean={rv.mean().item():.4f}, "
            f"t_minmax=({tv.min().item():.4f},{tv.max().item():.4f}), "
            f"r_minmax=({rv.min().item():.4f},{rv.max().item():.4f}), n={tv.numel()}"
        )

    def _inject_meanflow_r(self, model_input_kwargs):
        if getattr(self.args, "model_structure", "") != "LeoModelMeanFlow":
            raise ValueError("CFGD-MF-DISCRETE requires --model-structure LeoModelMeanFlow.")

        video_loss_fn = model_input_kwargs.get("diffusion_loss_fn", None)
        audio_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn", None)
        has_video = self._is_real_branch(model_input_kwargs.get("latents", None), video_loss_fn)
        has_audio = self._is_real_branch(model_input_kwargs.get("audio_latents", None), audio_loss_fn) \
            and audio_loss_fn.keywords.get("ut", None) is not None
        if not has_video:
            raise ValueError("CFGD-MF-DISCRETE expects a real video branch with flow time t.")

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
        model_input_kwargs["timestep_r"] = time_unary(v_r, self.video_denoiser.get_model_t)
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
            model_input_kwargs["audio_timestep_r"] = time_unary(a_r, self.audio_denoiser.get_model_t)
            model_input_kwargs["aut"] = a_ut
            audio_loss_fn.keywords.update(t=a_t, xt=a_xt, ut=a_ut)
        else:
            a_t, a_r = None, None

        if getattr(self, "_cfgd_discrete_dbg_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgd_discrete_dbg_left = getattr(self, "_cfgd_discrete_dbg_left", 3) - 1
            self._log_time_stats("video", v_t, v_r)
            if a_t is not None:
                self._log_time_stats("audio", a_t, a_r)
