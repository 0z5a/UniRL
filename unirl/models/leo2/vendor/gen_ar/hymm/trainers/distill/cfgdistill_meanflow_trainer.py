# MeanFlow-compatible CFG distillation trainer.
#
# This is a NEW file and does NOT modify any existing file.
#
# Objective:
#   student(x_t, t, r, cond) ~= teacher(x_t, t, uncond)
#                              + w * (teacher(x_t, t, cond) - teacher(x_t, t, uncond))
#
# Key points:
#   * Student is a real two-time MeanFlow backbone (`LeoModelMeanFlow`) and receives
#     `timestep_r` / `audio_timestep_r` in the gradient forward.
#   * Teacher target is the standard CFG velocity at time t only; teacher kwargs have
#     the second-time fields stripped, so teacher(x_t, t, cond/uncond) is used.
#   * r sampling is exactly the same as `imf_trainer`: r ~ U(0, t), sampled from the
#     resumable VAE generator and broadcast inside the CP group.
#   * Video/audio branch losses and weighting are inherited from cfgdistill_trainer_v3:
#     loss = image_loss_weight * video_cfgd_loss + audio_loss_weight * audio_cfgd_loss.

import functools

import torch
import torch.distributed as dist

from .cfgdistill_trainer_v3 import MultimodalTrainer as CfgDistillV3Trainer
from ...diffusion.flow.imf_loss import (
    broadcast_time_,
    rand_like_scaled,
    time_unary,
)

# Import installs the non-invasive model dispatch for `--model-structure LeoModelMeanFlow`.
from ...models.diffusion import leo_meanflow as _leo_meanflow  # noqa: F401


class MultimodalTrainer(CfgDistillV3Trainer):
    """CFG distillation whose student is conditioned on MeanFlow's second time r."""

    def after_initialize(self):
        super().after_initialize()
        self._disable_student_anchor()
        self.logger.info(
            "[CFGD-MF] initialized: student uses timestep_r/audio_timestep_r; "
            "teacher CFG target strips r and stays teacher(x_t, t, cond/uncond)."
        )

    def _disable_student_anchor(self):
        """Disable LeoModelMeanFlow's extra u(t,t) anchor recompute.

        CFGD-MF only needs student u(t,r) to regress to the teacher CFG target. The
        MeanFlow model's optional anchor path is for iMF loss and is unused here; leaving
        it on would waste the last 1/4 transformer tail+head and may pass
        `model_output_anchor` into ordinary FM loss hooks. This mirrors the safe opt-in
        switch provided by `leo_meanflow.py`.
        """
        n = 0
        seen = set()
        candidates = []
        m0 = getattr(self.model_engine, "model", None)
        if m0 is not None:
            candidates.append(m0)
        candidates.extend(list(getattr(self.model_engine, "fsdp_models", [])))
        for root in candidates:
            if root is None:
                continue
            for m in root.modules():
                if id(m) in seen:
                    continue
                seen.add(id(m))
                if hasattr(m, "_mf_run_tail"):
                    m._gan_disable_anchor = True
                    n += 1
        self.logger.info(f"[CFGD-MF] disabled MeanFlow anchor recompute on {n} student module(s).")

    def _sample_r(self, t):
        """Sample r with the same policy as iMF: r ~ U(0, t), CP-consistent."""
        r = rand_like_scaled(t, generator=self.vae.generator)
        cp_group = getattr(self.p_state, "cp_group", None)
        cp_size = getattr(self.p_state, "cp_size", 1)
        if cp_group is not None and cp_size > 1:
            src = dist.get_global_rank(cp_group, 0)
            broadcast_time_(r, src, cp_group)
        return r

    @staticmethod
    def _is_real_branch(latent_key, loss_fn):
        return latent_key is not None and isinstance(loss_fn, functools.partial) \
            and loss_fn.keywords.get("t", None) is not None

    def _inject_meanflow_r(self, model_input_kwargs):
        if getattr(self.args, "model_structure", "") != "LeoModelMeanFlow":
            raise ValueError(
                "CFGD-MF requires `--model-structure LeoModelMeanFlow` so the student "
                "can consume (t, r). Use jobs/examples/run_pure_torch_leo_cfgd_meanflow.sh "
                "or pass the flag explicitly."
            )

        video_loss_fn = model_input_kwargs.get("diffusion_loss_fn", None)
        audio_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn", None)

        has_video = self._is_real_branch(model_input_kwargs.get("latents", None), video_loss_fn)
        has_audio = self._is_real_branch(model_input_kwargs.get("audio_latents", None), audio_loss_fn) \
            and audio_loss_fn.keywords.get("ut", None) is not None

        if not has_video:
            raise ValueError("CFGD-MF expects a real video branch with diffusion_loss_fn carrying flow time t.")

        v_t = video_loss_fn.keywords["t"]
        v_r = self._sample_r(v_t)
        model_input_kwargs["timestep_r"] = time_unary(v_r, self.video_denoiser.get_model_t)

        if has_audio:
            a_t = audio_loss_fn.keywords["t"]
            a_r = self._sample_r(a_t)
            model_input_kwargs["audio_timestep_r"] = time_unary(a_r, self.audio_denoiser.get_model_t)

        if getattr(self, "_cfgdmf_dbg_left", 3) > 0 and getattr(self, "rank", 0) == 0:
            self._cfgdmf_dbg_left = getattr(self, "_cfgdmf_dbg_left", 3) - 1

            def _flat_time(x, out):
                if torch.is_tensor(x):
                    out.append(x.detach().float().reshape(-1))
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        _flat_time(y, out)

            def _stat(name, t, r):
                ts, rs = [], []
                _flat_time(t, ts)
                _flat_time(r, rs)
                if ts and rs:
                    tv = torch.cat(ts)
                    rv = torch.cat(rs)
                    self.logger.info(
                        f"[CFGD-MF][{name}] sampled r~U(0,t): "
                        f"t_mean={tv.mean().item():.4f}, r_mean={rv.mean().item():.4f}, "
                        f"mean(r/t)={(rv / tv.clamp_min(1e-6)).mean().item():.4f}, n={tv.numel()}"
                    )

            _stat("video", v_t, v_r)
            if has_audio:
                _stat("audio", a_t, a_r)

    def prepare_model_inputs(self, batch, device):
        # v3 builds cond kwargs + uncond text kwargs and keeps standard x_t/t sampling.
        model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen = super().prepare_model_inputs(batch, device)
        # Add MeanFlow's second time only to the student gradient forward.
        self._inject_meanflow_r(model_input_kwargs)
        return model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen

    @torch.no_grad()
    def prepare_cfgdistill_target(self, model_input_kwargs, uncond_model_input_kwargs, device):
        # Teacher target must be teacher(x_t, t, cond/uncond), not teacher(x_t, t, r, ...).
        teacher_kwargs = dict(model_input_kwargs)
        teacher_kwargs.pop("timestep_r", None)
        teacher_kwargs.pop("audio_timestep_r", None)
        return super().prepare_cfgdistill_target(teacher_kwargs, uncond_model_input_kwargs, device)
