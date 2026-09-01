# MeanFlow (iMF) sampling pipeline for the Leo two-time backbone.
#
# This is a NEW, self-contained module. It does NOT modify any existing file.
#
# ============================ Why a new pipeline ============================
# A plain flow-matching model predicts the INSTANTANEOUS velocity v(x_t, t). The Euler
# update of `FlowMatchDiscreteScheduler.step` is
#
#       prev_sample = sample + derivative * dt ,   derivative = v ,
#       dt = sigma_next - sigma                                          (< 0, reverse)
#
# i.e.  x_r = x_t + v * (sigma_r - sigma_t) = x_t - (t - r) * v .
#
# A MeanFlow / improved-MeanFlow (iMF) model instead predicts the AVERAGE velocity
# u(x_t, t, r) over the interval [r, t], where r is the NEXT timestep. The SAME Euler
# update is then EXACT for any step size:
#
#       x_r = x_t - (t - r) * u(x_t, t, r) .
#
# So nothing about the scheduler math changes. The ONLY thing the iMF model needs that a
# plain diffusion model does not is the SECOND time input `r` (== the next schedule
# timestep). With `r` fed in, one can use very few steps (even a single step).
#
# ============================ How it is done ============================
# We subclass the unmodified `Leo2Pipeline` and transparently wrap the model's forward
# for the duration of `__call__`, injecting `timestep_r` / `audio_timestep_r` (the next
# entry of each branch scheduler's `timesteps_full`) before delegating to the original,
# fully-tested sampling loop. The scheduler `.step()` (and everything else: CFG, channel
# extension, decoding) is reused verbatim.
#
# NOTE: this is intended for the Euler solver (the MeanFlow default). Each denoising step
# calls the model exactly once, in schedule order, so the per-step `r` is simply the
# next timestep `timesteps_full[i + 1]`.

import torch

from .pipeline_leo import Leo2Pipeline


class Leo2PipelineMeanFlow(Leo2Pipeline):
    """`Leo2Pipeline` that feeds the second time input ``r`` (next timestep) each step."""

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        # ---- iMF / MeanFlow hard requirements ----
        # (1) NO classifier-free guidance. The guidance is already distilled INTO the model
        #     (training target v_tgt = u_t + (1 - 1/w)(v_cond - v_uncond)), so the model's
        #     single (cond) forward already outputs the guided average velocity. Applying CFG
        #     again at sampling would double-count it. guidance_scale MUST be 1.0 -> this also
        #     makes cfg_factor == 1, i.e. ONE forward per step (no uncond pass).
        # (2) Euler solver only: x_r = x_t - (t - r) * u(x_t, t, r). The flow solver is already
        #     constrained to "euler" by the CLI (`--flow-solver`, choices=["euler"]), so the
        #     scheduler.step Euler update is exactly the MeanFlow update; nothing to override.
        kwargs["guidance_scale"] = 1.0

        # Resolve the per-branch schedulers exactly the way Leo2Pipeline.__call__ does,
        # so we can read each branch's NEXT timestep (== r) without duplicating the loop.
        image_size = kwargs.get("image_size", None)
        video_duration = kwargs.get("video_duration", 1)
        if image_size is not None:
            visual_scheduler = self.scheduler if video_duration == 1 else self.video_scheduler
        else:
            visual_scheduler = None
        audio_scheduler = self.audio_scheduler if kwargs.get("audio_duration", None) is not None else None

        model = self.model
        # Use the *class* forward so nn.Module.__call__ (and any FSDP / forward hooks)
        # keep working; we only shadow the bound forward with a thin injector closure.
        real_forward = type(model).forward
        state = {"i": 0}

        def mf_forward(*fwd_args, **fwd_kwargs):
            i = state["i"]
            ts = fwd_kwargs.get("timesteps", None)
            if ts is not None and fwd_kwargs.get("timestep_r", None) is None and visual_scheduler is not None:
                # r = next timestep, same (sigma * num_train_timesteps) model-time units as t.
                r = visual_scheduler.timesteps_full[i + 1]
                fwd_kwargs["timestep_r"] = r.repeat(ts.shape[0]).to(ts)
            ats = fwd_kwargs.get("audio_timesteps", None)
            if ats is not None and fwd_kwargs.get("audio_timestep_r", None) is None and audio_scheduler is not None:
                ar = audio_scheduler.timesteps_full[i + 1]
                fwd_kwargs["audio_timestep_r"] = ar.repeat(ats.shape[0]).to(ats)
            out = real_forward(model, *fwd_args, **fwd_kwargs)
            state["i"] = i + 1
            return out

        model.forward = mf_forward
        try:
            return super().__call__(*args, **kwargs)
        finally:
            # Restore the class-bound forward (drop the instance-level override).
            try:
                del model.forward
            except AttributeError:
                model.forward = real_forward.__get__(model, type(model))
