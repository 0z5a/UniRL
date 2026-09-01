# improved-MeanFlow (iMF) distillation trainer.
#
# This is a NEW, self-contained trainer. It does NOT modify any existing file.
# Launch it by pointing `--trainer` at `distill.imf_trainer.IMFTrainer`
# (see `jobs/examples/run_pure_torch_leo_imf.sh`).
#
# ============================ Design (方案 A) ============================
# The whole iMF objective is implemented by overriding ONLY `prepare_model_inputs`.
# We reuse the model's *existing* per-branch loss hook: `LeoModel.forward` calls
#     raw_visual_loss = diffusion_loss_fn(model_output=diff_pred)["loss"]
#     raw_audio_loss  = audio_diffusion_loss_fn(model_output=audio_diff_pred)["loss"]
# and then applies `visual_loss_weight` / `audio_loss_weight` and the global-average
# machinery. By *replacing* those two loss-fn hooks with iMF closures we get the iMF
# loss for free with the existing dual-branch weighting / averaging / metrics, and the
# inherited `train_step` (which reads `model_output.losses`) runs the single gradient
# forward unchanged.  ZERO changes to model / engine / base train_step.
#
# Audio and video are independent branches with their own weights (config:
# image-loss-weight / audio-loss-weight); we build an iMF target *per branch* and
# inject a *per-branch* loss fn. There is no t=r ratio / bernoulli mask: the
# flow-matching anchor and the meanflow-consistency term are combined inside each
# per-branch loss via `anchor_weight` (reference: loss + 10*loss_2).
#
# ============================ iMF target construction ============================
# Online self-distillation from the *student's own* detached predictions (NO teacher):
#   1. engine.eval() + module.train()  (the recipe required by leo.py's packed text path;
#      engine.training=False makes the engine return the prediction dict, while
#      module.training=True keeps the packed cond_text flattening correct -- exactly the
#      pattern proven in cfgdistill_trainer_v1.build_teacher_model / data_carrier_trainer).
#   2. v_cond   = student(cond)          # CFG positive
#      v_uncond = student(uncond)        # CFG negative (same <cfg> null-token logic as
#                                          cfgdistill_trainer_v2, shared by both branches)
#   3. du/dt via central finite difference: perturb the noised-latent channels along
#      v_cond and the flow-time along ±dt, then du/dt = (v_plus - v_minus) / span.
#   4. v_tgt = u_t + (1 - 1/w) * (v_cond - v_uncond),  w = 6 for both branches.
#   5. engine.train()  (restore for the gradient forward).
# Per-branch loss (under the no-timestep_r backbone, pred(t,r) == pred(t,t)):
#   L = mean((pred - v_tgt + (t - r) * du/dt)^2) + anchor_weight * mean((pred - v_tgt)^2)
#
# ============================ Ragged fields ============================
# In packed training every field (t, r, u_t, latents, prediction, model_t) is a *ragged*
# structure: a list[B] whose per-batch-item entry is a tensor [N_i, ...] (sub-samples
# stacked) or a list of such tensors. Time fields (t/r/model_t) carry 1-D leaves [N_i]
# (one flow-time per sub-sample). All iMF math is done through the ragged-aware helpers
# in `imf_loss.py`, which broadcast each per-sub-sample time onto the matching data leaf.
#
# ============================ t / r consistency ============================
# t is sampled inside the data pipeline via `vae.generator` (seeded with seed + dp_rank,
# so identical across the CP ranks that split one sample). We sample r from the *same*
# generator immediately afterwards (keeping it CP-consistent in the common case and
# tied to the resumable RNG stream) and additionally broadcast every r leaf within the
# CP group as belt-and-suspenders, so r is identical inside every context-parallel group
# while differing across data-parallel groups. Audio and video are independent branches,
# so each samples its own r in its own ragged structure (both CP-consistent).

import functools

import torch
import torch.distributed as dist

from ..pretrain_pure_torch_dit import Leo2Trainer
from ...core.data_provider_dit import encode_text
from ...core.global_vars import get_tkwrapper
from ...diffusion.flow.imf_loss import (
    broadcast_time_,
    compute_cfg_target,
    compute_dudt,
    make_imf_loss_fn,
    perturb_latents,
    rand_like_scaled,
    time_binary,
    time_unary,
)
from ...models.diffusion import leo_meanflow as _leo_meanflow  # noqa: F401


class IMFTrainer(Leo2Trainer):
    """iMF (improved MeanFlow) self-distillation trainer for Leo-2 (t2va)."""

    # ------------------------------------------------------------------ #
    # iMF hyper-parameters (baked as defaults; overridable via args).
    # ------------------------------------------------------------------ #
    def _imf_cfg(self):
        return dict(
            guidance=float(getattr(self.args, "imf_guidance", 6.0)),
            anchor_weight=float(getattr(self.args, "imf_anchor_weight", 1.0)),
            fd_dt=float(getattr(self.args, "imf_fd_dt", 1e-2)),
        )

    # ------------------------------------------------------------------ #
    # uncond (<cfg>) token construction -- identical logic to
    # cfgdistill_trainer_v2._build_uncond_tokens, shared by both branches.
    # ------------------------------------------------------------------ #
    def _build_uncond_tokens(self, tokens, text_mask, device):
        tkwrapper = getattr(self, "tkwrapper", None) or get_tkwrapper()
        uncond_token_id = getattr(tkwrapper, "cfg_token_id", None)
        if uncond_token_id is None:
            raise ValueError("Cannot build uncond tokens: cfg_token_id is unavailable.")

        tokens = tokens.to(device)
        text_mask_bool = text_mask.to(device).to(torch.bool)

        # Defensive length alignment (tokens and text_mask are expected equal length).
        if text_mask_bool.shape[1] < tokens.shape[1]:
            pad = torch.zeros(
                (tokens.shape[0], tokens.shape[1] - text_mask_bool.shape[1]),
                dtype=torch.bool, device=device,
            )
            text_mask_bool = torch.cat([text_mask_bool, pad], dim=1)
        elif text_mask_bool.shape[1] > tokens.shape[1]:
            text_mask_bool = text_mask_bool[:, :tokens.shape[1]]

        # Replace only real prompt content; keep structural special tokens intact.
        special_token_ids = [
            v for v in getattr(tkwrapper, "special_token_map", {}).values()
            if isinstance(v, int)
        ]
        if special_token_ids:
            special_ids = torch.tensor(special_token_ids, device=device, dtype=tokens.dtype)
            special_mask = torch.isin(tokens, special_ids)
            replace_mask = text_mask_bool & (~special_mask)
        else:
            replace_mask = text_mask_bool

        uncond_tokens = tokens.clone()
        uncond_tokens[replace_mask] = int(uncond_token_id)
        return uncond_tokens

    def _build_uncond_text(self, batch, device):
        uncond_tokens = self._build_uncond_tokens(
            batch["tokens"], batch["text_mask"], device,
        )
        uncond_batch = dict(batch)
        uncond_batch["tokens"] = uncond_tokens
        dataset_tag = batch["dataset_tag"][0]
        task_kwargs = getattr(self.args, f"{dataset_tag}_task_kwargs")
        uncond_text_states, uncond_text_mask = encode_text(
            self.text_encoder, uncond_batch, task_kwargs, device,
        )
        return uncond_text_states, uncond_text_mask

    # ------------------------------------------------------------------ #
    # CP-consistent r sampling (ragged: t is a list[B] of 1-D time tensors).
    # ------------------------------------------------------------------ #
    def _sample_r(self, t):
        # r ~ U(0, t)  (reference: r = rand_like(t) * t), drawn from the VAE generator
        # so it stays tied to the resumable RNG stream and CP-consistent in the common
        # (seed + dp_rank) seeding case.
        r = rand_like_scaled(t, generator=self.vae.generator)
        # Belt-and-suspenders: force r identical inside every context-parallel group.
        cp_group = getattr(self.p_state, "cp_group", None)
        cp_size = getattr(self.p_state, "cp_size", 1)
        if cp_group is not None and cp_size > 1:
            src = dist.get_global_rank(cp_group, 0)
            broadcast_time_(r, src, cp_group)
        return r

    # ------------------------------------------------------------------ #
    # Single student forward (eval recipe) -> per-branch velocity predictions.
    # ------------------------------------------------------------------ #
    def _student_predict(self, kwargs, has_audio):
        out = self.model_engine(**kwargs)
        v_video = out["diffusion_prediction"] if "latents" in kwargs else None
        v_audio = out["audio_diffusion_prediction"] if has_audio else None
        return v_video, v_audio

    # ------------------------------------------------------------------ #
    # The whole iMF machinery lives here; train_step is inherited unchanged.
    # ------------------------------------------------------------------ #
    def prepare_model_inputs(self, batch, device):
        cfg = self._imf_cfg()
        guidance = cfg["guidance"]
        anchor_weight = cfg["anchor_weight"]
        dt = cfg["fd_dt"]

        # 1) Standard inputs (VAE encode + noise + FM loss-fn partials).
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = super().prepare_model_inputs(batch, device)

        # 2) Extract per-branch flow-time t and FM velocity target u_t from the
        #    partials the data pipeline built (partial(training_losses_fn, t=, ut=, ...)).
        video_loss_fn = model_input_kwargs.get("diffusion_loss_fn", None)
        audio_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn", None)

        has_video = ("latents" in model_input_kwargs) and isinstance(
            video_loss_fn, functools.partial
        )
        has_audio = (
            ("audio_latents" in model_input_kwargs)
            and isinstance(audio_loss_fn, functools.partial)
            and audio_loss_fn.keywords.get("ut", None) is not None
        )

        # iMF requires the video branch (the production t2va config always has it). If for
        # some reason it is absent, fall back to the standard FM inputs untouched.
        if not has_video:
            return model_input_kwargs, bsz, seqlen

        # All time/data fields below are *ragged*: list[B] whose per-item entry is a
        # tensor [N_i, ...] (or a list of tensors); time fields carry 1-D leaves [N_i].
        v_t = video_loss_fn.keywords["t"]            # ragged flow time in [0, 1]
        v_ut = video_loss_fn.keywords["ut"]          # ragged FM velocity target
        a_t = audio_loss_fn.keywords["t"] if has_audio else None
        a_ut = audio_loss_fn.keywords["ut"] if has_audio else None

        # 3) uncond (<cfg>) text -- shared by both branches.
        with torch.autocast(device_type="cuda", enabled=False):
            uncond_text_states, uncond_text_mask = self._build_uncond_text(batch, device)

        # 4) CP-consistent r ~ U(0, t). Audio and video are independent branches, so each
        #    samples r in its OWN ragged structure (both CP-consistent).
        v_r = self._sample_r(v_t)
        a_r = self._sample_r(a_t) if has_audio else None

        # For the REAL two-time backbone we need r's model-time (== get_model_t(r)) both for
        # the JVP forwards (du/dt of u(t, r) at FIXED r) and the gradient forward (pred=u(t,r)).
        is_mf = getattr(self.args, "model_structure", "") == "LeoModelMeanFlow"
        if is_mf:
            v_r_mt = time_unary(v_r, self.video_denoiser.get_model_t)
            a_r_mt = time_unary(a_r, self.audio_denoiser.get_model_t) if has_audio else None

        # 5) Finite-difference window in flow time (clamped to [0, 1]); ragged per-leaf.
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

        # 6) eval recipe (engine.eval() returns prediction dict; module.train() keeps the
        #    packed cond_text flattening path correct).
        self.model_engine.eval()
        for m in self.model_engine.fsdp_models:
            m.train()

        try:
            with torch.no_grad():
                # cond forward
                v_cond_v, v_cond_a = self._student_predict(model_input_kwargs, has_audio)

                # uncond forward (only the text condition changes)
                uncond_kwargs = dict(model_input_kwargs)
                uncond_kwargs["cond_text_states"] = uncond_text_states
                uncond_kwargs["cond_text_mask"] = uncond_text_mask
                v_unc_v, v_unc_a = self._student_predict(uncond_kwargs, has_audio)

                distill_audio = has_audio and v_cond_a is not None and v_unc_a is not None
                if has_audio and not distill_audio and getattr(self, "rank", 0) == 0:
                    self.logger.warning(
                        "[iMF] audio_diffusion_prediction is unavailable; "
                        "skip audio iMF distillation and keep the original audio FM loss."
                    )

                # plus / minus forwards: perturb noised latents along v_cond and flow-time.
                # For the meanflow backbone these compute u(t +/- dt, r) at FIXED r, so the
                # finite difference is du/dt of the meanflow average velocity (not of v).
                plus_kwargs = dict(model_input_kwargs)
                plus_kwargs["latents"] = perturb_latents(
                    model_input_kwargs["latents"], v_cond_v, v_step_plus
                )
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
                minus_kwargs["latents"] = perturb_latents(
                    model_input_kwargs["latents"], v_cond_v, v_step_minus
                )
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

                # 7) per-branch v_tgt and du/dt
                v_tgt_v = compute_cfg_target(v_ut, v_cond_v, v_unc_v, guidance)
                dudt_v = compute_dudt(v_plus_v, v_minus_v, v_span)
                if distill_audio:
                    v_tgt_a = compute_cfg_target(a_ut, v_cond_a, v_unc_a, guidance)
                    dudt_a = compute_dudt(v_plus_a, v_minus_a, a_span)
        finally:
            # 8) restore train mode for the gradient forward.
            self.model_engine.train()

        # 9) Inject per-branch iMF loss fns; the model's existing weighting / global-average
        #    machinery and the inherited train_step do the rest.
        time_diff_v = time_binary(v_t, v_r, lambda a, b: a.float() - b.float())
        model_input_kwargs["diffusion_loss_fn"] = make_imf_loss_fn(
            time_diff_v, v_tgt_v, dudt_v, anchor_weight
        )
        if distill_audio:
            time_diff_a = time_binary(a_t, a_r, lambda a, b: a.float() - b.float())
            model_input_kwargs["audio_diffusion_loss_fn"] = make_imf_loss_fn(
                time_diff_a, v_tgt_a, dudt_a, anchor_weight
            )

        # 10) For the REAL two-time backbone (LeoModelMeanFlow), feed r into the *gradient*
        #     forward so its last 1/4 layers + head do AdaLN with r. The model then returns
        #     BOTH pred=u(t, r) (for the consistency term) and an internally-forked
        #     pred_anchor=u(t, t) (for the anchor term). The cond/uncond eval forwards above
        #     ran without r (r == t -> instantaneous velocity v(z, t)), which is what v_tgt
        #     needs. Other structures don't accept timestep_r, so we only inject it here.
        if is_mf:
            model_input_kwargs["timestep_r"] = v_r_mt
            if distill_audio:
                model_input_kwargs["audio_timestep_r"] = a_r_mt

        return model_input_kwargs, bsz, seqlen
