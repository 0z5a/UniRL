# iMF distillation (imfd) trainer.
#
# This module is intentionally self-contained and only adds a new trainer. It does
# not modify existing iMF / CFG-distillation files.
#
# Difference from IMFTrainer:
#   iMF:  v_tgt = ut + (1 - 1 / w) * (v_cond - v_uncond)
#   imfd: v_tgt = teacher(xt, t, cond) + (1 - 1 / w) * (v_cond - v_uncond)
#
# The same operation is applied independently to video and audio branches.

import copy
import functools
from typing import Optional, Type

import torch

from ...engines import find_engine
from ...engines.hy_base_engine import HyBaseEngine
from ...models import build_model
from ...diffusion.flow.imf_loss import (
    compute_cfg_target,
    compute_dudt,
    make_imf_loss_fn,
    perturb_latents,
    time_binary,
    time_unary,
)
from .imf_trainer import IMFTrainer


DEFAULT_IMFD_TEACHER_LOAD = "/apdcephfs_zwfy/share_303793872/zheyuan/Leo2_CFGD_ckpt/ckpt/iter_0000600/weights" 
# DEFAULT_IMFD_TEACHER_LOAD = "/apdcephfs_zwfy/share_303793872/zheyuan/Leo2_SFT_puretorch_ckpt/iter_0004000/weights"


class IMFDTrainer(IMFTrainer):
    """Improved MeanFlow distillation with a frozen text-conditioned teacher target."""

    def __init__(self, args):
        super().__init__(args)
        self.build_teacher_model()

    @staticmethod
    def _main_config(config_obj):
        if isinstance(config_obj, dict):
            return config_obj.get("main_branch")
        return config_obj

    def build_teacher_model(self):
        teacher_args = copy.deepcopy(self.args)
        teacher_args.model_name = getattr(self.args, "teacher_model_name", None) or self.args.model_name
        teacher_args.model_structure = getattr(self.args, "teacher_model_structure", None) or "LeoModel"
        teacher_args.load = getattr(self.args, "teacher_load", None) or DEFAULT_IMFD_TEACHER_LOAD
        teacher_args.add_guidance_token = False
        # The teacher is frozen and inference-only. Keeping it in bf16 avoids
        # paying fp32 parameter memory just because the student uses fp32 master params.
        teacher_args.main_params_fp32 = bool(getattr(self.args, "teacher_main_params_fp32", False))

        self.logger.info(
            f"[imfd] Building teacher model: model_name={teacher_args.model_name}, "
            f"model_structure={teacher_args.model_structure}, load={teacher_args.load}"
        )

        dtype = torch.bfloat16 if teacher_args.bf16 and not teacher_args.main_params_fp32 else torch.float32
        self.teacher_model, self.teacher_model_config = build_model(
            teacher_args,
            dtype=dtype,
            device=teacher_args.init_device,
            initialize_weights=teacher_args.init_device != "meta",
        )
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if teacher_args.reproduce and hasattr(self.teacher_model, "enable_deterministic"):
            self.teacher_model.enable_deterministic()

        teacher_main_config = self._main_config(self.teacher_model_config)
        teacher_use_mot = bool(getattr(teacher_main_config, "use_mot", False))
        self.teacher_model.collect_load_plans(
            None,
            teacher_args.load,
            fuse_experts_in_load=teacher_args.fuse_experts_in_load,
            copy_mot_in_load=teacher_args.copy_mot_in_load and teacher_use_mot,
        )
        self.teacher_model.load_before_fsdp()
        torch.distributed.barrier()

        ParallelEngine: Type[HyBaseEngine] = find_engine(teacher_args.model_name)
        self.teacher_engine: HyBaseEngine = ParallelEngine(
            model=self.teacher_model,
            optimizer_config=None,
            get_lr_scheduler_func=None,
            enable_autocast=teacher_args.autocast_dtype not in ["fp32", "float32"],
            autocast_prec=teacher_args.autocast_dtype,
            gradient_accumulation_steps=1,
            enable_gradient_checkpointing=False,
            initialize_meta_param=teacher_args.fsdp_impl == "new",
            dp_replicate_param_handler="none",
        )

        for plan in self.teacher_model.after_fsdp_plans:
            if plan.source == "dcp":
                default_states = self.teacher_engine.pre_load_state_dict()
                self.teacher_engine.load_checkpoint(**plan.metadata)
                self.teacher_engine.post_load_state_dict(default_states)
            elif plan.source == "resume":
                self.teacher_engine.load_checkpoint(**plan.metadata)
            else:
                self.logger.warning(f"[imfd teacher] unhandled after_fsdp plan source: {plan.source}")

        self.logger.info(
            f"[imfd teacher] load={teacher_args.load} | after_fsdp_plans "
            f"n={len(self.teacher_model.after_fsdp_plans)} "
            f"sources={[p.source for p in self.teacher_model.after_fsdp_plans]}"
        )

        # Engine-level eval returns prediction dictionaries. Module-level train keeps
        # packed text instantiation consistent with the student path.
        self.teacher_engine.eval()
        for teacher_fsdp_model in self.teacher_engine.fsdp_models:
            teacher_fsdp_model.train()
        self.logger.info("[imfd] Teacher model initialized.")

    def _teacher_predict(self, kwargs, has_audio):
        out = self.teacher_engine(**kwargs)
        v_video = out["diffusion_prediction"] if "latents" in kwargs else None
        v_audio = out["audio_diffusion_prediction"] if has_audio else None
        return v_video, v_audio

    def prepare_model_inputs(self, batch, device):
        cfg = self._imf_cfg()
        guidance = cfg["guidance"]
        anchor_weight = cfg["anchor_weight"]
        dt = cfg["fd_dt"]

        # 1) Standard FM inputs from the regular Leo2 data path.
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

        v_t = video_loss_fn.keywords["t"]
        a_t = audio_loss_fn.keywords["t"] if has_audio else None

        # 2) Build unconditioned text once; all other inputs stay shared.
        with torch.autocast(device_type="cuda", enabled=False):
            uncond_text_states, uncond_text_mask = self._build_uncond_text(batch, device)

        # 3) CP-consistent r ~ U(0, t).
        v_r = self._sample_r(v_t)
        a_r = self._sample_r(a_t) if has_audio else None

        is_mf = getattr(self.args, "model_structure", "") == "LeoModelMeanFlow"
        if is_mf:
            v_r_mt = time_unary(v_r, self.video_denoiser.get_model_t)
            a_r_mt = time_unary(a_r, self.audio_denoiser.get_model_t) if has_audio else None

        # 4) Finite-difference window in flow time.
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

        # 5) No-grad student/teacher forwards for detached targets.
        self.model_engine.eval()
        for m in self.model_engine.fsdp_models:
            m.train()
        self.teacher_engine.eval()
        for m in self.teacher_engine.fsdp_models:
            m.train()

        try:
            with torch.no_grad():
                # Student CFG delta at (xt, t, t).
                v_cond_v, v_cond_a = self._student_predict(model_input_kwargs, has_audio)

                uncond_kwargs = dict(model_input_kwargs)
                uncond_kwargs["cond_text_states"] = uncond_text_states
                uncond_kwargs["cond_text_mask"] = uncond_text_mask
                v_unc_v, v_unc_a = self._student_predict(uncond_kwargs, has_audio)
                del uncond_kwargs, uncond_text_states, uncond_text_mask

                # Teacher text-conditioned velocity replaces the FM target ut.
                teacher_kwargs = dict(model_input_kwargs)
                teacher_kwargs.pop("timestep_r", None)
                teacher_kwargs.pop("audio_timestep_r", None)
                teacher_ut_v, teacher_ut_a = self._teacher_predict(teacher_kwargs, has_audio)
                del teacher_kwargs

                v_tgt_v = compute_cfg_target(teacher_ut_v, v_cond_v, v_unc_v, guidance)
                del teacher_ut_v, v_unc_v
                if has_audio:
                    v_tgt_a = compute_cfg_target(teacher_ut_a, v_cond_a, v_unc_a, guidance)
                    del teacher_ut_a, v_unc_a

                # Student finite difference of u(xt, r, t) at fixed r.
                plus_kwargs = dict(model_input_kwargs)
                plus_kwargs["latents"] = perturb_latents(
                    model_input_kwargs["latents"], v_cond_v, v_step_plus
                )
                plus_kwargs["timesteps"] = v_ts_plus
                if has_audio:
                    plus_kwargs["audio_latents"] = perturb_latents(
                        model_input_kwargs["audio_latents"], v_cond_a, a_step_plus
                    )
                    plus_kwargs["audio_timesteps"] = a_ts_plus
                if is_mf:
                    plus_kwargs["timestep_r"] = v_r_mt
                    if has_audio:
                        plus_kwargs["audio_timestep_r"] = a_r_mt
                v_plus_v, v_plus_a = self._student_predict(plus_kwargs, has_audio)
                del plus_kwargs

                minus_kwargs = dict(model_input_kwargs)
                minus_kwargs["latents"] = perturb_latents(
                    model_input_kwargs["latents"], v_cond_v, v_step_minus
                )
                minus_kwargs["timesteps"] = v_ts_minus
                if has_audio:
                    minus_kwargs["audio_latents"] = perturb_latents(
                        model_input_kwargs["audio_latents"], v_cond_a, a_step_minus
                    )
                    minus_kwargs["audio_timesteps"] = a_ts_minus
                if is_mf:
                    minus_kwargs["timestep_r"] = v_r_mt
                    if has_audio:
                        minus_kwargs["audio_timestep_r"] = a_r_mt
                v_minus_v, v_minus_a = self._student_predict(minus_kwargs, has_audio)
                del minus_kwargs, v_cond_v

                dudt_v = compute_dudt(v_plus_v, v_minus_v, v_span)
                del v_plus_v, v_minus_v
                if has_audio:
                    del v_cond_a
                    dudt_a = compute_dudt(v_plus_a, v_minus_a, a_span)
                    del v_plus_a, v_minus_a
        finally:
            self.model_engine.train()
            self.teacher_engine.eval()
            for m in self.teacher_engine.fsdp_models:
                m.train()

        # 6) Inject imfd branch losses into the existing Leo loss hooks.
        time_diff_v = time_binary(v_t, v_r, lambda a, b: a.float() - b.float())
        model_input_kwargs["diffusion_loss_fn"] = make_imf_loss_fn(
            time_diff_v, v_tgt_v, dudt_v, anchor_weight
        )
        if has_audio:
            time_diff_a = time_binary(a_t, a_r, lambda a, b: a.float() - b.float())
            model_input_kwargs["audio_diffusion_loss_fn"] = make_imf_loss_fn(
                time_diff_a, v_tgt_a, dudt_a, anchor_weight
            )

        if is_mf:
            model_input_kwargs["timestep_r"] = v_r_mt
            if has_audio:
                model_input_kwargs["audio_timestep_r"] = a_r_mt

        return model_input_kwargs, bsz, seqlen
