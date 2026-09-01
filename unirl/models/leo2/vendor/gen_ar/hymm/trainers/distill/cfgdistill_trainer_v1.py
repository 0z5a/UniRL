# This trainer is based on pure torch.
# This trainer uses the same interface as PTMv2 trainer for ease of integration.
#
import addict
import os
import time
from argparse import Namespace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Type, Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.optim import AdamW
from hy_parallelism.parallel_states import init_parallel_state
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from ...core.data_provider_dit import (
    encode_text,
    prepare_model_inputs,
)
from ...core.global_vars import (
    set_logger,
)
from ...core.parallel_states import ParallelState
from ...core.extra_model_provider import (
    build_text_encoder,
    build_audio_vae,
    build_denoiser,
    build_repa_encoder,
)
from ...engines import find_engine
from ...engines.hy_base_engine import HyBaseEngine
from ...models import build_model
from ...models.multimodal.hunyuan_multimodal_state import ParameterSummary
from ...utils.file_utils import empty_logger, dump_configs, dump_codes
from ...utils.helpers import print_args
from ...utils.torch_utils import set_manual_seed, set_reproducibility
from ...utils.optimzer_utils import build_optimizer_factory_from_model_args, pre_optimizer_hook
from ..pretrain_pure_torch import MultimodalTrainer as PretrainMultimodalTrainer


class MultimodalTrainer(PretrainMultimodalTrainer):
    # vae 内部维护了 generator, 为了正确 resume，需要记录 vae generator 状态
    # 这里 tuple 含义为 [trainer 属性名, 保存到 client_state 的 key 名]
    # 会从这些属性中调用 generator.get_state() 和 generator.set_state() 方法
    _GENERATOR_STATE_BINDINGS = (
        ("vae", "vae_generator_state"),
        ("audio_vae", "audio_vae_generator_state"),
    )

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.build_teacher_model()

    def build_extra_model(self):
        super().build_extra_model()
        args = self.args

        assert args.use_text_encoder, "`use_text_encoder` must be True for Leo2Trainer."
        self.text_encoder = build_text_encoder()

        # Modality-specific SNR config; fall back to global args when not set (None).
        video_snr_type = args.flow_snr_type_video or args.flow_snr_type
        video_snr_mix_ratio = (
            args.flow_snr_mix_uniform_ratio_video
            if args.flow_snr_mix_uniform_ratio_video is not None
            else args.flow_snr_mix_uniform_ratio
        )
        self.video_denoiser = build_denoiser(
            denoiser_type="video",
            shift=args.flow_shift_video,
            snr_type=video_snr_type,
            snr_mix_uniform_ratio=video_snr_mix_ratio,
        )
        if args.audio_branch_model_name is not None:
            audio_snr_type = args.flow_snr_type_audio or args.flow_snr_type
            audio_snr_mix_ratio = (
                args.flow_snr_mix_uniform_ratio_audio
                if args.flow_snr_mix_uniform_ratio_audio is not None
                else args.flow_snr_mix_uniform_ratio
            )
            self.audio_denoiser = build_denoiser(
                denoiser_type="audio",
                shift=args.flow_shift_audio,
                snr_type=audio_snr_type,
                snr_mix_uniform_ratio=audio_snr_mix_ratio,
            )

        if args.use_audio_vae:
            self.audio_vae = build_audio_vae()

        if args.use_repa:
            self.repa_encoder = build_repa_encoder()

    def _collect_extra_client_state(self) -> dict:
        state = super()._collect_extra_client_state()
        for attr_name, key in self._GENERATOR_STATE_BINDINGS:
            obj = getattr(self, attr_name, None)
            if obj is not None and getattr(obj, "generator", None) is not None:
                state[key] = obj.generator.get_state().clone().cpu()
        return state

    def _consume_extra_client_state(self, client_state: dict) -> None:
        super()._consume_extra_client_state(client_state)
        for attr_name, key in self._GENERATOR_STATE_BINDINGS:
            state = client_state.get(key)
            if state is None:
                continue
            obj = getattr(self, attr_name, None)
            if obj is not None and getattr(obj, "generator", None) is not None:
                obj.generator.set_state(state)
                self.logger.info(f"Restored {attr_name}.generator state from checkpoint.")

    def after_initialize(self):
        super().after_initialize()

        args = self.args

        if args.audio_branch_model_name is not None:
            self.loss_names.extend([
                f"{dataset_tag}_audio_loss"
                for dataset_tag in self.mm_state.all_dataset_keys
                if "2a" in dataset_tag or "2va" in dataset_tag
            ])

        if args.moe_aux_loss and args.moe_aux_loss_coeff > 0:
            config = self.model.get_config()
            txt_config = self.model.get_txt_config()

            if config.num_experts > 0:
                self.loss_names.extend(["image_video_moe_loss", "image_video_capacity_rate"])
            if txt_config.num_experts > 0:
                self.loss_names.extend(["text_moe_loss", "text_capacity_rate"])

        if self.args.use_repa:
            self.loss_names.append("repa_loss")

        if getattr(args, "use_global_diffusion_loss_average", False):
            self.loss_names.extend(["global_image_video_loss", "global_audio_loss"])
            if self.args.use_repa:
                self.loss_names.append("global_repa_loss")

    def build_teacher_model(self):
        import copy
        teacher_args = copy.deepcopy(self.args)
        teacher_args.model_name = getattr(self.args, "teacher_model_name", None) or self.args.model_name
        teacher_args.load = getattr(self.args, "teacher_load", None) or self.args.load
        teacher_args.add_guidance_token = False
        self.logger.info(
            f"Building teacher model: model_name={teacher_args.model_name}, load={teacher_args.load}"
        )

        # Build model
        dtype = torch.bfloat16 if teacher_args.bf16 and not teacher_args.main_params_fp32 else torch.float32
        self.teacher_model, self.teacher_model_config = build_model(
            teacher_args,
            dtype=dtype,
            device=teacher_args.init_device,
            initialize_weights=teacher_args.init_device != "meta",
        )
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # Model Reproducibility.
        if teacher_args.reproduce and hasattr(self.teacher_model, "enable_deterministic"):
            self.teacher_model.enable_deterministic()
        # Load weights from pretrained checkpoint if specified. We first collect all possible load plans
        # into three groups: before_fsdp_plans, fsdp_plans, after_fsdp_plans.
        # - before_fsdp_plans: (init_device != 'meta') plans that weights loading can be immediately executed.
        #   For example, pretrained submodules, hugging face checkpoints. Loaded by self.model.load_before_fsdp()
        # - fsdp_plans: (init_device == 'meta') plans that weights loading need to be deferred but supported
        #   for fully loading or stream loading. For example, pretrained submodules, hugging face checkpoints.
        #   Loaded by load_and_apply_fsdp2() when applying fsdp.
        # - after_fsdp_plans: plans that weights loading need to be deferred until after fully_shard.
        #   For example, dcp checkpoints, resuming from existing training checkpoints.
        #   Loaded by self.load_after_fsdp()
        self.teacher_model.collect_load_plans(
            None, teacher_args.load,
            fuse_experts_in_load=teacher_args.fuse_experts_in_load,
            copy_mot_in_load=teacher_args.copy_mot_in_load and self.teacher_model_config.use_mot,
        )
        self.teacher_model.load_before_fsdp()
        torch.distributed.barrier()

        # Build Model Engine
        ParallelEngine: Type[HyBaseEngine] = find_engine(teacher_args.model_name)     # noqa
        self.teacher_engine: HyBaseEngine = ParallelEngine(
            model=self.teacher_model,
            optimizer_config=None,
            get_lr_scheduler_func=None,
            enable_autocast=teacher_args.autocast_dtype not in ["fp32", "float32"],
            autocast_prec=teacher_args.autocast_dtype,
            gradient_accumulation_steps=1,
            enable_gradient_checkpointing=True,
            # Put meta param materialization into init_param_and_apply_fsdp2 for memory-efficient initialization
            # For streaming fsdp implementation, we set initialize_meta_param to false, since meta params will be materialized in apply_fsdp.
            initialize_meta_param=teacher_args.fsdp_impl == 'new',
            dp_replicate_param_handler='none',
            # cpu_offload=self.args.teacher_cpu_offload
        )

        # ===== 让 teacher 和 student 加载同一份 ckpt（同路径 + 同加载动作）=====
        # teacher_args.load 已等于 args.load（未设 teacher_load），路径与 student 一致。
        # 但本 run 的 ckpt 是 DCP（ckpt.torch/iter_xxx）、init_device=meta，其加载计划落在
        # after_fsdp_plans，必须在 FSDP engine 建好后执行——这正是 student 的 load_after_fsdp 所做的。
        # 此前 build_teacher_model 缺这一步，导致 teacher 权重停在 meta 初始化（≈0），pred 恒为 0。
        # 这里与 pretrain_pure_torch.load_after_fsdp 的 dcp 分支保持完全一致（teacher 不需要 resume 的优化器/标量状态）。
        for plan in self.teacher_model.after_fsdp_plans:
            if plan.source == "dcp":
                default_states = self.teacher_engine.pre_load_state_dict()
                self.teacher_engine.load_checkpoint(**plan.metadata)
                self.teacher_engine.post_load_state_dict(default_states)
            elif plan.source == "resume":
                self.teacher_engine.load_checkpoint(**plan.metadata)
            else:
                self.logger.warning(
                    f"[teacher] unhandled after_fsdp plan source: {plan.source}"
                )
        self.logger.info(
            f"[teacher] load={teacher_args.load} | after_fsdp_plans "
            f"n={len(self.teacher_model.after_fsdp_plans)} "
            f"sources={[p.source for p in self.teacher_model.after_fsdp_plans]}"
        )

        self.teacher_engine.eval()
        for teacher_fsdp_model in self.teacher_engine.fsdp_models:
            teacher_fsdp_model.train()
        # leo.py 的 instantiate_text_tokens_full_seqlen（1561-1600）依据 self.training 走两条不同路径：
        # 只有当 self.training=True 时，text_states 才会按 cond_text_mask 压平成与 text_mask 一致的形状，断言才成立
        # 那为什么不能直接 engine.train()？
        # 差别全在 engine.training 这个引擎级标志上，它影响引擎 forward/__call__ 的返回行为（之前分析过）：
        # loss closure 触发：get_real_ret / _cache_ret_and_loss 在 engine.training=False 时直接返回模型输出 dict（供 ["diffusion_prediction"] 取用）；若 engine.training=True，引擎会走训练分支去触发 loss closure。teacher 根本没注册 loss closure，且整段在 @torch.no_grad() 下，走训练分支语义不对、甚至可能报错。
        # PP schedule：若启用 PP（本配置没启用），eval() 用 inference schedule（无 backward），train() 用 training schedule。teacher 只做前向推理，应该用 inference schedule。
        self.logger.info(f"Teacher model initialized.")

    @staticmethod
    def _calc_cfg_target(pred_cond, pred_uncond, guidance_scale):
        if isinstance(pred_cond, torch.Tensor):
            assert isinstance(pred_uncond, torch.Tensor), (
                f"pred_cond is Tensor but pred_uncond is {type(pred_uncond)}"
            )
            assert pred_cond.shape == pred_uncond.shape, (
                f"pred_cond and pred_uncond shape mismatch: {pred_cond.shape} vs {pred_uncond.shape}"
            )
            target = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # Store CFG target in bf16 (was fp32) to halve the resident memory of this tensor.
            # It is held by `loss_closure` through the whole student fwd+bwd, so its dtype directly
            # adds to peak memory. In `_cfgdistill_mse`, `pred.to(fp32) - target` type-promotes back
            # to fp32, so loss precision is essentially unaffected.
            return target.to(dtype=torch.bfloat16).detach().clone() # * 2 + 1

        assert isinstance(pred_cond, list) and isinstance(pred_uncond, list), (
            f"Unsupported CFG target types: {type(pred_cond)} vs {type(pred_uncond)}"
        )
        assert len(pred_cond) == len(pred_uncond), (
            f"pred_cond and pred_uncond list length mismatch: {len(pred_cond)} vs {len(pred_uncond)}"
        )
        return [
            MultimodalTrainer._calc_cfg_target(cond_i, uncond_i, guidance_scale)
            for cond_i, uncond_i in zip(pred_cond, pred_uncond)
        ]

    @staticmethod
    def _cfgdistill_mse(pred, target):
        if isinstance(pred, torch.Tensor):
            assert isinstance(target, torch.Tensor), (
                f"pred is Tensor but target is {type(target)}"
            )
            assert pred.shape == target.shape, (
                f"pred and target shape mismatch: {pred.shape} vs {target.shape}"
            )
            return torch.mean((pred.to(dtype=torch.float32) - target) ** 2)

        assert isinstance(pred, list) and isinstance(target, list), (
            f"Unsupported loss types: {type(pred)} vs {type(target)}"
        )
        assert len(pred) == len(target), (
            f"pred and target list length mismatch: {len(pred)} vs {len(target)}"
        )
        losses = [
            MultimodalTrainer._cfgdistill_mse(pred_i, target_i)
            for pred_i, target_i in zip(pred, target)
        ]
        return torch.stack(losses).mean()

    def prepare_model_inputs(self, batch, device):
        with torch.autocast(device_type="cuda", enabled=False):
            model_input_kwargs, bsz, seqlen = prepare_model_inputs(
                batch, device, model_config=self.model.get_config(),
            )
        return model_input_kwargs, model_input_kwargs, bsz, seqlen
        return model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen
    
    @torch.no_grad()
    def prepare_cfgdistill_target(self, model_input_kwargs, uncond_model_input_kwargs, device):
        guidance_scale_min = getattr(self.args, "guidance_scale_min", 6.0)
        guidance_scale_max = getattr(self.args, "guidance_scale_max", 6.0)
        sampled_guidance_scale = torch.empty((), device=device).uniform_(
            guidance_scale_min, guidance_scale_max
        )

        teacher_model_input_kwargs = model_input_kwargs.copy()
        for k in ("guidance_index", "guidance"):
            teacher_model_input_kwargs.pop(k, None)
        teacher_model_uncond_kwargs = dict(teacher_model_input_kwargs)
        teacher_model_uncond_kwargs.update(uncond_model_input_kwargs)

        pred_cond = self.teacher_engine(**teacher_model_input_kwargs)["diffusion_prediction"]
        pred_uncond = self.teacher_engine(**teacher_model_uncond_kwargs)["diffusion_prediction"]
        cfgdistill_target = self._calc_cfg_target(pred_cond, pred_uncond, sampled_guidance_scale)
        del pred_cond, pred_uncond
        torch.cuda.empty_cache()
        return cfgdistill_target, sampled_guidance_scale

    @torch.no_grad()
    def validation(self, timer):
        from hymm.samplers.leo2_sampler import Leo2Sampler

        timer.start("validation", sync=True, barrier=True)
        sampler = Leo2Sampler(None, self.rank, self.world_size, trainer=self)
        sampler.run_testsets(
            sample_save_base=self.exp_dir / f"samples/iter_{self.ss.update_steps:07d}",
        )
        timer.stop("validation")

    def train_step(self, batch):
        args = self.args
        device = torch.device("cuda", args.local_rank)

        model_input_kwargs, uncond_model_input_kwargs, bsz, seqlen = self.prepare_model_inputs(batch, device)

        cfgdistill_target, sampled_guidance_scale = self.prepare_cfgdistill_target(
            model_input_kwargs, uncond_model_input_kwargs, device
        )


        def loss_closure(model_output, input_args_kwargs_dict):     # noqa
            loss_dict = {}
            pred = model_output["diffusion_prediction"]
            cfgdistill_loss = self._cfgdistill_mse(pred, cfgdistill_target)
            loss_dict['loss'] = cfgdistill_loss

            return cfgdistill_loss, loss_dict

        self.model_engine.register_loss_closure(loss_closure)

        loss = self.model_engine(**model_input_kwargs)

        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(f"NaN loss encountered in rank {self.rank}, total NaN count: {self.nan_grad_count}")

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict['loss'] = loss

        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen
            }
        }

        return loss_dict, consumed_metrics
