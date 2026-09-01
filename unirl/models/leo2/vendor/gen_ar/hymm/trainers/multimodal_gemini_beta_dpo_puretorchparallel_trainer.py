import gc
import os
import time
import json
import random
from collections import OrderedDict
from typing import Dict, Union, List, Optional

import loguru
import torch
import torch.distributed as dist
from index_kits.sampler import DistributedSampler
from torch.utils.data import DataLoader
try:
    from torch.nn.attention.flex_attention import create_block_mask
except:
    pass

from .helpers import (
    ScalarStates,
    CycleStates,
    save_checkpoint,
    get_trainable_params,
    WarmupCosineHelper
)
from ..models.autoencoders import VAEEncodeOutput
from ..models import build_model, EMA, DistributedEMA
from ..models.autoregressive.flex_attn_layers import (
    create_text_image_mask_mod,
    create_batch_text_image_mask_mod,
)
from ..samplers.gemini_beta_sampler import GeminiBetaSampler
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.rl_t2i_loader import RLTextImageArrowStream
from .multimodal_gemini_alpha_trainer import GeminiTrainerAlphaMultiModal
from ..utils.torch_utils import (
    to_device,
    set_manual_seed,
    set_worker_seed_builder,
    profiler_context,
    is_torch_tensor,
    PRECISION_TO_TYPE,
)
from ..ds_config import get_deepspeed_config
import torchvision.transforms as transforms
from ..models.autoencoders import load_vae
from ..models.tokenizers import TokenizerWrapper
from ..diffusion import load_denoiser
from hymm.parallelism.parallel_states import init_parallel_state

gc.set_threshold(7000, 100, 100)


def flow_dpo_loss_fn(v_w_pred, v_l_pred, v_w_ref_pred, v_l_ref_pred, v_w_target, v_l_target, lose_exist, use_sft_loss=None, beta=500, pos_term_weight=0.0, neg_term_weight=0.0, hinge_threshold=0, only_kl_loss=False):
    reduce_dims = list(range(1, v_w_pred.ndim))

    model_w_err = (v_w_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    model_l_err = (v_l_pred - v_l_target).pow(2).mean(dim=reduce_dims)
    ref_w_err = (v_w_ref_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    ref_l_err = (v_l_ref_pred - v_l_target).pow(2).mean(dim=reduce_dims)

    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err

    # FIXME: Now lose_exist only support batch size == 1
    lose_exist = lose_exist[0]
    if not lose_exist:
        inside_term = -0.5 * beta * w_diff
        l_diff = torch.tensor(0.0, device=w_diff.device)
        pos_term = torch.tensor(0.0, device=w_diff.device)
        neg_term = torch.tensor(0.0, device=w_diff.device)
    else:
        if pos_term_weight > 0 and neg_term_weight > 0:
            pos_term = pos_term_weight * torch.clamp(ref_w_err - model_w_err, max=hinge_threshold)  # 希望 ref_w_err 越大于 model_w_err 越好
            neg_term = neg_term_weight * torch.clamp(ref_l_err - model_l_err, min=0)  # 希望 ref_l_err 越小于 model_w_err 越好
            inside_term = -0.5 * beta * (w_diff - l_diff - pos_term + neg_term)
        elif pos_term_weight > 0:
            pos_term = pos_term_weight * torch.clamp(ref_w_err - model_w_err, max=hinge_threshold)  # 希望 ref_w_err 越大于 model_w_err 越好
            inside_term = -0.5 * beta * (w_diff - l_diff - pos_term)
        elif neg_term_weight > 0:
            neg_term = neg_term_weight * torch.clamp(ref_l_err - model_l_err, min=0)  # 希望 ref_l_err 越小于 model_w_err 越好
            inside_term = -0.5 * beta * (w_diff - l_diff + neg_term)
        else:
            inside_term = -0.5 * beta * (w_diff - l_diff)

    loss = - torch.nn.functional.logsigmoid(inside_term)

    if only_kl_loss:
        # dummy kl loss
        if not lose_exist:
            loss = (v_w_pred - v_w_ref_pred).pow(2).mean(dim=reduce_dims)
        else:
            loss = (v_w_pred - v_w_ref_pred).pow(2).mean(dim=reduce_dims) + (v_l_pred - v_l_ref_pred).pow(2).mean(dim=reduce_dims)

    extra_info = {
        "w_diff_loss": w_diff.mean(),
        "l_diff_loss": l_diff.mean(),
        "w_err_loss": model_w_err.mean(),
        "l_err_loss": model_l_err.mean(),
        "ref_w_err_loss": ref_w_err.mean(),
        "ref_l_err_loss": ref_l_err.mean(),
    }
    if pos_term_weight > 0:
        extra_info.update(dict(
            pos_term_loss=pos_term.mean(),
        ))
    return loss, extra_info


class MultiModalGeminiBetaPureTorchParallelDPOTrainer(GeminiTrainerAlphaMultiModal):
    def __init__(self, args, all_dataset_keys=None):
        self.dp_rank_is_correct = False # 继承太多层，没有仔细看各个函数的调用顺序，增加这个flag，确保在用到dp_rank的时候都是正确的
        super().__init__(args)
        self.build_reference_model()
        # if self.args.get("online", False):
        self.build_sampler()
        assert args.launcher == 'pure_torch'

    
    def build_sampler(self):
        model_dict = dict(
            vae=self.vae,
            model_settings=self.model_settings,
        )
        # model在推理时(prepare_model_inputs)通过unwarp self.model_engine定义即可
        factor_kwargs = {"device": self.device, "dtype": PRECISION_TO_TYPE[self.args.precision]}
        self.model_dict = GeminiBetaSampler.build_extra_model(
            self.args,
            model_dict,
            factor_kwargs,
            logger=self.logger,
        )
        self.sampler = GeminiBetaSampler(
            self.args,
            model_dict=self.model_dict,
            pipeline_name=self.pipeline_name,
            rank=self.dp_rank,
            world_size=self.dp_size,
            device=self.device,
            logger=self.logger,
        )

    def init_distributed_env(self):
        super().init_distributed_env()
        # TODO: initialize two meshes
        self.parallel_dims = init_parallel_state(
            dp_replicate=1, # world_size//8 在单个Node下进行切片
            dp_shard=-1,
            sp=1,
            tp=1,
            pp=self.args.pp_size,
            ep=self.args.ep_size,
            world_size=dist.get_world_size(),
        )
        self.parallel_dims.build_mesh('cuda')
        self.dp_rank = self.parallel_dims.dp_mesh.get_local_rank()
        self.dp_size = self.parallel_dims.dp_mesh.size()
        self.dp_rank_is_correct = True

        assert self.parallel_dims.pp_enabled, '不开pp还要用这个trainer要改造一下 train_step'

    def initialize_puretorch_model_engine(self):
        from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
        from torch.optim import Adam, AdamW
        if (self.parallel_dims.ep or self.parallel_dims.pp) and self.args.get('pretrained_ckpt', None):
            load_ckpt_path = self.args.get('pretrained_ckpt', None)
        else:
            load_ckpt_path = None
        
        ds_config = get_deepspeed_config(self.args)
        if self.args.tensorboard:
            ds_config["tensorboard"] = {
                "enabled": True,
                "output_path": str(self.output_dir.absolute()),
                "job_name": self.exp_dir.name,
            }

        def scalar_state_to_dict(scalar_state):
            scalar_state_dict = scalar_state.to_dict()
            # sync scalar_state
            if dist.is_available() and dist.is_initialized():
                gather_results_list = [None for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(gather_results_list, scalar_state_dict)
                scalar_state_dict = gather_results_list

            client_state = {
                "config": self.args,
                "scalar_state": scalar_state_dict,
            }
            return client_state
        
        self.logger.info(f"!!!!!! Must check the gradient_accumulation_steps: {self.args.gradient_accumulation_steps}")
        self.model_engine = GeminiParallelEngine(
            model=self.model,
            ds_config=ds_config,
            load_ckpt_path=load_ckpt_path,
            # auto_benchmark=True,
            # training_benchmark_schedule=('1F1B',),
            # pipeline_parallel_schedule='1F1B',
            # m_microbatch=1,
            micro_batch_size=2, # grpo micro_batchsize should be 2
            optimizer_config=dict(
                optimizer_cls={'AdamW': AdamW, 'Adam': Adam}[self.args.optimizer_name],
                optimizer_kwargs=dict(
                    lr=self.args.optimizer_params['lr'],
                    betas=self.args.optimizer_params['betas'],
                    weight_decay=self.args.optimizer_params['weight_decay'],
                    eps=self.args.optimizer_params['eps'],
                )
            ),
            pp_enable_autocast=self.args.autocast_dtype != 'fp32',
            # pp_enable_autocast=False,
            autocast_prec=self.args.autocast_dtype,
            weight_prec=self.args.precision,
            # cpu_offload=True,
            initial_training_states=scalar_state_to_dict(self.get_states_cls('scalar')()),
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        # resume from checkpoint
        self.resume_puretorch = self.args.get("resume_puretorch", False)
        client_state = None
        if self.resume_puretorch:
            _, training_states = self.model_engine.load_checkpoint(self.resume_puretorch, load_optimizer_states=True)
            self.logger.info(f"training_states: {training_states}")
            # FIXME(yutaocui): 这里需要修改
            client_state = training_states['scalar_state']

            # Resume ScalarStates. Overlap the initial states.
            self.ss = self.get_states_cls('scalar').from_pretrained(
                client_state["scalar_state"],
                rank=self.rank,
                world_size=self.world_size,
                default_rank0_ss=self.args.default_rank0_ss,
                default={'lr': self.args.lr},
            )
            self.logger.info(f"Resume ScalarStates: {self.ss}")

    
    def build_extra_model(self):
        """ Extra frozen models. """
        args = self.args

        # ====================== Build VAE ========================
        self.logger.info("Building VAE...")
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
            only_encoder=False if self.args.get("online", False) else True,
            sample_size=args.get('vae_sample_size'),
        )
        if args.get('vae_spatial_tiling', False):
            self.vae.enable_spatial_tiling()
        self.vae_generater = torch.Generator(self.device).manual_seed(self.dp_rank)
        if args.get('prerun_vae', torch.backends.cudnn.benchmark):
            self.prerun_vae()

        # ====================== Build tokenizer ========================
        if self.dataset is not None:
            self.tkwrapper = self.dataset.tokenizer
        elif hasattr(self, 'dataset_dict'):
            self.tkwrapper = list(self.dataset_dict.values())[0].tokenizer
        else:
            self.tkwrapper = TokenizerWrapper(args.tokenizer_name, self.logger)

        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

        # ====================== Build reward models ========================
        # TODO

    
    def task_init(self, args, all_dataset_keys=None):
        self.sampling_probs_dict = json.loads(args.sampling_probs)
        self.all_dataset_keys = sorted(list(self.sampling_probs_dict.keys()))

        # Define what dummy token are incurred by each task.
        self.dummy_to_tasks = dict(
            t2i={"t2i", "editing", "subject_driven", "interleave", "face_id_clip"},
            mmu={"mmu", "mmu_interleave", "face_id_clip"},
            face={},
        )

        # Define the sequence batch size for each task for long sequence training.
        self.seq_batch_size = args.get("seq_batch_size", {})
        if isinstance(self.seq_batch_size, str):
            self.seq_batch_size = json.loads(self.seq_batch_size)

        # Visualize sampled images and corresponding rewards
        self.visualize_reward = args.get("visualize_reward", False)
        self.visualize_every = args.get("visualize_every", 100)

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.pipeline_name = "transfusion"

        # DPO params
        self.update_ref_model = args.get("update_ref_model", False)
        self.ema_update_ref_steps = args.get("ema_update_ref_steps", 50)
        self.ema_update_ref_decay = args.get("ema_update_ref_decay", 0.95)

    
    def build_reference_model(self):
        """
        Build the reference model. Note that model must be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[self.args.precision]}
        # assert self.args.get('pretrained_reference_model_ckpt', None) is not None, "pretrained_reference_model_ckpt is not set."
        self.ref_model, _ = build_model(self.args, self.args.get('pretrained_reference_model_ckpt', None), logger=self.logger, **factor_kwargs)
        self.ref_model.requires_grad = False

        if self.args.pp_size > 1 or self.args.ep_size > 1:
            from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
            if (self.parallel_dims.ep or self.parallel_dims.pp) and self.args.get('pretrained_reference_model_ckpt', None):
                load_ckpt_path = self.args.get('pretrained_reference_model_ckpt', None)
            else:
                load_ckpt_path = None
            self.ref_model = GeminiParallelEngine(
                model=self.ref_model,
                load_ckpt_path=load_ckpt_path,
                micro_batch_size=1,
                pp_enable_autocast=self.args.autocast_dtype != 'fp32',
                # pp_enable_autocast=False,
                autocast_prec=self.args.autocast_dtype,
                weight_prec=self.args.precision,
                # cpu_offload=True,
            )
            self.ref_model.eval()

        self.logger.info(f"Reference model built on device: {self.device}")

        assert self.dp_rank_is_correct
        loguru.logger.info(f'when seeding, {self.dp_rank=}')
        set_manual_seed(self.args.global_seed + self.dp_rank)
        

    def update_reference_model(self, decay=0.95):
        """
        Update reference model using EMA from current training model.
        Args:
            decay: EMA decay rate, default 0.95
        """
        start_time = time.time()
        ref_model_params = OrderedDict(self.ref_model.named_parameters())
        current_model_params = OrderedDict(self.model_engine.named_parameters())
        
        for name, ref_param in ref_model_params.items():
            ref_param.data.mul_(decay).add_(current_model_params[name].data, alpha=1.0 - decay)
        end_time = time.time()
        self.logger.info(f"Updated reference model with EMA decay={decay} at step {self.ss.update_steps}, time cost: {end_time - start_time}s")


    def build_data_iterator(self):
        self.dataloader = CombinedBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.dp_rank,
            world_size=self.dp_size,
            datasets=self.dataset_dict,
            samplers=self.sampler_dict,
            dataloaders=self.dataloader_dict,
            sampling_probs=self.sampling_probs_dict,
            initial_seed=self.args.global_seed,
            sampling_mode=self.args.get('combined_iterator_sampling_mode', 'random'),
            cache_shuffle=self.args.get('cache_shuffle'),
            fixed_key=self.cur_key,
            fixed_key_group=self.cur_key_group,
            force_sync_shuffle=False,
        )

    
    def build_dataloader(self):
        args = self.args
        self.dataloader_preliminary_setup()

        dataloader_kwargs = dict(
            **args.dataloader_params,
            worker_init_fn=set_worker_seed_builder(self.dp_rank),
            shuffle=False,
            drop_last=True,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )

        def _filter_dummies(dummy_list):
            # Filter out dummies that are not in the dataset keys
            valid_dummies = []
            for dummy_candidate in dummy_list:
                if any(task in self.all_dataset_keys for task in self.dummy_to_tasks[dummy_candidate]):
                    valid_dummies.append(dummy_candidate)
            return valid_dummies

        # =====================================
        #     Text(+Image) to image data
        # =====================================
        self.task_dummy_dict['t2i'] = _filter_dummies(['mmu'])
        self.task_dummy_dict['face_id_clip'] = []
        task_info_list = [
            dict(dataset_tag="t2i", cur_task="t2i", cls=RLTextImageArrowStream),
        ]
        for item in task_info_list:
            dataset_tag = item['dataset_tag']
            if dataset_tag not in self.all_dataset_keys:
                continue
            task_batch_size = args.get(f'{dataset_tag}_batch_size', self.micro_batch_size)
            if dataset_tag == "t2i":
                multireso = args["t2i_index_kwargs"]["multireso"]
            else:
                multireso = args.get(f'{dataset_tag}_index_kwargs')[f"{dataset_tag}_multireso"]
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                dataset_tag=dataset_tag,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                logger=self.logger,
                dummy_number=sum([self.dummy_dict[task] for task in self.task_dummy_dict[item['cur_task']]]),
                template=args.sequence_template,
                # if sequence batch is enabled, attention mask sequence length -1 is disabled in __getitem__,
                # and performed in seq_collate_fn instead.
                attn_mask_seq_m1=dataset_tag not in self.seq_batch_size,
            )
            # Build sampler and data loader
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.dataset_num_replicas[dataset_tag],
                rank=self.dataset_rank[dataset_tag],
                batch_size=task_batch_size if multireso else 1,  # Provide bsz to use multireso.
                **sampler_kwargs,
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=task_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )
    

    @property
    def optimizer(self):
        return self.model_engine.optimizer
    

    def _vae_encode_tensor_dpo(self, image, lose_image, sample_type=None, n_tokens=None, sample_timestep=None):
        # ===================================== prepare diffusion =====================================
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            vae_encode_result_w = self.vae.encode(image)
            vae_encode_result_l = self.vae.encode(lose_image)
            if is_torch_tensor(vae_encode_result_w):
                latents_w = vae_encode_result_w
                latents_l = vae_encode_result_l
            else:
                latents_w = vae_encode_result_w.latent_dist.sample(generator=self.vae_generater)
                latents_l = vae_encode_result_l.latent_dist.sample(generator=self.vae_generater)
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                latents_w.sub_(self.vae.config.shift_factor)
                latents_l.sub_(self.vae.config.shift_factor)
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                latents_w.mul_(self.vae.config.scaling_factor)
                latents_l.mul_(self.vae.config.scaling_factor)

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert latents_w.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents_w = latents_w.squeeze(2)
            assert latents_l.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents_l = latents_l.squeeze(2)

        if sample_type is not None:
            # Choose whether using the same t and x_0 for both win and lose samples, both use the same t
            if sample_type == "sample":
                t, w_x_0, w_x_1 = self.denoiser.sample(latents_w, n_tokens)
                l_x_1 = latents_l
                l_x_0 = w_x_0
                same_x0_for_win_lose = self.args.get('same_x0_for_win_lose', True)
                if not same_x0_for_win_lose:
                    _, l_x_0, l_x_1 = self.denoiser.sample(latents_l, n_tokens)
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
            
            if sample_timestep is not None:
                assert sample_timestep <= 1.0 and sample_timestep >= 0.0, "sample_timestep should be in [0.0, 1.0)"
                t = sample_timestep.to(w_x_0.device)

            # same t for win and lose
            t, w_x_t, w_u_t = self.denoiser.path_sampler.plan(t, w_x_0, w_x_1)
            t, l_x_t, l_u_t = self.denoiser.path_sampler.plan(t, l_x_0, l_x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000

            return (
                VAEEncodeOutput(t=t, model_t=model_t, x_0=w_x_0, x_t=w_x_t, u_t=w_u_t, latents=latents_w),
                VAEEncodeOutput(t=t, model_t=model_t, x_0=l_x_0, x_t=l_x_t, u_t=l_u_t, latents=latents_l),
            )

        return (
            VAEEncodeOutput(latents=latents_w),
            VAEEncodeOutput(latents=latents_l),
        )

    def vae_encode_dpo(self, images, lose_images, sample_type=None, n_tokens=None, sample_timestep=None):
        if isinstance(images, list):
            assert n_tokens is None, "n_tokens is not supported for list of images."
            batch_output_w, batch_output_l = [], []
            for image_item, lose_image_item in zip(images, lose_images):
                if is_torch_tensor(image_item) and image_item.ndim == 3:
                    image_item = image_item.unsqueeze(0)
                    lose_image_item = lose_image_item.unsqueeze(0)

                if isinstance(image_item, list):
                    vae_outputs_w, vae_outputs_l = [
                        self._vae_encode_tensor_dpo(
                            image[None].to(self.device), lose_image[None].to(self.device), sample_type=sample_type, n_tokens=n_tokens, sample_timestep=sample_timestep
                        )
                        for image, lose_image in zip(image_item, lose_image_item)
                    ]
                    outputs_w = VAEEncodeOutput.cat(vae_outputs_w)
                    outputs_l = VAEEncodeOutput.cat(vae_outputs_l)

                else:
                    image_item = image_item.to(self.device)
                    lose_image_item = lose_image_item.to(self.device)
                    outputs_w, outputs_l = self._vae_encode_tensor_dpo(image_item, lose_image_item, sample_type=sample_type, n_tokens=n_tokens, sample_timestep=sample_timestep)

                batch_output_w.append(outputs_w)
                batch_output_l.append(outputs_l)

            batch_output_w = VAEEncodeOutput.build(batch_output_w)
            batch_output_l = VAEEncodeOutput.build(batch_output_l)

        elif is_torch_tensor(images):
            images = images.to(self.device)
            lose_images = lose_images.to(self.device)
            if images.ndim == 4:
                batch_output_w, batch_output_l = self._vae_encode_tensor_dpo(images, lose_images, sample_type=sample_type, n_tokens=n_tokens, sample_timestep=sample_timestep)

            elif images.ndim == 5:
                vae_outputs_w, vae_outputs_l = [
                    self._vae_encode_tensor_dpo(image, lose_image, sample_type=sample_type, n_tokens=n_tokens, sample_timestep=sample_timestep)
                    for image, lose_image in zip(images, lose_images)
                ]
                batch_output_w = VAEEncodeOutput.build(vae_outputs_w)
                batch_output_l = VAEEncodeOutput.build(vae_outputs_l)

            else:
                raise ValueError(f"images should have shape [B, C, H, W] or [B, n, C, H, W], got {images.shape}")

        else:
            raise ValueError(f"Unknown images type, expected [list, torch.Tensor], got {type(images)}")

        return batch_output_w, batch_output_l

    def prepare_model_dpo_t2i_inputs(self, batch: Dict, device: Union[int, str], sample_timestep=None, **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        lose_exist = batch["lose_exist"]
        use_sft_loss = False #batch["use_sft_loss"]

        # concat win-image and lose-image
        tokens = tokens.repeat(2, 1)
        target_tokens = target_tokens.repeat(2, 1)
        text_mask = text_mask.repeat(2, 1)
        image_mask = image_mask.repeat(2, 1)

        # Add dummy tokens
        extra = dict(
            n_samples=batch["n_samples"].repeat(2).to(device),
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
        )
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"] * 2,
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=batch["iw_ih_scatter_index"].repeat(2, 1).to(device),        # [2 * b, 2]
                iw_ih_scatter_src=batch["iw_ih_scatter_src"].repeat(2, 1).to(device),            # [2 * b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=batch["timestep_scatter_index"].repeat(2, 1).to(device),  # [2 * b, 1]
            ))
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attn_type = self.args.get('t2i_task_kwargs', {}).get('attn_type', 'auto')
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)
            attention_mask = attention_mask.repeat(2, 1, 1, 1)  # win-image and lose-image share the same attention mask
        elif attn_type == 'flex':
            assert False, "Flex attention is not supported for DPO now."
            bsz, seq_len = tokens.shape
            batch["gen_image_slices"] = batch["gen_image_slices"] * 2
            if bsz == 1:
                image_slices = batch["gen_image_slices"][0]
                mask_mod = create_text_image_mask_mod(image_slices, seq_len, device)
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
            else:
                mask_mod = create_batch_text_image_mask_mod(batch["gen_image_slices"], seq_len, device)
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        # ===================================== prepare diffusion =====================================
        out_w, out_l = self.vae_encode_dpo(batch["image"], batch["lose_image"], sample_type="sample", sample_timestep=sample_timestep)
        t_w, model_t_w, x_0_w, x_t_w, u_t_w = out_w.t, out_w.model_t, out_w.x_0, out_w.x_t, out_w.u_t
        t_l, model_t_l, x_0_l, x_t_l, u_t_l = out_l.t, out_l.model_t, out_l.x_0, out_l.x_t, out_l.u_t
        
        if self.parallel_dims.pp_enabled:
            # Defensive programming
            torch.distributed.broadcast(x_t_w, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(x_t_l, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(u_t_w, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(u_t_l, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(model_t_w, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(model_t_l, group_src=0, group=self.parallel_dims.pp_group)
        
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                                      # [b, seqlen]
            x_t=torch.cat([x_t_w, x_t_l], dim=0),            # [b, c, h, w]
            t=torch.cat([model_t_w, model_t_l], dim=0),      # [b]
            # target=target_tokens,                            # [b, seqlen]
            attention_mask=attention_mask,                   # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                 # For loss
            return_loss=False,                               # Do not return diffusion loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, u_t_w, u_t_l, lose_exist, use_sft_loss, batch_size, n_tokens


    def prepare_model_ti2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)
        # For interleave data, src_image_mask and src_images can be None
        if "src_image_mask" in batch:
            src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        else:
            src_image_mask = None
        if "und_image_mask" in batch:
            und_image_mask = batch["und_image_mask"][:, :-1].contiguous().to(device)
        else:
            und_image_mask = None
        
        # concat win-image and lose-image
        tokens = tokens.repeat(2, 1)
        target_tokens = target_tokens.repeat(2, 1)
        text_mask = text_mask.repeat(2, 1)
        image_mask = image_mask.repeat(2, 1)
        if src_image_mask is not None:
            src_image_mask = src_image_mask.repeat(2, 1)
        if und_image_mask is not None:
            und_image_mask = und_image_mask.repeat(2, 1)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,            # [2b, seqlen]
            image_mask=image_mask,          # [2b, seqlen]
            **(dict(src_image_mask=src_image_mask) if src_image_mask is not None else {}),  # [2b, seqlen]
            **(dict(und_image_masks=und_image_mask) if und_image_mask is not None else {}),  # [2b, seqlen]
        )
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"] * 2,
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=to_device(batch["iw_ih_scatter_index"].repeat(2, 1), device),        # [2b, 2]
                iw_ih_scatter_src=to_device(batch["iw_ih_scatter_src"].repeat(2, 1), device),            # [2b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=to_device(batch["timestep_scatter_index"].repeat(2, 1), device),  # [2b, 1]
            ))
        # for task in self.task_dummy_dict['ti2i']:
        #     if self.dummy_dict[task]:
        #         tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
        #             tokens, target_tokens, extra,
        #             dummy_token_type=task, dummy_number=self.dummy_dict[task], device=device
        #         )
        batch_size, n_tokens = tokens.shape

        # Attention mask
        # TODO: support flex attention
        attn_type = 'auto'
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)
            attention_mask = attention_mask.repeat(2, 1, 1, 1)  # win-image and lose-image share the same attention mask
        elif attn_type == 'flex':
            assert False, "Flex attention is not supported for DPO now."
        else:
            raise NotImplementedError(f"Attention type {attn_type} is not supported.")

        # ===================================== prepare diffusion =====================================
        out_w, out_l = self.vae_encode_dpo(batch["image"], batch["lose_image"], sample_type="sample")
        t_w, model_t_w, x_0_w, x_t_w, u_t_w = out_w.t, out_w.model_t, out_w.x_0, out_w.x_t, out_w.u_t
        t_l, model_t_l, x_0_l, x_t_l, u_t_l = out_l.t, out_l.model_t, out_l.x_0, out_l.x_t, out_l.u_t

        if "src_images" in batch and batch["src_images"] is not None:
            sout = self.vae_encode(batch["src_images"], sample_type="sample_start")
            input_src_t, input_src_x = sout.model_t, sout.x_t
            input_src_t = input_src_t.repeat(2)
            input_src_x = input_src_x.repeat(2, 1, 1, 1)
        else:
            input_src_t, input_src_x = None, None

        # joint image mode
        if "und_images" in batch:
            und_images = to_device(batch["und_images"], device)
            assert "und_images" not in extra, "und_images should not be added for dummy token"
            extra.update(dict(
                und_images=und_images.repeat(2, 1, 1, 1),  # 需要check shape
            ))

        if "vision_encoder_kwargs" in batch:
            try:
                # 需要check shape
                vision_encoder_kwargs = {k: to_device(v.repeat(2, 1, 1, 1), device) for k, v in batch["vision_encoder_kwargs"].items()}
            except Exception as e:
                vision_encoder_kwargs = {k: [to_device(v_.repeat(2, 1, 1, 1), device) for v_ in v] for k, v in batch["vision_encoder_kwargs"].items()}
            
            extra.update(dict(
                vision_encoder_kwargs=vision_encoder_kwargs,
            ))

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                                      # [b, seqlen]
            x_t=torch.cat([x_t_w, x_t_l], dim=0),            # [b, c, h, w]
            t=torch.cat([model_t_w, model_t_l], dim=0),      # [b]
            src_x=input_src_x,                               # [b, c, h, w]
            src_t=input_src_t,                               # [b]
            target=target_tokens,                            # [b, seqlen]
            attention_mask=attention_mask,                   # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            return_loss=False,                               # Do not return diffusion loss
            data_type=batch['data_type'][0],                 # For loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens


    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], sample_timestep=None, **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_dpo_t2i_inputs(batch, device, sample_timestep=sample_timestep, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    def train_step(self, batch, sample_timestep=None, only_kl_loss=False, **kwargs):
        start1 = time.time()
        model_input_kwargs, u_t_w, u_t_l, lose_exist, use_sft_loss, cur_batch_size, n_tokens = self.prepare_model_inputs(
            batch, 
            self.device,
            sample_timestep=sample_timestep,
            step=kwargs.get("step", 0),
        )
        torch.cuda.synchronize()
        duration1 = time.time() - start1

        start2 = time.time()

        ############################ Reference Model ############################
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            with torch.no_grad():
                output = self.ref_model(**model_input_kwargs)
                ref_model_pred = output["diffusion_prediction"]

        ############################ Policy Model and Loss Closure ############################
        def loss_closure(model_output, moe_loss):
            model_pred = model_output["diffusion_prediction"]
            v_w_pred, v_l_pred = model_pred.chunk(2, dim=0)
            v_w_ref_pred, v_l_ref_pred = ref_model_pred.chunk(2, dim=0)

            pos_term_weight = self.args.get('pos_term_weight', 0.0)
            neg_term_weight = self.args.get('neg_term_weight', 0.0)
            hinge_threshold = self.args.get('hinge_threshold', 0.0)
            loss, extra_info = flow_dpo_loss_fn(
                v_w_pred,
                v_l_pred,
                v_w_ref_pred,
                v_l_ref_pred,
                u_t_w,
                u_t_l,
                lose_exist,
                use_sft_loss,
                pos_term_weight=pos_term_weight,
                neg_term_weight=neg_term_weight,
                hinge_threshold=hinge_threshold,
                only_kl_loss=only_kl_loss,
            )
            extra_info['loss'] = loss.detach()

            return loss, extra_info

        self.model_engine.register_loss_closure(loss_closure)
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            self.model_engine(**model_input_kwargs)
            loss_dict = self.model_engine.get_cached_result('loss_dict', merge_op='mean')
        
        self.logger.info(f"loss_dict: {loss_dict}")
    
        torch.cuda.synchronize()
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times
    
    def update_train_states(self, ss, cs, batch, batch_size, n_tokens, loss):
        # A forward-backward step is counted as one train step.
        ss.add(train_steps=1, epoch_train_steps=1)
        cs.add(log_steps=1, running_loss=loss)
        # If training long sequence, each sequence may contain multiple samples.
        # Therefore, we sum `n_samples` to get the real number of samples.
        samples = batch["n_samples"].sum().item()
        key = batch["dtype"][0]
        cs.running_samples[key] += samples
        cs.running_tokens[key] += batch_size * n_tokens

        # We enable `is_update_step` if the current step is the gradient accumulation boundary.
        is_update_step = self.ss.train_steps % self.grad_accu_steps == 0
        if is_update_step:
            ss.add(
                update_steps=1,
                epoch_update_steps=1,
                current_run_update_steps=1
            )
            ss.lr = self.optimizer.param_groups[0]["lr"]

        return is_update_step

    def train_loop(self):
        args = self.args
        self.model_engine.train()
        self.ss.current_run_update_steps = 0

        if args.init_save:
            save_checkpoint(args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

        # Training loop
        start_epoch = self.ss.epoch
        finished = False
        nan_grad_count = 0

        for epoch in range(start_epoch, args.max_epochs):
            self.shuffle_dataset_and_set_start_index(self.ss)

            with profiler_context(
                args.profile, self.exp_dir, worker_name=f"Rank_{self.rank}"
            ) as prof:
                self.logger.info(f"Beginning epoch {epoch}...")
                try:
                    self.logger.info(f"  Steps left this epoch: {len(self.dataloader) // self.grad_accu_steps:,}")
                except NotImplementedError:
                    pass
                # Define cycle states, which accumulate the training information between log_steps.
                cs = self.get_states_cls('cycle')()
                torch.cuda.synchronize()
                start_time = time.time()
                data_start = time.time()
                times = {}

                for bi, batch in enumerate(self.dataloader):
                    torch.cuda.synchronize()
                    times['data'] = time.time() - data_start

                    # Dry run dataloader to check data processing.
                    if args.get('dry_run_dataloader'):
                        if bi > 0 and bi % 20 == 0:
                            self.logger.info(
                                f"Dry run dataloader: {bi} batches processed. Average time: {times['data'] / 20:.2f}s."
                            )
                            data_start = time.time()

                        continue
                    
                    # FIXME: 临时支持
                    if self.args.get("dpo_for_low_noise_timestep", False):
                        num_inference_steps = 50
                        start_ts = self.args.get("dpo_for_low_noise_timestep_start", num_inference_steps - 10)
                        end_ts = self.args.get("dpo_for_low_noise_timestep_end", num_inference_steps - 1)
                        if self.args.get("dpo_only_kl_loss_trick", False):
                            if bi % 2 == 0:
                                timestep_idx = random.randint(0, start_ts)
                                only_kl_loss = True
                            else:
                                timestep_idx = random.randint(start_ts, end_ts)
                                only_kl_loss = False
                        else:
                            timestep_idx = random.randint(start_ts, end_ts)
                            only_kl_loss = False
                        self.sampler.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
                        # timesteps = self.sampler.pipeline.scheduler.timesteps
                        timestep = self.sampler.pipeline.scheduler.sigmas[timestep_idx:timestep_idx+1].to(self.device)
                        self.logger.info(f"timestep-{timestep_idx}, timestep-{timestep}")
                        loss_dict, batch_size, n_tokens, forward_times = self.train_step(batch, sample_timestep=timestep, step=self.ss.update_steps, only_kl_loss=only_kl_loss)
                    else:
                        loss_dict, batch_size, n_tokens, forward_times = self.train_step(batch, step=self.ss.update_steps)
                    times.update(forward_times)

                    backward_start = time.time()
                    loss = loss_dict["loss"].mean()
                    for k, v in loss_dict.items():
                        if "loss" in k and k != "loss":
                            cs.running_sub_loss_dict[k] += v.mean().item()
                            cs.running_sub_step_dict[k] += 1

                    self.model_engine.backward(loss)
                    torch.cuda.synchronize()
                    times['backward'] = time.time() - backward_start

                    is_update_step = self.update_train_states(self.ss, cs, batch, batch_size, n_tokens, loss.item())

                    # Update model parameters at the boundary of gradient accumulation.
                    update_start = time.time()
                    lrs = [group["lr"] for group in self.model_engine.optimizer_container.param_groups]
                    self.model_engine.step()
                    torch.cuda.synchronize()
                    times['update'] = time.time() - update_start

                    if self.ss.update_steps >= args.max_training_steps:
                        # Enter stopping routine if max steps reached after this step.
                        finished = True

                    # Update EMA model at the step of main model parameters update.
                    if args.use_ema and is_update_step:
                        self.ema.update(self.model_engine.module)

                    # Update reference model using EMA every n steps
                    if is_update_step and self.update_ref_model and self.ss.update_steps % self.ema_update_ref_steps == 0:
                        self.ref_model.reshard()
                        self.update_reference_model(self.ema_update_ref_decay)

                    # Log training information:
                    if is_update_step and self.ss.update_steps % args.log_every == 0:
                        # All-gather scalar states and cycle states.
                        all_cs: List[Optional[CycleStates]] = [None for _ in range(self.world_size)]
                        torch.distributed.all_gather_object(all_cs, cs)

                        # Calculate average main loss
                        avg_loss = sum([cs_i.running_loss for cs_i in all_cs]) / sum([cs_i.log_steps for cs_i in all_cs])
                        # Calculate average sub losses.
                        merged_loss_dict = {}
                        merged_step_dict = {}
                        for cs_i in all_cs:
                            for k, v in cs_i.running_sub_loss_dict.items():
                                if k not in merged_loss_dict:
                                    merged_loss_dict[k] = v
                                    merged_step_dict[k] = cs_i.running_sub_step_dict[k]
                                else:
                                    merged_loss_dict[k] += v
                                    merged_step_dict[k] += cs_i.running_sub_step_dict[k]
                        sorted_keys = sorted(list(merged_loss_dict.keys()))
                        avg_sub_loss_dict = {k: merged_loss_dict[k] / merged_step_dict[k] for k in sorted_keys}
                        # Calculate cumulated metrics.
                        cum_samples = self.update_log_states(self.ss, all_cs)

                        # Synchronize cuda to accurately measure training speed:
                        torch.cuda.synchronize()
                        end_time = time.time()
                        steps_per_sec = cs.log_steps / self.grad_accu_steps / (end_time - start_time)
                        seconds_per_step = (end_time - start_time) / (cs.log_steps / self.grad_accu_steps)
                        samples_per_sec = cum_samples / (end_time - start_time)

                        grad_norm = self.model_engine.get_global_grad_norm()
                        user_log_events, user_summary_events = self.get_events(self.ss, avg_loss)

                        log_events = [
                             f"Train Loss: {avg_loss:.4f}",
                             *[f"{k}: {v:.4f}" for k, v in avg_sub_loss_dict.items()],
                         ] + [f"Lr{lr_i}: {lr:.6g}" for lr_i, lr in enumerate(lrs)] + [
                             f"Steps/Sec: {steps_per_sec:.2f}",
                             f"Sec/Step: {seconds_per_step:.2f}",
                             f"Samples/Sec: {int(samples_per_sec):d}",
                             f"Global Grad Norm: {grad_norm:.4f}",
                             f"Nan Grad Count: {nan_grad_count}",
                         ] + user_log_events + [
                             f"T{time_key}: {duration:.4f}"
                             for time_key, duration in times.items()
                         ]
                        summary_events = [
                            ("Train/Steps/train_loss", avg_loss, self.ss.update_steps),
                            *[("Train/Steps/" + k, v, self.ss.update_steps) for k, v in avg_sub_loss_dict.items()],
                            ("Train/Steps/LR", self.ss.lr, self.ss.update_steps),
                            ("Train/Steps/steps_per_sec", steps_per_sec, self.ss.update_steps),
                            ("Train/Steps/samples_per_sec", int(samples_per_sec), self.ss.update_steps),
                            ("Train/Steps/seconds_per_step", seconds_per_step, self.ss.update_steps),
                            ("Train/Steps/grad_norm", grad_norm, self.ss.update_steps),
                            ("Train/ComputationsAttn/train_loss", avg_loss, self.ss.consumed_computations_attn),
                            ("Train/ComputationsTotal/train_loss", avg_loss, self.ss.consumed_computations_total),
                        ] + user_summary_events
                        # Log the training information to the logger.
                        self.logger.info(f"(step={self.ss.update_steps:07d}) " + ", ".join(log_events))
                        # Log the training information to the monitor.
                        if self.model_engine.monitor.enabled and self.rank == 0:
                            self.model_engine.monitor.write_events(summary_events)

                        # Reset monitoring variables:
                        cs.reset()
                        start_time = time.time()

                    # Save checkpoint:
                    if (is_update_step and self.ss.update_steps % args.ckpt_every == 0) or (
                        finished and args.final_save
                    ):
                        self.save_checkpoint()
                    
                    # Perform evaluation
                    if args.validation_every > 0 and (
                        (is_update_step and self.ss.update_steps % args.validation_every == 0)
                        or (
                            is_update_step
                            and self.ss.current_run_update_steps in args.validation_at_steps
                        )
                        or finished
                    ):
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()
                        # del loss_dict
                        self.model_engine.module.eval()
                        self.val_logger.info(
                            f"Start evaluation after train epoch={self.ss.epoch}, step={self.ss.update_steps} "
                            + (f"(update_step={self.ss.update_steps:07d}) " if self.grad_accu_steps > 1 else "")
                        )
                        with torch.no_grad():
                            self.eval_step(loss_dict)
                        # Return to training mode
                        self.model_engine.module.train()
                        # Clear the cache to save GPU memory.
                        torch.cuda.empty_cache()
                    
                    dist.barrier()

                    if prof:
                        prof.step()

                    if finished:
                        self.logger.info(f"Finished and breaking loop at step={self.ss.update_steps}.")
                        break

                    torch.cuda.synchronize()
                    data_start = time.time()

                if finished:
                    self.logger.info(f"Finished and breaking loop at epoch={epoch}.")
                    break

                # Reset epoch states
                new_epoch = self.ss.inc_epoch()
                self.logger.info(f"Increase epoch to {new_epoch}.")

    def train(self):
        self.before_train()
        dist.barrier()
        self.train_loop()
        self.after_train()
