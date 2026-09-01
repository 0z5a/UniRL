import gc
import os
import time
import json
import loguru
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Union, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import deepspeed
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
from index_kits.sampler import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex
try:
    from torch.nn.attention.flex_attention import create_block_mask
except:
    pass

from hymm.trainers.transfusion_parallel import tp_sp_decorator
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
from ..models.reward_models.text_ocr_groundingQwenVL import TextOCRGroundingQwenVL
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
from ..utils import lr_schedules
from ..utils.fsdp_wrapper import FSDPEngine
from ..utils.helpers import default, get_obj_from_str, to_2tuple
from ..utils.torch_distributions import gather_tensor
from ..models.autoencoders import load_vae
from ..models.tokenizers import TokenizerWrapper
from ..diffusion import load_denoiser
from ..utils.deepspeed_utils import unwrap_model_for_generation_deepspeed
from hymm.parallelism.parallel_states import get_parallel_state, init_parallel_state
from hymm.ar.pipelines.pipeline_transfusion_text2image_with_logprob import rescale_noise_cfg
from hymm.diffusion.pipelines.flow_sde_with_logprob import sde_step_with_logprob

gc.set_threshold(7000, 100, 100)


def reda_policy_loss_fn(image, raw_image, ref_image, sigmas, reward_model, alignment_guidance, space_guidance):
    """
    REDA policy loss function.
    
    Args:
        image: Decoded image tensor in [0, 1] range (from vae_decode)
        raw_image: Raw decoded image tensor in [0, 1] range (from vae_decode)
        ref_image: Reference image tensor in [-1, 1] range (raw batch image)
        sigmas: Noise schedule sigmas
        reward_model: CLIP reward model
        alignment_guidance: Alignment guidance scale
        space_guidance: Space guidance scale
    
    Returns:
        torch.Tensor: Computed loss value
    """
    # image and raw_image should already be in [0, 1] range from vae_decode
    # ref_image needs to be converted from [-1, 1] to [0, 1] range
    ref_image = (ref_image / 2 + 0.5).clamp(0, 1)
    
    outputs_real_i, outputs_real_i_feat = reward_model.get_fine_image_features(image)
    with torch.no_grad():
        output_real_raw, output_real_raw_feat = reward_model.get_fine_image_features(raw_image)
        outputs_real_ref, outputs_real_ref_feat = reward_model.get_fine_image_features(ref_image)

    def correlation(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.diagonal(a @ b.T)

    ex_scale = alignment_guidance
    dx_scale = space_guidance

    target = (outputs_real_ref + (output_real_raw - outputs_real_ref) * ex_scale).clone().detach()
    output_real_raw_feat = (outputs_real_ref_feat + (
                output_real_raw_feat - outputs_real_ref_feat) * dx_scale).clone().detach()

    time_alignment = F.mse_loss(output_real_raw_feat, outputs_real_i_feat) * sigmas ** 2
    raw_alignment = correlation(output_real_raw, outputs_real_i)
    detail_score = (time_alignment + raw_alignment * 2) / 3

    pseudo_alignment = correlation(outputs_real_i, target)
    loss = raw_alignment.clone().detach() * 2 - detail_score - pseudo_alignment
    return loss.mean()


class MultiModalGeminiBetaREDAPureTorchParallelTrainer(GeminiTrainerAlphaMultiModal):
    def __init__(self, args, all_dataset_keys=None):
        self.dp_rank_is_correct = False # 继承太多层，没有仔细看各个函数的调用顺序，增加这个flag，确保在用到dp_rank的时候都是正确的
        super().__init__(args)
        self.build_reference_model()
        self.build_sampler()
        assert args.launcher == 'pure_torch'

    
    def build_sampler(self):
        model_dict = dict(
            vae=self.vae,
            model_settings=self.model_settings,
        )
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

        # Resume from checkpoint
        # Must guarantee the `args.resume_policy_model_puretorch` is set. Otherwise, the loading process will raise error
        # that "Missing key in checkpoint state_dict: states.config.resume_policy_model_puretorch"
        assert (
            hasattr(self.args, 'resume_policy_model_puretorch')
        ), f"args.resume_policy_model_puretorch is not set, please set it to 'False' if you don't want to resume from checkpoint"
        self.resume_policy_model_puretorch = self.args.resume_policy_model_puretorch
        if self.resume_policy_model_puretorch:
            _, training_states = self.model_engine.load_checkpoint(self.resume_policy_model_puretorch, load_optimizer_states=True)
            client_state = training_states['scalar_state']
            self.logger.info(f"scalar_state: {client_state[0]}")

            # Resume ScalarStates. Overlap the initial states.
            self.ss = self.get_states_cls('scalar').from_pretrained(
                client_state[0],  # directly use the first-rank's scalar_state, since all ranks have the same scalar_state
                rank=self.rank,
                world_size=self.world_size,
                default_rank0_ss=self.args.default_rank0_ss,
                default={'lr': self.args.lr},
            )

            # Reset the consumed_samples_total and epoch_consumed_samples for resuming dataloder
            consumed_steps = client_state[0]['train_steps']
            self.ss.consumed_samples_total = defaultdict(int)
            self.ss.epoch_consumed_samples = defaultdict(int)
            for key in self.all_dataset_keys:
                self.ss.consumed_samples_total[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size)
                self.ss.epoch_consumed_samples[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size)
            self.logger.info(f"Resumed ScalarStates: {self.ss}")

    
    def initialize_puretorch_reference_model_engine(self):
        from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
        from torch.optim import Adam, AdamW
        if (self.parallel_dims.ep or self.parallel_dims.pp) and self.args.get('pretrained_reference_model_ckpt', None):
            load_ckpt_path = self.args.get('pretrained_reference_model_ckpt', None)
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
        self.ref_model_engine = GeminiParallelEngine(
            model=self.ref_model,
            ds_config=ds_config,
            load_ckpt_path=load_ckpt_path,
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

        # Resume from checkpoint
        # Must guarantee the `args.resume_reference_model_puretorch` is set. Otherwise, the loading process will raise error
        # that "Missing key in checkpoint state_dict: states.config.resume_reference_model_puretorch"
        assert (
            hasattr(self.args, 'resume_reference_model_puretorch')
        ), f"args.resume_reference_model_puretorch is not set, please set it to 'False' if you don't want to resume from checkpoint"
        self.resume_reference_model_puretorch = self.args.resume_reference_model_puretorch
        if self.resume_reference_model_puretorch:
            _, training_states = self.ref_model_engine.load_checkpoint(self.resume_reference_model_puretorch, load_optimizer_states=True)
            client_state = training_states['scalar_state']
            self.logger.info(f"scalar_state: {client_state[0]}")

            # Resume ScalarStates. Overlap the initial states.
            self.ss = self.get_states_cls('scalar').from_pretrained(
                client_state[0],  # directly use the first-rank's scalar_state, since all ranks have the same scalar_state
                rank=self.rank,
                world_size=self.world_size,
                default_rank0_ss=self.args.default_rank0_ss,
                default={'lr': self.args.lr},
            )

            # Reset the consumed_samples_total and epoch_consumed_samples for resuming dataloder
            consumed_steps = client_state[0]['train_steps']
            self.ss.consumed_samples_total = defaultdict(int)
            self.ss.epoch_consumed_samples = defaultdict(int)
            for key in self.all_dataset_keys:
                self.ss.consumed_samples_total[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size)
                self.ss.epoch_consumed_samples[key] = int(self.args.micro_batch_size * consumed_steps * self.dp_size)
            self.logger.info(f"Resumed ScalarStates: {self.ss}")

    
    def build_extra_model(self):
        from hymm.models.reward_models.clip import get_clip as clip
        """ Extra frozen models. """
        args = self.args

        # ====================== Build VAE ========================
        self.logger.info("Building VAE...")
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
            only_encoder=False,
            sample_size=args.get('vae_sample_size'),
        )
        self.clip = clip()
        if args.get('vae_spatial_tiling', False):
            self.vae.enable_spatial_tiling()
        self.vae_generater = torch.Generator(self.device).manual_seed(self.dp_rank)
        if args.get('prerun_vae', torch.backends.cudnn.benchmark):
            self.prerun_vae()

        self.vae.encoder.requires_grad_(False)
        self.vae.decoder.requires_grad_(True)
        self.vae.encoder.eval()
        self.vae.decoder.train()
        self.vae._set_gradient_checkpointing(self.vae.decoder, True)

        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

    
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

        # ===================================== ReDA-related parameters =====================================
        self.pipeline_name = "transfusion_with_logprob"

        self.min_start = args.get('min_start', 15)
        self.min_end = args.get('min_end', 40)
        self.max_disturb_ts_ind = args.get('max_disturb_ts_ind', 2)  # corresponds to `max_i`
        # TODO: check this 
        self.num_infer_timesteps = self.min_end + self.max_disturb_ts_ind
        args.diff_infer_steps = self.num_infer_timesteps  # sampling timesteps

        self.mid_timestep_ind = args.get('mid_timestep_ind', 10)  # TODO: 需要check
        self.policy_sample_range_start = args.get('policy_sample_range_start', 1.0)
        self.policy_sample_range_end = args.get('policy_sample_range_end', 0.8)
        self.policy_sample_steps = args.get('policy_sample_steps', 100)
        self.policy_infer_steps = args.get('policy_infer_steps', 100)
        self.k_steps = args.get('k_steps', 1)
        self.alignment_guidance = args.get('alignment_guidance', 0.0)  # corresponds to `ex_scale`
        self.space_guidance = args.get('space_guidance', 0.0)  # corresponds to `dx_scale`

        # ref-model
        self.ref_mid_timestep_ind = args.get('ref_mid_timestep_ind', 10)  # TODO: 需要check

        self.use_sde_sample = args.get('use_sde_sample', False)

    
    def build_reference_model(self):
        """
        Build the reference model. Note that model must be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[self.args.precision]}
        # assert self.args.get('pretrained_reference_model_ckpt', None) is not None, "pretrained_reference_model_ckpt is not set."
        self.ref_model, _ = build_model(self.args, self.args.get('pretrained_reference_model_ckpt', None), logger=self.logger, **factor_kwargs)
        self.ref_model.requires_grad = True

        self.initialize_puretorch_reference_model_engine()

        assert self.dp_rank_is_correct
        loguru.logger.info(f'when seeding, {self.dp_rank=}')
        set_manual_seed(self.args.global_seed + self.dp_rank)
        

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
    

    def _vae_encode_tensor(self, image, sample_type=None, n_tokens=None):
        # ===================================== prepare diffusion =====================================
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            vae_encode_result_w = self.vae.encode(image)
            if is_torch_tensor(vae_encode_result_w):
                latents_w = vae_encode_result_w
            else:
                latents_w = vae_encode_result_w.latent_dist.sample(generator=self.vae_generater)
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                latents_w.sub_(self.vae.config.shift_factor)
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                latents_w.mul_(self.vae.config.scaling_factor)

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert latents_w.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents_w = latents_w.squeeze(2)

        if sample_type is not None:
            # Choose whether using the same t and x_0 for both win and lose samples, both use the same t
            if sample_type == "sample":
                t, w_x_0, w_x_1 = self.denoiser.sample(latents_w, n_tokens)
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
            # same t for win and lose
            t, w_x_t, w_u_t = self.denoiser.path_sampler.plan(t, w_x_0, w_x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000

            return VAEEncodeOutput(t=t, model_t=model_t, x_0=w_x_0, x_t=w_x_t, u_t=w_u_t, latents=latents_w)

        return VAEEncodeOutput(latents=latents_w)


    def vae_encode(self, images, sample_type=None, n_tokens=None):
        if isinstance(images, list):
            assert n_tokens is None, "n_tokens is not supported for list of images."
            batch_output_w = []
            for image_item in images:
                if is_torch_tensor(image_item) and image_item.ndim == 3:
                    image_item = image_item.unsqueeze(0)

                if isinstance(image_item, list):
                    vae_outputs_w = [
                        self._vae_encode_tensor(
                            image[None].to(self.device), sample_type=sample_type, n_tokens=n_tokens
                        )
                        for image in image_item
                    ]
                    outputs_w = VAEEncodeOutput.cat(vae_outputs_w)

                else:
                    image_item = image_item.to(self.device)
                    outputs_w = self._vae_encode_tensor(image_item, sample_type=sample_type, n_tokens=n_tokens)

                batch_output_w.append(outputs_w)

            batch_output_w = VAEEncodeOutput.build(batch_output_w)

        elif is_torch_tensor(images):
            images = images.to(self.device)
            if images.ndim == 4:
                batch_output_w = self._vae_encode_tensor(images, sample_type=sample_type, n_tokens=n_tokens)

            elif images.ndim == 5:
                vae_outputs_w = [
                    self._vae_encode_tensor(image, sample_type=sample_type, n_tokens=n_tokens)
                    for image in images
                ]
                batch_output_w = VAEEncodeOutput.build(vae_outputs_w)

            else:
                raise ValueError(f"images should have shape [B, C, H, W] or [B, n, C, H, W], got {images.shape}")

        else:
            raise ValueError(f"Unknown images type, expected [list, torch.Tensor], got {type(images)}")

        return batch_output_w


    def vae_decode(self, latents, generator, output_type: Optional[str] = "pt"):
        """
        VAE decode function that returns tensor format compatible with loss_fn.
        
        Args:
            latents: Input latent tensors
            generator: Random generator
            output_type: Output type, should be "pt" for tensor output
            
        Returns:
            torch.Tensor: Decoded image tensor in range [0, 1] ready for reward model
        """
        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        # do_denormalize = [True if self.vae._trans_type=="-11" else False] * image.shape[0]
        # image = self.sampler.pipeline.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        # return image
    
        # Determine if we need to denormalize based on VAE configuration
        # If VAE outputs in [-1, 1] range, we need to convert to [0, 1] for reward model
        if hasattr(self.vae, '_trans_type') and self.vae._trans_type == "-11":
            # Convert from [-1, 1] to [0, 1] range
            image = (image / 2 + 0.5).clamp(0, 1)
        elif output_type == "pt":
            # Ensure image is in [0, 1] range for tensor output
            image = image.clamp(0, 1)
        
        # For reward model compatibility, always return tensor in [0, 1] range
        if output_type == "pt":
            return image
        else:
            # If other output types are needed, process with image_processor
            do_denormalize = [False] * image.shape[0]  # Already normalized above
            return self.sampler.pipeline.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=do_denormalize
            )

    def prepare_model_reda_t2i_inputs(self, bi, batch: Dict, device: Union[int, str], **kwargs):
        input_prompts = batch["text"]
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        # target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            n_samples=batch["n_samples"].to(device),
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
        )
        if "rope_image_info" in batch:
            extra.update(dict(
                rope_image_info=batch["rope_image_info"],
            ))
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
                iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=batch["timestep_scatter_index"].to(device),  # [b, 1]
            ))
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attn_type = self.args.get('t2i_task_kwargs', {}).get('attn_type', 'auto')
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)
        elif attn_type == 'flex':
            assert False, "Flex attention is not supported for REDA now."
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

        # ===================================== Prepare diffusion inputs =====================================
        with torch.inference_mode():
            out_w = self.vae_encode(batch["image"], sample_type="sample")  # win-image encoding
        latents_w = out_w.latents
        raw_img_w = batch["image"].to(self.device)
        image_size = raw_img_w.shape[2:]

        if self.parallel_dims.pp_enabled:
            # Defensive programming
            torch.distributed.broadcast(latents_w, group_src=0, group=self.parallel_dims.pp_group)
            torch.distributed.broadcast(raw_img_w, group_src=0, group=self.parallel_dims.pp_group)
        
        # ==================================== Sample latents (x1) using ref-model =============================
        self.ref_model_engine.eval()
        with torch.no_grad():
            with unwrap_model_for_generation_deepspeed(self.ref_model_engine) as unwrapped_model:
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    self.sampler.pipeline.model = unwrapped_model

                    if self.use_sde_sample:
                        determistic = [False] * self.num_infer_timesteps
                    else:
                        determistic = [True] * self.num_infer_timesteps

                    rng = np.random.default_rng(bi)
                    start_timestep_ind = rng.integers(0, self.max_disturb_ts_ind)
                    out_dict = self.sampler.batch_x2image(
                        [input_prompts],
                        # seed=seeds,
                        verbose=1,
                        task="t2i",
                        sequence_template=self.args.sequence_template,
                        predict_image_shape_token=self.args.predict_image_shape_token,
                        sample_image_size=[image_size],
                        return_only_samples=False,
                        pipeline_kwargs={
                            "kl_weight": 0, 
                            "determistic": determistic,
                            "start_timestep_ind": start_timestep_ind,
                            "x1_for_induce_start_xt": latents_w,
                        },
                    )
        self.ref_model_engine.train()

        # 重要：不然的话容易部分rank在batch_x2image, 部分rank走到train step，ep导致滞后的rank oom
        dist.barrier()

        all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs = out_dict["extra_outputs"]
        # Convert inference tensors to normal tensors that can be used in autograd, we need .clone() to create new tensors from inference tensors
        raw_latents = all_latents[-1].clone().detach()
        # Collect the noise, since we need to use it for policy model in ReDA
        noise_common = all_latents[0].clone().detach()
        del all_latents, all_log_probs, all_ref_prev_latents_mean, model_input_extra_kwargs
        torch.cuda.empty_cache()

        # ==================================== Get policy model inputs =============================
        mid_timestep = self.mid_timestep_ind
        start = mid_timestep + 1
        sigmas_l = torch.linspace(self.policy_sample_range_start, self.policy_sample_range_end, self.policy_sample_steps).to(device)
        sigmas = sigmas_l[self.policy_infer_steps- start]
        latents_xt = sigmas * noise_common + (1.0 - sigmas) * raw_latents
        sigmas = sigmas_l[self.policy_infer_steps - mid_timestep]
        target_gap = sigmas * noise_common
        policy_infer_time_train_kwargs = {
            "latents_xt": latents_xt,
            "latents_x1": raw_latents,
            "target_gap": target_gap,
            "sigmas_l": sigmas_l,
            "sigmas": sigmas,
            "num_inference_steps": self.policy_infer_steps,
            "start": start,
            "mid_timestep": mid_timestep,
            "k": self.k_steps
        }

        return (
            input_prompts,
            raw_img_w,
            latents_w,
            raw_latents,
            noise_common,
            policy_infer_time_train_kwargs,
            batch_size,
            n_tokens,
        )


    def prepare_model_inputs(self, bi, batch: Dict, device: Union[int, str], **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_reda_t2i_inputs(bi, batch, device, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs
    

    def train_step_policy_model(self, bi, batch, **kwargs):
        start1 = time.time()
        (
            input_prompts,
            raw_img_w,
            latents_w,
            raw_latents,
            noise_common,
            policy_infer_time_train_kwargs,
            cur_batch_size,
            n_tokens,
        ) = self.prepare_model_inputs(
            bi,
            batch, 
            self.device,
            step=kwargs.get("step", 0),
        )
        latents_xt = policy_infer_time_train_kwargs["latents_xt"]
        latents_x1 = policy_infer_time_train_kwargs["latents_x1"]
        target_gap = policy_infer_time_train_kwargs["target_gap"]
        sigmas_l = policy_infer_time_train_kwargs["sigmas_l"]
        sigmas = policy_infer_time_train_kwargs["sigmas"]
        num_inference_steps = policy_infer_time_train_kwargs["num_inference_steps"]
        start = policy_infer_time_train_kwargs["start"]
        mid_timestep = policy_infer_time_train_kwargs["mid_timestep"]
        k = policy_infer_time_train_kwargs["k"]

        torch.cuda.synchronize()
        duration1 = time.time() - start1
        start2 = time.time()

        # Only get the model_input_extra_kwargs, do not truely perform inference
        out_dict = self.sampler.batch_x2image(
            [input_prompts],
            # seed=seeds,
            verbose=1,
            task="t2i",
            sequence_template=self.args.sequence_template,
            predict_image_shape_token=self.args.predict_image_shape_token,
            sample_image_size=self.args.sample_image_size,  # TODO: 检查是否需要改成gt sample_size
            return_only_samples=False,
            pipeline_kwargs={
                "kl_weight": 0, 
                "only_get_infer_model_kwargs": True,
            },
        )
        model_input_extra_kwargs, generator = out_dict["extra_outputs"]
        model_input_extra_kwargs["return_loss"] = False

        ############################ Policy Model Training and Loss Closure ############################
        def _loss_closure(model_output, moe_loss):
            pred = model_output["diffusion_prediction"]
            pipeline = self.sampler.pipeline

            pred = pred.to(dtype=torch.float32)

            # perform guidance
            if pipeline.do_classifier_free_guidance:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = pred_uncond + pipeline.guidance_scale * (pred_cond - pred_uncond)

            if pipeline.do_classifier_free_guidance and pipeline.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=pipeline.guidance_rescale)
            
            latents = latents_xt.to(torch.float32)
            dt = sigmas_l[num_inference_steps - start + k + i * k] - sigmas_l[num_inference_steps - start + i * k]
            latents = latents + pred * dt

            latents = (latents-target_gap)/(1-sigmas)
            expand_temporal_dim = False

            # vae decode
            self.vae.decoder.train()
            self.vae.decoder.requires_grad_(True)
            image = self.vae_decode(latents, generator=generator)
            raw_image = self.vae_decode(raw_latents, generator=generator)
            self.vae.decoder.eval()

            loss = reda_policy_loss_fn(
                image,
                raw_img_w,
                raw_image,
                sigmas,
                self.clip,
                self.alignment_guidance,
                self.space_guidance
            )

            return loss, {"loss": loss.detach()}

        self.model_engine.register_loss_closure(_loss_closure)
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            pipeline = self.sampler.pipeline
            cfg_factor = 1
            if pipeline.do_classifier_free_guidance:
                cfg_factor = 2

            for i, t in enumerate(sigmas_l[(num_inference_steps - start):(num_inference_steps - mid_timestep):k]):
                # TODO: 目前只支持迭代一次，原始的reda代码的超参设置目前也是一次，之后再支持
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_xt] * cfg_factor)
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                t_expand = t.repeat(latent_model_input.shape[0]) * 1000  # TODO: 1000 参数化

                self.model_engine(
                    x_t=latent_model_input,
                    t=t_expand,
                    **model_input_extra_kwargs,
                )
                loss_dict = self.model_engine.get_cached_result('loss_dict', merge_op='mean')
                loss = loss_dict['loss']

        # for debug
        self.logger.info(f"loss_dict: {loss_dict}")
    
        torch.cuda.synchronize()
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        ref_train_input = (num_inference_steps, noise_common, sigmas_l, raw_latents, k, model_input_extra_kwargs, self.device)

        return loss_dict, cur_batch_size, ref_train_input, n_tokens, times


    def train_step_reference_model(self, ref_train_input, **kwargs):

        num_inference_steps, noise_common, sigmas_l, raw_latents, k, model_input_extra_kwargs, device = ref_train_input
        ref_mid_timestep = self.ref_mid_timestep_ind
        start_ref = ref_mid_timestep + 1
        reshuffle = 0.9  # TODO

        noise_ref = noise_common * reshuffle + (1 - reshuffle**2)**0.5 * torch.randn_like(noise_common)
        sigmas = sigmas_l[num_inference_steps - start_ref]
        latents_xt = sigmas * noise_ref + (1.0 - sigmas) * raw_latents

        latents_ref = latents_xt.detach().requires_grad_(True)

        cummulate_velosity = []
        self.model_engine.eval()
        with torch.no_grad():
            with unwrap_model_for_generation_deepspeed(self.model_engine) as unwrapped_model:
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                    cfg_factor = 1
                    if self.sampler.pipeline.do_classifier_free_guidance:
                        cfg_factor = 2
                    latent_model_input = torch.cat([latents_ref] * cfg_factor)

                    for i, t in enumerate(sigmas_l[(num_inference_steps - start_ref):(num_inference_steps - ref_mid_timestep):k]):
                        # TODO: 目前只支持迭代一次，原始的reda代码的超参设置目前也是一次，之后再支持
                        latent_model_input = self.sampler.pipeline.scheduler.scale_model_input(latent_model_input, t)
                        mid_t_expand = t.repeat(latent_model_input.shape[0]) * 1000  # TODO: 1000 参数化
                        pred = unwrapped_model(
                            x_t=latent_model_input,
                            t=mid_t_expand,
                            **model_input_extra_kwargs,
                        )["diffusion_prediction"]

                        # perform guidance
                        if self.sampler.pipeline.do_classifier_free_guidance:
                            pred_cond, pred_uncond = pred.chunk(2)
                            pred = pred_uncond + self.sampler.pipeline.guidance_scale * (pred_cond - pred_uncond)

                        if self.sampler.pipeline.do_classifier_free_guidance and self.sampler.pipeline.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.sampler.pipeline.guidance_rescale)
                        
                        dt = sigmas_l[num_inference_steps - start_ref + k + i * k] - sigmas_l[num_inference_steps - start_ref + i * k]
                        latent_model_input = latent_model_input + torch.cat([pred] * 2) * dt
                        mid_t_expand = mid_t_expand + dt.repeat(latent_model_input.shape[0]) * 1000
                        cummulate_velosity.append(dt)
        
        self.model_engine.train()
        # self.model_engine.cpu()
        # self.ref_model_engine.cuda()

        def _loss_closure(model_output, moe_loss):
            pred = model_output["diffusion_prediction"]
            pipeline = self.sampler.pipeline

            pred = pred.to(dtype=torch.float32)
            loss_ref = F.mse_loss(pred, torch.stack(cummulate_velosity, dim=0).mean(0)) # DMv8g
            return loss_ref, {"loss_ref": loss_ref.detach()}
        
        self.ref_model_engine.register_loss_closure(_loss_closure)
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            pipeline = self.sampler.pipeline
            cfg_factor = 1
            if pipeline.do_classifier_free_guidance:
                cfg_factor = 2

            self.ref_model_engine(
                x_t=latent_model_input,
                t=mid_t_expand,
                **model_input_extra_kwargs,
            )
            loss_dict = self.ref_model_engine.get_cached_result('loss_dict', merge_op='mean')
            loss = loss_dict['loss_ref']
        
        return loss_dict


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
                    
                    loss_dict, batch_size, ref_train_input, n_tokens, forward_times = self.train_step_policy_model(bi, batch, step=self.ss.update_steps)
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

                    
                    # Update reference model
                    ref_loss_dict = self.train_step_reference_model(ref_train_input, step=self.ss.update_steps)
                    ref_loss = ref_loss_dict["loss_ref"].mean()
                    self.ref_model_engine.backward(ref_loss)
                    self.ref_model_engine.step()
                    for k, v in ref_loss_dict.items():
                        cs.running_sub_loss_dict[k] += v.mean().item()
                        cs.running_sub_step_dict[k] += 1


                    if self.ss.update_steps >= args.max_training_steps:
                        # Enter stopping routine if max steps reached after this step.
                        finished = True

                    # Update EMA model at the step of main model parameters update.
                    if args.use_ema and is_update_step:
                        self.ema.update(self.model_engine.module)

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
 