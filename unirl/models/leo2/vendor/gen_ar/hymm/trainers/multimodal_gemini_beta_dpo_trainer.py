import gc
import os
import time
import json
import inspect
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Union, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import deepspeed
import torch
import torch.distributed as dist
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
# from ..models.reward_models.text_ocr_groundingpaddle import TextOCRGroundingPaddle
from ..models.reward_models.text_ocr_groundingQwenVL import TextOCRGroundingQwenVL
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.rl_t2i_loader import RLTextImageArrowStream
from .multimodal_gemini_beta_trainer import MultiModalGeminiBetaTrainer
from ..utils.torch_utils import (
    build_optimizer,
    set_manual_seed,
    set_worker_seed_builder,
    profiler_context,
    is_torch_tensor,
    PRECISION_TO_TYPE,
    NAME_TO_SHARDING_STRATEGY,
)
from ..utils.parallel_states import (
    initialize_sequence_parallel_state,
    initialize_teacher_student_parallel_state,
    nccl_info,
    is_teacher_group,
    is_student_group
)
from ..utils.communication import broadcast_within_ts_unit, broadcast, all_gather_ts
from ..constants import SCORE_METRICS, SAMPLE_METRICS, C_SCALE
from ..core.global_vars import get_nccl_timeout
from ..data_kits.csv_dataset import decode_csv_file
from ..ds_config import get_deepspeed_config
from ..utils import lr_schedules
from ..utils.file_utils import (
    safe_dir,
    get_experiment_max_number,
    empty_logger,
    dump_configs,
    dump_codes,
    resolve_resume_path,
    logger_filter,
    dict_repr,
)
from ..utils.fsdp_wrapper import FSDPEngine
from ..utils.helpers import default, get_obj_from_str, to_2tuple
from ..utils.torch_distributions import gather_tensor
import torchvision.transforms as transforms
from ..models.autoencoders import load_vae
from ..models.tokenizers import TokenizerWrapper
from ..diffusion import load_denoiser
import random

gc.set_threshold(7000, 100, 100)


def hy_text_length_getter(index_manager, ind):
    return index_manager.get_attribute(ind, "hy_ids_length")


def flow_dpo_loss_fn(v_w_pred, v_l_pred, v_w_ref_pred, v_l_ref_pred, v_w_target, v_l_target, beta=500, pos_term_weight=0.0, hinge_threshold=0):
    reduce_dims = list(range(1, v_w_pred.ndim))

    model_w_err = (v_w_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    model_l_err = (v_l_pred - v_l_target).pow(2).mean(dim=reduce_dims)
    ref_w_err = (v_w_ref_pred - v_w_target).pow(2).mean(dim=reduce_dims)
    ref_l_err = (v_l_ref_pred - v_l_target).pow(2).mean(dim=reduce_dims)

    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err
    if pos_term_weight > 0:
        pos_term = pos_term_weight * torch.clamp(ref_w_err - model_w_err, max=hinge_threshold)
        inside_term = -0.5 * beta * (w_diff - l_diff - pos_term)
    else:
        inside_term = -0.5 * beta * (w_diff - l_diff)
    loss = - torch.nn.functional.logsigmoid(inside_term)

    extra_info = {
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


@tp_sp_decorator
class MultiModalGeminiBetaDPOTrainer(MultiModalGeminiBetaTrainer):
    def __init__(self, args, all_dataset_keys=None):
        self.task_init(args, all_dataset_keys=all_dataset_keys)
        self.args = args

        self.dataset = None
        self.data_sampler = None
        self.dataloader = None

        self.resume_path = None

        self.sample_evaluator = None
        self.loss_evaluator = None
        self.score_evaluator = None

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.init_env()

        # Keep the order: dataloader -> model & optimizer -> extra model -> evaluator
        self.build_dataloader()
        
        if is_student_group():
            self.build_model_and_optimizer(process_group=nccl_info.ts_group if self.ref_policy_parallel else None)
        if is_teacher_group():
            self.build_reference_model()
        self.global_batch_size = (
            args.global_batch_size
            if args.global_batch_size is not None
            else self.micro_batch_size * self.dp_size * self.grad_accu_steps
        )
        self.build_extra_model()
        self.build_data_iterator()
        self.build_evaluator()
        if self.args.get("online", False):
            self.sampler = self.get_sampler()
            random.seed(self.args.get("global_seed", 0) + self.dp_rank)
        dist.barrier()
        print(f"Build models, dataloader and optimizer done. rank: {self.rank}, dp_rank: {self.dp_rank}, dp_size: {self.dp_size}")
    
    def build_extra_model(self):
        """ Extra frozen models. """
        args = self.args

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

        if self.dataset is not None:
            self.tkwrapper = self.dataset.tokenizer
        elif hasattr(self, 'dataset_dict'):
            self.tkwrapper = list(self.dataset_dict.values())[0].tokenizer
        else:
            self.tkwrapper = TokenizerWrapper(args.tokenizer_name, self.logger)

        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

        self.use_3d_rope = args.get('rope_type', 'default') in ['3d', '3d-interleave']
        if self.args.get("text", {}).get("ocr", {}).get("enable", False):
            self.text_reward_model = TextOCRGroundingQwenVL()

    def init_distributed_env(self):
        args = self.args
        nccl_timeout = get_nccl_timeout()
        if args.launcher == "deepspeed":
            deepspeed.init_distributed(timeout=nccl_timeout)
        else:
            dist.init_process_group(backend="nccl", timeout=nccl_timeout)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # Set current device for the current process, otherwise dist.barrier() will occupy more memory in rank 0.
        if args.launcher == "deepspeed":
            self.device = self.local_rank = args.local_rank
        else:
            self.device = self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.device)

        self.sequence_parallel_size = args.get('sequence_parallel_size', 1)
        initialize_sequence_parallel_state(self.sequence_parallel_size, self.rank, self.world_size)
        self.ref_policy_parallel = args.get('ref_policy_parallel', True)
        if self.ref_policy_parallel:
            initialize_teacher_student_parallel_state(self.sequence_parallel_size, self.rank, self.world_size)
        
        self.dp_rank = nccl_info.ts_unit_group_id if self.ref_policy_parallel else nccl_info.group_id
        self.dp_size = self.world_size // nccl_info.ts_unit_size if self.ref_policy_parallel else self.world_size // nccl_info.sp_size
    
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
    
    def build_reference_model(self):
        """
        Build the reference model. Note that model must be in `.eval()` mode.
        """
        self.logger.info("Building reference model...")
        factor_kwargs = {"device": "cpu", "dtype": PRECISION_TO_TYPE[self.args.precision]}
        assert self.args.get('pretrained_reference_model_ckpt', None) is not None, "pretrained_reference_model_ckpt is not set."
        self.ref_model, _ = build_model(self.args, self.args.pretrained_reference_model_ckpt, logger=self.logger, **factor_kwargs)
        self.ref_model.requires_grad = False
        self.ref_model = self.ref_model.to(self.device)
        self.ref_model.eval()
        print(f"Reference model built on device: {self.device}")

        # After model initialization, we set different seed for each process.
        set_manual_seed(self.args.global_seed + self.dp_rank)

        # Needed for save_checkpoint, which will be all_gathered for all ranks.
        scalar_state = {
            'lr': self.args.lr,
        }
        self.ss = self.get_states_cls('scalar')(**scalar_state)

        # Mixed precision training.
        self.target_dtype = PRECISION_TO_TYPE[self.args.autocast_dtype]
        self.autocast_enabled = self.args.autocast_dtype != torch.float32

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

    def _vae_encode_tensor(self, image, lose_image, sample_type=None, n_tokens=None):
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

    def vae_encode(self, images, lose_images, sample_type=None, n_tokens=None):
        # TODO (yutaocui): Now only support t2i task, to support ti2i (src_image_num_list)
        if isinstance(images, list):
            assert n_tokens is None, "n_tokens is not supported for list of images."
            batch_output_w, batch_output_l = [], []
            for image_item, lose_image_item in zip(images, lose_images):
                if is_torch_tensor(image_item) and image_item.ndim == 3:
                    image_item = image_item.unsqueeze(0)
                    lose_image_item = lose_image_item.unsqueeze(0)

                if isinstance(image_item, list):
                    vae_outputs_w, vae_outputs_l = [
                        self._vae_encode_tensor(
                            image[None].to(self.device), lose_image[None].to(self.device), sample_type=sample_type, n_tokens=n_tokens
                        )
                        for image, lose_image in zip(image_item, lose_image_item)
                    ]
                    outputs_w = VAEEncodeOutput.cat(vae_outputs_w)
                    outputs_l = VAEEncodeOutput.cat(vae_outputs_l)

                else:
                    image_item = image_item.to(self.device)
                    lose_image_item = lose_image_item.to(self.device)
                    outputs_w, outputs_l = self._vae_encode_tensor(image_item, lose_image_item, sample_type=sample_type, n_tokens=n_tokens)

                batch_output_w.append(outputs_w)
                batch_output_l.append(outputs_l)

            batch_output_w = VAEEncodeOutput.build(batch_output_w)
            batch_output_l = VAEEncodeOutput.build(batch_output_l)

        elif is_torch_tensor(images):
            images = images.to(self.device)
            lose_images = lose_images.to(self.device)
            if images.ndim == 4:
                batch_output_w, batch_output_l = self._vae_encode_tensor(images, lose_images, sample_type=sample_type, n_tokens=n_tokens)

            elif images.ndim == 5:
                vae_outputs_w, vae_outputs_l = [
                    self._vae_encode_tensor(image, lose_image, sample_type=sample_type, n_tokens=n_tokens)
                    for image, lose_image in zip(images, lose_images)
                ]
                batch_output_w = VAEEncodeOutput.build(vae_outputs_w)
                batch_output_l = VAEEncodeOutput.build(vae_outputs_l)

            else:
                raise ValueError(f"images should have shape [B, C, H, W] or [B, n, C, H, W], got {images.shape}")

        else:
            raise ValueError(f"Unknown images type, expected [list, torch.Tensor], got {type(images)}")

        return batch_output_w, batch_output_l

    def prepare_model_dpo_t2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # concat win-image and lose-image
        tokens = tokens.repeat(2, 1)
        target_tokens = target_tokens.repeat(2, 1)
        text_mask = text_mask.repeat(2, 1)
        image_mask = image_mask.repeat(2, 1)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
        )
        if "iw_ih_scatter_index" in batch:
            extra.update(dict(
                iw_ih_scatter_index=batch["iw_ih_scatter_index"].repeat(2, 1).to(device),        # [b, 2]
                iw_ih_scatter_src=batch["iw_ih_scatter_src"].repeat(2, 1).to(device),            # [b, 2]
            ))
        if "timestep_scatter_index" in batch:
            extra.update(dict(
                timestep_scatter_index=batch["timestep_scatter_index"].repeat(2, 1).to(device),  # [b, 1]
            ))
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attn_type = self.args.get('t2i_task_kwargs', {}).get('attn_type', 'auto')
        if attn_type == 'auto':
            attention_mask = batch["attention_mask"].to(device)
            attention_mask = attention_mask.repeat(2, 1, 1)  # win-image and lose-image attention mask
        elif attn_type == 'flex':
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
        rewards = None
        if self.args.get("online", False):
            # use in domain data as win/lose
            out_domain_percent = self.args.get("out_domain_percent", 0.2)
            if random.random() >= out_domain_percent:
                if self.args.get("text", {}).get("ocr", {}).get("enable", False):
                    prompt = batch["text"]
                    num_samples = self.args["text"]["ocr"].get("num_samples", 2)
                    prompt = [text for text in prompt for _ in range(num_samples)]
                    sample = self.iter_sample(
                        prompt=prompt,
                        seed=[self.args.global_seed + i + kwargs["step"] for i in range(len(prompt))],
                        bsz=self.args["text"]["ocr"].get("bsz", 1),
                        output_type="pil",
                    )
                    # ocr reward
                    gt = [self.text_reward_model.parse_gt(text) for text in prompt]
                    eval_res = self.text_reward_model.eval(
                        sample,
                        gt,
                        url=self.args["text"]["ocr"].get("url", None),
                        max_workers=self.args["text"]["ocr"].get("max_workers", 8),
                        show_progress=True,
                        if_split_by_character=self.args["text"]["ocr"].get("if_split_by_character", False),
                    )

                    # rank by reward
                    rewards = [eval_res[i][self.args["text"]["ocr"].get("reward_metric", "f1_score")] for i in range(len(eval_res))]
                    # gt = [eval_res[i]["gt_ocr_split1"] for i in range(len(eval_res))]
                    # pred = [eval_res[i]["pred_ocr_split1"] for i in range(len(eval_res))]

                    # visualize images and rewards
                    if self.visualize_reward and self.ss.update_steps % self.visualize_every == 0:
                        visualize_dir = os.path.join(self.exp_dir, "reward_visualizations")
                        os.makedirs(visualize_dir, exist_ok=True)
                        plt.figure(figsize=(8, 8))
                        plt.imshow(sample[0])
                        plt.title(f"Reward Score: {rewards[0]:.4f}")
                        plt.axis('off')
                    
                        save_path = os.path.join(visualize_dir, f"step_{self.ss.update_steps}_sample_{self.rank}_reward_{rewards[0]:.4f}.png")
                        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close()

                    win_images = []
                    lose_images = []
                    for i in range(0, len(sample), num_samples):
                        cur_sample = sample[i:i+num_samples]
                        cur_reward = rewards[i:i+num_samples]
                        sorted_indices = sorted(range(len(cur_reward)), key=lambda k: cur_reward[k], reverse=True)

                        cur_sample = [self.pil_image_to_tensor(x).to(self.device) for x in cur_sample]
                        win_images.append(cur_sample[sorted_indices[0]])
                        lose_images.append(cur_sample[sorted_indices[-1]])

                    batch["image"] = torch.stack(win_images, dim=0)
                    batch["lose_image"] = torch.stack(lose_images, dim=0)
                else:
                    # Sample image as lose
                    # use kv_cache for speed up
                    self.args.kv_cache = True
                    prompt = batch["text"]
                    out_cur = self.sampler.predict(
                        prompt=prompt,
                        guidance_scale=3.5,
                        diff_infer_steps=50,
                        flow_shift=3.0,
                        size=1024,
                        output_type="latent",
                    )
                    self.args.kv_cache = False
                    batch["lose_image"] = out_cur["samples"]
            else:
                rewards = batch["image"].shape[0] * self.args["text"]["ocr"].get("num_samples", 2) * [0.0]
            rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        out_w, out_l = self.vae_encode(batch["image"], batch["lose_image"], sample_type="sample")
        t_w, model_t_w, x_0_w, x_t_w, u_t_w = out_w.t, out_w.model_t, out_w.x_0, out_w.x_t, out_w.u_t
        t_l, model_t_l, x_0_l, x_t_l, u_t_l = out_l.t, out_l.model_t, out_l.x_0, out_l.x_t, out_l.u_t
        # Broadcast x_t/u_t within the ref-policy group, further ensuring consistency
        if self.ref_policy_parallel:
            broadcast_within_ts_unit(x_t_w)
            broadcast_within_ts_unit(x_t_l)
            broadcast_within_ts_unit(u_t_w)
            broadcast_within_ts_unit(u_t_l)
            broadcast_within_ts_unit(model_t_w)
            broadcast_within_ts_unit(model_t_l)
        elif self.sequence_parallel_size > 1:
            broadcast(x_t_w)
            broadcast(x_t_l)
            broadcast(u_t_w)
            broadcast(u_t_l)
            broadcast(model_t_w)
            broadcast(model_t_l)
        
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                                      # [b, seqlen]
            x_t=torch.cat([x_t_w, x_t_l], dim=0),            # [b, c, h, w]
            t=torch.cat([model_t_w, model_t_l], dim=0),      # [b]
            target=target_tokens,                            # [b, seqlen]
            attention_mask=attention_mask,                   # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type=batch['data_type'][0],                 # For loss
            return_loss=False,                               # Do not return diffusion loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, u_t_w, u_t_l, batch_size, n_tokens, rewards
    
    def iter_sample(self, prompt, seed, bsz, **kwargs):
        self.args.kv_cache = True
        sample = []
        for i in range(0, len(prompt), bsz):
            cur_prompt = prompt[i:i+bsz]
            cur_seed = seed[i:i+bsz]
            out_cur = self.sampler.predict(
                prompt=cur_prompt,
                guidance_scale=kwargs.get("guidance_scale", 3.5),
                diff_infer_steps=kwargs.get("diff_infer_steps", 50),
                flow_shift=kwargs.get("flow_shift", 3.0),
                size=kwargs.get("size", 1024),
                output_type=kwargs.get("output_type", "pil"),
                seed=cur_seed,
            )
            sample.extend(out_cur["samples"])
        self.args.kv_cache = False
        return sample

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        if batch["dtype"][0] == "t2i":
            inputs = self.prepare_model_dpo_t2i_inputs(batch, device, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected {self.all_dataset_keys}, got {batch['dtype']}")
        return inputs

    def train_step(self, batch, **kwargs):
        start1 = time.time()
        model_input_kwargs, u_t_w, u_t_l, cur_batch_size, n_tokens, rewards = self.prepare_model_inputs(
            batch, 
            self.device,
            step=kwargs.get("step", 0),
        )
        torch.cuda.synchronize()
        duration1 = time.time() - start1

        mbs = model_input_kwargs["x_t"].shape[0]
        cur_group_r = nccl_info.rank_within_group

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            ############################ Policy Model ############################
            if is_student_group():
                output = self.model_engine(**model_input_kwargs)
                model_pred = output["diffusion_prediction"]
            if self.ref_policy_parallel and is_teacher_group():
                model_pred = torch.zeros_like(model_input_kwargs["x_t"], requires_grad=True, dtype=self.target_dtype)
            if self.ref_policy_parallel:            
                ####### communication between ref and policy predictions #######
                model_pred_gathered = all_gather_ts(model_pred, dim=0)
                model_pred = model_pred_gathered[:(model_pred_gathered.shape[0] // 2)][cur_group_r*mbs:(cur_group_r+1)*mbs]
            
            ############################ Reference Model ############################
            if is_teacher_group():
                with torch.inference_mode():
                    output = self.ref_model(**model_input_kwargs)
                    ref_model_pred = output["diffusion_prediction"]
            if self.ref_policy_parallel and is_student_group():
                ref_model_pred = torch.zeros_like(model_input_kwargs["x_t"], dtype=self.target_dtype)
            if self.ref_policy_parallel:
                ####### communication between ref and policy predictions #######
                ref_model_pred_gathered = all_gather_ts(ref_model_pred, dim=0)
                ref_model_pred = ref_model_pred_gathered[(ref_model_pred_gathered.shape[0] // 2):][cur_group_r*mbs:(cur_group_r+1)*mbs]

        ############################ Loss Calculation ############################
        if is_student_group():
            v_w_pred, v_l_pred = model_pred.chunk(2, dim=0)
            v_w_ref_pred, v_l_ref_pred = ref_model_pred.chunk(2, dim=0)

            pos_term_weight = self.args.get('pos_term_weight', 0.0)
            loss, extra_info = flow_dpo_loss_fn(v_w_pred, v_l_pred, v_w_ref_pred, v_l_ref_pred, u_t_w, u_t_l, pos_term_weight=pos_term_weight)

        if self.ref_policy_parallel and is_teacher_group():
            loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            extra_info = {}
        
        loss_dict = {
            "loss": loss,
            **extra_info,
        }
        if rewards is not None:
            loss_dict["reward_dummyloss"] = rewards
    
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
            if is_student_group():
                ss.lr = self.optimizer.param_groups[0]["lr"]
            elif is_teacher_group():
                ss.lr = 0

        return is_update_step

    def train_loop(self):
        args = self.args
        if is_student_group():
            self.model_engine.train()
        elif is_teacher_group():
            self.ref_model.eval()
        self.ss.current_run_update_steps = 0

        if is_student_group() and args.init_save:
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

                    loss_dict, batch_size, n_tokens, forward_times = self.train_step(batch, step=self.ss.update_steps)
                    times.update(forward_times)

                    backward_start = time.time()
                    loss = loss_dict["loss"].mean()
                    for k, v in loss_dict.items():
                        if "loss" in k and k != "loss":
                            cs.running_sub_loss_dict[k] += v.mean().item()
                            cs.running_sub_step_dict[k] += 1

                    if is_student_group():
                        self.model_engine.backward(loss)
                    torch.cuda.synchronize()
                    times['backward'] = time.time() - backward_start

                    is_update_step = self.update_train_states(self.ss, cs, batch, batch_size, n_tokens, loss.item())

                    if is_student_group():
                        if args.skip_nan_grad and hasattr(self.model_engine.optimizer, "scaled_global_norm"):
                            scaled_grad_norm = self.model_engine.optimizer.scaled_global_norm()     
                            if torch.any(torch.isnan(scaled_grad_norm)):
                                nan_grad_count += 1
                                self.logger.info(f"Step {self.ss.update_steps:07d} grad norm is nan, skipping step. Total nan grad count: {nan_grad_count}.")
                                self.model_engine.optimizer.zero_grad()

                    # Update model parameters at the boundary of gradient accumulation.
                    update_start = time.time()
                    if is_student_group():
                        # Get the lr before optimizer.step()
                        lrs = [group["lr"] for group in self.optimizer.param_groups]
                        self.model_engine.step(lr_kwargs={"last_batch_iteration": self.lr_helper(self.ss.update_steps + 1)})
                    else:
                        lrs = [0]
                    torch.cuda.synchronize()
                    times['update'] = time.time() - update_start

                    if self.ss.update_steps >= args.max_training_steps:
                        # Enter stopping routine if max steps reached after this step.
                        finished = True

                    # Update EMA model at the step of main model parameters update.
                    if args.use_ema and is_update_step and is_student_group():
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

                        if is_student_group():
                            grad_norm = self.model_engine.get_global_grad_norm()
                        elif is_teacher_group():
                            grad_norm = 0

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
                        if is_student_group() and self.model_engine.monitor.enabled and self.rank == 0:
                            self.model_engine.monitor.write_events(summary_events)

                        # Reset monitoring variables:
                        cs.reset()
                        start_time = time.time()

                    if is_student_group():  
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

    def before_train(self):
        args = self.args

        # ============================= Print key info =============================
        print(f"[{self.rank}] Worker ready.")

        try:
            iters_per_epoch = len(self.dataloader) // self.grad_accu_steps
        except NotImplementedError:
            iters_per_epoch = 0
        model = self.model if is_student_group() else self.ref_model
        # `params_count` is used for all rank logging during training
        if hasattr(model, "params_count"):
            self.params_count = model.params_count()
        else:
            self.params_count = {
                "total": sum(p.numel() for p in model.parameters()),
                "attn+mlp": sum(p.numel() for name, p in model.named_parameters() if "attn" in name or "mlp" in name),                
            }

        if is_student_group():
            self.logger.info("****************************** Running training ******************************")
            self.logger.info(f"  Number GPUs:               {self.world_size}")
            if hasattr(self, 'dataset') and self.dataset is not None:
                self.logger.info(f"  Training samples(total):   {len(self.dataset):,}({self.dataset.total_length:,})")
            elif hasattr(self, 'dataset_dict'):
                for k, v in self.dataset_dict.items():
                    self.logger.info(f"  Training samples:          {k} = {len(v):,}({v.total_length:,})")
            for k, v in self.params_count.items():
                self.logger.info(f"  Number {k} parameters:   {v:,}")
            self.logger.info(f"  Number trainable params:   {self.num_trainable_params:,}")
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info(f"  Updates per epoch:         {iters_per_epoch:,}" + ("(unknown)" if iters_per_epoch == 0 else ""))
            self.logger.info(f"  Batch size per device:     {self.micro_batch_size}")
            self.logger.info(f"  Batch size all device:     {self.global_batch_size}")
            self.logger.info(f"  Gradient Accu steps:       {self.grad_accu_steps}")
            self.logger.info(f"  Training epochs:           {self.ss.epoch}/{args.max_epochs}")
            self.logger.info(f"  Training total steps:      {self.ss.update_steps:,}/{args.max_training_steps:,}")
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info(f"  Main model precision:      {args.precision}")
            self.logger.info(f"  Autocast precision:      {args.autocast_dtype}")
            self.logger.info(f"  Using EMA model:           {args.use_ema}")
            if args.use_ema:
                self.logger.info(f"      Using Distributed EMA: {args.distributed_ema}")
                self.logger.info(f"      EMA precision:         {args.ema_precision}")
                self.logger.info(f"      EMA decay:             {self.ema.decay if args.use_ema else None}")
                self.logger.info(f"      EMA warmup power:      {self.ema.power if args.use_ema else None}")
            self.logger.info("------------------------------------------------------------------------------")
            self.logger.info(f"  Media Tokenizer:           {args.vae_type} ({args.vae_precision})")
            self.logger.info(f"  VAE autocast precision:           {args.vae_autocast_dtype}")
            if hasattr(self, 'vae') and hasattr(self.vae, 'codebook_size'):
                self.logger.info(f"      Codebook size:         {self.vae.codebook_size}")
            if hasattr(self, 'vae') and hasattr(self.vae, 'downsample_factor'):
                self.logger.info(f"      Downsample factor:     {self.vae.downsample_factor}")
            self.logger.info("------------------------------------------------------------------------------")
            if self.resume_path:
                self.logger.info(f"  Resume from:               {self.resume_path}")
            self.logger.info(f"  Experiment directory:      {self.exp_dir}")
            self.logger.info("*******************************************************************************")
                