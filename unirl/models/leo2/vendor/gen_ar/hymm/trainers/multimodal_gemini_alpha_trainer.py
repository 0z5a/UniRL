import gc
import json
import os
import math
from functools import partial
from typing import Dict, Union

import torch
from index_kits.sampler import DistributedSampler, MultiIndexV2
from torch.utils.data import DataLoader
try:
    from torch.nn.attention.flex_attention import BlockMask
except:
    BlockMask = None

from .base_trainer import BaseTrainer
from .helpers import MultiModalScalarStates, MultiModalCycleStates
from ..constants import C_SCALE, VISION_ENCODER_META_INFO
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.mmu_loader import MultiModalUnderstandingArrowStream
from ..data_kits.text_image_transfusion_loader import TransfusionTextImageArrowStream
from ..data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from ..diffusion import load_denoiser, load_scheduler
from ..models.autoencoders import load_vae, VAEEncodeOutput
from ..models.tokenizers import TokenizerWrapper
from ..samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler
from ..utils.file_utils import safe_dir
from ..utils.helpers import default, to_2tuple
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE, is_torch_tensor

gc.set_threshold(7000, 100, 100)


def gemini_alpha_mm_pretrain_length_getter(index_manager: MultiIndexV2, ind):
    # hy_text_930m: hy_ids_length
    # gemini_alpha_caption: caption_v2_hy_ids_length
    arrow_name = index_manager.get_arrow_file(ind)
    if 't2i_caption' in arrow_name:
        len_col = "caption_v2_hy_ids_length"
        length_list = index_manager.get_attribute(ind, len_col, shadow=len_col)
        length = sum(length_list[1:])
    else:
        len_col = "hy_ids_length"
        length = index_manager.get_attribute(ind, len_col)
    return length


def hy_text_length_getter(index_manager, ind):
    return index_manager.get_attribute(ind, "hy_ids_length")


def recaption_length_getter(index_manager, ind):
    len_col = "caption_v2_hy_ids_length"
    length_list = index_manager.get_attribute(ind, len_col, shadow=len_col)
    length = sum(length_list[1:])
    return length


class GeminiTrainerAlphaMultiModal(BaseTrainer):
    def __init__(self, args, all_dataset_keys=None):
        self.task_init(args, all_dataset_keys)
        super().__init__(args)

    def get_pp_rank(self):
        return 0

    def get_pp_world_size(self):
        return 1

    def task_init(self, args, all_dataset_keys=None):
        if isinstance(args.sampling_probs, list):
            # Keys' order is important for CombinedBatchIterator and must be consistent with sampling_probs.
            self.all_dataset_keys = default(all_dataset_keys, ['t2i', 'lm_text', 'lm_recap', 'mmu'])
            self.sampling_probs_dict = {key: args.sampling_probs[i] for i, key in enumerate(self.all_dataset_keys)}
            # TODO: deprecated
        elif isinstance(args.sampling_probs, str):
            self.sampling_probs_dict = json.loads(args.sampling_probs)
            self.all_dataset_keys = sorted(list(self.sampling_probs_dict.keys()))
        else:
            raise ValueError(f"Invalid sampling_probs type: {type(args.sampling_probs)}")
        if 't2i' in self.sampling_probs_dict or 'mmu' in self.sampling_probs_dict:
            assert args.add_iw_ih_token is True
        if 't2i' in self.sampling_probs_dict:
            assert args.add_timestep_token is True

        # Define what dummy token are incurred by each task.
        self.dummy_to_tasks = dict(
            t2i={"t2i", "inpainting", "editing_OmniEdit", "editing_v2f", "subject_driven", "interleave", "editing",
                 "face_id_clip"},
            mmu={"mmu", "mmu_interleave", "face_id_clip"},
            face={"face_id"},
        )
        # Set the number of dummy tokens
        self.dummy_dict = dict(
            t2i=(
                1 + (2 if args.add_iw_ih_token else 0) + (1 if args.add_timestep_token else 0)
                if any(task in self.sampling_probs_dict for task in self.dummy_to_tasks['t2i'])
                else 0
            ),
            mmu=(
                1 + (2 if args.add_iw_ih_token else 0)
                if any(task in self.sampling_probs_dict for task in self.dummy_to_tasks['mmu'])
                else 0
            ),
            face=(
                self.args.get('face_aligner_num_queries', 16)
                if any(task in self.sampling_probs_dict for task in self.dummy_to_tasks['face'])
                else 0
            )
        )
        for key, value in self.dummy_dict.items():
            self.logger.info(f"Using {value} {key} dummy tokens.")

    def dataloader_preliminary_setup(self):
        args = self.args
        # Determine the dataloader allocation if combined_iterator_sampling_mode is not `random`.
        combined_iterator_sampling_mode = args.get('combined_iterator_sampling_mode', 'random')
        # The fixed key of current rank.
        self.cur_key = None
        # The list of ranks that are assigned to the same dataset with current rank.
        self.cur_key_group = None
        self.dataset_num_replicas = {key: self.dp_size for key in self.all_dataset_keys}
        self.dataset_rank = {key: self.dp_rank for key in self.all_dataset_keys}

        if combined_iterator_sampling_mode == "fixed":
            # Dataloaders are assigned to fixed ranks, following the sampling_probs as near as possible.
            # For example, if sampling_probs = [t2i: 0.4, lm: 0.2, mmu: 0.4], and world_size = 32, then
            # ranks 0-12 will be assigned to t2i, ranks 13-18 to lm, and ranks 19-31 to mmu.
            # In the other side, all the samples in the t2i dataset will be evently distributed among
            # the ranks 0-12, and so on.

            # Get weights and remove zero-weight dataset
            valid_keys = [key for key in self.all_dataset_keys if self.sampling_probs_dict[key] > 0]
            assert len(valid_keys) <= self.world_size, (
                f"Number of world_size({self.world_size}) should be greater than or equal to "
                f"valid datasets ({len(valid_keys)}, {valid_keys}) in fixed mode. "
                f"Try to use more GPUs, set sampling_probs to 0 for some datasets, or use random sampling mode "
                f"(combined_iterator_sampling_mode=random)."
            )
            weights = [self.sampling_probs_dict[key] for key in valid_keys]
            key_index, rank_bias, num_replicas, cum_ranks = \
                CombinedBatchIterator.fixed_dataloader_allocation(
                    weights=weights, keys=valid_keys, rank=self.dp_rank, world_size=self.dp_size, _logger=self.logger)
            self.cur_key = valid_keys[key_index]
            self.cur_key_group = list(range(cum_ranks[key_index], cum_ranks[key_index + 1]))
            # For invalid dataset (i.e. sample_probs=0), set the number of replicas to 1 to avoid
            # zero division error in the sampler.
            self.dataset_num_replicas = {key: 1 for key in self.all_dataset_keys}
            for i, key in enumerate(valid_keys):
                self.dataset_num_replicas[key] = num_replicas[i]
            # For invalid dataset (i.e. sample_probs=0), set the dataset_rank to 0 by default.
            self.dataset_rank = {key: 0 for key in self.all_dataset_keys}
            self.dataset_rank[self.cur_key] = self.dp_rank - rank_bias

        self.dataset_dict = {}
        self.sampler_dict = {}
        self.batch_sampler_dict = {}
        self.dataloader_dict = {}
        self.task_dummy_dict = {}

    def build_dataloader(self):
        args = self.args

        self.dataloader_preliminary_setup()

        # =====================================
        #          Text to image data
        # =====================================
        real_text_token_length = args.text_token_length - self.dummy_dict['mmu']

        # Build t2i dataset
        self.dataset_dict['t2i'] = self.dataset = TransfusionTextImageArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            text_token_length=real_text_token_length,
            uncond_p=args.uncond_p,
            tokenizer_name=args.tokenizer_name,
            multireso=args.multireso,               # Set multireso=True to make batch in resolution buckets. bsz must be provided in index dataset and sampler.
            index_kwargs=dict(
                batch_size=self.micro_batch_size,   # Provide bsz to use multireso.
                world_size=1,   # Dataset don't need to align with world_size. It will be handled by the sampler.
                **args.index_kwargs,
            ),
            post_kwargs=dict(
                return_attention_mask=True,
                dummy_number=self.dummy_dict['mmu'],
            ),
            debug=False,
            logger=self.logger,
        )
        # Build sampler and data loader
        dataloader_kwargs = dict(
            **args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.dp_rank)
        )
        self.sampler_dict['t2i'] = DistributedSampler(
            self.dataset_dict['t2i'],
            num_replicas=self.dataset_num_replicas['t2i'],
            rank=self.dataset_rank['t2i'],
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
            batch_size=self.micro_batch_size,   # Provide bsz to use multireso.
        )
        self.dataloader_dict['t2i'] = DataLoader(
            self.dataset_dict['t2i'],
            batch_size=self.micro_batch_size,
            sampler=self.sampler_dict['t2i'],
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs,
        )

        # =====================================
        #             Language data
        # =====================================
        lm_max_length = args.lm_token_length + 1 - self.dummy_dict['t2i'] - self.dummy_dict['mmu']
        assert args.lm_batch_sampler == "max_length", "`max_length` batch sampler is required in Pretrain."

        dataset_kwargs = dict(
            args=args,
            t2t_text_token_length=lm_max_length,
            tokenizer_name=args.tokenizer_name,
            index_kwargs=dict(
                **args.lm_index_kwargs,
            ),
            logger=self.logger,
        )
        sampler_kwargs = dict(
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )
        batch_sampler_kwargs = dict(
            batch_size=self.micro_batch_size,
            max_length=lm_max_length,
        )

        # ------------- LM Text ---------------
        self.dataset_dict['lm_text'] = TextArrowStream(
            index_file=args.lm_text_index_file,
            dataset_type="lm_text",
            **dataset_kwargs,
        )
        self.sampler_dict['lm_text'] = DistributedSampler(
            self.dataset_dict['lm_text'],
            num_replicas=self.dataset_num_replicas['lm_text'],
            rank=self.dataset_rank['lm_text'],
            **sampler_kwargs,
        )
        self.lm_text_batch_sampler = MaxLengthBatchSampler(
            self.dataset_dict['lm_text'].index_manager,
            self.sampler_dict['lm_text'],
            length_getter=hy_text_length_getter,
            **batch_sampler_kwargs
        )
        self.dataloader_dict['lm_text'] = DataLoader(
            self.dataset_dict['lm_text'], batch_sampler=self.lm_text_batch_sampler, **dataloader_kwargs)

        # ------------- LM Recaption ---------------
        self.dataset_dict['lm_recap'] = TextArrowStream(
            index_file=args.lm_recap_index_file,
            dataset_type="lm_recap",
            **dataset_kwargs,
        )
        self.sampler_dict['lm_recap'] = DistributedSampler(
            self.dataset_dict['lm_recap'],
            num_replicas=self.dataset_num_replicas['lm_recap'],
            rank=self.dataset_rank['lm_recap'],
            **sampler_kwargs,
        )
        self.lm_recap_batch_sampler = MaxLengthBatchSampler(
            self.dataset_dict['lm_recap'].index_manager,
            self.sampler_dict['lm_recap'],
            length_getter=recaption_length_getter,
            **batch_sampler_kwargs
        )
        self.dataloader_dict['lm_recap'] = DataLoader(
            self.dataset_dict['lm_recap'], batch_sampler=self.lm_recap_batch_sampler, **dataloader_kwargs)

        # ================================================
        #          Multimodal understanding data
        # ================================================
        # Set the number of dummy tokens for mmu sequences
        mmu_max_length = args.mmu_token_length + 1 - self.dummy_dict['t2i']

        self.dataset_dict['mmu'] = MultiModalUnderstandingArrowStream(
            args=args,
            index_file=args.mmu_index_file,
            max_token_length=mmu_max_length,
            tokenizer_name=args.tokenizer_name,
            index_kwargs=dict(
                **args.mmu_index_kwargs,
            ),
            dummy_number=self.dummy_dict['t2i'],
            logger=self.logger,
        )
        self.sampler_dict['mmu'] = DistributedSampler(
            self.dataset_dict['mmu'],
            num_replicas=self.dataset_num_replicas['mmu'],
            rank=self.dataset_rank['mmu'],
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
        )
        self.dataloader_dict['mmu'] = DataLoader(
            self.dataset_dict['mmu'],
            batch_size=self.micro_batch_size,
            sampler=self.sampler_dict['mmu'],
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs,
        )

    def resume_dataloader(self, ss):
        # The sampler states will be restored in self.shuffle_dataset()
        for sampler in self.sampler_dict.values():
            assert isinstance(sampler, DistributedSampler), (
                f"In {self.__class__.__name__}, only index_kits.samplers.DistributedSampler supports --resume-dataloader."
            )

    def get_trainable_params(self, model, training_parts):
        if training_parts == "inv_sampling_probs":
            mmu_proj = []
            img_proj = []
            hw_emb = []
            lm_head = []
            transformer = []

            sum_probs = sum(self.sampling_probs_dict.values())
            mmu_factor = (self.sampling_probs_dict.get('mmu', 0) / sum_probs) or 1
            t2i_factor = (self.sampling_probs_dict.get('t2i', 0) / sum_probs) or 1
            lm_factor = (
                (self.sampling_probs_dict.get('lm_text', 0) + self.sampling_probs_dict.get('lm_recap', 0)) / sum_probs
            ) or 1

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith('vision_model') or name.startswith('vision_aligner'):
                    mmu_proj.append(param)
                elif name.startswith('language_model.timestep_emb') or name.startswith('language_model.patch_embed') \
                        or name.startswith('language_model.final_layer') or name.startswith('language_model.time_embed'):
                    img_proj.append(param)
                elif name.startswith('language_model.w_emb') or name.startswith('language_model.h_emb'):
                    hw_emb.append(param)
                elif name.startswith('language_model.lm_head'):
                    lm_head.append(param)
                elif name.startswith('language_model.transformer'):
                    transformer.append(param)
                else:
                    raise ValueError(f"Undetermined parameter: {name}")

            params = [
                {'params': mmu_proj, 'lr': self.args.lr / mmu_factor},
                {'params': img_proj, 'lr': self.args.lr / t2i_factor},
                {'params': hw_emb, 'lr': self.args.lr / (mmu_factor + t2i_factor)},
                {'params': lm_head, 'lr': self.args.lr / (mmu_factor + lm_factor)},
                {'params': transformer, 'lr': self.args.lr},
            ]
            names = [
                'mmu_proj', 'img_proj', 'hw_emb', 'lm_head', 'transformer'
            ]

            self.logger.info(f"Using diverse learning rates for different parts of the model:")
            for name, item in zip(names, params):
                self.logger.info(f"    {name} ... ({len(item['params']):>4d} tensors) ... lr={item['lr']}")

        else:
            params = [{'params': [p for p in model.parameters() if p.requires_grad]}]

        self.trainable_params = params
        self.num_trainable_params = sum(
            p.numel() if isinstance(p, torch.Tensor)
            else sum(p2.numel() for p2 in p['params'])
            for p in self.trainable_params
        )
        return self.trainable_params

    @property
    def vae_prerun_sizes(self):
        from index_kits import MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2

        # Pre-run vae arguments. Each is a tuple of (batch_size, height, width)
        vae_prerun_sizes = set()
        vae_prerun_tasks = set(self.args.get("vae_prerun_tasks", []))
        cur_tasks = list(self.dataset_dict.keys()) if self.cur_key is None else [self.cur_key]
        for task in cur_tasks:
            if task in vae_prerun_tasks:
                dataset = self.dataset_dict[task]
                dataloader = self.dataloader_dict[task]
                if dataloader.batch_size is None:
                    bsz = dataloader.batch_sampler.batch_size
                else:
                    bsz = dataloader.batch_size
                if isinstance(dataset.index_manager, MultiResolutionBucketIndexV2):
                    for bucket in dataset.index_manager.buckets:
                        vae_prerun_sizes.add((bsz, bucket.height, bucket.width))
                elif isinstance(dataset.index_manager, MultiMultiResolutionBucketIndexV2):
                    for multireso in dataset.index_manager.buckets:
                        for bucket in multireso.buckets:
                            vae_prerun_sizes.add((bsz, bucket.height, bucket.width))
                elif hasattr(dataset, "vae_reso_group"):
                    for reso in dataset.vae_reso_group.data:
                        vae_prerun_sizes.add((bsz, reso.h, reso.w))

        return vae_prerun_sizes

    def prerun_vae(self):
        vae_prerun_sizes = sorted(list(self.vae_prerun_sizes), key=lambda x: (x[1], -x[2]))

        if len(vae_prerun_sizes) > 0:
            self.logger.info("Pre-running VAE...")

            vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
            with torch.autocast(
                    device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32
            ):
                for i, (bsz, height, width) in enumerate(vae_prerun_sizes):
                    self.logger.info(f"    Bucket {i}: {bsz}x3x{height}x{width}")
                    image = torch.rand((bsz, 3, height, width), device=self.device)
                    _ = self.vae.encode(image)
        else:
            self.logger.info("No task relies on VAE. Skip pre-running VAE.")

    def build_extra_model(self, **kwargs):
        """ Extra frozen models. """
        args = self.args

        if kwargs.get('skip_build_vae'):
            self.vae = None
        else:
            self.logger.info("Building VAE...")
            self.vae = load_vae(
                args.vae_type,
                args.vae_precision,
                device=self.device,
                logger=self.logger,
                only_encoder=True,
                sample_size=args.get('vae_sample_size'),
            )
            if args.get('vae_spatial_tiling', False):
                self.vae.enable_spatial_tiling()
            if args.get('vae_slicing_bsz') is not None and args.vae_slicing_bsz > 1:
                self.vae.slicing_bsz = args.vae_slicing_bsz
                self.vae.enable_slicing()
                self.logger.info(f"VAE slicing batch size set to {self.vae.slicing_bsz}")
            if args.get('vae_use_compile'):
                self.vae.use_compile = args.vae_use_compile
                self.logger.info(f"VAE use_compile set to {self.vae.use_compile}")
            self.vae_generater = torch.Generator(self.device).manual_seed(args.seed + self.dp_rank)
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

    def _vae_encode_tensor(self, image, sample_type=None, n_tokens=None, skip_encode=False):
        # ===================================== prepare diffusion =====================================
        if skip_encode:
            latents = image
        else:
            vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                vae_encode_result = self.vae.encode(image)
                if is_torch_tensor(vae_encode_result):
                    latents = vae_encode_result
                else:
                    latents = vae_encode_result.latent_dist.sample(generator=self.vae_generater)
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    latents.sub_(self.vae.config.shift_factor)
                if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                    latents.mul_(self.vae.config.scaling_factor)

            # b c t h w
            if hasattr(self.vae, "ffactor_temporal"):
                assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
                latents = latents.squeeze(2)

        if sample_type is not None:
            if sample_type == "sample":
                t, x_0, x_1 = self.denoiser.sample(latents, n_tokens, generator=self.vae_generater)
            elif sample_type == "sample_start":
                t, x_0, x_1 = self.denoiser.sample_start(latents, generator=self.vae_generater)
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
            t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000

            return VAEEncodeOutput(t=t, model_t=model_t, x_0=x_0, x_t=x_t, u_t=u_t, latents=latents)

        return VAEEncodeOutput(latents=latents)

    def vae_encode(self, images, sample_type=None, n_tokens=None):
        if isinstance(images, list):
            assert n_tokens is None, "n_tokens is not supported for list of images."
            batch_output = []
            for bsz_i, image_item in enumerate(images):
                if is_torch_tensor(image_item) and image_item.ndim == 3:
                    image_item = image_item.unsqueeze(0)
                if isinstance(image_item, list):
                    vae_outputs = [
                        self._vae_encode_tensor(
                            image[None].to(self.device), sample_type=sample_type, n_tokens=n_tokens
                        )
                        for image in image_item
                    ]
                    outputs = VAEEncodeOutput.cat(vae_outputs)
                else:
                    image_item = image_item.to(self.device)
                    outputs = self._vae_encode_tensor(image_item, sample_type=sample_type, n_tokens=n_tokens)
                batch_output.append(outputs)
            batch_output = VAEEncodeOutput.build(batch_output)

        elif is_torch_tensor(images):
            images = images.to(self.device)
            if images.ndim == 4:
                batch_output = self._vae_encode_tensor(images, sample_type=sample_type, n_tokens=n_tokens)
            elif images.ndim == 5:
                vae_outputs = [
                    self._vae_encode_tensor(image, sample_type=sample_type, n_tokens=n_tokens)
                    for image in images
                ]
                batch_output = VAEEncodeOutput.build(vae_outputs)
            else:
                raise ValueError(f"images should have shape [B, C, H, W] or [B, n, C, H, W], got {images.shape}")

        else:
            raise ValueError(f"Unknown images type, expected [list, torch.Tensor], got {type(images)}")

        return batch_output

    def build_data_iterator(self):
        self.dataloader = CombinedBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.dp_rank,
            world_size=self.world_size,
            datasets=self.dataset_dict,
            samplers=self.sampler_dict,
            dataloaders=self.dataloader_dict,
            sampling_probs=self.sampling_probs_dict,
            initial_seed=self.args.global_seed,
            sampling_mode=self.args.get('combined_iterator_sampling_mode', 'random'),
            cache_shuffle=self.args.get('cache_shuffle'),
            fixed_key=self.cur_key,
            fixed_key_group=self.cur_key_group,
        )

    def get_sampler(self):
        """ Get evaluation sampler """
        return Text2ImageTransfusionSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine,
                            model_settings=self.model_settings,
                            tokenizer=self.tkwrapper,
                            scheduler=load_scheduler(self.args),
                            ),
            rank=self.dp_rank,
            world_size=self.dp_size,
            device=self.device,
            logger=self.val_logger,
        )

    def save_first_training_samples(self, batch, key_not_save=None, extra=None):
        if self.args.save_n_training_data == 0:
            return
        # Save training data for debugging
        dp_rank = self.dp_rank if self.cur_key is None else self.dataset_rank[self.cur_key]
        pp_rank = self.get_pp_rank()
        cur_step = self.ss.current_run_update_steps
        # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
        check_data_path = safe_dir(self.exp_dir / "saved_training_data")
        save_path = check_data_path / f"data_batch{cur_step}_{self.cur_key}_dp{dp_rank}_pp{pp_rank}_global{self.rank}.pt"
        # If the file already exists (when grad_acc > 1), skip saving.
        if save_path.exists():
            return
        if cur_step < self.args.save_n_training_data and dp_rank < 8:
            # Filter BlockMask
            if BlockMask is not None:
                batch = {k: v for k, v in batch.items() if not isinstance(v, BlockMask)}
            if key_not_save is not None:
                if isinstance(key_not_save, str):
                    key_not_save = [key_not_save]
                batch = {k: v for k, v in batch.items() if k not in key_not_save}
            if extra is not None:
                for k, v in extra.items():
                    if k not in batch:
                        batch[k] = v
                    else:
                        batch[f"extra_{k}"] = v
            torch.save(batch, save_path)

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        if batch["dtype"][0].startswith("t2i"):
            inputs = self.prepare_model_t2i_inputs(batch, device, **kwargs)
        elif batch["dtype"][0].startswith("lm"):
            inputs = self.prepare_model_lm_inputs(batch, device, **kwargs)
        elif batch["dtype"][0].startswith("mmu"):
            inputs = self.prepare_model_mmu_inputs(batch, device, **kwargs)
        else:
            raise ValueError(f"Unknown batch dtype, expected [{self.all_dataset_keys}], got {batch['dtype']}")
        return inputs

    def add_dummy_tokens(self, tokens, target_tokens, extra,
                         dummy_token_type, dummy_number, device):
        """ Add dummy tokens to avoid hanging when deepspeed all-reduce gradients.

        Different modalities connect with different model parameters in the computation graph.
        If different batches correspond to different modalities, deepspeed cannot correctly perform
        all-reduce gradients. Therefore, we need to pad some dummy tokens to maintain consistent
        activated model parameters.

        ======================================== Dummy tokens in sequence ========================================
              t2i: ++++++++++++ mmu_dummy face_dummy
        ti2i\face: ++++++++++++ mmu_dummy face_dummy
              mmu: ++++++++++++ t2i_dummy face_dummy
               lm: ++++++++++++ t2i_dummy mmu_dummy face_dummy
             face: ++++++++++++ mmu_dummy                               # rely on t2i/t2i_dummy
        ==========================================================================================================

        ======================================== Dummy token affectation =========================================

         t2i_dummy: tokens, target_tokens, text_mask, und_image_masks, iw_ih, image_mask,                 timestep, x_t, t, diff_fn
                    ^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^  ***************  *+*+*  ++++++++++                  ++++++++  +++  +  +++++++
         mmu_dummy: tokens, target_tokens, text_mask, und_image_masks, iw_ih, image_mask, src_image_mask,                           und_image
                    ^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^  *+*+*+*+*+*+*+*  *+*+*  **********  **************                            +++++++++
        face_dummy: tokens, target_tokens, text_mask, und_image_masks,        image_mask, src_image_mask,                                     src_face_embedding
                    ^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^  *+*+*+*+*+*+*+*         **********  **************                                      ++++++++++++++++++

        Notice:
          ^^^: Fixed concat
          ***: Automatic extension
          *+*: Automatic extension or new ones.
          +++: New ones that cause dummy
        ==========================================================================================================

        """
        batch_size, n_tokens = tokens.shape

        # ^^^^^^^^ Fixed concat ^^^^^^^^
        dummy_tokens = torch.full((batch_size, dummy_number), self.tkwrapper.pad_token,
                                  dtype=tokens.dtype, device=device)
        dummy_target_tokens = (-100) * torch.ones((batch_size, dummy_number), dtype=tokens.dtype, device=device)
        tokens = torch.cat([tokens, dummy_tokens], dim=1)
        target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

        extra['text_mask'] = torch.cat([
            extra['text_mask'],
            torch.zeros_like(dummy_tokens, dtype=extra['text_mask'].dtype, device=device)
        ], dim=1)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        add_n_tokens = 0
        if dummy_token_type in ["t2i", "mmu"]:
            # *+*+*+*+ Automatic extension or new ones *+*+*+*+
            if self.args.add_iw_ih_token:
                scatter_index = torch.tensor([[n_tokens, n_tokens + 1]] * batch_size, dtype=torch.long, device=device)
                scatter_src = torch.tensor([[2, 2]] * batch_size, dtype=torch.long, device=device)
                if 'iw_ih_scatter_index' in extra:
                    if isinstance(extra['iw_ih_scatter_index'], list):
                        extra['iw_ih_scatter_index'] = [torch.cat([x1, x2]) for x1, x2 in zip(extra['iw_ih_scatter_index'], scatter_index)]
                        extra['iw_ih_scatter_src'] = [torch.cat([x1, x2]) for x1, x2 in zip(extra['iw_ih_scatter_src'], scatter_src)]
                    else:
                        extra['iw_ih_scatter_index'] = torch.cat([extra['iw_ih_scatter_index'], scatter_index], dim=1)
                        extra['iw_ih_scatter_src'] = torch.cat([extra['iw_ih_scatter_src'], scatter_src], dim=1)
                else:
                    extra['iw_ih_scatter_index'] = scatter_index
                    extra['iw_ih_scatter_src'] = scatter_src
                add_n_tokens += 2
            # *+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*

        if dummy_token_type == "t2i":
            # ++++++++ New ones that cause dummy ++++++++
            image_mask = torch.zeros_like(tokens, dtype=torch.bool, device=device)
            image_mask[:, -1] = True
            patch_size = self.args.patch_size
            latents = torch.randn((batch_size, self.args.vae_latent_dim, patch_size, patch_size), device=device)
            t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
            t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000
            extra.update(dict(
                x_t=x_t,
                t=model_t,
                image_mask=image_mask,
                diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            ))
            if self.args.add_timestep_token:
                if 'timestep_scatter_index' not in extra:
                    timestep_scatter_index = torch.tensor([[n_tokens + add_n_tokens]] * batch_size, dtype=torch.long, device=device)
                    extra.update(dict(
                        timestep_scatter_index=timestep_scatter_index,
                    ))
                else:
                    extra['timestep_scatter_index'] = torch.cat([
                        extra['timestep_scatter_index'],
                        torch.tensor([[n_tokens + add_n_tokens]] * batch_size, dtype=torch.long, device=device)
                    ], dim=1)
            # ++++++++++++++++++++++++++++++++++++++++++++

            # ******** Automatic extension ********
            if 'und_image_masks' in extra:
                extra['und_image_masks'] = torch.cat([
                    extra['und_image_masks'],
                    torch.zeros_like(dummy_tokens, dtype=extra['und_image_masks'].dtype, device=device)
                ], dim=1)
            if 'src_image_mask' in extra:
                extra['src_image_mask'] = torch.cat([
                    extra['src_image_mask'],
                    torch.zeros_like(dummy_tokens, dtype=extra['src_image_mask'].dtype, device=device)
                ], dim=1)
            # *************************************

        elif dummy_token_type == "mmu":
            # ++++++++ New ones that cause dummy ++++++++
            vision_encoder_meta_info = VISION_ENCODER_META_INFO[self.args.vision_model_type]
            downsample_factor = to_2tuple(vision_encoder_meta_info["downsample_factor"])
            
            
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                # siglip2-so400m-patch16-naflex needs und_images as 3D tensor: batch_size x seq_len x dim
                pixel_image = torch.rand((batch_size, 1, 3 * math.prod(downsample_factor)), device=device) * 2 - 1
                extra.update(dict(
                    vision_encoder_kwargs={
                        "spatial_shapes": torch.tensor([[1, 1]]*batch_size, dtype=torch.long, device=device),
                        "attention_mask": torch.ones((batch_size, 1), dtype=torch.int32, device=device),
                    },
                ))
            else:
                pixel_image = torch.rand((batch_size, 3, *downsample_factor), device=device) * 2 - 1
            extra.update(dict(
                und_images=pixel_image,
            ))
            # ++++++++++++++++++++++++++++++++++++++++++++

            # *+*+*+*+ Automatic extension or new ones *+*+*+*+
            if 'und_image_masks' in extra:
                extra['und_image_masks'] = torch.cat([
                    extra['und_image_masks'],
                    torch.zeros_like(dummy_tokens, dtype=extra['und_image_masks'].dtype, device=device)
                ], dim=1)
            else:
                extra['und_image_masks'] = torch.zeros_like(tokens, dtype=torch.bool, device=device)
            extra['und_image_masks'][:, -1] = True
            # *+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*

            # ******** Automatic extension ********
            if 'image_mask' in extra:
                extra['image_mask'] = torch.cat([
                    extra['image_mask'],
                    torch.zeros_like(dummy_tokens, dtype=extra['image_mask'].dtype, device=device)
                ], dim=1)

            if 'src_image_mask' in extra:
                extra['src_image_mask'] = torch.cat([
                    extra['src_image_mask'],
                    torch.zeros_like(dummy_tokens, dtype=extra['src_image_mask'].dtype, device=device)
                ], dim=1)
            # *************************************

        elif dummy_token_type == "face":
            # ++++++++ New ones that cause dummy ++++++++
            extra.update(dict(
                src_face_embedding=torch.rand((batch_size, 512), device=tokens.device),
            ))
            # ++++++++++++++++++++++++++++++++++++++++++++

            # *+*+*+*+ Automatic extension or new ones *+*+*+*+
            if 'und_image_masks' in extra:
                extra['und_image_masks'] = torch.cat([
                    extra['und_image_masks'],
                    torch.ones_like(dummy_tokens, dtype=extra['und_image_masks'].dtype, device=device)
                ], dim=1)
            else:
                extra['und_image_masks'] = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
                extra['und_image_masks'][:, -dummy_number:] = True
            # *+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*

            # ******** Automatic extension ********
            if 'image_mask' in extra:
                extra['image_mask'] = torch.cat([
                    extra['image_mask'],
                    torch.zeros_like(dummy_tokens, dtype=extra['image_mask'].dtype, device=device)
                ], dim=1)

            if 'src_image_mask' in extra:
                extra['src_image_mask'] = torch.cat([
                    extra['src_image_mask'],
                    torch.zeros_like(dummy_tokens, dtype=extra['src_image_mask'].dtype, device=device)
                ], dim=1)
            # *************************************

        n_tokens += dummy_number

        return tokens, target_tokens, extra, n_tokens

    def prepare_model_lm_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,
        )
        if self.dummy_dict['t2i']:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                tokens, target_tokens, extra, dummy_token_type="t2i", dummy_number=self.dummy_dict['t2i'], device=device)
        if self.dummy_dict['mmu']:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                tokens, target_tokens, extra, dummy_token_type="mmu", dummy_number=self.dummy_dict['mmu'], device=device)
        batch_size, n_tokens = tokens.shape

        # Attention mask
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)

        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="lm",         # For loss
            **extra,    # x_t, t, image_mask, und_images, und_image_masks, diffusion_loss_fn, iw, ih, timestep
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_t2i_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,        # [b, seqlen]
            image_mask=image_mask,      # [b, seqlen]
            iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
            iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
            timestep_scatter_index=batch["timestep_scatter_index"].to(device),  # [b, 1]
        )
        if self.dummy_dict['mmu']:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                tokens, target_tokens, extra, dummy_token_type="mmu", dummy_number=self.dummy_dict['mmu'], device=device)
        batch_size, n_tokens = tokens.shape

        # Attention mask
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        if kwargs.get('skip_vae_encode'):
            x_t, model_t, t, x_0, u_t = None, None, None, None, None
        else:
            image = batch["image"].to(device)
            vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                vae_encode_result = self.vae.encode(image)
                if isinstance(vae_encode_result, torch.Tensor):
                    latents = vae_encode_result
                else:
                    latents = vae_encode_result.latent_dist.sample()
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    latents.sub_(self.vae.config.shift_factor)
                if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                    latents.mul_(self.vae.config.scaling_factor)

            # b c t h w
            if hasattr(self.vae, "ffactor_temporal"):
                assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
                latents = latents.squeeze(2)

            t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
            t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,                         # [b, seqlen]
            x_t=x_t,                            # [b, c, h, w]
            t=model_t,                          # [b]
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            target=target_tokens,               # [b, seqlen]
            attention_mask=attention_mask,      # [b, seqlen, seqlen]
            image_loss_weight=self.args.image_loss_weight,
            data_type="t2i",        # For loss
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_mmu_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # und_image_mask is used to fill image_embeds, therefore should be shifted same as tokens
        und_image_mask = batch["und_image_mask"][:, :-1].contiguous().to(device)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,        # [b, seqlen]
            und_image_masks=und_image_mask,  # [b, seqlen]
            iw_ih_scatter_index=batch["iw_ih_scatter_index"].to(device),        # [b, 2]
            iw_ih_scatter_src=batch["iw_ih_scatter_src"].to(device),            # [b, 2]
        )
        if self.dummy_dict['t2i']:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens(
                tokens, target_tokens, extra, dummy_token_type="t2i", dummy_number=self.dummy_dict['t2i'], device=device)
        batch_size, n_tokens = tokens.shape

        # build attention mask. Intra-image is full attention, inter-image and image-text is causal attention.
        attention_mask = batch["attention_mask"].to(device)

        images = batch["image"].to(device)
        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="mmu",        # For loss
            und_images=images,
            **extra,
        )

        self.save_first_training_samples(model_intput_kwargs)
        return model_intput_kwargs, batch_size, n_tokens

    def shuffle_dataset_and_set_start_index(self, ss):
        # Shuffle and reset index are handled by the mix-data iterator, i.e., self.dataloader
        pass

    def get_states_cls(self, state_type):
        if state_type == "scalar":
            return MultiModalScalarStates
        elif state_type == "cycle":
            return MultiModalCycleStates
        else:
            raise ValueError(f"Unknown state type: {state_type}")

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

    def update_log_states(self, ss, all_cs):
        consumed_samples = 0
        consumed_tokens = 0
        for key in self.all_dataset_keys:
            part_samples = sum([cs_i.running_samples[key] for cs_i in all_cs])
            consumed_samples += part_samples
            self.ss.consumed_samples_total[key] += part_samples
            self.ss.epoch_consumed_samples[key] += part_samples
            part_tokens = sum([cs_i.running_tokens[key] for cs_i in all_cs])
            consumed_tokens += part_tokens
            self.ss.consumed_tokens_total[key] += part_tokens

        self.ss.add(
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * consumed_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * consumed_tokens / C_SCALE,
        )

        return consumed_samples

    def get_events(self, ss, loss):
        log_events = []
        for key in self.all_dataset_keys:
            log_events.extend([
                f"Consumed {key} Samples: {ss.consumed_samples_total[key]:,}",
                f"Consumed {key} Tokens: {ss.consumed_tokens_total[key]:,}",
            ])

        consumed_samples_total = sum([ss.consumed_samples_total[key] for key in self.all_dataset_keys])
        consumed_tokens_total = sum([ss.consumed_tokens_total[key] for key in self.all_dataset_keys])
        summary_events = [
            ("Train/TotalSamples/train_loss", loss, consumed_samples_total),
            ("Train/TotalTokens/train_loss", loss, consumed_tokens_total),
        ]
        return log_events, summary_events
