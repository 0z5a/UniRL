# FIXME
import time
from typing import Union, Dict
import copy
from einops import rearrange
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as TF

from hymm.models.diffusion.posemb_layers import get_nd_rotary_pos_embed
from hymm.diffusion import load_denoiser

from hymm.trainers.base_trainer import BaseTrainer
from hymm.data_kits.text_image_loader import TextImageArrowStream
from hymm.models import load_vae

from hymm.trainers.helpers import dynamic_values_wrapper
from hymm.utils.helpers import as_tuple
from hymm.utils.torch_utils import (
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
)


EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipToken2ImageDiffusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        
        # args.logger = self.logger # this will result in  error during ckpt saving
        
        eva_size = EVA_IMAGE_SIZE
        eva_mean = OPENAI_DATASET_MEAN
        eva_std = OPENAI_DATASET_STD
        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            # TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])
        

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args
        image = batch["image"]
                
        # ======================================== Build SDXL embedding for crop, time and text ======================================
        
        # log shape, range, dtype of image, mean, std
        self.logger.info(f"prepare_model_inputs: Input image shape: {image.shape=}; dtype: {image.dtype=}")
        self.logger.info(f"prepare_model_inputs: Input image range: {image.min().item()=} {image.max().item()=}")
        self.logger.info(f"prepare_model_inputs: Input image mean: {image.mean().item()}; std: {image.std().item()}")
        # hymm.trainers.cliptoken2image_diffusion_trainer:prepare_model_inputs:57 - Input image shape: image.shape=torch.Size([4, 3, 1024, 1024]); dtype: image.dtype=torch.float32
        # hymm.trainers.cliptoken2image_diffusion_trainer:prepare_model_inputs:58 - Input image range: image.min().item()=-1.0 image.max().item()=1.0
        # hymm.trainers.cliptoken2image_diffusion_trainer:prepare_model_inputs:60 - Input image mean: 0.11428836733102798
        # hymm.trainers.cliptoken2image_diffusion_trainer:prepare_model_inputs:61 - Input image std: 0.6745584011077881
        
        # default value, assume all high quality 
        sample_size  = list(image.shape[-2:])        
        height, width = sample_size
        crop_info = [0, 0]
        original_size = list(image.shape[-2:])
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(device)
        time_ids = torch.repeat_interleave(time_ids[None, ...], image.shape[0], dim=0)
        if 'kwargs' in batch:
            batch_kwargs = batch['kwargs']
            assert 'origin_size' in batch_kwargs, "origin_size should be in batch_kwargs"
            assert 'target_size' in batch_kwargs, "target_size should be in batch_kwargs"
            assert 'crop_coords_xy' in batch_kwargs, "crop_coords_xy should be in batch_kwargs"
            
            original_size = batch['kwargs']["origin_size"] # torch.Size([4, 2])
            sample_size = batch['kwargs']["target_size"] 
            crop_info = batch['kwargs']["crop_coords_xy"]
            time_ids = torch.cat([original_size, crop_info, sample_size], dim=1).to(device)

        # ======================================== End Build SDXL embedding for crop, time and text ======================================
        # Tokenization
        image = image.to(device)
        vae_dtype = PRECISION_TO_TYPE[args.clipvision_precision]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                # resize image to 448x448
                # FIXME, input resize should be avoided
                # check the range of image here to see if it is in [0, 1], if not, raise an error
                # change image range from [-1, 1] to [0, 1]
                image_min0_to_max1 = (image + 1.0) / 2.0
                
                clip_size_image = self.transform(image_min0_to_max1)
                # print(f"Input shape before CLIP encode_image: {clip_size_image.shape=}")                
                pooling_stride = self.args.get("pooling_stride", None)
                image_tokens = self.multimodal_encoder.model.encode_image(image=clip_size_image, pooling_stride=pooling_stride, logger=self.logger).float()
                # print(f"Output shape after CLIP encode_image: {image_tokens.shape=}")

        vae_dtype = PRECISION_TO_TYPE[args.vae_precision]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                # print(f"Input images shape: {image.shape=}")
                latents = self.vae.encode(image.to(device, dtype=torch.float32)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = latents.to(device, dtype=self.target_dtype)
                
                # print(f"Input latents shape: {latents.shape=}")

        # if len(image_tokens.shape) > 2:
        #     image_tokens = rearrange(image_tokens, "b w h -> b (w h) 1")

        # ======================================== Build RoPE ======================================
        # target_ndim = 2     # n-d RoPE
        # ndim = len(image.shape) - 2
        # image_size = list(image.shape[-ndim:])
        # assert all(s % self.model_settings.patch_size == 0 for s in image_size), \
        #     f"Image size(last {ndim} dimensions) should be divisible by patch size({self.model_settings.patch_size}), " \
        #     f"but got {image_size}."
        # rope_sizes = [s // self.model_settings.patch_size for s in image_size]
        # if len(rope_sizes) != target_ndim:
        #     rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        # head_dim = self.model_settings.hidden_size // self.model_settings.num_heads
        # rope_dim_list = self.model_settings.rope_dim_list
        # if rope_dim_list is None:
        #     rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        # assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        # freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        #     rope_dim_list=rope_dim_list,
        #     start=rope_sizes,
        #     theta=self.model_settings.rope_theta,
        #     use_real=True,
        #     theta_rescale_factor=1.0,
        # )
        unet_added_conditions = {}
        unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(image_tokens, dim=1)

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            cond=image_tokens,              # [b, l]
            # cond_mask=None,
            # freqs_cos=freqs_cos,            # [seqlen, head_dim]
            # freqs_sin=freqs_sin,            # [seqlen, head_dim]
            added_cond_kwargs = unet_added_conditions.copy(),
            return_dict=True
        )

        batch_size = image_tokens.shape[0]
        n_tokens = image_tokens.shape[1]

        self.logger.info(f"Prepare input shape image_tokens after CLIP encode_image: {model_kwargs['cond'].shape=}")
        self.logger.info(f"Prepare input latents shape: {latents.shape=}")
        for k, v in model_kwargs['added_cond_kwargs'].items():
            self.logger.info(f"Prepare input added_cond_kwargs: {k=} {v.shape=}")
        return latents, model_kwargs, batch_size, n_tokens

    def build_extra_model(self):
        args = self.args

        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )
        self.multimodal_encoder = load_vae(
            args.clipvision_type,
            args.clipvision_precision,
            device=self.device,
            logger=self.logger,
        )
        self.denoiser = load_denoiser(args)

    def build_dataloader(self):
        args = self.args
        self.dataset = TextImageArrowStream(
            image_size=args.image_size,
            image_len=args.image_len,
            tokenizer=args.tokenizer,
            text_len=args.text_len,
            uncond_p=args.uncond_p,
            index_file=args.index_file,
            multireso=args.multireso,
            index_kwargs=dict(
                batch_size=1 if args.mix_scale else self.micro_batch_size,
                world_size=1 if args.mix_scale else self.world_size,
                **args.index_kwargs,
            ),
            debug=False,
            logger=self.logger,
        )
        # Build sampler and data loader
        dataloader_kwargs = dict(
            **args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.rank)
        )
        # Dynamic anchor size for mix-scale training.
        self.anchor_sizes = as_tuple(args.anchor_size)
        self.dynamic_anchor_size = dynamic_values_wrapper(self.anchor_sizes, self.anchor_sizes)
        if args.mix_scale:
            self.data_sampler = DistributedSamplerWithStartIndex(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=args.global_seed,
                drop_last=True,
            )
            dynamic_batch_size = dynamic_values_wrapper(self.anchor_sizes, self.args.mix_micro_batch_size)
            batch_sampler = IndexBatchSampler(
                self.dataset.index_manager, self.data_sampler, batch_size=dynamic_batch_size, drop_last=True
            )
            self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler, **dataloader_kwargs)
        else:
            if args.multireso:
                self.data_sampler = BlockDistributedSampler(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=args.global_seed,
                    drop_last=True,
                    batch_size=self.micro_batch_size,
                )
            else:
                self.data_sampler = DistributedSamplerWithStartIndex(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    seed=args.global_seed,
                    drop_last=True,
                )
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.micro_batch_size,
                sampler=self.data_sampler,
                shuffle=False,
                drop_last=True,
                **dataloader_kwargs,
            )

    def train_step(self, batch):
        start1 = time.time()
        image_latents, model_kwargs, cur_batch_size, n_tokens = self.prepare_model_inputs(batch, self.device)
        duration1 = time.time() - start1

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            loss_dict = self.denoiser.training_losses(self.model_engine, image_latents, model_kwargs, n_tokens=n_tokens)
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times
