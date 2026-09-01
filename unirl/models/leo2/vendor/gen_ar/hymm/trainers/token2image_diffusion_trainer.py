import time
from typing import Union, Dict
from functools import partial

from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
import torch
from torch.utils.data import DataLoader

from hymm.models.diffusion.posemb_layers import get_nd_rotary_pos_embed
from hymm.diffusion import load_denoiser
from hymm.models.visual_encoders import load_vision_model
from hymm.models.autoencoders import load_vae, VAEEncodeOutput

from .base_trainer import BaseTrainer
from hymm.data_kits.siglip_diffusion_decoder_loader import SiglipDiffusionDecoderArrowStream

from hymm.trainers.helpers import dynamic_values_wrapper
from hymm.utils.helpers import as_tuple
from hymm.utils.torch_utils import (
    set_worker_seed_builder,
    PRECISION_TO_TYPE,
    is_torch_tensor,
)


class Token2ImageDiffusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def vae_encode_image_tensor(self, image, sample_type=None, n_tokens=None):
        # ===================================== prepare diffusion =====================================
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
                t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
            elif sample_type == "sample_start":
                t, x_0, x_1 = self.denoiser.sample_start(latents)
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
            t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
            model_t = self.denoiser.get_model_t(t)  # t*1000

            return VAEEncodeOutput(t=t, model_t=model_t, x_0=x_0, x_t=x_t, u_t=u_t, latents=latents)

        return VAEEncodeOutput(latents=latents)

    def build_extra_model(self):
        args = self.args

        self.logger.info("Building Vision Encoder...")
        self.vision_encoder = load_vision_model(
            args.vision_encoder_type,
            args.vision_encoder_precision,
            device=self.device,
            logger=self.logger,
        )
        
        # ====================== Build VAE ========================
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )
        self.vae_generater = torch.Generator(self.device).manual_seed(self.dp_rank)

        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args
        # b x max_num_patches x 768
        pixel_values = batch["pixel_values"].to(device)
        # b x max_num_patches
        pixel_attention_mask = batch["pixel_attention_mask"].to(device)
        # b x 2 (h, w)
        spatial_shapes = batch["spatial_shapes"].to(device)

        vision_encoder_autocast_dtype = PRECISION_TO_TYPE[self.args.vision_encoder_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vision_encoder_autocast_dtype, enabled=vision_encoder_autocast_dtype != torch.float32):
            vision_encoder_output = self.vision_encoder(
                pixel_values=pixel_values,
                attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            # b x max_num_patches x 1152
            latents = vision_encoder_output.last_hidden_state

        # Tokenization
        # batch_size x c x h x w
        image_tensor = batch["image_tensor"].to(device)
        vae_encode_output = self.vae_encode_image_tensor(image_tensor, sample_type="sample")

        # print(f"image_tensor.shape: {image_tensor.shape}")
        # print(f"vae_encode_output: {vae_encode_output.t}, {vae_encode_output.x_t.shape}, {vae_encode_output.x_0.shape}, {vae_encode_output.x_t.shape}, {vae_encode_output.u_t.shape}")

        # ======================================== Build RoPE ======================================
        target_ndim = 2     # n-d RoPE
        ndim = len(vae_encode_output.x_t.shape) - 2    # remove batch and channel dimension
        image_size = list(vae_encode_output.x_t.shape[-ndim:])
        assert all(s % self.model_settings.patch_size == 0 for s in image_size), \
            f"Image size(last {ndim} dimensions) should be divisible by patch size({self.model_settings.patch_size}), " \
            f"but got {image_size}."
        rope_sizes = [s // self.model_settings.patch_size for s in image_size]
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model_settings.hidden_size // self.model_settings.num_heads
        rope_dim_list = self.model_settings.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=self.model_settings.rope_theta,
            use_real=True,
            theta_rescale_factor=1.0,
        )

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            x=vae_encode_output.x_t,
            t=vae_encode_output.model_t,
            cond=latents,
            cond_mask=pixel_attention_mask,
            freqs_cos=freqs_cos,            # [seqlen, head_dim]
            freqs_sin=freqs_sin,            # [seqlen, head_dim]
            return_dict=True,
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=vae_encode_output.t, x0=vae_encode_output.x_0, xt=vae_encode_output.x_t, ut=vae_encode_output.u_t),
        )

        batch_size = latents.shape[0]
        n_tokens = image_tensor.shape[2] * image_tensor.shape[3]

        return model_intput_kwargs, batch_size, n_tokens

    def build_dataloader(self):
        args = self.args
        self.dataset = SiglipDiffusionDecoderArrowStream(
            args=args,
            training_image_size=args.training_image_size,
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
        model_intput_kwargs, cur_batch_size, n_tokens = self.prepare_model_inputs(batch, self.device)
        duration1 = time.time() - start1

        start2 = time.time()
        with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
            loss_dict = self.model_engine(**model_intput_kwargs)
        duration2 = time.time() - start2

        times = {
            "preprocess": duration1,
            "forward": duration2,
        }

        return loss_dict, cur_batch_size, n_tokens, times
