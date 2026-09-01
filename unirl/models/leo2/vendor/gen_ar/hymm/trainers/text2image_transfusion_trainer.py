import os
from functools import partial
import torch
from typing import Dict, Union

from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..models import load_vae
from ..utils.file_utils import safe_dir
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..diffusion import load_denoiser, load_scheduler
from ..data_kits.text_image_transfusion_loader import TransfusionTextImageArrowStream
from .helpers import dynamic_values_wrapper
from ..samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler
from ..utils.helpers import as_tuple


class Text2ImageTransfusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def build_dataloader(self):
        args = self.args
        self.dataset = TransfusionTextImageArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            text_token_length=args.text_token_length,
            uncond_p=args.uncond_p,
            tokenizer_name=args.tokenizer_name,
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
                    align=self.micro_batch_size,
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

    def build_extra_model(self):
        args = self.args

        self.logger.info("Building VAE...")
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )

        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

        self.use_3d_rope = args.get('rope_type', 'default') in ['3d', '3d-interleave']

    def get_sampler(self):
        return Text2ImageTransfusionSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine.module,
                            model_settings=self.model_settings,
                            tokenizer=self.dataset.tokenizer,
                            scheduler=load_scheduler(self.args),
                            ),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            logger=self.val_logger,
        )

    def strip_pad_tokens(self, batch):
        # Find the common number of <pad> tokens in the batch
        max_pads = batch["tokens"].eq(self.dataset.tokenizer.special_token_map["<pad>"]).sum(dim=1).min().item()
        if max_pads > 0:
            # Remove the padding from the target tokens
            batch["tokens"] = batch["tokens"][:, :-max_pads]
            batch["target_token"] = batch["target_token"][:, :-max_pads]
            batch["text_mask"] = batch["text_mask"][:, :-max_pads]
            batch["image_mask"] = batch["image_mask"][:, :-max_pads]
            if self.use_3d_rope:
                batch["freqs_cos"] = batch["freqs_cos"][:, :-max_pads]
                batch["freqs_sin"] = batch["freqs_sin"][:, :-max_pads]
        return batch

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # Strip ending pad tokens for training acceleration
        if self.args.get('strip_pad_tokens', False):
            batch = self.strip_pad_tokens(batch)

        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        # text: 256 + 1, image: 256, total: 513, 513 - 1 = 512
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_token is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_token
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # build attention mask
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(batch_size, 1, 1)
        image_mask_1 = image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(1, 2)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(1)

        # ===================================== prepare diffusion =====================================
        image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            vae_encode_result = self.vae.encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                latents.sub_(self.vae.config.shift_factor) #.mul_(self.vae.config.scaling_factor)
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                latents.mul_(self.vae.config.scaling_factor)

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        model_t = self.denoiser.get_model_t(t) # t*1000

        # ===================================== Pack model kwargs =====================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            x_t=x_t,  # [b, c, h, w]
            t=model_t, # [b]
            diffusion_loss_fn=diffusion_loss_fn,
            target=target_tokens,  # [b, 512]
            text_mask=text_mask,  # [b, 512]
            image_mask=image_mask,  # [b, 512]
            attention_mask=attention_mask, # [b, 512, 512]
            image_loss_weight=self.args.image_loss_weight,
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })
        if self.args.add_timestep_token:
            assert "timestep_scatter_index" in batch, "timestep_scatter_index is required for adding timestep token"
            model_intput_kwargs.update({
                "timestep_scatter_index": batch["timestep_scatter_index"].to(device),  # [b, 1]
            })
        if self.use_3d_rope:
            freqs_cos = batch["freqs_cos"].to(device)
            freqs_sin = batch["freqs_sin"].to(device)
            model_intput_kwargs.update(dict(
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            ))
        
        return model_intput_kwargs, batch_size, n_tokens
