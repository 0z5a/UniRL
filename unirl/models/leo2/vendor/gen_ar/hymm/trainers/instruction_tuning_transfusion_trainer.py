from functools import partial
import os
import torch
from typing import Dict, Union
import einops
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader

from ..models import load_vae
from ..diffusion import load_denoiser
from .helpers import dynamic_values_wrapper
from ..utils.helpers import as_tuple
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..utils.file_utils import safe_dir
from .base_trainer import BaseTrainer
from ..data_kits.instruction_tuning_transfusion_loader import InstructionTuningTransfusionArrowStream, InstructionTuningTransfusionArrowStream2


class InstructionTuningTransfusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
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

    def build_dataloader(self):
        args = self.args
        self.dataset = InstructionTuningTransfusionArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            text_token_length=args.text_token_length,
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

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        # text: 256 + 1, image: 256 * 2, total: 769, 769 - 1 = 768
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_token is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_token
        target_token = batch["target_token"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        tgt_image_mask = batch["tgt_image_mask"][:, :-1].contiguous().to(device)

        # build attention mask
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(batch_size, 1, 1)
        image_mask_1 = src_image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(1, 2)
        image_mask_3 = tgt_image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
        image_mask_4 = image_mask_3.transpose(1, 2)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (image_mask_3.bool() & image_mask_4.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(1)

        # ===================================== prepare diffusion =====================================
        tgt_image = batch["tgt_image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            tgt_latents = self.vae.encode(tgt_image).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                tgt_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
            else:
                tgt_latents.mul_(self.vae.config.scaling_factor)
        
        t, x_0, x_1 = self.denoiser.sample(tgt_latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        model_t = self.denoiser.get_model_t(t) # t*1000

        src_image = batch["src_image"].to(device)
        
        self.src_condition_type = self.args.get("src_condition_type", ["vae"])
        if isinstance(self.src_condition_type, str):
            self.src_condition_type = self.src_condition_type.split("_cat_")
        else: 
            assert isinstance(self.src_condition_type, list), f"src_condition_type should be a list,e.g, ['face_embed'], ['vae', 'face_embed'], but got {type(self.src_condition_type)}"

        if "vae" not in self.src_condition_type:
            src_image = None
        
        if src_image is not None:
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                src_latents = self.vae.encode(src_image).latent_dist.sample()
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                else:
                    src_latents.mul_(self.vae.config.scaling_factor)

            src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
            src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
            src_model_t = self.denoiser.get_model_t(src_t) # t*1000
        else:
            src_x = None
            src_model_t = None

        if "face_embed" in self.src_condition_type:
            src_face_embedding = batch["src_face_embedding"].to(self.device)
            src_face_embedding = einops.rearrange(src_face_embedding, 'b c -> b c 1 1')
        else:
            src_face_embedding = None
        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 768]
            x_t=x_t,  # [b, c, h, w]
            t=model_t, # [b]
            diffusion_loss_fn=diffusion_loss_fn,
            src_x=src_x,  # [b, c, h, w]
            src_t=src_model_t,  # [b]
            src_image_mask=src_image_mask,  # [b, 768]
            target=target_token,  # [b, 768]
            text_mask=text_mask,  # [b, 768]
            image_mask=tgt_image_mask,  # [b, 768]
            attention_mask=attention_mask, # [b, 768, 768]
            image_loss_weight=self.args.image_loss_weight,
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })
        if src_face_embedding is not None:
            model_intput_kwargs.update({
                "src_face_embedding": src_face_embedding.to(self.device),
            })
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_intput_kwargs, batch_size, n_tokens


class InstructionTuningTransfusionTrainer2(InstructionTuningTransfusionTrainer):

    def build_dataloader(self):
        args = self.args
        self.dataset = InstructionTuningTransfusionArrowStream2(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            text_token_length=args.text_token_length,
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
                collate_fn=self.dataset.collate_fn,
                **dataloader_kwargs,
            )

    def prepare_model_inputs(
            self,
            batch: Dict,
            device: Union[int, str],
    ):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        # text: 256 + 1, image: 256 * 2, total: 769, 769 - 1 = 768
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_token is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_token
        target_token = batch["target_token"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        tgt_image_mask = batch["tgt_image_mask"][:, :-1].contiguous().to(device)

        # build attention mask
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        # causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        # causal_mask = causal_mask.view(1, n_tokens, n_tokens).repeat(batch_size, 1, 1)
        # image_mask_1 = src_image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
        # image_mask_2 = image_mask_1.transpose(1, 2)
        # image_mask_3 = tgt_image_mask.view(batch_size, 1, n_tokens).repeat(1, n_tokens, 1)
        # image_mask_4 = image_mask_1.transpose(1, 2)
        # attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (
        #             image_mask_3.bool() & image_mask_4.bool())
        # # unsqueeze for attention head dim
        # attention_mask = attention_mask.unsqueeze(1)
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        tgt_image = batch["tgt_image"].to(device)
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            tgt_latents = self.vae.encode(tgt_image).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                tgt_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
            else:
                tgt_latents.mul_(self.vae.config.scaling_factor)

        t, x_0, x_1 = self.denoiser.sample(tgt_latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        model_t = self.denoiser.get_model_t(t)  # t*1000

        batch_src_xs = []
        batch_src_model_ts = []
        for src_images in batch["src_images"]:
            src_xs = []
            src_model_ts = []
            for src_image in src_images:
                src_image = src_image.to(device)
                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                    src_latents = self.vae.encode(src_image[None]).latent_dist.sample()
                    if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                        src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                    else:
                        src_latents.mul_(self.vae.config.scaling_factor)

                src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
                src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
                src_model_t = self.denoiser.get_model_t(src_t)  # t*1000
                src_xs.append(src_x)
                src_model_ts.append(src_model_t)
            batch_src_xs.append(torch.cat(src_xs))
            batch_src_model_ts.append(torch.cat(src_model_ts))

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 768]
            x_t=x_t,  # [b, c, h, w]
            t=model_t,  # [b]
            diffusion_loss_fn=diffusion_loss_fn,
            src_x=batch_src_xs,  # [b, c, h, w]
            src_t=batch_src_model_ts,  # [b]
            src_image_mask=src_image_mask,  # [b, 768]
            target=target_token,  # [b, 768]
            text_mask=text_mask,  # [b, 768]
            image_mask=tgt_image_mask,  # [b, 768]
            attention_mask=attention_mask,  # [b, 768, 768]
            image_loss_weight=self.args.image_loss_weight,
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })
        if self.use_3d_rope:
            freqs_cos = batch["freqs_cos"].to(device)
            freqs_sin = batch["freqs_sin"].to(device)
            model_intput_kwargs.update(dict(
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            ))

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_intput_kwargs, batch_size, n_tokens
