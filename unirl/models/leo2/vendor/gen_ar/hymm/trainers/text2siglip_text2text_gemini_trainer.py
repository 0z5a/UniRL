import gc
import os
from functools import partial
import torch
from typing import Dict, Union

from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader

from ..data_kits.samplers import SequentialSampler
from ..models.visual_encoders import load_vision_model
from ..utils.file_utils import safe_dir
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..diffusion import load_denoiser
from ..data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from ..data_kits.text_siglip_transfusion_loader import TransfusionTextSiglipArrowStream
from .text2image_text2text_gemini_trainer import GeminiTrainerAlpha


class Text2SiglipText2TextTrainer(GeminiTrainerAlpha):
    def __init__(self, args):
        super().__init__(args)
    
    def build_dataloader(self):
        args = self.args
        # ==== Image model ====
        self.dataset = TransfusionTextSiglipArrowStream(
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
        self.image_sampler = BlockDistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
            align=self.micro_batch_size,
        )
        self.image_dataloader = DataLoader(
            self.dataset,
            batch_size=self.micro_batch_size,
            sampler=self.image_sampler,
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs,
        )

        # ==== Text model ====
        # Set the number of dummy tokens for text sequences
        self.dummy_number = 1 + (2 if self.args.add_iw_ih_token else 0) + (
            1 if self.args.add_timestep_token else 0)
        t2t_max_length = args.text_token_length + args.image_token_length + 1 - self.dummy_number

        self.text_dataset = TextArrowStream(
            args=args,
            index_file=args.text_index_file,
            t2t_text_token_length=t2t_max_length,
            tokenizer_name=args.tokenizer_name,
            index_kwargs=args.lm_index_kwargs,
            logger=self.logger,
        )
        self.text_sampler = SequentialSampler(self.text_dataset)
        self.text_batch_sampler = MaxLengthBatchSampler(
            self.text_dataset.index_manager, self.text_sampler, batch_size=self.micro_batch_size,
            max_length=t2t_max_length,
            length_getter=lambda idm, ind: idm.get_attribute(ind, 'hy_ids_length'),
        )
        self.text_dataloader = DataLoader(self.text_dataset, batch_sampler=self.text_batch_sampler, **dataloader_kwargs)


    def build_extra_model(self):
        args = self.args

        self.logger.info("Building Vision Encoder...")
        self.vision_encoder = load_vision_model(
            args.vision_encoder_type,
            args.vision_encoder_precision,
            device=self.device,
            logger=self.logger,
        )
        self.compression_net = None
        if hasattr(args, "compression_net_type"):
            from hymm.models.visual_encoders.siglip2.compression import CompressionNet
            self.compression_net = CompressionNet(
                type=args.compression_net_type,
                device=self.device,
                dtype=PRECISION_TO_TYPE["bf16"],
            )
        # ====================== Build denoise scheduler ========================
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)

    def get_sampler(self):
        return None

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        data_type = batch["dtype"][0]
        if data_type == "t2i":
            inputs = self.prepare_model_siglip_inputs(batch, device)
        elif data_type == "lm":
            inputs = self.prepare_model_text_inputs(batch, device)
        else:
            raise ValueError("Unknown batch type, expected 'image' or 'text'.")
        return inputs

    def prepare_model_text_inputs(self, batch: Dict, device: Union[int, str]):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)

        batch_size, n_tokens = tokens.shape

        # ==== Add dummy tokens to avoid hanging when deepspeed all-reduce gradients ====
        # Different modalities connect with different model parameters in the computation graph.
        # If different batches correspond to different modalities, deepspeed cannot correctly perform
        # all-reduce gradients. Therefore, we need to pad dummy image tokens to text sequences to
        # maintain consistent activated model parameters.
        dummy_tokens = torch.zeros((batch_size, self.dummy_number), dtype=tokens.dtype, device=device)
        dummy_target_tokens = (-100) * torch.ones((batch_size, self.dummy_number), dtype=tokens.dtype, device=device)
        tokens = torch.cat([tokens, dummy_tokens], dim=1)
        target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

        image_mask = torch.zeros_like(tokens, dtype=torch.float32, device=device)
        image_mask[:, -1] = 1.0
        # Add iw,ih,timestep tokens to touch their embedding layers (include learnable parameters).
        if self.args.add_iw_ih_token:
            iw_ih_scatter_index = torch.tensor([[n_tokens, n_tokens + 1]] * batch_size, dtype=torch.long, device=device)
            iw_ih_scatter_src = torch.tensor([[2, 2]] * batch_size, dtype=torch.long, device=device)
        if self.args.add_timestep_token:
            timestep_scatter_index = torch.tensor([[n_tokens + 2]] * batch_size, dtype=torch.long, device=device)
        n_tokens += self.dummy_number

        text_mask = torch.cat([text_mask, torch.zeros_like(dummy_tokens, dtype=torch.float32, device=device)], dim=1)

        # Mixed attention mask
        _, n_tokens = tokens.shape
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)

        # Process dummy image tokens
        in_channel = self.args.vision_encoder_latent_dim
        if hasattr(self.args, "compression_net_type"):
            from hymm.models.visual_encoders.siglip2.compression import CompressionNet
            in_channel = CompressionNet.config[self.args.compression_net_type]["out_channels"]
        latents = torch.randn((batch_size, 1, in_channel), device=device)
        t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        model_t = self.denoiser.get_model_t(t) # t*1000

        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            attention_mask=attention_mask,  # [b, 512, 512]
            x_t=x_t,
            t=model_t,
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
            text_mask=text_mask,
            image_mask=image_mask,
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="text",
        )
        if self.args.add_iw_ih_token:
            model_intput_kwargs.update({
                "iw_ih_scatter_index": iw_ih_scatter_index,  # [b, 2]
                "iw_ih_scatter_src": iw_ih_scatter_src,  # [b, 2]
            })
        if self.args.add_timestep_token:
            model_intput_kwargs.update({
                "timestep_scatter_index": timestep_scatter_index,  # [b, 1]
            })
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(model_intput_kwargs, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_siglip_inputs(self, batch: Dict, device: Union[int, str]):
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
        # image = batch["image"].to(device)

        # b x 256 x 768
        pixel_values = batch["pixel_values"].to(device)
        # b x 256
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
            # b x 256 x 1152
            latents = vision_encoder_output.last_hidden_state

        if self.compression_net is not None:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # b x 256 x 1152 -> b x 256 x 64/128
                latents = self.compression_net(latents)

        t, x_0, x_1 = self.denoiser.sample(latents, None)
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
            data_type="image",
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
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(model_intput_kwargs, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        return model_intput_kwargs, batch_size, n_tokens
