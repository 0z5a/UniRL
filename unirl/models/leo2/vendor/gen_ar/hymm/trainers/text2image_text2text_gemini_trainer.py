import gc
import os
from functools import partial
import torch
from typing import Dict, Union

from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from .helpers import TextImageScalarStates, TextImageCycleStates
from ..constants import C_SCALE
from ..data_kits.samplers import SequentialSampler
from ..models import load_vae
from ..utils.file_utils import safe_dir
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..diffusion import load_denoiser, load_scheduler
from ..data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from ..data_kits.text_image_transfusion_loader import TransfusionTextImageArrowStream
from ..data_kits.text_image_iterator import TextImageBatchIterator
from ..samplers.text2image_transfusion_sampler import Text2ImageTransfusionSampler

gc.set_threshold(7000, 100, 100)

from .transfusion_parallel import tp_sp_decorator

@tp_sp_decorator
class GeminiTrainerAlpha(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def build_dataloader(self):
        args = self.args
        # ==== Image model ====
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
                # TODO: p0 这里的 world_size 用于 MultiResolutionBucketIndexV2 and MultiMultiResolutionBucketIndexV2
                #  的分桶逻辑，不确定是要传入真 world_size 还是 dp_size?
                world_size=1 if args.mix_scale else self.dp_size,
                **args.index_kwargs,
            ),
            debug=False,
            logger=self.logger,
        )
        # Build sampler and data loader
        dataloader_kwargs = dict(
            **args.dataloader_params, worker_init_fn=set_worker_seed_builder(self.dp_rank)
        )
        self.image_sampler = BlockDistributedSampler(
            self.dataset,
            num_replicas=self.dp_size,
            rank=self.dp_rank,
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

    def resume_dataloader(self, ss):
        # The sampler states will be restored in self.shuffle_dataset()
        for sampler in [self.image_sampler, self.text_sampler]:
            assert isinstance(sampler, (BlockDistributedSampler, SequentialSampler)), (
                "Only BlockDistributedSampler and SequentialSampler supports --resume-dataloader."
            )

    def get_states_cls(self, state_type):
        if state_type == "scalar":
            return TextImageScalarStates
        elif state_type == "cycle":
            return TextImageCycleStates
        else:
            raise ValueError(f"Unknown state type: {state_type}")

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

    def build_data_iterator(self):
        self.dataloader = TextImageBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.dp_rank,
            world_size=...,
            text_dataset=self.text_dataset,
            text_sampler=self.text_sampler,
            text_dataloader=self.text_dataloader,
            image_dataset=self.dataset,
            image_sampler=self.image_sampler,
            image_dataloader=self.image_dataloader,
            text_sampling_prob=self.args.text_sampling_prob,
            initial_seed=self.args.global_seed,
            logger=self.logger,
        )

    def get_sampler(self):
        return Text2ImageTransfusionSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine, # use fsdp wrapped module to handle sharding
                            model_settings=self.model_settings,
                            tokenizer=self.dataset.tokenizer,
                            scheduler=load_scheduler(self.args),
                            ),
            rank=self.dp_rank,
            world_size=self.dp_size,
            device=self.device,
            logger=self.val_logger,
        )

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        if "image" in batch:
            inputs = self.prepare_model_image_inputs(batch, device)
        elif "text" in batch:
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
        patch_size = self.args.patch_size
        latents = torch.randn((batch_size, self.args.vae_latent_dim, patch_size, patch_size), device=device)
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

    def prepare_model_image_inputs(self, batch: Dict, device: Union[int, str]):
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
        if self.use_3d_rope:
            freqs_cos = batch["freqs_cos"].to(device)
            freqs_sin = batch["freqs_sin"].to(device)
            model_intput_kwargs.update(dict(
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
            ))
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(model_intput_kwargs, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        return model_intput_kwargs, batch_size, n_tokens

    def shuffle_dataset_and_set_start_index(self, ss):
        # Shuffle and reset index are handled by the mix-data iterator, i.e., self.dataloader
        pass

    def update_train_states(self, ss, cs, batch, batch_size, n_tokens, loss):
        """
        `batch_size` is literal for image batch, but not for text batch. The actual number of `text samples`
        should be returned by text batch. Here one `text sample` represents a record in the text dataset.
        Accurately counting the number of text samples is useful for resume text dataset and sampler from
        a checkpoint.
        """

        # A forward-backward step is counted as one train step.
        ss.add(train_steps=1, epoch_train_steps=1)
        cs.add(log_steps=1, running_loss=loss)
        if "image" in batch:
            # ss.add(consumed_image_samples_per_dp=batch_size)  # Already accumulated in TextImageBatchIterator
            cs.add(running_image_samples=batch_size, running_image_tokens=batch_size * n_tokens)
        elif "text" in batch:
            # Use `text_batch_size` to count samples, while `batch_size` to count tokens (length-aligned).
            text_batch_size = batch["n_samples"].sum().item()
            # ss.add(consumed_text_samples_per_dp=text_batch_size)  # Already accumulated in TextImageBatchIterator
            cs.add(running_text_samples=text_batch_size, running_text_tokens=batch_size * n_tokens)

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
        cum_image_samples = sum([cs_i.running_image_samples for cs_i in all_cs])
        cum_image_tokens = sum([cs_i.running_image_tokens for cs_i in all_cs])
        cum_text_samples = sum([cs_i.running_text_samples for cs_i in all_cs])
        cum_text_tokens = sum([cs_i.running_text_tokens for cs_i in all_cs])
        self.ss.add(
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * cum_image_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * cum_image_tokens / C_SCALE,
            # ==== Image modal ====
            consumed_image_samples_total=cum_image_samples,
            consumed_image_tokens_total=cum_image_tokens,
            # ==== Text modal ====
            consumed_text_samples_total=cum_text_samples,
            consumed_text_tokens_total=cum_text_tokens,
        )
        return cum_image_samples

    def get_events(self, ss, loss):
        log_events = [
            f"Consumed Image Samples: {ss.consumed_image_samples_total:,}",
            f"Consumed Image Tokens: {ss.consumed_image_tokens_total:,}",
            f"Consumed Text Samples: {ss.consumed_text_samples_total:,}",
            f"Consumed Text Tokens: {ss.consumed_text_tokens_total:,}",
        ]
        summary_events = [
            ("Train/Tokens/train_loss", loss, ss.consumed_image_tokens_total + ss.consumed_text_tokens_total),
            ("Train/TotalSamples/train_loss", loss, ss.consumed_image_samples_total + ss.consumed_text_samples_total),
            ("Train/ImageSamples/train_loss", loss, ss.consumed_image_samples_total),
            ("Train/TextTokens/train_loss", loss, ss.consumed_text_tokens_total),
        ]
        return log_events, summary_events
