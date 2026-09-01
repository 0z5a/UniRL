import time
from typing import Union, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip import create_model_from_pretrained

from hymm.diffusion import load_denoiser
from hymm.trainers.base_trainer import BaseTrainer
from hymm.data_kits.visual_audio_diffusion_index_loader import build_visual_audio_diffusion_fixedlen_dataloader
from hymm.constants import C_SCALE
from hymm.trainers.helpers import (
    CycleStates,
    save_checkpoint,
    get_trainable_params,
)
from hymm.utils.torch_utils import (
    profiler_context,
    all_gather_sum,
)


# cpied from https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
def patch_clip(clip_model):
    # a hack to make it output last hidden states
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x, self.attn_mask

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class VisualText2AudioDiffTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        if self.args.use_clip_text_feat:
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")  # same as 'ViT-H-14'
            self.clip_model = create_model_from_pretrained(
                "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
            )
            self.clip_model = patch_clip(self.clip_model).to(self.device)

    @torch.inference_mode()
    def encode_text_clip(self, text: List[str], device) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        # x: (B, L)
        tokens = self.tokenizer(text).to(device)
        return self.clip_model.encode_text(tokens, normalize=True)

    def move_only_tensor_to_device(self, batch, maybe_tensor_name, device):
        if maybe_tensor_name not in batch:
            return None
        maybe_tensor = batch[maybe_tensor_name]
        if isinstance(maybe_tensor, torch.Tensor):
            return maybe_tensor.to(device)
        return maybe_tensor

    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        clip_embedding = batch["clip_embedding"].to(device)
        audio_embedding = batch["audio_token"].to(device)  # [B x N x T] or [B x N x T x F]
        sync_embedding = self.move_only_tensor_to_device(batch, "sync_embedding", device)  # B, num_segments * 8, 768
        drop_visual = self.move_only_tensor_to_device(batch, "drop_visual", device)
        text_embedding = self.move_only_tensor_to_device(batch, "t5_embedding", device)
        audio_attn_mask = self.move_only_tensor_to_device(batch, "audio_attn_mask", device)
        cond_mask = self.move_only_tensor_to_device(batch, "cond_mask", device)

        # Use clip to extract text feature, replacing the t5 feature
        if "caption" in batch and self.args.use_clip_text_feat:
            caption = batch["caption"]
            # [16, 77, 1024], [77, 77]
            text_clip_feat, _ = self.encode_text_clip(caption, device)
            text_embedding = text_clip_feat.detach()
            if text_clip_feat.shape[1] > self.model_settings.t5_length:
                text_embedding = text_embedding[:, : self.model_settings.t5_length]
            if audio_attn_mask is not None:
                cond_mask = torch.ones(
                    (text_embedding.shape[0], text_embedding.shape[1]), dtype=torch.bool, device=self.device
                )
            else:
                cond_mask = None

        batch_size = audio_embedding.shape[0]
        if audio_embedding.ndim >= 3 and audio_embedding.ndim <= 4:
            n_tokens = np.prod(audio_embedding.shape[2:])
        else:
            raise ValueError(f"audio_embedding should have 3 or 4 dimensions but got {audio_embedding.ndim}")

        # ===================================== Pack model kwargs ==================================
        model_kwargs = dict(
            clip_feat=clip_embedding,  # [b, xxx, 768]
            cond=text_embedding,  # [b, xxx, 4096]
            cond_mask=cond_mask,
            audio_mask=audio_attn_mask,
            sync_feat=sync_embedding,
            drop_visual=drop_visual,
            return_dict=True,
        )

        return audio_embedding, model_kwargs, batch_size, n_tokens

    def build_extra_model(self):
        args = self.args

        # Build denoiser
        self.denoiser = load_denoiser(args)

    def build_dataloader(self):
        self.dataloader, self.dataset, self.data_sampler = build_visual_audio_diffusion_fixedlen_dataloader(
            self.args, self.rank, self.world_size
        )

    def before_train(self):
        args = self.args
        try:
            iters_per_epoch = len(self.dataloader) // self.grad_accu_steps
        except NotImplementedError:
            iters_per_epoch = 0
        self.params_count = self.model.params_count()
        self.logger.info("****************************** Running training ******************************")
        self.logger.info(f"  Number GPUs:               {self.world_size}")
        for k, v in self.params_count.items():
            self.logger.info(f"  Number {k} parameters:   {v:,}")
        self.logger.info(
            f"  Number trainable params:   {sum(p.numel() for p in get_trainable_params(self.model, args.training_parts)):,}"
        )
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Updates per epoch:         {iters_per_epoch:,}(0 means unknown)")
        self.logger.info(f"  Batch size per device:     {self.micro_batch_size}")
        self.logger.info(f"  Batch size all device:     {self.global_batch_size}")
        self.logger.info(f"  Gradient Accu steps:       {self.grad_accu_steps}")
        self.logger.info(f"  Training epochs:           {self.ss.epoch}/{args.max_epochs}")
        self.logger.info(f"  Training total steps:      {self.ss.update_steps:,}/{args.max_training_steps:,}")
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Main model precision:      {args.precision}")
        self.logger.info(f"  Using EMA model:           {args.use_ema}")
        self.logger.info(f"  Using Distributed EMA model:           {args.distributed_ema}")
        if args.use_ema:
            self.logger.info(f"  EMA precision:             {args.ema_precision}")
            self.logger.info(f"  EMA decay:                 {self.ema.decay if args.use_ema else None}")
            self.logger.info(f"  EMA warmup power:          {self.ema.power if args.use_ema else None}")
        self.logger.info("------------------------------------------------------------------------------")
        self.logger.info(f"  Experiment directory:      {self.exp_dir}")
        self.logger.info("*******************************************************************************")

    def after_train(self):
        self.logger.info("Training Finished!")

    def train_loop(self):
        args = self.args
        target_dtype = None
        autocast_enabled = False
        if self.model_engine.bfloat16_enabled():
            self.logger.info("using bf16 for training")
            target_dtype = torch.bfloat16
            autocast_enabled = True
        elif self.model_engine.fp16_enabled():
            target_dtype = torch.half
            autocast_enabled = True

        self.model_engine.train()

        if args.init_save:
            save_checkpoint(args, self.rank, self.logger, self.model_engine, self.ema, self.ss, self.ckpt_dir)

        # Training loop
        start_epoch = self.ss.epoch
        finished = False
        self.ss.current_run_update_steps = 0
        # for resuming
        start_index = self.ss.consumed_samples_total % self.data_sampler.total_size
        for epoch in range(start_epoch, args.max_epochs):
            self.logger.info(f"Start random shuffle(seed={args.global_seed + epoch})")

            self.data_sampler.set_epoch(epoch)
            self.dataset.shuffle(
                seed=args.global_seed + epoch,
                fast=args.fast_shuffle,
            )
            self.data_sampler.start_index = start_index
            self.logger.info(f"End of random shuffle")
            try:
                self.logger.info(f"  Iters left this epoch: {len(self.dataloader):,}")
            except NotImplementedError:
                self.logger.info(f"  Iters left this epoch: unknown")

            with profiler_context(args.profile, self.exp_dir, worker_name=f"Rank_{self.rank}") as prof:
                self.logger.info(f"Beginning epoch {epoch}...")
                # Define cycle states, which accumulate the training information between log_steps.
                cs = CycleStates()
                start_time = time.time()

                for batch in self.dataloader:
                    audio_embedding, model_kwargs, cur_batch_size, n_tokens = self.prepare_model_inputs(
                        batch, self.device
                    )
                    # A forward-backward step
                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                        loss_dict = self.denoiser.training_losses(
                            self.model_engine, audio_embedding, model_kwargs, n_tokens=n_tokens
                        )

                    loss = loss_dict["loss"].mean()
                    self.model_engine.backward(loss)
                    # Update accumulated states
                    self.ss.add(train_steps=1, epoch_train_steps=1, epoch_consumed_samples_per_dp=cur_batch_size)
                    # We enable `is_update_step` if the current step is the gradient accumulation boundary.
                    is_update_step = self.ss.train_steps % self.grad_accu_steps == 0
                    if is_update_step:
                        self.ss.add(update_steps=1, epoch_update_steps=1, current_run_update_steps=1)
                    self.ss.lr = self.optimizer.param_groups[0]["lr"]
                    # Update model parameters at the step of gradient accumulation.
                    self.model_engine.step(lr_kwargs={"last_batch_iteration": self.lr_helper(self.ss.update_steps + 1)})
                    if self.ss.update_steps >= args.max_training_steps:
                        # Enter stopping routine if max steps reached after this step.
                        finished = True

                    # Update EMA model at the step of main model parameters update.
                    if args.use_ema and is_update_step:
                        self.ema.update(self.model_engine.module)

                    # Log training information:
                    cs.add(
                        log_steps=1,
                        running_loss=loss.item(),
                        running_samples=cur_batch_size,
                        running_tokens=cur_batch_size * n_tokens,
                    )
                    if is_update_step and self.ss.update_steps % args.log_every == 0:
                        # Reduce loss history over all processes:
                        avg_loss = all_gather_sum(cs.running_loss / cs.log_steps, self.device) / self.world_size
                        cum_samples = all_gather_sum(cs.running_samples, self.device)
                        cum_tokens = all_gather_sum(cs.running_tokens, self.device)
                        # Measure training speed:
                        torch.cuda.synchronize()
                        end_time = time.time()
                        steps_per_sec = cs.log_steps / self.grad_accu_steps / (end_time - start_time)
                        seconds_per_step = (end_time - start_time) / (cs.log_steps / self.grad_accu_steps)
                        samples_per_sec = cum_samples / (end_time - start_time)
                        self.ss.add(
                            epoch_consumed_samples_total=cum_samples,
                            consumed_samples_total=cum_samples,
                            consumed_tokens_total=cum_tokens,
                            consumed_computations_attn=6 * self.params_count["attn+mlp"] * cum_tokens / C_SCALE,
                            consumed_computations_total=6 * self.params_count["total"] * cum_tokens / C_SCALE,
                        )
                        log_events = [
                            f"Train Loss: {avg_loss:.4f}",
                            f"Lr: {self.optimizer.param_groups[0]['lr']:.6g}",
                            f"Steps/Sec: {steps_per_sec:.2f}",
                            f"Sec/Step: {seconds_per_step:.2f}",
                            f"Samples/Sec: {int(samples_per_sec):d}",
                            f"Consumed Samples: {self.ss.consumed_samples_total:,}",
                            f"Consumed Tokens: {self.ss.consumed_tokens_total:,}",
                        ]
                        summary_events = [
                            ("Train/Steps/train_loss", avg_loss, self.ss.update_steps),
                            ("Train/Steps/steps_per_sec", steps_per_sec, self.ss.update_steps),
                            ("Train/Steps/samples_per_sec", int(samples_per_sec), self.ss.update_steps),
                            ("Train/Tokens/train_loss", avg_loss, self.ss.consumed_tokens_total),
                            ("Train/ComputationsAttn/train_loss", avg_loss, self.ss.consumed_computations_attn),
                            ("Train/ComputationsTotal/train_loss", avg_loss, self.ss.consumed_computations_total),
                        ]
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
                        save_checkpoint(
                            args,
                            self.rank,
                            self.logger,
                            self.model_engine,
                            self.ema,
                            self.ss,
                            self.ckpt_dir,
                        )

                    if prof:
                        prof.step()

                    if finished:
                        self.logger.info(f"Finished and breaking loop at step={self.ss.update_steps}.")
                        break

                if finished:
                    self.logger.info(f"Finished and breaking loop at epoch={epoch}.")
                    break

                # Reset epoch states
                self.ss.epoch += 1
                self.ss.epoch_train_steps = 0
                self.ss.epoch_update_steps = 0
                start_index = 0
                if self.data_sampler.start_index != 0:
                    self.data_sampler.start_index = 0
