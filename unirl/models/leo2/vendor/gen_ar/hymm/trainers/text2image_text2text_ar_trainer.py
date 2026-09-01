import os
from functools import partial
import torch
from typing import Dict, Union

from einops import rearrange
from index_kits.sampler import BlockDistributedSampler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..constants import C_SCALE
from ..data_kits.samplers import SequentialSampler
from ..models import load_vae
from ..utils.file_utils import safe_dir
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..diffusion import load_denoiser, load_scheduler
from ..data_kits.text_loader import TextArrowStream, MaxLengthBatchSampler
from ..data_kits.text_image_loader import TextImageArrowStream
from ..data_kits.text_image_iterator import TextImageBatchIterator


class Text2imageText2textTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def build_dataloader(self):
        args = self.args
        # ==== Image model ====
        self.dataset = TextImageArrowStream(
            args=args,
            index_file=args.index_file,
            training_image_size=args.training_image_size,
            image_token_length=args.image_token_length,
            image_token_offset=args.get("image_token_offset", 0),
            use_pre_extracted_token=args.get("use_pre_extracted_token", False),
            text_token_length=args.text_token_length,
            uncond_p=args.uncond_p,
            tokenizer_name=args.tokenizer_name,
            multireso=args.multireso,
            add_iw_ih_token=args.add_iw_ih_token,
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
        self.dummy_number = 1 + (2 if self.args.add_iw_ih_token else 0)
        t2t_max_length = args.text_token_length + 1

        self.text_dataset = TextArrowStream(
            args=args,
            index_file=args.text_index_file,
            t2t_text_token_length=t2t_max_length,
            tokenizer_name=args.tokenizer_name,
            logger=self.logger,
        )
        self.text_sampler = SequentialSampler(self.text_dataset)
        self.text_batch_sampler = MaxLengthBatchSampler(
            self.text_dataset.index_manager, self.text_sampler, batch_size=self.micro_batch_size,
            max_length=t2t_max_length, length_col='hy_ids_length',
        )
        self.text_dataloader = DataLoader(self.text_dataset, batch_sampler=self.text_batch_sampler, **dataloader_kwargs)

        self.dataloader = TextImageBatchIterator(
            args=args,
            rank=self.rank,
            world_size=self.world_size,
            text_dataset=self.text_dataset,
            text_sampler=self.text_sampler,
            text_dataloader=self.text_dataloader,
            image_dataset=self.dataset,
            image_sampler=self.image_sampler,
            image_dataloader=self.image_dataloader,
            text_sampling_prob=args.text_sampling_prob,
            initial_seed=args.global_seed,
            logger=self.logger,
        )

    def resume_dataloader(self, ss):
        # The sampler states will be restored in self.shuffle_dataset()
        for sampler in [self.image_sampler, self.text_sampler]:
            assert isinstance(sampler, (BlockDistributedSampler, SequentialSampler)), (
                "Only BlockDistributedSampler and SequentialSampler supports --resume-dataloader."
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

        self.use_3d_rope = args.get('rope_type', 'default') in ['3d', '3d-interleave']


    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        if "image" in batch:
            inputs = self.prepare_model_image_inputs(batch, device)
        elif "text" in batch:
            inputs = self.prepare_model_text_inputs(batch, device)
        else:
            raise ValueError("Unknown batch type, expected 'image' or 'text'.")
        return inputs

    def prepare_model_text_inputs(self, batch: Dict, device: Union[int, str]):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)

        batch_size, n_tokens = tokens.shape

        # ==== Add dummy tokens to avoid hanging when deepspeed all-reduce gradients ====
        # Different modalities connect with different model parameters in the computation graph.
        # If different batches correspond to different modalities, deepspeed cannot correctly perform
        # all-reduce gradients. Therefore, we need to pad dummy image tokens to text sequences to
        # maintain consistent activated model parameters.
        dummy_tokens = torch.zeros((batch_size, self.dummy_number), dtype=tokens.dtype, device=device)
        dummy_tokens[:, -1] = self.args.image_token_offset  # 设置为一个任意的image token index
        tokens = torch.cat([tokens, dummy_tokens], dim=1)
        target_tokens = torch.cat([target_tokens, dummy_tokens], dim=1)
        # image_mask = torch.zeros_like(tokens, dtype=torch.float32, device=device)
        # image_mask[:, -1] = 1.0
        if self.args.add_iw_ih_token:
            iw_ih_scatter_index = torch.tensor([[n_tokens, n_tokens + 1]] * batch_size, dtype=torch.long, device=device)
            iw_ih_scatter_src = torch.tensor([[1, 1]] * batch_size, dtype=torch.long, device=device)
        n_tokens += self.dummy_number

        target_tokens[target_tokens == self.dataset.tokenizer.special_token_map["<pad>"]] = -100
        target_tokens[target_tokens == self.dataset.tokenizer.special_token_map["<iw>"]] = -100
        target_tokens[target_tokens == self.dataset.tokenizer.special_token_map["<ih>"]] = -100

        model_intput_kwargs = dict(
            idx=tokens,  # [b, 1280]
            target=target_tokens,  # [b, 1280]
            image_loss_weight=0.0,
            data_type="text",
        )

        if self.args.add_iw_ih_token:
            model_intput_kwargs.update({
                "iw_ih_scatter_index": iw_ih_scatter_index,  # [b, 2]
                "iw_ih_scatter_src": iw_ih_scatter_src,  # [b, 2]
            })
       
        return model_intput_kwargs, batch_size, n_tokens

    def prepare_model_image_inputs(self, batch: Dict, device: Union[int, str]):
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))

        # Online image tokenization
        use_pre_extracted_token = self.args.get("use_pre_extracted_token", False)
        if not use_pre_extracted_token:
            image = batch["image"].to(device)
            vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                    image_tokens = self.vae.vq_encode(image).long()
            image_tokens = image_tokens + self.args.image_token_offset
            if len(image_tokens.shape) > 2:
                image_tokens = rearrange(image_tokens, "b h w -> b (h w)")

            # text: 256 + 1, image: 1024, total: 1281, 1281 - 1 = 1280
            tokens = batch["tokens"].contiguous().to(device)
            for i in range(tokens.shape[0]):
                tokens[i, tokens[i] == self.dataset.tokenizer.special_token_map["<img>"]] = image_tokens[i]
            
            target_token = tokens.clone()
            target_token[target_token == self.dataset.tokenizer.special_token_map["<pad>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<cfg>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<iw>"]] = -100
            target_token[target_token == self.dataset.tokenizer.special_token_map["<ih>"]] = -100
        else:
            # text: 256 + 1, image: 1024, total: 1281, 1281 - 1 = 1280
            tokens = batch["tokens"]
            target_token = batch["target_token"]
        
        tokens = tokens[:, :-1].contiguous().to(device)
        target_token = target_token[:, 1:].contiguous().to(device)
        text_loss_mask = batch["text_loss_mask"][:, 1:].contiguous().to(device)
        image_loss_mask = batch["image_loss_mask"][:, 1:].contiguous().to(device)

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 1280]
            target=target_token,  # [b, 1280]
            text_loss_mask=text_loss_mask,  # [b, 1280]
            image_loss_mask=image_loss_mask,  # [b, 1280]
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_intput_kwargs, batch_size, n_tokens

    def shuffle_dataset_and_set_start_index(self, ss):
        # Assign ScalarState to dataloader for maintain the state of epoch-related attributes.
        self.dataloader.ss = ss
        # shuffle_dataset will be executed only once in the first epoch.
        # The epoch loop is useless because the TextImageBatchIterator is infinite.
        self.dataloader.shuffle_and_initialize_index()

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
            ss.add(epoch_consumed_samples_per_dp=batch_size)
            cs.add(running_samples=batch_size, running_tokens=batch_size * n_tokens)
        elif "text" in batch:
            # Use `text_batch_size` to count samples, while `batch_size` to count tokens (length-aligned).
            text_batch_size = batch["n_samples"].sum().item()
            ss.add(epoch_consumed_text_samples_per_dp=text_batch_size)
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
        cum_samples = sum([cs_i.running_samples for cs_i in all_cs])
        cum_tokens = sum([cs_i.running_tokens for cs_i in all_cs])
        cum_text_samples = sum([cs_i.running_text_samples for cs_i in all_cs])
        cum_text_tokens = sum([cs_i.running_text_tokens for cs_i in all_cs])
        self.ss.add(
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * cum_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * cum_tokens / C_SCALE,
            # ==== Image modal ====
            epoch_consumed_samples_total=cum_samples,
            consumed_samples_total=cum_samples,
            consumed_tokens_total=cum_tokens,
            # ==== Text modal ====
            epoch_consumed_text_samples_total=cum_text_samples,
            consumed_text_samples_total=cum_text_samples,
            consumed_text_tokens_total=cum_text_tokens,
        )
        return cum_samples

    def get_events(self, ss, loss):
        log_events = [
            f"Consumed Image Samples: {ss.consumed_samples_total:,}",
            f"Consumed Image Tokens: {ss.consumed_tokens_total:,}",
            f"Consumed Text Samples: {ss.consumed_text_samples_total:,}",
            f"Consumed Text Tokens: {ss.consumed_text_tokens_total:,}",
        ]
        summary_events = [
            ("Train/Tokens/train_loss", loss, ss.consumed_tokens_total + ss.consumed_text_tokens_total),
            ("Train/TotalSamples/train_loss", loss, ss.consumed_samples_total + ss.consumed_text_samples_total),
            ("Train/ImageSamples/train_loss", loss, ss.consumed_samples_total),
            ("Train/TextTokens/train_loss", loss, ss.consumed_text_tokens_total),
        ]
        return log_events, summary_events
