import os
from typing import Dict, Union

import torch
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader
from einops import rearrange

from .helpers import dynamic_values_wrapper
from ..models import load_vae
from ..utils.file_utils import safe_dir
from ..data_kits.janus_text_image_loader import JanusTextImageArrowStream
from .text2image_ar_trainer import Text2ImageARTrainer
from ..utils.helpers import as_tuple
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE


class Text2ImageJanusCoTTrainer(Text2ImageARTrainer):
    def __init__(self, args):
        super().__init__(args)
    

    def build_extra_model(self):
        args = self.args
        use_pre_extracted_token = self.args.get("use_pre_extracted_token", False)
        if not use_pre_extracted_token:
            self.vae = load_vae(
                args.vae_type,
                args.vae_precision,
                device=self.device,
                logger=self.logger,
            )
            self.image_token_offset = args.image_token_offset
            self.logger.info(f"Image token offset: {self.image_token_offset}")


    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        """
        Prepare model inputs for training, including tokenized text and images.
        """
        
        # Save training data for debugging
        cur_step = self.ss.current_run_update_steps
        if cur_step < self.args.save_n_training_data and self.rank < 8:
            check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
            torch.save(batch, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))

        # Retrieve special token IDs
        img_token_id = self.dataset.tokenizer.img_token_id
        gen_img_token_id = self.dataset.tokenizer.gen_img_token_id

        # Handle image tokenization
        use_pre_extracted_token = self.args.get("use_pre_extracted_token", False)
        if not use_pre_extracted_token:
            # Process images
            images = batch["images"].to(device)  # Shape: [B, N, C, H, W]
            batch_size, max_n_imgs_per_cot = images.shape[:2]
            images = images.view(-1, *images.shape[2:])  # Flatten to [B * N, C, H, W]

            # Encode images using VAE
            vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                    images_tokens = self.vae.vq_encode(images).long().detach()
                    images_tokens += self.image_token_offset

            # Rearrange image tokens if necessary
            if len(images_tokens.shape) > 2:
                images_tokens = rearrange(images_tokens, "(b n) h w -> b n (h w)", b=batch_size)

            # Replace <gen_img> tokens with actual image tokens
            tokens = batch["tokens"].contiguous().to(device)
            for i in range(tokens.shape[0]):
                enable_think_mode = self.args.get("enable_think_mode", False)
                if enable_think_mode:
                    eff_image_token = images_tokens[i, :(batch["eff_images_num"] + 1)].view(-1)
                else:
                    eff_image_token = images_tokens[i, :batch["eff_images_num"]].view(-1)
                tokens[i, tokens[i] == gen_img_token_id] = eff_image_token

            target_token = tokens.clone()
        else:
            raise NotImplementedError("Pre-extracted token handling is not implemented.")

        # Prepare input and target tokens
        tokens = tokens[:, :-1].contiguous().to(device)
        target_token = target_token[:, 1:].contiguous().to(device)
        text_loss_mask = batch["text_loss_mask"][:, 1:].contiguous().to(device)
        image_loss_mask = batch["image_loss_mask"][:, 1:].contiguous().to(device)

        # Pack model input kwargs
        model_input_kwargs = {
            "idx": tokens,  # [B, L]
            "target": target_token,  # [B, L]
            "text_loss_mask": text_loss_mask,  # [B, L]
            "image_loss_mask": image_loss_mask,  # [B, L]
            "imgs_input": batch["und_images_tensor"],  # [B, 3, C, H, W]
            "image_token_id": img_token_id,
            "eff_images_num": batch["eff_images_num"],  # List of effective image counts, len = B
        }

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_input_kwargs, batch_size, n_tokens


    def build_dataloader(self):
        args = self.args
        self.dataset = JanusTextImageArrowStream(
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
