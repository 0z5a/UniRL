from typing import Union, Dict

import torch
from einops import rearrange
from index_kits.sampler import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer
from ..data_kits.text_image_loader import TextImageArrowStream
from ..models import load_vae

from .helpers import dynamic_values_wrapper
from ..samplers.text2image_ar_sampler import Text2ImageARSampler
from ..samplers.logits_processor import get_logits_processors
from ..utils.helpers import as_tuple
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE


class Text2ImageARTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def prepare_model_inputs(self, batch: Dict, device: Union[int, str]):
        args = self.args
        _ = batch["kwargs"]
        image = batch["image"]
        tokens = batch["tokens"]

        # Tokenization
        image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image_tokens = self.vae.vq_encode(image).long()
        image_tokens = image_tokens + self.image_token_offset
        if len(image_tokens.shape) > 2:
            image_tokens = rearrange(image_tokens, "b h w -> b (h w)")

        # text: 256 + 1, image: 1024, total: 1281, 1281 - 1 = 1280
        tokens = batch["tokens"].contiguous().to(device)
        for i in range(tokens.shape[0]):
            tokens[i, tokens[i] == self.dataset.tokenizer.special_token_map["<img>"]] = image_tokens[i]

        target_token = tokens.clone()
        target_token[target_token == self.dataset.tokenizer.special_token_map["<pad>"]] = -100
        target_token[target_token == self.dataset.tokenizer.special_token_map["<cfg>"]] = -100

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
        if args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": batch["iw_ih_scatter_index"].to(device),  # [b, 2]
                "iw_ih_scatter_src": batch["iw_ih_scatter_src"].to(device),  # [b, 2]
            })

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        return model_intput_kwargs, batch_size, n_tokens

    def build_extra_model(self):
        """leave tokenizer to be implemented by subclass,
        since we may neeed to do experiments on different modalities with different tokenizers
        """
        args = self.args
        self.vae = load_vae(
            args.vae_type,
            args.vae_precision,
            device=self.device,
            logger=self.logger,
        )
        self.image_token_offset = args.image_token_offset
        self.logger.info(f"Image token offset: {self.image_token_offset}")

    def get_sampler(self):
        return Text2ImageARSampler(
            self.args,
            model_dict=dict(vae=self.vae,
                            model=self.model_engine.module,
                            model_settings=self.model_settings,
                            tokenizer=self.dataset.tokenizer,
                            logits_processor=get_logits_processors(self.args.logits_processors_cfg),
                            ),
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            logger=self.val_logger,
        )

    def build_dataloader(self):
        args = self.args
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
