import os
from functools import partial
from typing import Dict, Union

import torch
import torch.nn.functional as F
import einops
from torch.utils.data import DataLoader

from index_kits.sampler import DistributedSampler

from ..models import load_vae
from ..diffusion import load_denoiser
from .helpers import MultiModalScalarStates, MultiModalCycleStates
from ..constants import C_SCALE
from ..utils.torch_utils import set_worker_seed_builder, PRECISION_TO_TYPE
from ..utils.file_utils import safe_dir
from .base_trainer import BaseTrainer
from ..models.tokenizers.tokenizer_wrapper import TokenizerWrapper
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.text_loader import TextArrowStream
from ..data_kits.t2i_loader import TextImageArrowStream, TextImageToImageArrowStream, FaceIDArrowStream
from ..data_kits.instruction_template import (
    text2image_instructions,
    inpainting_instructions,
    editing_instructions,
    subject_driven_instructions,
    face_id_instructions,
)

class InstructionTuningGeminiAlphaTrainer(BaseTrainer):
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

        self.tokenizer_wrapper = TokenizerWrapper(args.tokenizer_name, self.logger)

    def build_dataloader(self):
        args = self.args

        dataloader_kwargs = dict(
            worker_init_fn=set_worker_seed_builder(self.rank),
            **args.dataloader_params,  # num_workers, prefetch_factor
        )

        self.dataset_dict = {}
        self.sampler_dict = {}
        self.dataloader_dict = {}

        self.all_dataset_keys = ['t2i', 'inpainting_randomly_sampled_from_1024_high_quality_v2', 'editing_OmniEdit', 'editing_video2frame_in_house', 'subject_driven', 'face_id_preserve', 'lm']
        self.sampling_probs_dict = {key: args.sampling_probs[i] for i, key in enumerate(self.all_dataset_keys)}

        # vision
        task_info_list = [
            dict(dataset_tag="t2i", cls=TextImageArrowStream, instruction_candidates=text2image_instructions),
            dict(dataset_tag="inpainting_randomly_sampled_from_1024_high_quality_v2", cls=TextImageToImageArrowStream, instruction_candidates=inpainting_instructions),
            dict(dataset_tag="editing_OmniEdit", cls=TextImageToImageArrowStream, instruction_candidates=editing_instructions),
            dict(dataset_tag="editing_video2frame_in_house", cls=TextImageToImageArrowStream, instruction_candidates=editing_instructions),
            dict(dataset_tag="subject_driven", cls=TextImageToImageArrowStream, instruction_candidates=subject_driven_instructions),
            dict(dataset_tag="face_id_preserve", cls=FaceIDArrowStream, instruction_candidates=face_id_instructions),
        ]

        for item in task_info_list:
            dataset_tag = item['dataset_tag']
            self.dataset_dict[dataset_tag] = item['cls'](
                args=args,
                tokenizer_name=args.tokenizer_name,
                task_kwargs=args.get(f'{dataset_tag}_task_kwargs'),
                index_kwargs=dict(
                    batch_size=self.micro_batch_size,  # Provide bsz to use multireso.
                    world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                    **args.get(f'{dataset_tag}_index_kwargs'),
                ),
                dummy_number=0,  # we process dummy in prepare_model_inputs
                template="instruct",
                instruction_candidates=item['instruction_candidates'],
                logger=self.logger,
                dataset_tag=dataset_tag,
            )
            # Build sampler and data loader
            self.sampler_dict[dataset_tag] = DistributedSampler(
                self.dataset_dict[dataset_tag],
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=args.global_seed,
                drop_last=True,
                batch_size=self.micro_batch_size,   # Provide bsz to use multireso.
            )
            self.dataloader_dict[dataset_tag] = DataLoader(
                self.dataset_dict[dataset_tag],
                batch_size=self.micro_batch_size,
                sampler=self.sampler_dict[dataset_tag],
                shuffle=False,
                drop_last=True,
                collate_fn=(self.dataset_dict[dataset_tag].collate_fn
                            if hasattr(self.dataset_dict[dataset_tag], "collate_fn")
                            else None),
                **dataloader_kwargs,
            )

        # lm
        self.dataset_dict["lm"] = TextArrowStream(
            args=args,
            index_file=args.lm_index_file,
            t2t_text_token_length=args.lm_token_length,
            tokenizer_name=args.tokenizer_name,
            index_kwargs=dict(
                batch_size=self.micro_batch_size,  # Provide bsz to use multireso.
                world_size=1,  # Dataset don't need to align with world_size. It will be handled by the sampler.
                index_strategy=args.lm_index_strategy,
                index_probability=args.lm_index_probability,
                **args.lm_index_kwargs,
            ),
            template="instruct",
            logger=self.logger,
            dataset_type="lm",
        )
        self.sampler_dict["lm"] = DistributedSampler(
            dataset=self.dataset_dict["lm"],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            seed=args.global_seed,
            drop_last=True,
            batch_size=self.micro_batch_size,
        )
        self.dataloader_dict["lm"] = DataLoader(
            dataset=self.dataset_dict["lm"],
            batch_size=self.micro_batch_size,
            sampler=self.sampler_dict["lm"],
            shuffle=False,
            drop_last=True,
            **dataloader_kwargs,
        )


    def build_data_iterator(self):

        self.dataloader = CombinedBatchIterator(
            ss=self.ss,
            fast_shuffle=self.args.fast_shuffle,
            rank=self.rank,
            world_size=self.world_size,
            datasets=self.dataset_dict,
            samplers=self.sampler_dict,
            dataloaders=self.dataloader_dict,
            sampling_probs=self.sampling_probs_dict,
            initial_seed=self.args.global_seed,
            sampling_mode=self.args.get('combined_iterator_sampling_mode', 'random'),
            cache_shuffle=self.args.cache_shuffle,
        )

    def add_dummy_tokens_for_gen_image(self, tokens, target_tokens, extra, dummy_number):
        """ Add dummy tokens to avoid hanging when deepspeed all-reduce gradients.

        Different modalities connect with different model parameters in the computation graph.
        If different batches correspond to different modalities, deepspeed cannot correctly perform
        all-reduce gradients. Therefore, we need to pad some dummy tokens to maintain consistent
        activated model parameters.
        """
        batch_size, n_tokens = tokens.shape
        dummy_tokens = torch.full((batch_size, dummy_number), self.tokenizer_wrapper.pad_token, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, dummy_tokens], dim=1)
        dummy_target_tokens = (-100) * torch.ones((batch_size, dummy_number), dtype=tokens.dtype, device=target_tokens.device)
        target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

        # add_iw_ih_token
        scatter_index = torch.tensor([[n_tokens, n_tokens + 1]] * batch_size, dtype=torch.long, device=tokens.device)
        scatter_src = torch.tensor([[2, 2]] * batch_size, dtype=torch.long, device=tokens.device)
        extra['iw_ih_scatter_index'] = torch.cat([extra['iw_ih_scatter_index'], scatter_index], dim=1) \
            if 'iw_ih_scatter_index' in extra else scatter_index
        extra['iw_ih_scatter_src'] = torch.cat([extra['iw_ih_scatter_src'], scatter_src], dim=1) \
            if 'iw_ih_scatter_src' in extra else scatter_src

        # add_timestep_token
        scatter_index = torch.tensor([[n_tokens + 2]] * batch_size, dtype=torch.long, device=tokens.device)
        if 'timestep_scatter_index' in extra:
            extra['timestep_scatter_index'] = torch.cat([extra['timestep_scatter_index'], scatter_index], dim=1)
        else:
            extra['timestep_scatter_index'] = scatter_index

        patch_size = self.args.patch_size
        latents = torch.randn((batch_size, self.args.vae_latent_dim, patch_size, patch_size), device=tokens.device)
        t, x_0, x_1 = self.denoiser.sample(latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        model_t = self.denoiser.get_model_t(t)  # t*1000

        extra.update(dict(
            x_t=x_t,
            t=model_t,
            diffusion_loss_fn=partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        ))

        self.extend_mask(extra, batch_size, dummy_number)

        # add image_mask after extend_mask to avoid extra extend
        gen_image_mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        gen_image_mask[:, -1] = True
        extra.update(dict(
            image_mask=gen_image_mask,
        ))

        n_tokens += dummy_number

        return tokens, target_tokens, extra, n_tokens

    def add_dummy_tokens_for_src_face_embedding(self, tokens, target_tokens, extra, dummy_number):
        """ Add dummy tokens to avoid hanging when deepspeed all-reduce gradients.

        Different modalities connect with different model parameters in the computation graph.
        If different batches correspond to different modalities, deepspeed cannot correctly perform
        all-reduce gradients. Therefore, we need to pad some dummy tokens to maintain consistent
        activated model parameters.
        """
        batch_size, n_tokens = tokens.shape
        dummy_tokens = torch.full((batch_size, dummy_number), self.tokenizer_wrapper.pad_token, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, dummy_tokens], dim=1)
        dummy_target_tokens = (-100) * torch.ones((batch_size, dummy_number), dtype=tokens.dtype, device=target_tokens.device)
        target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

        extra = self.extend_mask(extra, batch_size, dummy_number)

        # add image_mask after extend_mask to avoid extra extend
        und_image_mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        und_image_mask[:, -dummy_number:] = True
        extra.update(dict(
            src_face_embedding=torch.rand((batch_size, 1, 512), device=tokens.device),
            und_image_masks=und_image_mask,
        ))

        n_tokens += dummy_number

        return tokens, target_tokens, extra, n_tokens

    def extend_mask(self, extra, batch_size, dummy_number):
        # image_mask
        if 'image_mask' in extra:
            extra['image_mask'] = torch.cat([
                extra['image_mask'],
                torch.zeros((batch_size, dummy_number), dtype=extra['image_mask'].dtype, device=extra['image_mask'].device)
            ], dim=1)

        # src_image_mask
        if 'src_image_mask' in extra:
            extra['src_image_mask'] = torch.cat([
                extra['src_image_mask'],
                torch.zeros((batch_size, dummy_number), dtype=extra['src_image_mask'].dtype, device=extra['src_image_mask'].device)
            ], dim=1)

        # text_mask
        if 'text_mask' in extra:
            extra['text_mask'] = torch.cat([
                extra['text_mask'],
                torch.zeros((batch_size, dummy_number), dtype=extra['text_mask'].dtype, device=extra['text_mask'].device)
            ], dim=1)

        # und_image_masks
        if 'und_image_masks' in extra:
            extra['und_image_masks'] = torch.cat([
                extra['und_image_masks'],
                torch.zeros((batch_size, dummy_number), dtype=extra['und_image_masks'].dtype, device=extra['und_image_masks'].device)
            ], dim=1)

        # attention_mask
        if "attention_mask" in extra:
            L = extra["attention_mask"].shape[2]
            # B x 1 x L x L -> B x 1 x (L + dummy_number) x (L + dummy_number)
            extra["attention_mask"] = F.pad(extra["attention_mask"], (0, dummy_number, 0, dummy_number), value=False)
            for i in range(L, L + dummy_number):
                extra["attention_mask"][:, :, i, i] = True

        return extra

    def prepare_t2i_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # text: 256 + 1, image: 256, total: 513, 513 - 1 = 512
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_tokens is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_tokens
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # build attention mask
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        attention_mask = batch["attention_mask"].to(device)

        faceid_dummy_token_number = self.args.get("faceid_dummy_token_number", None)
        extra = {}
        if faceid_dummy_token_number is not None:
            # the key must correspond to that in model_intput_kwargs
            extra = {
                "text_mask": text_mask,
                "image_mask": image_mask,
                "attention_mask": attention_mask,
            }
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens_for_src_face_embedding(tokens, target_tokens, extra, faceid_dummy_token_number)

        # ===================================== prepare diffusion =====================================
        image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            latents = self.vae.encode(image).latent_dist.sample()
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
            data_type="t2i",
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
        
        model_intput_kwargs.update(extra)
        
        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            self.save_model_inputs(model_intput_kwargs, cur_step)

        return model_intput_kwargs, batch_size, n_tokens

    def prepare_lm_model_inputs(self, batch: Dict, device: Union[int, str], **kwargs):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]

        # Attention mask
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)

        # Add dummy tokens
        extra = dict(
            text_mask=text_mask,
            attention_mask=attention_mask,
        )
        
        target_image_dummy_token_number = self.args.get("target_image_dummy_token_number", None)
        if target_image_dummy_token_number is not None:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens_for_gen_image(
                    tokens, target_tokens, extra, dummy_number=target_image_dummy_token_number)

        faceid_dummy_token_number = self.args.get("faceid_dummy_token_number", None)
        if faceid_dummy_token_number is not None:
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens_for_src_face_embedding(
                    tokens, target_tokens, extra, dummy_number=faceid_dummy_token_number)

        model_intput_kwargs = dict(
            idx=tokens,  # [b, 512]
            target=target_tokens,  # [b, 512]
            image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
            data_type="lm",         # For loss
        )

        # x_t, t, image_mask, und_images, und_image_masks, diffusion_loss_fn, iw, ih, timestep, attention_mask
        model_intput_kwargs.update(extra)

        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            self.save_model_inputs(model_intput_kwargs, cur_step)

        return model_intput_kwargs, batch_size, n_tokens

    def prepare_ti2i_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        # text: 256 + 1, image: 1024 * 2, total: 2305, 2305 - 1 = 2304
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_token is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_token
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        tgt_image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        attention_mask = batch["attention_mask"].to(device)


        faceid_dummy_token_number = self.args.get("faceid_dummy_token_number", None)
        extra = {}
        if faceid_dummy_token_number is not None:
            # the key must correspond to that in model_intput_kwargs
            extra = {
                "text_mask": text_mask,
                "src_image_mask": src_image_mask,
                "image_mask": tgt_image_mask,
                "attention_mask": attention_mask,
            }
            tokens, target_tokens, extra, n_tokens = self.add_dummy_tokens_for_src_face_embedding(tokens, target_tokens, extra, faceid_dummy_token_number)

        # ===================================== prepare diffusion =====================================
        tgt_image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            tgt_latents = self.vae.encode(tgt_image).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                tgt_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
            else:
                tgt_latents.mul_(self.vae.config.scaling_factor)

        if hasattr(self.vae, "ffactor_temporal"):
            assert tgt_latents.shape[2] == 1, "tgt_latents should have shape [B, C, T, H, W] and T should be 1"
            tgt_latents = tgt_latents.squeeze(2)

        t, x_0, x_1 = self.denoiser.sample(tgt_latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        model_t = self.denoiser.get_model_t(t) # t*1000


        src_is_list_of_tensor = isinstance(batch["src_images"], list)
        src_is_stacked_tensor = isinstance(batch["src_images"], torch.Tensor)

        # batch["src_images"] is a list of a list of tensors, the length of the first list is batch_size, the length of the second list is the number of source images of the i-th sample, and each tensor is c x h x w
        # some data_type does not have src_images, such as faceid, it would be list of None
        if src_is_list_of_tensor:
            batch_src_xs = []
            batch_src_model_ts = []
            for src_images in batch["src_images"]:
                # in t2i_loader, src_image returns src_images[0] if len(src_images) == 1 else src_images
                # therefore, for subject driven, if one sample only contains one src_image, batch["src_images"] can be
                # [ c x h x w, n_srcs_{2} * [ c x h x w] ]
                # we convert one-src-image to list to ensure batch["src_images"] is a list of a list of tensors
                if isinstance(src_images, torch.Tensor) and src_images.dim() == 3:
                    src_images = [src_images]
                src_xs = []
                src_model_ts = []
                for src_image in src_images:
                    # c x h x w
                    src_image = src_image.to(device)
                    with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                        # c x h x w -> 1 x c x h x w -> 1 x c x 1 x h x w
                        src_latents = self.vae.encode(src_image[None]).latent_dist.sample()
                        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                            src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                        else:
                            src_latents.mul_(self.vae.config.scaling_factor)

                    if hasattr(self.vae, "ffactor_temporal"):
                        assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                        # 1 x c x h x w
                        src_latents = src_latents.squeeze(2)

                    src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
                    src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
                    src_model_t = self.denoiser.get_model_t(src_t)  # t*1000
                    src_xs.append(src_x)
                    src_model_ts.append(src_model_t)
                
                try:
                    # batch_src_xs is a list of tensors, the length of the list is batch_size.
                    # each tensor is n_src_{i} x c x h x w, n_src_{i} is the number of source images of the i-th sample.
                    batch_src_xs.append(torch.cat(src_xs))
                except Exception as e:
                    # batch_src_xs is a list of list of tensors, the length of the first list is batch_size, the length of the second list is the number of source images of the i-th sample, and each tensor is c x h x w
                    batch_src_xs.append(src_xs)
                # batch_src_model_ts is a list of tensors, the length of the list is batch_size.
                # each tensor is n_src_{i}, n_src_{i} is the number of source images of the i-th sample.
                batch_src_model_ts.append(torch.cat(src_model_ts))

            input_src_x = batch_src_xs
            input_src_t = batch_src_model_ts

        elif src_is_stacked_tensor:
            src_images = batch["src_images"].to(device)
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                src_latents = self.vae.encode(src_images).latent_dist.sample()
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                else:
                    src_latents.mul_(self.vae.config.scaling_factor)

            if hasattr(self.vae, "ffactor_temporal"):
                assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                src_latents = src_latents.squeeze(2)

            src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
            src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
            src_model_t = self.denoiser.get_model_t(src_t) # t*1000

            input_src_x = src_x
            input_src_t = src_model_t

        else:
            raise NotImplementedError(f"src_images is not a list of tensors or a stacked tensor")

        # ===================================== Pack model kwargs ==================================
        model_intput_kwargs = dict(
            idx=tokens,  # [b, 768]
            x_t=x_t,  # [b, c, h, w]
            t=model_t, # [b]
            diffusion_loss_fn=diffusion_loss_fn,
            src_x=input_src_x,  # [b, c, h, w]
            src_t=input_src_t,  # [b]
            src_image_mask=src_image_mask,  # [b, 768]
            target=target_tokens,  # [b, 768]
            text_mask=text_mask,  # [b, 768]
            image_mask=tgt_image_mask,  # [b, 768]
            attention_mask=attention_mask, # [b, 768, 768]
            image_loss_weight=self.args.image_loss_weight,
            data_type="ti2i",
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": [t.cuda() for t in batch["iw_ih_scatter_index"]] if isinstance(batch["iw_ih_scatter_index"], list) else batch["iw_ih_scatter_index"].to(device),  # tensor: [b, 2(n_src_{i} + 1)]; list: each tensor is 2(n_src_{i} + 1)
                "iw_ih_scatter_src": [t.cuda() for t in batch["iw_ih_scatter_src"]] if isinstance(batch["iw_ih_scatter_src"], list) else batch["iw_ih_scatter_src"].to(device),  # tensor: [b, 2(n_src_{i} + 1)]; list: each tensor is 2(n_src_{i} + 1)
            })
        if self.args.add_timestep_token:
            assert "timestep_scatter_index" in batch, "timestep_scatter_index is required for adding timestep token"
            model_intput_kwargs.update({
                "timestep_scatter_index": [t.cuda() for t in batch["timestep_scatter_index"]] if isinstance(batch["timestep_scatter_index"], list) else batch["timestep_scatter_index"].to(device),  # tensor: [b, k]; list: each tensor is (n_src_{i} + 1)
            })

        model_intput_kwargs.update(extra)

        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            self.save_model_inputs(model_intput_kwargs, cur_step)

        return model_intput_kwargs, batch_size, n_tokens

    def prepare_faceid_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        tokens = batch["tokens"][:, :-1].contiguous().to(device)

        # ===================================== IMPORTANT =====================================
        # target_tokens is only used to calculate losses on text tokens and some special tokens
        # <img> is set to -100 in target_tokens
        target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
        text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
        # image_loss is computed inplace, therefore image_mask is shifted same as tokens
        src_image_mask = batch["src_image_mask"][:, :-1].contiguous().to(device)
        tgt_image_mask = batch["image_mask"][:, :-1].contiguous().to(device)

        # build attention mask
        batch_size = tokens.shape[0]
        n_tokens = tokens.shape[1]
        attention_mask = batch["attention_mask"].to(device)

        # ===================================== prepare diffusion =====================================
        tgt_image = batch["image"].to(device)
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            tgt_latents = self.vae.encode(tgt_image).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                tgt_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
            else:
                tgt_latents.mul_(self.vae.config.scaling_factor)

        if hasattr(self.vae, "ffactor_temporal"):
            assert tgt_latents.shape[2] == 1, "tgt_latents should have shape [B, C, T, H, W] and T should be 1"
            tgt_latents = tgt_latents.squeeze(2)

        t, x_0, x_1 = self.denoiser.sample(tgt_latents, n_tokens)
        t, x_t, u_t = self.denoiser.path_sampler.plan(t, x_0, x_1)
        diffusion_loss_fn = partial(self.denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t)
        model_t = self.denoiser.get_model_t(t) # t*1000

        # If "vae" in src_condition_type, src_images is not None
        src_is_list_of_tensor = isinstance(batch["src_images"], list)
        src_is_stacked_tensor = isinstance(batch["src_images"], torch.Tensor)

        # batch["src_images"] is a list of a list of tensors, the length of the first list is batch_size, the length of the second list is the number of source images of the i-th sample, and each tensor is c x h x w
        # some data_type does not have src_images, such as faceid, it would be list of None
        if src_is_list_of_tensor:
            batch_src_xs = []
            batch_src_model_ts = []
            for src_images in batch["src_images"]:
                src_xs = []
                src_model_ts = []
                for src_image in src_images:
                    # c x h x w
                    src_image = src_image.to(device)
                    with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                        # c x h x w -> 1 x c x h x w -> 1 x c x 1 x h x w
                        src_latents = self.vae.encode(src_image[None]).latent_dist.sample()
                        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                            src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                        else:
                            src_latents.mul_(self.vae.config.scaling_factor)

                    if hasattr(self.vae, "ffactor_temporal"):
                        assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                        # 1 x c x h x w
                        src_latents = src_latents.squeeze(2)

                    src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
                    src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
                    src_model_t = self.denoiser.get_model_t(src_t)  # t*1000
                    src_xs.append(src_x)
                    src_model_ts.append(src_model_t)
                
                # batch_src_xs is a list of tensors, the length of the list is batch_size.
                # each tensor is n_src_{i} x c x h x w, n_src_{i} is the number of source images of the i-th sample.
                batch_src_xs.append(torch.cat(src_xs))
                # batch_src_model_ts is a list of tensors, the length of the list is batch_size.
                # each tensor is n_src_{i}, n_src_{i} is the number of source images of the i-th sample.
                batch_src_model_ts.append(torch.cat(src_model_ts))

                input_src_x = batch_src_xs
                input_src_t = batch_src_model_ts

        elif src_is_stacked_tensor:
            src_images = batch["src_images"].to(device)
            with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
                src_latents = self.vae.encode(src_images).latent_dist.sample()
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    src_latents.sub_(self.vae.config.shift_factor).mul_(self.vae.config.scaling_factor)
                else:
                    src_latents.mul_(self.vae.config.scaling_factor)

            if hasattr(self.vae, "ffactor_temporal"):
                assert src_latents.shape[2] == 1, "src_latents should have shape [B, C, T, H, W] and T should be 1"
                src_latents = src_latents.squeeze(2)

            src_t, src_x_0, src_x_1 = self.denoiser.sample_start(src_latents)
            src_t, src_x, _ = self.denoiser.path_sampler.plan(src_t, src_x_0, src_x_1)
            src_model_t = self.denoiser.get_model_t(src_t) # t*1000

            input_src_x = src_x
            input_src_t = src_model_t

        else:
            # some data_type does not have src_images, such as faceid
            input_src_x = None
            input_src_t = None

        # If "face_embedding" in src_condition_type, src_face_embedding is not None
        if "src_face_embedding" in batch.keys():
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
            src_x=input_src_x,  # [b, c, h, w]
            src_t=input_src_t,  # [b]
            src_image_mask=src_image_mask,  # [b, 768]
            target=target_tokens,  # [b, 768]
            text_mask=text_mask,  # [b, 768]
            image_mask=tgt_image_mask,  # [b, 768]
            attention_mask=attention_mask, # [b, 768, 768]
            image_loss_weight=self.args.image_loss_weight,
            data_type="faceid",
            src_face_embedding=src_face_embedding,
        )
        if self.args.add_iw_ih_token:
            assert "iw_ih_scatter_index" in batch and "iw_ih_scatter_src" in batch, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"
            model_intput_kwargs.update({
                "iw_ih_scatter_index": [t.cuda() for t in batch["iw_ih_scatter_index"]] if isinstance(batch["iw_ih_scatter_index"], list) else batch["iw_ih_scatter_index"].to(device),  # tensor: [b, 2(n_src_{i} + 1)]; list: each tensor is 2(n_src_{i} + 1)
                "iw_ih_scatter_src": [t.cuda() for t in batch["iw_ih_scatter_src"]] if isinstance(batch["iw_ih_scatter_src"], list) else batch["iw_ih_scatter_src"].to(device),  # tensor: [b, 2(n_src_{i} + 1)]; list: each tensor is 2(n_src_{i} + 1)
            })
        if self.args.add_timestep_token:
            assert "timestep_scatter_index" in batch, "timestep_scatter_index is required for adding timestep token"
            model_intput_kwargs.update({
                "timestep_scatter_index": [t.cuda() for t in batch["timestep_scatter_index"]] if isinstance(batch["timestep_scatter_index"], list) else batch["timestep_scatter_index"].to(device),  # tensor: [b, k]; list: each tensor is (n_src_{i} + 1)
            })

        # Save training data for debugging
        if (cur_step := self.ss.current_run_update_steps) < self.args.save_n_training_data and self.rank < 8:
            self.save_model_inputs(model_intput_kwargs, cur_step)

        return model_intput_kwargs, batch_size, n_tokens


    def prepare_model_inputs(
        self,
        batch: Dict,
        device: Union[int, str],
    ):
        data_type = batch["data_type"]
        if isinstance(data_type, list):
            data_type = data_type[0]
        if data_type == "t2i":
            inputs = self.prepare_t2i_model_inputs(batch, device)
        elif data_type == "lm":
            inputs = self.prepare_lm_model_inputs(batch, device)
        elif data_type == "ti2i":
            inputs = self.prepare_ti2i_model_inputs(batch, device)
        elif data_type == "faceid":
            inputs = self.prepare_faceid_model_inputs(batch, device)
        else:
            raise ValueError("Unknown batch dtype, expected 'ti2i' or 'faceid'.")
        return inputs

    def save_model_inputs(self, model_intput_kwargs, cur_step):
        # If gradient_accumulation_steps > 1, data of the boundary step will be finally saved.
        check_data_path = safe_dir(os.path.join(self.exp_dir, "saved_training_data"))
        torch.save(model_intput_kwargs, os.path.join(check_data_path, f"data_batch{cur_step}_rank{self.rank}.pt"))

    def resume_dataloader(self, ss):
        # The sampler states will be restored in self.shuffle_dataset()
        for sampler in self.sampler_dict.values():
            assert isinstance(sampler, DistributedSampler), (
                f"In {self.__class__.__name__}, only index_kits.samplers.DistributedSampler supports --resume-dataloader."
            )

    def shuffle_dataset_and_set_start_index(self, ss):
        # Shuffle and reset index are handled by the mix-data iterator, i.e., self.dataloader
        pass

    def get_states_cls(self, state_type):
        if state_type == "scalar":
            return MultiModalScalarStates
        elif state_type == "cycle":
            return MultiModalCycleStates
        else:
            raise ValueError(f"Unknown state type: {state_type}")

    def update_train_states(self, ss, cs, batch, batch_size, n_tokens, loss):
        # A forward-backward step is counted as one train step.
        ss.add(train_steps=1, epoch_train_steps=1)
        cs.add(log_steps=1, running_loss=loss)
        # If training long sequence, each sequence may contain multiple samples.
        # Therefore, we sum `n_samples` to get the real number of samples.
        samples = batch["n_samples"].sum().item()
        key = batch["dtype"][0]
        cs.running_samples[key] += samples
        cs.running_tokens[key] += batch_size * n_tokens

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
        consumed_samples = 0
        consumed_tokens = 0
        for key in self.all_dataset_keys:
            part_samples = sum([cs_i.running_samples[key] for cs_i in all_cs])
            consumed_samples += part_samples
            self.ss.consumed_samples_total[key] += part_samples
            self.ss.epoch_consumed_samples[key] += part_samples
            part_tokens = sum([cs_i.running_tokens[key] for cs_i in all_cs])
            consumed_tokens += part_tokens
            self.ss.consumed_tokens_total[key] += part_tokens

        self.ss.add(
            consumed_computations_attn=6 * self.params_count["attn+mlp"] * consumed_tokens / C_SCALE,
            consumed_computations_total=6 * self.params_count["total"] * consumed_tokens / C_SCALE,
        )

        return consumed_samples

    def get_events(self, ss, loss):
        log_events = []
        for key in self.all_dataset_keys:
            log_events.extend([
                f"Consumed {key} Samples: {ss.consumed_samples_total[key]:,}",
                f"Consumed {key} Tokens: {ss.consumed_tokens_total[key]:,}",
            ])

        consumed_samples_total = sum([ss.consumed_samples_total[key] for key in self.all_dataset_keys])
        consumed_tokens_total = sum([ss.consumed_tokens_total[key] for key in self.all_dataset_keys])
        summary_events = [
            ("Train/TotalSamples/train_loss", loss, consumed_samples_total),
            ("Train/TotalTokens/train_loss", loss, consumed_tokens_total),
        ]
        return log_events, summary_events