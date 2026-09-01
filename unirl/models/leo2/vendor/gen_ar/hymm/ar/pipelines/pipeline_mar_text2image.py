# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by @jarvizhang
# Modified from diffusers==0.29.2
#
# ==============================================================================
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import logging

from ..mask_schedulers import adap_sche
from ...ar.mask_schedulers import create_attention_mask_t2i
from ...models.basic.rope import get_mlm_rope
from ...models.tokenizers import TokenizerWrapper
from ...models.arrangement import Arrangement
from ...utils.helpers import repeat_interleave, default
from ...utils.torch_utils import PRECISION_TO_TYPE

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Text2ImageMARPipeline(DiffusionPipeline):
    def __init__(self,
                 vae,
                 model,
                 tokenizer: TokenizerWrapper,
                 arrange: Arrangement,
                 pipeline: DiffusionPipeline,
                 progress_bar_config: Dict[str, Any] = None,
                 args=None,
                 model_settings=None,
                 ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        self.set_progress_bar_config(**progress_bar_config)

        self.args = args
        self.model_settings = model_settings
        # ==========================================================================================

        self.register_modules(
            vae=vae,
            model=model,
            tokenizer=tokenizer,
            arrange=arrange,
            pipeline=pipeline,
        )
        self.tokenizer = tokenizer
        self.arrange = arrange
        self.vae_scale_factor = self.vae.downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_whole_tokens(
            self,
            prompts,
            image_token_len,
            negative_prompts: Optional[List[str]] = None,
            do_classifier_free_guidance: bool = False,
            device: Optional[torch.device] = None,
    ):
        """
        Returns
        -------
        batch_whole_tokens: torch.LongTensor
            [batch_size, text_token_len + image_token_len], the whole tokens to predict
        batch_uncond_whole_tokens: torch.LongTensor
            [batch_size, text_token_len + image_token_len], the whole tokens to predict unconditionally
        """
        kwargs = dict(
            image_token_len=image_token_len,
            max_text_token_length=self.arrange.s_text_maxlen,
            dtype=torch.int,
        )
        batch_whole_tokens = []
        for prompt in prompts:
            whole_tokens = self.tokenizer.encode_mlm_t2i(
                text=prompt, text_uncond_p=0, **kwargs)[:-1]  # :-1 remove the eos token
            batch_whole_tokens.append(whole_tokens)
        batch_whole_tokens = torch.stack(batch_whole_tokens, dim=0)
        batch_whole_tokens = batch_whole_tokens.to(device)

        if do_classifier_free_guidance:
            batch_uncond_whole_tokens = []
            for prompt, neg_prompt in zip(prompts, negative_prompts):
                if neg_prompt == '':
                    text = prompt
                    uncond_p = 1
                else:
                    text = neg_prompt
                    uncond_p = 0
                neg_whole_tokens = self.tokenizer.encode_mlm_t2i(
                    text=text, text_uncond_p=uncond_p, **kwargs)[:-1]  # :-1 remove the eos token
                batch_uncond_whole_tokens.append(neg_whole_tokens)
            batch_uncond_whole_tokens = torch.stack(batch_uncond_whole_tokens, dim=0)
            batch_uncond_whole_tokens = batch_uncond_whole_tokens.to(device)
        else:
            batch_uncond_whole_tokens = None

        return batch_whole_tokens, batch_uncond_whole_tokens

    @staticmethod
    def adap_sche(step, image_seq_len, mode="arccos"):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":  # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":  # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":  # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":  # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":  # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * image_seq_len
        sche = sche.round()
        sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
        sche[-1] += image_seq_len - sche.sum()  # need to sum up nb of code
        return sche.int()

    def prepare_latents_and_mask(self, batch_size, height, width, device):
        shape = (
            batch_size,
            int(height) // self.vae_scale_factor // self.args.patch_size,
            int(width) // self.vae_scale_factor // self.args.patch_size,
            self.model.token_embed_dim,
        )
        init_tokens = torch.zeros(shape).to(device)
        init_mask = torch.ones(shape[:-1], dtype=torch.bool).to(device)

        return init_tokens, init_mask

    def guidance_scale(self, step):
        if self._guidance_dynamic == 'const':
            guidance = self._guidance_scale
        elif self._guidance_dynamic == 'linear_inc':
            guidance = (self._guidance_scale - 1) * (step / (self.num_timesteps - 1)) + 1
        elif self._guidance_dynamic == 'linear_dec':
            guidance = (self._guidance_scale - 1) * ((self.num_timesteps - 1 - step) / (self.num_timesteps - 1)) + 1
        else:
            raise ValueError(f"Unknown guidance dynamic: {self._guidance_dynamic}")
        return guidance

    @staticmethod
    def mask_by_order(mask_len, order, bsz, seq_len, device):
        masking = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        masking = torch.scatter(masking,
                                dim=-1,
                                index=order[:, :mask_len],
                                src=torch.ones(bsz, seq_len, dtype=torch.bool, device=device))
        return masking

    @staticmethod
    def sample_orders(bsz, seq_len, device):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(device).long()
        return orders

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        height: int,
        width: int,
        num_inference_steps: int = 12,
        guidance_scale: float = 6.0,
        guidance_dynamic: str = 'const',
        temperature: float = 1.0,
        schedule_type: Optional[str] = None,
        shift: float = None,
        negative_prompt: Optional[List[str]] = None,
        num_sample_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        self._guidance_scale = guidance_scale
        self._guidance_dynamic = guidance_dynamic

        # 2. Define call parameters
        batch_size = len(prompt)

        device = self._execution_device

        target_dtype = PRECISION_TO_TYPE[self.args.autocast_dtype]
        autocast_enabled = (target_dtype != torch.float32)

        # 3. Prepare initial masked tokens
        latents, mask = self.prepare_latents_and_mask(
            batch_size * num_sample_per_prompt,
            height,
            width,
            device,
        )
        bsz, tk_height, tk_width, token_embed_dim = latents.shape
        image_token_len = tk_height * tk_width
        # Flatten the spatial dimensions
        latents = latents.reshape(bsz, image_token_len, token_embed_dim)
        mask = mask.reshape(bsz, image_token_len)

        # 4. Prepare initial tokens
        # 4.1 MLM use <cfg> token when negative prompt is empty string.
        # Remap the negative prompt to the prompt for providing the same length.
        assert all([x == '' for x in negative_prompt]) or all([x != '' for x in negative_prompt]), (
            f"Negative prompt should be all empty or all non-empty, got {negative_prompt}"
        )
        # 4.2 prepare the whole tokens
        whole_tokens, uncond_whole_tokens = self.prepare_whole_tokens(
            repeat_interleave(prompt, num_sample_per_prompt),
            image_token_len,
            repeat_interleave(negative_prompt, num_sample_per_prompt),
            self.do_classifier_free_guidance,
            device,
        )
        if self.do_classifier_free_guidance:
            whole_tokens = torch.cat([uncond_whole_tokens, whole_tokens], dim=0)
            bsz_factor = 2
        else:
            bsz_factor = 1
        imgs_start_pos = torch.tensor([self.arrange.s_image_range[0]] * bsz * bsz_factor, device=device)

        #   Create attention mask: [bs, 1, seqlen, seqlen]
        attn_mask_kwargs = dict(
            pad_id=self.arrange.pad_id, boi_id=self.arrange.boi_id, eoi_id=self.arrange.eoi_id,
            mask_pad=True, return_inverse_mask=True, dtype=target_dtype,
        )
        attention_mask = create_attention_mask_t2i(whole_tokens, **attn_mask_kwargs)
        attention_mask = attention_mask.to(device)

        #    Prepare RoPE
        if self.model_settings.rope_type in ["3d", "3d-interleave"]:
            head_dim = self.model_settings.n_embd // self.model_settings.n_head
            text_dim = int(head_dim * self.model_settings.rotary_percentage)
            image_half_dim = (head_dim - text_dim) // 2
            assert text_dim + 2 * image_half_dim == head_dim, "RoPE dimension mismatch"
            rope_dim_list = [text_dim, image_half_dim, image_half_dim]
            rope_kwargs = dict(theta=self.args.get('rope_theta', 10000),
                               interleave=self.model_settings.rope_type == "3d-interleave")
            freqs_cos, freqs_sin = get_mlm_rope(
                rope_dim_list, tk_height, tk_width, self.arrange.s_text_maxlen, self.arrange.s_text_maxlen - 2,
                device, **rope_kwargs)
        elif self.model_settings.rope_type == "default":
            freqs_cos, freqs_sin = None, None
        else:
            raise ValueError(f"Unknown RoPE type: {self.model_settings.rope_type}")

        # 5. Prepare the mask scheduler
        orders = self.sample_orders(bsz, image_token_len, device)
        schedule_type = default(schedule_type, self.args.schedule_type)
        shift = default(shift, self.args.mask_ratio_shift)
        scheduler = adap_sche(num_inference_steps, image_token_len, mode=schedule_type, shift=shift)
        t_count = 0

        # 6. Denoising loop
        self._num_timesteps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, step in enumerate(range(num_inference_steps)):
                if self.do_classifier_free_guidance:
                    latents_input = torch.cat([latents, latents], dim=0)
                    mask_input = torch.cat([mask, mask], dim=0)
                else:
                    latents_input = latents
                    mask_input = mask

                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    logits = self.model(
                        idx=whole_tokens,
                        attention_mask=attention_mask,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        imgs=latents_input,
                        imgs_pos=imgs_start_pos,
                        imgs_mask=mask_input,
                    )['logits']

                t_count += scheduler[step]
                mask_len = image_token_len - t_count
                mask_next = self.mask_by_order(mask_len, orders, bsz, image_token_len, device)
                if step == num_inference_steps - 1:
                    mask_to_pred = mask.bool()
                else:
                    mask_to_pred = torch.logical_xor(mask, mask_next)
                mask = mask_next
                if self.do_classifier_free_guidance:
                    mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
                cond = logits[mask_to_pred]

                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    sampled_token_latent = self.pipeline(
                        cond=cond,
                        generator=generator,
                        num_inference_steps=self.args.diff_infer_steps,
                        guidance_scale=self.guidance_scale(step),
                    )[0]
                latents[mask_to_pred[:bsz]] = sampled_token_latent

                progress_bar.update()

        # 7. unpatchify
        latents = self.model.unpatchify(latents, tk_height, tk_width)

        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor
        
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if logger is not None: logger.info(f"image after postprocess with do_denormalize, output_type {output_type}")        
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
