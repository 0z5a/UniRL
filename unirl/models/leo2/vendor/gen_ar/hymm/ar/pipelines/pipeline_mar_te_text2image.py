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


class Text2ImageMARTEPipeline(DiffusionPipeline):
    def __init__(self,
                 vae,
                 model,
                 tokenizer: TokenizerWrapper,
                 arrange: Arrangement,
                 pipeline: DiffusionPipeline,
                 text_encoder: Optional[Any],
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
            text_encoder=text_encoder,
        )
        self.tokenizer = tokenizer
        self.arrange = arrange
        self.vae_scale_factor = self.vae.downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        text_encoder=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            text_encoder (TextEncoder, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(prompt)

            prompt_outputs = text_encoder.encode(text_inputs)
            prompt_embeds = prompt_outputs.hidden_state

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_images_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens)

            # if hasattr(text_encoder.model.config, "use_attention_mask") and text_encoder.model.config.use_attention_mask:
            #     attention_mask = uncond_input.attention_mask.to(device)
            # else:
            #     attention_mask = None

            negative_prompt_outputs = text_encoder.encode(uncond_input)
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(1, num_images_per_prompt)
                negative_attention_mask = negative_attention_mask.view(batch_size * num_images_per_prompt, seq_len)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

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
        # 4.2 prepare the text tokens
        prompt_embeds, negative_prompt_embeds, _, _ = \
            self.encode_prompt(
                prompt,
                device,
                num_sample_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
            )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

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
                rope_dim_list, tk_height, tk_width, self.args.text_token_length + 1, self.args.text_token_length,
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
                        text_states=prompt_embeds,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        imgs=latents_input,
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

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
