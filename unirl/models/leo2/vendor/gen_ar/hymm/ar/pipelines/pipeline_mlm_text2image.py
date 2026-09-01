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

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import (
    logging,
)

from ..mask_schedulers import adap_sche, create_attention_mask_t2i
from ...models.basic.rope import get_mlm_rope
from ...models.tokenizers import TokenizerWrapper
from ...models.arrangement import Arrangement
from ...utils.helpers import default, repeat_interleave
from ...utils.torch_distributions import (
    categorical_sample,
    gumbel_sample,
)
from ...utils.torch_utils import PRECISION_TO_TYPE

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Text2ImageMLMPipeline(DiffusionPipeline):
    def __init__(self,
                 vae,
                 model,
                 tokenizer: TokenizerWrapper,
                 arrange: Arrangement,
                 progress_bar_config: Dict[str, Any] = None,
                 args=None,
                 model_settings=None,
                 logits_processor: Optional["LogitsProcessorList"] = None,
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
        )
        self.logits_processor = logits_processor
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
        batch_whole_tokens = []
        for prompt in prompts:
            whole_tokens = self.tokenizer.encode_mlm_t2i(
                text=prompt,
                image_token_len=image_token_len,
                max_text_token_length=self.arrange.s_text_maxlen,
                text_uncond_p=0,
            )[:-1]  # :-1 remove the eos token
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
                    text=text,
                    image_token_len=image_token_len,
                    max_text_token_length=self.arrange.s_text_maxlen,
                    text_uncond_p=uncond_p,
                )[:-1]  # :-1 remove the eos token
                batch_uncond_whole_tokens.append(neg_whole_tokens)
            batch_uncond_whole_tokens = torch.stack(batch_uncond_whole_tokens, dim=0)
            batch_uncond_whole_tokens = batch_uncond_whole_tokens.to(device)
        else:
            batch_uncond_whole_tokens = None

        return batch_whole_tokens, batch_uncond_whole_tokens

    def prepare_tokens_and_mask(self, batch_size, height, width, device, init_tokens=None, init_mask=None):
        shape = (
            batch_size,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if init_tokens is None:
            init_tokens = torch.full(shape, self.arrange.mask_id).to(device)
        else:
            init_tokens = init_tokens.to(device)
        if init_mask is None:
            init_mask = torch.ones(shape).to(device)
        else:
            init_mask = init_mask.to(device)

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
        randomize: str = 'linear',
        temperature: float = 4.5,
        schedule_type: str = 'arccos',
        shift: float = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        negative_prompt: Optional[List[str]] = None,
        num_sample_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_tokens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        self._guidance_scale = guidance_scale
        self._guidance_dynamic = guidance_dynamic

        if self.logits_processor is not None:
            kwargs = {}
            if top_k is not None:
                kwargs['top_k'] = top_k
            if top_p is not None:
                kwargs['top_p'] = top_p
            if kwargs:
                self.logits_processor.update(**kwargs)

        # 2. Define call parameters
        batch_size = len(prompt)

        device = self._execution_device

        target_dtype = PRECISION_TO_TYPE[self.args.autocast_dtype]
        autocast_enabled = (target_dtype != torch.float32)

        # 3. Prepare initial masked tokens
        masked_tokens, mask = self.prepare_tokens_and_mask(
            batch_size * num_sample_per_prompt,
            height,
            width,
            device,
            image_tokens,
            mask,
        )
        total_bs, tk_height, tk_width = masked_tokens.shape
        image_token_len = tk_height * tk_width
        # Flatten the spatial dimensions
        masked_tokens = masked_tokens.flatten(1)
        mask = mask.flatten(1)

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
        #   Slice of the runtime sequence image range
        slice_rt_s_image = slice(self.arrange.s_image_range[0], self.arrange.s_image_range[0] + image_token_len)

        #   Create attention mask: [bs, 1, seqlen, seqlen]
        attention_mask = create_attention_mask_t2i(
            whole_tokens,
            self.arrange.pad_id,
            self.arrange.boi_id,
            self.arrange.eoi_id,
            mask_pad=True,
            return_inverse_mask=True,
        )
        if self.do_classifier_free_guidance:
            if all([x == '' for x in negative_prompt]):
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            elif all([x != '' for x in negative_prompt]):
                negative_attention_mask = create_attention_mask_t2i(
                    uncond_whole_tokens,
                    self.arrange.pad_id,
                    self.arrange.boi_id,
                    self.arrange.eoi_id,
                    mask_pad=True,
                    return_inverse_mask=True,
                )
                attention_mask = torch.cat([negative_attention_mask, attention_mask], dim=0)
            else:
                raise ValueError(f"Negative prompt should be all empty or all non-empty, got {negative_prompt}")
        attention_mask = attention_mask.to(device)

        #    Prepare RoPE
        if self.model_settings.rope_type in ["3d", "3d-interleave"]:
            head_dim = self.model_settings.n_embd // self.model_settings.n_head
            text_dim = int(head_dim * self.model_settings.rotary_percentage)
            image_half_dim = (head_dim - text_dim) // 2
            assert text_dim + 2 * image_half_dim == head_dim, "RoPE dimension mismatch"
            rope_dim_list = [text_dim, image_half_dim, image_half_dim]
            rope_kwargs = dict(theta=self.args.get('rope_theta', 10000),
                               interleave=self.model_settings.rope_type == "3d-interleave",
                               theta_rescale_factor=self.args.rope_base_rescale_factor)
            freqs_cos, freqs_sin = get_mlm_rope(
                rope_dim_list, tk_height, tk_width, self.arrange.s_text_maxlen, self.arrange.s_text_maxlen - 2,
                device, **rope_kwargs)
        elif self.model_settings.rope_type == "default":
            freqs_cos, freqs_sin = None, None
        else:
            raise ValueError(f"Unknown RoPE type: {self.model_settings.rope_type}")

        # 5. Prepare schedule
        schedule_type = default(schedule_type, self.args.schedule_type)
        scheduler = adap_sche(num_inference_steps, image_token_len, mode=schedule_type, shift=shift)

        # 6. Denoising loop
        self._num_timesteps = len(scheduler)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                # Fill the image tokens and concat uncond batches
                whole_tokens_input = whole_tokens.clone()
                whole_tokens_input[:, slice_rt_s_image] = masked_tokens
                if self.do_classifier_free_guidance:
                    uncond_whole_tokens_input = uncond_whole_tokens.clone()
                    uncond_whole_tokens_input[:, slice_rt_s_image] = masked_tokens
                    whole_tokens_input = torch.cat([uncond_whole_tokens_input, whole_tokens_input], dim=0)

                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    logits = self.model(
                        idx=whole_tokens_input,
                        attention_mask=attention_mask,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                    )['logits']
                    # logits should be float32
                    logits = logits[:, slice_rt_s_image, slice(*self.arrange.image_id_range)]

                if self.do_classifier_free_guidance:
                    logits_uncond, logits_text = logits.chunk(2)
                    logits = logits_uncond + self.guidance_scale(i) * (logits_text - logits_uncond)

                if self.logits_processor is not None:
                    logits = self.logits_processor(None, logits)

                probs = torch.softmax(logits, -1)
                # Sample the code from the softmax prediction
                # pred_tokens = torch.distributions.Categorical(probs=probs).sample()
                pred_tokens = categorical_sample(probs, generator=generator)
                # Compute the confidence of the prediction
                conf = torch.gather(probs, 2, pred_tokens.unsqueeze(-1)).squeeze(-1)

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = i / (self.num_timesteps - 1)
                    rand = temperature * gumbel_sample(shape=conf.shape, generator=generator) * (1 - ratio)
                    conf = torch.log(conf) + rand.to(device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if i < 2 else conf
                elif randomize == "random":  # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                thresh_conf, indice_mask = torch.topk(conf.view(conf.size(0), -1), k=t, dim=-1)
                thresh_conf = thresh_conf[:, -1:]

                # replace the chosen tokens
                f_mask = (mask.float() * (conf >= thresh_conf).float()).bool()
                masked_tokens[f_mask] = pred_tokens[f_mask] + self.arrange.image_id_range[0]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0

                progress_bar.update()

        # 7. Clamp and reshape the final masked tokens
        masked_tokens = masked_tokens - self.arrange.image_id_range[0]
        masked_tokens = masked_tokens.view(total_bs, tk_height, tk_width)

        # 8. Decode the final masked tokens
        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image = self.vae.vq_decode(masked_tokens)

        # 9. Postprocess the final image
        transform_range = kwargs.get('transform_range', '-11')
        if transform_range == '01':
            image = self.image_processor.postprocess(
                image.clamp(0, 1), output_type=output_type, do_denormalize=[False] * image.shape[0])
        elif transform_range == 'minmax':
            low, high = float(image.min()), float(image.max())
            image.clamp_(low, high)
            image.sub_(low).div_(max(high - low, 1e-5))
            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=[False] * image.shape[0])
        else:
            image = self.image_processor.postprocess(
                image, output_type=output_type, do_denormalize=[True] * image.shape[0])

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
