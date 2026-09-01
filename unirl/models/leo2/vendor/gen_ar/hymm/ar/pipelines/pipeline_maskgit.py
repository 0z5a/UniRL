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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.utils import (
    logging,
)

from ...utils.torch_distributions import (
    categorical_sample,
    gumbel_sample,
)
from ...utils.torch_utils import PRECISION_TO_TYPE

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class MaskGITPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[Image.Image], np.ndarray]


class MaskGITPipeline(DiffusionPipeline):
    def __init__(self,
                 vae,
                 model,
                 model_settings=None,
                 progress_bar_config: Dict[str, Any] = None,
                 args=None,
                 ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        self.set_progress_bar_config(**progress_bar_config)

        self.args = args
        if model_settings.vocab_image_first:
            self.image_vocab_range = (0, model_settings.media_vocab_size)
            self.mask_id = model_settings.media_vocab_size
            self.label_vocab_range = (model_settings.media_vocab_size + 1,
                                      model_settings.media_vocab_size + 1 + model_settings.padded_vocab_size)
        else:
            self.label_vocab_range = (0, model_settings.padded_vocab_size)
            self.image_vocab_range = (model_settings.padded_vocab_size,
                                      model_settings.padded_vocab_size + model_settings.media_vocab_size)
            self.mask_id = model_settings.padded_vocab_size + model_settings.media_vocab_size
        self.uncond_id = model_settings.media_vocab_size + 1 + model_settings.padded_vocab_size
        # ==========================================================================================

        self.register_modules(
            vae=vae,
            model=model,
        )
        self.vae_scale_factor = self.vae.downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_label(
            self,
            labels,
            device,
            num_images_per_label,
            do_classifier_free_guidance,
    ):
        if isinstance(labels, int):
            labels = [labels]
        if isinstance(labels, list):
            labels = torch.LongTensor(labels)
        labels = labels.view(-1, 1).to(device)
        labels = labels.repeat(1, num_images_per_label).view(-1, 1)
        label_tokens = labels + self.label_vocab_range[0]

        if do_classifier_free_guidance:
            uncond_label_tokens = torch.full_like(labels, self.uncond_id)
            label_tokens = torch.cat([uncond_label_tokens, label_tokens], dim=0)

        return label_tokens

    def adap_sche(self, step, image_seq_len, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
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

    def prepare_tokens_and_mask(self, batch_size, height, width, device, init_tokens=None, init_mask=None):
        shape = (
            batch_size,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if init_tokens is None:
            init_tokens = torch.full(shape, self.mask_id).to(device)
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
        height: int,
        width: int,
        label: Union[int, List[int], torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        guidance_dynamic: str = 'linear_inc',
        randomize: str = 'linear',
        temperature: float = 4.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_label: Optional[int] = 1,
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

        # 2. Define call parameters
        if isinstance(label, int):
            batch_size = 1
        else:
            batch_size = len(label)

        device = self._execution_device

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (target_dtype != torch.float32)

        # 3. Prepare labels
        label_tokens = self.encode_label(
            label,
            device,
            num_images_per_label,
            self.do_classifier_free_guidance,
        )

        # 4. Prepare initial masked tokens
        masked_tokens, mask = self.prepare_tokens_and_mask(
            batch_size * num_images_per_label,
            height,
            width,
            device,
            image_tokens,
            mask,
        )
        total_bs, token_height, token_width = masked_tokens.shape
        image_seq_len = token_height * token_width
        # Flatten the spatial dimensions
        masked_tokens = masked_tokens.flatten(1)
        mask = mask.flatten(1)

        # 5. Prepare schedule
        scheduler = self.adap_sche(num_inference_steps, image_seq_len, mode=self.args.schedule_type)

        # 6. Denoising loop
        self._num_timesteps = len(scheduler)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                masked_tokens_input = torch.cat([masked_tokens] * 2) if self.do_classifier_free_guidance else masked_tokens
                masked_tokens_input = torch.cat([masked_tokens_input, label_tokens], dim=1)

                with torch.cuda.amp.autocast():
                    logits = self.model(
                        x=masked_tokens_input,
                    )['logits']
                    logits = logits[:, :image_seq_len, slice(self.image_vocab_range[0], self.image_vocab_range[1] + 1)]

                if self.do_classifier_free_guidance:
                    logits_uncond, logits_label = logits.chunk(2)
                    logits = logits_uncond + self.guidance_scale(i) * (logits_label - logits_uncond)

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
                masked_tokens[f_mask] = pred_tokens[f_mask] + self.image_vocab_range[0]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0

                progress_bar.update()

        # 7. Clamp and reshape the final masked tokens
        masked_tokens = torch.clamp(
            masked_tokens, self.image_vocab_range[0], self.image_vocab_range[1] - 1
        ) - self.image_vocab_range[0]
        masked_tokens = masked_tokens.view(total_bs, token_height, token_width)

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

        return MaskGITPipelineOutput(images=image)
