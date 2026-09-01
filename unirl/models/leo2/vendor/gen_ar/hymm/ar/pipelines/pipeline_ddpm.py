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
# Modified by jarvizhang
# ==============================================================================

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


@dataclass
class PipelineOutput(BaseOutput):
    """
    Output class for DDPM 1d pipelines.

    Args:
        x (torch.Tensor):
            denoised logits of shape `(batch_size, channels)`.
    """

    x: torch.Tensor


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for 1d-tensor generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        model ([`ModelMixin`]):
            A model to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "model"

    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @torch.no_grad()
    def __call__(
        self,
        cond: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        guidance_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[PipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            cond (`torch.Tensor`):
                The condition tensor of shape `(batch_size, channels)`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                A higher guidance scale value encourages the model to generate images closely linked to the condition
                at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        self._guidance_scale = guidance_scale
        bsz_factor = 2 if self.do_classifier_free_guidance else 1

        # 1. Prepare latents
        x_shape = (cond.shape[0] // bsz_factor, *cond.shape[1:-1], self.model.in_channels)
        if generator is not None and isinstance(generator, list):
            gen_bs = len(generator)
            # Repeat generator to match batch size
            assert x_shape[0] % gen_bs == 0, "The batch size must be a multiple of the number of generators."
            if x_shape[0] != gen_bs:
                generator = [gen for gen in generator for _ in range(x_shape[0] // gen_bs)]
        x = randn_tensor(x_shape, generator=generator, device=self.device, dtype=self.model.dtype)

        # 2. set step values
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 3. Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            x_input = torch.cat([x] * 2) if self.do_classifier_free_guidance else x
            t_expand = t.repeat(x_input.shape[0])

            # 3. predict noise model_output
            output = self.model(x_input, t_expand, cond)['x']
            # output, rest = torch.chunk(output, 2, dim=1)

            if self.do_classifier_free_guidance:
                output_uncond, output_cond = torch.chunk(output, 2)
                output = output_uncond + self.guidance_scale * (output_cond - output_uncond)

            # output = torch.cat([output, torch.chunk(rest, 2)[0]], dim=1)

            # 2. compute previous image: x_t -> x_t-1
            x = self.scheduler.step(output, t, x, generator=generator).prev_sample

        if not return_dict:
            return (x,)

        return PipelineOutput(x=x)
