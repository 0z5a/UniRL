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
# Created by @ckczzjzhang
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from einops import rearrange
from PIL import Image

import numpy as np
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import BaseOutput, deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.models.modeling_utils import ModelMixin

from hymm.utils.torch_utils import PRECISION_TO_TYPE


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Token2ImagePipelineOutput(BaseOutput):
    samples: Union[List[Any], np.ndarray]


def rescale_noise_cfg(noise_cfg, noise_pred_cond, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" samples
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Token2ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        diffusion_model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = ""
    _optional_components = []
    _exclude_from_cpu_offload = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        diffusion_model: ModelMixin,
        scheduler: SchedulerMixin,
        vae,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()
        self.args = args

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        # ==========================================================================================

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            vae=vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.latent_scale_factor = None

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    # transform generated latents to samples according to output_type
    def postprocess_latents(self, latents, output_type):
        if output_type == "latent":
            return latents

        do_denormalize = [True] * latents.shape[0]

        images = torch.stack(
            [self.denormalize(latents[i]) if do_denormalize[i] else latents[i] for i in range(latents.shape[0])]
        )

        if output_type == "pt":
            return images

        images = self.pt_to_numpy(images)

        if output_type == "np":
            return images

        if output_type == "pil":
            return self.numpy_to_pil(images)

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_kwargs[k] = v
        return extra_kwargs

    def prepare_latents(self, batch_size, latent_channel, sample_size, dtype, device, generator, latents=None):
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(sample_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(sample_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(self.latent_scale_factor, list):
            assert len(self.latent_scale_factor) == len(sample_size), "len(latent_scale_factor) shoudl be the same as len(sample_size)"
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(f"latent_scale_factor should be either None, int, tuple of int, or list of int, but got {self.latent_scale_factor}")

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(sample_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        sample_size: Union[Tuple[int], List[int]] = (512, 512),
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        scheduler_set_timesteps_extra_kwargs: Dict[str, Any] = {},   # kwargs to pass to scheduler.set_timesteps, such as n_tokens for flux_shift
        scheduler_step_extra_kwargs: Dict[str, Any] = {},   # kwargs to pass to scheduler.step, such as eta for DDIMScheduler
        model_input_extra_kwargs: Dict[str, Any] = {},   # kwargs to pass to diffusion_model.forward, such as RoPE freqs
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            condition (`Any` or `List[Any]`):
                The condition to guide sample generation. If not defined, you need to pass `condition_embeds`.
            sample_size (`Tuple[int]` or `List[int]`):
                The size of the generated sample.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality sample at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the
                `condition` at the expense of lower sample quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_condition (`str` or `List[str]`, *optional*):
                The condition to guide what to not include in sample generation. If not defined, you need to
                pass `negative_condition_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            condition_embeds (`torch.Tensor`, *optional*):
                Pre-generated condition embeddings. Can be used to easily tweak condition inputs. If not
                provided, condition embeddings are generated from the `condition` input argument.
            negative_condition_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative condition embeddings. Can be used to easily tweak condition inputs. If
                not provided, `negative_condition_embeds` are generated from the `negative_condition` input argument.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
                generation. Can be used to tweak the same generation with different conditions. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~DiffusionPipelineOutput`] instead of a
                plain tuple.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~DiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~DiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated samples.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        batch_size = model_input_extra_kwargs["cond"].shape[0]
        if self.do_classifier_free_guidance:
            batch_size = batch_size // 2

        device = self._execution_device

        # Prepare timesteps
        _scheduler_set_timesteps_extra_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, scheduler_set_timesteps_extra_kwargs
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, **_scheduler_set_timesteps_extra_kwargs,
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.diffusion_model.config.model_config.in_channels,
            sample_size=sample_size,
            dtype=self.diffusion_model.config.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # Prepare extra step kwargs.
        scheduler_step_extra_kwargs.update({"generator": generator})
        _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, scheduler_step_extra_kwargs,
        )

        # Sampling loop
        target_dtype = PRECISION_TO_TYPE[self.args.autocast_dtype]
        autocast_enabled = (target_dtype != torch.float32)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                # predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    pred = self.diffusion_model(
                        x=latent_model_input,
                        t=t_expand,
                        **model_input_extra_kwargs,
                        return_dict=True,
                    )['x']

                pred = pred.to(dtype=torch.float32)

                # perform guidance
                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2)
                    pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)


        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        vae_autocast_dtype = PRECISION_TO_TYPE[self.args.vae_autocast_dtype]
        with torch.autocast(device_type="cuda", dtype=vae_autocast_dtype, enabled=vae_autocast_dtype != torch.float32):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        samples = self.postprocess_latents(image, output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples, )

        return Token2ImagePipelineOutput(samples=samples)
