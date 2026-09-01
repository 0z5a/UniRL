import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from typing import Optional, Tuple, Union
import math
import copy

import numpy as np
import torch
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler: SchedulerMixin,
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


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
            self,
            pred_cond: torch.Tensor,
            pred_uncond: Optional[torch.Tensor],
            guidance_scale: float,
            step: int,
    ) -> torch.Tensor:

        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


@dataclass
class HunyuanMultimodalPipelineOutput(BaseOutput):
    samples: Union[List[Any], np.ndarray]


class HunyuanMultimodalPipeline(DiffusionPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model ([`ModelMixin`]):
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
            model,
            scheduler: SchedulerMixin,
            vae,
            vae_autocast_dtype: Optional[torch.dtype] = None,
            progress_bar_config: Dict[str, Any] = None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.vae_autocast_dtype = vae_autocast_dtype
        # ==========================================================================================

        self.register_modules(
            model=model,
            scheduler=scheduler,
            vae=vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.latent_scale_factor = self.model.config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

        # Must start with APG_mode_
        self.cfg_operator = ClassifierFreeGuidance()

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

    @staticmethod
    def prepare_extra_func_kwargs(func, kwargs):
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

    def prepare_latents(self, batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
        # Variable resolution: image_size is a list of (h, w) tuples with different sizes
        is_variable_res = isinstance(image_size, list)
        if is_variable_res:
            assert len(image_size) == batch_size, f"image_size length (={len(image_size)}) should be the same as batch_size (={batch_size})"
            return self._prepare_latents_variable_res(
                batch_size, latent_channel, image_size, dtype, device, generator, latents,
            )

        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(self.latent_scale_factor, list):
            assert len(self.latent_scale_factor) == len(image_size), \
                "len(latent_scale_factor) shoudl be the same as len(image_size)"
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {self.latent_scale_factor}"
            )

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        n_tokens = math.prod([int(s) // f for s, f in zip(image_size, latent_scale_factor)])
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

        return latents, n_tokens

    def _prepare_latents_variable_res(
        self, batch_size, latent_channel, image_sizes, dtype, device, generator, latents=None,
    ):
        """
        Prepare per-sample latents for variable resolution batch inference.
        """
        assert len(image_sizes) == batch_size

        ndim = len(image_sizes[0])
        if self.latent_scale_factor is None:
            scale_factors = (1,) * ndim
        elif isinstance(self.latent_scale_factor, int):
            scale_factors = (self.latent_scale_factor,) * ndim
        elif isinstance(self.latent_scale_factor, (tuple, list)):
            scale_factors = tuple(self.latent_scale_factor)
        else:
            raise ValueError(f"Unsupported latent_scale_factor: {self.latent_scale_factor}")

        latents_list = []
        n_tokens_list = []
        for i, img_size in enumerate(image_sizes):
            lat_dims = tuple(int(s) // f for s, f in zip(img_size, scale_factors))
            shape = (1, latent_channel, *lat_dims)
            gen = generator[i] if isinstance(generator, list) else generator
            if latents is None:
                lat = randn_tensor(shape, generator=gen, device=device, dtype=dtype)
            else:
                lat = latents[i] if isinstance(latents, list) else latents[i:i+1]
                lat = lat.to(device)
            if hasattr(self.per_sample_schedulers[i], "init_noise_sigma"):
                lat = lat * self.per_sample_schedulers[i].init_noise_sigma
            latents_list.append(lat)
            n_tokens_list.append(math.prod(lat_dims))

        return latents_list, n_tokens_list

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
        return self._cfg_factor > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def set_scheduler(self, new_scheduler):
        self.register_modules(scheduler=new_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        image_size: tuple[int, int] | list[tuple[int, int]],
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],    # noqa
        model_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> Union[HunyuanMultimodalPipelineOutput, tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The text to guide image generation.
            image_size (`Tuple[int]` or `List[int]`):
                The size (height, width) of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
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
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
                generation. Can be used to tweak the same generation with different conditions. If not provided,
                a latents tensor is generated by sampling using the supplied random `generator`.
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

        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        cfg_factor = kwargs.pop('cfg_factor', None)
        if cfg_factor is None:
            cfg_factor = 1 + (self._guidance_scale > 1.0)
        self._cfg_factor = cfg_factor
        cfg_distilled = kwargs.pop('cfg_distilled', False)
        meanflow = kwargs.pop('meanflow', False)
        device = self.model.device

        is_variable_res = isinstance(image_size, list)
        # batch多分辨率推理： 为每个sample生成独立的 scheduler
        if is_variable_res:
            self.per_sample_schedulers = [copy.deepcopy(self.scheduler) for _ in range(batch_size)]

        # Prepare latent variables
        latents, n_tokens = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.model.config.vae_latent_dim,
            image_size=image_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        # Prepare timesteps
        # TODO(kevinkhwu): check the correctness of n_tokens (image/video).
        if is_variable_res:
            # batch多分辨率推理：每 sample 独立 timesteps
            per_sample_timesteps = []
            for b_idx, nt in enumerate(n_tokens):
                ts_i, num_inference_steps = retrieve_timesteps(
                    self.per_sample_schedulers[b_idx], num_inference_steps, device, timesteps, sigmas, n_tokens=nt,
                )
                per_sample_timesteps.append(ts_i)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas, n_tokens=n_tokens,
            )

        # Prepare extra step kwargs.
        if is_variable_res:
            if isinstance(generator, list):
                _scheduler_step_extra_kwargs_list = [self.prepare_extra_func_kwargs(
                    self.per_sample_schedulers[b_idx].step, {"generator": generator[b_idx]}
                ) for b_idx in range(batch_size)]
            else:
                _scheduler_step_extra_kwargs_list = [self.prepare_extra_func_kwargs(
                    self.per_sample_schedulers[b_idx].step, {"generator": generator}
                ) for b_idx in range(batch_size)]
        else:
            _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
                self.scheduler.step, {"generator": generator}
            )

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device)

        # cfg_distilled
        if cfg_distilled:
            model_kwargs["guidance"] = torch.tensor(
                [[1000.0*self._guidance_scale]], device=device, dtype=torch.float32
            ).expand(batch_size, -1)

        # Sampling loop
        if is_variable_res:
            timesteps = per_sample_timesteps[0]
            scheduler_order = self.per_sample_schedulers[0].order
        else:
            scheduler_order = self.scheduler.order
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler_order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if is_variable_res:
                    # batch多分辨率推理始终传 list：
                    # -- 1. 第一步中instantiate_vae_image_tokens会逐sample scatter到统一的hidden_states 中；
                    # -- 2. 后续步中会逐sample做patch_embed再进行 1D padding，保证有效token在前、pad在后
                    latent_model_input = [lat.clone() for lat in latents] * cfg_factor

                    # NOTE: 每个sample有独立的timestep（flux-style shift 依赖 n_tokens）
                    per_t = [per_sample_timesteps[j][i] for j in range(batch_size)]
                    per_t_all = per_t * cfg_factor
                    latent_model_input = [
                        self.per_sample_schedulers[k%batch_size].scale_model_input(lat, t_i)
                        for k, (lat, t_i) in enumerate(zip(latent_model_input, per_t_all))
                    ]
                    t_expand = torch.stack(per_t_all)
                    if meanflow:
                        per_t_r = [self.per_sample_schedulers[j].get_timestep_r(per_sample_timesteps[j][i]) for j in range(batch_size)]
                        per_t_r_all = per_t_r * cfg_factor
                        t_r_expand = torch.stack(per_t_r_all)
                    else:
                        t_r_expand = None
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * cfg_factor)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    t_expand = t.repeat(latent_model_input.shape[0])
                    if meanflow:
                        t_r = self.scheduler.get_timestep_r(t)
                        t_r_expand = t_r.repeat(latent_model_input.shape[0])
                    else:
                        t_r_expand = None

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    images=latent_model_input,
                    timesteps=t_expand,
                    timestep_r=t_r_expand,
                    **model_kwargs,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = self.model(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]

                # 多分辨率走 per-sample 路径，统一分辨率走 batched 路径
                if is_variable_res:
                    pred = [p.to(dtype=torch.float32) for p in pred]

                    if self.do_classifier_free_guidance:
                        half = len(pred) // 2
                        pred_cond_list = pred[:half]
                        pred_uncond_list = pred[half:]
                        pred = [
                            self.cfg_operator(pc, pu, self.guidance_scale, step=i)
                            for pc, pu in zip(pred_cond_list, pred_uncond_list)
                        ]

                        if self.guidance_rescale > 0.0:
                            pred = [
                                rescale_noise_cfg(p, pc, guidance_rescale=self.guidance_rescale)
                                for p, pc in zip(pred, pred_cond_list)
                            ]

                    # per-sample scheduler step。每个 sample 使用自己的 timestep，scheduler 是单例，需要在每个 sample 调用前：
                    # -- 1. 设置正确的 step_index（防止被多次 +1）
                    # -- 2. 换入该 sample 对应的 sigmas（flux-shift 下不同 n_tokens 有不同 schedule）
                    for j in range(batch_size):
                        latents[j] = self.per_sample_schedulers[j].step(
                            pred[j], per_sample_timesteps[j][i], latents[j],
                            **_scheduler_step_extra_kwargs_list[j], return_dict=False,
                        )[0]
                else:
                    pred = pred.to(dtype=torch.float32)
                    if pred.ndim == 5 and pred.size(2) == 1 and latents.ndim == 4:
                        pred = pred.squeeze(2)

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        pred_cond, pred_uncond = pred.chunk(2)
                        pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

                if i != len(timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    input_ids = None
                    # if input_ids.shape[1] != model_kwargs["input_pos"].shape[1]:
                    #     input_ids = torch.gather(input_ids, 1, index=model_kwargs["input_pos"])

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler_order == 0):
                    progress_bar.update()

        # ---- VAE decode ----
        if is_variable_res:
            # 多分辨率batch推理：逐sample VAE decode
            all_images = []
            for lat in latents:
                if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
                    lat = lat / self.vae.config.scaling_factor
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                    lat = lat + self.vae.config.shift_factor
                if hasattr(self.vae, "ffactor_temporal"):
                    lat = lat.unsqueeze(2)
                with torch.autocast(
                    device_type="cuda", dtype=self.vae_autocast_dtype,
                    enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32,
                ):
                    img = self.vae.decode(lat, return_dict=False, generator=generator)[0]
                if hasattr(self.vae, "ffactor_temporal"):
                    assert img.shape[2] == 1
                    img = img.squeeze(2)
                do_denormalize = [True] * img.shape[0]
                pil_imgs = self.image_processor.postprocess(img, output_type=output_type, do_denormalize=do_denormalize)
                all_images.extend(pil_imgs)

            if not return_dict:
                return (all_images,)
            return HunyuanMultimodalPipelineOutput(samples=all_images)

        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        with torch.autocast(
                device_type="cuda", dtype=self.vae_autocast_dtype,
                enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32
        ):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image,)

        return HunyuanMultimodalPipelineOutput(samples=image)
