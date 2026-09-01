import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from hymm.models.visual_encoders.qwen.qwen_vit import (
    flatten_qwen3vl_pixel_values as _flatten_qwen3vl_pixel_values,
    flatten_qwen3vl_grid_thw as _flatten_qwen3vl_grid_thw,
)

from hymm.models.autoencoders import denormalize_vae_latents

if TYPE_CHECKING:
    from hymm.data_kits.utils.audio_utils import AudioProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _pad_cond_vae_channels(cond_vae_images, target_latent_channels: Optional[int]):
    """Match cond_vae latent channels to the DiT patch_embed input channels.
    """
    if cond_vae_images is None or target_latent_channels is None:
        return cond_vae_images

    def _pad_one(latent: torch.Tensor) -> torch.Tensor:
        channel_dim = 1 if latent.dim() == 5 else 0
        in_channels = latent.shape[channel_dim]
        if in_channels == target_latent_channels:
            return latent
        if in_channels > target_latent_channels:
            raise ValueError(
                f"cond_vae latent has {in_channels} channels, but model expects "
                f"{target_latent_channels}: shape={tuple(latent.shape)}"
            )
        pad_shape = list(latent.shape)
        pad_shape[channel_dim] = target_latent_channels - in_channels
        pad = torch.zeros(pad_shape, dtype=latent.dtype, device=latent.device)
        return torch.cat([latent, pad], dim=channel_dim)

    if isinstance(cond_vae_images, torch.Tensor):
        return _pad_one(cond_vae_images)
    return [
        [_pad_one(t) for t in item] if isinstance(item, list) else _pad_one(item)
        for item in cond_vae_images
    ]


def _merge_cond_vit_slices(image_slices, video_slices):
    """Merge per-sample reference-image and source-video ViT placeholder slices (r2v).
    """
    if not image_slices and not video_slices:
        return None
    image_slices = list(image_slices or [])
    video_slices = list(video_slices or [])
    batch_size = max(len(image_slices), len(video_slices))
    if image_slices and len(image_slices) != batch_size:
        raise ValueError(
            f"Image ViT slice batch has {len(image_slices)} rows, expected {batch_size}."
        )
    if video_slices and len(video_slices) != batch_size:
        raise ValueError(
            f"Video ViT slice batch has {len(video_slices)} rows, expected {batch_size}."
        )
    if not image_slices:
        image_slices = [[] for _ in range(batch_size)]
    if not video_slices:
        video_slices = [[] for _ in range(batch_size)]

    def _to_slice_list(sample_slices):
        if isinstance(sample_slices, (list, tuple)):
            return list(sample_slices)
        return [sample_slices]

    merged = []
    for sample_image_slices, sample_video_slices in zip(image_slices, video_slices):
        merged.append(_to_slice_list(sample_image_slices) + _to_slice_list(sample_video_slices))
    return merged


def _build_cond_text_scatter_mask(text_mask, cond_vit_image_slices):
    """Full-seqlen scatter target for r2v cond_text_states.
    """
    if not cond_vit_image_slices:
        return None
    scatter_mask = text_mask.clone().bool()
    seqlen = scatter_mask.size(1)
    for batch_idx, slices_i in enumerate(cond_vit_image_slices):
        if isinstance(slices_i, slice):
            slices_i = [slices_i]
        for sli in slices_i:
            scatter_mask[batch_idx, max(sli.start - 1, 0):min(sli.stop + 1, seqlen)] = True
    return scatter_mask.to(text_mask.dtype)


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


def black_image(width, height):
    # 创建一张黑色图像
    black_image = Image.new('RGB', (width, height), (0, 0, 0))
    return black_image


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
class Leo2PipelineOutput(BaseOutput):
    visuals: Union[List[Any], np.ndarray]
    audios: Optional[Union[List[Any], np.ndarray]] = None
    visual_latents: Optional[torch.Tensor] = None
    audio_latents: Optional[torch.Tensor] = None


class Leo2Pipeline(DiffusionPipeline):
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
            text_encoder,
            vae_autocast_dtype: Optional[torch.dtype] = None,
            progress_bar_config: Dict[str, Any] = None,
            args=None,
            video_scheduler: SchedulerMixin = None,
            audio_processor: "AudioProcessor" = None,
            audio_scheduler: SchedulerMixin = None,
            audio_vae=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.vae_autocast_dtype = vae_autocast_dtype
        self.args = args
        # ==========================================================================================

        self.register_modules(
            model=model,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            audio_vae=audio_vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.vae_spatial_downsample_factor = self.model.config.vae_spatial_downsample_factor
        self.vae_temporal_downsample_factor = self.model.config.vae_temporal_downsample_factor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_downsample_factor)
        self.image_processor = self.video_processor
        # Here we use custom audio processor due to the unique calculation of audio token duration.
        self.audio_processor = audio_processor

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

    def encode_prompt(
        self,
        model_kwargs: Dict[str, Any],
        text_encoder=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            model_kwargs (Dict[str, Any]):
                The model kwargs passed to the pipeline's `__call__` method. Should include `tokens` and `text_mask`
                for encoding the prompt.
            text_encoder (TextEncoder, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        tokens = model_kwargs["input_ids"]
        text_mask = model_kwargs["text_mask"]
        tokenizer_output = model_kwargs.get("tokenizer_output")
        cond_vit_image_slices = getattr(tokenizer_output, "vit_image_slices", None) if tokenizer_output is not None else None
        cond_vit_video_slices = getattr(tokenizer_output, "vit_video_slices", None) if tokenizer_output is not None else None
        cond_vit_video_context_slices = (
            getattr(tokenizer_output, "vit_video_context_slices", None)
            if tokenizer_output is not None
            else None
        )
        all_cond_vit_slices = _merge_cond_vit_slices(
            cond_vit_image_slices,
            cond_vit_video_slices,
        )
        # Remove tailing tokens with 0 attention mask to reduce the computation of text encoder without affecting the results.
        max_valid_len = (text_mask.size(1) - 1 - text_mask.flip(dims=[1]).argmax(dim=1)).max().item() + 1
        if all_cond_vit_slices is not None:
            max_cond_vit_len = max(
                (
                    sli.stop + 1
                    for slices_i in all_cond_vit_slices
                    for sli in (slices_i if isinstance(slices_i, (list, tuple)) else [slices_i])
                ),
                default=0,
            )
            max_valid_len = max(max_valid_len, max_cond_vit_len)
        if cond_vit_video_context_slices is not None:
            max_valid_len = max(
                max_valid_len,
                max(
                    (
                        sli.stop
                        for slices_i in cond_vit_video_context_slices
                        for sli in (
                            slices_i
                            if isinstance(slices_i, (list, tuple))
                            else [slices_i]
                        )
                    ),
                    default=0,
                ),
            )
        tokens = tokens[:, :max_valid_len]
        text_mask = text_mask[:, :max_valid_len]
        cond_vit_image_kwargs = model_kwargs.get("cond_vit_image_kwargs") or {}
        pixel_values = _flatten_qwen3vl_pixel_values(model_kwargs.get("cond_vit_images"))
        image_grid_thw = _flatten_qwen3vl_grid_thw(cond_vit_image_kwargs.get("grid_thw"))
        cond_vit_video_kwargs = model_kwargs.get("cond_vit_video_kwargs") or {}
        pixel_values_videos = _flatten_qwen3vl_pixel_values(
            model_kwargs.get("cond_vit_videos")
        )
        video_grid_thw = _flatten_qwen3vl_grid_thw(
            cond_vit_video_kwargs.get("video_grid_thw")
        )
        text_batch = dict(tokens=tokens, text_mask=text_mask)
        if tokenizer_output is not None:
            text_batch.update(dict(
                cond_vit_image_slices=cond_vit_image_slices,
                cond_vit_video_slices=cond_vit_video_slices,
                cond_vit_video_context_slices=cond_vit_video_context_slices,
            ))
        prompt_outputs = text_encoder.batch_encode_with_sp(
            text_batch,
            system_prompt=model_kwargs["system_prompt"],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        model_kwargs["cond_text_states"] = prompt_outputs.hidden_state.contiguous()   # (bsz, seq_len, hidden_dim)
        model_kwargs["cond_text_mask"] = prompt_outputs.attention_mask.contiguous().to(torch.int32)  # (bsz, seq_len)

        model_kwargs["cond_text_scatter_mask"] = _build_cond_text_scatter_mask(
            model_kwargs["text_mask"], all_cond_vit_slices,
        )

        return model_kwargs

    def prepare_audio_latents(self, batch_size, audio_duration, dtype, device, generator, latents=None,
                              token_length=None):
        # `token_length` (when provided) pins the audio latent length to the sequence's audio token count,
        # keeping generation consistent when the token count was pinned (e.g. validation-loss debug-gen).
        token_len = token_length if token_length is not None \
            else self.audio_processor.audio_vae_info.calc_token_duration(audio_duration)
        latents_shape = (
            batch_size,
            self.audio_processor.audio_vae_info.latent_dim,
            token_len,
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
        if hasattr(self.audio_scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.audio_scheduler.init_noise_sigma

        return latents

    def prepare_latents(self, batch_size, latent_channel, video_duration, height, width, dtype, device, generator,
                        latents=None):
        vae_spatial_downsample_factor = self.vae_spatial_downsample_factor
        if isinstance(vae_spatial_downsample_factor, int):
            vae_spatial_downsample_factor = (vae_spatial_downsample_factor,) * 2
        vae_temporal_downsample_factor = self.vae_temporal_downsample_factor
        assert isinstance(vae_spatial_downsample_factor, (tuple, list)) and len(vae_spatial_downsample_factor) == 2, \
            f"vae_spatial_downsample_factor should be a tuple or list of length 2, but got {vae_spatial_downsample_factor}"
        assert isinstance(vae_temporal_downsample_factor, int), \
            f"vae_temporal_downsample_factor should be an int, but got {vae_temporal_downsample_factor}"

        assert (video_duration - 1) % 4 == 0, \
            f"video_video_duration - 1 should be divisible by 4, but got {video_duration - 1}"
        latents_shape = (
            batch_size,
            latent_channel,
            (video_duration - 1) // vae_temporal_downsample_factor + 1,
            height // vae_spatial_downsample_factor[0],
            width // vae_spatial_downsample_factor[1],
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
        video_duration = self.scheduler if video_duration == 1 else self.video_scheduler
        if hasattr(video_duration, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * video_duration.init_noise_sigma

        return latents

    def prepare_channel_cond_latents(self, channel_cond_vae_images, latents):
        bsz, ch, tk_duration, tk_height, tk_width = latents.shape
        if channel_cond_vae_images is None:
            channel_cond_vae_images = [[] for _ in range(bsz)]

        cond_latents = torch.zeros(latents.shape, dtype=latents.dtype, device=latents.device)
        cond_mask = torch.zeros((tk_duration,), dtype=latents.dtype, device=latents.device)

        channel_task_type = None
        for i, channel_cond_vae_images_i in enumerate(channel_cond_vae_images):
            if len(channel_cond_vae_images_i) > 0:
                if channel_cond_vae_images_i.ndim == 4:
                    channel_cond_vae_images_i = channel_cond_vae_images_i.unsqueeze(0)
                # Now channel_cond_vae_images_i should be of shape [num_cond_frames, ch, 1, tk_height, tk_width]
                assert channel_cond_vae_images_i.ndim == 5, \
                    (f"Each element of channel_cond_vae_images should be a 4D tensor or a 5D tensor, "
                     f"but the element {i} got shape {channel_cond_vae_images_i.shape}")

            # TODO: Find a better way to determine the channel conditions.
            if len(channel_cond_vae_images_i) == 1:
                # i2v
                cond_first_frame = channel_cond_vae_images_i[0]
                assert cond_first_frame.shape == (ch, 1, tk_height, tk_width), \
                    (f"Expected cond_first_frame shape to be {(ch, 1, tk_height, tk_width)}, "
                     f"but got {cond_first_frame.shape}")
                cond_latents[i, :, 0:1] = cond_first_frame
                cond_mask[0] = 1.0
                if channel_task_type is not None:
                    assert channel_task_type == "i2v", \
                        (f"Got inconsistent channel condition types, expected all to be i2v but got "
                         f"{channel_task_type} for element {i}")
                channel_task_type = "i2v"
            elif len(channel_cond_vae_images_i) == 2:
                # fl2v
                cond_first_frame, cond_last_frame = channel_cond_vae_images_i
                assert cond_first_frame.shape == (ch, 1, tk_height, tk_width), \
                    (f"Expected cond_first_frame shape to be {(ch, 1, tk_height, tk_width)}, "
                     f"but got {cond_first_frame.shape}")
                assert cond_last_frame.shape == (ch, 1, tk_height, tk_width), \
                    (f"Expected cond_last_frame shape to be {(ch, 1, tk_height, tk_width)}, "
                     f"but got {cond_last_frame.shape}")
                cond_latents[i, :, 0:1] = cond_first_frame
                cond_latents[i, :, -1:] = cond_last_frame
                cond_mask[0] = 1.0
                cond_mask[-1] = 1.0
                if channel_task_type is not None:
                    assert channel_task_type == "fl2v", \
                        (f"Got inconsistent channel condition types, expected all to be fl2v but got "
                         f"{channel_task_type} for element {i}")
                channel_task_type = "fl2v"
            elif len(channel_cond_vae_images_i) > 2:
                raise ValueError(
                    f"Only up to 2 channel condition images are supported, but got {len(channel_cond_vae_images_i)}"
                )

        if channel_task_type is None:
            channel_task_type = "t2v"

        cond_mask = cond_mask.view(1, 1, tk_duration, 1, 1).repeat(bsz, 1, 1, tk_height, tk_width)
        return cond_latents, cond_mask, channel_task_type

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def guidance_scale_audio(self):
        return self._guidance_scale_audio

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def set_scheduler(self, new_scheduler):
        self.register_modules(scheduler=new_scheduler)

    @staticmethod
    def duplicate_generator(generator):
        if generator is None:
            return generator

        if isinstance(generator, list):
            return [g.clone_state() for g in generator]

        return generator.clone_state()

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int,
            image_size: List[int] | None,
            video_duration: int = 1,
            audio_duration: Optional[int] = None,
            audio_token_length: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
            guidance_scale_audio: Optional[float] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            negative_attention_mask: Optional[torch.Tensor] = None,
            output_type: Optional[str | dict] = "pil",
            return_dict: bool = True,
            guidance_rescale: float = 0.0,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],    # noqa
            model_kwargs: Dict[str, Any] = None,
            **kwargs,
    ) -> Union[Leo2PipelineOutput, tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The text to guide image generation.
            image_size (`Tuple[int]` or `List[int]`):
                The size (height, width) of the generated image/video.
            video_duration (`int`, *optional*, defaults to 1):
                The duration of the generated video in frames.
            audio_duration (`int`, *optional*, defaults to None):
                The duration of the generated audio in (video frames / fps * sample_rate).
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
            output_type (`str` or `dict`, *optional*, defaults to `"pil"`):
                The output format of the generated sample. If dict, it should specify the output type for each
                modality, e.g. `{"video": "np", "audio": "np"}`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~DiffusionPipelineOutput`] instead of a
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
        if isinstance(output_type, str):
            output_type = dict(visual=output_type)
        if image_size is not None:
            visual_scheduler = self.scheduler if video_duration == 1 else self.video_scheduler
            main_scheduler = visual_scheduler
        else:
            main_scheduler = self.audio_scheduler

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._guidance_scale_audio = guidance_scale_audio if guidance_scale_audio is not None else guidance_scale

        cfg_factor = 1 + self.do_classifier_free_guidance

        # Define call parameters
        device = self.model.device
        # device = self._execution_device

        model_kwargs = self.encode_prompt(model_kwargs)

        if image_size is not None:
            # Compute n_tokens for flux shift in scheduler
            vae_spatial_downsample_factor = self.vae_spatial_downsample_factor
            if isinstance(vae_spatial_downsample_factor, int):
                vae_spatial_downsample_factor = (vae_spatial_downsample_factor,) * 2
            vae_temporal_downsample_factor = self.vae_temporal_downsample_factor
            n_tokens = ((video_duration - 1) // vae_temporal_downsample_factor + 1) * \
                (image_size[0] // vae_spatial_downsample_factor[0]) * (image_size[1] // vae_spatial_downsample_factor[1])
            if model_kwargs.get("cond_vae_images") is not None:
                target_latent_channels = (
                    getattr(self.model.config, "img_latent_in_channels", None)
                    or getattr(self.args, "img_latent_in_channels", None)
                    or getattr(self.args, "vae_latent_dim", None)
                )
                model_kwargs["cond_vae_images"] = _pad_cond_vae_channels(
                    model_kwargs["cond_vae_images"], target_latent_channels=target_latent_channels
                )

            # Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                visual_scheduler, num_inference_steps, device, timesteps, sigmas, n_tokens=n_tokens,
            )
            main_timesteps = timesteps
        else:
            main_timesteps = None

        # Prepare audio latents
        if audio_duration is not None:
            audio_timesteps, _ = retrieve_timesteps(
                self.audio_scheduler, num_inference_steps, device, None, sigmas,
            )
            if main_timesteps is None:
                timesteps = audio_timesteps
                main_timesteps = audio_timesteps

            audio_generator = self.duplicate_generator(generator)
            audio_latents = self.prepare_audio_latents(
                batch_size=batch_size,
                audio_duration=audio_duration,
                dtype=torch.float32,
                device=device,
                generator=audio_generator,
                token_length=audio_token_length,
            )
        else:
            audio_timesteps = timesteps
            audio_latents = None

        if image_size is not None:
            # Get number of channel conditional images
            channel_cond_vae_images = model_kwargs.pop("channel_cond_vae_images")
            if channel_cond_vae_images is None:
                num_channel_cond_images = 0
            else:
                num_channel_cond_images = len(channel_cond_vae_images[0])

            # Prepare latent variables
            latents = self.prepare_latents(
                batch_size=batch_size,
                latent_channel=self.model.config.vae_latent_dim,
                video_duration=video_duration,
                height=image_size[0],
                width=image_size[1],
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
            )

            # Squeeze temporal dim for 2D projection (align with training data_provider_dit.py)
            if latents.ndim == 5 and latents.size(2) == 1 and self.model.config.img_proj_ndim == 2:
                latents = latents.squeeze(2)

            # Prepare channel conditional latents
            if self.args.extend_latent_channels:
                channel_cond_latents, channel_cond_mask, channel_task_type = self.prepare_channel_cond_latents(
                    channel_cond_vae_images,
                    latents,
                )
            else:
                channel_cond_latents, channel_cond_mask, channel_task_type = None, None, "t2v"

            # Prepare extra step kwargs.
            _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
                visual_scheduler.step, {"generator": generator}
            )
        else:
            _scheduler_step_extra_kwargs = {}

        if audio_latents is not None:
            _scheduler_step_extra_kwargs_audio = self.prepare_extra_func_kwargs(
                self.audio_scheduler.step, {"generator": generator}
            )
        else:
            _scheduler_step_extra_kwargs_audio = {}

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device)

        # Sampling loop
        num_warmup_steps = len(main_timesteps) - num_inference_steps * main_scheduler.order
        self._num_timesteps = len(main_timesteps)

        cache_context = getattr(self.model, "cache_context", None)
        cache_context = cache_context("denoise") if cache_context is not None else nullcontext()
        with cache_context, self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t, at) in enumerate(zip(timesteps, audio_timesteps)):

                # expand the channel
                if image_size is not None:
                    if self.args.extend_latent_channels:
                        latent_model_input = torch.cat([latents, channel_cond_latents, channel_cond_mask], dim=1)
                    else:
                        latent_model_input = latents
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent_model_input] * cfg_factor)
                    latent_model_input = visual_scheduler.scale_model_input(latent_model_input, t)
                    t_expand = t.repeat(latent_model_input.shape[0])
                else:
                    latent_model_input = None
                    t_expand = None

                if audio_latents is not None:
                    audio_latent_model_input = torch.cat([audio_latents] * cfg_factor)
                    audio_latent_model_input = self.audio_scheduler.scale_model_input(audio_latent_model_input, at)
                    audio_t_expand = at.repeat(audio_latent_model_input.shape[0])
                else:
                    audio_latent_model_input = None
                    audio_t_expand = None

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    latents=latent_model_input,
                    timesteps=t_expand,
                    audio_latents=audio_latent_model_input,
                    audio_timesteps=audio_t_expand,
                    **model_kwargs,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = self.model(**model_inputs)
                    pred = model_output.get("diffusion_prediction", None)
                    audio_pred = model_output.get("audio_diffusion_prediction", None)

                if pred is not None:
                    pred = pred.to(dtype=torch.float32)
                    if pred.ndim == 5 and pred.size(2) == 1 and latents.ndim == 4:
                        pred = pred.squeeze(2)

                if audio_pred is not None:
                    audio_pred = audio_pred.to(dtype=torch.float32)

                # perform guidance
                if self.do_classifier_free_guidance:
                    if pred is not None:
                        pred_cond, pred_uncond = pred.chunk(2)
                        pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)
                    if audio_pred is not None:
                        audio_pred_cond, audio_pred_uncond = audio_pred.chunk(2)
                        audio_pred = self.cfg_operator(audio_pred_cond, audio_pred_uncond, self.guidance_scale_audio, step=i)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    if pred is not None:
                        pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)
                    if audio_pred is not None:
                        audio_pred = rescale_noise_cfg(
                            audio_pred, audio_pred_cond, guidance_rescale=self.guidance_rescale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                if latents is not None and pred is not None:
                    latents = visual_scheduler.step(
                        pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False
                    )[0]
                if audio_latents is not None and audio_pred is not None:
                    audio_latents = self.audio_scheduler.step(
                        audio_pred, at, audio_latents, **_scheduler_step_extra_kwargs_audio, return_dict=False
                    )[0]

                if i != len(main_timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    # input_ids = None
                    # if input_ids.shape[1] != model_kwargs["input_pos"].shape[1]:
                    #     input_ids = torch.gather(input_ids, 1, index=model_kwargs["input_pos"])

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    audio_latents = callback_outputs.pop("audio_latents", audio_latents)

                # call the callback, if provided
                if i == len(main_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % main_scheduler.order == 0):
                    progress_bar.update()

        # ================================ Decode latents to Image/Video ================================
        if latents is not None:
            if channel_task_type == "fl2v":
                # For fl2v, we need to remove the last latent frame since it contains artifacts.
                latents = latents[:, :, :-1]

            latents = denormalize_vae_latents(self.vae, latents)

            if hasattr(self.vae, "ffactor_temporal") and latents.ndim == 4:
                latents = latents.unsqueeze(2)

            with torch.autocast(
                    device_type="cuda", dtype=self.vae_autocast_dtype,
                    enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32
            ):
                visuals = self.vae.decode(latents, return_dict=False, generator=generator)[0]

            # b c t h w
            if hasattr(self.vae, "ffactor_temporal") and visuals.shape[2] == 1:
                visuals = visuals.squeeze(2)

            if visuals.ndim == 4:
                do_denormalize = [True] * visuals.shape[0]
                visuals = self.video_processor.postprocess(
                    visuals, output_type=output_type['visual'], do_denormalize=do_denormalize
                )
            elif visuals.ndim == 5:
                visuals = self.video_processor.postprocess_video(visuals, output_type=output_type['visual'])
                if output_type['visual'] == "np":
                    visuals = (visuals * 255).round().astype("uint8")
            else:
                raise ValueError(f"Unexpected visuals shape {visuals.shape}, expected 4 or 5 dimensions.")
        else:
            visuals = None

        # ================================ Decode latents to Audio ================================
        if audio_latents is not None:
            audios = self.audio_vae.decode(audio_latents)
            audios = self.audio_processor.postprocess_audio(audios, output_type=output_type['audio'])
        else:
            audios = None

        if not return_dict:
            return visuals, audios

        output = Leo2PipelineOutput(visuals=visuals, audios=audios)
        if kwargs.get("return_latents", False):
            output.visual_latents = latents
            output.audio_latents = audio_latents
        return output

    @torch.no_grad()
    def generate_validation_loss(
            self,
            video_vae_out,
            audio_vae_out,
            model_kwargs: Dict[str, Any] = None,
            video_denoiser=None,
            audio_denoiser=None,
            **kwargs,
    ) -> Dict[str, Any]:

        # Define call parameters
        device = self.model.device
        # device = self._execution_device

        model_kwargs = self.encode_prompt(model_kwargs)

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device)

        latent_model_input = video_vae_out.x_t
        t_expand = video_vae_out.model_t
        audio_latent_model_input = audio_vae_out.x_t
        audio_t_expand = audio_vae_out.model_t

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            latents=latent_model_input,
            timesteps=t_expand,
            audio_latents=audio_latent_model_input,
            audio_timesteps=audio_t_expand,
            **model_kwargs,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            model_output = self.model(**model_inputs)
            pred = model_output.get("diffusion_prediction", None)
            audio_pred = model_output.get("audio_diffusion_prediction", None)

        # Compute the flow-matching validation loss the same way training does:
        # `denoiser.training_losses_fn(t, x0, xt, ut, model_output)["loss"]`, where the ground-truth velocity `u_t`
        # and the sampled `t / x_0 / x_t` come from the noise-adding step (VAEEncodeOutput). `mean_flat` returns a
        # per-sample loss; we average over the batch to obtain a scalar.
        def _flow_loss(denoiser, vae_out, model_pred):
            if denoiser is None or model_pred is None or vae_out is None or vae_out.u_t is None:
                return None
            loss = denoiser.training_losses_fn(
                t=vae_out.t,
                x0=vae_out.x_0,
                xt=vae_out.x_t,
                ut=vae_out.u_t,
                model_output=model_pred.to(vae_out.u_t.dtype),
            )["loss"]
            return loss.mean()

        video_loss = _flow_loss(video_denoiser, video_vae_out, pred)
        audio_loss = _flow_loss(audio_denoiser, audio_vae_out, audio_pred)

        return {
            "video_loss": video_loss,
            "audio_loss": audio_loss,
            "diffusion_prediction": pred,
            "audio_diffusion_prediction": audio_pred,
        }
