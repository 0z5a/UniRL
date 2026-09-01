from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from typing import Optional, Tuple, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import logging
from hymm.ar.pipelines.pipeline_leo import Leo2Pipeline, Leo2PipelineOutput, retrieve_timesteps, rescale_noise_cfg
from hymm.models.autoencoders import denormalize_vae_latents
from hymm.models.diffusion.leo import LeoOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class Leo2GRPOPipelineOutput(Leo2PipelineOutput):
    # Visual-side per-step quantities.
    all_latents: Optional[List[torch.Tensor]] = None
    # Per-modality log probs are returned SEPARATELY (not pre-summed): the caller decides whether audio
    # participates in the GRPO objective. ``all_log_probs`` is the visual log prob, ``all_log_probs_audio``
    # the audio one. A joint objective is simply ``all_log_probs + all_log_probs_audio``.
    all_log_probs: Optional[List[torch.Tensor]] = None
    all_prev_means: Optional[List[torch.Tensor]] = None
    all_std_devs: Optional[List[torch.Tensor]] = None
    # Audio-side per-step quantities, stored separately because they are modality-specific tensors
    # (different shape from the visual ones) and therefore cannot be summed with the visual versions.
    all_latents_audio: Optional[List[torch.Tensor]] = None
    all_log_probs_audio: Optional[List[torch.Tensor]] = None
    all_prev_means_audio: Optional[List[torch.Tensor]] = None
    all_std_devs_audio: Optional[List[torch.Tensor]] = None


class Leo2GRPOPipeline(Leo2Pipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion with GRPO modifications.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    This pipeline inherits from Leo2Pipeline and provides custom modifications
    for GRPO (Group Relative Policy Optimization) training.
    - provides a single step denoise function for GRPO training
    - calculate the log probs for GRPO training

    The pipeline supports visual and/or audio modalities:
    - visual (image / video) when ``image_size`` is provided,
    - audio when ``audio_duration`` is provided.
    When both are present, ``sde_step_with_logprob`` is run separately for each modality and each modality's
    per-step log prob is returned SEPARATELY (``all_log_probs`` for visual, ``all_log_probs_audio`` for audio);
    they are NOT pre-summed, so the caller decides whether audio participates in the objective (a joint
    objective is ``all_log_probs + all_log_probs_audio``). ``prev_mean`` / ``std_dev`` / ``latents`` are
    modality-specific tensors (different shapes), so the audio versions are tracked / returned separately too.

    Args:
        model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

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
            audio_scheduler: SchedulerMixin = None,
            audio_vae=None,
            audio_processor=None,
    ):
        super().__init__(
            model=model,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            vae_autocast_dtype=vae_autocast_dtype,
            progress_bar_config=progress_bar_config,
            args=args,
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            audio_vae=audio_vae,
            audio_processor=audio_processor,
        )
        from hymm.core.global_vars import get_logger
        self.logger = get_logger()

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int,
            image_size: List[int] | None,
            video_duration: int = 1,
            audio_duration: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
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
            sde_type: str = "dance_grpo",
            eta: float = 0.0,
            determistic: Union[bool, List[bool]] = False,  # Used in GRPO for progressive training (note: typo kept for consistency)
            **kwargs,
    ) -> Union[Leo2GRPOPipelineOutput, tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image_size (`Tuple[int]` or `List[int]` or `None`):
                The size (height, width) of the generated image/video. If ``None``, the visual modality is disabled
                and ``audio_duration`` must be provided (audio-only). When both ``image_size`` and ``audio_duration``
                are provided, visual and audio are sampled jointly and their per-step log probs are returned
                separately (``all_log_probs`` / ``all_log_probs_audio``).
            video_duration (`int`, *optional*, defaults to 1):
                The duration of the generated video in frames.
            audio_duration (`int`, *optional*, defaults to None):
                The duration of the generated audio in (video frames / fps * sample_rate). If ``None``, the audio
                modality is disabled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the condition.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) used by the SDE step with log-prob.

        Returns:
            [`Leo2GRPOPipelineOutput`] or `tuple`.
        """

        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)
        if isinstance(output_type, str):
            output_type = dict(visual=output_type)

        if image_size is None and audio_duration is None:
            raise ValueError("At least one of `image_size` or `audio_duration` must be provided for GRPO sampling.")

        # Select the main modality scheduler that drives the loop length / progress bar:
        # visual when image_size is provided, otherwise audio.
        if image_size is not None:
            visual_scheduler = self.scheduler if video_duration == 1 else self.video_scheduler
            main_scheduler = visual_scheduler
        else:
            visual_scheduler = None
            main_scheduler = self.audio_scheduler

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        # Define call parameters
        device = self.model.device

        model_kwargs = self.encode_prompt(model_kwargs)

        if image_size is not None:
            # Compute n_tokens for flux shift in scheduler
            vae_spatial_downsample_factor = self.vae_spatial_downsample_factor
            if isinstance(vae_spatial_downsample_factor, int):
                vae_spatial_downsample_factor = (vae_spatial_downsample_factor,) * 2
            vae_temporal_downsample_factor = self.vae_temporal_downsample_factor
            n_tokens = ((video_duration - 1) // vae_temporal_downsample_factor + 1) * \
                (image_size[0] // vae_spatial_downsample_factor[0]) * (image_size[1] // vae_spatial_downsample_factor[1])

            # Prepare visual timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                visual_scheduler, num_inference_steps, device, timesteps, sigmas, n_tokens=n_tokens,
            )
            main_timesteps = timesteps
        else:
            main_timesteps = None

        # Prepare audio timesteps / latents
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
        else:
            latents = None
            channel_cond_latents, channel_cond_mask, channel_task_type = None, None, "t2v"

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device).to(dtype=torch.long)

        # Sampling loop
        num_warmup_steps = len(main_timesteps) - num_inference_steps * main_scheduler.order
        self._num_timesteps = len(main_timesteps)

        # Visual-side trackers
        all_latents = [latents] if latents is not None else None
        all_log_probs = [] if latents is not None else None
        all_prev_means = [] if latents is not None else None
        all_std_devs = [] if latents is not None else None
        # Audio-side trackers (kept separate since the tensors are modality-specific and the per-step
        # log probs are NOT pre-summed with the visual ones; the caller decides whether to combine them)
        all_latents_audio = [audio_latents] if audio_latents is not None else None
        all_log_probs_audio = [] if audio_latents is not None else None
        all_prev_means_audio = [] if audio_latents is not None else None
        all_std_devs_audio = [] if audio_latents is not None else None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t, at) in enumerate(zip(timesteps, audio_timesteps)):
                # Handle determistic parameter: can be a list (per-timestep) or a single bool
                if isinstance(determistic, list):
                    assert len(determistic) == num_inference_steps, \
                        f"determistic list length ({len(determistic)}) must match num_inference_steps ({num_inference_steps})"
                    determistic_i = determistic[i]
                else:
                    determistic_i = determistic

                pred, audio_pred, model_output = self.denoise_step(
                    latents,
                    t,
                    i,
                    model_kwargs=model_kwargs,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    channel_cond_latents=channel_cond_latents,
                    channel_cond_mask=channel_cond_mask,
                    audio_latents=audio_latents,
                    audio_timestep=at,
                    return_audio_pred=True,
                )

                # compute the previous noisy sample x_t -> x_t-1 separately per modality and record each
                # modality's per-step log prob separately (do NOT sum here; the caller combines if desired).
                if latents is not None and pred is not None:
                    latents, _pred_orig_v, log_prob_v, prev_mean_v, std_dev_v = visual_scheduler.sde_step_with_logprob(
                        pred, latents, t, eta=eta, prev_sample=None, grpo=True, sde_type=sde_type, determistic=determistic_i
                    )
                    all_latents.append(latents)
                    all_log_probs.append(log_prob_v)
                    all_prev_means.append(prev_mean_v)
                    all_std_devs.append(std_dev_v)

                if audio_latents is not None and audio_pred is not None:
                    audio_latents, _pred_orig_a, log_prob_a, prev_mean_a, std_dev_a = self.audio_scheduler.sde_step_with_logprob(
                        audio_pred, audio_latents, at, eta=eta, prev_sample=None, grpo=True, sde_type=sde_type, determistic=determistic_i
                    )
                    all_latents_audio.append(audio_latents)
                    all_log_probs_audio.append(log_prob_a)
                    all_prev_means_audio.append(prev_mean_a)
                    all_std_devs_audio.append(std_dev_a)

                if i != len(main_timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    input_ids = None

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

        if all_latents is not None:
            all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
            all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)
            all_prev_means = torch.stack(all_prev_means, dim=1)  # (batch_size, num_steps, ...)
            all_std_devs = torch.stack(all_std_devs, dim=0)  # (num_steps, ...)
        if all_latents_audio is not None:
            all_latents_audio = torch.stack(all_latents_audio, dim=1)  # (batch_size, num_steps + 1, ...)
            all_log_probs_audio = torch.stack(all_log_probs_audio, dim=1)  # (batch_size, num_steps)
            all_prev_means_audio = torch.stack(all_prev_means_audio, dim=1)  # (batch_size, num_steps, ...)
            all_std_devs_audio = torch.stack(all_std_devs_audio, dim=0)  # (num_steps, ...)

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
            return (visuals,)

        return Leo2GRPOPipelineOutput(
            visuals=visuals,
            audios=audios,
            all_latents=all_latents,
            all_log_probs=all_log_probs,
            all_prev_means=all_prev_means,
            all_std_devs=all_std_devs,
            all_latents_audio=all_latents_audio,
            all_log_probs_audio=all_log_probs_audio,
            all_prev_means_audio=all_prev_means_audio,
            all_std_devs_audio=all_std_devs_audio,
        )

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        step: int,
        model_kwargs: Dict[str, Any] = None,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        channel_cond_latents: torch.Tensor = None,
        channel_cond_mask: torch.Tensor = None,
        audio_latents: torch.Tensor = None,
        audio_timestep: torch.Tensor = None,
        return_audio_pred: bool = False,
    ) -> Tuple[torch.Tensor, LeoOutput | tuple]:
        """
        Perform a single denoising step in the diffusion process.

        By default returns ``(main_pred, model_output)`` where ``main_pred`` is the visual prediction when
        ``latents`` is provided, otherwise the audio prediction. This keeps a stable 2-tuple contract relied
        on by the GRPO trainer.

        When ``return_audio_pred=True`` returns ``(pred, audio_pred, model_output)`` so callers can run
        ``sde_step_with_logprob`` separately for each modality (used by the dual-modality ``__call__``).
        Either ``pred`` or ``audio_pred`` may be ``None`` if the corresponding modality is absent.
        """
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        cfg_factor = 1 + self.do_classifier_free_guidance

        # ---- visual branch ----
        if latents is not None:
            # expand the channel
            if self.args.extend_latent_channels:
                if channel_cond_latents is None or channel_cond_mask is None:
                    assert "channel_cond_vae_images" in model_kwargs, \
                        "channel_cond_vae_images must be provided in model_kwargs when extend_latent_channels is True"
                    channel_cond_latents, channel_cond_mask, _ = self.prepare_channel_cond_latents(
                        model_kwargs["channel_cond_vae_images"],
                        latents,
                    )
                latent_model_input = torch.cat([latents, channel_cond_latents, channel_cond_mask], dim=1)
            else:
                latent_model_input = latents
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latent_model_input] * cfg_factor)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

            # NOTE: timestep can be either a scalar (from pipeline.__call__) or a 1D tensor
            # of shape [batch_size] (from grpo_one_step in GRPO training when
            # mini_batch_size_per_rollout > 1). torch.Tensor.repeat(N) on a 1D tensor of
            # length L returns length L*N, not N, so we need to dispatch on the two cases.
            if timestep.dim() == 0 or (timestep.dim() == 1 and timestep.shape[0] == 1):
                t_expand = timestep.repeat(latent_model_input.shape[0])
            else:
                # Batched timestep [B] -> tile by cfg_factor: [t0,..,tB-1, t0,..,tB-1]
                # so it aligns with latent_model_input which is cat([latents] * cfg_factor).
                t_expand = timestep.repeat(cfg_factor)
        else:
            latent_model_input = None
            t_expand = None

        # ---- audio branch ----
        if audio_latents is not None:
            audio_latent_model_input = torch.cat([audio_latents] * cfg_factor)
            audio_latent_model_input = self.audio_scheduler.scale_model_input(audio_latent_model_input, audio_timestep)
            if audio_timestep.dim() == 0 or (audio_timestep.dim() == 1 and audio_timestep.shape[0] == 1):
                audio_t_expand = audio_timestep.repeat(audio_latent_model_input.shape[0])
            else:
                audio_t_expand = audio_timestep.repeat(cfg_factor)
        else:
            audio_latent_model_input = None
            audio_t_expand = None

        model_inputs = self.model.prepare_inputs_for_generation(
            None,
            latents=latent_model_input,
            timesteps=t_expand,
            audio_latents=audio_latent_model_input,
            audio_timesteps=audio_t_expand,
            **model_kwargs,
        )

        # GRPO needs the raw prediction (with grad in the update path) but no diffusion
        # loss. The model only enters its loss branch in train() mode, where
        # diffusion_loss_fn must be callable; inject zero-cost no-op loss fns so forward
        # still returns diffusion_prediction without intruding into leo.py. The returned
        # `losses` are ignored here. NOTE: this relies on use_repa=False; if REPA is ever
        # enabled, get_aux_losses would dereference the (unprovided) repa_feats and crash.
        def _noop_loss_fn(model_output, **_):
            ref = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
            return {"loss": ref.new_zeros(1)}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            model_output = self.model(
                **model_inputs,
                diffusion_loss_fn=_noop_loss_fn,
                audio_diffusion_loss_fn=_noop_loss_fn,
            )
            pred = model_output.get("diffusion_prediction", None)
            audio_pred = model_output.get("audio_diffusion_prediction", None)

        if pred is not None:
            pred = pred.to(dtype=torch.float32)
            if pred.ndim == 5 and pred.size(2) == 1 and latents is not None and latents.ndim == 4:
                pred = pred.squeeze(2)
        if audio_pred is not None:
            audio_pred = audio_pred.to(dtype=torch.float32)

        # perform guidance
        if self.do_classifier_free_guidance:
            if pred is not None:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=step)
            if audio_pred is not None:
                audio_pred_cond, audio_pred_uncond = audio_pred.chunk(2)
                audio_pred = self.cfg_operator(audio_pred_cond, audio_pred_uncond, self.guidance_scale, step=step)

        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            if pred is not None:
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)
            if audio_pred is not None:
                audio_pred = rescale_noise_cfg(audio_pred, audio_pred_cond, guidance_rescale=self.guidance_rescale)

        if return_audio_pred:
            return pred, audio_pred, model_output

        # Return the main-modality prediction (visual if available, else audio).
        main_pred = pred if pred is not None else audio_pred
        return main_pred, model_output


class Leo2ReFLPipeline(Leo2GRPOPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion with ReFL modifications.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    This pipeline inherits from Leo2GRPOPipeline and runs a truncated sampling loop (``sample_step`` steps),
    returning the stacked main-modality latents for ReFL training. It supports joint visual + audio sampling, as
    well as audio-only sampling when ``image_size`` is ``None``.

    Args:
        model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents.
    """

    # @torch.no_grad()
    # def __call__(
    #         self,
    #         batch_size: int,
    #         image_size: List[int] | None,
    #         video_duration: int = 1,
    #         audio_duration: Optional[int] = None,
    #         num_inference_steps: int = 50,
    #         timesteps: List[int] = None,
    #         sigmas: List[float] = None,
    #         guidance_scale: float = 7.5,
    #         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #         latents: Optional[torch.Tensor] = None,
    #         prompt_embeds: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         negative_prompt_embeds: Optional[torch.Tensor] = None,
    #         negative_attention_mask: Optional[torch.Tensor] = None,
    #         output_type: Optional[str | dict] = "pil",
    #         return_dict: bool = True,
    #         guidance_rescale: float = 0.0,
    #         callback_on_step_end: Optional[
    #             Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    #         ] = None,
    #         callback_on_step_end_tensor_inputs: List[str] = ["latents"],    # noqa
    #         model_kwargs: Dict[str, Any] = None,
    #         sample_step: Optional[int] = None,
    #         **kwargs,
    # ) -> Union[Leo2PipelineOutput, tuple]:
    #     r"""
    #     The call function to the pipeline for ReFL generation.

    #     Args:
    #         image_size (`Tuple[int]` or `List[int]` or `None`):
    #             The size (height, width) of the generated image/video. If ``None``, the pipeline runs in audio-only
    #             mode and ``audio_duration`` must be provided.
    #         audio_duration (`int`, *optional*, defaults to None):
    #             The duration of the generated audio in (video frames / fps * sample_rate).
    #         sample_step (`int`, *optional*):
    #             The number of denoising steps to actually run (truncated sampling for ReFL).

    #     Returns:
    #         `Tuple[torch.Tensor, Optional[torch.Tensor]]`: ``(all_latents, all_latents_audio)`` where
    #         ``all_latents`` is the stacked main-modality latents of shape (batch_size, num_steps + 1, ...),
    #         and ``all_latents_audio`` is the stacked audio latents (same step layout) for joint visual+audio
    #         runs, or ``None`` otherwise.
    #     """

    #     callback_steps = kwargs.pop("callback_steps", None)
    #     pbar_steps = kwargs.pop("pbar_steps", None)
    #     if isinstance(output_type, str):
    #         output_type = dict(visual=output_type)

    #     if image_size is not None:
    #         visual_scheduler = self.scheduler if video_duration == 1 else self.video_scheduler
    #         main_scheduler = visual_scheduler
    #     else:
    #         visual_scheduler = None
    #         main_scheduler = self.audio_scheduler

    #     if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
    #         callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    #     self._guidance_scale = guidance_scale
    #     self._guidance_rescale = guidance_rescale

    #     cfg_factor = 1 + self.do_classifier_free_guidance

    #     # Define call parameters
    #     device = self.model.device

    #     model_kwargs = self.encode_prompt(model_kwargs)

    #     if image_size is not None:
    #         # Compute n_tokens for flux shift in scheduler
    #         vae_spatial_downsample_factor = self.vae_spatial_downsample_factor
    #         if isinstance(vae_spatial_downsample_factor, int):
    #             vae_spatial_downsample_factor = (vae_spatial_downsample_factor,) * 2
    #         vae_temporal_downsample_factor = self.vae_temporal_downsample_factor
    #         n_tokens = ((video_duration - 1) // vae_temporal_downsample_factor + 1) * \
    #             (image_size[0] // vae_spatial_downsample_factor[0]) * (image_size[1] // vae_spatial_downsample_factor[1])

    #         # Prepare timesteps
    #         timesteps, num_inference_steps = retrieve_timesteps(
    #             visual_scheduler, num_inference_steps, device, timesteps, sigmas, n_tokens=n_tokens,
    #         )
    #         main_timesteps = timesteps
    #     else:
    #         main_timesteps = None

    #     # Prepare audio latents
    #     if audio_duration is not None:
    #         audio_timesteps, _ = retrieve_timesteps(
    #             self.audio_scheduler, num_inference_steps, device, None, sigmas,
    #         )
    #         if main_timesteps is None:
    #             timesteps = audio_timesteps
    #             main_timesteps = audio_timesteps

    #         audio_generator = self.duplicate_generator(generator)
    #         audio_latents = self.prepare_audio_latents(
    #             batch_size=batch_size,
    #             audio_duration=audio_duration,
    #             dtype=torch.float32,
    #             device=device,
    #             generator=audio_generator,
    #         )
    #     else:
    #         audio_timesteps = timesteps
    #         audio_latents = None

    #     if image_size is not None:
    #         # Get number of channel conditional images
    #         channel_cond_vae_images = model_kwargs.pop("channel_cond_vae_images")
    #         if channel_cond_vae_images is None:
    #             num_channel_cond_images = 0
    #         else:
    #             num_channel_cond_images = len(channel_cond_vae_images[0])

    #         # Prepare latent variables
    #         latents = self.prepare_latents(
    #             batch_size=batch_size,
    #             latent_channel=self.model.config.vae_latent_dim,
    #             video_duration=video_duration,
    #             height=image_size[0],
    #             width=image_size[1],
    #             dtype=torch.float32,
    #             device=device,
    #             generator=generator,
    #             latents=latents,
    #         )

    #         # Squeeze temporal dim for 2D projection (align with training data_provider_dit.py)
    #         if latents.ndim == 5 and latents.size(2) == 1 and self.model.config.img_proj_ndim == 2:
    #             latents = latents.squeeze(2)

    #         # Prepare channel conditional latents
    #         if self.args.extend_latent_channels:
    #             channel_cond_latents, channel_cond_mask, channel_task_type = self.prepare_channel_cond_latents(
    #                 channel_cond_vae_images,
    #                 latents,
    #             )
    #         else:
    #             channel_cond_latents, channel_cond_mask, channel_task_type = None, None, "t2v"

    #         # Prepare extra step kwargs.
    #         _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
    #             visual_scheduler.step, {"generator": generator}
    #         )
    #     else:
    #         latents = None
    #         channel_cond_latents, channel_cond_mask, channel_task_type = None, None, "t2v"
    #         _scheduler_step_extra_kwargs = {}

    #     if audio_latents is not None:
    #         _scheduler_step_extra_kwargs_audio = self.prepare_extra_func_kwargs(
    #             self.audio_scheduler.step, {"generator": generator}
    #         )
    #     else:
    #         _scheduler_step_extra_kwargs_audio = {}

    #     # Prepare model kwargs
    #     input_ids = model_kwargs.pop("input_ids")
    #     attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
    #         input_ids, self.model.generation_config, model_kwargs=model_kwargs,
    #     )
    #     model_kwargs["attention_mask"] = attention_mask.to(device)

    #     # Sampling loop
    #     num_warmup_steps = len(main_timesteps) - num_inference_steps * main_scheduler.order
    #     self._num_timesteps = len(main_timesteps)

    #     # The main modality latents tracked for ReFL.
    #     main_latents = latents if image_size is not None else audio_latents
    #     all_latents = [main_latents]
    #     # Track audio latents separately for ReFL reuse ONLY when running joint visual+audio
    #     # (image_size present). In audio-only mode the audio track already lives in all_latents,
    #     # so there is no separate audio track to expose.
    #     track_audio = image_size is not None and audio_latents is not None
    #     all_latents_audio = [audio_latents] if track_audio else None
    #     timesteps = timesteps[:sample_step]
    #     audio_timesteps = audio_timesteps[:sample_step]
    #     num_inference_steps = sample_step

    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, (t, at) in enumerate(zip(timesteps, audio_timesteps)):
    #             # expand the channel
    #             if image_size is not None:
    #                 if self.args.extend_latent_channels:
    #                     latent_model_input = torch.cat([latents, channel_cond_latents, channel_cond_mask], dim=1)
    #                 else:
    #                     latent_model_input = latents
    #                 # expand the latents if we are doing classifier free guidance
    #                 latent_model_input = torch.cat([latent_model_input] * cfg_factor)
    #                 latent_model_input = visual_scheduler.scale_model_input(latent_model_input, t)
    #                 t_expand = t.repeat(latent_model_input.shape[0])
    #             else:
    #                 latent_model_input = None
    #                 t_expand = None

    #             if audio_latents is not None:
    #                 audio_latent_model_input = torch.cat([audio_latents] * cfg_factor)
    #                 audio_latent_model_input = self.audio_scheduler.scale_model_input(audio_latent_model_input, at)
    #                 audio_t_expand = at.repeat(audio_latent_model_input.shape[0])
    #             else:
    #                 audio_latent_model_input = None
    #                 audio_t_expand = None

    #             # Scheme B: pass input_ids=None so leo.py forward takes the compact (packed)
    #             model_inputs = self.model.prepare_inputs_for_generation(
    #                 None,
    #                 latents=latent_model_input,
    #                 timesteps=t_expand,
    #                 audio_latents=audio_latent_model_input,
    #                 audio_timesteps=audio_t_expand,
    #                 **model_kwargs,
    #             )

    #             # The policy model is in train() mode during the ReFL rollout (forward_step keeps it
    #             # trainable), so leo.py forward does NOT take the early inference return and instead
    #             # enters its loss branch, where diffusion_loss_fn must be callable. Inject zero-cost
    #             # no-op loss fns so the forward still returns diffusion_prediction without computing a
    #             # real loss (the returned losses are ignored here). Mirrors denoise_step's handling.
    #             def _noop_loss_fn(model_output, **_):
    #                 ref = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
    #                 return {"loss": ref.new_zeros(1)}

    #             with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    #                 model_output = self.model(
    #                     **model_inputs,
    #                     diffusion_loss_fn=_noop_loss_fn,
    #                     audio_diffusion_loss_fn=_noop_loss_fn,
    #                 )
    #                 pred = model_output.get("diffusion_prediction", None)
    #                 audio_pred = model_output.get("audio_diffusion_prediction", None)

    #             if pred is not None:
    #                 pred = pred.to(dtype=torch.float32)
    #                 if pred.ndim == 5 and pred.size(2) == 1 and latents.ndim == 4:
    #                     pred = pred.squeeze(2)

    #             if audio_pred is not None:
    #                 audio_pred = audio_pred.to(dtype=torch.float32)

    #             # perform guidance
    #             if self.do_classifier_free_guidance:
    #                 if pred is not None:
    #                     pred_cond, pred_uncond = pred.chunk(2)
    #                     pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)
    #                 if audio_pred is not None:
    #                     audio_pred_cond, audio_pred_uncond = audio_pred.chunk(2)
    #                     audio_pred = self.cfg_operator(audio_pred_cond, audio_pred_uncond, self.guidance_scale, step=i)

    #             if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
    #                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    #                 if pred is not None:
    #                     pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)
    #                 if audio_pred is not None:
    #                     audio_pred = rescale_noise_cfg(
    #                         audio_pred, audio_pred_cond, guidance_rescale=self.guidance_rescale
    #                     )

    #             # compute the previous noisy sample x_t -> x_t-1
    #             if latents is not None and pred is not None:
    #                 latents = visual_scheduler.step(
    #                     pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False
    #                 )[0]
    #             if audio_latents is not None and audio_pred is not None:
    #                 audio_latents = self.audio_scheduler.step(
    #                     audio_pred, at, audio_latents, **_scheduler_step_extra_kwargs_audio, return_dict=False
    #                 )[0]

    #             main_latents = latents if image_size is not None else audio_latents
    #             all_latents.append(main_latents)
    #             if all_latents_audio is not None:
    #                 all_latents_audio.append(audio_latents)

    #             if i != len(timesteps) - 1:
    #                 model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
    #                     model_output,
    #                     model_kwargs,
    #                 )

    #             if callback_on_step_end is not None:
    #                 callback_kwargs = {}
    #                 for k in callback_on_step_end_tensor_inputs:
    #                     callback_kwargs[k] = locals()[k]
    #                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

    #                 latents = callback_outputs.pop("latents", latents)
    #                 audio_latents = callback_outputs.pop("audio_latents", audio_latents)

    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % main_scheduler.order == 0):
    #                 progress_bar.update()

    #     all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
    #     if all_latents_audio is not None:
    #         all_latents_audio = torch.stack(all_latents_audio, dim=1)  # (batch_size, num_steps + 1, ...)

    #     # Return both tracks so ReFL training can reuse the rollout audio latents for the JOINT
    #     # av forward (audio is a frozen input; only the visual reward is back-propagated).
    #     # ``all_latents_audio`` is None for non-av (visual-only) runs.
    #     return all_latents, all_latents_audio

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int,
            image_size: List[int] | None,
            video_duration: int = 1,
            audio_duration: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
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
            sde_type: str = "dance_grpo",
            eta: float = 0.0,
            determistic: Union[bool, List[bool]] = True,  # Used in GRPO for progressive training (note: typo kept for consistency)
            sample_step: Optional[int] = None,
            **kwargs,
    ) -> Union[Leo2GRPOPipelineOutput, tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image_size (`Tuple[int]` or `List[int]` or `None`):
                The size (height, width) of the generated image/video. If ``None``, the visual modality is disabled
                and ``audio_duration`` must be provided (audio-only). When both ``image_size`` and ``audio_duration``
                are provided, visual and audio are sampled jointly and their per-step log probs are returned
                separately (``all_log_probs`` / ``all_log_probs_audio``).
            video_duration (`int`, *optional*, defaults to 1):
                The duration of the generated video in frames.
            audio_duration (`int`, *optional*, defaults to None):
                The duration of the generated audio in (video frames / fps * sample_rate). If ``None``, the audio
                modality is disabled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the condition.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) used by the SDE step with log-prob.

        Returns:
            [`Leo2GRPOPipelineOutput`] or `tuple`.
        """

        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)
        if isinstance(output_type, str):
            output_type = dict(visual=output_type)

        if image_size is None and audio_duration is None:
            raise ValueError("At least one of `image_size` or `audio_duration` must be provided for GRPO sampling.")

        # Select the main modality scheduler that drives the loop length / progress bar:
        # visual when image_size is provided, otherwise audio.
        if image_size is not None:
            visual_scheduler = self.scheduler if video_duration == 1 else self.video_scheduler
            main_scheduler = visual_scheduler
        else:
            visual_scheduler = None
            main_scheduler = self.audio_scheduler

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        # Define call parameters
        device = self.model.device

        model_kwargs = self.encode_prompt(model_kwargs)

        if image_size is not None:
            # Compute n_tokens for flux shift in scheduler
            vae_spatial_downsample_factor = self.vae_spatial_downsample_factor
            if isinstance(vae_spatial_downsample_factor, int):
                vae_spatial_downsample_factor = (vae_spatial_downsample_factor,) * 2
            vae_temporal_downsample_factor = self.vae_temporal_downsample_factor
            n_tokens = ((video_duration - 1) // vae_temporal_downsample_factor + 1) * \
                (image_size[0] // vae_spatial_downsample_factor[0]) * (image_size[1] // vae_spatial_downsample_factor[1])

            # Prepare visual timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                visual_scheduler, num_inference_steps, device, timesteps, sigmas, n_tokens=n_tokens,
            )
            main_timesteps = timesteps
        else:
            main_timesteps = None

        # Prepare audio timesteps / latents
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
        else:
            latents = None
            channel_cond_latents, channel_cond_mask, channel_task_type = None, None, "t2v"

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(device).to(dtype=torch.long)

        # Sampling loop
        num_warmup_steps = len(main_timesteps) - num_inference_steps * main_scheduler.order
        self._num_timesteps = len(main_timesteps)

        # Visual-side trackers
        all_latents = [latents] if latents is not None else None

        # Audio-side trackers (kept separate since the tensors are modality-specific and the per-step
        all_latents_audio = [audio_latents] if audio_latents is not None else None

        timesteps = timesteps[:sample_step]
        audio_timesteps = audio_timesteps[:sample_step]
        num_inference_steps = sample_step

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t, at) in enumerate(zip(timesteps, audio_timesteps)):
                # Handle determistic parameter: can be a list (per-timestep) or a single bool
                if isinstance(determistic, list):
                    assert len(determistic) == num_inference_steps, \
                        f"determistic list length ({len(determistic)}) must match num_inference_steps ({num_inference_steps})"
                    determistic_i = determistic[i]
                else:
                    determistic_i = determistic

                pred, audio_pred, model_output = self.denoise_step(
                    latents,
                    t,
                    i,
                    model_kwargs=model_kwargs,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    channel_cond_latents=channel_cond_latents,
                    channel_cond_mask=channel_cond_mask,
                    audio_latents=audio_latents,
                    audio_timestep=at,
                    return_audio_pred=True,
                )

                # compute the previous noisy sample x_t -> x_t-1 separately per modality and record each
                # modality's per-step log prob separately (do NOT sum here; the caller combines if desired).
                if latents is not None and pred is not None:
                    latents, _pred_orig_v, log_prob_v, prev_mean_v, std_dev_v = visual_scheduler.sde_step_with_logprob(
                        pred, latents, t, eta=eta, prev_sample=None, grpo=True, sde_type=sde_type, determistic=determistic_i
                    )
                    all_latents.append(latents)

                if audio_latents is not None and audio_pred is not None:
                    audio_latents, _pred_orig_a, log_prob_a, prev_mean_a, std_dev_a = self.audio_scheduler.sde_step_with_logprob(
                        audio_pred, audio_latents, at, eta=eta, prev_sample=None, grpo=True, sde_type=sde_type, determistic=determistic_i
                    )
                    all_latents_audio.append(audio_latents)

                if i != len(main_timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    input_ids = None

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

        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, ...)
        if all_latents_audio is not None:
            all_latents_audio = torch.stack(all_latents_audio, dim=1)  # (batch_size, num_steps + 1, ...)

        # Return both tracks so ReFL training can reuse the rollout audio latents for the JOINT
        # av forward (audio is a frozen input; only the visual reward is back-propagated).
        # ``all_latents_audio`` is None for non-av (visual-only) runs.
        return all_latents, all_latents_audio