"""Leo2 t2v pipeline -- prompt -> video via the UniRL Sample protocol."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from .bundle import Leo2Bundle
from .config import (
    LEO2_AUDIO_LATENT_CHANNELS,
    LEO2_AUDIO_SAMPLE_RATE,
    LEO2_VAE_LATENT_CHANNELS,
    LEO2_VAE_SPATIAL,
    LEO2_VAE_TEMPORAL,
    Leo2PipelineConfig,
)
from .diffusion import Leo2DiffusionStage
from .text_embed import Leo2CondStage
from .vae import Leo2AudioDecodeStage, Leo2VideoDecodeStage


class Leo2Pipeline(Pipeline):
    """Text -> video through a sequence-packed denoising loop."""

    def __init__(
        self,
        *,
        bundle: Leo2Bundle,
        cond_stage: Leo2CondStage,
        diffusion: Leo2DiffusionStage,
        video_decode: Leo2VideoDecodeStage,
        audio_decode: Leo2AudioDecodeStage | None,
        config: Leo2PipelineConfig,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.cond_stage = cond_stage
        self.diffusion = diffusion
        self.video_decode = video_decode
        self.audio_decode = audio_decode
        self.config = config

    @classmethod
    def from_config(cls, config: Leo2PipelineConfig, strategy: StepStrategy) -> "Leo2Pipeline":
        return cls.from_bundle(Leo2Bundle.from_config(config), config=config, strategy=strategy)

    @classmethod
    def from_bundle(
        cls,
        bundle: Leo2Bundle,
        *,
        config: Leo2PipelineConfig,
        strategy: StepStrategy,
    ) -> "Leo2Pipeline":
        return cls(
            bundle=bundle,
            cond_stage=Leo2CondStage(bundle),
            diffusion=Leo2DiffusionStage(
                bundle,
                strategy,
                autocast_precision=config.autocast_precision,
                trajectory_precision=config.trajectory_precision,
                logprob_precision=config.logprob_precision,
                profile_forward=config.profile_forward,
                enable_audio=config.enable_audio,
                video_shift=config.video_shift,
                audio_shift=config.audio_shift,
                audio_joint_sde=config.audio_joint_sde,
            ),
            video_decode=Leo2VideoDecodeStage(bundle),
            audio_decode=Leo2AudioDecodeStage(bundle) if config.enable_audio else None,
            config=config,
        )

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        return FlowMatchSchedulePolicy.static_only(shift=float(self.config.video_shift))

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Per-sample UNPACKED video latent shape for the driver x_T recipe."""
        del model_config
        frames = int(sampling_spec.num_frames)
        return (
            LEO2_VAE_LATENT_CHANNELS,
            (frames - 1) // LEO2_VAE_TEMPORAL + 1,
            int(sampling_spec.height) // LEO2_VAE_SPATIAL,
            int(sampling_spec.width) // LEO2_VAE_SPATIAL,
        )

    @classmethod
    def _latent_shape_from_conditions(cls, conditions: Any) -> Tuple[int, ...]:
        """Return the VAE shape for hymm's effective (possibly snapped) media bucket."""
        blobs = getattr(conditions, "hymm", None)
        require(isinstance(blobs, list) and blobs, "Leo2Pipeline: conditions.hymm must be a non-empty list")
        geometries = {(tuple(blob["image_size"]), int(blob["video_duration"])) for blob in blobs}
        require(
            len(geometries) == 1,
            f"Leo2Pipeline: one forward requires one effective media geometry, got {sorted(geometries)!r}",
        )
        (height, width), frames = next(iter(geometries))
        require(
            type(height) is int
            and type(width) is int
            and height > 0
            and width > 0
            and height % LEO2_VAE_SPATIAL == 0
            and width % LEO2_VAE_SPATIAL == 0,
            "Leo2Pipeline: effective image_size must contain positive dimensions divisible by "
            f"{LEO2_VAE_SPATIAL}, got {(height, width)!r}",
        )
        require(
            frames > 0 and (frames - 1) % LEO2_VAE_TEMPORAL == 0,
            "Leo2Pipeline: effective video_duration must satisfy "
            f"(frames - 1) % {LEO2_VAE_TEMPORAL} == 0, got {frames!r}",
        )
        return (
            LEO2_VAE_LATENT_CHANNELS,
            (frames - 1) // LEO2_VAE_TEMPORAL + 1,
            height // LEO2_VAE_SPATIAL,
            width // LEO2_VAE_SPATIAL,
        )

    def _debug_hymm_sample(self, prompt: str, params, seed: int) -> None:
        """Dump a native hymm sample when ``LEO2_DEBUG_HYMM_SAMPLE=1``."""
        from .vae import _maybe_dump_frames

        model = self.bundle.model
        try:
            with torch.no_grad(), self.bundle.text_encoder_ctx():
                out = model.generate_video(
                    message_list=[[{"role": "user", "content": prompt}]],
                    seed=[int(seed)],
                    video_size=(int(params.height), int(params.width)),
                    num_frames=int(params.num_frames),
                    video_fps=24,
                    bot_task="av" if self.config.enable_audio else "video",
                    use_system_prompt=self.bundle.use_system_prompt,
                    diff_guidance_scale=1.0,
                    diff_infer_steps=int(params.num_inference_steps),
                    output_type={"visual": "latent", "audio": "latent"},
                    verbose=0,
                )
            latents = out.videos if hasattr(out, "videos") else out
            if isinstance(latents, (list, tuple)):
                latents = latents[0]
            require(isinstance(latents, torch.Tensor), f"hymm A/B: unexpected output type {type(latents)}")
            print(
                f"[leo2 hymm-ab] latents {tuple(latents.shape)} dtype={latents.dtype} std={latents.float().std():.3f}",
                flush=True,
            )
            visuals = self.video_decode.decode_to_tensor(latents)
            _maybe_dump_frames(visuals, tag="hymm")
        except Exception as exc:
            print(f"[leo2 hymm-ab] failed: {exc!r}", flush=True)

    def generate(self, sample: Sample) -> Sample:
        gen = sample.parts[-1]
        params = gen.sampling_params
        require(params is not None, "Leo2Pipeline.generate: generation Part carries no sampling params")
        require(
            params.sigmas is not None,
            "Leo2Pipeline.generate: params.sigmas is None -- the hosting engine pins the schedule first.",
        )

        conditioning = list(sample.conditioning())
        texts = next((c for c in conditioning if isinstance(c, Texts)), None)
        require(texts is not None, "Leo2Pipeline.generate: no text prompt in the sample conditioning")

        base_seed = int(params.seed) if params.seed is not None else 0
        conditions = self.cond_stage.build(
            texts,
            height=int(params.height),
            width=int(params.width),
            num_frames=int(params.num_frames),
            seeds=[base_seed] * len(list(texts.texts)),
        )

        if os.environ.get("LEO2_DEBUG_HYMM_SAMPLE") and not getattr(self, "_hymm_ab_done", False):
            self._hymm_ab_done = True
            self._debug_hymm_sample(str(list(texts.texts)[0]), params, base_seed)

        recipe = NoiseRecipe.from_sample(sample)
        shape = self._latent_shape_from_conditions(conditions)
        video_noise = recipe.resolve(device=self.bundle.device, latent_shape=shape)
        require(
            video_noise is not None,
            "Leo2Pipeline.generate: no initial latents from the driver x_T recipe.",
        )
        audio_noise = None
        if self.config.enable_audio:
            audio_token_lengths = [blob.get("audio_token_length") for blob in conditions.hymm]
            require(
                all(type(length) is int and length > 0 for length in audio_token_lengths)
                and len(set(audio_token_lengths)) == 1,
                "Leo2Pipeline.generate: packed AV conditions require one shared positive "
                f"audio_token_length, got {audio_token_lengths}",
            )
            audio_token_length = audio_token_lengths[0]
            audio_noise = recipe.resolve(
                device=self.bundle.device,
                salt="audio",
                latent_shape=(LEO2_AUDIO_LATENT_CHANNELS, audio_token_length),
            )
            require(
                audio_noise is not None,
                "Leo2Pipeline.generate: no initial audio latents from the driver noise recipe",
            )

        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=params.sigmas.to(self.bundle.device),
            initial_latents=video_noise,
            initial_audio_latents=audio_noise,
            sde_indices=list(params.sde_indices) if params.sde_indices is not None else None,
            denoise_seed_keys=[str(sample_id) for sample_id in gen.sample_ids],
            denoise_base_seed=base_seed,
        )

        final_latents = segment.latents_at(int(params.num_inference_steps))
        videos = self.video_decode.decode(final_latents)
        primitives = {"video": videos}
        primitive_metadata = {}
        if self.config.enable_audio:
            require(
                self.audio_decode is not None and segment.aux_latents is not None,
                "Leo2Pipeline.generate: AV rollout produced no audio decoder/trajectory",
            )
            final_audio_latents = segment.aux_latents_at(int(params.num_inference_steps))
            primitives["audio"] = self.audio_decode.decode(final_audio_latents)
            primitive_metadata["audio"] = {"sample_rate": LEO2_AUDIO_SAMPLE_RATE}

        filled = gen.fill(
            segment=segment,
            primitives=primitives,
            primitive_metadata=primitive_metadata,
            conditions=conditions.to_dict(),
        )
        forward_pack = tuple(gen.sample_ids)
        filled = dataclasses.replace(
            filled,
            forward_pack_sample_ids=[forward_pack] * gen.batch_size,
        )
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["Leo2Pipeline"]
