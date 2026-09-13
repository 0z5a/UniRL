"""Leo2 t2v pipeline -- prompt -> video via the UniRL Sample protocol."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor

from unirl.config.require import require
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sample_id import branch_of
from unirl.utils.profiling import emit_phase_profile, profile_region, tensor_tree_nbytes

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
                audio_stochastic_rollout=config.audio_stochastic_rollout,
                audio_joint_sde=config.audio_joint_sde,
                store_sde_means=config.store_sde_means,
                store_initial_latents=config.store_initial_latents,
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

    @staticmethod
    def _native_seeds(sample: Sample, *, base_seed: int, same_noise: bool) -> list[int]:
        """Reproduce Hymm's per-row seed expansion."""
        frontier = sample.parts[-1]
        metadata = sample.root_metadata(-1)
        seeds = []
        for sample_id, row in zip(frontier.sample_ids, metadata):
            seed = int((row or {}).get("seed", base_seed))
            if not same_noise:
                seed += int(branch_of(sample_id) or 0)
            seeds.append(seed)
        return seeds

    @classmethod
    def _native_generators(cls, sample: Sample, *, base_seed: int, same_noise: bool, device: torch.device):
        """Build Hymm-compatible per-row CUDA generators."""
        generators = [
            torch.Generator(device=device).manual_seed(seed)
            for seed in cls._native_seeds(sample, base_seed=base_seed, same_noise=same_noise)
        ]
        return generators[0] if len(generators) == 1 else generators

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
        native_seeds = (
            self._native_seeds(sample, base_seed=base_seed, same_noise=bool(params.init_same_noise))
            if self.config.native_rng_compat
            else [base_seed] * len(list(texts.texts))
        )
        with profile_region("leo2.rollout.condition_build", batch_size=gen.batch_size):
            conditions = self.cond_stage.build(
                texts,
                height=int(params.height),
                width=int(params.width),
                num_frames=int(params.num_frames),
                seeds=native_seeds,
            )
        emit_phase_profile("leo2.rollout.conditions", tensor_bytes=tensor_tree_nbytes(conditions))

        if os.environ.get("LEO2_DEBUG_HYMM_SAMPLE") and not getattr(self, "_hymm_ab_done", False):
            self._hymm_ab_done = True
            self._debug_hymm_sample(str(list(texts.texts)[0]), params, base_seed)

        with profile_region("leo2.rollout.noise_init", batch_size=gen.batch_size):
            recipe = NoiseRecipe.from_sample(sample)
            shape = self._latent_shape_from_conditions(conditions)
            native_generators = None
            if self.config.native_rng_compat and recipe.initial_latents is None:
                native_generators = self._native_generators(
                    sample,
                    base_seed=base_seed,
                    same_noise=bool(params.init_same_noise),
                    device=self.bundle.device,
                )
                audio_generators = (
                    [generator.clone_state() for generator in native_generators]
                    if isinstance(native_generators, list)
                    else native_generators.clone_state()
                )
                video_noise = randn_tensor(
                    (gen.batch_size, *shape),
                    generator=native_generators,
                    device=self.bundle.device,
                    dtype=torch.float32,
                )
            else:
                audio_generators = None
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
                if audio_generators is not None:
                    audio_noise = randn_tensor(
                        (gen.batch_size, LEO2_AUDIO_LATENT_CHANNELS, audio_token_length),
                        generator=audio_generators,
                        device=self.bundle.device,
                        dtype=torch.float32,
                    )
                else:
                    audio_noise = recipe.resolve(
                        device=self.bundle.device,
                        salt="audio",
                        latent_shape=(LEO2_AUDIO_LATENT_CHANNELS, audio_token_length),
                    )
                require(
                    audio_noise is not None,
                    "Leo2Pipeline.generate: no initial audio latents from the driver noise recipe",
                )

        with profile_region("leo2.rollout.denoise", batch_size=gen.batch_size):
            segment = self.diffusion.generate(
                conditions,
                params=params,
                sigmas=params.sigmas.to(dtype=torch.float32, device="cpu"),
                initial_latents=video_noise,
                initial_audio_latents=audio_noise,
                sde_indices=list(params.sde_indices) if params.sde_indices is not None else None,
                denoise_seed_keys=(
                    None
                    if self.config.native_rng_compat
                    else [str(sample_id) for sample_id in gen.sample_ids]
                ),
                denoise_base_seed=base_seed,
            )
        emit_phase_profile(
            "leo2.rollout.trajectory",
            tensor_bytes=tensor_tree_nbytes(segment),
            latents_bytes=tensor_tree_nbytes(segment.latents),
            initial_latents_bytes=tensor_tree_nbytes(segment.initial_latents),
            sde_logp_bytes=tensor_tree_nbytes(segment.sde_logp),
            sde_means_bytes=tensor_tree_nbytes(segment.sde_means),
            aux_latents_bytes=tensor_tree_nbytes(segment.aux_latents),
        )

        final_latents = segment.latents_at(int(params.num_inference_steps))
        with profile_region("leo2.rollout.video_decode", batch_size=gen.batch_size):
            videos = self.video_decode.decode(final_latents)
        primitives = {"video": videos}
        primitive_metadata = {}
        if self.config.enable_audio:
            require(
                self.audio_decode is not None and segment.aux_latents is not None,
                "Leo2Pipeline.generate: AV rollout produced no audio decoder/trajectory",
            )
            final_audio_latents = segment.aux_latents_at(int(params.num_inference_steps))
            with profile_region("leo2.rollout.audio_decode", batch_size=gen.batch_size):
                primitives["audio"] = self.audio_decode.decode(final_audio_latents)
            primitive_metadata["audio"] = {"sample_rate": LEO2_AUDIO_SAMPLE_RATE}
        emit_phase_profile("leo2.rollout.decoded", tensor_bytes=tensor_tree_nbytes(primitives))

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
