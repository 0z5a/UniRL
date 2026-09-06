"""Build Leo2 prompt conditions through the native input-preparation boundary."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .conditions import LEO2_MODEL_KWARGS, Leo2Conditions
from .preprocessing_cache import Leo2PreprocessingCache, map_tensors

if TYPE_CHECKING:
    from .bundle import Leo2Bundle


class _CaptureDone(Exception):
    def __init__(self, kwargs: Dict[str, Any]):
        self.kwargs = kwargs


class _PipelineRecorder:
    """Stands in for ``model.diffusion_pipeline`` during input capture."""

    def __call__(self, **kwargs):
        raise _CaptureDone(kwargs)

    def __getattr__(self, name):  # tolerate attribute peeks before the call
        raise AttributeError(name)


class Leo2CondStage:
    """Prompt -> ready-to-step hymm conditioning blob."""

    def __init__(self, bundle: "Leo2Bundle") -> None:
        self.bundle = bundle
        self._cache_size = int(getattr(bundle.config, "condition_cache_size", 0))
        self._cache: OrderedDict[tuple[str, int, int, int, int], Leo2Conditions] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_cache = (
            Leo2PreprocessingCache(bundle.config)
            if getattr(bundle.config, "preprocessing_cache_mode", "off") == "readonly"
            else None
        )

    def _capture(self, prompt: str, *, height: int, width: int, num_frames: int, seed: int) -> Dict[str, Any]:
        model = self.bundle.model
        # build_diffusion_pipeline() guards on `_diffusion_pipeline is None`,
        # and `diffusion_pipeline` may be a read-only property -- swap the
        # private attribute so generate()'s internal rebuild keeps the recorder.
        real_pipeline = model._diffusion_pipeline
        model._diffusion_pipeline = _PipelineRecorder()
        try:
            model.generate_video(
                # prepare_model_inputs only supports message_list; this is the
                # exact shape video_prompt_dataset.default_prompt_fn produces.
                message_list=[[{"role": "user", "content": prompt}]],
                seed=[int(seed)],
                video_size=(int(height), int(width)),
                num_frames=int(num_frames),
                video_fps=24,
                bot_task="video",
                use_system_prompt=self.bundle.use_system_prompt,
                diff_guidance_scale=1.0,
                output_type={"visual": "latent"},
                verbose=0,
            )
        except _CaptureDone as done:
            return done.kwargs
        finally:
            model._diffusion_pipeline = real_pipeline
        raise RuntimeError(
            "Leo2CondStage: generate_video returned without calling diffusion_pipeline -- "
            "the capture boundary moved; check leo_hf.generate for the pipeline call site."
        )

    def build(self, texts: Texts, *, height: int, width: int, num_frames: int, seeds: List[int]) -> Leo2Conditions:
        prompts = list(texts.texts)
        require(
            len(prompts) > 0 and len(prompts) == len(seeds), "Leo2CondStage: prompts/seeds must be nonempty and aligned"
        )
        if self.disk_cache is not None:
            blobs = []
            for prompt in prompts:
                key = self.disk_cache.condition_key(prompt, height=height, width=width, num_frames=num_frames)
                blobs.extend(self.disk_cache.read_condition(key).hymm)
            self.cache_hits += len(prompts)
            return Leo2Conditions.from_dict({"hymm": blobs})
        if self._cache_size <= 0 or len(prompts) != 1 or len(seeds) != 1:
            return self._build_uncached(
                texts,
                height=height,
                width=width,
                num_frames=num_frames,
                seeds=seeds,
            )

        key = (str(prompts[0]), int(height), int(width), int(num_frames), int(seeds[0]))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return _clone_conditions(cached)

        result = self._build_uncached(
            texts,
            height=height,
            width=width,
            num_frames=num_frames,
            seeds=seeds,
        )
        self.cache_misses += 1
        self._cache[key] = Leo2Conditions.from_dict(map_tensors(result.to_dict(), "cpu"))
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    @torch.no_grad()
    def _build_uncached(
        self,
        texts: Texts,
        *,
        height: int,
        width: int,
        num_frames: int,
        seeds: List[int],
    ) -> Leo2Conditions:
        prompts: List[str] = list(texts.texts)
        require(len(prompts) > 0, "Leo2CondStage: no prompts")
        require(len(seeds) == len(prompts), f"Leo2CondStage: {len(prompts)} prompts vs {len(seeds)} seeds")

        pipeline = self.bundle.model.diffusion_pipeline
        blobs: List[Dict[str, Any]] = []
        embeds: List[torch.Tensor] = []
        with self.bundle.text_encoder_ctx():
            for prompt, seed in zip(prompts, seeds):
                captured = self._capture(prompt, height=height, width=width, num_frames=num_frames, seed=seed)
                model_kwargs = captured.get("model_kwargs")
                require(model_kwargs is not None, "Leo2CondStage: captured call carries no model_kwargs")
                image_size = captured.get("image_size")
                video_duration = captured.get("video_duration")
                if isinstance(image_size, (list, tuple)):
                    image_size = tuple(int(value) if isinstance(value, np.integer) else value for value in image_size)
                if isinstance(video_duration, np.integer):
                    video_duration = int(video_duration)
                require(
                    isinstance(image_size, (list, tuple))
                    and len(image_size) == 2
                    and all(type(value) is int and value > 0 for value in image_size),
                    "Leo2CondStage: captured image_size must contain two positive integers; "
                    f"requested={(height, width)!r}, received={image_size!r}",
                )
                require(
                    type(video_duration) is int and video_duration > 0,
                    "Leo2CondStage: captured video_duration must be a positive integer; "
                    f"requested={num_frames!r}, received={video_duration!r}",
                )

                # __call__ never runs (recorder aborts it), so set the guidance
                # attributes its prelude would have set before using the pipeline.
                pipeline._guidance_scale = 1.0
                pipeline._guidance_rescale = 0.0
                pipeline._guidance_scale_audio = 1.0
                # Mirrors pipeline_leo.__call__: encode_prompt mutates/extends
                # model_kwargs (adds cond_text_states etc.), then input_ids are
                # popped and an attention mask derived.
                model_kwargs = pipeline.encode_prompt(model_kwargs)
                # pipeline_leo.py:810-814 verbatim
                input_ids = model_kwargs.pop("input_ids")
                attention_mask = self.bundle.model._prepare_attention_mask_for_generation(  # noqa
                    input_ids,
                    self.bundle.model.generation_config,
                    model_kwargs=model_kwargs,
                )
                model_kwargs["attention_mask"] = attention_mask.to(self.bundle.device)
                model_kwargs = _to_transport_tree(
                    {key: model_kwargs[key] for key in LEO2_MODEL_KWARGS if key in model_kwargs},
                    path="model_kwargs",
                )
                blobs.append(
                    dict(
                        input_ids=input_ids,
                        model_kwargs=model_kwargs,
                        # hymm snaps requests to its supported media buckets.
                        # Persist that effective geometry so x_T matches the
                        # visual token mask instead of the unsnapped request.
                        image_size=(int(image_size[0]), int(image_size[1])),
                        video_duration=int(video_duration),
                        captured_call=_slim(captured),
                    )
                )
                text_states = model_kwargs.get("cond_text_states")
                embeds.append(
                    text_states.detach().to("cpu", copy=True)
                    if isinstance(text_states, torch.Tensor)
                    else torch.zeros(1, 1, 1)
                )

        return Leo2Conditions.from_dict(
            {
                "text": TextEmbedCondition(embeds=torch.cat([e[:1] for e in embeds], dim=0) if embeds else None),
                "hymm": blobs,
            }
        )


def _clone_conditions(conditions: Leo2Conditions) -> Leo2Conditions:
    """Clone mutable containers while sharing immutable frozen-condition tensors."""
    text = conditions.text
    cloned_text = (
        None
        if text is None
        else TextEmbedCondition(
            embeds=text.embeds,
            pooled=text.pooled,
            attn_mask=text.attn_mask,
        )
    )
    return Leo2Conditions(
        text=cloned_text,
        hymm=_clone_transport_tree(conditions.hymm),
    )


def _clone_transport_tree(value: Any) -> Any:
    """Copy builtin containers without duplicating tensor storage."""
    if isinstance(value, torch.Tensor) or value is None or type(value) in (bool, int, float, str, slice):
        return value
    if isinstance(value, list):
        return [_clone_transport_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_transport_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_transport_tree(item) for key, item in value.items()}
    raise TypeError(
        "Leo2 condition cache expected Tensor/builtin transport data, "
        f"got {type(value).__module__}.{type(value).__qualname__}: {value!r}"
    )


def _to_transport_tree(value: Any, *, path: str) -> Any:
    """Copy a value tree into Tensor/builtin-only transport form."""
    if isinstance(value, torch.Tensor) or value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, np.generic):
        # hymm stores numpy integer bounds inside rope_media_info slices.
        # Normalize them so condition blobs remain tensor/builtin-only.
        return value.item()
    if isinstance(value, slice):
        return slice(
            _to_transport_tree(value.start, path=f"{path}.start"),
            _to_transport_tree(value.stop, path=f"{path}.stop"),
            _to_transport_tree(value.step, path=f"{path}.step"),
        )
    if isinstance(value, list):
        return [_to_transport_tree(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_to_transport_tree(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            out[key] = _to_transport_tree(item, path=f"{path}.{key}")
        return out
    raise TypeError(
        f"{path} contains non-transportable {type(value).__module__}.{type(value).__qualname__}; "
        "flatten it to Tensor/builtin values before crossing Ray actors"
    )


def _slim(captured: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only small scalar call kwargs for debugging; drop tensors."""
    out = {}
    for k, v in captured.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


__all__ = ["Leo2CondStage"]
