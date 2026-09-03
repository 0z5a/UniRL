"""Build Leo2 prompt conditions through the native input-preparation boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .conditions import LEO2_MODEL_KWARGS, Leo2Conditions

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

    @torch.no_grad()
    def build(self, texts: Texts, *, height: int, width: int, num_frames: int, seeds: List[int]) -> Leo2Conditions:
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
                        image_size=(int(height), int(width)),
                        video_duration=int(num_frames),
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


def _to_transport_tree(value: Any, *, path: str) -> Any:
    """Copy a value tree into Tensor/builtin-only transport form."""
    if isinstance(value, torch.Tensor) or value is None or type(value) in (bool, int, float, str):
        return value
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
