"""Leo2 conditioning stage -- capture the native input prep, then embed.

Instead of re-implementing ``leo_hf.generate_video``'s input preparation
(template application, tokenization, media layout), we let the native code run
it and CAPTURE the kwargs at the diffusion-pipeline call boundary. That makes
this stage immune to drift in hymm's prep logic: whatever the verified
inference path feeds its denoising loop is exactly what the RL stage gets.

Flow per prompt (batch-1, mirrors pipeline_leo.__call__ up to its loop):
  1. swap ``model.diffusion_pipeline`` for a recorder, call ``generate_video``
     with guidance_scale=1.0 -> recorder raises after storing the call kwargs
  2. run ``pipeline.encode_prompt(model_kwargs)`` (Qwen3.5, frozen)
  3. pop input_ids, build attention_mask (pipeline_leo.py:810-815)
  4. stash {input_ids, model_kwargs, image_size, video_duration} on the
     conditions blob for the diffusion stage / replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .conditions import Leo2Conditions

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
                captured = self._capture(
                    prompt, height=height, width=width, num_frames=num_frames, seed=seed
                )
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
                    input_ids, self.bundle.model.generation_config, model_kwargs=model_kwargs,
                )
                model_kwargs["attention_mask"] = attention_mask.to(self.bundle.device)
                _report_foreign_types(model_kwargs)
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

        return Leo2Conditions(
            text=TextEmbedCondition(embeds=torch.cat([e[:1] for e in embeds], dim=0) if embeds else None),
            hymm=blobs,
        )


_FOREIGN_REPORTED = False


def _report_foreign_types(model_kwargs: Dict[str, Any]) -> None:
    """One-time diagnostic: model_kwargs entries whose class is neither torch nor
    builtin. Their pickles drag hymm onto the driver's / reward actors' sys.path
    (R13), so this lists what a pure-tensor blob would have to encode instead."""
    global _FOREIGN_REPORTED
    if _FOREIGN_REPORTED:
        return
    _FOREIGN_REPORTED = True

    def _cls(v):
        mod = type(v).__module__
        return None if mod == "builtins" or mod.startswith("torch") else f"{mod}.{type(v).__qualname__}"

    foreign = {}
    for k, v in model_kwargs.items():
        c = _cls(v)
        if c:
            foreign[k] = c
        elif isinstance(v, (list, tuple)):
            inner = {_cls(x) for x in v if _cls(x)}
            if inner:
                foreign[k] = f"{type(v).__name__}[{','.join(sorted(inner))}]"
        elif isinstance(v, dict):
            inner = {_cls(x) for x in v.values() if _cls(x)}
            if inner:
                foreign[k] = f"dict[{','.join(sorted(inner))}]"
    print(f"[leo2 cond] non-builtin model_kwargs types: {foreign}", flush=True)


def _slim(captured: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only small scalar call kwargs for debugging; drop tensors."""
    out = {}
    for k, v in captured.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


__all__ = ["Leo2CondStage"]
