"""Leo2 model package with its vendored gen-ar runtime."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Leo2Bundle": (".bundle", "Leo2Bundle"),
    "Leo2Conditions": (".conditions", "Leo2Conditions"),
    "Leo2PipelineConfig": (".config", "Leo2PipelineConfig"),
    "Leo2DiffusionStage": (".diffusion", "Leo2DiffusionStage"),
    "Leo2FlowSDEStrategy": (".sde", "Leo2FlowSDEStrategy"),
    "Leo2Pipeline": (".pipeline", "Leo2Pipeline"),
    "Leo2CondStage": (".text_embed", "Leo2CondStage"),
    "Leo2AudioDecodeStage": (".vae", "Leo2AudioDecodeStage"),
    "Leo2VideoDecodeStage": (".vae", "Leo2VideoDecodeStage"),
    "Leo2PreprocessingCache": (".preprocessing_cache", "Leo2PreprocessingCache"),
    "Leo2CachedSupervisedTrackBuilder": (".sft", "Leo2CachedSupervisedTrackBuilder"),
    "Leo2RolloutEngine": (".rollout", "Leo2RolloutEngine"),
}


def __getattr__(name: str) -> Any:
    """Load public Leo2 components only when requested."""
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, package=__name__), object_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
