"""Typed Leo2 diffusion conditions with dependency-free transport payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition

LEO2_MODEL_KWARGS = (
    "attention_mask",
    "rope_media_info",
    "cond_vae_images",
    "cond_text_states",
    "cond_text_mask",
    "visual_mask",
    "text_mask",
    "timesteps_index",
    "audio_mask",
    "cond_vae_mask",
    "cond_timesteps",
    "cond_text_scatter_mask",
    "und_token_indices",
    "gen_token_indices",
    "audio_token_indices",
)
LEO2_REQUIRED_MODEL_KWARGS = frozenset(
    {
        "attention_mask",
        "rope_media_info",
        "cond_text_states",
        "cond_text_mask",
        "visual_mask",
        "text_mask",
        "timesteps_index",
        "audio_mask",
        "und_token_indices",
        "gen_token_indices",
        "audio_token_indices",
    }
)
_REQUIRED_TENSOR_MODEL_KWARGS = frozenset(
    {
        "attention_mask",
        "cond_text_states",
        "cond_text_mask",
        "visual_mask",
        "text_mask",
        "timesteps_index",
        "und_token_indices",
        "gen_token_indices",
    }
)
_NULLABLE_TENSOR_MODEL_KWARGS = frozenset(
    {
        "audio_mask",
        "audio_token_indices",
        "cond_vae_images",
        "cond_vae_mask",
        "cond_timesteps",
        "cond_text_scatter_mask",
    }
)


@dataclass
class Leo2Conditions(Batch):
    """Conditions passed to the Leo2 diffusion stage."""

    # Human-readable / logging-friendly view of the text conditioning.
    text: Optional[TextEmbedCondition] = concat_field(default=None)
    # Per-sample Tensor/builtin blobs: {"input_ids": Tensor, "model_kwargs": dict}.
    hymm: Optional[List[Dict[str, Any]]] = concat_field(default=None)

    def __post_init__(self) -> None:
        _validate_conditions(self.text, self.hymm)

    @classmethod
    def from_dict(cls, d: dict) -> "Leo2Conditions":
        if not isinstance(d, dict):
            raise TypeError(f"Leo2Conditions.from_dict expects dict, got {type(d).__name__}")
        invalid_keys = [key for key in d if not isinstance(key, str)]
        if invalid_keys:
            raise TypeError(f"Leo2Conditions.from_dict keys must be strings, got {invalid_keys[:8]}")
        unknown = sorted(set(d) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Leo2Conditions.from_dict received unknown fields: {unknown}")
        if "hymm" not in d or d["hymm"] is None:
            raise ValueError("Leo2Conditions.from_dict requires non-null field 'hymm'")
        return cls(**d)

    def to_dict(self) -> dict:
        _validate_conditions(self.text, self.hymm)
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


def _validate_conditions(
    text: Optional[TextEmbedCondition],
    hymm: Optional[List[Dict[str, Any]]],
) -> None:
    if text is not None:
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(f"Leo2Conditions.text must be TextEmbedCondition or None, got {type(text).__name__}")
        for name in ("embeds", "pooled", "attn_mask"):
            value = getattr(text, name)
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"Leo2Conditions.text.{name} must be Tensor or None, got {type(value).__name__}")
    if hymm is None:
        raise ValueError("Leo2Conditions.hymm is required")
    if not isinstance(hymm, list):
        raise TypeError(f"Leo2Conditions.hymm must be a list, got {type(hymm).__name__}")
    if not hymm:
        raise ValueError("Leo2Conditions.hymm must contain at least one sample blob")
    for index, blob in enumerate(hymm):
        _validate_hymm_blob(blob, path=f"hymm[{index}]")


def _validate_hymm_blob(blob: Any, *, path: str) -> None:
    if not isinstance(blob, dict):
        raise TypeError(f"Leo2Conditions.{path} must be a dict, got {type(blob).__name__}")
    invalid_keys = [key for key in blob if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError(f"Leo2Conditions.{path} keys must be strings, got {invalid_keys[:8]}")
    required = {"input_ids", "model_kwargs", "image_size", "video_duration"}
    allowed = required | {"captured_call", "_device"}
    missing = sorted(required - set(blob))
    unknown = sorted(set(blob) - allowed)
    if missing or unknown:
        raise ValueError(f"Leo2Conditions.{path} fields mismatch: missing={missing}, unknown={unknown}")
    if not isinstance(blob["input_ids"], torch.Tensor):
        raise TypeError(f"Leo2Conditions.{path}.input_ids must be Tensor, got {type(blob['input_ids']).__name__}")
    if not isinstance(blob["model_kwargs"], dict):
        raise TypeError(f"Leo2Conditions.{path}.model_kwargs must be a dict, got {type(blob['model_kwargs']).__name__}")
    model_kwargs = blob["model_kwargs"]
    invalid_model_keys = [key for key in model_kwargs if not isinstance(key, str)]
    if invalid_model_keys:
        raise TypeError(f"Leo2Conditions.{path}.model_kwargs keys must be strings, got {invalid_model_keys[:8]}")
    missing_model_keys = sorted(LEO2_REQUIRED_MODEL_KWARGS - set(model_kwargs))
    unknown_model_keys = sorted(set(model_kwargs) - set(LEO2_MODEL_KWARGS))
    if missing_model_keys or unknown_model_keys:
        raise ValueError(
            f"Leo2Conditions.{path}.model_kwargs fields mismatch: "
            f"missing={missing_model_keys}, unknown={unknown_model_keys}"
        )
    for key in _REQUIRED_TENSOR_MODEL_KWARGS:
        if not isinstance(model_kwargs[key], torch.Tensor):
            raise TypeError(
                f"Leo2Conditions.{path}.model_kwargs.{key} must be Tensor, got {type(model_kwargs[key]).__name__}"
            )
    if type(model_kwargs["rope_media_info"]) is not list:
        raise TypeError(
            f"Leo2Conditions.{path}.model_kwargs.rope_media_info must be a list, "
            f"got {type(model_kwargs['rope_media_info']).__name__}"
        )
    for key in _NULLABLE_TENSOR_MODEL_KWARGS & set(model_kwargs):
        value = model_kwargs[key]
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Leo2Conditions.{path}.model_kwargs.{key} must be Tensor or None, got {type(value).__name__}"
            )
    image_size = blob["image_size"]
    if (
        not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError(f"Leo2Conditions.{path}.image_size must contain two positive integers")
    if type(blob["video_duration"]) is not int or blob["video_duration"] <= 0:
        raise ValueError(f"Leo2Conditions.{path}.video_duration must be a positive integer")
    if "_device" in blob and not isinstance(blob["_device"], str):
        raise TypeError(f"Leo2Conditions.{path}._device must be a string")
    _validate_transport_tree(blob, path=f"Leo2Conditions.{path}")


def _validate_transport_tree(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor) or value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, slice):
        for name in ("start", "stop", "step"):
            _validate_transport_tree(getattr(value, name), path=f"{path}.{name}")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_transport_tree(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            _validate_transport_tree(item, path=f"{path}.{key}")
        return
    raise TypeError(
        f"{path} contains non-transportable {type(value).__module__}.{type(value).__qualname__}; "
        "Leo2 conditions may contain only Tensor and builtin values"
    )


__all__ = ["LEO2_MODEL_KWARGS", "LEO2_REQUIRED_MODEL_KWARGS", "Leo2Conditions"]
