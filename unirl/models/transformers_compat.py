"""Shared compatibility fixes for supported Transformers releases."""

from __future__ import annotations

import functools
import importlib.util
import logging
import sys
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

from packaging.version import Version

logger = logging.getLogger(__name__)

_AFFECTED_TRANSFORMERS_RELEASE = (5, 6, 0)
_FLASH_ATTENTION_BACKENDS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
_PATCH_MARKER = "_unirl_optional_s_aux_compat"


class _NullAttentionSink:
    """Adapt ``None`` to the Transformers 5.6.0 ``s_aux.to(...)`` call."""

    def to(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_NULL_ATTENTION_SINK = _NullAttentionSink()


def _needs_optional_s_aux_compat(transformers_version: str) -> bool:
    return Version(transformers_version).release == _AFFECTED_TRANSFORMERS_RELEASE


def _wrap_optional_s_aux(attention_forward: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve ``s_aux=None`` through the Transformers 5.6.0 FA dispatcher."""
    if getattr(attention_forward, _PATCH_MARKER, False):
        return attention_forward

    @functools.wraps(attention_forward)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("s_aux") is None:
            kwargs["s_aux"] = _NULL_ATTENTION_SINK
        return attention_forward(*args, **kwargs)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch_attention_registry(registry: Any) -> tuple[str, ...]:
    patched: list[str] = []
    for backend in _FLASH_ATTENTION_BACKENDS:
        try:
            attention_forward = registry[backend]
        except KeyError:
            continue
        wrapped = _wrap_optional_s_aux(attention_forward)
        if wrapped is attention_forward:
            continue
        registry.register(backend, wrapped)
        patched.append(backend)
    return tuple(patched)


def _register_flash_attn_3_distribution_mapping(import_utils: Any) -> bool:
    """Register source-built FA3 in Transformers' import-name mapping."""
    try:
        module_available = importlib.util.find_spec("flash_attn_interface") is not None
    except ImportError:
        module_available = False
    except ValueError:
        module_available = "flash_attn_interface" in sys.modules
    if not module_available:
        return False

    mapping = import_utils.PACKAGE_DISTRIBUTION_MAPPING
    distributions = list(mapping.get("flash_attn_interface", ()))
    if "flash-attn-3" in {name.replace("_", "-") for name in distributions}:
        return False
    distributions.append("flash-attn-3")
    mapping["flash_attn_interface"] = distributions
    availability = getattr(import_utils, "is_flash_attn_3_available", None)
    if callable(getattr(availability, "cache_clear", None)):
        availability.cache_clear()
    return True


def install_transformers_flash_attention_compat() -> tuple[str, ...]:
    """Install the Transformers 5.6.0 optional-attention-sink fix once per process."""
    if not _needs_optional_s_aux_compat(version("transformers")):
        return ()

    from transformers.utils import import_utils

    if _register_flash_attn_3_distribution_mapping(import_utils):
        logger.info("Registered source-built flash_attn_interface as flash-attn-3 for Transformers")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    patched = _patch_attention_registry(ALL_ATTENTION_FUNCTIONS)
    if patched:
        logger.info(
            "Installed Transformers 5.6.0 optional s_aux compatibility for: %s",
            ", ".join(patched),
        )
    return patched


__all__ = ["install_transformers_flash_attention_compat"]
