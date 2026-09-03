"""Backward-compatible import for the shared Transformers compatibility shim."""

from unirl.models.transformers_compat import install_transformers_flash_attention_compat

__all__ = ["install_transformers_flash_attention_compat"]
