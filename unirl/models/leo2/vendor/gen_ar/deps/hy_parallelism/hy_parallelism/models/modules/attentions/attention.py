"""Unified attention dispatch.

Provides:
- attention(): single entry point for all attention backends (flex, sdpa, magi, flash)
- convert_dense_mask(): convert dense bool mask to the format expected by a given backend

Usage:
    from hy_parallelism.models.modules.attentions.attention import attention, convert_dense_mask

    # At model.forward entry — convert mask once:
    attention_mask = convert_dense_mask(attention_mask, mode=attn_mode)

    # In each attention layer — dispatch to backend:
    y = attention(q, k, v, mask, mode=attn_mode, scale=scale)
"""

import torch
from torch.nn.attention.flex_attention import BlockMask


def attention(q, k, v, mask, *, mode, scale=None, **kwargs):
    """Unified attention dispatch.

    Args:
        q, k, v: (B, H, S, D) — all backends use BHSD input layout
        mask: format depends on mode
            - "flex": BlockMask
            - "sdpa": dense bool/float tensor (B, 1, S, S) or None
            - "magi": tuple (q_ranges, k_ranges, attn_type_map)
            - "flash"/"flash_packed"/"flash3"/"flash3_packed": pad mask (B, S)
        mode: attention backend to use
        scale: softmax scale, defaults to 1/sqrt(D) inside each backend
        **kwargs: backend-specific options
            - deterministic (bool): for flash backends, default False

    Returns:
        (B, S, H, D) — all backends return BSHD output layout
    """
    if mode == "flex":
        from .flex import flex_attention
        q = q.to(dtype=v.dtype)
        k = k.to(dtype=v.dtype)
        return flex_attention(q, k, v, block_mask=mask, scale=scale).transpose(1, 2)

    if mode == "sdpa":
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
            is_causal=mask is None and q.size(2) > 1,
        ).transpose(1, 2)

    if mode == "magi":
        from .magi import magi_scaled_dot_product_attention
        return magi_scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale,
        ).transpose(1, 2)

    if mode in ("flash", "flash_packed", "flash3", "flash3_packed"):
        if "3" in mode:
            from .flash import flash_attn_no_pad_v3 as flash_fn
        else:
            from .flash import flash_attn_no_pad as flash_fn
        qkv = torch.stack([
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        ], dim=2)
        return flash_fn(
            qkv, mask, causal=False, dropout_p=0.0, softmax_scale=scale,
            deterministic=kwargs.get("deterministic", False),
            packed="packed" in mode,
        )

    raise NotImplementedError(f"Unsupported attention mode: {mode}")


def convert_dense_mask(dense_mask, *, mode):
    """Convert dense bool mask to the format expected by the given attention mode.

    Call once at model.forward entry. If mask is already in the target format
    (BlockMask, tuple, etc.), returns it unchanged.

    Args:
        dense_mask: bool tensor (B, 1, S, S), or None, or already-converted mask
        mode: target attention backend

    Returns:
        Converted mask, or input unchanged for sdpa/flash/None
    """
    if dense_mask is None:
        return None
    if isinstance(dense_mask, (BlockMask, tuple)):
        return dense_mask

    if mode == "flex":
        from .flex import dense_binary_mask_to_block_mask
        return dense_binary_mask_to_block_mask(dense_mask)

    if mode == "magi":
        from .magi import dense_mask_to_magi_ranges
        return dense_mask_to_magi_ranges(dense_mask)

    return dense_mask
