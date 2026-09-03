import functools
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _create_sparse_block_from_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    flex_attention as _flex_attention,
    noop_mask,
)

_compiled_flex_by_grad: dict[bool, Any] = {}


@functools.wraps(_flex_attention)
def flex_attention(*args, **kwargs):
    grad_enabled = torch.is_grad_enabled()
    with torch._dynamo.utils.disable_cache_limit():
        if grad_enabled not in _compiled_flex_by_grad:
            _compiled_flex_by_grad[grad_enabled] = torch.compile(_flex_attention, dynamic=True)
        return _compiled_flex_by_grad[grad_enabled](*args, **kwargs)


def _normalize_dense_binary_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim == 2:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    elif attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.ndim != 4:
        raise ValueError(
            f"Expected attention mask with 2/3/4 dims [B, H?, Q, KV], "
            f"got shape {tuple(attention_mask.shape)}"
        )

    mask = attention_mask
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        if mask.numel() > 0:
            mask_min = int(mask.min())
            mask_max = int(mask.max())
            assert 0 <= mask_min <= 1 and 0 <= mask_max <= 1, (
                f"Integer attention mask must contain only 0 and 1, "
                f"got min={mask_min}, max={mask_max}, shape={tuple(mask.shape)}"
            )
        return mask != 0
    raise TypeError(
        f"Dense attention mask for flex BlockMask conversion must be bool or "
        f"integer 0/1, got dtype={mask.dtype}"
    )


def _convert_mask_to_block_mask_bool_reduce(
    mask: torch.Tensor,
    Q_BLOCK_SIZE: int,
    KV_BLOCK_SIZE: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 避免 mask.sum() 嘅 int reduction 帶嚟嘅高顯存
    assert mask.dtype == torch.bool
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0

    block_mask = mask.view(
        B,
        H,
        Q // Q_BLOCK_SIZE,
        Q_BLOCK_SIZE,
        KV // KV_BLOCK_SIZE,
        KV_BLOCK_SIZE,
    ).permute(0, 1, 2, 4, 3, 5)
    block_any = block_mask.any(dim=(-2, -1))
    full_blocks = block_mask.all(dim=(-2, -1))
    partial_blocks = block_any & ~full_blocks
    return partial_blocks.to(torch.int8), full_blocks.to(torch.int8)


class _DenseMaskMod:

    __slots__ = ("dense_mask", "broadcast_b", "broadcast_h")

    def __init__(self, dense_mask: torch.Tensor) -> None:
        self.broadcast_b = dense_mask.size(0) == 1
        self.broadcast_h = dense_mask.size(1) == 1
        if self.broadcast_b and self.broadcast_h:
            self.dense_mask = dense_mask[0, 0].contiguous()
        elif self.broadcast_b:
            self.dense_mask = dense_mask[0].contiguous()
        elif self.broadcast_h:
            self.dense_mask = dense_mask[:, 0].contiguous()
        else:
            self.dense_mask = dense_mask.contiguous()

    def __call__(self, b, h, q_idx, kv_idx):
        q_idx = q_idx.to(torch.int64)
        kv_idx = kv_idx.to(torch.int64)
        if self.broadcast_b and self.broadcast_h:
            return self.dense_mask[q_idx, kv_idx]
        if self.broadcast_b:
            return self.dense_mask[h.to(torch.int64), q_idx, kv_idx]
        if self.broadcast_h:
            return self.dense_mask[b.to(torch.int64), q_idx, kv_idx]
        return self.dense_mask[b.to(torch.int64), h.to(torch.int64), q_idx, kv_idx]


def dense_binary_mask_to_block_mask(
    attention_mask: torch.Tensor,
    B: int=None,
    H: int=None,
    Q_LEN: int=None,
    KV_LEN: int=None,
    block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    dense_bool = _normalize_dense_binary_mask(attention_mask)
    dense_bool = dense_bool.cuda()

    if B is None:
        B = dense_bool.size(0)
    if H is None:
        H = dense_bool.size(1)
    if Q_LEN is None:
        Q_LEN = dense_bool.size(2)
    if KV_LEN is None:
        KV_LEN = dense_bool.size(3)


    mask_B, mask_H, mask_Q_LEN, mask_KV_LEN = dense_bool.shape
    if mask_B not in (1, B):
        raise ValueError(f"Mask batch dim must be 1 or B={B}, got {mask_B}")
    if mask_H not in (1, H):
        raise ValueError(f"Mask head dim must be 1 or H={H}, got {mask_H}")
    if mask_Q_LEN != Q_LEN or mask_KV_LEN != KV_LEN:
        raise ValueError(
            f"Mask sequence dims must match Q_LEN/KV_LEN=({Q_LEN}, {KV_LEN}), "
            f"got ({mask_Q_LEN}, {mask_KV_LEN})"
        )

    pad_q = (block_size - Q_LEN % block_size) % block_size
    pad_kv = (block_size - KV_LEN % block_size) % block_size
    padded_dense_bool = (
        F.pad(dense_bool, (0, pad_kv, 0, pad_q), value=False)
        if pad_q or pad_kv
        else dense_bool
    )
    padded_dense_bool = padded_dense_bool.contiguous()
    partial_blocks, full_blocks = _convert_mask_to_block_mask_bool_reduce(
        padded_dense_bool,
        Q_BLOCK_SIZE=block_size,
        KV_BLOCK_SIZE=block_size,
    )
    mask_mod = _DenseMaskMod(padded_dense_bool) if partial_blocks.any() else noop_mask
    return _create_sparse_block_from_block_mask(
        (partial_blocks, full_blocks),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE=block_size,
        KV_BLOCK_SIZE=block_size,
    )
