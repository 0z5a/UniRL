from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
    unpad_input_for_concatenated_sequences,
    index_first_axis,
    index_put_first_axis,
)
from einops import rearrange
from hy_parallelism.models.modules.attentions.flash import FlashAttnMaskInfo

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except:
    flash_attn_varlen_func_v3 = None
    print("flash_attn_varlen_func_v3 not available")


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False, packed=False,
):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    if not isinstance(key_padding_mask, FlashAttnMaskInfo):
        key_padding_mask = FlashAttnMaskInfo.from_attention_mask(
            key_padding_mask, pack=packed
        )
    assert key_padding_mask.pack == packed
    x_unpad, indices, cu_seqlens, max_s, *_ = key_padding_mask.unpad(x)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    x_unpad_dtype = x_unpad.dtype
    # Flash attention always requires bfloat16 or float16 input
    if x_unpad_dtype not in [torch.bfloat16, torch.float16]:
        x_unpad = x_unpad.to(torch.bfloat16)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output_unpad = output_unpad.to(x_unpad_dtype)
    if packed:
        output = index_put_first_axis(output_unpad, indices, batch_size * seqlen).unsqueeze(0)
    else:
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    return output


def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False, packed=False,
):
    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")
    
    if isinstance(qkv, tuple):
        query, key, value = qkv
        batch_size, seqlen, nheads, head_dim = query.shape
    else:
        batch_size, seqlen, _, nheads, head_dim = qkv.shape
        query, key, value = qkv.unbind(dim=2)

    if not isinstance(key_padding_mask, FlashAttnMaskInfo):
        key_padding_mask = FlashAttnMaskInfo.from_attention_mask(
            key_padding_mask, pack=packed
        )
    assert key_padding_mask.pack == packed
    query_unpad, indices, cu_seqlens_q, max_seqlen_q, *_ = key_padding_mask.unpad(
        rearrange(query, "b s h d -> b s (h d)")
    )
    key_unpad, _, cu_seqlens_k, *_ = key_padding_mask.unpad(
        rearrange(key, "b s h d -> b s (h d)")
    )
    value_unpad, *_ = key_padding_mask.unpad(
        rearrange(value, "b s h d -> b s (h d)")
    )
    
    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_q, 
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic
    )

    if packed:
        output = index_put_first_axis(output_unpad, indices, batch_size * seqlen).unsqueeze(0)
    else:
        output = rearrange(
            pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
            "b s (h d) -> b s h d", h=nheads
        )
    return output


def test_flash_attn_alignment(
    batch_size: int = 4,
    seq_len: int = 128,
    n_heads: int = 8,
    head_dim: int = 64,
    pct_valid: float = 0.8,  # 有效token比例
    causal: bool = True,
    tol: float = 1e-5
) -> dict:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qkv = torch.randn(
        batch_size, seq_len, 3, n_heads, head_dim,
        device=device, dtype=torch.float16
    )
    
    key_padding_mask = torch.ones(batch_size, seq_len, device=device)
    for i in range(batch_size):
        valid_len = int(seq_len * pct_valid)
        key_padding_mask[i, valid_len:] = 0  # 设置padding部分为0

    output_v2 = flash_attn_no_pad(
        qkv, key_padding_mask, causal=causal, deterministic=True
    )
    output_v3 = flash_attn_no_pad_v3(
        qkv, key_padding_mask, causal=causal, deterministic=True
    )

    valid_indices = key_padding_mask == 1
    valid_output_v2 = output_v2[valid_indices]
    valid_output_v3 = output_v3[valid_indices]

    abs_diff = (valid_output_v2 - valid_output_v3).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    is_aligned = torch.allclose(
        valid_output_v2, valid_output_v3, rtol=tol, atol=tol
    )
    
    return {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "is_aligned": bool(is_aligned),
        "output_shape": tuple(output_v2.shape),
        "valid_token_count": int(valid_indices.sum().item()),
        "causal_mode": causal,
        "tolerance": tol
    }


if __name__ == "__main__":
    base_test = test_flash_attn_alignment(pct_valid=0.7)
    print("基础测试结果:", base_test)
    
    full_valid_test = test_flash_attn_alignment(pct_valid=1.0)
    print("\n全有效token测试:", full_valid_test)
    
    high_pad_test = test_flash_attn_alignment(pct_valid=0.3)
    print("\n高padding比例测试:", high_pad_test)
    
    non_causal_test = test_flash_attn_alignment(causal=False)
    print("\n非因果注意力测试:", non_causal_test)
