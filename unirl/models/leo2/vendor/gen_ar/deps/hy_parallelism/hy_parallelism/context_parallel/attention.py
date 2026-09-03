# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import einops
import torch
from typing import Optional


from loguru import logger
import numpy as np
import torch.nn.functional as F
from einops import rearrange

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.cache_size_limit = 192
    torch._dynamo.config.accumulated_cache_size_limit = 192
    flex_mask_cache = {}
except Exception:
    logger.warning("Could not load Sliding Tile Attention of FlexAttn.")



from .core import CPInfo, maybe_to_split_head, maybe_to_split_seq
from hy_parallelism.parallel_states import get_parallel_state


def _resolve_magi_mask(
    attn_mask,
    *,
    batch_size: int,
    sequence_length: int,
    causal: bool,
    device: torch.device,
):
    if attn_mask is not None:
        if isinstance(attn_mask, tuple):
            return attn_mask
        if isinstance(attn_mask, torch.Tensor):
            from hy_parallelism.models.modules.attentions.magi import dense_mask_to_magi_ranges
            return dense_mask_to_magi_ranges(attn_mask, device=device)
        raise TypeError(
            f"magi attn_mask must be a tuple (q_ranges, k_ranges, attn_type_map) "
            f"or a dense bool tensor, got {type(attn_mask)}"
        )
    if causal:
        from hy_parallelism.models.modules.attentions.magi import causal_to_magi_ranges
        return causal_to_magi_ranges(batch_size * sequence_length, device=device)
    raise ValueError("magi attention requires attn_mask or causal=True")


def _magi_attention_local(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask,
    causal: bool,
) -> torch.Tensor:
    from hy_parallelism.models.modules.attentions.magi import magi_scaled_dot_product_attention

    magi_mask = _resolve_magi_mask(
        attn_mask,
        batch_size=query.size(0),
        sequence_length=query.size(1),
        causal=causal,
        device=query.device,
    )
    hidden_states = magi_scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=magi_mask,
    )
    return hidden_states.transpose(1, 2)


def _magi_attention_cp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_info: CPInfo,
) -> torch.Tensor:
    from magi_attention.api import calc_attn

    if cp_info.magi_key is None:
        raise ValueError(
            "magi attention with context parallel requires CPInfo.magi_key; "
            "build it via CPInfo.from_magi_metas() and pass query_cp_info"
        )

    b, s, h_q, d = query.shape
    _, _, h_kv, _ = key.shape
    q_flat = query.reshape(b * s, h_q, d).contiguous()
    k_flat = key.reshape(b * s, h_kv, d).contiguous()
    v_flat = value.reshape(b * s, h_kv, d).contiguous()
    hidden_states, _ = calc_attn(q_flat, k_flat, v_flat, cp_info.magi_key)
    return hidden_states.reshape(b, s, h_q, d)


def _resolve_magi_cp_info(
    query_cp_info: Optional[CPInfo],
    key_cp_info: Optional[CPInfo],
    value_cp_info: Optional[CPInfo],
) -> CPInfo:
    for cp_info in (query_cp_info, key_cp_info, value_cp_info):
        if cp_info is not None:
            return cp_info
    raise ValueError(
        "magi attention with context parallel requires query_cp_info (or key/value cp_info) "
        "with magi_key from CPInfo.from_magi_metas()"
    )



def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output

def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
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

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads
    )
    return output

@torch.compiler.disable
def attention_no_sp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    drop_rate: float = 0.0,
    text_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    query_cp_info: Optional[CPInfo] = None,
    key_cp_info: Optional[CPInfo] = None,
    value_cp_info: Optional[CPInfo] = None,
    attn_mode: str = "flash",
) -> torch.Tensor:

    sequence_length = query.size(1)

    if text_mask is not None:
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
    else:
        attn_mask = None

    if attn_mask is not None:
        if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
            assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
            attn_mask = attn_mask.to(torch.bool)
        elif attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(query.dtype)
            raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')

    # transpose q,k,v dim to fit scaled_dot_product_attention
    query = query.transpose(1, 2)  # B * Head_num * length * dim
    key = key.transpose(1, 2)      # B * Head_num * length * dim
    value = value.transpose(1, 2)  # B * Head_num * length * dim

    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(attn_mask[b, q_idx] & attn_mask[b, kv_idx], score, float('-inf'))

    if attn_mask is not None:
        raise NotImplementedError("pytorch experimental context parallel does not support flex attention.")
        hidden_states = flex_attention(query, key, value, score_mod=score_mod)
    else:
        # hidden_states = flex_attention(query, key, value)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=causal)

    # transpose back
    hidden_states = hidden_states.transpose(1, 2)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states



@torch.compiler.disable
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    drop_rate: float = 0.0,
    text_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    query_cp_info: Optional[CPInfo] = None,
    key_cp_info: Optional[CPInfo] = None,
    value_cp_info: Optional[CPInfo] = None,
    attn_mode: str = "flash",
) -> torch.Tensor:
    """
    Compute attention using flash_attn_no_pad or torch scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        drop_rate: Dropout rate for attention weights.
        attn_mask: Optional attention mask of shape [B, L].
        causal: Whether to apply causal masking.
        query_cp_info: Optional split metadata from maybe_scatter_seq(return_split_meta=True).
            When provided, all_to_all / all_gather skip runtime length collection.
        attn_mode: Attention mode: "torch", "flash2", "flash3", "sageattn", or "magi".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    enable_cp = get_parallel_state().cp_enabled
    use_ulysses_cp_ops = enable_cp and attn_mode != "magi"
    if use_ulysses_cp_ops:
        query = maybe_to_split_head(query, cp_info=query_cp_info)
        key = maybe_to_split_head(key, cp_info=key_cp_info)
        value = maybe_to_split_head(value, cp_info=value_cp_info)

    sequence_length = query.size(1)

    if attn_mode == "magi":
        if enable_cp:
            cp_info = _resolve_magi_cp_info(query_cp_info, key_cp_info, value_cp_info)
            hidden_states = _magi_attention_cp(query, key, value, cp_info)
        else:
            hidden_states = _magi_attention_local(
                query, key, value, attn_mask=attn_mask, causal=causal,
            )

    elif attn_mode == "torch":

        if text_mask is not None:
            attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        else:
            attn_mask = None

        if attn_mask is not None:
            if attn_mask.dtype != torch.bool and attn_mask.dtype in [torch.int64, torch.int32]:
                assert attn_mask.max() <= 1 and attn_mask.min() >= 0, f'Integer attention mask must be between 0 and 1 for torch attention.'
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(query.dtype)
                raise NotImplementedError(f'Float attention mask is not implemented for torch attention.')

        # transpose q,k,v dim to fit scaled_dot_product_attention
        query = query.transpose(1, 2)  # B * Head_num * length * dim
        key = key.transpose(1, 2)      # B * Head_num * length * dim
        value = value.transpose(1, 2)  # B * Head_num * length * dim

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(attn_mask[b, q_idx] & attn_mask[b, kv_idx], score, float('-inf'))

        if attn_mask is not None:
            hidden_states = flex_attention(query, key, value, score_mod=score_mod)
        else:
            hidden_states = flex_attention(query, key, value)

        # transpose back
        hidden_states = hidden_states.transpose(1, 2)

    elif attn_mode == "flash2":
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)

        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "flash3":
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, dropout_p=0.0, softmax_scale=None)

    elif attn_mode == "sageattn":
        from sageattention import sageattn
        hidden_states = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    else:
        raise NotImplementedError(
            f'Unsupported attention mode: {attn_mode}. '
            'Supported: torch, flash2, flash3, sageattn, magi.'
        )


    if use_ulysses_cp_ops:
        hidden_states = maybe_to_split_seq(hidden_states, cp_info=query_cp_info)

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, a*d)

    return hidden_states

# for jvp
def jvp_flash_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    JVP_PLAN = 'amorehead-open-source'

    if JVP_PLAN == 'amorehead-open-source':
        dtype = torch.float32
        query, key, value = map(lambda x: x.to(dtype), [query, key, value])

        from .jvp_attention import JVPAttn, MIN_SEQUENCE_LENGTH
        from .jvp_attention import attention as jvp_attention
        from torch._functorch.eager_transforms import JVP_NESTING


        if JVP_NESTING > 0:
            attn_forward = JVPAttn.fwd_dual
        else:
            attn_forward = jvp_attention
        # attn_forward = JVPAttn.apply
        MIN_SEQ_LEN = MIN_SEQUENCE_LENGTH
        batch, heads, seqlen, head_dim = query.shape

        # Compute required padding
        pad_len = (MIN_SEQ_LEN - (seqlen % MIN_SEQ_LEN)) % MIN_SEQ_LEN

        if pad_len > 0:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_len))
            key = torch.nn.functional.pad(key, (0, 0, 0, pad_len))
            value = torch.nn.functional.pad(value, (0, 0, 0, pad_len))
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_len, 0, pad_len))
        else:
            pass

        ret = attn_forward(
            query,
            key,
            value,
            attn_mask=attn_mask.expand(query.shape[0], query.shape[1], query.shape[2], attn_mask.shape[3]) if attn_mask is not None else None,
            causal=is_causal,
            sm_scale=scale,
            warp_specialize=True,
            USE_TMA=False,
            # dropout_p=attn_dropout_p if self.training else 0.0,  # NOTE: Attention dropout is currently unsupported
        )
        ret = ret[:, :, :seqlen, :]

        return ret
    else:
        raise NotImplementedError("JVP flash attention is not implemented for non-open-source version")