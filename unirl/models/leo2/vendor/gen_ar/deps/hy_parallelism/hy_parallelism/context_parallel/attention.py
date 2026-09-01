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



from .core import all_to_all_4D, all_gather, maybe_to_split_head, maybe_gather_head, maybe_to_split_seq, maybe_gather_seq
from hy_parallelism.parallel_states import get_parallel_state




def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    # FA2 removed: delegate to the shared FA3 no-pad implementation (vendored padding).
    # Lazy import keeps this deps package importable without hymm at module load.
    from hymm.models.basic.flash_attn_no_pad import flash_attn_no_pad as _fa3_no_pad
    return _fa3_no_pad(
        qkv, key_padding_mask, causal=causal, dropout_p=dropout_p,
        softmax_scale=softmax_scale, deterministic=deterministic,
    )


# Back-compat alias: both names now resolve to the same FA3 implementation.
flash_attn_no_pad_v3 = flash_attn_no_pad

@torch.compiler.disable
def attention_no_sp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    drop_rate: float = 0.0,
    text_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    query_pad: int = None,
    key_pad: int = None,
    value_pad: int = None,
    attn_mode: str = "flash",
) -> torch.Tensor:
    # raise NotImplementedError('This function is only for pytest.')

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
    query_pad: int = None,
    key_pad: int = None,
    value_pad: int = None,
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
        attn_mode: Attention mode, either "flash" or "torch". Defaults to "flash".

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    enable_sp = get_parallel_state().sp_enabled
    if enable_sp:
        query = maybe_to_split_head(query, input_pad=query_pad)
        key = maybe_to_split_head(key, input_pad=key_pad)
        value = maybe_to_split_head(value, input_pad=value_pad)

    sequence_length = query.size(1)

    if attn_mode == "torch":

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
        raise NotImplementedError(f'Unsupported attention mode: {attn_mode}. Only torch, flash, flash3, sageattn and flex-block-attn are supported.')


    if enable_sp:
        hidden_states = maybe_to_split_seq(hidden_states, input_pad=query_pad)

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