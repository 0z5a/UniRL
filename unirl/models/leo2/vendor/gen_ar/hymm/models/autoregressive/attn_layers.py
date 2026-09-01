from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask
    # Compile the flex_attention function
    flex_attention = torch.compile(flex_attention, dynamic=False)
except:
    BlockMask = None

from .utils import batched_index_copy_
from ..basic.pos_emb_layers import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        block_idx: int,
        tp_friendly_qkv = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch

        self.tp_friendly_qkv = tp_friendly_qkv
        if not tp_friendly_qkv:
            self.attn = nn.Linear(config.n_embd, shape, bias=config.attention_bias, **factory_kwargs)
        else:
            if config.attention_bias:
                raise NotImplementedError('Checkpoint conversion with bias is not implemented for TP-friendly qkv yet.')
            self.attn_q = nn.Linear(config.n_embd, config.n_head * config.head_size, bias=config.attention_bias, **factory_kwargs)
            self.attn_k = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.attention_bias, **factory_kwargs)
            self.attn_v = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.attention_bias, **factory_kwargs)


        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.attention_o_proj_bias, **factory_kwargs)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = config.norm_class(config.head_size, eps=config.norm_eps, **factory_kwargs)
            self.k_norm = config.norm_class(config.head_size, eps=config.norm_eps, **factory_kwargs)
            
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and block_idx % config.sliding_window_layer_placing == 0
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        if not self.tp_friendly_qkv:
            qkv = self.attn(x)
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
        else:
            new_q = self.attn_q(x).view(B, T, self.config.n_query_groups, self.config.n_head // self.config.n_query_groups, self.config.head_size)
            new_k = self.attn_k(x).view(B, T, self.config.n_query_groups, 1, self.config.head_size)
            new_v = self.attn_v(x).view(B, T, self.config.n_query_groups, 1, self.config.head_size)
            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [new_q, new_k, new_v])


        # ============================================================================================
        # following codes are used for traditional qkv format, we leave them here for testing
        # q_per_kv = self.config.n_head // self.config.n_query_groups
        # total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        # qkv = qkv.view(B, T, self.config.n_query_groups * total_qkv, self.config.head_size)

        # # split batched computation into three
        # q, k, v = qkv.split((q_per_kv * self.config.n_query_groups, 1 * self.config.n_query_groups, 1 * self.config.n_query_groups), dim=2)
        # q = q.permute(0, 2, 1, 3)
        # k = k.permute(0, 2, 1, 3).unsqueeze(2)
        # v = v.permute(0, 2, 1, 3).unsqueeze(2)
        # ============================================================================================

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        rope_kwargs = dict(apply_rope_in_fp32=True)
        if self.config.rope_type in ["default", "interleave", "2d"]:
            if self.config.rope_n_elem == q.shape[-1]:
                q = apply_rope(q, cos, sin, interleave=self.config.rope_type == "interleave", **rope_kwargs)
                k = apply_rope(k, cos, sin, interleave=self.config.rope_type == "interleave", **rope_kwargs)
            else:
                q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin, interleave=self.config.rope_type == "interleave", **rope_kwargs)
                k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin, interleave=self.config.rope_type == "interleave", **rope_kwargs)
                q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
                k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
        elif self.config.rope_type in ["3d", "3d-interleave"]:
            # TODO: `3d` gives wrong apply_rope results, it should be deprecated in future.
            q = apply_rope(q, cos, sin, interleave=self.config.rope_type == "3d-interleave", **rope_kwargs)
            k = apply_rope(k, cos, sin, interleave=self.config.rope_type == "3d-interleave", **rope_kwargs)
        else:
            raise ValueError(f"Unknown rope type: {self.config.rope_type}")

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            scores = q @ k.mT * scale
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            )
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            if mask is None:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=True
                )
            elif isinstance(mask, torch.Tensor):
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
                )
            elif BlockMask is not None and isinstance(mask, BlockMask):
                q = q.to(dtype=v.dtype)
                k = k.to(dtype=v.dtype)
                y = flex_attention(q, k, v, block_mask=mask, scale=scale)
            else:
                raise NotImplementedError(f"Attention type {self.attn_type} not implemented")

        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


# Full Attention
class SelfAttention(CausalSelfAttention):
    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False,
        )
        return y.transpose(1, 2)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        n = k.size(0)
        k = batched_index_copy_(self.k[:n, ...], dim=-2, idx=input_pos, val=k)
        v = batched_index_copy_(self.v[:n, ...], dim=-2, idx=input_pos, val=v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)
