import math
from functools import cache
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, BlockMask
try:
    from .flash_attn_no_pad import flash_attn_no_pad, flash_attn_no_pad_v3
except Exception as e:
    print(f"Failed to import flash_attn_no_pad or flash_attn_no_pad_v3: {e}")
    flash_attn_no_pad = None
    flash_attn_no_pad_v3 = None

from hymm.models.basic.pos_emb_layers import apply_rope
from .model_config import TransformerConfig

compiled_flex_attention = None


@cache
def _load_sage_sm90_backend():
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90, sm90_compile
    # we need this function becuase the source from pypi sage==2.2.0 has some error.
    # please refer to https://github.com/thu-ml/SageAttention/commit/d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4
    sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf = (
        torch.ops.sageattention_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.default
    )
    return sageattn_qk_int8_pv_fp8_cuda_sm90


class SelfAttention(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.deterministic = False

        if not config.split_qkv:
            self.qkv_proj = nn.Linear(
                config.hidden_size, (config.num_attention_heads + 2 * config.num_kv_heads) * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
            self.k_proj = nn.Linear(
                config.hidden_size, config.num_kv_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
            self.v_proj = nn.Linear(
                config.hidden_size, config.num_kv_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )

        self.o_proj = nn.Linear(
            config.attention_head_size * config.num_attention_heads, config.hidden_size,
            bias=config.attention_bias, **factory_kwargs
        )

        if config.use_qk_norm:
            self.query_layernorm = config.qk_norm_class(
                config.attention_head_size, **config.get_norm_kwargs(config.qk_norm_type), **factory_kwargs
            )
            self.key_layernorm = config.qk_norm_class(
                config.attention_head_size, **config.get_norm_kwargs(config.qk_norm_type), **factory_kwargs
            )

    def enable_deterministic(self) -> None:
        self.deterministic = True

    def disable_deterministic(self) -> None:
        self.deterministic = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        if not self._config.split_qkv:
            qkv = self.qkv_proj(hidden_states)
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(bsz, seqlen, n_kv_head, total_qkv, head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (bsz, num_kv_heads, total_qkv, T, head_size)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
        else:
            new_q = self.q_proj(hidden_states).view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
            new_k = self.k_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
            new_v = self.v_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [new_q, new_k, new_v])

        q = q.reshape(bsz, -1, seqlen, head_size)  # (B, n_q_head, T, hs)
        k = k.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)
        v = v.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)

        # QWen VL use qk norm before rotary pos emb
        if self._config.use_qk_norm and self._config.pre_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        apply_rope_kwargs = dict(
            apply_rope_in_fp32=self._config.apply_rope_in_fp32,
            interleave=self._config.rope_interleave,
        )
        q = apply_rope(q, *rotary_position_embeddings, **apply_rope_kwargs)
        k = apply_rope(k, *rotary_position_embeddings, **apply_rope_kwargs)

        # Some others use qk norm after rotary pos emb
        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = k.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if q_per_kv != 1:
            k = k.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            v = v.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        y = self.scaled_dot_product_attention(q, k, v, attention_mask)

        y = y.reshape(bsz, seqlen, head_size * n_q_head)  # re-assemble all head outputs side by side

        # output projection
        return self.o_proj(y)

    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input q, k, v: (bsz, n_head, seqlen, head_size)
        # Output q, k, v: (bsz, seqlen, n_head, head_size)
        scale = 1.0 / math.sqrt(self._config.attention_head_size)
        attn_impl = self._config.attn_impl
        if not torch.is_grad_enabled() and not self.training and self._config.inference_attn_impl is not None:
            attn_impl = self._config.inference_attn_impl

        if attn_impl == "flex":
            assert isinstance(mask, BlockMask), \
                "config.attn_impl is set to `flex`, a BlockMask must be provided to the forward function."
            global compiled_flex_attention
            if compiled_flex_attention is None:
                compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            y = compiled_flex_attention(q, k, v, block_mask=mask, scale=scale)  # noqa
            return y.transpose(1, 2)

        elif attn_impl == "sdpa":
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
                # If q only has one token (typically in AR model decoding stage), we should use full attention.
                is_causal=mask is None and q.size(2) > 1
            )
            return y.transpose(1, 2)

        elif attn_impl in ["flash", "flash_packed"]:
            assert flash_attn_no_pad is not None
            # Transpose for FA layout: bsz, n_head, seqlen, head_size -> bsz, seqlen, n_head, head_size
            qkv = torch.stack([
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
            ], dim=2)   # (bsz, seqlen, 3, n_head, head_size)
            # The mask should be in shape (bsz, seqlen) with True for all image tokens and valid text tokens.
            y = flash_attn_no_pad(qkv, mask, causal=False, dropout_p=0.0, softmax_scale=None,
                                  deterministic=self.deterministic, packed=attn_impl == "flash_packed")
            return y

        elif attn_impl in ["flash3", "flash3_packed"]:
            assert flash_attn_no_pad_v3 is not None
            # Transpose for FA layout: bsz, n_head, seqlen, head_size -> bsz, seqlen, n_head, head_size
            qkv = (
                q.permute(0, 2, 1, 3),
                k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
            )
            # The mask should be in shape (bsz, seqlen) with True for all image tokens and valid text tokens.
            y = flash_attn_no_pad_v3(qkv, mask, causal=False, dropout_p=0.0, softmax_scale=None,
                                     deterministic=self.deterministic, packed=attn_impl == "flash3_packed")
            return y

        elif attn_impl == "sageattn":
            if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
                raise RuntimeError("attn_impl=sageattn currently supports NVIDIA SM90 GPUs only")
            # SageAttention currently supports full attention only; mask is intentionally ignored.
            sage_func = _load_sage_sm90_backend()
            y = sage_func(
                q,
                k,
                v,
                tensor_layout="HND",
                is_causal=False,
                qk_quant_gran="per_warp",
                pv_accum_dtype="fp32+fp32",
            )
            return y.transpose(1, 2)

        else:
            raise NotImplementedError(f"Unsupported attention implementation: {attn_impl}")
