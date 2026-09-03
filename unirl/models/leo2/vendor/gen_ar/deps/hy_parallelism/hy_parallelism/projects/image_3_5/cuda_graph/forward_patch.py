from __future__ import annotations

import contextlib
import math
import os
from typing import Callable, Optional, Union

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from hymm.core.global_vars import get_parallel_state
from hymm.models.autoregressive.custom_cache import HunyuanStaticCache
from hymm.models.basic.pos_emb_layers import apply_rope_qk
from hy_parallelism.models.modules.attentions.flex import flex_attention
from hy_parallelism.tools.profiling import profile_func

ENV = "HY35_DECODE_CUDA_GRAPH"


def enabled() -> bool:
    return os.environ.get(ENV, "1") == "1"


def resolve_attn_impl(config) -> str:
    fallback_backend = "magi" if get_parallel_state().cp_size == 1 else "flex"
    return getattr(config, "attn_impl", fallback_backend)


def _kv_cache_tensors(past_key_values: HunyuanStaticCache, layer_idx: int):
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    layer = past_key_values.layers[layer_idx]
    if layer.keys is None:
        raise RuntimeError(f"KV cache layer {layer_idx} is not initialized")
    return layer.keys, layer.values


def make_graph_safe_update_kv_cache(layer_idx: int) -> Callable:
    def update_kv_cache(past_key_values, k, v, input_pos):
        k_out, v_out = _kv_cache_tensors(past_key_values, layer_idx)
        pos = input_pos if input_pos.is_cuda else input_pos.cuda(non_blocking=True)
        if pos.dim() == 2:
            assert pos.shape[0] == 1, "decode CUDA graph only supports batch size 1"
            pos = pos.reshape(-1)
        k_out.index_copy_(2, pos, k.to(k_out.dtype))
        v_out.index_copy_(2, pos, v.to(v_out.dtype))
        return k_out, v_out

    return update_kv_cache


def bind_graph_safe_kv_updates(model) -> None:
    from hymm.models.multimodal.hunyuan_multimodal import CausalSelfAttention

    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn, CausalSelfAttention):
            attn._update_kv_cache = make_graph_safe_update_kv_cache(attn.layer_idx)


def gpu_input_pos(input_pos: torch.Tensor) -> torch.Tensor:
    return input_pos if input_pos.is_cuda else input_pos.cuda(non_blocking=True)


def sync_decode_magi_attn_mask(
    model,
    past_key_values: HunyuanStaticCache,
    input_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build magi ranges for static-KV decode: q[0] attends to k[0:pos+1].

    Equivalent to dense mask ``(arange(kv_len) <= pos)`` with q_len=1, but uses
    FFA instead of SDPA math backend. Updated outside CUDA graph replay.
    """

    k_out, _ = _kv_cache_tensors(past_key_values, 0)
    valid_kv_len = int(input_pos.reshape(-1)[-1].item()) + 1
    device = k_out.device

    magi_mask = getattr(model, "_static_decode_attn_mask", None)
    if magi_mask is None:
        q_ranges = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, valid_kv_len]], dtype=torch.int32, device=device)
        attn_type_map = torch.zeros(1, dtype=torch.int32, device=device)
        magi_mask = (q_ranges, k_ranges, attn_type_map)
        model._static_decode_attn_mask = magi_mask # 关键
    else:
        _, k_ranges, _ = magi_mask
        k_ranges[0, 1] = valid_kv_len
    return magi_mask


def sync_decode_flex_block_mask(
    model,
    past_key_values: HunyuanStaticCache,
    input_pos: torch.Tensor,
) -> BlockMask:
    k_out, _ = _kv_cache_tensors(past_key_values, 0)
    pos = int(input_pos.reshape(-1)[-1].item())
    kv_cache_len = int(k_out.size(2))
    device = k_out.device

    pos_buf = getattr(model, "_static_decode_flex_pos", None)
    if (
        not isinstance(pos_buf, torch.Tensor)
        or pos_buf.numel() != 1
        or pos_buf.device != device
    ):
        pos_buf = torch.tensor([pos], dtype=torch.int64, device=device)
        model._static_decode_flex_pos = pos_buf
    else:
        pos_buf.fill_(pos)

    def mask_mod(b, h, q_idx, kv_idx):
        return kv_idx.to(torch.int64) <= pos_buf[0]

    new_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=1,
        KV_LEN=kv_cache_len,  # static cache, no crop
        device=device,
        _compile=True,
    )

    old_mask = getattr(model, "_static_decode_attn_mask", None)
    if not isinstance(old_mask, BlockMask) or old_mask.shape[-1] != kv_cache_len:
        model._static_decode_attn_mask = new_mask
        return new_mask

    # Keep the captured BlockMask object; only refresh metadata + pos_buf.
    old_mask.kv_num_blocks.copy_(new_mask.kv_num_blocks)
    old_mask.kv_indices.copy_(new_mask.kv_indices)
    if old_mask.full_kv_num_blocks is not None and new_mask.full_kv_num_blocks is not None:
        old_mask.full_kv_num_blocks.copy_(new_mask.full_kv_num_blocks)
        old_mask.full_kv_indices.copy_(new_mask.full_kv_indices)
    if old_mask.q_num_blocks is not None and new_mask.q_num_blocks is not None:
        old_mask.q_num_blocks.copy_(new_mask.q_num_blocks)
        old_mask.q_indices.copy_(new_mask.q_indices)
    if old_mask.full_q_num_blocks is not None and new_mask.full_q_num_blocks is not None:
        old_mask.full_q_num_blocks.copy_(new_mask.full_q_num_blocks)
        old_mask.full_q_indices.copy_(new_mask.full_q_indices)
    return old_mask


def sync_decode_flash_cache_seqlens(
    model,
    past_key_values: HunyuanStaticCache,
    input_pos: torch.Tensor,
) -> torch.Tensor:
    k_out, _ = _kv_cache_tensors(past_key_values, 0)
    bsz = int(k_out.size(0))
    kv_cache_len = int(k_out.size(2))
    device = k_out.device
    # Outside graph: .item() sync is intentional (same as magi k_ranges update).
    valid_kv_len = int(input_pos.reshape(-1)[-1].item()) + 1
    if valid_kv_len > kv_cache_len:
        valid_kv_len = kv_cache_len

    seqlens = getattr(model, "_static_decode_cache_seqlens", None)
    if (
        not isinstance(seqlens, torch.Tensor)
        or seqlens.dtype != torch.int32
        or seqlens.numel() != bsz
        or seqlens.device != device
    ):
        seqlens = torch.zeros(bsz, dtype=torch.int32, device=device)
        model._static_decode_cache_seqlens = seqlens
    seqlens.fill_(valid_kv_len)
    # Reuse attn_mask slot: int32 rank-1 tensor means flash_kvcache path.
    model._static_decode_attn_mask = seqlens
    return seqlens


def sync_decode_attn_mask(
    model,
    past_key_values: HunyuanStaticCache,
    input_pos: torch.Tensor,
) -> Union[BlockMask, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    attn_impl = resolve_attn_impl(model._config)
    # FFA is slow in long-seq decoding
    return sync_decode_flash_cache_seqlens(model, past_key_values, input_pos)
    if attn_impl == "flex":
        return sync_decode_flex_block_mask(model, past_key_values, input_pos)
    if attn_impl == "magi":
        return sync_decode_magi_attn_mask(model, past_key_values, input_pos)
    # sdpa / flash / flash_kvcache: flash_attn_with_kvcache + static cache_seqlens
    return sync_decode_flash_cache_seqlens(model, past_key_values, input_pos)


def gather_rope_from_cache(model, input_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_cache = model.cached_rope.cos_cache
    sin_cache = model.cached_rope.sin_cache
    if cos_cache is None or sin_cache is None:
        raise RuntimeError("cached_rope is empty; run prefill before decode CUDA graph capture")

    head_size = cos_cache.size(-1)
    pos = gpu_input_pos(input_pos)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    idx = pos.unsqueeze(-1).expand(-1, -1, head_size)
    cos = torch.gather(cos_cache, 1, idx)
    sin = torch.gather(sin_cache, 1, idx)
    if not model._config.apply_rope_in_fp32:
        cos = cos.to(dtype=model.dtype)
        sin = sin.to(dtype=model.dtype)
    return cos, sin


def graph_safe_decode_forward(model, *args, **kwargs):
    import torch.nn.functional as F

    from hymm.models.multimodal.hunyuan_multimodal_state import HunyuanMultimodalOutput

    if args:
        input_ids = args[0]
    else:
        input_ids = kwargs["input_ids"]

    input_pos = kwargs["input_pos"]
    past_key_values = kwargs["past_key_values"]
    und_token_indices = kwargs["und_token_indices"]
    gen_token_indices = kwargs["gen_token_indices"]

    bsz = input_ids.shape[0]
    pos = gpu_input_pos(input_pos)
    cos, sin = gather_rope_from_cache(model, pos)

    hidden_states = model.model["embed_tokens"](input_ids)
    if model._config.use_mot and model._config_mot_gen.hidden_size != model._config.hidden_size:
        hidden_states = (
            hidden_states,
            hidden_states.new_zeros(
                bsz, hidden_states.shape[1], model._config_mot_gen.hidden_size,
            ),
        )

    und_idx = und_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states[0].shape[-1])
    gen_idx = gen_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states[1].shape[-1])
    hidden_states = (
        hidden_states[0].gather(dim=1, index=und_idx),
        hidden_states[1].gather(dim=1, index=gen_idx),
    )

    und_can_skip = False
    gen_can_skip = True

    magi_mask = getattr(model, "_static_decode_attn_mask", None)
    assert magi_mask is not None, "decode attention mask must be prepared before graph replay"

    for layer in model.model.layers:
        hidden_states = layer(
            hidden_states,
            magi_mask,
            (cos, sin),
            pos,
            past_key_values,
            und_token_indices,
            gen_token_indices,
            und_can_skip,
            gen_can_skip,
        )

    und_hidden_states, _ = hidden_states
    und_hidden_states = model.model["norm"](und_hidden_states)
    if not model._config.tie_word_embeddings:
        logits = model.lm_head(und_hidden_states)
    else:
        logits = F.linear(und_hidden_states, model.model.embed_tokens.weight)

    return HunyuanMultimodalOutput(logits=logits, past_key_values=past_key_values)


def forward_fast_single_stream_no_graph_break_impl(
    self,
    hidden_states: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    past_key_values: Optional[HunyuanStaticCache] = None,
    und_token_indices: Optional[torch.Tensor] = None,
    gen_token_indices: Optional[torch.Tensor] = None,
    und_can_skip: bool = False,
    gen_can_skip: bool = False,
) -> torch.Tensor:
    assert not und_can_skip, "forward_fast_single_stream_no_graph_break requires und_can_skip=False"
    assert gen_can_skip, "forward_fast_single_stream_no_graph_break requires gen_can_skip=True"

    und_hidden_states, gen_hidden_states = hidden_states
    assert gen_hidden_states.numel() == 0

    bsz = und_hidden_states.shape[0]
    und_seqlen = und_hidden_states.shape[1]
    gen_seqlen = 0
    head_size = self._config.attention_head_size
    n_q_head = self._config.num_attention_heads
    n_kv_head = self._config.num_kv_heads
    q_per_kv = n_q_head // n_kv_head

    if not self._config.split_qkv:
        qkv = self.qkv_proj(und_hidden_states)
        total_qkv = q_per_kv + 2
        qkv = qkv.view(bsz, und_seqlen, n_kv_head, total_qkv, head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
    else:
        q = self.q_proj(und_hidden_states)
        k = self.k_proj(und_hidden_states)
        v = self.v_proj(und_hidden_states)
        q = q.view(bsz, und_seqlen, n_kv_head, q_per_kv, head_size).permute(0, 2, 3, 1, 4)
        k = k.view(bsz, und_seqlen, n_kv_head, 1, head_size).permute(0, 2, 3, 1, 4)
        v = v.view(bsz, und_seqlen, n_kv_head, 1, head_size).permute(0, 2, 3, 1, 4)

    q = q.reshape(bsz, n_q_head, und_seqlen, head_size)
    k = k.reshape(bsz, n_kv_head, und_seqlen, head_size)
    v = v.reshape(bsz, n_kv_head, und_seqlen, head_size)

    merged_seqlen = und_seqlen + gen_seqlen

    q_merge, k_merge, v_merge = q, k, v

    if self._config.use_qk_norm and self._config.pre_qk_norm:
        q_merge = self.query_layernorm(q_merge)
        k_merge = self.key_layernorm(k_merge)

    q_merge, k_merge = apply_rope_qk(
        q_merge, k_merge, *rotary_position_embeddings,
        apply_rope_in_fp32=self._config.apply_rope_in_fp32,
    )

    if self._config.use_qk_norm and not self._config.pre_qk_norm:
        q_merge = self.query_layernorm(q_merge)
        k_merge = self.key_layernorm(k_merge)

    q_merge = q_merge.to(v_merge.dtype)
    k_merge = k_merge.to(v_merge.dtype)

    if input_pos is not None:
        k_merge, v_merge = self._update_kv_cache(
            past_key_values, k_merge, v_merge, input_pos,
        )

    attn_mask = attention_mask
    scale = 1.0 / math.sqrt(self._config.attention_head_size)
    if isinstance(attn_mask, BlockMask):
        q_merge = q_merge.to(dtype=v_merge.dtype)
        k_merge = k_merge.to(dtype=v_merge.dtype)
        y = flex_attention(
            q_merge, k_merge, v_merge, block_mask=attn_mask, scale=scale,
            enable_gqa=q_merge.size(1) != k_merge.size(1),
        ).transpose(1, 2)
    elif isinstance(attn_mask, tuple):
        from hy_parallelism.models.modules.attentions.magi import magi_scaled_dot_product_attention
        y = magi_scaled_dot_product_attention(
            q_merge, k_merge, v_merge, attn_mask=attn_mask, scale=scale,
        )
    elif (
        isinstance(attn_mask, torch.Tensor)
        and attn_mask.dtype == torch.int32
        and attn_mask.ndim == 1
    ):
        # Transpose keeps headdim contiguous (stride(-1)==1), which FA requires;
        # no full .contiguous() copy of the KV cache.
        from flash_attn import flash_attn_with_kvcache
        y = flash_attn_with_kvcache(
            q_merge.transpose(1, 2),
            k_merge.transpose(1, 2),
            v_merge.transpose(1, 2),
            cache_seqlens=attn_mask,
            softmax_scale=scale,
            causal=False,
        )
    else:
        if q_merge.size(2) == 1 and input_pos is not None:
            pos = input_pos.reshape(-1)[-1]
            kv_len = k_merge.size(2)
            kv_idx = torch.arange(kv_len, device=k_merge.device, dtype=pos.dtype)
            attn_mask = (kv_idx <= pos).view(1, 1, 1, kv_len)
        y = torch.nn.functional.scaled_dot_product_attention(
            q_merge, k_merge, v_merge, attn_mask=attn_mask, dropout_p=0.0, scale=scale,
            is_causal=attn_mask is None and q_merge.size(2) > 1,
            enable_gqa=q_merge.size(1) != k_merge.size(1),
        )

    y = y.reshape(bsz, merged_seqlen, head_size * n_q_head)
    core_attn_out = y
    gen_core_attn_out = y.new_zeros(bsz, 0, y.size(-1))

    und_hidden_states = self.o_proj(core_attn_out)
    gen_hidden_states = self.o_proj_mot_gen(gen_core_attn_out)
    return und_hidden_states, gen_hidden_states


def bind_fast_attn_forward(attn):
    return forward_fast_single_stream_no_graph_break_impl.__get__(attn, type(attn))


@contextlib.contextmanager
def use_fast_forward_context(model):
    original_forward_dict = {}
    for layer in model.model.layers:
        original_forward_dict[id(layer.self_attn)] = layer.self_attn.forward
        layer.self_attn.forward = bind_fast_attn_forward(layer.self_attn)
    try:
        yield
    finally:
        for layer in model.model.layers:
            layer.self_attn.forward = original_forward_dict[id(layer.self_attn)]
