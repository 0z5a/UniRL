# coding=utf-8
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
#
""" PyTorch HunYuan model."""
import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.cuda import nvtx

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_hunyuan import HunYuanConfig
try:
    import flashinfer
except Exception as e:
    flashinfer = None

try:
    from torch.nn.attention.flex_attention import flex_attention, BlockMask

    # Compile the flex_attention function
    flex_attention = torch.compile(flex_attention, dynamic=False)
except:
    BlockMask = None

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

logger = logging.get_logger(__name__)


def topkgating(
        logits: Tensor,
        topk: int,
        group_limited_greedy: bool = False,
        n_group: int = None,
        topk_group: int = None,
        norm_topk_prob: bool = False,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    if group_limited_greedy:
        group_shape = list(gates.shape[:-1]) + [n_group, gates.shape[-1] // n_group]
        group_scores = (
            gates.reshape(group_shape).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                group_shape
            )
            .reshape(list(gates.shape))
        )  # [n, e]
        gates = gates.masked_fill(~score_mask.bool(), 0.0)

    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    # expert_mask_aux = expert_mask.max(dim=-2)[0]
    # tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    # router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    # l_aux = num_experts ** 2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)
    if drop_tokens:
        expert_capacity = int(max(topk, topk * gates.shape[0] // gates.shape[1]) * capacity_factor)
    else:
        expert_index_flat = expert_index.flatten()
        tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
        expert_capacity = torch.max(tokens_per_expert).item()

    # Force norm topk for gemini A13B
    if norm_topk_prob and topk > 1:
        gates_s = torch.clamp(
            torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps
        )
        router_probs = gates / gates_s
    else:
        router_probs = gates * routed_scaling_factor
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    # Shape: [num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(-1)

    # Create mask out of indices.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))
    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [tokens_per_group, num_experts].
    token_priority = torch.max(token_priority, dim=1)[0]

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [tokens_per_group, num_experts, expert_capacity].
    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    # exp_counts_capacity = torch.sum(dispatch_mask)
    # exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    # return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts
    return [None, None], combine_weights, dispatch_mask, None


def top1gating(logits: Tensor, random_routing_dropped_token: bool = False):
    """Implements Top1Gating on logits."""
    # everything is in fp32 in this function
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    capacity = gates.shape[0]

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions
    # exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')
    exp_counts = torch.sum(mask1, dim=0).detach()

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    mask1_rand = mask1

    top_idx = torch.topk(mask1_rand, k=capacity, dim=0)[1]

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    mask1_bk = mask1
    if random_routing_dropped_token:
        not_full = capacity - new_mask1.sum(dim=0)
        sorted_notfull, indices_notfull = torch.sort(not_full, descending=True)
        sorted_notfull = sorted_notfull.to(torch.int64)
        not_full_experts_ids = torch.repeat_interleave(indices_notfull, sorted_notfull)
        shuffle_not_full_ids = torch.randperm(not_full_experts_ids.shape[0])
        not_full_experts_ids = not_full_experts_ids[shuffle_not_full_ids]
        indices1_s_after_drop = torch.argmax(new_mask1, dim=1)
        # get drop idx
        drop_mask = 1 - new_mask1.sum(dim=1)
        drop_mask = drop_mask.bool()
        drop_idx = drop_mask.nonzero().reshape(-1)
        drop_num = drop_mask.sum().to(torch.int64)
        indices1_s_after_drop.scatter_(0, drop_idx, not_full_experts_ids[:drop_num])
        nodrop_mask1 = F.one_hot(indices1_s_after_drop, num_classes=num_experts)
        mask1 = nodrop_mask1

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()  # one hot to float
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    exp_counts_capacity = torch.sum(mask1_bk)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0])
    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be "
        "removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in "
        "v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class HunYuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HunYuanRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(HunYuanRMSNorm)


class HunYuanRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # inv_freq = inv_freq.bfloat16()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1).float()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class HunYuanLinearScalingRotaryEmbedding(HunYuanRotaryEmbedding):
    """HunYuanRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class HunYuanDynamicNTKScalingRotaryEmbedding(HunYuanRotaryEmbedding):
    """
    HunYuanRotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class HunYuanDynamicNTKAlphaRotaryEmbedding(HunYuanRotaryEmbedding):
    """
    HunYuanRotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_alpha=1.0):
        self.scaling_alpha = scaling_alpha
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        base = self.base * self.scaling_alpha ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(HunYuanRotaryEmbedding):

    def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1,
            mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
                self.base
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
                self.scaling_factor
                * self.base
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if mla:
        b, h, s, d = q.shape
        q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_xdrope(q, k, cos, sin, position_ids, xdrope_section, output_size=None, bf16=False):
    # xd-rope 支持统一token任意维度的position id，最终的cos和sin按照xdrope_section里指定的比例提取后拼接
    x_dim = len(xdrope_section)

    cos = cos[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
    sin = sin[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()

    xdrope_section = [int(x * q.shape[-1] / 2) for x in xdrope_section] * 2

    # for xd concat
    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"
    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))], dim=-1)

    # for head repeat
    cos = cos.view(output_size[0], 1, output_size[2], -1)  # .repeat(1, output_size[1], 1, 1)
    sin = sin.view(output_size[0], 1, output_size[2], -1)  # .repeat(1, output_size[1], 1, 1)

    if bf16:
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    else:
        return (q * cos) + (rotate_half(q) * sin).to(q.dtype), (k * cos) + (rotate_half(k) * sin).to(k.dtype)


class HunYuanMLP(nn.Module):
    """
    使用 SwiGLU 的 MLP
    """

    def __init__(self, config: HunYuanConfig, layer_idx=None, is_shared_mlp=False, is_moe=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            # 如果是 moe 的话，优先用 moe_intermediate_size
            if config.moe_intermediate_size is not None:
                self.intermediate_size = config.moe_intermediate_size if isinstance(config.moe_intermediate_size,
                                                                                    int) else \
                config.moe_intermediate_size[layer_idx]

            if is_shared_mlp:
                num_shared_expert = config.num_shared_expert if isinstance(config.num_shared_expert, int) else \
                config.num_shared_expert[layer_idx]
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=2)
            down_proj = self.down_proj(x1 * self.act_fn(x2))
            return down_proj
        elif self.hidden_act == "gelu":
            intermediate = self.gate_and_up_proj(x)
            intermediate = self.act_fn(intermediate)
            output = self.down_proj(intermediate)
            return output
        else:
            assert False, "other hidden_act are not supported"


class HunYuanTopKGate(nn.Module):
    def __init__(self, config: HunYuanConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[layer_idx]
        self.drop_tokens = config.moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = config.moe_random_routing_dropped_token
        num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        self.wg = nn.Linear(config.hidden_size, num_experts, bias=False, dtype=torch.float32)

        # DeepSeek gating args
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.group_limited_greedy = config.group_limited_greedy

    def forward(self, hidden_states, type_=0):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if type_ == 0:
            if self.moe_topk == 1:
                gate_output = top1gating(logits, random_routing_dropped_token=self.random_routing_dropped_token)
                print(f"HunYuanTopKGate gate_output self.moe_topk=1")
            else:
                gate_output = topkgating(logits, self.moe_topk, group_limited_greedy=self.group_limited_greedy,
                                         n_group=self.n_group, topk_group=self.topk_group,
                                         norm_topk_prob=self.norm_topk_prob,
                                         routed_scaling_factor=self.routed_scaling_factor,
                                         capacity_factor=self.config.capacity_factor,
                                         drop_tokens=self.drop_tokens)
        else:
            gate_output = self.easyTopK(logits, self.moe_topk)

        return gate_output

    def easyTopK(self, logits, moe_topk):
        gates = F.softmax(logits, dim=1)
        topk_weight_1, expert_index = torch.topk(gates, moe_topk)
        weight_sums = topk_weight_1.sum(dim=1, keepdim=True)
        # 防止除零
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        topk_weight = topk_weight_1 / weight_sums

        return topk_weight, expert_index


class HunYuanMoE(nn.Module):
    def __init__(self, config: HunYuanConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.moe_topk = config.moe_topk
        self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunYuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
        self.gate = HunYuanTopKGate(config, layer_idx=layer_idx)
        self.experts = nn.ModuleList(
            [HunYuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True) for _ in range(self.num_experts)]
        )

        self._use_fused_moe = False
        self.moe_weight = None
        self.moe_weight_2 = None
        self._weights_initialized = False

    def enable_fused_moe(self, enable=True):
        self._use_fused_moe = enable
        if not enable:
            self._fused_weights_cache = None  # clear cache

    def clear_fused_cache(self):
        """清除融合权重缓存（权重更新后需要调用）"""
        self._fused_weights_cache = None

    def forward(self, hidden_states):
        torch.cuda.set_device(hidden_states.device.index)
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        # if self.layer_idx == 0:
        #     print("shared mlp output:", hidden_states_mlp, hidden_states_mlp.shape)

        reshaped_input = hidden_states.reshape(-1, hidden_size) # [bsz*seq_len, hidden_size]

        with nvtx.range("MoE"):
            if self._use_fused_moe:
                # Get expert weights
                if not self._weights_initialized:
                    self._initialize_weights_on_device(hidden_states.device)
                topk_weight, topk_index = self.gate(hidden_states, 1)

                combined_output = torch.zeros_like(reshaped_input)
                _ = flashinfer.fused_moe.cutlass_fused_moe(
                    reshaped_input.contiguous(),
                    topk_index.to(torch.int).contiguous(),
                    topk_weight.to(torch.float).contiguous(),
                    self.moe_weight,
                    self.moe_weight_2,
                    torch.bfloat16,
                    output=combined_output,
                    quant_scales=None,
                )
            else:
                # Original implementation - fallback for compatibility
                _, combine_weights, dispatch_mask, _ = self.gate(hidden_states, 0)
                dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
                chunks = dispatched_input.chunk(self.num_experts, dim=0)
                expert_outputs = []
                for chunk, expert in zip(chunks, self.experts):
                    expert_outputs.append(expert(chunk))

                expert_output = torch.cat(expert_outputs, dim=0)
                combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)

        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output

    def _initialize_weights_on_device(self, device):
        expert_weights_gate_up = []
        expert_weights_down = []

        for expert in self.experts:
            expert.to(device)
            expert_weights_gate_up.append(expert.gate_and_up_proj.weight.to(device))
            expert_weights_down.append(expert.down_proj.weight.to(device))

        self.moe_weight = torch.stack(expert_weights_gate_up).contiguous()
        self.moe_weight_2 = torch.stack(expert_weights_down).contiguous()
        # empty the expert weights
        for expert in self.experts:
            expert.gate_and_up_proj.weight.data = torch.empty(0, device=device)
            if expert.gate_and_up_proj.bias is not None:
                expert.gate_and_up_proj.bias.data = torch.empty(0, device=device)
            expert.down_proj.weight.data = torch.empty(0, device=device)
            if expert.down_proj.bias is not None:
                expert.down_proj.bias.data = torch.empty(0, device=device)

        self._weights_initialized = True


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HunYuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HunYuanConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # layer_idx 从 0 开始
        self.attention_type = 'cross' if config.use_cla and layer_idx % config.cla_share_factor != 0 else 'self'
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.use_qk_norm = config.use_qk_norm
        self.use_rotary_pos_emb = config.use_rotary_pos_emb
        self.hidden_size_q = self.head_dim * self.num_heads
        self.hidden_size_kv = self.head_dim * self.num_key_value_heads

        # custom
        self.use_flash_attn = config.use_flash_attn

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )

        if self.attention_type == 'self':
            self.qkv_proj = nn.Linear(
                self.hidden_size,
                self.hidden_size_q + 2 * self.hidden_size_kv,
                bias=config.attention_bias
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size_q, bias=config.attention_bias)

        self.o_proj = nn.Linear(self.hidden_size_q, self.hidden_size, bias=config.attention_bias)
        if self.use_qk_norm:
            self.query_layernorm = HunYuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunYuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.use_rotary_pos_emb:
            self._init_rope()
            self.position_embedding_xdrope = config.position_embedding_xdrope
            self.xdrope_section = config.xdrope_section

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = HunYuanRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            scaling_alpha = self.config.rope_scaling["alpha"]
            if scaling_type == "linear":
                self.rotary_emb = HunYuanLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                if scaling_alpha:
                    self.rotary_emb = HunYuanDynamicNTKAlphaRotaryEmbedding(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_alpha=scaling_alpha,
                        base=self.rope_theta,
                    )
                else:
                    self.rotary_emb = HunYuanDynamicNTKScalingRotaryEmbedding(
                        self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        scaling_factor=scaling_factor,
                        base=self.rope_theta,
                    )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            elif scaling_type is None:
                # Using custom rotary embedding
                self.rotary_emb = None
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            frist_step: bool = True,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            query_states = self.q_proj(hidden_states)
            orig_key_states, orig_value_states = kv_states
            key_states, value_states = kv_states
        else:
            qkv_states = self.qkv_proj(hidden_states)
            qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                            self.head_dim)
            query_states, key_states, value_states = torch.split(qkv_states, (self.num_key_value_groups, 1, 1), dim=3)
            orig_key_states, orig_value_states = key_states, value_states

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if self.use_rotary_pos_emb:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if self.position_embedding_xdrope:
                assert self.xdrope_section
                output_size = (query_states.size(0),
                               query_states.size(1),
                               query_states.size(2),
                               key_states.size(2))
                bf16 = self.config.torch_dtype == torch.bfloat16
                query_states, key_states = apply_rotary_pos_emb_xdrope(query_states, key_states, cos, sin, position_ids,
                                                                       self.xdrope_section, output_size, bf16)
            # elif self.config.use_gemini_rope:
            #     query_states, key_states = apply_rotary_pos_emb_gemini(query_states, key_states, cos, sin, position_ids)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, (orig_key_states, orig_value_states)


class HunYuanFlashAttention2(HunYuanAttention):
    """
    HunYuan flash attention module. This module inherits from `HunYuanAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # HunYuanFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            query_states = self.q_proj(hidden_states)
            orig_key_states, orig_value_states = kv_states
            key_states, value_states = kv_states
        else:
            qkv_states = self.qkv_proj(hidden_states)
            qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                            self.head_dim)
            query_states, key_states, value_states = torch.split(qkv_states, (self.num_key_value_groups, 1, 1), dim=3)
            orig_key_states, orig_value_states = key_states, value_states

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if self.use_rotary_pos_emb:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if self.position_embedding_xdrope:
                assert self.xdrope_section
                output_size = (query_states.size(0),
                               query_states.size(1),
                               query_states.size(2),
                               key_states.size(2))
                bf16 = self.config.torch_dtype == torch.bfloat16
                query_states, key_states = apply_rotary_pos_emb_xdrope(query_states, key_states, cos, sin, position_ids,
                                                                       self.xdrope_section, output_size, bf16)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (HunYuanRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, (orig_key_states, orig_value_states)

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class HunYuanSdpaAttention(HunYuanAttention):
    """
    HunYuan attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `HunYuanAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt
    to SDPA API.
    """

    # Adapted from HunYuanAttention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            frist_step: bool = True,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'HunYuanModel is using HunYuanSdpaAttention,'
                'but `torch.nn.functional.scaled_dot_product_attention`'
                'does not support `output_attentions=True`. Falling back to the manual attention implementation, '
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            query_states = self.q_proj(hidden_states)
            orig_key_states, orig_value_states = kv_states
            key_states, value_states = kv_states
        else:
            qkv_states = self.qkv_proj(hidden_states)
            qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                            self.head_dim)
            query_states, key_states, value_states = torch.split(qkv_states, (self.num_key_value_groups, 1, 1), dim=3)
            orig_key_states, orig_value_states = key_states, value_states

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if self.use_rotary_pos_emb:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            if self.position_embedding_xdrope:
                assert self.xdrope_section
                output_size = (query_states.size(0),
                               query_states.size(1),
                               query_states.size(2),
                               key_states.size(2))
                bf16 = self.config.torch_dtype == torch.bfloat16
                query_states, key_states = apply_rotary_pos_emb_xdrope(query_states, key_states, cos, sin, position_ids,
                                                                       self.xdrope_section, output_size, bf16)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a
            # causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, (orig_key_states, orig_value_states)


class HunYuanGeminiAttention(HunYuanAttention):
    """
    HunYuan attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `HunYuanAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt
    to SDPA API.
    """

    # Adapted from HunYuanAttention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            first_step: bool = True,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'HunYuanModel is using HunYuanSdpaAttention,'
                'but `torch.nn.functional.scaled_dot_product_attention`'
                'does not support `output_attentions=True`. Falling back to the manual attention implementation, '
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            query_states = self.q_proj(hidden_states)
            orig_key_states, orig_value_states = kv_states
            key_states, value_states = kv_states
        else:
            qkv_states = self.qkv_proj(hidden_states)
            qkv_states = qkv_states.reshape(bsz, q_len, self.num_key_value_heads, self.num_key_value_groups + 2,
                                            self.head_dim)
            query_states, key_states, value_states = torch.split(qkv_states, (self.num_key_value_groups, 1, 1), dim=3)
            orig_key_states, orig_value_states = key_states, value_states

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if self.use_flash_attn:
            from flash_attn import flash_attn_func
            target_dtype = key_states.dtype if key_states.dtype in [torch.bfloat16, torch.float16] else torch.bfloat16

            q_fa = query_states.to(target_dtype).transpose(1, 2).contiguous()
            k_fa = key_states.to(target_dtype).transpose(1, 2).contiguous()
            v_fa = value_states.to(target_dtype).transpose(1, 2).contiguous()

            with nvtx.range("attention"):
                timestep_index = gen_timestep_scatter_index[0,0].item()
                if timestep_index == -1:
                    # text attention
                    if attention_mask is None:
                        # decode attention
                        attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=False)
                    else:
                        # prefill attention
                        # attn_output = flash_attn_func(q_fa, k_fa, v_fa, causal=True)

                        attn_output = torch.nn.functional.scaled_dot_product_attention(
                            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
                        )
                        attn_output = attn_output.transpose(1, 2).contiguous()
                else:
                    # image attention
                    if first_step:
                        # casual_len = timestep_index + 1
                        # text_query_states = q_fa[:, :casual_len, :, :]
                        # text_key_states = k_fa[:, :casual_len, :, :]
                        # text_value_states = v_fa[:, :casual_len, :, :]
                        # text_attn_output = flash_attn_func(text_query_states, text_key_states, text_value_states, causal=True)
                        # image_query_states = q_fa[:, casual_len:, :, :]
                        # image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                        # attn_output = torch.cat((text_attn_output, image_attn_output), dim=1)

                        attn_output = torch.nn.functional.scaled_dot_product_attention(
                            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
                        )
                        attn_output = attn_output.transpose(1, 2).contiguous()
                    else:
                        casual_len = timestep_index + 1
                        timestep_query_states = q_fa[:, 0:1, :, :]
                        timestep_key_states = k_fa[:, :casual_len, :, :]
                        timestep_value_states = v_fa[:, :casual_len, :, :]
                        timestep_attn_output = flash_attn_func(timestep_query_states, timestep_key_states, timestep_value_states, causal=True)
                        image_query_states = q_fa[:, 1:, :, :]
                        image_attn_output = flash_attn_func(image_query_states, k_fa, v_fa, causal=False)
                        attn_output = torch.cat((timestep_attn_output, image_attn_output), dim=1)

        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0
            )
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, (orig_key_states, orig_value_states)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HunYuanConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # layer_idx 从 0 开始
        self.attention_type = 'cross' if config.use_cla and layer_idx % config.cla_share_factor != 0 else 'self'
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = HunYuanRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = HunYuanRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = HunYuanRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = HunYuanLinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = HunYuanDynamicNTKAlphaRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_alpha=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.reshape(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        norm_compressed_kv = (None, None)
        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            norm_compressed_kv = kv_states
            kv = (
                self.kv_b_proj(compressed_kv)
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            k_nope, value_states = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k_pe = self.qk_rope_a_proj_with_mqa(hidden_states)  # [b, s, qk_rope_head_dim]
            k_pe = k_pe.reshape(bsz, 1, q_len, self.qk_rope_head_dim)  # [b, s, 1, qk_rope_head_dim]
        else:
            compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
            kv = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )

            k_nope, value_states = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, mla=True)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, norm_compressed_kv


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV2
class DeepseekV2FlashAttention2(DeepseekV2Attention):
    """
    DeepseekV2 flash attention module. This module inherits from `DeepseekV2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_states: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.reshape(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        norm_compressed_kv = (None, None)
        if self.attention_type == "cross" and kv_states is not None and isinstance(kv_states, tuple):
            norm_compressed_kv = kv_states
            kv = (
                self.kv_b_proj(compressed_kv)
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            k_nope, value_states = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k_pe = self.qk_rope_a_proj_with_mqa(hidden_states)  # [b, s, qk_rope_head_dim]
            k_pe = k_pe.reshape(bsz, 1, q_len, self.qk_rope_head_dim)  # [b, s, 1, qk_rope_head_dim]
        else:
            compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            k_pe = k_pe.reshape(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
            kv = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
                .reshape(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )

            k_nope, value_states = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, mla=True)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, norm_compressed_kv

    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
            self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


HUNYUAN_ATTENTION_CLASSES = {
    "eager": HunYuanAttention,
    "flash_attention_2": HunYuanFlashAttention2,
    "sdpa": HunYuanSdpaAttention,
    "gemini_attention": HunYuanGeminiAttention,
}

MLA_ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
}


class HunYuanDecoderLayer(nn.Module):
    def __init__(self, config: HunYuanConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if config.use_mla:
            if config._attn_implementation == "sdpa":
                print("[Warning] Using MLA with SDPA attention is not supported, will use eager instead.")
                config._attn_implementation = "eager"
            self.self_attn = MLA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        else:
            config._attn_implementation = "gemini_attention"
            self.self_attn = HUNYUAN_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        if ((isinstance(config.num_experts, int) and config.num_experts > 1) or (
                isinstance(config.num_experts, list) and max(
                config.num_experts) > 1)) and layer_idx >= config.moe_layer_num_skipped:
            self.mlp = HunYuanMoE(config, layer_idx=layer_idx)
            if config.use_fused_moe:
                self.mlp.enable_fused_moe(True)
        else:
            self.mlp = HunYuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=False)
        if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
            self.input_layernorm = HunYuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = HunYuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
            self.input_layernorm = norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            kv_states: Optional[Tuple[torch.Tensor]] = None,
            custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
            first_step: Optional[bool] =False,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            kv_states (`Tuple(torch.FloatTensor)`, *optional*): Used when CLA is enabled,
                key and value states from past attention blocks
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
                "`attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, kv_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kv_states=kv_states,
            custom_pos_emb=custom_pos_emb,
            first_step=first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # import pdb; pdb.set_trace()
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (kv_states,)

        return outputs
