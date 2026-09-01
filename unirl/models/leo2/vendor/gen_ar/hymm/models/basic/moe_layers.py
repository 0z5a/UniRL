from functools import partial
from typing import Optional

import os
import loguru
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
try:
    from colt5_attention import topk as maybe_differentiable_topk
except Exception as e:
    print(f"Warning: Colt5 attention is not available: {e}")
    maybe_differentiable_topk = None
from einops import rearrange, reduce

import warnings
from torch.cuda import nvtx
from torch.distributed.tensor import DTensor

from hymm.models.basic.model_config import TransformerConfig
from hymm.core.global_vars import get_parallel_state
from hymm.core.parallel_states import ParallelState

from hy_parallelism.utils import is_recomputing
from hy_parallelism.context_parallel.core import maybe_gather_seq, maybe_scatter_seq

try:
    import flashinfer
except Exception as e:
    flashinfer = None

ASSUME_EVEN_CP_SPLITTING = False


def _requires_cp_seq_gather(
    cp_size: int,
    training: bool,
    need_aux_loss: bool,
    need_capacity_rate: bool,
    need_drop_token: bool,
) -> bool:
    # cp 下三类操作需要注意：
    # 1. aux loss
    # 2. capacity rate
    # 3. drop token
    if cp_size <= 1:
        return False
    if not training and not torch.is_grad_enabled():
        if need_aux_loss or need_capacity_rate or need_drop_token:
            # 非训练模式， 通常无需算 loss
            # 如需统计 capacity rate 和 aux loss， 请在训练模式下进行
            warnings.warn(f'Skiping CP sequence gathering for faster inference. The returned capacity rate and aux loss will be incorrect.')
        return False
    return need_aux_loss or need_capacity_rate or need_drop_token


def swap(weight):
    is_dtensor = isinstance(weight, DTensor)
    local_weight = weight.to_local() if is_dtensor else weight

    *leading_dims, out_ch, in_ch = local_weight.size()
    flipped_weight = (
        local_weight
        .view(*leading_dims, 2, out_ch // 2, in_ch)
        .flip(dims=(len(leading_dims),))
        .view(*leading_dims, out_ch, in_ch)
    )

    if is_dtensor:
        weight.to_local().copy_(flipped_weight)
    else:
        weight.copy_(flipped_weight)

def _maybe_reduce_pi_fi_for_cp_balance_loss(pi, fi, seqlen, assume_even_cp_splitting=False):
    from torch.distributed.nn.functional import all_reduce

    if get_parallel_state().cp_size > 1:
        if assume_even_cp_splitting:
            pi = all_reduce(pi, op=torch.distributed.ReduceOp.AVG, group=get_parallel_state().cp_group)
            fi = all_reduce(fi, op=torch.distributed.ReduceOp.AVG, group=get_parallel_state().cp_group)
        else:
            total_seq_len = torch.tensor(seqlen, dtype=torch.int32, device=pi.device)
            torch.distributed.all_reduce(total_seq_len, op=torch.distributed.ReduceOp.SUM, group=get_parallel_state().cp_group)

            factor = seqlen / total_seq_len
            pi.mul_(factor)
            fi.mul_(factor)

            pi = all_reduce(pi, op=torch.distributed.ReduceOp.SUM, group=get_parallel_state().cp_group) 
            fi = all_reduce(fi, op=torch.distributed.ReduceOp.SUM, group=get_parallel_state().cp_group) 
    return pi, fi


class HunyuanMLP(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            is_shared_mlp: bool = False,
            is_moe: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.ffn_hidden_size = config.ffn_hidden_size

        # For expert
        if is_shared_mlp or is_moe:
            self.ffn_hidden_size = (
                config.moe_ffn_hidden_size
                if isinstance(config.moe_ffn_hidden_size, int)
                else config.moe_ffn_hidden_size[layer_idx % config.num_layers]
            )
            if is_shared_mlp:
                num_shared_expert = (
                    config.moe_mixed_mlp
                    if isinstance(config.moe_mixed_mlp, int)
                    else config.moe_mixed_mlp[layer_idx % config.num_layers]
                )
                self.ffn_hidden_size *= num_shared_expert

        if config.split_gate_and_up:
            self.gate_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias, **factory_kwargs)
            self.up_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias, **factory_kwargs)
        else:
            self.gate_and_up_proj = nn.Linear(
                config.hidden_size, self.ffn_hidden_size * 2, bias=config.mlp_bias, **factory_kwargs
            )
        self.down_proj = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=config.mlp_bias, **factory_kwargs)
        self.act_fn = config.act_class()

    def forward(self, x):
        if self._config.split_gate_and_up:
            up = self.up_proj(x)
            gate = self.gate_proj(x)
        else:
            gate_and_up_proj = self.gate_and_up_proj(x)
            up, gate = gate_and_up_proj.chunk(2, dim=-1)
        out = self.down_proj(up * self.act_fn(gate))
        return out

    @torch.no_grad()
    def swap_gate_and_up_weights(self):
        """
        For gate_and_up layers, the linear weights have two parts: gate and up.
        Torch/FlashInfer use [up, gate] layout, while PTMv2/TE use [gate, up] layout.
        This function swaps the two parts in-place.
        """
        assert not self._config.split_gate_and_up, \
            "swap_gate_and_up_weights only works for combined gate_and_up_proj."
        swap(self.gate_and_up_proj.weight)


class DeepSeekMoEGate(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx]
        )
        self.top_k = (
            config.moe_topk
            if isinstance(config.moe_topk, int)
            else config.moe_topk[layer_idx % config.num_layers]
        )
        # Force the gating layer to be in float32 for stability
        self.wg = nn.Linear(config.hidden_size, self.num_experts, bias=False, **factory_kwargs)
        self.moe_drop_token_enabled = getattr(config, 'moe_drop_token_enabled', False)
        self._moe_compute_capacity_rate = getattr(config, 'moe_compute_capacity_rate', True)

        if config.moe_enable_router_expert_bias:
            # expert_bias: persistent=True, will be loaded from checkpoint, safe on meta device
            self.register_buffer('expert_bias', torch.zeros(self.num_experts, dtype=torch.float32, device=torch.cuda.current_device()))
            self.register_buffer('local_tokens_per_expert', torch.zeros(self.num_experts, dtype=torch.float32, device=torch.cuda.current_device()), persistent=False)
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    @property
    def moe_compute_capacity_rate(self):
        return self._moe_compute_capacity_rate

    def set_moe_compute_capacity_rate(self, value: bool):
        self._moe_compute_capacity_rate = value

    def reset_parameters(self):
        """Initialize router parameters for FSDP meta-parameter materialization."""
        # init.normal_(self.wg.weight, std=self._config.init_std)
        # Also reinitialize expert_bias buffer after to_empty() materialization.
        # The actual values will be overwritten by checkpoint loading if available.
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            self.expert_bias.zero_()
            self.local_tokens_per_expert.zero_()
   

    def forward(self, hidden_states):
        bsz, seqlen, hdim = hidden_states.size()
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1)).float()
        logits = self.wg(hidden_states)
        requires_seq_gather = _requires_cp_seq_gather(
            cp_size=get_parallel_state().cp_size,
            training=self.training,
            need_aux_loss=self._config.moe_aux_loss,
            need_capacity_rate=self.moe_compute_capacity_rate,
            need_drop_token=self.moe_drop_token_enabled,
        )
        if requires_seq_gather:
            logits = einops.rearrange(
                maybe_gather_seq(einops.rearrange(logits, '(b s) k -> b s k', b=bsz)),
                'b s k -> (b s) k',
            )
            seqlen = logits.size(0) // bsz

        if self._config.moe_score_func == "softmax":
            # scores = logits.softmax(dim=1, dtype=torch.float)    # (bsz * seqlen, n_experts)
            scores = F.softmax(logits, dim=1)    # (bsz * seqlen, n_experts)
            topk_weights, topk_idx = torch.topk(scores, self.top_k, dim=-1) # (bsz * seqlen, top_k)
        elif self._config.moe_score_func == "sigmoid":
            # Implement sigmoid score function for expert bias
            scores = torch.sigmoid(logits.float()).type_as(logits)
            if self.expert_bias is not None:
                scores_for_routing = scores + self.expert_bias
                _, topk_idx = torch.topk(scores_for_routing, self.top_k, dim=1)
                topk_weights = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
            else:
                topk_weights, topk_idx = torch.topk(scores, self.top_k, dim=1)
            # Keep aux-loss score definition aligned with ptmv2 implementation: normalized sigmoid scores.
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            raise NotImplementedError(f"MoE score function {self._config.moe_score_func} not implemented.")

        if self.top_k > 1 and self._config.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(topk_weights.dtype).eps)
            topk_weights = topk_weights / denominator
        
        if self._config.routed_scaling_factor:
            topk_weights = topk_weights * self._config.routed_scaling_factor
        
        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self._config.moe_enable_router_expert_bias and self.expert_bias is not None and not is_recomputing():
            with torch.no_grad():
                routing_map = torch.zeros_like(logits).int().scatter(1, topk_idx, 1).bool()
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        if self.training and self._config.moe_aux_loss and torch.is_grad_enabled():
            # Auxiliary loss to encourage balanced expert usage.
            # fi is the fraction of tokens assigned to expert i
            # pi is the fraction of the router probability allocated to expert i
            # (See Switch Transformers paper for details: http://arxiv.org/abs/2101.03961)

            # Align with official implementation under expert_bias:
            # use top-k from aux-loss scores (without expert_bias) for fi statistics.
            if self._config.moe_enable_router_expert_bias and self.expert_bias is not None:
                _, aux_topk_idx = torch.topk(scores, self.top_k, dim=1)
            else:
                aux_topk_idx = topk_idx

            if self._config.moe_seq_aux_loss:
                fi = torch.zeros(bsz, self.num_experts, dtype=torch.float32, device=hidden_states.device)
                fi.scatter_add_(
                    dim=1,
                    index=aux_topk_idx.view(bsz, -1),   # (bsz, seqlen * top_k)
                    src=torch.ones(bsz, seqlen * self.top_k, dtype=torch.float32, device=hidden_states.device)
                ).div_(seqlen * self.top_k / self.num_experts)      # (bsz, n_experts)
                pi = scores.view(bsz, seqlen, self.num_experts).mean(dim=1)     # (bsz, n_experts)
                # pi, fi = _maybe_reduce_pi_fi_for_cp_balance_loss(pi, fi, seqlen, assume_even_cp_splitting=ASSUME_EVEN_CP_SPLITTING)
                balance_loss = (fi * pi).sum(dim=1).mean()
            else:
                fi = F.one_hot(aux_topk_idx.view(-1), num_classes=self.num_experts).float().mean(dim=0)     # (n_experts,)
                fi = fi * self.num_experts
                pi = scores.mean(dim=0)     # (n_experts,)
                # pi, fi = _maybe_reduce_pi_fi_for_cp_balance_loss(pi, fi, seqlen, assume_even_cp_splitting=ASSUME_EVEN_CP_SPLITTING)
                balance_loss = (fi * pi).sum()
        else:
            balance_loss = None

        # Calculate capacity rate
        if self.training and (self.moe_compute_capacity_rate or self.moe_drop_token_enabled):
            with torch.no_grad():

                def cumsum_exclusive(t, dim=-3):
                    dim_pos = dim if dim >= 0 else t.dim() + dim
                    if t.size(dim_pos) == 0:
                        return t.clone()
                    t_perm = t.movedim(dim_pos, -1).contiguous()
                    out = F.pad(t_perm, (1, 0)).cumsum(-1)[..., :-1].movedim(-1, dim_pos)
                    return out

                MIN_EXPERT_CAPACITY = 4

                # Each sequence sends (at most?) expert_capacity positions to each expert.
                # Static expert_capacity dimension is needed for expert batch sizes
                # expert_capacity need times top_k and batch_size
                expert_capacity = bsz * min(seqlen, int(self.top_k * (seqlen * self._config.capacity_factor) / self.num_experts))
                expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
                expert_capacity_f = float(expert_capacity)

                # create masks
                one_hot_gate_indices = F.one_hot(topk_idx.transpose(0, 1), self.num_experts)  # (top_k, bsz * seqlen, n_experts)
                mask = one_hot_gate_indices.float()  # (top_k, bsz * seqlen, n_experts)

                # (top_k, bsz * seqlen, num_gates), cumsum along group_size dim
                mask_cumsum = cumsum_exclusive(mask, dim=-2)

                # capacity
                prev_expert_count = 0.
                for n in range(self.top_k):
                    position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]  # (bsz * seqlen, n_experts)

                    # Remove the elements that don't fit. (batch, sequence, experts)
                    mask[n] *= (position_in_expert < expert_capacity_f).float()  # mask[n] updated, (bsz * seqlen, n_experts)

                    # How many examples in this sequence go to this expert - needed for the next iteration as offset
                    prev_expert_count = reduce(
                        mask[n], '... gs e -> ... 1 e', 'sum'
                    ) + prev_expert_count  # prev_expert_count shape: (1, num_gates)

                if self.moe_drop_token_enabled:
                    mask_flat = reduce(mask, 'k ... gs e -> ... gs k', 'sum')
                    topk_weights = topk_weights * mask_flat

                # Calculate capacity rate
                total_tokens = bsz * seqlen * self.top_k
                routed_tokens = mask.sum()
                capacity_rate = routed_tokens / total_tokens

        else:
            capacity_rate = None

        if requires_seq_gather:
            def recover_cp_region(tensor):
                return einops.rearrange(
                    maybe_scatter_seq(einops.rearrange(tensor, '(b s) k -> b s k', b=bsz)),
                    'b s k -> (b s) k',
                )
            topk_weights, topk_idx = map(recover_cp_region, [topk_weights, topk_idx])

        return topk_weights, topk_idx, balance_loss, capacity_rate


def topkgating(
        logits: torch.Tensor,
        topk: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
        divide_top_k: bool = False,
):
    assert get_parallel_state().cp_size == 1, "This topkgating implementation is not supported in context parallel"
    logits = logits.float()
    gates = F.softmax(logits, dim=1)

    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts ** 2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)
    if divide_top_k:
        l_aux = l_aux / topk

    if drop_tokens:
        expert_capacity = int(max(topk, topk * gates.shape[0] // gates.shape[1]) * capacity_factor)
    else:
        expert_index_flat = expert_index.flatten()
        tokens_per_expert = torch.bincount(expert_index_flat, minlength=num_experts)
        expert_capacity = torch.max(tokens_per_expert).item()

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
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


def topkgating_impl2(
        gate_logits,  # input logis
        top_n,  # top_k
        num_gates,  # number of experts
        threshold,  # threshold for top-n
        capacity_factor,  # capacity factor
        differentiable_topk,  # True or False, use differential topk
        differentiable_topk_fused,  # True or False, wether fused topk
        return_loss,  # True or False, wether return loss
        # straight_through_dispatch_tensor, #TODO True or False, whether use straight through dispatch tensor
        noise_mult=1.,
        noise_gates=False,
        eps=1e-9,
        divide_top_k=False,
        drop_tokens=False,
):
    def log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -log(-log(noise))

    def cumsum_exclusive(t, dim=-3):
        dim_pos = dim if dim >= 0 else t.dim() + dim
        if t.size(dim_pos) == 0:
            return t.clone()
        t_perm = t.movedim(dim_pos, -1).contiguous()
        out = F.pad(t_perm, (1, 0)).cumsum(-1)[..., :-1].movedim(-1, dim_pos)
        return out

    MIN_EXPERT_CAPACITY = 4

    def cast_tuple(el, len=1):
        return el if isinstance(el, tuple) else ((el,) * len)

    # Implements Top1Gating on logits.
    b, group_size, dim = gate_logits.shape

    top_n_minus_1 = top_n - 1
    threshold = cast_tuple(threshold, top_n_minus_1)
    assert len(threshold) == top_n_minus_1
    threshold = torch.tensor([eps, *threshold]).to(gate_logits.device)

    # Each sequence sends (at most?) expert_capacity positions to each expert.
    # Static expert_capacity dimension is needed for expert batch sizes
    # expert_capacity need times top_k and batch_size
    expert_capacity = b * min(group_size, int(top_n * (group_size * capacity_factor) / num_gates))
    expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
    expert_capacity_f = float(expert_capacity)

    maybe_noised_gate_logits = gate_logits  # shape: (b, group_size, num_gates)

    if noise_gates:
        noise = gumbel_noise(maybe_noised_gate_logits)  # shape: (b, group_size, num_gates)
        maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult  # shape: (b, group_size, num_gates)

    # raw_gates = maybe_noised_gate_logits.softmax(dim = -1) # shape: (b, group_size, num_gates)

    # kevinkhwu: float softmax
    raw_gates = maybe_noised_gate_logits.float().softmax(dim=-1)  # shape: (b, group_size, num_gates)
    raw_gates = raw_gates.to(maybe_noised_gate_logits.dtype)

    # find top N experts per position
    topk_func = partial(
        maybe_differentiable_topk,
        non_differentiable=not differentiable_topk,
        fused=differentiable_topk_fused  # use triton fused coordinate descent if possible by default
    )
    topk_return = topk_func(raw_gates, k=top_n)
    gate_indices = topk_return.indices  # shape: (b, group_size, top_n)
    if differentiable_topk:
        # allow for differentiable topk using coordinate descent
        # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention
        gates = topk_return.coor_descent_values  # shape: (b, group_size, top_n)
    else:
        gates = topk_return.values  # shape: (b, group_size, top_n)
    # move the top-n dimension to be first
    gates = rearrange(gates, '... k -> k ...')  # shape: (top_n, b, group_size)
    gate_indices = rearrange(gate_indices, '... k -> k ...')  # shape: (top_n, b, group_size)

    # create masks
    one_hot_gate_indices = F.one_hot(gate_indices, num_gates)  # shape: (top_n, b, group_size, num_gates)
    mask = one_hot_gate_indices.float()  # shape: (top_n, b, group_size, num_gates)
    count_top_k_rate = torch.sum(mask, dim=[0, 1, 2]).detach().cpu() / torch.sum(mask).detach().cpu()

    mask_1 = mask[0]  # shape: (b, group_size, num_gates), needed for balancing loss
    count_top_1_rate = torch.sum(mask_1, dim=[0, 1]).detach().cpu() / torch.sum(mask_1).detach().cpu()

    # normalize top-n gate scores
    denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min=eps)  # shape: (1, b, group_size)
    gates = gates / denom  # shape: (top_n, b, group_size) (normalized)

    # best performing policy was to route to the second expert, with probability of min(1., score / threshold),
    # where score = gate2 / (gate1 + gate2)
    # optimal threshold was ~ 0.2
    # generalized to more than 2 experts

    probs = torch.zeros_like(gates).uniform_(0., 1.)  # shape: (top_n, b, group_size)
    # shape: (top_n, b, group_size), boolean
    should_route = probs < (gates / threshold.clamp(min=eps).view(-1, 1, 1))

    # tokens should always be routed to first expert
    # threshold for first expert already set to very small number, but just in case
    should_route[0, ...] = True  # shape: (top_n, b, group_size), boolean, first slice is True

    # rearrange(should_route.float(), '... -> ... 1') shape: (top_n, b, group_size, 1)
    # shape: (top_n, b, group_size, num_gates), updated based on should_route
    mask *= rearrange(should_route.float(), '... -> ... 1')

    # shape: (top_n, b, group_size, num_gates), cumsum along group_size dim
    mask_cumsum = cumsum_exclusive(mask, dim=-2)

    # capacity
    prev_expert_count = 0.
    for n in range(top_n):
        position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]  # shape: (b, group_size, num_gates)

        # Remove the elements that don't fit. (batch, sequence, experts)
        mask[n] *= (position_in_expert < expert_capacity_f).float() # mask[n] updated, shape: (b, group_size, num_gates)

        # How many examples in this sequence go to this expert - needed for the next iteration as offset
        prev_expert_count = reduce(
            mask[n], '... gs e -> ... 1 e', 'sum'
        ) + prev_expert_count  # prev_expert_count shape: (b, 1, num_gates)

    if drop_tokens:
        mask_flat = reduce(mask, 'k ... gs e -> k ... gs', 'sum') # shape: (top_n, b, group_size) (gs is group_size)
        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903

        gates = gates * mask_flat # shape: (top_n, b, group_size), these are the final combining weights for chosen experts

    # balance losses - (batch, experts)
    # We want to equalize the fraction of the batch assigned to each expert
    if return_loss:
        # ====== Old implementation ======
        # density_1 = reduce(mask_1, '... s e -> ... e', 'mean') # shape: (b, num_gates)
        # density_1_proxy = reduce(raw_gates, '... s e -> ... e', 'mean') # shape: (b, num_gates)
        # balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2) # scalar

        # ====== New implementation ======
        mask_all = reduce(one_hot_gate_indices.float(), 'k ... gs e -> ... gs e', 'sum')

        # Calculate global stats (reduce over batch '...' and seqlen 's'/'gs') to match topkgating logic
        # topkgating calculates stats over the entire flattened input (bsz*seqlen).
        # Previously: density_1 = reduce(mask_all, '... s e -> ... e', 'mean') calculated per-sample stats.

        # We flatten batch and seqlen dimensions to treat them as a single stream of tokens
        density_1 = mask_all.view(-1, num_gates).mean(dim=0)  # shape: (num_gates)
        density_1_proxy = raw_gates.view(-1, num_gates).mean(dim=0)  # shape: (num_gates)

        # density_1, density_1_proxy = _maybe_reduce_pi_fi_for_cp_balance_loss(density_1, density_1_proxy, group_size, assume_even_cp_splitting=ASSUME_EVEN_CP_SPLITTING)

        # Note: topkgating uses: num_experts^2 * mean(density * proxy)
        # mean(density * proxy) = sum(density * proxy) / num_experts
        # So num_experts^2 * mean(...) = num_experts * sum(...)
        balance_loss = (density_1_proxy * density_1).sum() * float(num_gates)  # scalar

        # Hunyuan moe balance loss implementation isn't divided by k. Here we fix it to align with deepseek moe.
        if divide_top_k:
            balance_loss = balance_loss / top_n
    else:
        balance_loss = None

    # calculate the router z-loss proposed in paper
    if return_loss:
        router_z_loss_tmp = torch.logsumexp(gate_logits, dim=-1)  # shape: (b, group_size)
        router_z_loss_tmp = torch.square(router_z_loss_tmp)  # shape: (b, group_size)
        router_z_loss = router_z_loss_tmp.mean()  # scalar
    else:
        router_z_loss = None

    # Calculate capacity rate
    total_tokens = b * group_size * top_n
    routed_tokens = mask.sum()
    capacity_rate = routed_tokens / total_tokens

    topk_idx = gate_indices.permute(1, 2, 0).contiguous()
    topk_weight = gates.permute(1, 2, 0).contiguous()
    return balance_loss, router_z_loss, topk_idx, topk_weight, count_top_1_rate, count_top_k_rate, capacity_rate


class HunyuanTopKGate(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx]
        )
        self.top_k = (
            config.moe_topk
            if isinstance(config.moe_topk, int)
            else config.moe_topk[layer_idx % config.num_layers]
        )
        self.moe_drop_token_enabled = getattr(config, 'moe_drop_token_enabled', False)
        self.moe_compute_capacity_rate = getattr(config, 'moe_compute_capacity_rate', True)

        self.wg = nn.Linear(config.hidden_size, self.num_experts, bias=False, **factory_kwargs)

    def forward(self, hidden_states):
        original_bsz = hidden_states.size(0)
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1)).float()
        logits = self.wg(hidden_states)

        gate_impl = self._config.gate_impl or self._config.moe_impl
        if gate_impl == "ep_moe":
            assert self._config.moe_score_func == "softmax", "Only softmax score function is supported for HunyuanTopKGate"
            requires_seq_gather = _requires_cp_seq_gather(
                cp_size=get_parallel_state().cp_size,
                training=self.training,
                need_aux_loss=self._config.moe_aux_loss,
                need_capacity_rate=self.moe_compute_capacity_rate,
                need_drop_token=self.moe_drop_token_enabled,
            )
            if requires_seq_gather:
                logits = einops.rearrange(
                    maybe_gather_seq(einops.rearrange(logits, '(b s) k -> b s k', b=original_bsz)),
                    'b s k -> (b s) k',
                )
            (
                balance_loss, router_z_loss,
                topk_idx, topk_weight,
                count_top_1_rate, count_top_k_rate, capacity_rate
            ) = topkgating_impl2(logits[None], self.top_k,
                                 num_gates=self.num_experts,
                                 threshold=0,
                                 capacity_factor=self._config.capacity_factor,
                                 differentiable_topk=False,
                                 differentiable_topk_fused=False,
                                 return_loss=self.training,
                                 divide_top_k=True,
                                 drop_tokens=self.moe_drop_token_enabled)
            if requires_seq_gather:
                topk_idx, topk_weight = map(lambda x: einops.rearrange(maybe_scatter_seq(einops.rearrange(x, '1 (b s) k -> b s k', b=original_bsz)), 'b s k -> 1 (b s) k'), [topk_idx, topk_weight])
            gate_output = (
                balance_loss, router_z_loss,
                topk_idx[0], topk_weight[0],
                count_top_1_rate, count_top_k_rate, capacity_rate
            )
        else:
            gate_output = topkgating(logits, self.top_k,
                                     norm_topk_prob=self._config.norm_topk_prob,
                                     routed_scaling_factor=self._config.routed_scaling_factor,
                                     capacity_factor=self._config.capacity_factor,
                                     drop_tokens=self._config.moe_drop_tokens,
                                     divide_top_k=True)

        return gate_output


MOE_GATE_LAYER = {
    "deepseek": DeepSeekMoEGate,
    "hunyuan": HunyuanTopKGate,
    "flashinfer": DeepSeekMoEGate,
    "ep_moe": HunyuanTopKGate,
}


class DeepSeekMoE(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx % config.num_layers]
        )
        self.top_k = (
            config.moe_topk
            if isinstance(config.moe_topk, int)
            else config.moe_topk[layer_idx % config.num_layers]
        )
        gate_impl = config.gate_impl or config.moe_impl
        assert gate_impl in MOE_GATE_LAYER, f"gate_impl `{gate_impl}` not supported"
        self.moe_drop_token_enabled = getattr(config, 'moe_drop_token_enabled', False)
        # Make sure the gate layer is named ending with "gate" for compatibility with FSDP
        self.gate = MOE_GATE_LAYER[gate_impl](config, layer_idx, device=device, dtype=torch.float32)

        self.experts = nn.ModuleList(
            [HunyuanMLP(config, layer_idx, is_moe=True, **factory_kwargs)
             for _ in range(self.num_experts)]
        )
        # Only create shared_mlp if moe_mixed_mlp > 0
        moe_mixed_mlp = (
            config.moe_mixed_mlp
            if isinstance(config.moe_mixed_mlp, int)
            else config.moe_mixed_mlp[layer_idx % config.num_layers]
        )
        if moe_mixed_mlp > 0:
            self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True, **factory_kwargs)
        else:
            self.shared_mlp = None

        # Output container for MoE training loss and monitors
        self._moe_output_container = {}

        self.__post_init__()

    def __post_init__(self, **kwargs):
        pass

    def get_balance_loss(self):
        """ Get MoE auxiliary loss for balancing expert usage.
        Returns:
            aux_loss (torch.Tensor): Auxiliary loss for balancing expert usage.
        """
        return self._moe_output_container.get("balance_loss", None)

    def get_capacity_rate(self):
        """ Get MoE expert capacity rate.
        Returns:
            capacity_rate (torch.Tensor): Expert capacity rate.
        """
        return self._moe_output_container.get("capacity_rate", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Implementation from DeepSeek MoE
        https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/blob/main/modeling_deepseek.py#L375
        """
        bsz, seqlen, hdim = hidden_states.size()
        input_hidden_states = hidden_states

        with torch.autocast('cuda', enabled=False):
            topk_weights, topk_idx, balance_loss, capacity_rate = self.gate(hidden_states)  # (bsz * seqlen, top_k)
        # we cast back to the input dtype
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Flatten for easier indexing
        flat_topk_idx = topk_idx.view(-1)
        hidden_states = hidden_states.view(-1, hdim)    # (bsz * seqlen, hdim)
        hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)  # (bsz * seqlen * k, hdim)

        # Forward through experts
        expert_outputs = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        for i in range(self.num_experts):
            expert_mask = (flat_topk_idx == i)
            selected_inputs = hidden_states[expert_mask]
            expert_output = self.experts[i](selected_inputs)    # compatible with zero tensor
            expert_outputs[expert_mask] = expert_output.to(hidden_states.dtype)

        # Weighted sum of expert outputs and plus shared MLP output
        #   Autocast works here, weighted_outputs will be float32 due to sum op.
        weighted_outputs = (expert_outputs.view(
            bsz * seqlen, self.top_k, hdim) * topk_weights.unsqueeze(-1)).sum(dim=1) # (bsz * seqlen, hdim)
        weighted_outputs = weighted_outputs.to(hidden_states.dtype).view(bsz, seqlen, hdim)
        if self.shared_mlp is not None:
            final_hidden_states = weighted_outputs + self.shared_mlp(input_hidden_states)
        else:
            final_hidden_states = weighted_outputs

        # Store aux loss in output container
        if not is_recomputing():
            self._moe_output_container["balance_loss"] = balance_loss
            self._moe_output_container["capacity_rate"] = capacity_rate

        return final_hidden_states


class HunyuanMoE(DeepSeekMoE):

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.shared_mlp is not None:
            hidden_states_mlp = self.shared_mlp(hidden_states)
        else:
            hidden_states_mlp = torch.zeros_like(hidden_states)
        reshaped_input = hidden_states.reshape(-1, hidden_size) # [bsz*seq_len, hidden_size]

        with torch.autocast('cuda', enabled=False):
            l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
        chunks = dispatched_input.chunk(self.num_experts, dim=0)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs.append(expert(chunk))

        expert_output = torch.cat(expert_outputs, dim=0)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output.float()).to(hidden_states.dtype)

        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)
        output = hidden_states_mlp + combined_output

        # Store aux loss in output container
        if not is_recomputing():
            self._moe_output_container["balance_loss"] = l_moe[0]
            self._moe_output_container["capacity_rate"] = l_moe[1]

        return output.to(hidden_states.dtype)


class FlashInferMoE(DeepSeekMoE):
    def __post_init__(self, **kwargs):
        self.__materialize_and_init_state = None
        self.expert_gate_and_up_weights = None
        self.expert_down_weights = None
        self.expert_rearranged = False
        assert flashinfer is not None, "Package 'flashinfer' is not installed."

    @property
    def _materialize_and_init_state(self):
        return self.__materialize_and_init_state

    @_materialize_and_init_state.setter
    def _materialize_and_init_state(self, value):
        self.__materialize_and_init_state = value
        # After materialization, rearrange expert weights
        self.rearrange_expert_weights()

    def rearrange_expert_weights(self, dtype=None, device=None):
        if self.expert_rearranged:
            return

        if dtype is None:
            dtype = self.experts[0].down_proj.weight.dtype
        if device is None:
            device = self.experts[0].down_proj.weight.device

        gate_and_up_weights = []
        down_weights = []

        for expert in self.experts:
            if self._config.split_gate_and_up:
                gate_and_up_weights.append(torch.cat(
                    (expert.up_proj.weight.data.to(device), expert.gate_proj.weight.data.to(device)),
                    dim=0,
                ))
            else:
                gate_and_up_weights.append(expert.gate_and_up_proj.weight.data.to(device))
            down_weights.append(expert.down_proj.weight.to(device))

        self.expert_gate_and_up_weights = nn.Parameter(
            torch.stack(gate_and_up_weights).contiguous(), requires_grad=False
        )
        self.expert_gate_and_up_weights._name_hymm_engine_ = "expert_gate_and_up_weights"
        self.expert_down_weights = nn.Parameter(
            torch.stack(down_weights).contiguous(), requires_grad=False
        )
        self.expert_down_weights._name_hymm_engine_ = "expert_down_weights"

        # Empty the original weights to save memory
        for expert in self.experts:
            if self._config.split_gate_and_up:
                expert.up_proj.weight.data = torch.empty(0, dtype=dtype, device=device)
                expert.gate_proj.weight.data = torch.empty(0, dtype=dtype, device=device)
            else:
                expert.gate_and_up_proj.weight.data = torch.empty(0, dtype=dtype, device=device)
            expert.down_proj.weight.data = torch.empty(0, dtype=dtype, device=device)

        del self.experts

        self.expert_rearranged = True

    @torch.no_grad()
    def swap_gate_and_up_weights(self):
        """
        For gate_and_up layers, the linear weights have two parts: gate and up.
        Torch/FlashInfer use [up, gate] layout, while PTMv2/TE use [gate, up] layout.
        This function swaps the two parts in-place.
        """
        assert self.expert_rearranged, "Expert weights must be rearranged before swapping."
        swap(self.expert_gate_and_up_weights)
        if self.shared_mlp is not None:
            swap(self.shared_mlp.gate_and_up_proj.weight)

    def forward(self, hidden_states):
        assert not self.training, "FlashInferMoE only supports inference mode."
        # Ensure FlashInfer using the correct GPU device in single-process-multi-GPU setting
        torch.cuda.set_device(hidden_states.device.index)

        bsz, seqlen, hdim = hidden_states.size()
        input_hidden_states = hidden_states

        with nvtx.range("MoE"):
            self.rearrange_expert_weights(hidden_states.dtype, hidden_states.device)

            with torch.autocast('cuda', enabled=False):
                topk_weights, topk_idx, _, _ = self.gate(hidden_states)  # (bsz * seqlen, top_k)

            hidden_states = hidden_states.view(-1, hdim).contiguous()    # (bsz * seqlen, hdim)
            flashinfer_dtype = torch.bfloat16
            weighted_outputs = torch.zeros_like(hidden_states, dtype=flashinfer_dtype, device=hidden_states.device)

            _ = flashinfer.fused_moe.cutlass_fused_moe(  # noqa
                hidden_states.to(flashinfer_dtype),
                topk_idx.to(torch.int).contiguous(),
                topk_weights.to(torch.float).contiguous(),
                self.expert_gate_and_up_weights.data.to(flashinfer_dtype),
                self.expert_down_weights.data.to(flashinfer_dtype),
                flashinfer_dtype,
                output=weighted_outputs,
                quant_scales=None,
            )
        weighted_outputs = weighted_outputs.to(hidden_states.dtype).view(bsz, seqlen, hdim)

        if self.shared_mlp is not None:
            final_hidden_states = weighted_outputs + self.shared_mlp(input_hidden_states)
        else:
            final_hidden_states = weighted_outputs
        return final_hidden_states


class HunyuanFusedExpert(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.weight_format = getattr(config, "ep_moe_weight_format", "ep_moe")
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx % config.num_layers]
        )
        self.ffn_hidden_size = (
            config.moe_ffn_hidden_size
            if isinstance(config.moe_ffn_hidden_size, int)
            else config.moe_ffn_hidden_size[layer_idx % config.num_layers]
        )

        p_state: ParallelState = get_parallel_state()
        self.num_local_experts = self.num_experts // p_state.ep_size

        if self.weight_format == "flashinfer":
            self.expert_gate_and_up_weights = nn.Parameter(
                torch.empty(
                    self.num_local_experts, 2 * self.ffn_hidden_size, config.hidden_size, **factory_kwargs
                ),
            )
            self.expert_down_weights = nn.Parameter(
                torch.empty(
                    self.num_local_experts, config.hidden_size, self.ffn_hidden_size, **factory_kwargs
                ),
            )
        elif self.weight_format == "ep_moe":
            self.gate_proj_weights = nn.Parameter(
                torch.empty(self.num_local_experts, self.ffn_hidden_size, config.hidden_size, **factory_kwargs),
            )
            self.up_proj_weights = nn.Parameter(
                torch.empty(self.num_local_experts, self.ffn_hidden_size, config.hidden_size, **factory_kwargs),
            )
            self.down_proj_weights = nn.Parameter(
                torch.empty(self.num_local_experts, config.hidden_size, self.ffn_hidden_size, **factory_kwargs),
            )
        else:
            raise ValueError(f"Unknown ep_moe_weight_format: {self.weight_format!r}")

    def reset_parameters(self):
        if self.weight_format == "flashinfer":
            init.normal_(self.expert_gate_and_up_weights, std=self._config.init_std)
            init.normal_(self.expert_down_weights, std=self._config.init_std)
        else:
            init.normal_(self.gate_proj_weights, std=self._config.init_std)
            init.normal_(self.up_proj_weights, std=self._config.init_std)
            init.normal_(self.down_proj_weights, std=self._config.init_std)

    def _split_gate_up(self):
        up = self.expert_gate_and_up_weights[:, : self.ffn_hidden_size].contiguous()
        gate = self.expert_gate_and_up_weights[:, self.ffn_hidden_size :].contiguous()
        return gate, up

    @torch.no_grad()
    def swap_gate_and_up_weights(self):
        assert self.weight_format == "flashinfer", (
            "swap_gate_and_up_weights only works for weight_format='flashinfer'"
        )
        swap(self.expert_gate_and_up_weights)

    def forward(self, hidden_states, num_global_sum_tokens_per_local_expert=None, index=None):
        """
        如果给了 index, 就是 forward 单个 epxert
        如果给了 num_global_sum_tokens_per_local_expert, 使用 EPGroupGemm

        之所以要在 forward 里有这两种不同语义的设计，是为了方便 fsdp pre_forward
        """
        # Lazy import to make CI happy
        from hy_parallelism.models.modules.moe.moe_parallel import ep_grouped_gemm

        assert not (num_global_sum_tokens_per_local_expert is not None and index is not None), \
            "index and num_global_sum_tokens_per_local_expert cannot be given at the same time"

        if self.weight_format == "flashinfer":
            # gate_proj_weights, up_proj_weights = self._split_gate_up()
            # down_proj_weights = self.expert_down_weights
            pass
        else:
            gate_proj_weights = self.gate_proj_weights
            up_proj_weights = self.up_proj_weights
            down_proj_weights = self.down_proj_weights

        if num_global_sum_tokens_per_local_expert is not None:
            if self.weight_format == "flashinfer":
                from hy_parallelism.models.modules.moe.moe_parallel import grouped_gemm_cutlass_fused_weights
                final_permute_tokens = grouped_gemm_cutlass_fused_weights(
                    hidden_states.to(torch.bfloat16),
                    num_global_sum_tokens_per_local_expert,
                    self.expert_gate_and_up_weights.to(torch.bfloat16),
                    self.expert_down_weights.to(torch.bfloat16),
                )
            else:
                with torch.autocast('cuda', enabled=False):
                    final_permute_tokens = ep_grouped_gemm(
                        hidden_states.to(torch.bfloat16),
                        num_global_sum_tokens_per_local_expert,
                        gate_proj_weights.to(torch.bfloat16),
                        up_proj_weights.to(torch.bfloat16),
                        down_proj_weights.to(torch.bfloat16),
                    )
            return final_permute_tokens
        else:
            assert self.weight_format != "flashinfer", "flashinfer does not support single expert"
            gate_out = torch.matmul(hidden_states, gate_proj_weights[index].transpose(0, 1))
            up_out = torch.matmul(hidden_states, up_proj_weights[index].transpose(0, 1))
            inter_out = F.silu(gate_out) * up_out
            out = torch.matmul(inter_out, down_proj_weights[index].transpose(0, 1))
            return out

    def extra_repr(self):
        return (
            f"num_local_experts={self.num_local_experts}, num_experts={self.num_experts}, "
            f"ffn_hidden_size={self.ffn_hidden_size}, hidden_size={self._config.hidden_size}, "
            f"weight_format={self.weight_format}"
        )


class ExpertParallelMoE(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,
            layer_idx: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.num_experts = (
            config.num_experts
            if isinstance(config.num_experts, int)
            else config.num_experts[layer_idx % config.num_layers]
        )
        self.top_k = (
            config.moe_topk
            if isinstance(config.moe_topk, int)
            else config.moe_topk[layer_idx % config.num_layers]
        )
        self.gate_impl = config.gate_impl or config.moe_impl
        self.moe_drop_token_enabled = getattr(config, 'moe_drop_token_enabled', False)
        self.enable_deepep = getattr(config, 'moe_enable_deepep', os.environ.get('ENABLE_DEEPEP', '0').lower() in ('1', 'true', 'yes'))
        
        self.enable_comm_overlap = getattr(
            config, 'moe_enable_comm_overlap',
            os.environ.get('ENABLE_MOE_COMM_OVERLAP', '1').lower() in ('1', 'true', 'yes'),
        )
        # lazy init stream
        self._comm_overlap_stream = None


        assert self.gate_impl in MOE_GATE_LAYER, f"gate_impl `{self.gate_impl}` not supported"
        # Make sure the gate layer is named ending with "gate" for compatibility with FSDP
        self.gate = MOE_GATE_LAYER[self.gate_impl](config, layer_idx, device=device, dtype=torch.float32)

        self.fused_expert = config.moe_fused_expert

        if self.fused_expert:
            # assert get_parallel_state().ep_size > 1, "Fused expert can only be used with EP > 1. May be supported in the future."
            self.experts = HunyuanFusedExpert(config, layer_idx, **factory_kwargs)
        else:
            if getattr(config, "ep_moe_weight_format", "ep_moe") == "flashinfer":
                raise ValueError(
                    "Non-fused experts (moe_fused_expert=False) are incompatible with "
                    "ep_moe_weight_format='flashinfer'. "
                    "Use moe_fused_expert=True, or set ep_moe_weight_format='ep_moe'."
                )
            assert get_parallel_state().ep_size == 1, "Non-fused expert can only be used with EP = 1. NO PLAN TO SUPPORT EP+NonFusedExpert."
            p_state: ParallelState = get_parallel_state()
            self.num_local_experts = self.num_experts // p_state.ep_size
            self.experts = nn.ModuleList([
                HunyuanMLP(config, layer_idx, is_moe=True, **factory_kwargs)
                for _ in range(self.num_local_experts)
            ])

        # Only create shared_mlp if moe_mixed_mlp > 0
        moe_mixed_mlp = (
            config.moe_mixed_mlp
            if isinstance(config.moe_mixed_mlp, int)
            else config.moe_mixed_mlp[layer_idx % config.num_layers]
        )
        if moe_mixed_mlp > 0:
            self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True, **factory_kwargs)
        else:
            self.shared_mlp = None

        # Output container for MoE training loss and monitors
        self._moe_output_container = {}

        self.__post_init__()

    def __post_init__(self, **kwargs):
        pass

    @torch.no_grad()
    def swap_gate_and_up_weights(self):
        if self.fused_expert:
            self.experts.swap_gate_and_up_weights()
        if self.shared_mlp is not None:
            self.shared_mlp.swap_gate_and_up_weights()

    def get_balance_loss(self):
        """ Get MoE auxiliary loss for balancing expert usage.
        Returns:
            aux_loss (torch.Tensor): Auxiliary loss for balancing expert usage.
        """
        return self._moe_output_container.get("balance_loss", None)

    def get_capacity_rate(self):
        """ Get MoE expert capacity rate.
        Returns:
            capacity_rate (torch.Tensor): Expert capacity rate.
        """
        return self._moe_output_container.get("capacity_rate", None)


    def forward_expert(self, chunk_hidden_states: torch.Tensor, index: int) -> torch.Tensor:
        if self.fused_expert:
            return self.experts(chunk_hidden_states, index=index)
        return self.experts[index](chunk_hidden_states)

    def forward_non_ep(self, hidden_states):
        # TODO: Test this implementation before using it
        input_hidden_states = hidden_states
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        num_tokens = bsz * seq_len
        device = flat_hidden.device
        router_z_loss = None

        with torch.autocast('cuda', enabled=False):
            if self.gate_impl == "ep_moe":
                (
                    balance_loss, router_z_loss,
                    topk_idx, topk_weights,
                    count_top_1_rate, count_top_k_rate, capacity_rate
                ) = self.gate(hidden_states)
            elif self.gate_impl == "deepseek":
                topk_weights, topk_idx, balance_loss, capacity_rate = self.gate(hidden_states)  # (bsz * seqlen, top_k)
            else:
                raise NotImplementedError(f"Gate impl `{self.gate_impl}` not implemented in ExpertParallelMoE.")

        expert_mask = torch.nn.functional.one_hot(
            topk_idx, num_classes=self.num_experts).permute(2, 1, 0)    # (num_experts, top_k, bsz * seqlen)

        if self.moe_drop_token_enabled:
            expert_mask = expert_mask * topk_weights.ne(0).T.unsqueeze(0)
        
        ENABLE_GROUPED_GEMM = os.getenv('ENABLE_GROUPED_GEMM', '1').lower() in ('true', '1')
        if ENABLE_GROUPED_GEMM:
            from hy_parallelism.models.modules.moe.moe_utils import permute, unpermute, generate_weights_idx, permute_no_sync
            enable_fused_unpermute = not self.moe_drop_token_enabled

            num_global_sum_tokens_per_local_expert = expert_mask.sum(dim=(1, 2)).to(
                torch.device("cpu"), non_blocking=True
            )
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            routing_map = expert_mask.sum(dim=1)
            if self.moe_drop_token_enabled:
                permute_tokens, local_input_permutation_mapping = permute(flat_hidden, routing_map, cast_dtype=torch.bfloat16)
            else:
                permute_tokens, local_input_permutation_mapping = permute_no_sync(flat_hidden, topk_idx, cast_dtype=torch.bfloat16)

            if self.shared_mlp is not None:
                if self.enable_comm_overlap:
                    from hy_parallelism.distributed.communications.utils import run_on_async_stream
                    shared_mlp_output = run_on_async_stream(
                        lambda: self.shared_mlp(input_hidden_states),
                        torch.device('cuda'),
                    )
                else:
                    shared_mlp_output = self.shared_mlp(input_hidden_states)

            event.synchronize()
            expert_outputs = self.experts(permute_tokens, num_global_sum_tokens_per_local_expert)

            topk_idx = topk_idx.reshape(num_tokens, self.top_k)
            topk_weights = topk_weights.reshape(num_tokens, self.top_k).to(input_hidden_states.dtype)
            weights_idx = generate_weights_idx(topk_weights, topk_idx, self.num_experts)

            weighted_outputs = unpermute(
                expert_outputs,
                flat_hidden.shape,
                local_input_permutation_mapping,
                routing_map,
                routing_weights=weights_idx,
                use_fused_unpermute=enable_fused_unpermute,
            ).view(bsz, seq_len, hidden_dim)
        else:
            if self.shared_mlp is not None:
                if self.enable_comm_overlap:
                    from hy_parallelism.distributed.communications.utils import run_on_async_stream
                    shared_mlp_output = run_on_async_stream(
                        lambda: self.shared_mlp(input_hidden_states),
                        torch.device('cuda'),
                    )
                else:
                    shared_mlp_output = self.shared_mlp(input_hidden_states)
            topk_idx = topk_idx.reshape(num_tokens, self.top_k).to(device)
            topk_weights = topk_weights.reshape(num_tokens, self.top_k).to(device)
            topk_weights = topk_weights.to(input_hidden_states.dtype)

            flat_topk_idx = topk_idx.view(-1)
            hs = flat_hidden.repeat_interleave(self.top_k, dim=0)
            y = torch.empty_like(hs, dtype=hs.dtype)
            for i in range(self.num_experts):
                chunk_hidden_states = hs[flat_topk_idx == i]
                if chunk_hidden_states.numel() > 0:
                    y[flat_topk_idx == i] = self.forward_expert(chunk_hidden_states, index=i).to(y.dtype)


            weighted_outputs = (
                y.view(bsz, seq_len, self.top_k, hidden_dim) * topk_weights.view(bsz, seq_len, self.top_k).unsqueeze(-1)
            ).sum(dim=2)
            weighted_outputs = weighted_outputs.to(input_hidden_states.dtype)

        if self.shared_mlp is not None:
            if hasattr(shared_mlp_output, 'wait'):
                shared_mlp_output = shared_mlp_output.wait()
            final_hidden_states = weighted_outputs + shared_mlp_output
        else:
            final_hidden_states = weighted_outputs

        if not is_recomputing():
            if self.gate_impl == "ep_moe" and balance_loss is not None and router_z_loss is not None:
                self._moe_output_container["balance_loss"] = balance_loss + router_z_loss
            else:
                self._moe_output_container["balance_loss"] = balance_loss
            self._moe_output_container["capacity_rate"] = capacity_rate

        return final_hidden_states


    def forward_ep(self, hidden_states):
        # Lazy import to make CI happy
        if self.enable_deepep:
            from hy_parallelism.models.modules.moe.moe_parallel_deepep import (
                preprocess as moe_preprocess,
                token_pre_all2all,
                tokens_post_all2all,
            )
        else:
            from hy_parallelism.models.modules.moe.moe_parallel import (
                preprocess as moe_preprocess,
                token_pre_all2all,
                tokens_post_all2all,
            )

        p_state: ParallelState = get_parallel_state()

        input_hidden_states = hidden_states

        overlap_shared = (
            self.enable_comm_overlap
            and self.shared_mlp is not None
            and hidden_states.is_cuda
        )
        if overlap_shared:
            if self._comm_overlap_stream is None:
                self._comm_overlap_stream = torch.cuda.Stream()
            overlap_stream = self._comm_overlap_stream
            overlap_stream.wait_stream(torch.cuda.current_stream())
            input_hidden_states.record_stream(overlap_stream)
            with torch.cuda.stream(overlap_stream):
                shared_mlp_output = self.shared_mlp(input_hidden_states)
        elif self.shared_mlp is not None:
            shared_mlp_output = self.shared_mlp(input_hidden_states)
        else:
            shared_mlp_output = 0

        with torch.autocast('cuda', enabled=False):
            if self.gate_impl == "ep_moe":
                (
                    balance_loss, router_z_loss,
                    topk_idx, topk_weights,
                    count_top_1_rate, count_top_k_rate, capacity_rate
                ) = self.gate(hidden_states)
            elif self.gate_impl == "deepseek":
                topk_weights, topk_idx, balance_loss, capacity_rate = self.gate(hidden_states)  # (bsz * seqlen, top_k)
            else:
                raise NotImplementedError(f"Gate impl `{self.gate_impl}` not implemented in ExpertParallelMoE.")

        expert_mask = torch.nn.functional.one_hot(
            topk_idx, num_classes=self.num_experts).permute(2, 1, 0)    # (num_experts, top_k, bsz * seqlen)

        if self.moe_drop_token_enabled:
            expert_mask = expert_mask * topk_weights.ne(0).T.unsqueeze(0)

        input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
            moe_preprocess(
                expert_mask=expert_mask,
                num_experts=self.num_experts,
                ep_group=p_state.ep_group,
            )
        )

        permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
            hidden_states=hidden_states,
            expert_mask=expert_mask,
            num_experts=self.num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            ep_group=p_state.ep_group,
            routing_weights=topk_weights,
        )

        if self.fused_expert:
            final_permute_tokens = self.experts(permute_tokens, num_global_sum_tokens_per_local_expert)
        else:
            raise NotImplementedError(
                "Non-fused experts with EP is not supported yet (even thought the implementation below may work). "
                "Using this requires code change of FSDP Engine and Checkpoint Manager."
            )
            cumsum = torch.cat([torch.tensor([0]), num_global_sum_tokens_per_local_expert.cumsum(dim=0)])

            # Loop over all available experts in the model and perform the computation on each expert
            final_permute_tokens = torch.zeros(
                permute_tokens.shape,
                dtype=permute_tokens.dtype,
                device=permute_tokens.device,
            )

            for expert_idx in range(self.num_local_experts):
                start_idx = cumsum[expert_idx]
                end_idx = cumsum[expert_idx + 1]

                current_permute_tokens = permute_tokens[start_idx:end_idx]
                final_permute_tokens[start_idx:end_idx] = \
                    self.experts[expert_idx](current_permute_tokens).to(final_permute_tokens.dtype)

        unpermute_tokens = tokens_post_all2all(
            expert_outputs=final_permute_tokens,
            selected_experts=topk_idx,
            num_experts=self.num_experts,
            input_splits=input_splits,
            output_splits=output_splits,
            num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
            routing_map=routing_map,
            local_input_permutation_mapping=local_input_permutation_mapping,
            org_hidden_states_shape=org_hidden_states_shape,
            routing_weights=topk_weights,
            ep_group=get_parallel_state().ep_group,
        )
        weighted_outputs = unpermute_tokens.to(hidden_states.dtype).view(hidden_states.shape)

        if overlap_shared:
            torch.cuda.current_stream().wait_stream(self._comm_overlap_stream)
            shared_mlp_output.record_stream(torch.cuda.current_stream())

        final_hidden_states = weighted_outputs + shared_mlp_output

        # Store aux loss in output container
        if not is_recomputing():
            self._moe_output_container["balance_loss"] = balance_loss
            self._moe_output_container["capacity_rate"] = capacity_rate

        return final_hidden_states

    def forward(self, hidden_states):
        if get_parallel_state().backend == 'megatron':
            ep_size = get_parallel_state().ep_size
        else:
            from hy_parallelism.parallel_states import get_parallel_state as get_pure_torch_parallel_state
            ep_size = get_pure_torch_parallel_state().ep
        if ep_size > 1:
            return self.forward_ep(hidden_states)
        else:
            return self.forward_non_ep(hidden_states)


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_ffn_hidden_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim, **factory_kwargs))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size), **factory_kwargs))
        self.act_fn = config.act_class()
        self._config = config
    
    # add reset_parameters method for compatibility with FSDP
    def reset_parameters(self):
        init.normal_(self.gate_up_proj, std=self._config.init_std)
        init.normal_(self.down_proj, std=self._config.init_std)

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                gated_output = up * self.act_fn(gate)
                out = gated_output @ self.down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen3VLSparesMoeBlock(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False, **factory_kwargs)
        self.experts = Qwen3VLMoeTextExperts(config, layer_idx, **factory_kwargs)
        # Output container for MoE training loss and monitors
        self._moe_output_container = {}

        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def get_balance_loss(self):
        """ Get MoE auxiliary loss for balancing expert usage.
        Returns:
            aux_loss (torch.Tensor): Auxiliary loss for balancing expert usage.
        """
        return self._moe_output_container.get("balance_loss", None)

    def get_capacity_rate(self):
        """ Get MoE expert capacity rate.
        Returns:
            capacity_rate (torch.Tensor): Expert capacity rate.
        """
        return self._moe_output_container.get("capacity_rate", None)
    
    def get_load_std(self):
        """ Get MoE expert load std.
        Returns:
            load_std (torch.Tensor): Expert load std.
        """
        return self._moe_output_container.get("load_std", None)
    
    def get_expert_utilization(self):
        """ Get MoE expert utilization.
        Returns:
            expert_utilization (torch.Tensor): Expert utilization.
        """
        return self._moe_output_container.get("expert_utilization", None)
    
    def calculate_load_balancing_loss(self, router_logits, router_indices, full_routing_weights, divide_top_k=False):
        """
        计算Switch Transformer风格的负载均衡损失
        # Calculate balance loss: encourage balanced expert usage
        # fi is the fraction of tokens assigned to expert i
        # pi is the fraction of the router probability allocated to expert i
        # (See Switch Transformers paper for details: http://arxiv.org/abs/2101.03961)
        """
        num_tokens = router_logits.shape[0]
        
        # π_i: 每个专家的平均路由概率
        pi = full_routing_weights.mean(dim=0)  # [num_experts]
        
        # f_i: 每个专家被选中的token比例
        # router_indices shape: [num_tokens, top_k]
        router_indices_flat = router_indices.view(-1)  # [num_tokens * top_k]
        
        # 创建one-hot mask
        expert_mask = F.one_hot(
            router_indices_flat, 
            num_classes=self.num_experts
        ).float()  # [num_tokens * top_k, num_experts]
        
        # 统计每个专家被选中的次数
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        
        # 计算频率（注意：分母是num_tokens，不是num_tokens * top_k）
        fi = tokens_per_expert / num_tokens  # [num_experts]
        
        # 负载均衡损失
        balance_loss = self.num_experts * (fi * pi).sum()

        # 专家利用率：有多少专家至少处理了一个token
        expert_utilization = (tokens_per_expert > 0).float().mean()  # 理想值：1.0

        # 负载不均衡度
        load_std = tokens_per_expert.float().std() / tokens_per_expert.float().mean()  # 理想值：0.0

        if divide_top_k:
            balance_loss = balance_loss / self.top_k
        
        return balance_loss, expert_utilization, load_std

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)

        # Calculate full softmax weights for balance loss calculation
        # 始终使用 float32 进行概率计算以保证数值稳定性
        full_routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        # Select top_k
        routing_weights, router_indices = torch.topk(full_routing_weights, self.top_k, dim=-1)
        # 归一化 routing_weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # 将 routing_weights 转换回原始精度
        routing_weights = routing_weights.to(hidden_states.dtype)
        # Use advanced indexing instead of scatter to avoid cudagraphs incompatibility
        # Create batch indices for advanced indexing
        router_weights = torch.zeros_like(router_logits)
        batch_indices = torch.arange(router_logits.size(0), device=router_logits.device, dtype=torch.long).unsqueeze(1).expand(-1, self.top_k)
        router_weights[batch_indices, router_indices] = routing_weights
        
        # Calculate MoE auxiliary losses if needed
        if self.training and self._config.moe_aux_loss:
            # old implementation
            # fi = F.one_hot(router_indices.view(-1), num_classes=self.num_experts).float().mean(dim=0)  # (n_experts,)
            # fi = fi * self.num_experts
            # # Use the full routing weights (softmax of router_logits) for pi calculation
            # pi = full_routing_weights.mean(dim=0)  # (n_experts,) - use full softmax weights before topk
            # balance_loss = (fi * pi).sum()
            # divide_top_k=True: divide the balance loss by top_k to align with deepseek moe
            # Note that the official implementation of qwen3vl does not include division by topk.
            balance_loss, expert_utilization, load_std = self.calculate_load_balancing_loss(
                router_logits, router_indices, full_routing_weights, divide_top_k=True)
            # Store aux loss in output container
            if not is_recomputing():
                self._moe_output_container["balance_loss"] = balance_loss
                self._moe_output_container["capacity_rate"] = torch.tensor(1.0)
                self._moe_output_container["expert_utilization"] = expert_utilization
                self._moe_output_container["load_std"] = load_std
        else:
            self._moe_output_container["balance_loss"] = None
            self._moe_output_container["capacity_rate"] = None
            self._moe_output_container["load_std"] = None
            self._moe_output_container["expert_utilization"] = None
        
        # 恢复 batch 维度传递给 experts (取决于 experts 模块的具体实现，通常需要 [B, S, H])
        hidden_states = hidden_states.view(batch_size, seq_len, self.hidden_size)
        
        # 执行专家计算
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        
        # 确保输出类型一致
        routed_out = routed_out.to(hidden_states.dtype) # TODO: check if this is necessary, it is not required in original transformers
        
        return routed_out
