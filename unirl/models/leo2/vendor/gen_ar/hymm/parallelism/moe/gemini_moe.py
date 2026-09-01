# PYTHONPATH=`pwd` torchrun --nproc_per_node=8 hymm/parallelism/moe/gemini_moe.py
from typing import *
import os

import loguru
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import init_device_mesh
from torch.distributed.tensor import Shard

from transformers.activations import ACT2FN

from hymm.parallelism.parallel_states import get_parallel_state, init_parallel_state


def topkgating(
        logits: Tensor,
        topk: int,
        group_limited_greedy: bool=False,
        n_group: int=None,
        topk_group: int=None,
        norm_topk_prob: bool=False,
        routed_scaling_factor: float=1.0
):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    # logits, gates: [bs*seqlen, num_experts]

    if group_limited_greedy:
        group_shape = list(gates.shape[:-1])+[n_group,gates.shape[-1] // n_group]
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

    expert_capacity = topk * gates.shape[0]
    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    # [b*seqlen, topk]
    # expert_gate 每一个token 最高的 topk 个专家的概率, expert_gate[a][b]: token a 分配到的第 rank b of topk 个专家的概率
    # expert_index 每一个token 最高的 topk 个专家的索引, expert_index[a][b]: token a 分配到的第 rank b of topk 个专家的索引
    expert_mask = F.one_hot(expert_index, num_experts)
    # expert_mask 每一个token， 分配到的专家的one hot



    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts**2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

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
    # expert_mask[i,j]: Token i 分配到的第 rank j of topk 个专家的 one hot

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
    # token_priority: [bsz * seqlen, num_experts]
    dispatch_mask = F.one_hot(token_priority, num_classes=expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


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

    locations1_sc = F.one_hot(locations1_s, num_classes=capacity).float()   # one hot to float
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    exp_counts_capacity = torch.sum(mask1_bk)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0])
    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


class HunYuanMLP(nn.Module):
    """
    使用 SwiGLU 的 MLP
    """
    def __init__(self, config, block_idx=None, is_shared_mlp=False, is_moe=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.hidden_size = config.n_embd
        self.hidden_act = config.hidden_act

        self.intermediate_size = config.intermediate_size
        if is_shared_mlp or is_moe:
            # 如果是 moe 的话，优先用 moe_intermediate_size
            if config.moe_intermediate_size is not None:
                self.intermediate_size = config.moe_intermediate_size if isinstance(config.moe_intermediate_size, int) else config.moe_intermediate_size[block_idx]

            if is_shared_mlp:
                num_shared_expert = config.num_shared_expert if isinstance(config.num_shared_expert, int) else config.num_shared_expert[block_idx]
                self.intermediate_size *= num_shared_expert

        self.act_fn = ACT2FN[config.hidden_act]
        if self.hidden_act == "silu":
            self.intermediate_size *= 2  # SwiGLU
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
            self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=config.mlp_bias, **factory_kwargs)
        elif self.hidden_act == "gelu":
            self.gate_and_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, **factory_kwargs)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, **factory_kwargs)
        else:
            assert False, "other hidden_act are not supported"

    def forward(self, x):
        if self.hidden_act == "silu":
            gate_and_up_proj = self.gate_and_up_proj(x)
            x1, x2 = gate_and_up_proj.chunk(2, dim=-1)
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
    def __init__(self, config, block_idx=None, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.moe_topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[block_idx]
        self.drop_tokens = config.moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = config.moe_random_routing_dropped_token
        num_experts = config.n_expert if isinstance(config.n_expert, int) else config.n_expert[block_idx]

        # gate 必须使用 float32 类型，否则会导致精度问题
        self.wg = nn.Linear(config.n_embd, num_experts, bias=False, dtype=torch.float32, device=device)

        # DeepSeek gating args
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.group_limited_greedy = config.group_limited_greedy
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if self.moe_topk == 1:
            gate_output = top1gating(logits, random_routing_dropped_token=self.random_routing_dropped_token)
        else:
            gate_output = topkgating(logits, self.moe_topk, group_limited_greedy=self.group_limited_greedy, n_group=self.n_group, topk_group=self.topk_group, norm_topk_prob=self.norm_topk_prob, routed_scaling_factor=self.routed_scaling_factor)

        return gate_output

class HunYuanExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, has_bias = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = torch.nn.Parameter(
            torch.empty(num_experts, self.intermediate_size, self.hidden_size),
            requires_grad=True
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(num_experts, self.intermediate_size, self.hidden_size),
            requires_grad=True
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(num_experts, self.hidden_size, self.intermediate_size),
            requires_grad=True
        )
        if has_bias:
            self.gate_bias = torch.nn.Parameter(
                torch.zeros(num_experts, self.intermediate_size),  
                requires_grad=True
            )
            self.up_bias = torch.nn.Parameter(
                torch.zeros(num_experts, self.intermediate_size),  
                requires_grad=True
            )
            self.down_bias = torch.nn.Parameter(
                torch.zeros(num_experts, self.hidden_size),  
                requires_grad=True
            )
        else:
            self.gate_bias = [0 for _ in range(num_experts)]
            self.up_bias = [0 for _ in range(num_experts)]
            self.down_bias = [0 for _ in range(num_experts)]

        def init_param(param):
            import math
            for i in range(param.shape[0]):
                nn.init.kaiming_uniform_(param[i], a=math.sqrt(5), nonlinearity='leaky_relu')

        init_param(self.up_proj)
        init_param(self.gate_proj)
        init_param(self.down_proj)

        self.act_fn = ACT2FN['silu']

    def forward(self, hidden_states, index=None, cumsum=None):
        if cumsum is not None:
            assert get_parallel_state().ep_enabled
            assert hidden_states.dtype in [torch.bfloat16, torch.float16]
            gate_proj_weight = getattr(self, f'gate_proj_{ep_rank}').bfloat16()
            up_proj_weight = getattr(self, f'up_proj_{ep_rank}').bfloat16()
            down_proj_weight = getattr(self, f'down_proj_{ep_rank}').bfloat16()

            final_permute_tokens = EPGroupGemm.apply(
                hidden_states,
                cumsum,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            )
            return final_permute_tokens
        elif index is not None:
            if get_parallel_state().ep_enabled:
                ep_rank = dist.get_rank(group=get_parallel_state().ep_group)
                gate_out = torch.matmul(hidden_states, getattr(self, f'gate_proj')[index].transpose(0, 1)) + getattr(self, f'gate_bias')[index]
                up_out = torch.matmul(hidden_states, getattr(self, f'up_proj')[index].transpose(0, 1)) + getattr(self, f'up_bias')[index]
                inter_out = self.act_fn(gate_out) * up_out
                out = torch.matmul(inter_out, getattr(self, f'down_proj')[index].transpose(0, 1)) + getattr(self, f'down_bias')[index]
            else:
                gate_out = torch.matmul(hidden_states, self.gate_proj[index].transpose(0, 1)) + self.gate_bias[index]
                up_out = torch.matmul(hidden_states, self.up_proj[index].transpose(0, 1)) + self.up_bias[index]
                inter_out = self.act_fn(gate_out) * up_out
                out = torch.matmul(inter_out, self.down_proj[index].transpose(0, 1)) + self.down_bias[index]
            return out
        else:
            raise NotImplementedError("Only support index and cumsum")

class HunYuanMoE(nn.Module):
    def __init__(self, config, block_idx=None, device=None, dtype=None, expert_plan='default'):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.moe_topk = config.moe_topk if isinstance(config.moe_topk, int) else config.moe_topk[block_idx]
        self.num_experts = config.n_expert if isinstance(config.n_expert, int) else config.n_expert[block_idx]
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunYuanMLP(config, block_idx=block_idx, is_shared_mlp=True, **factory_kwargs)
        self.gate = HunYuanTopKGate(config, block_idx=block_idx, **factory_kwargs)
        if expert_plan == 'default':
            self.experts = nn.ModuleList(
                [HunYuanMLP(config, block_idx=block_idx, is_shared_mlp=False, is_moe=True, **factory_kwargs) for _ in range(self.num_experts)]
            )
        else:
            self.experts = HunYuanExpert(config.n_embd, config.intermediate_size, self.num_experts, config.mlp_bias)
        self.expert_plan = expert_plan

    def forward_expert(self, chunk, index):
        if self.expert_plan == 'ep':
            return self.experts(chunk, index=index)
        else:
            return self.experts[index](chunk)


    def forwardep(self, hidden_states):
        assert self.expert_plan == 'ep'

        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states)
        # hidden_states: [b, seq_len, hidden_size]
        # combine_weights: [bsz * seq_len, num_experts, capacity]
        # dispatch_mask: [bsz * seq_len, num_experts, capacity], dispatch_mask[a, i, j] is expert i's j-th slot handles a-th token
        # capacity = topk * bsz * seq_len

        reshaped_input = hidden_states.reshape(-1, hidden_size)

        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
        # dispatched_input[i][j]: 第 i 个专家拿到的第 j 个token


        topk_idx = torch.where(dispatch_mask.sum(-1))[-1].reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        assert topk_idx.shape[-1] == self.moe_topk
        cws = combine_weights.sum(-1)
        topk_weight = cws[cws > 0].reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        assert topk_weight.shape[-1] == self.moe_topk



        combined_output = moe_ep_forward(self.experts, hidden_states, topk_idx, topk_weight, self.num_experts, get_parallel_state().ep, get_parallel_state().ep_group)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output


    def forward2(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states)
        # hidden_states: [b, seq_len, hidden_size]
        # combine_weights: [bsz * seq_len, num_experts, capacity]
        # dispatch_mask: [bsz * seq_len, num_experts, capacity], dispatch_mask[a, i, j] is expert i's j-th slot handles a-th token
        # capacity = topk * bsz * seq_len

        reshaped_input = hidden_states.reshape(-1, hidden_size)

        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
        # dispatched_input[i][j]: 第 i 个专家拿到的第 j 个token


        topk_idx = torch.where(dispatch_mask.sum(-1))[-1].reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        assert topk_idx.shape[-1] == self.moe_topk
        cws = combine_weights.sum(-1)
        topk_weight = cws[cws > 0].reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        assert topk_weight.shape[-1] == self.moe_topk


        flat_topk_idx = topk_idx.view(-1)
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        hidden_states = hidden_states.repeat_interleave(self.config.moe_topk, dim=0)

        y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
        for i in range(self.num_experts):
            chunk_hidden_states = hidden_states[flat_topk_idx == i]
            if chunk_hidden_states.numel() > 0:  # 只在有token时计算
                # y[flat_topk_idx == i] = self.experts[i](chunk_hidden_states).to(y.dtype)
                # y[flat_topk_idx == i] = self.experts(chunk_hidden_states, index=i).to(y.dtype)
                y[flat_topk_idx == i] = self.forward_expert(chunk_hidden_states, index=i).to(y.dtype)


        combined_output = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=2)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output

    def forward1(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states)

        l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states)
        # hidden_states: [b, seq_len, hidden_size]
        # combine_weights: [bsz * seq_len, num_experts, capacity]
        # dispatch_mask: [bsz * seq_len, num_experts, capacity], dispatch_mask[a, i, j] is expert i's j-th slot handles a-th token
        # capacity = topk * bsz * seq_len

        reshaped_input = hidden_states.reshape(-1, hidden_size)

        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)

        chunks = dispatched_input.chunk(self.num_experts, dim=0)
        expert_outputs = []
        # for chunk, expert in zip(chunks, self.experts):
        #     expert_outputs.append(expert(chunk))
        for i, chunk in enumerate(chunks):
            # expert_outputs.append(self.experts(chunk, index=i))
            expert_outputs.append(self.forward_expert(chunk, index=i))

        expert_output = torch.cat(expert_outputs, dim=0)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)
        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output

    def forward(self, hidden_states):
        return self.forward1(hidden_states)

def moe_ep_forward(experts, hidden_states, topk_idx, topk_weight, num_experts, ep_size, ep_group, fused_experts=False):
    """

    experts: <class HunYuanExpert>
    hidden_states: [b, seqlen, hidden_dim]
    topk_idx: [b, seqlen, moe_topk]: topk_idx[b][x][i] token x 分给 rank i of topk 专家的 index (also called expert_index)
    topk_weight: [b, seqlen, moe_topk]: topk_idx[b][x][i] token x 分给 rank i of topk 专家的 weight, 用于 combine


    Equals to:

            flat_topk_idx = topk_idx.view(-1)
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            hidden_states = hidden_states.repeat_interleave(self.config.moe_topk, dim=0)

            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i in range(self.num_experts):
                chunk_hidden_states = hidden_states[flat_topk_idx == i]
                if chunk_hidden_states.numel() > 0:  # 只在有token时计算
                    y[flat_topk_idx == i] = self.experts(chunk_hidden_states, index=i).to(y.dtype)

            combined_output = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=2)
    """

    from hymm.parallelism.moe.moe_parallel import preprocess, token_pre_all2all, tokens_post_all2all

    selected_experts = topk_idx.view(-1, topk_idx.shape[-1])
    topk_weight = topk_weight.view(-1, topk_weight.shape[-1])

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    # expert_mask: [num_experts, moe_topk, bs*seqlen]
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        preprocess(
            expert_mask=expert_mask,
            num_experts=num_experts,
            ep_group=ep_group,
        )
    )

    permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
        hidden_states=hidden_states,
        expert_mask=expert_mask,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
        ep_group=ep_group,
    )

    if fused_experts:
        cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)
        final_permute_tokens = experts(permute_tokens, cumsum=cumsum)
    else:
        cumsum = torch.cat([torch.tensor([0]), num_global_sum_tokens_per_local_expert.cumsum(dim=0)])

        # Loop over all available experts in the model and perform the computation on each expert
        final_permute_tokens = torch.zeros(
            (permute_tokens.shape),
            dtype=permute_tokens.dtype,
            device=permute_tokens.device,
        )

        num_local_experts = num_experts // ep_size

        for expert_idx in range(num_local_experts):
            start_idx = cumsum[expert_idx]
            end_idx = cumsum[expert_idx + 1]

            current_permute_tokens = permute_tokens[start_idx:end_idx]
            final_permute_tokens[start_idx:end_idx] = experts(current_permute_tokens, index=expert_idx).to(final_permute_tokens.dtype)

    unpermute_tokens = tokens_post_all2all(
        expert_outputs=final_permute_tokens,
        selected_experts=selected_experts,
        num_experts=num_experts,
        input_splits=input_splits,
        output_splits=output_splits,
        num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
        routing_map=routing_map,
        local_input_permutation_mapping=local_input_permutation_mapping,
        org_hidden_states_shape=org_hidden_states_shape,
        routing_weights=topk_weight,
        ep_group=get_parallel_state().ep_group,
    )

    # combined_output = unpermute_tokens.to(hidden_states.dtype)
    combined_output = unpermute_tokens.to(hidden_states.dtype).view(hidden_states.shape)
    return combined_output


def apply_ep(model, ep_mesh):

    def set_module_from_path(model: nn.Module, path: str, path_new: str, value: any):
        attrs = path.split(".")
        attrs_new = path_new.split(".")
        if len(attrs) == 1:
            setattr(model, attrs_new[0], value)
            if attrs_new[0] != attrs[0]:
                delattr(model, attrs[0])
        else:
            next_obj = getattr(model, attrs[0])
            set_module_from_path(next_obj, ".".join(attrs[1:]), ".".join(attrs_new[1:]), value)


    ep_fqn_list = []
    ep_fqn_ep_list = []
    ep_local_chunk_list = []
    for fqn, param in model.named_parameters():
        if 'experts' in fqn:
            from torch.distributed.tensor import distribute_tensor
            dtensor = distribute_tensor(
                param.data,
                ep_mesh,
                placements=[Shard(0)],
            )

            local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
            new_fqn = fqn

            ep_fqn_list.append(fqn)
            ep_fqn_ep_list.append(new_fqn)
            ep_local_chunk_list.append(local_chunk)

    for fqn, fqn_ep, local_chunk in zip(ep_fqn_list, ep_fqn_ep_list, ep_local_chunk_list):
        set_module_from_path(model, fqn, fqn_ep, local_chunk)

def main():
    from accelerate.utils import set_seed
    set_seed(0)

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    init_parallel_state(
        dp_replicate=1,  # world_size//8 在单个Node下进行切片
        dp_shard=-1,
        sp=1,
        tp=1,
        pp=1,
        ep=2,
        world_size=world_size,
    )

    parallel_dims = get_parallel_state()
    parallel_dims.build_mesh(device_type='cuda')

    from dataclasses import dataclass
    @dataclass
    class Config():
        # MoE config
        n_expert: Union[int, List[int]] = 0
        n_expert_per_token: int = 0  # this is not used in HunYuanMoE
        hidden_act: Literal["silu", "gelu"] = "silu"
        use_mixed_mlp_moe: bool = False
        moe_intermediate_size: Union[int, List[int]] = None
        num_shared_expert: Union[int, List[int]] = None
        moe_topk: Union[int, List[int]] = None
        moe_drop_tokens: bool = False
        moe_random_routing_dropped_token: bool = False
        routed_scaling_factor: float = 1.0
        norm_topk_prob: bool = False
        n_group: bool = False
        topk_group: bool = False
        group_limited_greedy: bool = False
        n_embd: int = 1024
        intermediate_size: Union[int, List[int]] = None
        mlp_bias: bool = False,



    config = Config(
        moe_topk=4,
        n_embd=1024,
        intermediate_size=1024,
        n_expert=16,
        mlp_bias=False,
    )
    def compare(a, b, *args, **kwargs):
        ret = torch.allclose(a, b, *args, **kwargs)
        if not ret:
            loguru.logger.warning(f'差值最大值 {(a - b).abs().max()}', )
            loguru.logger.warning(f'差值均值 {(a - b).abs().mean()}', )
            loguru.logger.warning(f'相对误差 {((a - b).abs() / a.abs()).mean()}', )
        return ret
    def basic_test(moe):
        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size
        output1 = moe.forward1(input)
        output2 = moe.forward2(input)
        assert compare(output1, output2, atol=1e-6)
        loguru.logger.info(f'basic test pass: {moe.expert_plan}')

    def sd_transform_test():
        from hymm.parallelism.moe import ckpt_utils
        moe = HunYuanMoE(config).cuda()
        moe_ep = HunYuanMoE(config, expert_plan='ep').cuda()
        moe_ep.load_state_dict(ckpt_utils.GeminiDefaultToGeminiEP().transform(moe.state_dict()), strict=True)

        set_seed(parallel_dims.dp_mesh.get_local_rank())
        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size
        assert compare(moe_ep(input), moe(input), atol=1e-6)
        loguru.logger.info('sd transform test pass')


    def fuck():
        moe = HunYuanMoE(config, expert_plan='ep').cuda()
        set_seed(parallel_dims.dp_mesh.get_local_rank())
        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size
        output1 = moe.forward1(input)
        output2 = moe.forward2(input)
        apply_ep(moe, parallel_dims.ep_mesh)


        output3 = moe.forwardep(input)
        assert torch.allclose(output3, output2, atol=1e-6)
        assert torch.allclose(output3, output1, atol=1e-6)
        loguru.logger.info('All close')


    def ep_test1():
        set_seed(0)
        moe_ep = HunYuanMoE(config, expert_plan='ep').cuda()
        set_seed(parallel_dims.dp_mesh.get_local_rank())
        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size
        output1 = moe_ep.forward2(input)
        apply_ep(moe_ep, parallel_dims.ep_mesh)
        output2 = moe_ep.forwardep(input)
        assert compare(output1, output2, atol=1e-6)
        loguru.logger.info('ep test1 pass')

    def ep_test2():
        from hymm.parallelism.moe import ckpt_utils

        set_seed(0)
        moe = HunYuanMoE(config).cuda()
        moe_ep = HunYuanMoE(config, expert_plan='ep').cuda()

        set_seed(parallel_dims.dp_mesh.get_local_rank())
        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size

        output1 = moe.forward2(input)
        moe_ep.load_state_dict(ckpt_utils.GeminiDefaultToGeminiEP().transform(moe.state_dict()), strict=True)
        apply_ep(moe_ep, parallel_dims.ep_mesh)


        output2 = moe_ep.forwardep(input)
        assert compare(output1, output2, atol=1e-6)
        loguru.logger.info('ep test pass')

    def final_test():
        from hymm.parallelism.moe import ckpt_utils
        set_seed(0)
        moe = HunYuanMoE(config).cuda()
        moe_ep = HunYuanMoE(config, expert_plan='ep').cuda()
        moe_ep.load_state_dict(ckpt_utils.GeminiDefaultToGeminiEP().transform(moe.state_dict()), strict=True)
        apply_ep(moe_ep, parallel_dims.ep_mesh)

        set_seed(parallel_dims.dp_mesh.get_local_rank())


        input = torch.randn(2, 7, 1024).cuda()  # batch_size, seq_len, hidden_size
        assert compare(moe.forward1(input), moe_ep.forwardep(input), atol=1e-6)
        loguru.logger.info('Final test pass')


    basic_test(HunYuanMoE(config).cuda())
    basic_test(HunYuanMoE(config, expert_plan='ep').cuda())
    sd_transform_test()
    ep_test1()
    ep_test2()
    final_test()
    loguru.logger.info('All close')


if __name__ == '__main__':
    main()
