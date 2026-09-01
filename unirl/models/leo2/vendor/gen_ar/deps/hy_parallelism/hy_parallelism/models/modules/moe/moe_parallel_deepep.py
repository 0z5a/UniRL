# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer
except ImportError as e:
    raise ImportError(
        "deep_ep is required for moe_parallel_deepep. "
        "Install from https://github.com/deepseek-ai/DeepEP."
        "For Taiji server, please use a correct mirror or contact kevinkhwu."
    ) from e


_buffer: Optional[Buffer] = None


@dataclass
class _DispatchState:
    handle: tuple
    sort_indices: torch.Tensor
    sorted_scores: torch.Tensor
    num_recv: int
    input_dtype: torch.dtype


_state_cache: dict[int, _DispatchState] = {}
_cache_counter: int = 0


def _next_cache_id() -> int:
    global _cache_counter
    _cache_counter += 1
    return _cache_counter


def _hidden_bytes(x: torch.Tensor) -> int:
    return x.size(-1) * max(x.element_size(), 2)


def _get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for cfg in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer



def _topk_idx_from_expert_mask(expert_mask: torch.Tensor) -> torch.Tensor:
    # [num_experts, moe_topk, num_tokens] -> [num_tokens, moe_topk]
    return expert_mask.permute(2, 1, 0).long().argmax(dim=-1)


def _indices_to_multihot(
    indices: torch.Tensor, scores: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = indices.shape[0]
    routing_map = torch.zeros(
        (num_tokens, num_local_experts), dtype=torch.long, device=indices.device
    )
    weight_map = torch.zeros(
        (num_tokens, num_local_experts), dtype=scores.dtype, device=indices.device
    )
    mask = indices != -1
    valid = indices[mask]
    rows = torch.arange(num_tokens, device=indices.device).repeat_interleave(
        mask.sum(dim=1)
    )
    routing_map[rows, valid] = 1
    weight_map[rows, valid] = scores[mask]
    return routing_map.bool(), weight_map


def _permute_by_expert(
    tokens: torch.Tensor, routing_map: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens so expert-0 rows come first, then expert-1, etc."""
    num_tokens, num_local_experts = routing_map.shape
    rmap_t = routing_map.bool().T.contiguous()
    token_idx = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_local_experts, -1)
    )
    sort_indices = token_idx.masked_select(rmap_t)
    sorted_tokens = tokens.index_select(0, sort_indices)
    sorted_scores = scores.T.contiguous().masked_select(rmap_t)
    return sorted_tokens, sorted_scores, sort_indices


def _unpermute_by_expert(
    sorted_tokens: torch.Tensor, sort_indices: torch.Tensor, num_recv: int
) -> torch.Tensor:
    hidden = sorted_tokens.shape[-1]
    out = torch.zeros(
        (num_recv, hidden), dtype=sorted_tokens.dtype, device=sorted_tokens.device
    )
    out.scatter_add_(
        0, sort_indices.unsqueeze(1).expand(-1, hidden), sorted_tokens
    )
    return out



class _Dispatch(torch.autograd.Function):
    """fwd: buffer.dispatch, bwd: buffer.combine."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, num_experts, cache_id, group):
        buffer = _get_buffer(group, _hidden_bytes(x))

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert_disp,
            is_token_in_rank,
            _layout_event,
        ) = buffer.get_dispatch_layout(topk_idx=topk_idx, num_experts=num_experts)

        recv_x, recv_idx, recv_scores, num_recv_list, handle, _event = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert_disp,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.handle = handle
        ctx.group = group
        ctx.input_dtype = x.dtype

        _state_cache[cache_id] = _DispatchState(
            handle=handle,
            sort_indices=torch.empty(0, dtype=torch.long, device=x.device),
            sorted_scores=torch.empty(0, device=x.device),
            num_recv=recv_x.shape[0],
            input_dtype=x.dtype,
        )

        num_recv = torch.tensor(num_recv_list, dtype=torch.int32, device="cpu")
        return recv_x, recv_idx, recv_scores, num_recv

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_idx, grad_recv_scores, grad_num_recv):
        if grad_recv_x is None:
            return None, None, None, None, None, None
        buffer = _get_buffer(ctx.group, _hidden_bytes(grad_recv_x))
        grad_x, grad_scores, _event = buffer.combine(
            x=grad_recv_x.contiguous().bfloat16(), # deepEP requires bf16 in internode combine
            handle=ctx.handle,
            topk_weights=(
                grad_recv_scores.float() if grad_recv_scores is not None else None
            ),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return (
            grad_x.to(ctx.input_dtype),
            None,
            (grad_scores.to(ctx.input_dtype) if grad_scores is not None else None),
            None,
            None,
            None,
        )


class _Combine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, handle, group):
        buffer = _get_buffer(group, _hidden_bytes(x))
        combined, _, _event = buffer.combine(
            x=x.contiguous(),
            handle=handle,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        ctx.handle = handle
        ctx.group = group
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = _get_buffer(ctx.group, _hidden_bytes(grad_combined))
        grad_x, *_, _event = buffer.dispatch(
            x=grad_combined.contiguous(),
            handle=ctx.handle,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, None


# ======== 以下是 public 接口，和传统 ALL2ALL EP 接口保持一致，但增加了 routing_weights 传入 =================

@torch._dynamo.disable
def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
):
    # In the DeepEP path the token layout is resolved inside `buffer.dispatch`
    # (notify_dispatch), so the standalone all_gather of per-expert token counts
    # that the all2all `preprocess` performs here is redundant. The only value
    # consumed downstream is `num_global_sum_tokens_per_local_expert` (the
    # group_gemm cumsum), which equals the number of tokens dispatched to each
    # local expert and is therefore only known *after* dispatch has run.
    #
    # To keep this function's signature/return count identical to the all2all
    # `preprocess` (so the caller stays unchanged whether or not DeepEP is on), we
    # return an empty counts buffer now and let `token_pre_all2all` fill it
    # in-place: the caller forwards this same object as
    # `num_global_tokens_per_local_expert`, then reads it back as
    # `num_global_sum_tokens_per_local_expert` after dispatch. The two returned
    # slots intentionally alias the same tensor so the in-place fill is visible
    # at the experts call. `input_splits`/`output_splits` are unused by DeepEP.
    num_local_experts = num_experts // ep_group.size()
    num_global_sum_tokens_per_local_expert = torch.empty(
        num_local_experts, dtype=torch.long, device=expert_mask.device
    )
    return (
        None,
        None,
        num_global_sum_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
    )


@torch._dynamo.disable
def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_weights: torch.Tensor = None,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    """DeepEP token dispatch.

    Args:
        routing_weights: [bs*seqlen, moe_topk]. Required for topk > 1;
            defaults to 1 when None.

    Returns:
        (sorted_tokens, dispatched_routing_map, cache_id_tensor, org_shape)
    """
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim).contiguous()
    org_hidden_states_shape = hidden_states.shape

    topk_idx = _topk_idx_from_expert_mask(expert_mask)

    if routing_weights is not None:
        topk_weights = routing_weights.reshape(-1, topk_idx.shape[-1]).contiguous()
    else:
        topk_weights = torch.ones_like(topk_idx, dtype=torch.float32)

    num_local_experts = num_experts // ep_group.size()
    cache_id = _next_cache_id()

    recv_x, recv_idx, recv_scores, _num_recv = _Dispatch.apply(
        hidden_states, topk_idx, topk_weights, num_experts, cache_id, ep_group
    )

    dispatched_routing_map, dispatched_scores = _indices_to_multihot(
        recv_idx, recv_scores, num_local_experts
    )

    # The per-local-expert token counts (group_gemm cumsum) are known only now,
    # after dispatch. Fill the buffer that `preprocess` allocated and the caller
    # forwards as `num_global_tokens_per_local_expert`, so its aliased sibling
    # `num_global_sum_tokens_per_local_expert` is populated without an all_gather.
    if num_global_tokens_per_local_expert is not None:
        num_global_tokens_per_local_expert.copy_(dispatched_routing_map.sum(dim=0))

    sorted_tokens, sorted_scores, sort_indices = _permute_by_expert(
        recv_x, dispatched_routing_map, dispatched_scores
    )

    state = _state_cache[cache_id]
    state.sort_indices = sort_indices
    state.sorted_scores = sorted_scores
    state.num_recv = recv_x.shape[0]

    cache_id_tensor = torch.tensor([cache_id], dtype=torch.int64, device="cpu")
    return sorted_tokens, dispatched_routing_map, cache_id_tensor, org_hidden_states_shape


@torch._dynamo.disable
def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    selected_experts,
    num_experts: int,
    input_splits,
    output_splits,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    routing_weights: torch.Tensor = None,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    """DeepEP token combine.

    Routing weights were already applied during dispatch; we multiply
    expert_outputs by them here, then do an unweighted buffer.combine.
    """
    cache_id = int(local_input_permutation_mapping.item())
    state = _state_cache.pop(cache_id)

    if state.sorted_scores.numel() > 0:
        scores = state.sorted_scores.to(expert_outputs.dtype).reshape(-1, 1)
        weighted = expert_outputs * scores
    else:
        weighted = expert_outputs

    unsorted = _unpermute_by_expert(weighted, state.sort_indices, state.num_recv)
    combined = _Combine.apply(unsorted, state.handle, ep_group)

    return combined.view(org_hidden_states_shape).to(state.input_dtype)
