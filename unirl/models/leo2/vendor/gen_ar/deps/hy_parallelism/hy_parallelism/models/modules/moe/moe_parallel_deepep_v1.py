# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from hy_parallelism.bing_utils import gather_obj
from hy_parallelism.tools.profiling import profile_range

try:
    from deep_ep import Buffer
    from deep_ep import EventOverlap
    from deep_ep.utils import EventHandle
except ImportError as e:
    raise ImportError(
        "deep_ep is required for moe_parallel_deepep. "
        "Install from https://github.com/deepseek-ai/DeepEP."
        "For Taiji server, please use a correct mirror or contact kevinkhwu."
    ) from e


_buffer: Optional[Buffer] = None
low_latency: bool = False

# Fast path: sort recv_idx directly + fill per-expert counts from num_recv_list.
# Legacy path: build dense multihot routing_map, then permute.
USE_DEEPEP_FAST_EXPERT_SORT = os.environ.get(
    "HY_PARALLELISM_DEEPEP_FAST_EXPERT_SORT", "1"
).lower() in ("1", "true", "yes")

USE_DEEPEP_ASYNC_FINISH = os.environ.get(
    "HY_PARALLELISM_DEEPEP_ASYNC_FINISH", "0"
).lower() in ("1", "true", "yes")

# trmt DeepEP low-latency kernels only support fixed hidden sizes (see launch.cuh).
# LL DeepEP does not support topk > 11...  Yes, eleven
LOW_LATENCY_SUPPORTED_HIDDEN = (2560, 5120, 7168)
NUM_MAX_DISPATCH_TOKENS_PER_RANK = int(os.environ.get("HY_PARALLELISM_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "10240"))


def set_low_latency(val: bool):
    global low_latency
    low_latency = val


def _create_event_if_async(async_finish: bool) -> Optional[EventOverlap]:
    # FIXME: 创建这个会让所有 rank 在 cuda:0 有几百 M 的显存占用
    return EventOverlap(EventHandle()) if async_finish else None


def _sync_stream_if_async(async_finish: bool, after_event: Optional[EventOverlap]) -> None:
    if async_finish and after_event is not None:
        after_event.current_stream_wait()


@dataclass
class _DispatchState:
    handle: tuple
    sort_indices: torch.Tensor
    sorted_scores: torch.Tensor
    num_recv: int
    input_dtype: torch.dtype

    # Below states are only used for low latency mode
    low_latency: bool = False
    topk_idx: Optional[torch.Tensor] = None
    topk_weights: Optional[torch.Tensor] = None
    recv_count: Optional[torch.Tensor] = None
    packed_shape: Optional[torch.Size] = None
    hidden_dim: Optional[int] = None


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
        or _buffer.low_latency_mode
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def _get_low_latency_buffer(
    group: dist.ProcessGroup,
    hidden: int,
    num_experts: int,
    num_max_dispatch_tokens_per_rank: int,
) -> Buffer:
    global _buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts
    )
    num_local_experts = num_experts // group.size()
    assert num_experts % group.size() == 0
    if (
        _buffer is None
        or _buffer.group != group
        or not _buffer.low_latency_mode
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        if _buffer is not None:
            if _buffer.low_latency_mode:
                assert hasattr(_buffer, 'destroy'), "DeepEP version without destroy API can not init low latency buffer twice when EP > 8. EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);"
                _buffer.destroy()
            del _buffer
        _buffer = Buffer(
            group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_local_experts,
        )
        _buffer.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
    return _buffer


def _low_latency_padded_hidden(hidden_dim: int) -> int:
    for padded_hidden in LOW_LATENCY_SUPPORTED_HIDDEN:
        if hidden_dim <= padded_hidden:
            return padded_hidden
    raise ValueError(
        f"hidden_dim={hidden_dim} exceeds max low-latency hidden "
        f"{LOW_LATENCY_SUPPORTED_HIDDEN[-1]}"
    )


def _pad_hidden_last_dim(x: torch.Tensor, padded_hidden: int) -> torch.Tensor:
    """Pad the last dim to a DeepEP-supported hidden size; keep row-major contiguous."""
    hidden_dim = x.size(-1)
    if hidden_dim == padded_hidden:
        return x.contiguous()
    if hidden_dim > padded_hidden:
        return x[..., :padded_hidden].contiguous()
    return torch.nn.functional.pad(x.contiguous(), (0, padded_hidden - hidden_dim))


def _slice_hidden_last_dim(x: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    if x.size(-1) == hidden_dim:
        return x
    return x[..., :hidden_dim].contiguous()


def _flatten_low_latency_dispatch(
    recv_x: torch.Tensor, recv_count: torch.Tensor
) -> torch.Tensor:
    chunks = [
        recv_x[i, : recv_count[i].item()]
        for i in range(recv_x.size(0))
        if recv_count[i].item() > 0
    ]
    if not chunks:
        return recv_x.new_zeros((0, recv_x.size(-1)))
    return torch.cat(chunks, dim=0)


def _pack_low_latency_expert_outputs(
    expert_outputs: torch.Tensor,
    recv_count: torch.Tensor,
    packed_shape: torch.Size,
) -> torch.Tensor:
    packed = torch.zeros(
        packed_shape, dtype=expert_outputs.dtype, device=expert_outputs.device
    )
    offset = 0
    for i in range(recv_count.numel()):
        count = recv_count[i].item()
        if count > 0:
            packed[i, :count] = expert_outputs[offset : offset + count]
            offset += count
    return packed


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
    sorted_tokens = tokens.index_select(0, sort_indices) # slow on cuda for long seq
    sorted_scores = scores.T.contiguous().masked_select(rmap_t)
    return sorted_tokens, sorted_scores, sort_indices


def sort_by_expert_from_indices(
    tokens: torch.Tensor, expert_indices: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens so expert-0 rows come first, then expert-1, etc."""
    num_tokens = expert_indices.shape[0]
    if expert_indices.shape[1] == 1:
        experts = expert_indices.squeeze(1)
        sort_indices = experts.argsort(stable=True)
        return (
            tokens.index_select(0, sort_indices),
            scores.squeeze(1).index_select(0, sort_indices),
            sort_indices,
        )

    mask = expert_indices != -1
    token_ids = torch.arange(num_tokens, device=tokens.device).unsqueeze(1).expand_as(
        expert_indices
    )
    flat_token = token_ids[mask]
    flat_expert = expert_indices[mask]
    flat_score = scores[mask]
    perm = flat_expert.argsort(stable=True)
    sort_indices = flat_token[perm]
    return (
        tokens.index_select(0, sort_indices),
        flat_score[perm],
        sort_indices,
    )


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

        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        recv_x, recv_idx, recv_scores, num_recv_list, handle, after_event = buffer.dispatch(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights.float(),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert_disp,
            previous_event=previous_event,
            async_finish=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)

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
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        grad_x, grad_scores, after_event = buffer.combine(
            x=grad_recv_x.contiguous().bfloat16(), # deepEP requires bf16 in internode combine
            handle=ctx.handle,
            topk_weights=(
                grad_recv_scores.float() if grad_recv_scores is not None else None
            ),
            previous_event=previous_event,
            async_finish=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
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
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        combined, _, after_event = buffer.combine(
            x=x.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
        ctx.handle = handle
        ctx.group = group
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = _get_buffer(ctx.group, _hidden_bytes(grad_combined))
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        grad_x, *_, after_event = buffer.dispatch(
            x=grad_combined.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
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
        low_latency: use DeepEP low-latency IBGDA kernels for decoding inference.
            Hidden dim is zero-padded to the nearest supported size in
            ``LOW_LATENCY_SUPPORTED_HIDDEN`` for communication, then sliced
            back for expert compute and output.
        num_max_dispatch_tokens_per_rank: max tokens per rank for low-latency mode;
            defaults to the current token count when unset.

    Returns:
        (sorted_tokens, dispatched_routing_map, cache_id_tensor, org_shape)
    """
    hidden_dim = hidden_states.size(-1)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.reshape(-1, hidden_dim).contiguous()
    org_hidden_states_shape = hidden_states.shape

    topk_idx = _topk_idx_from_expert_mask(expert_mask)

    if routing_weights is not None:
        topk_weights = routing_weights.reshape(-1, topk_idx.shape[-1]).contiguous()
    else:
        topk_weights = torch.ones_like(topk_idx, dtype=torch.float32)

    cache_id = _next_cache_id()

    if low_latency:
        num_tokens = hidden_states.shape[0]
        # if num_max_dispatch_tokens_per_rank is None:
        #     num_max_dispatch_tokens_per_rank = num_tokens
        #     # num_max_dispatch_tokens_per_rank = min(max(gather_obj(num_max_dispatch_tokens_per_rank, group=ep_group)), 128)
        #     num_max_dispatch_tokens_per_rank = 10240
        assert num_tokens <= NUM_MAX_DISPATCH_TOKENS_PER_RANK, f"num_tokens={num_tokens} > num_max_dispatch_tokens_per_rank={NUM_MAX_DISPATCH_TOKENS_PER_RANK}"

        padded_hidden = _low_latency_padded_hidden(hidden_dim)
        hidden_states_comm = _pad_hidden_last_dim(
            hidden_states.to(torch.bfloat16), padded_hidden
        )

        buffer = _get_low_latency_buffer(
            ep_group,
            # padded_hidden,
            LOW_LATENCY_SUPPORTED_HIDDEN[-1], # create large enough buffer, since trmt deepep does not suuport `destroy` and ll bubber can not recreate
            num_experts, NUM_MAX_DISPATCH_TOKENS_PER_RANK
        )

        with profile_range("DeepEP LowLatency Dispatch", barrier=False, enable_sync=False):
            recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
                hidden_states_comm,
                topk_idx,
                NUM_MAX_DISPATCH_TOKENS_PER_RANK,
                num_experts,
                use_fp8=False,
                async_finish=False,
                return_recv_hook=False,
            )

        sorted_tokens = _slice_hidden_last_dim(
            _flatten_low_latency_dispatch(recv_x, recv_count), hidden_dim
        )
        dispatched_routing_map = torch.empty(
            0, dtype=torch.bool, device=hidden_states.device
        )
        if num_global_tokens_per_local_expert is not None:
            num_global_tokens_per_local_expert.copy_(
                recv_count.to(
                    device=num_global_tokens_per_local_expert.device,
                    dtype=num_global_tokens_per_local_expert.dtype,
                )
            )

        _state_cache[cache_id] = _DispatchState(
            handle=handle,
            sort_indices=torch.empty(0, dtype=torch.long, device=hidden_states.device),
            sorted_scores=torch.empty(0, device=hidden_states.device),
            num_recv=sorted_tokens.shape[0],
            input_dtype=input_dtype,
            low_latency=True,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            recv_count=recv_count,
            packed_shape=recv_x.shape,
            hidden_dim=hidden_dim,
        )
        cache_id_tensor = torch.tensor([cache_id], dtype=torch.int64, device="cpu")
        return sorted_tokens, dispatched_routing_map, cache_id_tensor, org_hidden_states_shape

    with profile_range("DeepEP Dispatch", barrier=False, enable_sync=False):
        recv_x, recv_idx, recv_scores, num_recv_per_expert = _Dispatch.apply(
            hidden_states, topk_idx, topk_weights, num_experts, cache_id, ep_group
        )

    if USE_DEEPEP_FAST_EXPERT_SORT:
        # 为了 preprocess 省一个 all_gather
        # The per-local-expert token counts (group_gemm cumsum) are known only now,
        # after dispatch. Fill the buffer that `preprocess` allocated and the caller
        # forwards as `num_global_tokens_per_local_expert`, so its aliased sibling
        # `num_global_sum_tokens_per_local_expert` is populated without an all_gather.
        if num_global_tokens_per_local_expert is not None:
            num_global_tokens_per_local_expert.copy_(
                num_recv_per_expert.to(
                    device=num_global_tokens_per_local_expert.device,
                    dtype=num_global_tokens_per_local_expert.dtype,
                )
            )

        sorted_tokens, sorted_scores, sort_indices = sort_by_expert_from_indices(
            recv_x, recv_idx, recv_scores
        )
        # DeepEP `tokens_post_all2all` does not read routing_map; keep an empty placeholder
        # for signature parity with the all2all path.
        dispatched_routing_map = torch.empty(0, dtype=torch.bool, device=hidden_states.device)
    else:
        num_local_experts = num_experts // ep_group.size()
        dispatched_routing_map, dispatched_scores = _indices_to_multihot(
            recv_idx, recv_scores, num_local_experts
        )

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

    In low-latency mode, expert outputs are repacked into the dispatch layout
    and weights are applied inside `buffer.low_latency_combine`.
    """
    cache_id = int(local_input_permutation_mapping.item())
    state = _state_cache.pop(cache_id)

    if state.low_latency:
        _, _, num_max_dispatch_tokens_per_rank, padded_hidden, _ = state.handle

        buffer = _get_low_latency_buffer(
            ep_group, padded_hidden, num_experts, num_max_dispatch_tokens_per_rank
        )

        expert_outputs_comm = _pad_hidden_last_dim(
            expert_outputs.contiguous().to(torch.bfloat16), padded_hidden
        )
        packed_outputs = _pack_low_latency_expert_outputs(
            expert_outputs_comm,
            state.recv_count,
            state.packed_shape,
        )

        with profile_range("DeepEP LowLatency Combine", barrier=False, enable_sync=False):
            combined, _, _ = buffer.low_latency_combine(
                packed_outputs,
                state.topk_idx,
                state.topk_weights.float(),
                state.handle,
                async_finish=False,
                return_recv_hook=False,
            )
        combined = _slice_hidden_last_dim(combined, state.hidden_dim)
        return combined.view(org_hidden_states_shape).to(state.input_dtype)

    if state.sorted_scores.numel() > 0:
        scores = state.sorted_scores.to(expert_outputs.dtype).reshape(-1, 1)
        weighted = expert_outputs * scores
    else:
        weighted = expert_outputs

    unsorted = _unpermute_by_expert(weighted, state.sort_indices, state.num_recv)
    with profile_range("DeepEP Combine", barrier=False, enable_sync=False):
        combined = _Combine.apply(unsorted, state.handle, ep_group)

    return combined.view(org_hidden_states_shape).to(state.input_dtype)
