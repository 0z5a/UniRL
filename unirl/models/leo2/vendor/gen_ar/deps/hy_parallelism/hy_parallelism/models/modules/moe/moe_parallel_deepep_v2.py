# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================
import loguru
import gc
from dataclasses import dataclass
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from hy_parallelism.tools.profiling import profile_range
from hy_parallelism.parallel_states import get_parallel_state


try:
    import deep_ep
    from deep_ep import ElasticBuffer, EPHandle, EventHandle, EventOverlap
except ImportError as e:
    raise ImportError(
        "deep_ep is required for moe_parallel_deepep. "
        "Install from https://github.com/deepseek-ai/DeepEP."
        "For Taiji server, please use a correct mirror or contact kevinkhwu."
    ) from e


_buffer: Optional[ElasticBuffer] = None
_buffer_hidden: int = 0
_num_comm_sms: int = 0
low_latency: bool = False

USE_DEEPEP_ASYNC_FINISH = os.environ.get(
    "HY_PARALLELISM_DEEPEP_ASYNC_FINISH", "0"
).lower() in ("1", "true", "yes")

NUM_MAX_DISPATCH_TOKENS_PER_RANK = int(
    os.environ.get("HY_PARALLELISM_NUM_MAX_DISPATCH_TOKENS_PER_RANK", str(40*1024))
)

# EPv2 combine JIT requires hidden % (32 * sizeof(int4) / sizeof(bf16)) == 0, i.e. hidden % 256 == 0.
DEEPEP_HIDDEN_ALIGN = 256

def set_low_latency(val: bool):
    global low_latency
    low_latency = val


def _create_event_if_async(async_finish: bool) -> Optional[EventHandle]:
    return EventHandle() if async_finish else None


def _sync_stream_if_async(async_finish: bool, after_event: Optional[EventOverlap]) -> None:
    if async_finish and after_event is not None and after_event.event is not None:
        after_event.current_stream_wait()


@dataclass
class _DispatchState:
    handle: EPHandle
    sort_indices: torch.Tensor
    sorted_scores: torch.Tensor
    num_recv: Union[int, torch.Tensor]
    input_dtype: torch.dtype
    recv_topk_weights: Optional[torch.Tensor] = None
    hidden_dim: int = 0
    padded_hidden: int = 0

    low_latency: bool = False


_state_cache: dict[int, _DispatchState] = {}
_cache_counter: int = 0


def _next_cache_id() -> int:
    global _cache_counter
    _cache_counter += 1
    return _cache_counter


def _assert_num_tokens_within_limit(num_tokens: int) -> None:
    if _buffer is not None and num_tokens > _buffer.num_max_tokens_per_rank:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds "
            f"num_max_tokens_per_rank={_buffer.num_max_tokens_per_rank}"
        )


def _padded_hidden_dim(hidden_dim: int) -> int:
    if hidden_dim % DEEPEP_HIDDEN_ALIGN == 0:
        return hidden_dim
    return ((hidden_dim + DEEPEP_HIDDEN_ALIGN - 1) // DEEPEP_HIDDEN_ALIGN) * DEEPEP_HIDDEN_ALIGN


def _pad_hidden_last_dim(x: torch.Tensor, padded_hidden: int) -> torch.Tensor:
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


def _to_dispatch_x(x: torch.Tensor, padded_hidden: Optional[int] = None) -> torch.Tensor:
    if padded_hidden is None:
        padded_hidden = _padded_hidden_dim(x.size(-1))
    return _pad_hidden_last_dim(x, padded_hidden).to(torch.bfloat16)


def _low_latency_num_recv_tokens(
    handle: EPHandle, num_recv: Optional[Union[int, torch.Tensor]] = None
) -> int:
    if num_recv is not None:
        if isinstance(num_recv, torch.Tensor):
            return int(num_recv.item())
        return num_recv
    return int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())


def _per_expert_recv_counts_tensor(
    handle: EPHandle, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if handle.num_recv_tokens_per_expert_list:
        return torch.tensor(
            handle.num_recv_tokens_per_expert_list,
            device=device,
            dtype=dtype,
        )
    psum = handle.psum_num_recv_tokens_per_expert
    if psum.numel() == 0:
        return torch.empty(0, device=device, dtype=dtype)
    prev = torch.cat([psum.new_zeros(1), psum[:-1]])
    counts = psum - prev
    if counts.device != device or counts.dtype != dtype:
        counts = counts.to(device=device, dtype=dtype)
    return counts


def _decode_dispatch(
    buffer: ElasticBuffer,
    x_comm: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, EPHandle]:
    """DeepEP v2 inference decode dispatch (async comm, full layout each step)."""
    topk_idx_typed = topk_idx.to(deep_ep.topk_idx_t)
    # previous_event = EventHandle()
    previous_event = None
    max_token_per_rank = 4 # 考虑 spec decoding
    assert x_comm.shape[0] <= max_token_per_rank, f"In low latency mode, bs*seqlen {x_comm.shape[0]} must be less than or equal to {max_token_per_rank}"
    recv_x, recv_topk_idx, recv_topk_weights, handle, after_event = buffer.dispatch(
        x_comm,
        topk_idx=topk_idx_typed,
        topk_weights=topk_weights.float(),
        num_experts=num_experts,
        num_max_tokens_per_rank=max_token_per_rank,
        do_cpu_sync=False,
        num_sms=_num_comm_sms,
        previous_event=previous_event,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    _sync_stream_if_async(True, after_event)

    return recv_x, recv_topk_idx, recv_topk_weights.float(), handle


def _prepare_low_latency_combine_x(
    expert_outputs: torch.Tensor,
    padded_hidden: int,
    handle: EPHandle,
) -> torch.Tensor:
    expert_comm = _to_dispatch_x(expert_outputs, padded_hidden)
    worst_case_rows = handle.recv_src_metadata.shape[0]
    if expert_comm.shape[0] > worst_case_rows:
        raise RuntimeError(
            f"low_latency combine: expert_outputs rows {expert_comm.shape[0]} "
            f"> dispatch rows {worst_case_rows}"
        )
    if expert_comm.shape[0] == worst_case_rows:
        return expert_comm.contiguous()
    combine_x = expert_comm.new_zeros((worst_case_rows, padded_hidden))
    combine_x[:expert_comm.shape[0]] = expert_comm
    return combine_x.contiguous()


def _safe_num_sms(buffer: ElasticBuffer, num_experts: int, num_topk: int) -> int:
    theoretical = buffer.get_theoretical_num_sms(num_experts, num_topk)
    if buffer.allow_hybrid_mode:
        # hybrid: num_qps = num_sms*16+1，安全条件 num_sms <= num_allocated_qps-1
        max_sms = buffer.num_allocated_qps - 1   # 17 QP → 最多 16 SM
    else:
        # direct: num_qps = min(num_sms, 9)，安全条件 num_sms <= 8
        max_sms = min(8, buffer.num_allocated_qps - 1)
    num_sms = min(theoretical, max_sms)
    num_sms = max(4, num_sms - num_sms % 2)  # 偶数，最小 4
    return num_sms


def _get_elastic_buffer(
    group: dist.ProcessGroup,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> ElasticBuffer:
    global _buffer, _buffer_hidden, _num_comm_sms
    buffer_kwargs = dict(
        group=group,
        num_max_tokens_per_rank=NUM_MAX_DISPATCH_TOKENS_PER_RANK,
        hidden=hidden,
        num_topk=0,
        # sl_idx=0,
        explicitly_destroy=True,
    )
    if get_parallel_state().ep > 16:
        # ep 32 設 True 會 RuntimeError: NCCL exception (/root/DeepEP/csrc/kernels/backend/nccl.cu:104): 2 (DOCA Error 21)
        # 但係繫裹設 False, ep 通訊耗時會有比較大嘅波動
        # 所以 暫時嘅方案系 True + num_allocated_qps 設 17
        # 並且之前發現繫裹 num_allocated_qps 設 65, 本身系可以 ep 32 run 嘅，但係繫裹重新創建 buffer, 一樣會 DOCA Error 21
        buffer_kwargs.update(dict(
            allow_hybrid_mode=True,
            num_allocated_qps=17,
        ))
    elif get_parallel_state().ep <= 8:
        buffer_kwargs.update(dict(
            allow_hybrid_mode=False,
        ))
    else:
        buffer_kwargs.update(dict(
            allow_hybrid_mode=True,
        ))

    global _buffer, _num_comm_sms
    # Check if we can reuse the existing buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group, NUM_MAX_DISPATCH_TOKENS_PER_RANK, hidden,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    if _buffer is not None:
        _buffer.destroy()
        del _buffer
    gc.collect()
    torch.cuda.empty_cache()
    # Allocate a new buffer with MoE settings
    # NOTES: V2 buffer size consumption is larger than V1
    _buffer = ElasticBuffer(
        **buffer_kwargs,
    )


    # 避免 torch reverse 太多 (实际用不到) 然后 cuda 单独分配的 buffer 又常驻
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f"cuda:{local_rank}")
    total_mem = torch.cuda.get_device_properties(device).total_memory
    used_bytes = getattr(_buffer, 'num_bytes', 0) if _buffer is not None else 0
    buffer = 4 * 1024**3 / total_mem
    fraction = max(0.1, min(0.98, float(total_mem - used_bytes) / total_mem) - buffer)
    if os.getenv('LOCAL_RANK', '0') == '0':
        theoretical_num_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
        loguru.logger.info(f'Creating DeepEP Buffer. {theoretical_num_sms=} {used_bytes/1024**3:.2f}GB. Setting torch memory fraction to {fraction}')
    torch.cuda.set_per_process_memory_fraction(fraction, device)

    # V2 analytically calculates the optimal SM count — no more auto-tuning needed
    # You may also specify `num_sms` manually in dispatch/combine calls to override
    # _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
    # _num_comm_sms = _safe_num_sms(_buffer, num_experts, num_topk)
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
    if os.getenv('LOCAL_RANK', '0') == '0':
        loguru.logger.info(f'Using {_num_comm_sms=} SMs')
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

def _low_latency_num_sorted_tokens(
    handle: EPHandle, num_recv: Union[int, torch.Tensor]
) -> Union[int, torch.Tensor]:
    if handle.topk_idx.shape[-1] > 1:
        return handle.psum_num_recv_tokens_per_expert[-1]
    raise NotImplementedError("top1 deepep is not tested")
    return num_recv


def sort_by_expert_from_indices_low_latency(
    tokens: torch.Tensor, expert_indices: torch.Tensor, scores: torch.Tensor, handle: EPHandle
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens so expert-0 rows come first, then expert-1, etc.

    Uses fixed-shape ``argsort`` + ``index_select`` only (no boolean indexing) so
    dispatch does not sync on the recv count. Padding rows beyond ``num_recv`` and
    invalid ``-1`` experts are mapped to a sentinel and sorted to the tail; combine
    masks them off with GPU-side counts.
    """
    num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
    recv_count = num_recv if num_recv.device == tokens.device else num_recv.to(tokens.device)
    num_local_experts = handle.psum_num_recv_tokens_per_expert.numel()
    sentinel = num_local_experts

    num_tokens = expert_indices.shape[0]
    if expert_indices.shape[1] == 1:
        experts = expert_indices.squeeze(1)
        row_ids = torch.arange(num_tokens, device=tokens.device)
        is_invalid = (experts < 0) | (row_ids >= recv_count)
        sort_key = torch.where(is_invalid, sentinel, experts)
        sort_indices = sort_key.argsort(stable=True)
        return (
            tokens.index_select(0, sort_indices),
            scores.squeeze(1).index_select(0, sort_indices),
            sort_indices,
            num_recv,
        )

    row_ids = torch.arange(num_tokens, device=tokens.device).unsqueeze(1).expand_as(
        expert_indices
    )
    flat_token = row_ids.reshape(-1)
    flat_expert = expert_indices.reshape(-1)
    flat_score = scores.reshape(-1)
    is_invalid = (flat_expert < 0) | (row_ids.reshape(-1) >= recv_count)
    sort_key = torch.where(is_invalid, sentinel, flat_expert)
    perm = sort_key.argsort(stable=True)
    sort_indices = flat_token[perm]
    return (
        tokens.index_select(0, sort_indices),
        flat_score[perm],
        sort_indices,
        num_recv,
    )



def _unpermute_by_expert(
    sorted_tokens: torch.Tensor, sort_indices: torch.Tensor, num_recv: int
) -> torch.Tensor:
    return _unpermute_by_expert_fixed_rows(sorted_tokens, sort_indices, num_recv)


def _unpermute_by_expert_fixed_rows(
    sorted_tokens: torch.Tensor, sort_indices: torch.Tensor, num_rows: int
) -> torch.Tensor:
    hidden = sorted_tokens.shape[-1]
    out = torch.zeros(
        (num_rows, hidden), dtype=sorted_tokens.dtype, device=sorted_tokens.device
    )
    out.scatter_add_(
        0, sort_indices.unsqueeze(1).expand(-1, hidden), sorted_tokens
    )
    return out


class _Dispatch(torch.autograd.Function):
    """fwd: buffer.dispatch, bwd: buffer.combine."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, num_experts, cache_id, group):
        hidden_dim = x.size(-1)
        padded_hidden = _padded_hidden_dim(hidden_dim)
        _assert_num_tokens_within_limit(x.shape[0])
        x_comm = _to_dispatch_x(x, padded_hidden)
        buffer = _get_elastic_buffer(
            group,
            padded_hidden,
            topk_idx.shape[-1],
            num_experts,
        )

        topk_idx = topk_idx.to(deep_ep.topk_idx_t)
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        recv_x, recv_topk_idx, recv_topk_weights, handle, after_event = buffer.dispatch(
            x=x_comm,
            topk_idx=topk_idx,
            topk_weights=topk_weights.float(),
            num_experts=num_experts,
            num_max_tokens_per_rank=NUM_MAX_DISPATCH_TOKENS_PER_RANK,
            num_sms=_num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)

        ctx.handle = handle
        ctx.group = group
        ctx.input_dtype = x.dtype
        ctx.hidden_dim = hidden_dim
        ctx.padded_hidden = padded_hidden

        _state_cache[cache_id] = _DispatchState(
            handle=handle,
            sort_indices=torch.empty(0, dtype=torch.long, device=x.device),
            sorted_scores=torch.empty(0, device=x.device),
            num_recv=recv_x.shape[0],
            input_dtype=x.dtype,
            recv_topk_weights=recv_topk_weights,
            hidden_dim=hidden_dim,
            padded_hidden=padded_hidden,
        )

        num_recv = torch.tensor(
            handle.num_recv_tokens_per_expert_list, dtype=torch.int32, device="cpu"
        )
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_num_recv):
        if grad_recv_x is None:
            return None, None, None, None, None, None
        num_experts = ctx.handle.num_experts
        buffer = _get_elastic_buffer(
            ctx.group,
            ctx.padded_hidden,
            ctx.handle.topk_idx.shape[-1],
            num_experts,
        )
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        grad_x, grad_topk_weights, after_event = buffer.combine(
            x=_to_dispatch_x(grad_recv_x, ctx.padded_hidden),
            handle=ctx.handle,
            topk_weights=(
                grad_recv_topk_weights.float()
                if grad_recv_topk_weights is not None
                else None
            ),
            num_sms=_num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
        return (
            _slice_hidden_last_dim(grad_x, ctx.hidden_dim).to(ctx.input_dtype),
            None,
            (grad_topk_weights.to(ctx.input_dtype) if grad_topk_weights is not None else None),
            None,
            None,
            None,
        )


class _Combine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, handle, group, topk_weights, padded_hidden, hidden_dim):
        buffer = _get_elastic_buffer(
            group,
            padded_hidden,
            handle.topk_idx.shape[-1],
            handle.num_experts,
        )
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        combined, _, after_event = buffer.combine(
            x=_to_dispatch_x(x, padded_hidden),
            handle=handle,
            topk_weights=topk_weights.float() if topk_weights is not None else None,
            num_sms=_num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
        ctx.handle = handle
        ctx.group = group
        ctx.padded_hidden = padded_hidden
        ctx.hidden_dim = hidden_dim
        return _slice_hidden_last_dim(combined, hidden_dim)

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = _get_elastic_buffer(
            ctx.group,
            ctx.padded_hidden,
            ctx.handle.topk_idx.shape[-1],
            ctx.handle.num_experts,
        )
        previous_event = _create_event_if_async(USE_DEEPEP_ASYNC_FINISH)
        grad_x, _, _, _, after_event = buffer.dispatch(
            x=_to_dispatch_x(grad_combined, ctx.padded_hidden),
            handle=ctx.handle,
            num_sms=_num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=USE_DEEPEP_ASYNC_FINISH,
            allocate_on_comm_stream=USE_DEEPEP_ASYNC_FINISH,
        )
        _sync_stream_if_async(USE_DEEPEP_ASYNC_FINISH, after_event)
        return _slice_hidden_last_dim(grad_x, ctx.hidden_dim), None, None, None, None, None


# ======== 以下是 public 接口，和传统 ALL2ALL EP 接口保持一致，但增加了 routing_weights 传入 =================

@torch._dynamo.disable
def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
):
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
        low_latency: inference decode path with async ElasticBuffer dispatch/combine.

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
    padded_hidden = _padded_hidden_dim(hidden_dim)

    _assert_num_tokens_within_limit(hidden_states.shape[0])

    if low_latency:
        buffer = _get_elastic_buffer(
            ep_group,
            padded_hidden,
            topk_idx.shape[-1],
            num_experts,
        )

        with profile_range("DeepEP LowLatency Dispatch", barrier=False, enable_sync=False):
            recv_x, recv_topk_idx, recv_scores, handle = _decode_dispatch(
                buffer,
                _to_dispatch_x(hidden_states, padded_hidden),
                topk_idx,
                topk_weights,
                num_experts,
            )

        recv_x_local = _slice_hidden_last_dim(recv_x, hidden_dim)
        sorted_tokens, sorted_scores, sort_indices, num_recv = (
            sort_by_expert_from_indices_low_latency(
                recv_x_local.to(input_dtype), recv_topk_idx, recv_scores, handle
            )
        )
        dispatched_routing_map = torch.empty(
            0, dtype=torch.bool, device=hidden_states.device
        )
        if num_global_tokens_per_local_expert is not None:
            num_global_tokens_per_local_expert.copy_(
                _per_expert_recv_counts_tensor(
                    handle,
                    num_global_tokens_per_local_expert.device,
                    num_global_tokens_per_local_expert.dtype,
                )
            )

        _state_cache[cache_id] = _DispatchState(
            handle=handle,
            sort_indices=sort_indices,
            sorted_scores=sorted_scores,
            num_recv=num_recv,
            input_dtype=input_dtype,
            hidden_dim=hidden_dim,
            padded_hidden=padded_hidden,
            low_latency=True,
        )
        cache_id_tensor = torch.tensor([cache_id], dtype=torch.int64, device="cpu")
        return sorted_tokens, dispatched_routing_map, cache_id_tensor, org_hidden_states_shape

    profile_deepep = os.getenv('HY_PARALLELISM_ENABLE_PROFILING', '0').lower() in ('1', 'true', 'yes')
    with profile_range("DeepEP Dispatch", barrier=profile_deepep, enable_sync=False):
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert = _Dispatch.apply(
            hidden_states, topk_idx, topk_weights, num_experts, cache_id, ep_group
        )

    recv_x_local = _slice_hidden_last_dim(recv_x, hidden_dim)
    recv_scores = (
        recv_topk_weights.float()
        if recv_topk_weights is not None
        else torch.ones(recv_topk_idx.shape, dtype=torch.float32, device=recv_topk_idx.device)
    )
    sorted_tokens, sorted_scores, sort_indices = sort_by_expert_from_indices(
        recv_x_local.to(input_dtype), recv_topk_idx, recv_scores
    )

    if num_global_tokens_per_local_expert is not None:
        num_global_tokens_per_local_expert.copy_(
            num_recv_per_expert.to(
                device=num_global_tokens_per_local_expert.device,
                dtype=num_global_tokens_per_local_expert.dtype,
            )
        )

    dispatched_routing_map = torch.empty(0, dtype=torch.bool, device=hidden_states.device)

    state = _state_cache[cache_id]
    state.recv_topk_weights = recv_topk_weights
    state.sort_indices = sort_indices
    state.sorted_scores = sorted_scores
    state.num_recv = recv_x.shape[0]

    # cache_id_tensor = torch.tensor([cache_id], dtype=torch.int64, device="cpu")
    cache_id_tensor = cache_id
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

    Routing weights are applied on expert outputs (sorted_scores) before combine,
    matching the v1 DeepEP path and the all2all `unpermute` weighting semantics.
    """
    # cache_id = int(local_input_permutation_mapping.item())
    cache_id = int(local_input_permutation_mapping)
    state = _state_cache.pop(cache_id)

    padded_hidden = state.padded_hidden or _padded_hidden_dim(state.hidden_dim)
    hidden_dim = state.hidden_dim or expert_outputs.size(-1)

    if state.low_latency:
        num_recv = state.num_recv
        num_sorted = _low_latency_num_sorted_tokens(state.handle, num_recv)
        num_topk = state.handle.topk_idx.shape[-1]
        buffer = _get_elastic_buffer(
            ep_group,
            padded_hidden,
            num_topk,
            num_experts,
        )
        if state.sorted_scores.numel() > 0:
            scores = state.sorted_scores.to(expert_outputs.dtype).reshape(-1, 1)
            if scores.shape[0] != expert_outputs.shape[0]:
                scores = scores[:expert_outputs.shape[0]]
            weighted = expert_outputs * scores
        else:
            weighted = expert_outputs
        if isinstance(num_sorted, torch.Tensor):
            num_sorted = num_sorted.to(device=weighted.device)
        valid_rows = torch.arange(weighted.shape[0], device=weighted.device) < num_sorted
        weighted = torch.where(valid_rows.unsqueeze(1), weighted, torch.zeros_like(weighted))
        sort_indices = state.sort_indices
        if sort_indices.shape[0] != weighted.shape[0]:
            sort_indices = sort_indices[:weighted.shape[0]]
        unsorted = _unpermute_by_expert_fixed_rows(
            weighted, sort_indices, state.handle.recv_src_metadata.shape[0]
        )
        combine_x = _prepare_low_latency_combine_x(unsorted, padded_hidden, state.handle)

        with profile_range("DeepEP LowLatency Combine", barrier=False, enable_sync=False):
            # previous_event = EventHandle()
            previous_event = None
            combined, _, after_event = buffer.combine(
                combine_x,
                handle=state.handle,
                topk_weights=None,
                num_sms=_num_comm_sms,
                previous_event=previous_event,
                async_with_compute_stream=False,
                allocate_on_comm_stream=False,
            )
        _sync_stream_if_async(True, after_event)
        combined = _slice_hidden_last_dim(combined, hidden_dim)
        # assert torch.isfinite(combined).all(), "combined is not finite"
        return combined.view(org_hidden_states_shape).to(state.input_dtype)

    profile_deepep = os.getenv('HY_PARALLELISM_ENABLE_PROFILING', '0').lower() in ('1', 'true', 'yes')
    with profile_range("DeepEP Combine", barrier=profile_deepep, enable_sync=False):
        if state.sorted_scores.numel() > 0:
            scores = state.sorted_scores.to(expert_outputs.dtype).reshape(-1, 1)
            weighted = expert_outputs * scores
        else:
            weighted = expert_outputs
        unsorted = _unpermute_by_expert(weighted, state.sort_indices, state.num_recv)
        combined = _Combine.apply(
            unsorted,
            state.handle,
            ep_group,
            None,
            padded_hidden,
            hidden_dim,
        )

    return combined.view(org_hidden_states_shape).to(state.input_dtype)
