import math

import torch
import triton
import triton.language as tl

def get_capacity(
    num_tokens: int, num_experts: int, capacity_factor: float, topk: int, min_capacity=None
) -> int:
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Int: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    
    if capacity * topk >= num_tokens:
        print(f'WARNING: capacity * topk: {capacity * topk} exceeds num_tokens: {num_tokens}!')

    return min(capacity * topk, num_tokens)

@triton.jit
def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    z = tl.reshape(indices, shape)

    mask = tl.arange(0, 2)[None, :, None]

    l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )
    r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(
        x.dtype
    )

    l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
    r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)

    idtype = tl.int32

    il_value = l_value.to(idtype, bitcast=True)
    ir_value = r_value.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    flag1 = tl.where((l_value > r_value) ^ flip, il_value ^ ir_value, tl.zeros_like(ix))
    ret = ix ^ flag1
    flag2 = tl.where((l_value > r_value) ^ flip, l_indice ^ r_indice, tl.zeros_like(ix))
    ind = indices ^ flag2

    return ret.to(x.dtype, bitcast=True), ind


@triton.jit
def _bitonic_merge(x, indices, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    if order == 2:
        shape: tl.constexpr = [n_outer * (2 ** (n_dims - 1 - stage)), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = tl.full(x.shape, value=order, dtype=tl.int32)
    for i in tl.static_range(stage):
        x, indices = _compare_and_swap(x, indices, flip, i + (n_dims - stage), n_dims)
    return x, indices


@triton.jit
def _argsort(x, indices, n_dims: tl.constexpr):
    for i in tl.static_range(1, n_dims + 1):
        x, indices = _bitonic_merge(x, indices, i, 2 if i < n_dims else 1, n_dims)
    return x, indices


# tl.softmax reduce in axis-0, but we want axis-1(-1)
@triton.jit
def _block_softmax(block):
    z = block - tl.max(block, 1)[:, None]
    num = tl.exp(z)
    den = tl.sum(num, 1)
    return tl.fdiv(num, den[:, None])


@triton.jit
def _topk_softmax_softmax_fwd(
    # input
    input_ptr,
    used_token_ptr,
    # output
    probs_ptr,
    topk_map_ptr,
    scores_ptr,
    prob_sum_ptr,
    # const expr
    input_stride_0: tl.constexpr,
    input_stride_1: tl.constexpr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    block_tokens: tl.constexpr,
    topk: tl.constexpr,
    n_dims: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_0 = pid * block_tokens + tl.arange(0, block_tokens)[:, None]
    offset_1 = tl.arange(0, BLOCK_SIZE)[None, :]
    experts_block_off = offset_0 * num_experts + offset_1

    input_block_ptrs = input_ptr + (offset_0 * input_stride_0 + offset_1 * input_stride_1)
    load_mask = offset_0 < num_tokens and offset_1 < num_experts
    # float("-inf") will break argsort, so we use -1e5
    block = tl.load(input_block_ptrs, mask=load_mask, other=-1e5).to(tl.float32)
    used_token = tl.load(used_token_ptr+offset_0, mask=offset_0<num_tokens, other=1)
    scores = _block_softmax(block)

    # topk, but returns row tensor of shape [num_experts]
    indices = tl.broadcast_to(tl.arange(0, BLOCK_SIZE)[None, :], block.shape)
    values, indices = _argsort(scores, indices, n_dims)
    values = tl.where(offset_1 < topk, values, tl.full(values.shape, 0, values.dtype))

    if topk == 1:
        probs = values.to(probs_ptr.dtype.element_ty)
    else:
        den = tl.sum(values, 1)
        sum_offset = pid * block_tokens + tl.arange(0, block_tokens)
        tl.store(prob_sum_ptr+sum_offset, den, mask=sum_offset<num_tokens)
        probs = tl.fdiv(values, den[:, None]).to(probs_ptr.dtype.element_ty)

    scatter_offset = offset_0 * num_experts + indices
    scatter_mask = offset_0 < num_tokens and offset_1 < topk

    tl.store(probs_ptr + experts_block_off, 0, mask=load_mask)
    tl.debug_barrier()
    tl.store(probs_ptr + scatter_offset, probs, mask=scatter_mask)

    tl.store(topk_map_ptr + experts_block_off, 0, mask=load_mask)
    tl.debug_barrier()
    tl.store(topk_map_ptr + scatter_offset, used_token, mask=scatter_mask)

    tl.store(scores_ptr + experts_block_off, scores, mask=load_mask)


@triton.jit
def _reduce_sum(
    # input
    topk_map_ptr,
    # output,
    tokens_per_expert_ptr,
    # const expr
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    reduce_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    # Calculate tokens_per_expert
    topk_map = tl.load(
        topk_map_ptr + pid + offset * num_experts, mask=offset < num_tokens, other=0
    ).to(reduce_dtype)
    tokens_per_expert = tl.sum(topk_map)
    tl.store(
        tokens_per_expert_ptr + pid, tokens_per_expert.to(tokens_per_expert_ptr.dtype.element_ty)
    )


@triton.jit
def _topk_softmax_softmax_bwd(
    # input
    probs_ptr,
    scores_ptr,
    probs_grad_ptr,
    scores_grad_ptr,
    prob_sum_ptr,
    # output
    logits_grad_ptr,
    # const expr
    probs_grad_stride_0: tl.constexpr,
    probs_grad_stride_1: tl.constexpr,
    scores_grad_stride_0: tl.constexpr,
    scores_grad_stride_1: tl.constexpr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    probs = tl.load(probs_ptr + pid * num_experts + offset, mask=offset < num_experts, other=0).to(
        tl.float32
    )
    scores = tl.load(scores_ptr + pid * num_experts + offset, mask=offset < num_experts, other=0)
    probs_grad = tl.load(
        probs_grad_ptr + pid * probs_grad_stride_0 + offset * probs_grad_stride_1,
        mask=offset < num_experts,
        other=0,
    ).to(tl.float32)
    scores_grad = tl.load(
        scores_grad_ptr + pid * scores_grad_stride_0 + offset * scores_grad_stride_1,
        mask=offset < num_experts,
        other=0,
    )

    if topk == 1:
        logits_grad_topk = probs_grad + scores_grad
        logits_grad = scores * (
            logits_grad_topk - tl.expand_dims(tl.sum(logits_grad_topk * scores, axis=0), axis=0)
        )       
    else:
        den = tl.load(prob_sum_ptr+pid)
        middle_grad = probs_grad - tl.expand_dims(tl.sum(probs_grad * probs, axis=0), axis=0)
        logits_grad_topk = tl.fdiv(middle_grad, den[:, None]) + scores_grad
        logits_grad = scores * (
            logits_grad_topk - tl.expand_dims(tl.sum(logits_grad_topk * scores, axis=0), axis=0)
        )


    tl.store(
        logits_grad_ptr + pid * num_experts + offset,
        logits_grad.to(logits_grad_ptr.dtype.element_ty),
        mask=offset < num_experts,
    )


def _dropless_fwd(logits, topk, used_token):
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    block_tokens = 16
    num_blocks = math.ceil(num_tokens / block_tokens)

    device = logits.device
    probs = torch.empty(num_tokens, num_experts, dtype=logits.dtype, device=device)
    scores = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_map = torch.empty(num_tokens, num_experts, dtype=torch.bool, device=device)
    prob_sum = torch.empty(num_tokens, dtype=logits.dtype, device=device)
    n_dims = math.ceil(math.log2(num_experts))

    _topk_softmax_softmax_fwd[(num_blocks,)](
        logits,
        used_token,
        probs,
        topk_map,
        scores,
        prob_sum,
        logits.stride(0),
        logits.stride(1),
        num_tokens,
        num_experts,
        block_tokens,
        topk,
        n_dims,
        triton.next_power_of_2(num_experts),
    )

    tokens_per_expert = torch.empty(num_experts, dtype=torch.int64, device=device)
    _reduce_sum[(num_experts,)](
        topk_map,
        tokens_per_expert,
        num_tokens,
        num_experts,
        tl.int64,
        triton.next_power_of_2(num_tokens),
    )

    return probs, scores, topk_map, prob_sum, tokens_per_expert


def _dropless_bwd(probs, scores, probs_grad, scores_grad, prob_sum, topk):
    num_tokens = probs.shape[0]
    num_experts = probs.shape[1]

    logits_grad = torch.empty(
        num_tokens, num_experts, device=probs_grad.device, dtype=probs_grad.dtype
    )

    _topk_softmax_softmax_bwd[(num_tokens,)](
        probs,
        scores,
        probs_grad,
        scores_grad,
        prob_sum,
        logits_grad,
        probs_grad.stride(0),
        probs_grad.stride(1),
        scores_grad.stride(0),
        scores_grad.stride(1),
        num_experts,
        topk,
        triton.next_power_of_2(num_experts),
    )
    return logits_grad


class FusedDroplessRouter(torch.autograd.Function):
    """Autograd function for FusedDroplessRouter."""

    @staticmethod
    def forward(ctx, logits, topk, used_token):
        """Forward."""
        ctx.topk = topk
        probs, scores, indices, prob_sum, tokens_per_expert = _dropless_fwd(logits, topk, used_token)
        ctx.save_for_backward(probs, scores, prob_sum)
        return probs, scores, indices, tokens_per_expert

    @staticmethod
    def backward(ctx, probs_grad, scores_grad, *_):
        """Backward."""
        probs, scores, prob_sum = ctx.saved_tensors
        logits_grad = _dropless_bwd(probs, scores, probs_grad, scores_grad, prob_sum, ctx.topk)
        return logits_grad, None, None


@triton.jit
def _drop_and_pad_fwd(
    # input
    capacity_indices_ptr,
    topk_map_ptr,
    probs_ptr,
    # output,
    final_probs_ptr,
    tokens_per_expert_ptr,
    # const expr
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    expert_capacity: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    pad_to_capacity: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    # Calculate tokens_per_expert
    topk_map = tl.load(
        topk_map_ptr + pid + offset * num_experts, mask=offset < num_tokens, other=0
    ).to(tl.int64)
    tokens_per_expert = tl.sum(topk_map)
    tl.store(tokens_per_expert_ptr + pid, tokens_per_expert)

    # Calculate capacity_mask
    capacity_indices = tl.load(
        capacity_indices_ptr + pid + offset * num_experts, mask=offset < expert_capacity, other=-1
    )
    tl.debug_barrier()
    # store twice to do scatter
    tl.store(topk_map_ptr + pid + offset * num_experts, 0, mask=offset < num_tokens)
    tl.debug_barrier()
    tl.store(topk_map_ptr + pid + capacity_indices * num_experts, 1, mask=offset < expert_capacity)
    tl.debug_barrier()
    capacity_mask = tl.load(
        topk_map_ptr + pid + offset * num_experts, mask=offset < num_tokens, other=0
    ).to(tl.int1)
    if not pad_to_capacity:
        capacity_mask = capacity_mask and topk_map.to(tl.int1)
        tl.store(topk_map_ptr + pid + offset * num_experts, capacity_mask, mask=offset < num_tokens)

    # Calculate probs
    probs = tl.load(probs_ptr + pid + offset * num_experts, mask=capacity_mask, other=0)
    tl.debug_barrier()
    tl.store(final_probs_ptr + pid + offset * num_experts, probs, mask=offset < num_tokens)


def _token_drop_fwd(logits, used_token, topk, expert_capacity, drop_policy, pad_to_capacity):
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    block_tokens = 16
    num_blocks = math.ceil(num_tokens / block_tokens)

    device = logits.device
    probs = torch.empty(num_tokens, num_experts, dtype=logits.dtype, device=device)
    topk_map = torch.empty(num_tokens, num_experts, dtype=torch.bool, device=device)
    scores = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=device)
    prob_sum = torch.empty(num_tokens, dtype=logits.dtype, device=device)
    n_dims = math.ceil(math.log2(num_experts))

    _topk_softmax_softmax_fwd[(num_blocks,)](
        logits,
        used_token, 
        probs,
        topk_map,
        scores,
        prob_sum,
        logits.stride(0),
        logits.stride(1),
        num_tokens,
        num_experts,
        block_tokens,
        topk,
        n_dims,
        triton.next_power_of_2(num_experts),
    )
    # Triton sort is too slow, use torch topk instead.
    if drop_policy == "probs":
        _, capacity_indices = torch.topk(probs, k=expert_capacity, dim=0, sorted=False)
    elif drop_policy == "position":
        _, capacity_indices = torch.topk(topk_map.int(), k=expert_capacity, dim=0, sorted=False)

    tokens_per_expert = torch.empty(num_experts, dtype=torch.int64, device=device)
    final_probs = torch.empty(num_tokens, num_experts, dtype=logits.dtype, device=device)

    _drop_and_pad_fwd[(num_experts,)](
        capacity_indices,
        topk_map,
        probs,
        final_probs,
        tokens_per_expert,
        num_tokens,
        num_experts,
        expert_capacity,
        triton.next_power_of_2(num_tokens),
        pad_to_capacity,
    )

    return final_probs, scores, topk_map, tokens_per_expert, probs, prob_sum


@triton.jit
def _drop_and_pad_bwd(
    final_probs_grad_ptr,
    final_map_ptr,
    probs_grad_ptr,
    final_probs_stride_0: tl.constexpr,
    final_probs_stride_1: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    probs_grad = tl.load(
        final_probs_grad_ptr + pid * final_probs_stride_0 + offset * final_probs_stride_1,
        mask=offset < num_experts,
        other=0,
    )
    final_map = tl.load(
        final_map_ptr + pid * num_experts + offset, mask=offset < num_experts, other=0
    )

    tl.store(
        probs_grad_ptr + pid * num_experts + offset,
        probs_grad * final_map,
        mask=offset < num_experts,
    )


def _token_drop_bwd(probs, scores, final_probs_grad, scores_grad, final_map, topk, prob_sum):
    num_tokens = probs.shape[0]
    num_experts = probs.shape[1]

    logits_grad = torch.empty(num_tokens, num_experts, device=probs.device, dtype=probs.dtype)
    probs_grad = torch.empty(num_tokens, num_experts, device=probs.device, dtype=probs.dtype)

    _drop_and_pad_bwd[(num_tokens,)](
        final_probs_grad,
        final_map,
        probs_grad,
        final_probs_grad.stride(0),
        final_probs_grad.stride(1),
        num_experts,
        triton.next_power_of_2(num_experts),
    )

    _topk_softmax_softmax_bwd[(num_tokens,)](
        probs,
        scores,
        probs_grad,
        scores_grad,
        prob_sum,
        logits_grad,
        probs_grad.stride(0),
        probs_grad.stride(1),
        scores_grad.stride(0),
        scores_grad.stride(1),
        num_experts,
        topk,
        triton.next_power_of_2(num_experts),
    )
    return logits_grad


class FusedTokenDropRouter(torch.autograd.Function):
    """Autograd function for FusedTokenDropRouter."""

    @staticmethod
    def forward(ctx, logits, used_token, topk, capacity_factor, min_capacity, drop_policy, pad_to_capacity):
        """Forward."""
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]
        expert_capacity = get_capacity(
            num_tokens=num_tokens, num_experts=num_experts, capacity_factor=capacity_factor, topk=topk, min_capacity=min_capacity
        )
        final_probs, scores, final_map, tokens_per_expert, probs, prob_sum = _token_drop_fwd(
            logits, used_token, topk, expert_capacity, drop_policy, pad_to_capacity
        )

        ctx.topk = topk
        ctx.save_for_backward(probs, scores, final_map, prob_sum)
        return final_probs, scores, final_map, tokens_per_expert, expert_capacity * num_experts

    @staticmethod
    def backward(ctx, final_probs_grad, scores_grad, *_):
        """Backward."""
        probs, scores, final_map, prob_sum = ctx.saved_tensors
        logits_grad = _token_drop_bwd(probs, scores, final_probs_grad, scores_grad, final_map, ctx.topk, prob_sum)
        return logits_grad, None, None, None, None, None, None


def fused_aux_loss_router(
    logits: torch.Tensor,
    used_token: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    min_capacity: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    drop_tokens: bool = True
):
    """Fused aux loss router."""
    # if capacity_factor is None:
    if not drop_tokens:
        probs, scores, indices, tokens_per_expert = FusedDroplessRouter.apply(logits, topk, used_token)
        num_out_tokens = torch.sum(tokens_per_expert).item()
    else:
        assert drop_policy in ("probs", "position"), f"Invalid drop_policy: {drop_policy}"
        probs, scores, indices, tokens_per_expert, num_out_tokens = FusedTokenDropRouter.apply(
            logits, used_token, topk, capacity_factor, min_capacity, drop_policy, pad_to_capacity
        )

    return probs, scores, indices, tokens_per_expert, num_out_tokens
