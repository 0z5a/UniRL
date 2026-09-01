import torch
import triton
import triton.language as tl

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
def _aux_loss_fwd(
    # input
    probs_per_expert_ptr,
    tokens_per_expert_ptr,
    # output,
    aux_loss_ptr,
    # const expr
    topk: tl.constexpr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_SIZE)
    # Calculate tokens_per_expert
    probs_per_expert = tl.load(probs_per_expert_ptr + offset, mask=offset < num_experts, other=0)
    tokens_per_expert = tl.load(tokens_per_expert_ptr + offset, mask=offset < num_experts, other=0)
    aux_loss = tl.sum(probs_per_expert * tokens_per_expert) * (
        num_experts / (num_tokens * num_tokens)
    )
    tl.store(aux_loss_ptr, aux_loss)


def _calculate_aux_loss_fwd(
    probs, tokens_per_expert, topk, num_tokens, num_experts
):
    aux_loss = torch.empty([], dtype=probs.dtype, device=probs.device)
    probs_per_expert = torch.empty(num_experts, dtype=probs.dtype, device=probs.device)
    _reduce_sum[(num_experts,)](
        probs,
        probs_per_expert,
        num_tokens,
        num_experts,
        tl.float32,
        triton.next_power_of_2(num_tokens),
    )

    _aux_loss_fwd[(1,)](
        probs_per_expert,
        tokens_per_expert,
        aux_loss,
        topk,
        num_tokens,
        num_experts,
        triton.next_power_of_2(num_experts),
    )
    return aux_loss

@triton.jit
def _aux_loss_bwd(
    # input
    aux_loss_grad_ptr,
    tokens_per_expert_ptr,
    # output
    probs_grad_ptr,
    # const expr
    topk: tl.constexpr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    aux_loss_grad = tl.load(aux_loss_grad_ptr)
    tokens_per_expert = tl.load(tokens_per_expert_ptr + offset, mask=offset < num_experts, other=0)
    probs_grad = (
        aux_loss_grad
        * tokens_per_expert
        * (num_experts / (num_tokens * num_tokens))
    )
    tl.store(probs_grad_ptr + pid * num_experts + offset, probs_grad, mask=offset < num_experts)


def _calculate_aux_loss_bwd(
    aux_loss_grad,
    tokens_per_expert,
    topk,
    num_tokens,
    num_experts,
):
    probs_grad = torch.empty(
        (num_tokens, num_experts), dtype=aux_loss_grad.dtype, device=aux_loss_grad.device
    )

    _aux_loss_bwd[(num_tokens,)](
        aux_loss_grad,
        tokens_per_expert,
        probs_grad,
        topk,
        num_tokens,
        num_experts,
        triton.next_power_of_2(num_experts),
    )
    return probs_grad


class FusedCalculateAuxLoss(torch.autograd.Function):
    """Autograd function for FusedCalculateAuxLoss."""

    @staticmethod
    def forward(ctx, probs, tokens_per_expert, topk):
        """Forward."""

        num_tokens = probs.shape[0]
        num_experts = probs.shape[1]
        aux_loss = _calculate_aux_loss_fwd(
            probs,
            tokens_per_expert,
            topk,
            num_tokens,
            num_experts,
        )
        ctx.save_for_backward(tokens_per_expert)
        ctx.topk = topk
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        return aux_loss

    @staticmethod
    def backward(ctx, aux_loss_grad):
        """Backward."""
        (tokens_per_expert,) = ctx.saved_tensors
        probs_grad = _calculate_aux_loss_bwd(
            aux_loss_grad,
            tokens_per_expert,
            ctx.topk,
            ctx.num_tokens,
            ctx.num_experts,
        )
        return probs_grad, None, None


def fused_calculate_aux_loss(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
):
    """Fused calculate aux loss."""
    return FusedCalculateAuxLoss.apply(
        probs, tokens_per_expert, topk
    )
