import contextlib
import os
from typing import Optional

import loguru
import torch
import torch.distributed as dist


from hy_parallelism.bing_utils import gather_obj
from hy_parallelism.common.logging import trace_log
from hy_parallelism.models.modules.moe.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from hy_parallelism.models.modules.moe.moe_utils import generate_weights_idx, permute, sort_chunks_by_idxs, unpermute
from hy_parallelism.training.jvp_utils import jvp_guard, tolist


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(group, input, output_split_sizes, input_split_sizes):

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        group, input, output_split_sizes, input_split_sizes = inputs
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

    @staticmethod
    def jvp(ctx, group_tangent, input_tangent, output_split_sizes_tangent, input_split_sizes_tangent):
        """
        JVP implementation for _AllToAll.
        
        For AllToAll operations, the JVP applies the same all-to-all transformation
        to the tangent vector as was applied to the primal input.
        
        Args:
            ctx: Context from forward pass
            group_tangent: Tangent for group (always None)
            input_tangent: Tangent vector for input tensor
            output_split_sizes_tangent: Tangent for output_split_sizes (always None)
            input_split_sizes_tangent: Tangent for input_split_sizes (always None)
        
        Returns:
            Tangent of the output tensor
        """
        # Only the input tensor has a meaningful tangent
        if input_tangent is None:
            return None
            
        # Apply the same AllToAll transformation to the tangent vector
        return _AllToAll.apply(ctx.group, input_tangent, ctx.output_split_sizes, ctx.input_split_sizes)


@torch._dynamo.disable
def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


def _print_token_dist(counts: list[int], n_bins: int, width: int = 40) -> None:
    import random
    counts = list(counts)
    # random.shuffle(counts)
    bin_size = len(counts) // n_bins
    bins = [sum(counts[i * bin_size:(i + 1) * bin_size]) for i in range(n_bins)]
    max_bin = max(bins) if bins else 1
    eighths = ' ▏▎▍▌▋▊▉'
    print(f'shuffled into {n_bins} bins (bin_size={bin_size}, max={max_bin}):')
    for b, s in enumerate(bins):
        units = int(s / max_bin * width * 8) if max_bin > 0 else 0
        full, rem = divmod(units, 8)
        bar = '█' * full + (eighths[rem] if rem else '')
        print(f'  bin {b:3d}: {s:8d} │{bar}')


@torch._dynamo.disable
def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # [ep_size] represent the number of sum tokens in each rank
    input_splits = tolist(num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1))

    # gather all the number of tokens per expert from all ep ranks
    # [ep_size, num_experts]
    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    # if dist.get_rank() == 0:
    #     import random
    #     tokens_per_expert = num_global_tokens_per_expert.sum(dim=0).tolist()
    #     print(f'{tokens_per_expert=}')
    #     _print_token_dist(tokens_per_expert, 10)


    # [ep_size, num_local_experts]
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    # [ep_size]
    output_splits = tolist(num_global_tokens_per_local_expert.sum(dim=1))

    # [num_local_expert]
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device("cpu"), non_blocking=True
    )

    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        torch.device("cpu"), non_blocking=True
    )

    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


@torch._dynamo.disable
def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_weights: torch.Tensor = None,  # unused here; signature parity with moe_parallel_deepep.token_pre_all2all
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)

    global_permuted_hidden_states = all_to_all(ep_group, local_permuted_hidden_states, output_splits, input_splits)

    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    permute_order = tolist(torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel())
    global_permuted_hidden_states = sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )

    return global_permuted_hidden_states, routing_map, local_input_permutation_mapping, org_hidden_states_shape


@torch._dynamo.disable
def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    selected_experts: int,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    routing_weights: torch.Tensor = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = tolist(torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel())
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)
    del expert_outputs

    weights_idx = None
    if routing_weights is not None:
        # [tokens, experts]
        weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)
    del routing_weights

    unpermute_outputs = unpermute(
        unpermute_outputs,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
        routing_weights = weights_idx
    )
    del weights_idx

    return unpermute_outputs

def _cumsum_to_tokens_per_expert(cumsum: torch.Tensor) -> torch.Tensor:
    tokens_per_expert = torch.empty_like(cumsum)
    if cumsum.numel() == 0:
        return tokens_per_expert.cpu()
    tokens_per_expert[0] = cumsum[0]
    if cumsum.numel() > 1:
        tokens_per_expert[1:] = cumsum[1:] - cumsum[:-1]
    return tokens_per_expert.cpu()


def _ep_grouped_gemm_cutlass(
    permute_tokens,
    tokens_per_expert,
    fc1_1_weight,
    fc1_2_weight,
    fc2_weight,
):
    try:
        import grouped_gemm  # pyright: ignore[reportMissingImports]
    except ImportError:
        grouped_gemm = None
    if grouped_gemm is None:
        raise RuntimeError("CUTLASS grouped GEMM is not available. Please install grouped_gemm via `pip3 install grouped_gemm` or unset "
            "HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM. "
            "pssh -i -P -t 0 -h /root/hosts \"pip3 install grouped_gemm\"")
    # grouped_gemm_installed = grouped_gemm is not None
    # if not all(gather_obj(grouped_gemm_installed)):
    #     if os.environ.get('LOCAL_RANK', '0') == '0':
    #         os.system("pip3 install grouped_gemm")
    #     dist.barrier()
    #     try:
    #         import grouped_gemm
    #     except ImportError as e:
    #         raise RuntimeError(
    #             "CUTLASS grouped GEMM is not available. Please install grouped_gemm via `pip3 install grouped_gemm` or unset "
    #             "HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM. "
    #             "pssh -i -P -t 0 -h /root/hosts \"pip3 install grouped_gemm\""
    #         ) from e

    tokens_per_expert = tokens_per_expert.cpu() # grouped_gemm requires `batch_sizes.is_cpu()` to be true
    gg_ops = grouped_gemm.ops

    fc1_1_output = gg_ops.gmm(
        permute_tokens, fc1_1_weight, tokens_per_expert, trans_b=True,
    )
    fc1_2_output = gg_ops.gmm(
        permute_tokens, fc1_2_weight, tokens_per_expert, trans_b=True,
    )
    fc1_output = torch.ops.aten.silu(fc1_1_output) * fc1_2_output
    return gg_ops.gmm(
        fc1_output, fc2_weight, tokens_per_expert, trans_b=True,
    )


def grouped_gemm_cutlass_fused_weights(
    permute_tokens,
    tokens_per_expert,
    fc1_weight, # [up, gate]
    fc2_weight,
):
    try:
        import grouped_gemm  # pyright: ignore[reportMissingImports]
    except ImportError:
        grouped_gemm = None
    if grouped_gemm is None:
        raise RuntimeError(
            "CUTLASS grouped GEMM is not available. Please install grouped_gemm via "
            "`pip3 install grouped_gemm` or unset HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM. "
            "pssh -i -P -t 0 -h /root/hosts \"pip3 install grouped_gemm\""
        )

    tokens_per_expert = tokens_per_expert.cpu()  # grouped_gemm requires `batch_sizes.is_cpu()`
    gg_ops = grouped_gemm.ops

    fc1_output = gg_ops.gmm(
        permute_tokens, fc1_weight, tokens_per_expert, trans_b=True,
    )
    fc1_2_output, fc1_1_output = fc1_output.chunk(2, dim=-1)
    intermediate = torch.ops.aten.silu(fc1_1_output) * fc1_2_output
    return gg_ops.gmm(
        intermediate, fc2_weight, tokens_per_expert, trans_b=True,
    )


def ep_grouped_gemm(
    permute_tokens,
    tokens_per_expert,
    fc1_1_weight,
    fc1_2_weight,
    fc2_weight,
    d2h_event: Optional[torch.cuda.Event] = None,
):
    use_cutlass_env = os.environ.get("HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM")
    use_cutlass = use_cutlass_env is None or use_cutlass_env.lower() in ("1", "true", "yes")
    if use_cutlass:
        try:
            # 避免後續 cuda kernel 中 cpu 想讀取時未 ready
            if d2h_event is not None:
                d2h_event.synchronize()
            # else:
            #     torch.cuda.synchronize()
            return _ep_grouped_gemm_cutlass(
                permute_tokens, tokens_per_expert, fc1_1_weight, fc1_2_weight, fc2_weight,
            )
        except RuntimeError:
            if use_cutlass_env is None:
                trace_log(
                    "HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM is unset, but CUTLASS grouped GEMM is unavailable; "
                    "fallback to EPGroupGemm.apply."
                )
            else:
                raise
    cumsum = torch.cumsum(tokens_per_expert, dim=0).to(permute_tokens.device)
    return EPGroupGemm.apply(permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight)[0]


class EPGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]
        num_tokens = permute_tokens.shape[0]

        # compute linear layer fc1-1
        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_a=False,
            transpose_b=True,
        )

        # compute linear layer fc1-2
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_a=False,
            transpose_b=True,
        )

        # compute the actication of linear layer fc1-1
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # compute final result of linear layer fc1
        fc1_output = fc1_1_activation * fc1_2_output
        del fc1_1_activation

        # weighted projection is outside this function
        # compute linear layer fc2
        fc2_output = group_gemm_same_nk(
            a=fc1_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_a=False,
            transpose_b=True,
        )
        del fc1_output

        return fc2_output, fc1_1_output, fc1_2_output

    @staticmethod
    def setup_context(ctx, inputs, output):
        permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight = inputs
        fc2_output, fc1_1_output, fc1_2_output = output

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        ctx.save_for_forward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        ctx.mark_non_differentiable(fc1_1_output, fc1_2_output)

    @staticmethod
    def backward(ctx, grad_output, grad_fc1_1_output, grad_fc1_2_output):
        # grad_output: [tokens, hidden_dim]
        num_tokens = grad_output.shape[0]
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts]

        # dgrad fc1
        grad_fc1_output = group_gemm_same_nk(
            a=grad_output,
            b=fc2_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_b=False,
        )

        # recompute
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # wgrad fc2
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_output,
                b=fc1_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum,
                max_K=num_tokens,
                transpose_a=True,
                transpose_b=False,
            )
        del fc2_weight, fc1_output, grad_output

        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output
        del fc1_1_activation, grad_fc1_output, fc1_2_output

        # dgrad output 2
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_b=False,
        )

        # wgrad fc1-2
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=permute_tokens,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum,
                max_K=num_tokens,
                transpose_a=True,
                transpose_b=False,
            )
        del fc1_2_weight, grad_fc1_2_output

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        del grad_fc1_1_activation, fc1_1_output

        # dgrad output 1
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum,
            max_M=num_tokens,
            transpose_b=False,
        )

        # wgrad fc1-1
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=permute_tokens,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum,
                max_K=num_tokens,
                transpose_a=True,
                transpose_b=False,
            )
        del fc1_1_weight, grad_fc1_1_output, permute_tokens

        # grad input
        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2
        del grad_scatter_output_1, grad_scatter_output_2

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )

    @staticmethod
    def jvp(
        ctx,
        permute_tokens_t,
        cumsum_t,
        fc1_1_weight_t,
        fc1_2_weight_t,
        fc2_weight_t,
    ):
        """
        group_gemm 前向 (per-expert, transpose_b=True means A @ Wᵀ):

        f1 = x @ W1ᵀ, f2 = x @ W2ᵀ
        act = silu(f1), fc1 = act * f2
        out = fc1 @ W3ᵀ

        给定 tangents: dx, dW1, dW2, dW3 (cumsum 没有 tangent) 的 jvp:

        d_f1 = dx @ W1ᵀ + x @ dW1ᵀ, d_f2 = dx @ W2ᵀ + x @ dW2ᵀ
        d_act = silu'(f1) ⊙ d_f1 = silu_backward(d_f1, f1)
        d_fc1 = d_act ⊙ f2 + act ⊙ d_f2
        d_out = d_fc1 @ W3ᵀ + fc1 @ dW3ᵀ
        """
        # Tangents follow the forward inputs; `cumsum` is integer indexing -> no tangent.
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors
        num_tokens = permute_tokens.shape[0]

        def ggemm_nt(a, b):
            # a @ b^T per expert, matching the forward's group_gemm_same_nk(transpose_b=True)
            return group_gemm_same_nk(
                a=a.contiguous(),
                b=b.contiguous(),
                cumsum_M=cumsum,
                max_M=num_tokens,
                transpose_a=False,
                transpose_b=True,
            )

        # d(fc1_1_output) = dx @ W1^T + x @ dW1^T
        d_fc1_1_output = None
        if permute_tokens_t is not None:
            d_fc1_1_output = ggemm_nt(permute_tokens_t, fc1_1_weight)
        if fc1_1_weight_t is not None:
            term = ggemm_nt(permute_tokens, fc1_1_weight_t)
            d_fc1_1_output = term if d_fc1_1_output is None else d_fc1_1_output + term

        # d(fc1_2_output) = dx @ W2^T + x @ dW2^T
        d_fc1_2_output = None
        if permute_tokens_t is not None:
            d_fc1_2_output = ggemm_nt(permute_tokens_t, fc1_2_weight)
        if fc1_2_weight_t is not None:
            term = ggemm_nt(permute_tokens, fc1_2_weight_t)
            d_fc1_2_output = term if d_fc1_2_output is None else d_fc1_2_output + term

        # recompute fc1 primals
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # d(fc1_output) = silu'(f1)*d(f1) * f2 + silu(f1) * d(f2)
        d_fc1_output = None
        if d_fc1_1_output is not None:
            d_fc1_1_activation = torch.ops.aten.silu_backward(d_fc1_1_output, fc1_1_output)
            d_fc1_output = d_fc1_1_activation * fc1_2_output
        if d_fc1_2_output is not None:
            term = fc1_1_activation * d_fc1_2_output
            d_fc1_output = term if d_fc1_output is None else d_fc1_output + term

        # d(fc2_output) = d(fc1_output) @ W3^T + fc1_output @ dW3^T
        d_fc2_output = None
        if d_fc1_output is not None:
            d_fc2_output = ggemm_nt(d_fc1_output, fc2_weight)
        if fc2_weight_t is not None:
            term = ggemm_nt(fc1_output, fc2_weight_t)
            d_fc2_output = term if d_fc2_output is None else d_fc2_output + term

        if d_fc2_output is None:
            d_fc2_output = torch.zeros(
                num_tokens, fc2_weight.shape[1], dtype=fc1_output.dtype, device=fc1_output.device
            )

        # One tangent per forward output; fc1_1_output / fc1_2_output are non-differentiable.
        return d_fc2_output, None, None
