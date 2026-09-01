import os
from typing import Optional, Sequence, Union

import torch
from hy_parallelism.training.jvp_utils import in_jvp_context, jvp_guard, tolist

USE_FUSED_SORT_CHUNKS = os.environ.get("HY_PARALLELISM_FUSED_SORT_CHUNKS", "0").lower() in (
    "1",
    "true",
    "yes",
)

def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
    routing_weights: torch.Tensor = None,
):
    """
    Unpermutes the tokens and apply the weight.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_weights (torch.Tensor): The routing weights, [num_tokens, num_experts].
        hidden_states_shape (torch.Size): The shape of the hidden states, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    Returns:
        torch.Tensor: The unpermuted token tensor, [num_tokens, hidden_dim].
    """
    if routing_weights is not None:
        tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
        tokens = tokens * tokens_weight.unsqueeze(-1)
        
    hidden_dim = hidden_states_shape[-1]

    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)

    # Scatter add the permuted_input back to the original positions
    # torch.use_deterministic_algorithms(False) # True will lead to OOM in scatter_add_
    unpermuted_tokens.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens)
    return unpermuted_tokens

def test():
    """
    30.203.138.22: hidden_states_shape=torch.Size([15624, 3072])
    30.203.138.22: shape1=torch.Size([15624, 3072]) shape2=torch.Size([249984, 3072]) shape3=torch.Size([249984, 3072])
    30.203.138.22: permutation_mapping.is_contiguous()=True permutation_mapping.numel()=249984 permutation_mapping.element_size()=8
    """
    a = torch.tensor(1., requires_grad=True)
    tokens = torch.randn(249984, 3072, device='cuda', dtype=torch.float32, requires_grad=True)
    hidden_states_shape = torch.Size([15624, 3072])
    unpermuted_tokens = torch.zeros(hidden_states_shape, device='cuda', dtype=torch.float32) * a
    permutation_mapping = torch.randint(0, 15624, (249984,), device='cuda', dtype=torch.int64)
    foo = permutation_mapping.unsqueeze(1).expand(-1, 3072)
    unpermuted_tokens.scatter_add_(0, foo, tokens)


def generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts) -> torch.Tensor:
    """
    Generate the weight index for the unpermute operation.

    Args:
        routing_weights (torch.Tensor): The routing weights. shape [num_tokens, topk].
        selected_experts (torch.Tensor): The selected experts. shape [num_tokens, topk].
        num_experts (int): The number of experts. shape [num_tokens, num_experts].

    Returns:
        torch.Tensor: The weight index.
    """
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)

    weights_idx.scatter_add_(1, selected_experts, routing_weights)

    return weights_idx


def sort_chunks_by_idxs_old(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, tolist(split_sizes), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: Union[torch.Tensor, Sequence[int]],
    use_fused: Optional[bool] = None,
    row_id_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if use_fused is None:
        use_fused = USE_FUSED_SORT_CHUNKS
    if use_fused and input.is_cuda and not in_jvp_context():
        return _sort_chunks_by_idxs_fused(input, split_sizes, sorted_idxs, row_id_map)
    chunks = torch.split(input, tolist(split_sizes), dim=0)
    if isinstance(sorted_idxs, torch.Tensor):
        order = sorted_idxs.tolist()
    else:
        order = list(sorted_idxs)
    return torch.cat([chunks[i] for i in order], dim=0)


def _sort_chunks_by_idxs_fused(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: Union[torch.Tensor, Sequence[int]],
    row_id_map: Optional[torch.Tensor],
) -> torch.Tensor:
    from hy_parallelism.models.modules.moe.ptm_moe.fused_moe.fused_permutation import (
        fused_sort_chunks_by_idxs,
    )

    # # TODO: REMOVE REF
    # ref = sort_chunks_by_idxs_old(input, split_sizes, sorted_idxs)

    device = input.device
    # split_sizes from preprocess live on CPU; must not use non_blocking before the Triton kernel
    split_sizes = split_sizes.reshape(-1)
    if split_sizes.device != device:
        split_sizes = split_sizes.to(device=device, dtype=torch.int32)
    elif split_sizes.dtype != torch.int32:
        split_sizes = split_sizes.to(dtype=torch.int32)

    if isinstance(sorted_idxs, torch.Tensor):
        sorted_idxs_t = sorted_idxs.reshape(-1).to(device=device, dtype=torch.long)
    else:
        sorted_idxs_t = torch.tensor(sorted_idxs, device=device, dtype=torch.long)

    torch.cuda.synchronize()
    output, _ = fused_sort_chunks_by_idxs(input, split_sizes, sorted_idxs_t, row_id_map)

    # assert torch.equal(output, ref), (
    #     f"fused sort != split/cat: max_diff={(output - ref).abs().max().item()}, "
    #     f"shape_fused={tuple(output.shape)}, shape_ref={tuple(ref.shape)}, "
    #     f"split_sum={int(split_sizes.sum())}, num_tokens={input.shape[0]}"
    # )

    return output
