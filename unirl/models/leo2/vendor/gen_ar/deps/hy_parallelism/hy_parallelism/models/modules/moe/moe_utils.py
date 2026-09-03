import os
from typing import Optional, Sequence, Tuple, Union

import torch
from hy_parallelism.training.jvp_utils import in_jvp_context, jvp_guard, tolist

USE_FUSED_SORT_CHUNKS = os.environ.get("HY_PARALLELISM_FUSED_SORT_CHUNKS", "0").lower() in (
    "1",
    "true",
    "yes",
)

def permute(tokens: torch.Tensor, routing_map: torch.Tensor, cast_dtype: Optional[torch.dtype] = None):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    if cast_dtype is not None:
        tokens = tokens.to(cast_dtype)

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def permute_no_sync(
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    cast_dtype: Optional[torch.dtype] = None,
):
    """
    Permute tokens by sorting ``topk_idx`` into expert-major order.

    Unlike :func:`permute`, the output length is known a priori as
    ``num_tokens * top_k``, so no CUDA→CPU sync is required.

    Assumes every token has exactly ``top_k`` valid expert assignments
    (i.e. dropless / no token dropping).

    Args:
        tokens (torch.Tensor): Input tokens, [num_tokens, hidden_dim].
        topk_idx (torch.Tensor): Gate top-k expert indices, [num_tokens, top_k].
        cast_dtype (torch.dtype, optional): Optional dtype to cast tokens to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Permuted tokens and the token index
        mapping used for unpermute, both of length ``num_tokens * top_k``.
    """
    num_tokens, _ = tokens.shape
    top_k = topk_idx.shape[-1]
    if cast_dtype is not None:
        tokens = tokens.to(cast_dtype)

    topk_idx = topk_idx.reshape(num_tokens, top_k)
    expert_ids = topk_idx.reshape(-1)
    token_ids = (
        torch.arange(num_tokens, device=topk_idx.device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    # Stable sort keeps ascending token order within each expert, matching
    # routing_map.masked_select over [num_experts, num_tokens].
    sorted_indices = token_ids[torch.argsort(expert_ids, stable=True)]
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def build_fused_unpermute_row_id_map(
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive fused_unpermute inputs from dropless unpermute args.

    fused_unpermute expects:
      - routing_map: [num_tokens, num_experts]
      - row_id_map:  [num_experts * num_tokens], 1-indexed into permuted rows
        (kernel does ``src_row = row_id_map[...] - 1``)

    permutation_mapping is the expert-major token index list from ``permute``.
    """
    routing_map_bool = routing_map.bool()
    num_experts, num_tokens = routing_map_bool.shape
    device = permutation_mapping.device
    tokens_per_expert = routing_map_bool.sum(dim=1)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.long),
        tokens_per_expert.long(),
        output_size=permutation_mapping.numel(), # avoid stream sync
    )
    row_id_2d = torch.zeros(num_experts, num_tokens, dtype=torch.int64, device=device)
    row_id_2d[expert_ids, permutation_mapping] = (
        torch.arange(permutation_mapping.numel(), device=device, dtype=torch.int64) + 1
    )
    routing_map_fused = routing_map_bool.T.contiguous().to(torch.int64)
    return row_id_2d.reshape(-1), routing_map_fused


def unpermute(
    tokens: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
    routing_weights: torch.Tensor = None,
    use_fused_unpermute: bool = False,
    *,
    row_id_map=None, routing_map_fused=None, # only required for fused implementation
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
    if (
        use_fused_unpermute
        and tokens.is_cuda
        and permutation_mapping.numel() > 0
        and not in_jvp_context()
    ):
        from hy_parallelism.models.modules.moe.ptm_moe.fused_moe.fused_permutation import (
            fused_unpermute,
        )

        if row_id_map is None or routing_map_fused is None:
            row_id_map, routing_map_fused = build_fused_unpermute_row_id_map(
                permutation_mapping, routing_map
            )
        probs = routing_weights.contiguous() if routing_weights is not None else None
        return fused_unpermute(
            tokens.contiguous(),
            row_id_map,
            hidden_states_shape,
            probs,
            routing_map_fused,
        )

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
