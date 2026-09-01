import loguru
import torch

# import deepspeed.utils.groups as groups
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast, List

import torch
from torch import Tensor
from torch.nn import Module
from .ptm_hunyuan_moe import TopKGate
from .fused_moe.dropless_permutation import _moe_permute_mask_map, _moe_unpermute_mask_map

try:
    from deep_ep import EventOverlap, Buffer
    # Communication buffer (will allocate at runtime)
    _buffer: Optional[Buffer] = None

    # Set the number of SMs to use
    # NOTES: this is a static variable
    Buffer.set_num_sms(24)
except ImportError:
    EventOverlap = None
    Buffer = None

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


def fused_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    """
    output, row_id_map, _ = _moe_permute_mask_map.apply(inp, routing_map, num_out_tokens, None)
    return output, row_id_map


def fused_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: torch.Tensor = None,
    restore_shape: torch.Tensor = None,
    probs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Tensor
        The output shape after the unpermute operation.
    probs: torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    """
    # if torch.distributed.get_rank() == 0: import debugpy; debugpy.listen(('30.203.138.137', 56789)); debugpy.wait_for_client()
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    return _moe_unpermute_mask_map.apply(inp, row_id_map, merging_probs, restore_shape)


def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    if fused:
        if fused_permute is None:
            raise ValueError("fused_permute is not available. Please install TE >= 2.1.0.")
        return fused_permute(tokens, routing_map, num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        return fused_unpermute(permuted_tokens, sorted_indices, probs, restore_shape)

    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(
        restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
    )
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens


def indices_to_multihot(indices, probs, num_local_experts):
    """
    Converts a tensor of indices to a multihot vector efficiently in PyTorch.

    Args:
        indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
        probs (torch.Tensor): [num_tokens, topk] token probabilities.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - routing_map: Multihot vector.
            - probs: Multihot probabilities.
    """
    batch_size = indices.shape[0]
    multihot_routing_map = torch.zeros(
        (batch_size, num_local_experts), dtype=torch.long, device=indices.device
    )

    multihot_probs = torch.zeros(
        (batch_size, num_local_experts), dtype=torch.float, device=indices.device
    )

    mask = indices != -1
    valid_indices = indices[mask]
    row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
        mask.sum(dim=1)
    )
    multihot_routing_map[row_indices, valid_indices] = 1

    multihot_probs[row_indices, valid_indices] = probs[mask]
    return multihot_routing_map.bool(), multihot_probs


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, 
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                topk_idx: torch.Tensor, 
                topk_weights: torch.Tensor,
                num_experts: int, 
                group: torch.distributed.ProcessGroup,
                previous_event: Optional[EventOverlap] = None,
                async_finish: bool = False):
        # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency 
        # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
        # refer to the docs of `Buffer.dispatch`
        buffer = get_buffer(group, get_hidden_bytes(x))

        # Calculate layout before actual dispatch
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
            buffer.get_dispatch_layout(topk_idx, num_experts,
                                        previous_event=previous_event, async_finish=False,
                                        allocate_on_comm_stream=previous_event is not None)
        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
        # For more advanced usages, please refer to the docs of the `dispatch` function
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
            buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights.float(),
                            num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                            is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                            previous_event=previous_event, async_finish=async_finish,
                            allocate_on_comm_stream=False)
        # For event management, please refer to the docs of the `EventOverlap` class
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.group = group

        return recv_x, recv_topk_weights, recv_topk_idx, num_recv_tokens_per_expert_list, handle, event

    @staticmethod
    def backward(ctx, grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, grad_recv_topk_idx: None, grad_num_recv_tokens_per_expert_list: None, grad_handle: None, grad_event: None):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_recv_x))

        # The backward process of MoE dispatch is actually a combine
        # For more advanced usages, please refer to the docs of the `combine` function
        combined_grad_x, combined_grad_recv_topk_weights, event = \
            buffer.combine(grad_recv_x.contiguous(), ctx.handle, topk_weights=grad_recv_topk_weights.float(), async_finish=True)
        
        event.current_stream_wait()

        # For event management, please refer to the docs of the `EventOverlap` class
        return combined_grad_x, None, combined_grad_recv_topk_weights, None, None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor, 
                handle: Tuple, 
                group: torch.distributed.ProcessGroup,
                previous_event: Optional[EventOverlap] = None):
        buffer = get_buffer(group, get_hidden_bytes(x))

        # Do MoE combine
        # For more advanced usages, please refer to the docs of the `combine` function
        combined_x, _, event = buffer.combine(x, handle, async_finish=False, previous_event=previous_event,
                                            allocate_on_comm_stream=previous_event is not None)
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.group = group

        # For event management, please refer to the docs of the `EventOverlap` class
        return combined_x, event
    
    @staticmethod
    def backward(ctx, grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], grad_event: None):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_combined_x))

        # The backward process of MoE combine is actually a dispatch
        # For more advanced usages, please refer to the docs of the `combine` function
        grad_x, _, _, _, _, event = buffer.dispatch(grad_combined_x.contiguous(), handle=ctx.handle, async_finish=False)

        # For event management, please refer to the docs of the `EventOverlap` class
        return grad_x, None, None, None


class DeepEP_MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_moe = moe.l_moe
        > aux_loss = l_moe[0]
        > z_loss = l_moe[1]

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """
    def __init__(self,
                 gate: TopKGate,
                 experts: Module,
                 ep_group_name,
                 ep_size,
                 num_local_experts: int,
                 use_tutel: bool = False,
                 gather_seq: bool = False,
                 a2a_ffn_overlap_degree: int = 1,
                 a2a_ffn_overlap_chunks: int = 2,
                 expert_choice_routing: bool = False,
                 use_fused_moe: bool = False,
                 fp8_a2a: bool = False,
                 ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        # use _set_ep_group
        self.ep_group = None
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.a2a_ffn_overlap_chunks = a2a_ffn_overlap_chunks
        # from megatron import get_timers
        # self.timers = get_timers()
        self.wall_clock_breakdown = False
        self.expert_choice_routing = expert_choice_routing
        self.gather_seq = gather_seq

        self.use_fused_moe = use_fused_moe
        self.gate.ds_fp32 = True

    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def _unset_ep_group(self):
        self.ep_group = None

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers('moe', log_level=2).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = input[0].reshape(-1, d_model)
        num_tokens = reshaped_input.shape[0]

        '''
        reshaped_input: (MBS * seq_len, dim)
        dispatched_input: (#experts, #tokens, dim)
        '''

        gate_reshaped_input = reshaped_input

        extra_return = {}

        if self.use_fused_moe and not self.gate.disable_fused_topk_gating:
            # self.l_moe, topk_weights, topk_idx, self.exp_counts, num_out_tokens, extra_return = self.gate(gate_reshaped_input, input[1], self.ep_group)
            out = self.gate(gate_reshaped_input, input[1], self.ep_group)
            if len(out) == 6:
                self.l_moe, topk_weights, topk_idx, self.exp_counts, num_out_tokens, extra_return = out
            else:
                # raise NotImplementedError('noaltian requires returning extra_return, fused topk gating is not supported.')
                self.l_moe, topk_weights, topk_idx, self.exp_counts, num_out_tokens = out
            
            if self.gate.drop_tokens:
                # topk_weights and topk_idx: (num_tokens, num_experts) -> (num_tokens, topk)
                values, indices = torch.topk(topk_idx.int(), dim=1, k=self.gate.k)
                topk_idx = torch.where(values > 0, indices, -1)
                topk_weights = topk_weights.gather(dim=1, index=indices)
            else:
                topk_weights, topk_idx = torch.topk(topk_weights, dim=1, k=self.gate.k)
        else:
            self.l_moe, topk_weights, topk_idx, self.exp_counts, extra_return = self.gate(gate_reshaped_input, input[1], self.ep_group)

        for k, v in extra_return.items():
            self.register_object(k, v)
        
        num_experts = self.gate.wg.weight.shape[0]

        if self.wall_clock_breakdown:
            self.timers('falltoall', log_level=2).start()

        dispatched_input, recv_topk_weights, recv_topk_idx, num_recv_tokens_per_expert_list, handle, dispatch_event = Dispatch.apply(reshaped_input, topk_idx, topk_weights, num_experts, self.ep_group, None, self.a2a_ffn_overlap_degree > 1)

        if 0 in dispatched_input.shape:
            pass
            # raise RuntimeError(
            #     'Some expert is not assigned any tokens, which is invalid for deepep. '
            #     'This could happen with PP shape inference or when experts offloading is extremely unbalanced. '
            #     'To handle this gracefully, set model._allow_empty_experts=True (returns zeros) '
            #     'or model._pad_empty_experts=True (adds dummy tokens). '
            #     'Please report a bug to kevinkhwu if this occurs during normal training.'
            # )

        if self.wall_clock_breakdown:
            self.timers('falltoall').stop()

        if hasattr(self, "mlp_layer"):
            if self.a2a_ffn_overlap_degree == 2:
                with dispatch_event:
                    s_output, _ = self.mlp_layer(input[0])

        dispatched_routing_map, dispatched_probs = indices_to_multihot(recv_topk_idx, recv_topk_weights, self.num_local_experts)

        # 后面的 not self.use_fused_moe 其实应该改成判断是否用 fused topk
        if self.gate.drop_tokens and not self.use_fused_moe: # kevinkhwu: Fused permute implementation ignores gate.drop_tokens. 
            import math
            # to(torch.int64) works around a bug in torch.onnx.export:
            # it should cast k to int64 when converting torch.topk but it doesn't.
            expert_capacity = math.ceil((num_tokens / num_experts) * self.gate.capacity_factor)
            if expert_capacity < self.gate.min_capacity:
                expert_capacity = self.gate.min_capacity.to(torch.int64)
            expert_capacity = self.ep_size * self.gate.k * expert_capacity  # 是收之后算的 capacity，所以要乘以 ep_size
            num_out_tokens = expert_capacity * self.num_local_experts
            # 重新计算 capacity rate
            capacity_rate = sum([min(expert_capacity, recv_tokens) for recv_tokens in num_recv_tokens_per_expert_list]) / (sum(num_recv_tokens_per_expert_list) + 1e-8)
            self.l_moe[-1] = torch.tensor(capacity_rate).cuda()
        else:
            num_out_tokens = sum(num_recv_tokens_per_expert_list)

        hidden_shape_before_permute = dispatched_input.shape
        permuted_dispatched_input, reversed_mapping_for_combine = permute(
            dispatched_input,
            dispatched_routing_map,
            num_out_tokens=num_out_tokens,
            fused=self.use_fused_moe,
            drop_and_pad=self.gate.drop_tokens
        )

        if self.gate.drop_tokens and not self.use_fused_moe:
            tokens_per_expert = [min(expert_capacity, dispatched_routing_map.shape[0])] * self.num_local_experts
        else:
            tokens_per_expert = num_recv_tokens_per_expert_list
        
        expert_output = self.experts(permuted_dispatched_input, num_tokens_per_local_expert=torch.tensor(tokens_per_expert))
        
        expert_output = unpermute(
            expert_output,
            reversed_mapping_for_combine,
            restore_shape=hidden_shape_before_permute,
            routing_map=dispatched_routing_map,
            probs=dispatched_probs,     # FP32 here
            fused=self.use_fused_moe,
            drop_and_pad=self.gate.drop_tokens
        )
        expert_output = expert_output.to(reshaped_input.dtype)

        if self.wall_clock_breakdown:
            self.timers('salltoall', log_level=2).start()


        
        # DEBUG MESSAGE
        import os
        if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
            from hy_parallelism.utils import gather_obj
            # wrong_text_dispatch = input[1].sum() % 5 != 0

            if os.environ.get('HY_PARALLELISM_ALWAYS_PRINT_EP_MSG', '0') == '1' or (0 in dispatched_input.shape):
                tokens_per_expert_gathered = gather_obj(tokens_per_expert, group=self.ep_group)
                with torch.no_grad():
                    input_fp32 = gate_reshaped_input.float()

                    # with torch.cuda.amp.autocast(enabled=False):
                    #     logits = self.gate.wg(input_fp32)
                debug_msg = (
                    f'{reshaped_input.shape=} {tokens_per_expert_gathered=} {permuted_dispatched_input.shape=} '
                    f'({topk_idx.shape} tokens) {(topk_idx == -1).sum()} tokens are masked ({topk_idx.numel() - (topk_idx == -1).sum()} tokens are not masked)\n'
                    f'used_token ({input[1].shape})= {input[1]=} {input[1].sum()=}\n'
                    f'({self.num_local_experts=} {self.gate.k=} {num_experts=} ep_size:{self.ep_size})' # ' ({logits.shape=} {logits=}.'
                    f'topk_weights ({topk_weights.shape})= \n'
                    # f'{topk_weights=} {topk_idx=} \n'
                )
                loguru.logger.warning(f'Some expert is not assigned any tokens. {debug_msg}.')


        # if 0 in dispatched_input.shape:
        #     has_problem = True
        # else:
        #     has_problem = False

        # if has_problem:
        #     torch.save(
        #         {
        #             'expert_output': expert_output, # topk 0 c
        #             'dispatched_input': dispatched_input, # 0 c
        #             'recv_topk_idx': recv_topk_idx, # 0 topk
        #             'recv_topk_weights': recv_topk_weights, # 0 topk
        #             'num_recv_tokens_per_expert_list': num_recv_tokens_per_expert_list, # [0, 0]
        #             'permuted_dispatched_input': permuted_dispatched_input, # [0, c]
        #             'reversed_mapping_for_combine': reversed_mapping_for_combine,
        #             'tokens_per_expert': tokens_per_expert,
        #         }, f'/tmp/fuck_{torch.distributed.get_rank()}.pt'
        #     )
        #     raise RuntimeError('FUCK')


        
        combined_output, combine_event = Combine.apply(expert_output, handle, self.ep_group)

        if self.wall_clock_breakdown:
            self.timers('salltoall').stop()

        # Re-shape back: gecm -> ecm
        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers('moe').stop()
        
        if hasattr(self, "mlp_layer"):
            if self.a2a_ffn_overlap_degree == 2:
                f_out = a + s_output
                return f_out

        return a