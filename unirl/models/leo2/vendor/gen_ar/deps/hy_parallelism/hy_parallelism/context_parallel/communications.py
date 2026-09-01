# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os
from typing import Any, Tuple, Union
import torch
import torch.distributed as dist

from torch.nn import functional as F
from hy_parallelism.parallel_states import get_parallel_state

__enable_sp_padding = True

def set_enable_sp_padding(enable: bool):
    global __enable_sp_padding
    __enable_sp_padding = enable


def broadcast(input_: torch.Tensor, group: dist.ProcessGroup):
    src = dist.get_global_rank(group, 0)
    dist.broadcast(input_, src=src, group=group)


def _all_to_all_4D(
        input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, input_pad: Union[int, None] = None, group=None
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        input_pad (int, None): default None. When the user does not set input_pad, all_gather operation is executed. 
                               When the user explicitly sets input_pad, all_gather operation can be skipped. 
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
            input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    group_rank = dist.get_group_rank(group, dist.get_rank())

    if scatter_idx == 2 and gather_idx == 1: # to split head

        if input_pad is None:

            seq_lens = [None] * seq_world_size
            dist.all_gather_object(seq_lens, input.shape[1], group)

            if seq_lens[-1] != seq_lens[0] :
                if not __enable_sp_padding:
                    raise RuntimeError('SP error')
                assert seq_lens[0] > seq_lens[-1], f'seq_lens: {seq_lens}'
                if seq_lens[group_rank] != seq_lens[0]:
                    local_pad = seq_lens[0] - seq_lens[group_rank]
                    input = F.pad(input, (0, 0, 0, 0, 0, local_pad))

                input_pad = seq_lens[0] * seq_world_size - sum(seq_lens)
            else:
                input_pad = 0
        elif input_pad > 0 and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
            raise NotImplementedError('Feeding input_pad to all_to_all_4D is not supported anymore.')
            input = F.pad(input, (0, 0, 0, 0, 0, input_pad))

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert hc % seq_world_size == 0, f'Invalid Head size: {hc}, which should be divisible by spsize {seq_world_size}'
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            # dist.all_to_all_single(output, input_t, group=group)

            from torch.distributed.nn.functional import _AlltoAllSingle
            output = _AlltoAllSingle.apply(group, output, None, None, input_t)
            # torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads k to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if input_pad > 0:
            output = output[:, :-input_pad]

        return output

    elif scatter_idx == 1 and gather_idx == 2: # to split seq

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        original_seqlen = seqlen

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            if not __enable_sp_padding:
                raise RuntimeError(f'seqlen({seqlen}) should be divisible by spsize({seq_world_size}), or you can try hy_parallelism.context_parallel.core.set_enable_sp_padding(True)')
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0


        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        # input_t = einops.rearrange(input, 'b (sp sl) sc hs -> sp sc sl b hs', sp=seq_world_size).contiguous()
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            # dist.all_to_all_single(output, input_t, group=group)
            from torch.distributed.nn.functional import _AlltoAllSingle
            output = _AlltoAllSingle.apply(group, output, None, None, input_t)

            # torch.cuda.synchronize()
        else:
            output = input_t
        # output = einops.rearrange(output, 'sp sc sl b hs -> b sl (sp sc) hs')

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        local_chunk_len_pad = seqlen // seq_world_size
        local_chunks_nopad = [ min(original_seqlen, (i+1) * local_chunk_len_pad) - i * local_chunk_len_pad for i in range(seq_world_size)]
        if gap > 0 and local_chunks_nopad[group_rank] != local_chunk_len_pad:
            output = output[:, :-(local_chunk_len_pad - local_chunks_nopad[group_rank])]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
            group: dist.ProcessGroup,
            input: torch.Tensor,
            scatter_idx: int,
            gather_idx: int,
            input_pad: Union[int, None],
    ) -> torch.Tensor:

        return _all_to_all_4D(input, scatter_idx, gather_idx, input_pad, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(
                ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, None
            ),
            None,
            None,
            None,
        )
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        group, input_, scatter_idx, gather_idx, input_pad = inputs
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
    
    @staticmethod
    def jvp(ctx, group_tangent, input_tangent, scatter_idx_tangent, gather_idx_tangent, input_pad_tangent):
        """
        JVP implementation for SeqAllToAll4D.
        
        For AllToAll operations, the JVP applies the same all-to-all transformation
        to the tangent vector as was applied to the primal input.
    
        Args:
            ctx: Context from forward pass
            group_tangent: Tangent for group (always None)
            input_tangent: Tangent vector for input tensor
            scatter_idx_tangent: Tangent for scatter_idx (always None)
            gather_idx_tangent: Tangent for gather_idx (always None)
        
        Returns:
            Tangent of the output tensor
        """
        # Only the input tensor has a meaningful tangent
        if input_tangent is None:
            return None
            
        # Apply the same AllToAll transformation to the tangent vector
        # Must use apply, other wise, all_gather communication will raise Storage error.
        return SeqAllToAll4D.apply(ctx.group, input_tangent, ctx.scatter_idx, ctx.gather_idx, None)

        # Storage error: NotImplementedError: Cannot access storage of TensorWrapper
        # return _all_to_all_4D(
        #     input_tangent,
        #     scatter_idx=ctx.scatter_idx,
        #     gather_idx=ctx.gather_idx,
        #     group=ctx.group,
        # )


def all_to_all_4D(
        input_: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1, input_pad: Union[int, None] = None,
):
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim, input_pad)


def _all_to_all(
        input_: torch.Tensor,
        world_size: int,
        group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(input_, process_group, scatter_dim, gather_dim):
        world_size = dist.get_world_size(process_group)
        output = _all_to_all(
            input_, world_size, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        input_, process_group, scatter_dim, gather_dim = inputs
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)

    @staticmethod
    def jvp(ctx, input_tangent, process_group_tangent, scatter_dim_tangent, gather_dim_tangent):
        """
        JVP implementation for _AllToAll in context parallel.
        
        For AllToAll operations, the JVP applies the same all-to-all transformation
        to the tangent vector as was applied to the primal input.
        
        Args:
            ctx: Context from forward pass
            input_tangent: Tangent vector for input tensor
            process_group_tangent: Tangent for process_group (always None)
            scatter_dim_tangent: Tangent for scatter_dim (always None)
            gather_dim_tangent: Tangent for gather_dim (always None)
        
        Returns:
            Tangent of the output tensor
        """
        # Only the input tensor has a meaningful tangent
        if input_tangent is None:
            return None
            
        # Apply the same AllToAll transformation to the tangent vector
        return _all_to_all(
            input_tangent,
            ctx.world_size,
            ctx.process_group,
            ctx.scatter_dim,
            ctx.gather_dim,
        )


def all_to_all(
        input_: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1
):
    return _AllToAll.apply(input_, group, scatter_dim, gather_dim)

class _Reduce_Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)



# https://github.com/hao-ai-lab/FastVideo/blob/4112507e99b06ffca502d34fe2da1120371c2bd5/fastvideo/distributed/device_communicators/base_device_communicator.py#L20
# https://github.com/pytorch/xla/issues/5784
# https://github.com/pytorch/pytorch/blob/5a114f72bf0aa93268ba51707a20f99b79ed048d/torch/distributed/nn/functional.py#L335C78-L335C78
# 错误的backward，罪魁祸首：
# https://github.com/shortcut-guide/fastvideo/blob/e087e85e09d0a18faf6e5662973a750c9a65c240/fastvideo/utils/communications.py


# deepspeed 风格的实现是  先每个rank只有自己的partial grad, 然后在 zero 里 handle reduce
# torchtitan 的风格是每个rank都 reduce好。
class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(input_, dim, pad, group):
        world_size = dist.get_world_size(group)

        # rank = get_parallel_state().sp_mesh.get_local_rank()
        if pad is None:
            sizes = [None] * world_size
            dist.all_gather_object(sizes, input_.shape, group)
            tensor_list = [torch.empty(sizes[i], dtype=input_.dtype, device=input_.device) for i in range(world_size)]

        else:
            # TODO: update test_sp.py
            raise NotImplementedError('AllGather with pad is not supported for training. For inference, you can use all_gather_pad instead.')
            tensor_list = [torch.empty(input_.shape, dtype=input_.dtype, device=input_.device) for i in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        rank = get_parallel_state().sp_rank
        dim = ctx.dim
        input_size = ctx.input_size


        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_size, group=group)

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]


        # Original fastvideo implementation do not reduce scatter the gradients, which is incorrect.
        # grad_input = grad_input.contiguous()
        # grad_input_list = [t.contiguous() for t in grad_input_list]
        # dist.reduce_scatter(grad_input, grad_input_list, group=group, op=dist.ReduceOp.SUM)

        # Why do we need a reduce here (i.e. why the original fastvideo implementation is buggy): 
        # Because other ranks have gathered my portion of the tensor, 
        # so the gradients from their subsequent computations involving my data need to be communicated back to me!
        # 
        # Rank 0         Rank 1
        # a1 (1) a2(1)   a2 (1)  a1(2)
        #       \______
        #              \   a1 的梯度应该要能够传回去
        #               \
        # [a1, a2]       [a1, a2]
        # (a1+a2).sum()  (2*a1+a2).sum()
        #
        # FSDP example:
        #   sp=1, 4 GPUs:
        #     g1 g2 g3 g4
        #   fsdp reduce: (g1+g2+g3+g4) / 4
        #
        #   sp=2, 8 GPUs:
        #   Without reduce_scatter:
        #     g1/2 g1/2  g2/2  g2/2  g3/2  g3/2  g4/2  g4/2
        #   fsdp reduce: (g1+g2+g3+g4) / 8
        #   With reduce_scatter:
        #     g1    g1    g2    g2    g3    g3    g4    g4
        #   fsdp reduce: (g1+g2+g3+g4) / 4
        # from torch.distributed import _AllGather
        grad_input = _Reduce_Scatter.apply(dist.ReduceOp.SUM, group, grad_input, *grad_input_list)

        return grad_input, None, None, None

    
    @staticmethod
    def setup_context(ctx, inputs, output):
        input_, dim, pad, group = inputs
        ctx.dim = dim
        ctx.group = group
        input_size = list(input_.size())
        ctx.input_size = input_size[dim]
    
    @staticmethod
    def jvp(ctx, input_tangent, dim_tangent, pad_tangent, group_tangent):
        """
        JVP implementation for _AllGather.
        
        For AllGather operations, the JVP applies the same all-gather transformation
        to the tangent vector as was applied to the primal input.
        
        Args:
            ctx: Context from forward pass
            input_tangent: Tangent vector for input tensor
            dim_tangent: Tangent for dim (always None)
            group_tangent: Tangent for group (always None)
        
        Returns:
            Tangent of the output tensor
        """
        # Only the input tensor has a meaningful tangent
        if input_tangent is None:
            return None
            
        # Apply the same AllGather transformation to the tangent vector
        return _AllGather.apply(input_tangent, ctx.dim, None, ctx.group)



def all_gather(input_: torch.Tensor, dim: int = 1, pad: Union[None, int] = None, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, pad, group)
    # from torch.distributed.nn.functional import _AllGather
    # gathered_tensor = _AllGather.apply(group, input_)
    # return torch.cat(gathered_tensor, dim=dim)


    

def all_gather_pad(input_: torch.Tensor, dim: int = 1, pad: Union[None, int] = None, group=None):
    """历史遗留代码，不建议使用，这里保留只为了支持一些旧服务"""

    world_size = dist.get_world_size(group)

    # rank = get_parallel_state().sp_mesh.get_local_rank()
    seq_world_size = dist.get_world_size(group)
    assert pad is not None

    if pad > 0 and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
        if input_.dim() == 4:
            input_ = F.pad(input_, (0, 0, 0, 0, 0, pad))
        elif input_.dim() == 3:
            input_ = F.pad(input_, (0, 0, 0, pad))

    tensor_list = [torch.empty(input_.shape, dtype=input_.dtype, device=input_.device) for i in range(world_size)]
    input_ = input_.contiguous()
    dist.all_gather(tensor_list, input_, group=group)

    output = torch.cat(tensor_list, dim=dim)
    if pad > 0:
        output = output[:, :-pad]
    return output

