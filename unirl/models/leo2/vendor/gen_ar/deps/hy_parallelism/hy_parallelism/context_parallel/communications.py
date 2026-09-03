# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os
from typing import Optional, Sequence, Tuple
import torch
import torch.distributed as dist

from torch.nn import functional as F
from hy_parallelism.distributed.communications.utils import run_on_async_stream
from hy_parallelism.parallel_states import get_parallel_state
from hy_parallelism.tools.profiling import profile_range

__enable_sp_padding = True

def set_enable_sp_padding(enable: bool):
    global __enable_sp_padding
    __enable_sp_padding = enable


def broadcast(input_: torch.Tensor, group: dist.ProcessGroup):
    src = dist.get_global_rank(group, 0)
    dist.broadcast(input_, src=src, group=group)

class _AlltoAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(group, output, output_split_sizes, input_split_sizes, input):
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        group, _output, output_split_sizes, input_split_sizes, input = inputs
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_size = output.size()
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

    @staticmethod
    def backward(ctx, grad_output):
        tensor = torch.empty(
            ctx.input_size, device=grad_output.device, dtype=grad_output.dtype
        )
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.input_split_sizes,
                ctx.output_split_sizes,
                grad_output.contiguous(),
            ),
        )

    @staticmethod
    def jvp(
        ctx,
        group_tangent,
        output_tangent,
        output_split_sizes_tangent,
        input_split_sizes_tangent,
        input_tangent,
    ):
        """JVP for AlltoAllSingle: apply the same all-to-all to the input tangent."""
        if input_tangent is None:
            return None
        tensor = torch.empty(
            ctx.output_size, device=input_tangent.device, dtype=input_tangent.dtype
        )
        return _AlltoAllSingle.apply(
            ctx.group,
            tensor,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            input_tangent.contiguous(),
        )


def _resolve_head_split_seq_lens(
    seq_lens: Sequence[int],
    group_rank: int,
) -> Tuple[int, int]:
    """Return (output_trim, local_pad) for to_split_head all_to_all."""
    if seq_lens[-1] != seq_lens[0]:
        if not __enable_sp_padding:
            raise RuntimeError('SP error')
        assert seq_lens[0] > seq_lens[-1], f'seq_lens: {seq_lens}'
        local_pad = seq_lens[0] - seq_lens[group_rank] if seq_lens[group_rank] != seq_lens[0] else 0
        output_trim = seq_lens[0] * len(seq_lens) - sum(seq_lens)
    else:
        local_pad = 0
        output_trim = 0
    return output_trim, local_pad


def all_to_all_4D(
        input_: torch.Tensor,
        group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
        split_seq_lens: Optional[Sequence[int]] = None,
        async_op: bool = False,
):
    """
    all-to-all for QKV

    Args:
        input_ (torch.Tensor): a tensor sharded along dim scatter dim
        group: torch process group
        scatter_dim (int): default 2
        gather_dim (int): default 1
        split_seq_lens (Sequence[int], optional): per-rank local sequence lengths after maybe_scatter_seq.
            When provided, the runtime all_gather_object for sequence lengths is skipped.
        async_op (bool): if True, run on a side CUDA stream and return CudaStreamWork

    Returns:
        torch.Tensor: resharded tensor, or CudaStreamWork when async_op=True
    """
    if async_op and not input_.is_cuda:
        raise RuntimeError("all_to_all_4D(async_op=True) requires CUDA tensor input")

    input = input_
    assert (
            input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    group_rank = dist.get_group_rank(group, dist.get_rank())

    if scatter_dim == 2 and gather_dim == 1: # to split head

        if split_seq_lens is not None:
            seq_lens = list(split_seq_lens)
            output_trim, local_pad = _resolve_head_split_seq_lens(seq_lens, group_rank)
            if local_pad > 0:
                input = F.pad(input, (0, 0, 0, 0, 0, local_pad))
        else:
            seq_lens = [None] * seq_world_size
            dist.all_gather_object(seq_lens, input.shape[1], group)

            output_trim, local_pad = _resolve_head_split_seq_lens(seq_lens, group_rank)
            if local_pad > 0:
                input = F.pad(input, (0, 0, 0, 0, 0, local_pad))

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

        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        output_t = torch.empty_like(input_t)
        def run_head_split() -> torch.Tensor:
            if seq_world_size == 1:
                return input_t
            return _AlltoAllSingle.apply(group, output_t, None, None, input_t)

        def finalize_head_split(output_t: torch.Tensor) -> torch.Tensor:
            output_t = output_t.reshape(seqlen, bs, shard_hc, hs)
            output_t = output_t.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
            if output_trim > 0:
                output_t = output_t[:, :-output_trim]
            return output_t

        if async_op:
            return run_on_async_stream(
                run_head_split,
                input_.device,
                record_tensors=(input_t,),
                on_wait=finalize_head_split,
            )
        return finalize_head_split(run_head_split())

    elif scatter_dim == 1 and gather_dim == 2: # to split seq

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

        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        local_chunk_len_pad = seqlen // seq_world_size
        local_chunks_nopad = [ min(original_seqlen, (i+1) * local_chunk_len_pad) - i * local_chunk_len_pad for i in range(seq_world_size)]
        output_t = torch.empty_like(input_t)

        def run_seq_split() -> torch.Tensor:
            if seq_world_size == 1:
                return input_t
            return _AlltoAllSingle.apply(group, output_t, None, None, input_t)

        def finalize_seq_split(output_t: torch.Tensor) -> torch.Tensor:
            output_t = output_t.reshape(hc, shard_seqlen, bs, hs)
            output_t = output_t.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
            if gap > 0 and local_chunks_nopad[group_rank] != local_chunk_len_pad:
                output_t = output_t[:, :-(local_chunk_len_pad - local_chunks_nopad[group_rank])]
            return output_t

        if async_op:
            return run_on_async_stream(
                run_seq_split,
                input_.device,
                record_tensors=(input_t,),
                on_wait=finalize_seq_split,
            )
        return finalize_seq_split(run_seq_split())
    else:
        raise RuntimeError("scatter_dim must be 1 or 2 and gather_dim must be 1 or 2")


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
    def forward(input_, dim, group, split_seq_lens=None):
        world_size = dist.get_world_size(group)

        if split_seq_lens is not None:
            sizes = list(split_seq_lens)
            tensor_list = [
                torch.empty(
                    *input_.shape[:dim],
                    sizes[i],
                    *input_.shape[dim + 1:],
                    dtype=input_.dtype,
                    device=input_.device,
                )
                for i in range(world_size)
            ]
        else:
            sizes = [None] * world_size
            with profile_range(f'gather sizes for maybe_gather_seq'):
                dist.all_gather_object(sizes, input_.shape[dim], group)
            tensor_list = [
                torch.empty(
                    *input_.shape[:dim],
                    sizes[i],
                    *input_.shape[dim + 1:],
                    dtype=input_.dtype,
                    device=input_.device,
                )
                for i in range(world_size)
            ]

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

        if ctx.split_seq_lens is not None:
            sizes = list(ctx.split_seq_lens)
        else:
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
        input_, dim, group, split_seq_lens = inputs
        ctx.dim = dim
        ctx.group = group
        ctx.split_seq_lens = split_seq_lens
        input_size = list(input_.size())
        ctx.input_size = input_size[dim]
    
    @staticmethod
    def jvp(ctx, input_tangent, dim_tangent, group_tangent, split_seq_lens_tangent):
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
        return _AllGather.apply(input_tangent, ctx.dim, ctx.group, ctx.split_seq_lens)



def all_gather(
    input_: torch.Tensor,
    dim: int = 1,
    group=None,
    split_seq_lens: Optional[Sequence[int]] = None,
):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, group, split_seq_lens)
    # from torch.distributed.nn.functional import _AllGather
    # gathered_tensor = _AllGather.apply(group, input_)
    # return torch.cat(gathered_tensor, dim=dim)
