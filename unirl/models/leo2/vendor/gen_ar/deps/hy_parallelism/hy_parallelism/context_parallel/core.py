# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

# Deepspeed Ulysses gradient scaling implementation
# https://github.com/deepspeedai/DeepSpeed/issues/5248
# https://github.com/microsoft/DeepSpeed/blob/535a908f1b60f819df4ccf1071f7c917c39dabbe/deepspeed/runtime/zero/stage_1_and_2.py#L1107

# Deepspeed Ulysees test
# https://github.com/deepspeedai/DeepSpeed/blob/master/tests/unit/ulysses_alst/test_ulysses_sp_hf.py


# Unified cp diffusers implementation
# Including discussion about summing the gradients across the sp ranks for ulysses
# https://github.com/huggingface/diffusers/issues/12570


# PyTorch experimental sp support
# https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246



import math

import torch
from functools import wraps
from typing import Any, List, Set, Tuple, Union

from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from hy_parallelism.context_parallel.communications import (
    all_gather,
    all_gather_pad,
    all_to_all_4D,
    set_enable_sp_padding,
)
from hy_parallelism.parallel_states import get_parallel_state

# 提供一个全局控制，实现伪 CP，即虽然有 cp mesh, 但是不做任何切分，不做任何通信
# 使用场景是：RL 需要使用 CP 在生图加速，但生文只有一个 token，额外的通信影响速度
# 重新创建单独的 mesh 容易造成 text rollout 和 image rollout 的 mismatch
# 因此短期 workaround 是禁用 CP 的通信，但仍然保留 CP 的 mesh，虽然生文无法获得
# 和不开 CP 一样的吞吐，但至少不要带来额外通信拖累速度
_DISABLE_CP_OPS = False
def set_disable_cp_ops(disable: bool):
    global _DISABLE_CP_OPS
    _DISABLE_CP_OPS = disable


class SequentialSharder:
    """
    This load balancer chunks the buffer into cp_world_size and rank0 gets
    0th shard, rank1 gets 1st shard, ...
    So this doesn't have any load balancing effect when using the causal masking.
    """

    @classmethod
    def shard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        assert buffer.size()[seq_dim] % mesh.size() == 0
        return buffer.chunk(mesh.size(), dim=seq_dim)[mesh.get_local_rank()]

    @classmethod
    def unshard(
        cls, buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
    ) -> torch.Tensor:
        import torch.distributed._functional_collectives as ft_c
        buffer = buffer.contiguous()
        all_buffers = [torch.empty_like(buffer) for _ in range(mesh.size())]
        ft_c.all_gather_inplace(all_buffers, buffer, mesh)
        return torch.cat(all_buffers, dim=seq_dim)




def no_sp_runnable(fn):
    # make the input object unchanged when sp is not enabled

    @wraps(fn)
    def wrapper(x, *args, **kwargs):
        if get_parallel_state().sp > 1:
            return fn(x, *args, **kwargs)
        else:
            return x

    return wrapper


@no_sp_runnable
def maybe_to_split_head(x, input_pad=None):
    if _DISABLE_CP_OPS:
        return x
    assert x.ndim == 4  # b shard_seq_len, hc, hs
    return all_to_all_4D(x, get_parallel_state().sp_group, scatter_dim=2, gather_dim=1, input_pad=input_pad)

    from hy_parallelism.context_parallel.communications import _all_to_all_4D
    return _all_to_all_4D(x, group=get_parallel_state().sp_group, scatter_idx=2, gather_idx=1, input_pad=input_pad)

@no_sp_runnable
def maybe_to_split_seq(x, input_pad=None):
    if _DISABLE_CP_OPS:
        return x
    assert x.ndim == 4  # b seq_len, shard_hc, hs
    return all_to_all_4D(x, get_parallel_state().sp_group, scatter_dim=1, gather_dim=2, input_pad=input_pad)


    from hy_parallelism.context_parallel.communications import _all_to_all_4D
    return _all_to_all_4D(x, group=get_parallel_state().sp_group, scatter_idx=1, gather_idx=2, input_pad=input_pad)





class AllReduceGradientsForSequenceParallel(torch.autograd.Function):
    """ All-Reduce split input gradients for sequence parallel training. """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        from hy_parallelism.parallel_states import get_parallel_state
        torch.distributed.all_reduce(grad_output, group=get_parallel_state().sp_group)
        return grad_output
                
class CPInfo:
    ...

def get_split_seq_info(seq_len, sp_size, sp_rank):
    """
    splitting logic is like: 
        x[chunk_len * sp_rank : chunk_len * (sp_rank + 1)]

    This function is used to get the size of the chunk for the given sp_rank.
    """
    assert sp_size > 0, f"sp_size must be positive, got {sp_size}"
    assert 0 <= sp_rank < sp_size, f"sp_rank ({sp_rank}) must be in [0, {sp_size})"
    assert seq_len >= 0, f"seq_len must be non-negative, got {seq_len}"

    chunk_len = math.ceil(seq_len / sp_size)
    start = chunk_len * sp_rank
    end = min(chunk_len * (sp_rank + 1), seq_len)

    length = max(0, end - start)
    return {'start': start, 'end': end, 'length': length}

@no_sp_runnable
def _sp_scatter(x, dim, return_pad=False):
    if _DISABLE_CP_OPS:
        return x
    sp_size = get_parallel_state().sp
    sp_rank = get_parallel_state().sp_rank
    if x.shape[dim] % sp_size != 0:
        n_token = x.shape[dim]
        # assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'{"Token" if dim == 1 else "Head size"} ({n_token}) is too short for SP {sp_size}'
    chunk_len = math.ceil(x.shape[dim] / sp_size)
    assert not return_pad, 'return_pad is not supported anymore.'
    if dim == 0:
        return x[chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    elif dim == 1:
        return x[:, chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    elif dim == 2:
        return x[:, :, chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    else:
        raise ValueError(f'Invalid dimension: {dim}')
    chunks = torch.chunk(x, sp_size, dim=dim) # can produce fewer chunks
    if return_pad:
        seq_lens = [chunk.shape[dim] for chunk in chunks]
        if seq_lens[-1] != seq_lens[0] :
            assert seq_lens[0] > seq_lens[-1]
            input_pad = seq_lens[0] - seq_lens[-1]
        else:
            input_pad = 0

        return chunks[sp_rank], input_pad
    else:
        return chunks[sp_rank]


@no_sp_runnable
def _sp_gather(x, dim: int = 1, pad: Union[int, None] = None):
    if _DISABLE_CP_OPS:
        dist.broadcast(x, src=0, group=get_parallel_state().sp_group)
        return x
    return all_gather(x, dim, pad, get_parallel_state().sp_group)

@no_sp_runnable
def _sp_gather_pad(x, dim: int = 1, pad: Union[int, None] = None):
    raise DeprecationWarning('_sp_gather_pad is deprecated, use _sp_gather instead')
    if _DISABLE_CP_OPS:
        return x
    return all_gather_pad(x, dim, pad, get_parallel_state().sp_group)

@no_sp_runnable
def maybe_scatter_seq(x, return_pad=False):
    if _DISABLE_CP_OPS:
        return x
    return _sp_scatter(x, 1, return_pad=return_pad)


@no_sp_runnable
def maybe_gather_seq(x, pad=None):
    if _DISABLE_CP_OPS:
        # 中间缺乏同步，强加这个保证模型端到端输出一致
        dist.broadcast(x, group_src=0, group=get_parallel_state().sp_group)
        return x
    return _sp_gather(x, dim=1, pad=pad)


@no_sp_runnable
def maybe_scatter_head(x):
    if _DISABLE_CP_OPS:
        return x
    return _sp_scatter(x, 2)


@no_sp_runnable
def maybe_gather_head(x, pad=None):
    if _DISABLE_CP_OPS:
        return x
    return _sp_gather(x, dim=2, pad=pad)


@no_sp_runnable
def maybe_gather_seq_pad(x, pad):
    """历史遗留代码，不建议使用，这里保留只为了支持一些旧服务"""
    raise DeprecationWarning('maybe_gather_seq_pad is deprecated, use maybe_gather_seq instead')
    return _sp_gather_pad(x, dim=1, pad=pad)

def _sp_scatter_list(x, dim):
    """历史遗留代码，不建议使用，这里保留只为了支持一些旧服务"""
    raise DeprecationWarning('get_chunk_seqs_list is deprecated, use maybe_scatter_seq instead')
    sp_size = get_parallel_state().sp
    if x.shape[dim] % sp_size != 0:
        n_token = x.shape[dim]
        assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'{"Token" if dim == 1 else "Head size"} ({n_token}) is too short for SP {sp_size}'
    chunks = torch.chunk(x, sp_size, dim=dim)
    return chunks

@no_sp_runnable
def get_chunk_seqs_list(x):
    """历史遗留代码，不建议使用，这里保留只为了支持一些旧服务"""
    raise DeprecationWarning('get_chunk_seqs_list is deprecated, use maybe_scatter_seq instead')
    return _sp_scatter_list(x, 1)

def wrap_list(t):
    if not isinstance(t, list):
        t = [t]
    return t

all_to_all_sp2hp = maybe_to_split_head
all_to_all_hp2sp = maybe_to_split_seq

def maybe_to_cp_region_num_head(num_head):
    if _DISABLE_CP_OPS:
        return num_head
    if get_parallel_state().cp_size > 1:
        assert num_head % get_parallel_state().cp_size == 0
        return num_head // get_parallel_state().cp_size
    return num_head

def maybe_to_normal_region_num_head(cp_num_head):
    if _DISABLE_CP_OPS:
        return cp_num_head
    if get_parallel_state().cp_size > 1:
        return cp_num_head * get_parallel_state().cp_size
    return cp_num_head

maybe_to_normal_reigion_num_head = maybe_to_normal_region_num_head # for legacy code with typo


# Pytorch experimental sp support
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
):

    raise NotImplementedError(
        "pytorch experimental context parallel only support ring attention with is_causal=True."
        "For flex attention, create_cp_block_mask should be called."
    )
    try:
        from torch.distributed.tensor.experimental import context_parallel
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )

