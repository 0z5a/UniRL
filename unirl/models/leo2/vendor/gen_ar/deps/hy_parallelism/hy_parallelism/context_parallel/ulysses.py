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


import os
import math

import torch
from functools import wraps
from typing import List, Optional, Set, Tuple

from torch.distributed.device_mesh import DeviceMesh

from hy_parallelism.context_parallel.communications import (
    all_gather,
    all_to_all_4D,
)
from hy_parallelism.distributed.communications import gather_obj
from hy_parallelism.parallel_states import get_parallel_state
from hy_parallelism.utils import auto_broadcast

from .cp_info import CPInfo, get_split_seq_info, is_cp_ops_disabled


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
    @wraps(fn)
    def wrapper(x, *args, **kwargs):
        if get_parallel_state().sp > 1:
            return fn(x, *args, **kwargs)
        else:
            return x

    return wrapper


@no_sp_runnable
def maybe_to_split_head(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    if is_cp_ops_disabled():
        return x
    assert x.ndim == 4  # b shard_seq_len, hc, hs
    split_seq_lens = cp_info.seq_lens if cp_info is not None else None
    return all_to_all_4D(
        x,
        get_parallel_state().sp_group,
        scatter_dim=2,
        gather_dim=1,
        split_seq_lens=split_seq_lens,
        async_op=async_op,
    )


@no_sp_runnable
def maybe_to_split_seq(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    if is_cp_ops_disabled():
        return x
    assert x.ndim == 4  # b seq_len, shard_hc, hs
    split_seq_lens = cp_info.seq_lens if cp_info is not None else None
    return all_to_all_4D(
        x,
        get_parallel_state().sp_group,
        scatter_dim=1,
        gather_dim=2,
        split_seq_lens=split_seq_lens,
        async_op=async_op,
    )


class AllReduceGradientsForSequenceParallel(torch.autograd.Function):
    """All-Reduce split input gradients for sequence parallel training."""

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        from hy_parallelism.parallel_states import get_parallel_state
        torch.distributed.all_reduce(grad_output, group=get_parallel_state().sp_group)
        return grad_output


def _sp_scatter(x, dim, return_split_meta=False):
    if is_cp_ops_disabled():
        return (x, None) if return_split_meta else x
    sp_size = get_parallel_state().sp
    sp_rank = get_parallel_state().sp_rank
    seq_len = x.shape[dim]
    cp_info = CPInfo.from_seq_len(seq_len, sp_size, sp_rank) if return_split_meta else None
    chunk_len = math.ceil(seq_len / sp_size)
    if dim == 0:
        out = x[chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    elif dim == 1:
        out = x[:, chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    elif dim == 2:
        out = x[:, :, chunk_len * sp_rank : chunk_len * (sp_rank + 1)]
    else:
        raise ValueError(f'Invalid dimension: {dim}')
    return (out, cp_info) if return_split_meta else out


@no_sp_runnable
def _sp_gather(x, dim: int = 1, cp_info: Optional[CPInfo] = None):
    if is_cp_ops_disabled():
        return auto_broadcast(x, group_src=0, group=get_parallel_state().sp_group)
    split_seq_lens = cp_info.seq_lens if cp_info is not None else None
    return all_gather(x, dim, get_parallel_state().sp_group, split_seq_lens=split_seq_lens)


def maybe_scatter_seq(x, return_split_meta=False, *, cp_info=None):
    if get_parallel_state().sp <= 1:
        return (x, None) if return_split_meta else x
    if is_cp_ops_disabled():
        return (x, None) if return_split_meta else x
    if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
        shape = x.shape
        shapes = gather_obj(shape, group=get_parallel_state().sp_group)
        assert all(x == shapes[0] for x in shapes), f'Shape mismatch in CP group. {shapes}'
    return _sp_scatter(x, 1, return_split_meta=return_split_meta)


@no_sp_runnable
def maybe_gather_seq(x, cp_info: Optional[CPInfo] = None):
    if is_cp_ops_disabled():
        return auto_broadcast(x, group_src=0, group=get_parallel_state().sp_group)
    return _sp_gather(x, dim=1, cp_info=cp_info)


@no_sp_runnable
def maybe_scatter_head(x):
    if is_cp_ops_disabled():
        return x
    return _sp_scatter(x, 2)


@no_sp_runnable
def maybe_gather_head(x):
    if is_cp_ops_disabled():
        return x
    return _sp_gather(x, dim=2)


def wrap_list(t):
    if not isinstance(t, list):
        t = [t]
    return t


all_to_all_sp2hp = maybe_to_split_head
all_to_all_hp2sp = maybe_to_split_seq


def maybe_to_cp_region_num_head(num_head):
    if is_cp_ops_disabled():
        return num_head
    if get_parallel_state().cp_size > 1:
        assert num_head % get_parallel_state().cp_size == 0
        return num_head // get_parallel_state().cp_size
    return num_head


def maybe_to_normal_region_num_head(cp_num_head):
    if is_cp_ops_disabled():
        return cp_num_head
    if get_parallel_state().cp_size > 1:
        return cp_num_head * get_parallel_state().cp_size
    return cp_num_head


maybe_to_normal_reigion_num_head = maybe_to_normal_region_num_head  # for legacy code with typo


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
