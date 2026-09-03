import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGather,
    DefaultAllGather,
)
from torch.types import _device

# An internal map from device to the symmetric memory pool for that device.
_symm_mem_pools: dict[_device, torch.cuda.MemPool] = {}

# (device, dtype, shape) -> ([buffers], next_idx)
_staging_rings: dict[tuple[Any, ...], tuple[list[torch.Tensor], int]] = {}


def get_mem_pool(device: _device) -> torch.cuda.MemPool:
    """
    Get the symmetric memory pool for a given device. If not found, create a new
    pool.

    The tensor allocations with this pool must be symmetric across ranks.  The
    allocated tensors can be used with symmetric operations, for example,
    operations defined under `torch.ops.symm_mem`.

    Args:
        device (`torch.device` or str): the device for which to get the symmetric memory pool.

    Returns:
        `torch.cuda.MemPool`: the symmetric memory pool for the given device.

    Example::

        >>> # doctest: +SKIP
        >>> pool = torch.distributed._symmetric_memory.get_mem_pool("cuda:0")
        >>> with torch.cuda.use_mem_pool(pool):
        >>>     tensor = torch.randn(1000, device="cuda:0")
        >>> tensor = torch.ops.symm_mem.one_shot_all_reduce(tensor, "sum", group_name)

    """
    # This function is a wrapper around the `torch.cuda.MemPool` constructor.
    # Due to special requirements of SymmetricMemory, we preset certain options for the pool.
    # - use_on_oom=False: we don't want to lend the space of the pool for
    # non-symmetric allocations because this could desync the allocation state
    # across ranks.
    # - no_split=True: we don't want to split segments, because today a segment
    # is associated with a signal pad, if two allocated tensors share a segment
    # and their kernels concurrently use (the same) signal pad, this could cause
    # undefined behaviors. We could consider relaxing this in the future if we
    # establish stream tracking and implicit synchronization around an
    # allocation.
    if device not in _symm_mem_pools:
        allocator = symm_mem.get_mempool_allocator(device)
        # Create a new pool with the given allocator and the preset options.
        _symm_mem_pools[device] = torch.cuda.MemPool(
            allocator,
            use_on_oom=False,
            no_split=True,
        )

    return _symm_mem_pools[device]


def init_nccl_symm_heap(pg, device, *, backend="NCCL"):
    world_name = dist.group.WORLD.group_name
    pg_name = pg.group_name
    symm_mem.set_backend(backend)
    symm_mem.enable_symm_mem_for_group(world_name)
    symm_mem.enable_symm_mem_for_group(pg_name)
    dist.barrier()
    t = symm_mem.empty(4096, dtype=torch.uint8, device=device)
    symm_mem.rendezvous(t, group=pg_name)
    del t
    torch.cuda.synchronize()
    dist.barrier(pg)
    t = symm_mem.empty(4096, dtype=torch.uint8, device=device)
    import loguru
    loguru.logger.info(f"symm heap ptr: {t.data_ptr()}")


class SymmMemAllocMixin:
    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
        ring_n: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        self._group = group
        self._ring_n = ring_n
        symm_mem.set_backend(backend)
        # NCCL symm mem alloc/rendezvous require group info in C++ group_info_map.
        # WORLD ("0") is needed at alloc time; the FSDP shard group at rendezvous.
        # pytorch 2.10.rc6 需要这个，后面好像弃用了
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        symm_mem.enable_symm_mem_for_group(group.group_name)
        # Force initialization of communicator; otherwise, the rendezvous may
        # see empty communicator.
        # TODO: Remove this, maybe by warning user to perform eager dist init.
        # For now, it is okay since it isjust a one-time cost at init.
        dist.barrier(group=group)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        shape = tuple(int(s) for s in size)

        # Fixed N-slot ring per (device, dtype, shape): all ranks advance the
        # same slot index, avoiding MemPool min-ptr reuse split across ranks.
        if self._ring_n >= 2:
            key = (str(device), dtype, shape)
            entry = _staging_rings.get(key)
            if entry is None:
                mempool = get_mem_pool(device)
                bufs: list[torch.Tensor] = []
                with torch.cuda.use_mem_pool(mempool):
                    for _ in range(self._ring_n):
                        bufs.append(torch.empty(shape, dtype=dtype, device=device))
                entry = (bufs, 0)
                _staging_rings[key] = entry
            bufs, idx = entry
            _staging_rings[key] = (bufs, (idx + 1) % len(bufs))
            return bufs[idx]

        mempool = get_mem_pool(device)
        with torch.cuda.use_mem_pool(mempool):
            return torch.empty(shape, dtype=dtype, device=device)


class SymmMemAllGather(SymmMemAllocMixin, AllGather):

    def __init__(
        self,
        group: dist.ProcessGroup,
        backend: Literal["NCCL"] = "NCCL",
        ring_n: int = 0,
    ) -> None:
        super().__init__(group, backend, ring_n=ring_n)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> dist.Work | None:
        # We are doing inplace all-gather, so we need to rendezvous the output tensor only
        symm_mem.rendezvous(output_tensor, group=group.group_name)
        # Calling regular all-gather would already cause libraries like NCCL to
        # use its optimized all-gather implementation for symmetric memory:
        # - Copy Engine All-Gather (when zero-CTA policy is enabled)
        # - Symmetric Kernel All-Gather (when zero-CTA policy is not enabled)
        return dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )


# fsdp_param_group
def set_symm_mem(
    self, backend: Literal["NCCL"] = "NCCL", ring_n: int = 0
) -> None:
    if not isinstance(self._all_gather_comm, (DefaultAllGather | SymmMemAllGather)):
        raise AssertionError(
            "cannot call set_symm_mem() "
            f"when all gather comm is custom: {self._all_gather_comm.__class__.__name__}"
        )
    self._all_gather_comm = SymmMemAllGather(
        self._all_gather_process_group, backend, ring_n=ring_n
    )


# fsdp module
def set_symm_mem_for_comm(
    self,
    backend: Literal["NCCL"] = "NCCL",
    recursive: bool = True,
    ring_n: int = 0,
) -> None:
    """
    Sets the symmetric memory (``symm_mem``) backend for allocating the
    staging buffers used in all-gather collectives. This allows NCCL to use
    optimized all-gather implementations via symmetric memory. Such
    optimization may depend on the topology of the system.  For single node,
    Copy Engine All-Gather may be used. For multi-node, Symmetric Kernel
    All-Gather may be used.

    To enable Copy Engine All-Gather, you need to set the NCCL process group
    with the zero-CTA policy.
    ```python
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
    ```
    Alternatively, you can set the environment variable `NCCL_CTA_POLICY` to 2.
    ```bash
    export NCCL_CTA_POLICY=2
    ```
    For more details, see [Copy Engine
    Collectives](https://docs.pytorch.org/docs/2.11/symmetric_memory.html#copy-engine-collectives).

    This cannot be used together with :meth:`set_custom_all_gather` or
    :meth:`set_custom_reduce_scatter`.

    Args:
        backend (str): The symmetric memory backend to use. Defaults to
            ``"NCCL"``. Currently, only ``"NCCL"`` is supported.
        ring_n (int): When ``>= 2``, each staging shape uses a fixed ring of
            ``N`` buffers and round-robin slot selection instead of MemPool
            free-list reuse (avoids cross-rank generation mismatch). ``0``
            disables and uses MemPool only.
    """
    from torch.distributed.fsdp import FSDPModule
    if recursive:
        modules = self.modules()
    else:
        modules = [self]

    for module in modules:
        if not isinstance(module, FSDPModule):
            continue

        state = module._get_fsdp_state()
        if hasattr(state, '_fsdp_param_groups'):
            for fsdp_param_group in state._fsdp_param_groups:
                set_symm_mem(fsdp_param_group, backend, ring_n=ring_n)
        else:
            set_symm_mem(state._fsdp_param_group, backend, ring_n=ring_n)
