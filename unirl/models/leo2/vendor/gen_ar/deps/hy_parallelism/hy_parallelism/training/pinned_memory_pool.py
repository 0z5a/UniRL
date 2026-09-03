"""Pinned memory pool for activation offloading.

Implements a **per-class bump allocator** over pinned host memory.  Each
allocation is routed to a power-of-two size class; within each class,
tensors are packed tightly via a bump pointer inside 512 MiB chunks.
Chunk tails that can't fit the next tensor are recycled via a free-list
for subsequent smaller allocations within the same class.

When a memory budget is configured, low-frequency classes are shrunk
after each iteration to prevent unbounded growth from rare tensor shapes.

A single ``reset()`` at the iteration boundary rewinds every class's bump
pointers without freeing the underlying chunks, so subsequent iterations
re-use the same pinned memory at zero ``cudaHostAlloc`` cost.

Typical usage::

    pool = get_pinned_memory_pool("activation", max_pool_bytes=6 * 1024**3)
    offloader = ActivationOffloader(enabled=True, pool=pool)
    ...
    pool.reset()  # call once per iteration
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import subprocess

import torch

from hy_parallelism.tools.profiling import profile_func

torch_logger = logging.getLogger(__name__)
torch_logger.setLevel(logging.WARNING)
logger = torch_logger
# from loguru import logger


@dataclass
class PoolStats:
    """Runtime statistics of a :class:`PinnedMemoryPool`.

    The first six fields are kept identical to the legacy implementation
    so that any external consumers continue to work unchanged; the
    remaining fields expose slab-specific diagnostics and default to
    zero for backward compatibility.
    """

    total_bytes: int
    used_bytes: int
    num_chunks: int
    num_allocs: int
    num_expansions: int
    peak_used_bytes: int
    num_classes: int = 0
    roundup_waste_bytes: int = 0
    live_used_bytes: int = 0


@dataclass
class _SizeClass:
    """A size class with per-class bump allocation within chunks.

    Multiple tensors pack tightly within each chunk (no per-slot waste).
    Tails that can't fit the next tensor are recycled via a free-list
    so subsequent smaller allocations can reclaim them.
    """

    class_id: int  # the power-of-two upper bound that defines this class
    chunk_size: int  # bytes per chunk (>= min_chunk_bytes)
    chunks: List[torch.Tensor] = field(default_factory=list)
    next_chunk: int = 0
    current_offset: int = 0  # bump pointer within current chunk
    # Recycled tails: [(chunk_idx, offset, remaining_bytes)]
    free_tails: List[List[int]] = field(default_factory=list)
    # Frequency tracking for shrink decisions.
    hit_count: int = 0  # number of iters this class was used
    miss_count: int = 0  # number of iters this class was NOT used


# Smallest size class is 64 MiB.  All allocations <= 64 MiB are served
# from the same class, reducing class proliferation for small tensors.
_MIN_SLOT_BYTES = 64 * 1024 * 1024

_pools: Dict[str, "PinnedMemoryPool"] = {}


def get_pinned_memory_pool(
    name: str,
    min_chunk_bytes: int = 512 * 1024 * 1024,
    max_pool_bytes: int = 200 * 1024 * 1024 * 1024,
) -> "PinnedMemoryPool":
    """Return a named pool, creating it on first call for ``name``.

    Args:
        name: Unique pool identifier within the process.
        min_chunk_bytes: Minimum bytes per cudaHostAlloc expansion.  Only
            applied when the pool is first created.
        max_pool_bytes: Soft cap on total pool size.  Only applied when the
            pool is first created.
    """
    pool = _pools.get(name)
    if pool is None:
        pool = PinnedMemoryPool(min_chunk_bytes=min_chunk_bytes, max_pool_bytes=max_pool_bytes)
        _pools[name] = pool
    return pool


def has_pinned_memory_pool(name: str) -> bool:
    """Return True if a pool with ``name`` has been created."""
    return name in _pools


def maybe_reset_memory_pool(name: str) -> None:
    """Reset the named pool if it has been created, otherwise no-op."""
    pool = _pools.get(name)
    if pool is not None:
        pool.reset()


def reset_pinned_memory_pools() -> None:
    """Destroy all named pools (intended for tests only)."""
    _pools.clear()


def _round_up_to_class(nbytes: int) -> int:
    """Round nbytes up to the next power-of-two size class.

    Coarse granularity reduces the number of distinct classes, which in
    turn limits the per-class high-water-mark accumulation that causes
    pool growth over many iterations with diverse tensor shapes.
    Worst-case per-allocation waste is 100 %; average ~33 %.
    All requests <= _MIN_SLOT_BYTES go to the same class.
    """
    if nbytes <= _MIN_SLOT_BYTES:
        return _MIN_SLOT_BYTES
    # Next power of 2 >= nbytes.
    return 1 << (nbytes - 1).bit_length()


class PinnedMemoryPool:
    """Pinned-memory bump allocator backed by cudaHostAlloc chunks.

    Allocations are routed to a size class by power-of-two rounding;
    each class maintains a bump pointer over a list of 512 MiB pinned
    chunks, with tail recycling for leftover space.  ``reset()`` rewinds
    every class's pointers in O(C) where C is the number of classes.
    An optional memory budget triggers frequency-based shrink to cap
    total pool size.

    Use :func:`get_pinned_memory_pool` to obtain process-level shared
    pools keyed by name.
    """

    ALIGNMENT = 512

    def __init__(
        self,
        min_chunk_bytes: int = 512 * 1024 * 1024,
        max_pool_bytes: int = 80 * 1024 * 1024 * 1024,
    ):
        self._min_chunk_bytes: int = min_chunk_bytes
        self._max_pool_bytes: int = max_pool_bytes
        self._classes: Dict[int, _SizeClass] = {}
        self._registered_streams: Set[torch.cuda.Stream] = set()
        self._num_allocs: int = 0
        self._num_expansions: int = 0
        self._peak_used: int = 0
        self._roundup_waste: int = 0
        self._live_used: int = 0
        self._verbose: bool = False

    @profile_func(msg="pinned_pool_allocate")
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate a pinned CPU tensor from the pool.

        The returned tensor is a view over a chunk owned by the pool and
        remains valid until the next :meth:`reset` call.  Its data
        pointer is aligned to :attr:`ALIGNMENT` bytes so async DMA
        copies (``copy_(..., non_blocking=True)``) work correctly.
        """
        nbytes = _numel(shape) * _element_size(dtype)
        aligned = _align_up(nbytes, self.ALIGNMENT)
        class_id = _round_up_to_class(aligned)

        cls = self._classes.get(class_id)
        if cls is None:
            cls = self._make_class(class_id)
            self._classes[class_id] = cls

        # Fast path: try fitting in current chunk.
        if cls.chunks and cls.current_offset + aligned <= cls.chunk_size:
            chunk = cls.chunks[cls.next_chunk]
            offset = cls.current_offset
            cls.current_offset = offset + aligned
            bump_advanced = True
        else:
            # Current chunk can't fit. Try recycled tails (best-fit).
            fit_idx = -1
            fit_size = 0
            for i, tail in enumerate(cls.free_tails):
                if tail[2] >= aligned and (fit_idx == -1 or tail[2] < fit_size):
                    fit_idx = i
                    fit_size = tail[2]
            if fit_idx >= 0:
                tail = cls.free_tails[fit_idx]
                chunk = cls.chunks[tail[0]]
                offset = tail[1]
                # Update or remove tail entry.
                remaining = tail[2] - aligned
                if remaining >= self.ALIGNMENT:
                    cls.free_tails[fit_idx] = [tail[0], offset + aligned, remaining]
                else:
                    cls.free_tails.pop(fit_idx)
                bump_advanced = False
            else:
                # No recyclable tail fits. Record current tail and advance.
                if cls.chunks:
                    remaining = cls.chunk_size - cls.current_offset
                    if remaining >= self.ALIGNMENT:
                        cls.free_tails.append([cls.next_chunk, cls.current_offset, remaining])
                    cls.next_chunk += 1
                    cls.current_offset = 0
                if cls.next_chunk >= len(cls.chunks):
                    self._expand_class(cls)
                chunk = cls.chunks[cls.next_chunk]
                offset = cls.current_offset
                cls.current_offset = offset + aligned
                bump_advanced = True

        raw = chunk[offset : offset + nbytes]
        tensor = raw.view(dtype).reshape(shape)

        self._num_allocs += 1
        self._live_used += nbytes
        self._roundup_waste += aligned - nbytes
        if bump_advanced:
            self._update_peak()
        return tensor

    def reset(self) -> None:
        """Reclaim all allocations from the current iteration.

        Synchronizes every registered CUDA stream first, then rewinds
        each class's bump pointers.  If a max_pool_bytes budget is set
        and the pool exceeds it, low-frequency classes are shrunk to
        bring the total back under budget.
        """
        if self._verbose and self._num_allocs > 0:
            self._log_stats()
        for stream in self._registered_streams:
            stream.synchronize()

        # Update per-class frequency stats before rewinding.
        for cls in self._classes.values():
            used_this_iter = cls.next_chunk + (1 if cls.current_offset > 0 else 0)
            if used_this_iter > 0:
                cls.hit_count += 1
            else:
                cls.miss_count += 1

        # Rewind bump pointers.
        for cls in self._classes.values():
            cls.next_chunk = 0
            cls.current_offset = 0
            cls.free_tails.clear()
        self._num_allocs = 0
        self._roundup_waste = 0
        self._live_used = 0

        # Budget-based shrink: only triggers when total exceeds the cap.
        if self._max_pool_bytes > 0:
            total = self._total_bytes()
            if total > self._max_pool_bytes:
                self._shrink_to_budget(total)

    def register_stream(self, stream: torch.cuda.Stream) -> None:
        """Register a CUDA stream to be synchronized on :meth:`reset`."""
        self._registered_streams.add(stream)

    def stats(self) -> PoolStats:
        """Return a snapshot of pool statistics."""
        total = 0
        used = 0
        num_chunks = 0
        for cls in self._classes.values():
            total += len(cls.chunks) * cls.chunk_size
            used += cls.next_chunk * cls.chunk_size + cls.current_offset
            num_chunks += len(cls.chunks)
        return PoolStats(
            total_bytes=total,
            used_bytes=used,
            num_chunks=num_chunks,
            num_allocs=self._num_allocs,
            num_expansions=self._num_expansions,
            peak_used_bytes=self._peak_used,
            num_classes=len(self._classes),
            roundup_waste_bytes=self._roundup_waste,
            live_used_bytes=self._live_used,
        )

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    # -- internals --

    def _shrink_to_budget(self, current_total: int) -> None:
        """Release chunks from low-frequency classes until total <= budget.

        All chunks are candidates for release (reset already invalidated
        the data). Classes are sorted by hit_rate: the least frequently
        used class gets its chunks released first.  If a released chunk
        is needed again next iteration, it will be re-allocated on demand
        (one cudaHostAlloc, ~1 ms — acceptable).
        """
        excess = current_total - self._max_pool_bytes

        # Sort classes by hit_rate ascending (least frequent first).
        candidates = []
        for cls in self._classes.values():
            if not cls.chunks:
                continue
            total_iters = cls.hit_count + cls.miss_count
            hit_rate = cls.hit_count / max(total_iters, 1)
            candidates.append((hit_rate, cls))
        candidates.sort(key=lambda x: x[0])

        freed_bytes = 0
        for _, cls in candidates:
            if freed_bytes >= excess:
                break
            while cls.chunks and freed_bytes < excess:
                cls.chunks.pop()
                freed_bytes += cls.chunk_size

        if freed_bytes > 0:
            logger.info(
                "[PinnedMemoryPool] shrink: freed {}, pool {} -> {} (budget {})".format(
                    _fmt_bytes(freed_bytes),
                    _fmt_bytes(current_total),
                    _fmt_bytes(current_total - freed_bytes),
                    _fmt_bytes(self._max_pool_bytes),
                )
            )


    def _make_class(self, class_id: int) -> _SizeClass:
        # Each chunk is at least min_chunk_bytes. Multiple tensors pack
        # tightly via bump pointer within the chunk. For very large tensors
        # (> min_chunk_bytes), chunk_size = class_id to fit at least one.
        chunk_size = max(self._min_chunk_bytes, class_id)
        return _SizeClass(class_id=class_id, chunk_size=chunk_size)

    def _expand_class(self, cls: _SizeClass) -> None:
        nbytes = cls.chunk_size

        # Budget check: if expanding would exceed the limit, try to reclaim
        # unused chunks from other classes first (mid-iteration shrink).
        if self._max_pool_bytes > 0:
            total_after = self._total_bytes() + nbytes
            if total_after > self._max_pool_bytes:
                self._shrink_unused(total_after - self._max_pool_bytes)
                total_after = self._total_bytes() + nbytes
                if total_after > self._max_pool_bytes:
                    raise RuntimeError(
                        f"PinnedMemoryPool: expanding would exceed budget "
                        f"({_fmt_bytes(total_after)} > {_fmt_bytes(self._max_pool_bytes)}). "
                    )

        try:
            chunk = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            try:
                subprocess.run(
                    "echo 1 > /proc/sys/vm/drop_caches",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.warning(
                    "[PinnedMemoryPool] pinned alloc failed, dropped caches, retrying"
                )
            except Exception as drop_err:
                logger.warning(
                    f"[PinnedMemoryPool] drop_caches failed: {drop_err}, retrying anyway"
                )
            try:
                chunk = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as exc:
                # echo 1 > /proc/sys/vm/drop_caches
                # echo 1 > /proc/sys/vm/compact_memory
                os.system(f'cat /proc/{os.getpid()}/maps')
                os.system(f'sysctl vm.max_map_count')
                os.system(f'cat /proc/{os.getpid()}/maps | wc -l')

                def _run_and_log(cmd, label=None):
                    try:
                        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
                        if label:
                            logger.warning(f"[PinnedMemoryPool][OOM][{label}] {output.strip()}")
                        else:
                            logger.warning(f"[PinnedMemoryPool][OOM] {output.strip()}")
                    except Exception as err:
                        logger.warning(f"[PinnedMemoryPool][OOM][{label if label else cmd}] Failed: {err}")

                # BAR1 Used 是否接近 Total?
                # _run_and_log("nvidia-smi -q | grep -A3 BAR1", label="nvidia-smi_BAR1")

                # /proc/<PID>/maps | wc -l, 检查条数
                _run_and_log(f"cat /proc/{os.getpid()}/maps | wc -l", label="maps_wc_l")
                # grep -i vmalloc /proc/meminfo           # VmallocUsed 是否顶到 VmallocTotal
                _run_and_log("grep -i vmalloc /proc/meminfo", label="vmalloc")
                # sysctl vm.max_map_count                 # 上限
                _run_and_log("sysctl vm.max_map_count", label="max_map_count")
                # cat /proc/cmdline | grep -o 'intel_iommu=[^ ]*' # IOMMU setting
                _run_and_log("cat /proc/cmdline | grep -o 'intel_iommu=[^ ]*'", label="iommu")
                # 关键:进程真实的锁页额度(容器里 ulimit 命令可能骗人,直接读 /proc)
                # 进程最大可锁住内存额度
                _run_and_log(f"grep -i 'Max locked memory' /proc/{os.getpid()}/limits", label="max_locked_memory")
                # 当前已锁住的内存量,看是否顶到额度
                _run_and_log(f"grep -i 'VmLck' /proc/{os.getpid()}/status", label="vmlck")
                # 系统范围 mlocked 内存
                _run_and_log("cat /proc/meminfo | grep -i Mlocked", label="meminfo_mlocked")
                _run_and_log("dmesg -T | tail -n 100", label="dmesg")

                raise RuntimeError(
                    f"PinnedMemoryPool: pinned alloc failed for {_fmt_bytes(nbytes)} "
                    f"({nbytes} bytes): {exc}"
                ) from exc
                import loguru
                loguru.logger.warning(
                    f"PinnedMemoryPool: pinned alloc failed, falling back to pageable"
                )
                self._log_stats(logger=loguru.logger)

                chunk = torch.empty(nbytes, dtype=torch.uint8)

        cls.chunks.append(chunk)
        self._num_expansions += 1

        total = self._total_bytes()
        live = self._live_used
        frag = (total - live) / total * 100.0 if total > 0 else 0.0
        total_chunks = sum(len(c.chunks) for c in self._classes.values())
        logger.info(
            "[PinnedMemoryPool] expand #{} cls={} (chunk={}) chunks_in_cls={} | pool: classes={} chunks={} total={} live={} frag={:.1f}%".format(
                self._num_expansions,
                _fmt_bytes(cls.class_id),
                _fmt_bytes(nbytes),
                len(cls.chunks),
                len(self._classes),
                total_chunks,
                _fmt_bytes(total),
                _fmt_bytes(live),
                frag,
            )
        )


    def _shrink_unused(self, needed: int) -> None:
        """Release chunks not used in the current iteration to free space.

        Only chunks beyond each class's current bump pointer are safe to
        release — they contain no live tensor data this iteration.
        """
        freed = 0
        for other_cls in self._classes.values():
            if freed >= needed:
                break
            # Determine how many trailing chunks are unused this iter.
            if other_cls.next_chunk == 0 and other_cls.current_offset == 0:
                # Class not touched this iter — all chunks are unused.
                safe_keep = 0
            else:
                # Chunks 0..next_chunk contain live data; the rest are unused.
                safe_keep = other_cls.next_chunk + 1
            while len(other_cls.chunks) > safe_keep and freed < needed:
                other_cls.chunks.pop()
                freed += other_cls.chunk_size
        if freed > 0:
            logger.info(
                f"[PinnedMemoryPool] mid-iter shrink: freed {_fmt_bytes(freed)} from unused chunks"
            )


    def _total_bytes(self) -> int:
        return sum(len(cls.chunks) * cls.chunk_size for cls in self._classes.values())

    def _used_bytes(self) -> int:
        return sum(
            cls.next_chunk * cls.chunk_size + cls.current_offset for cls in self._classes.values()
        )

    def _update_peak(self) -> None:
        used = self._used_bytes()
        if used > self._peak_used:
            self._peak_used = used

    def _log_stats(self, logger=None) -> None:
        s = self.stats()
        # frag = (total pool bytes - live caller bytes) / total pool bytes.
        # This captures every kind of overhead (alignment, class round-up,
        # last-chunk partial fill) in one number.
        frag = (s.total_bytes - s.live_used_bytes) / max(s.total_bytes, 1) * 100.0
        util = s.used_bytes / max(s.total_bytes, 1) * 100.0
        if logger is None:
            logger = torch_logger
        logger.info(
            "[PinnedMemoryPool] reset: live={} used={} total={} "
            "(util={:.1f}% frag={:.1f}% roundup={}) | classes={} chunks={} "
            "allocs={} expansions={}".format(
                _fmt_bytes(s.live_used_bytes),
                _fmt_bytes(s.used_bytes),
                _fmt_bytes(s.total_bytes),
                util,
                frag,
                _fmt_bytes(s.roundup_waste_bytes),
                s.num_classes,
                s.num_chunks,
                s.num_allocs,
                s.num_expansions,
            )
        )



def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def _fmt_bytes(n: int) -> str:
    """Compact byte formatter for log lines (B / KiB / MiB / GiB)."""
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f}GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}KiB"
    return f"{n}B"
