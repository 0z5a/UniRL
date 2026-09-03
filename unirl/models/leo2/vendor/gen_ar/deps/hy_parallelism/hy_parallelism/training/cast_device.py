"""Device casting utilities for nested containers (list, dict, tuple, dataclass, …).

When the target device is CPU, :func:`cast_to_device` supports ``pin_memory``
and a pinned-memory pool (name or
:class:`~hy_parallelism.training.pinned_memory_pool.PinnedMemoryPool`).
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Union, overload

import torch
from torch.nn.utils.rnn import PackedSequence

from hy_parallelism.training.pinned_memory_pool import (
    PinnedMemoryPool,
    get_pinned_memory_pool,
)

__all__ = ["CopyWork", "cast_to_device"]

R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)
Q = TypeVar("Q")
PoolRef = Union[str, PinnedMemoryPool, None]

_copy_streams: dict[int, torch.cuda.Stream] = {}


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _append_copy_event(
    copy_events: Optional[list[torch.cuda.Event]],
    stream: torch.cuda.Stream,
    device_index: int,
) -> None:
    if copy_events is None:
        return
    with torch.cuda.device(device_index):
        event = torch.cuda.Event()
        event.record(stream)
        copy_events.append(event)


@dataclass
class CopyWork:
    """Async device-copy handle, modeled after ``c10d.Work``.

    ``value`` holds the cast result immediately; call :meth:`wait` before using
    CUDA outputs on the current stream or reusing source GPU tensors.
    """

    value: Any
    _events: list[torch.cuda.Event] = field(default_factory=list)

    def wait(self) -> Any:
        """Block the current CUDA stream until all copies complete."""
        if not self._events:
            return self.value
        stream = torch.cuda.current_stream()
        for event in self._events:
            stream.wait_event(event)
        # Outputs allocated on the copy stream must be pinned to the consumer
        # stream so the caching allocator cannot reuse their storage early.
        def _record(t: torch.Tensor) -> torch.Tensor:
            if t.is_cuda:
                t.record_stream(stream)
            return t

        _apply_to_tensors(_record, self.value)
        return self.value


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Q], container: torch.Tensor) -> Q: ...


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R: ...


def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: Any) -> Any:
    def apply(x: Any) -> Any:
        from torch.nn.parallel.scatter_gather import _is_namedtuple

        if isinstance(x, torch.Tensor):
            return fn(x)
        if hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            changes = {
                f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
            }
            return dataclasses.replace(dc, **changes)
        if isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        if isinstance(x, PackedSequence):
            x.data = apply(x.data)
            return x
        if isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        if _is_namedtuple(x):
            return type(x)(*(apply(el) for el in x))
        if isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        return x

    return apply(container)


def _resolve_pin_memory_pool(
    pool: PoolRef,
    *,
    pin_memory: bool = False,
) -> Optional[PinnedMemoryPool]:
    if pool is not None:
        if isinstance(pool, str):
            return get_pinned_memory_pool(pool)
        return pool
    if pin_memory:
        return None
    return None


def _get_copy_stream(
    tensor: torch.Tensor, target_device: torch.device
) -> torch.cuda.Stream:
    """Return a cached side stream for a device copy.

    D2H uses a background stream on the source GPU; other copies use the target
    device stream.
    """
    if target_device.type == "cpu":
        device_index = _cuda_device_index(tensor.device)
    else:
        device_index = _cuda_device_index(target_device)

    stream = _copy_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream(device=device_index)
        _copy_streams[device_index] = stream
    return stream



def _cast_tensor_to_device(
    tensor: torch.Tensor,
    target_device: torch.device,
    *,
    use_side_stream_for_tensor_copies: bool = True,
    pin_memory: bool = False,
    pool: PoolRef = None,
    async_op: bool = False,
    copy_events: Optional[list[torch.cuda.Event]] = None,
) -> torch.Tensor:
    if tensor.device == target_device:
        return tensor

    if target_device.type == "cpu" and pin_memory:
        with torch.enable_grad():
            resolved_pool = _resolve_pin_memory_pool(pool, pin_memory=pin_memory)
            if resolved_pool is not None:
                cpu_buf = resolved_pool.allocate(tensor.shape, tensor.dtype)
            else:
                cpu_buf = torch.empty_like(tensor, device="cpu", pin_memory=True)

            non_blocking = cpu_buf.is_pinned()
            device_mod = getattr(torch, tensor.device.type, None)
            if (
                use_side_stream_for_tensor_copies
                and tensor.device.type != "cpu"
                and device_mod is not None
            ):
                device_index = _cuda_device_index(tensor.device)
                stream = _get_copy_stream(tensor, target_device)
                with device_mod.device(device_index):
                    caller_stream = device_mod.current_stream()
                    stream.wait_stream(caller_stream)
                    with device_mod.stream(stream):
                        cpu_buf.copy_(tensor, non_blocking=non_blocking)
                        if async_op:
                            _append_copy_event(copy_events, stream, device_index)
                    if async_op:
                        tensor.record_stream(stream)
                        if resolved_pool is not None:
                            resolved_pool.register_stream(stream)
                    else:
                        caller_stream.wait_stream(stream)
                        if resolved_pool is not None:
                            resolved_pool.register_stream(stream)
            else:
                if tensor.device.type != "cpu" and device_mod is not None:
                    device_index = _cuda_device_index(tensor.device)
                    with device_mod.device(device_index):
                        cpu_buf.copy_(tensor, non_blocking=non_blocking)
                        if async_op and non_blocking:
                            stream = device_mod.current_stream()
                            _append_copy_event(copy_events, stream, device_index)
                            tensor.record_stream(stream)
                else:
                    cpu_buf.copy_(tensor, non_blocking=non_blocking)

            assert cpu_buf.requires_grad == tensor.requires_grad, (
                f"{cpu_buf.requires_grad=} {tensor.requires_grad=}"
            )
        return cpu_buf

    if not use_side_stream_for_tensor_copies:
        return tensor.to(target_device)

    device_mod = getattr(torch, tensor.device.type, None)
    if tensor.device.type == "cpu" or device_mod is None:
        return tensor.to(target_device)

    device_index = _cuda_device_index(tensor.device)
    stream = _get_copy_stream(tensor, target_device)
    with device_mod.device(device_index):
        caller_stream = device_mod.current_stream()
        stream.wait_stream(caller_stream)
        with device_mod.stream(stream):
            output = tensor.to(target_device)
            if async_op:
                _append_copy_event(copy_events, stream, device_index)
        if async_op:
            tensor.record_stream(stream)
        else:
            caller_stream.wait_stream(stream)
            output.record_stream(caller_stream)
    return output


def cast_to_device(
    container: Any,
    target_device: torch.device | str = "cuda",
    *,
    use_side_stream_for_tensor_copies: bool = False,
    pin_memory: bool = True,
    pool: PoolRef = None,
    async_op: bool = False,
) -> Any | CopyWork:
    """Move every tensor in ``container`` to ``target_device``.

    ``pin_memory`` and ``pool`` are only consulted when ``target_device`` is CPU;
    they control how the destination CPU buffer is allocated (pageable,
    ``pin_memory=True``, or a pinned-memory pool). Other device paths ignore them.

    When ``async_op=False`` (default), copies are synchronized before return.
    When ``async_op=True``, returns a :class:`CopyWork` handle; call
    :meth:`CopyWork.wait` before consuming the result. ``async_op`` requires
    ``use_side_stream_for_tensor_copies=True``.
    """
    if async_op and not use_side_stream_for_tensor_copies:
        raise ValueError(
            "async_op=True requires use_side_stream_for_tensor_copies=True"
        )
    device = torch.device(target_device)
    copy_events: list[torch.cuda.Event] = []
    value = _apply_to_tensors(
        lambda t: _cast_tensor_to_device(
            t,
            device,
            use_side_stream_for_tensor_copies=use_side_stream_for_tensor_copies,
            pin_memory=pin_memory,
            pool=pool,
            async_op=async_op,
            copy_events=copy_events if async_op else None,
        ),
        container,
    )
    if async_op:
        return CopyWork(value, copy_events)
    return value


def _recursive_to(
    inputs: Any,
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool = True,
    pin_memory: bool = False,
    pool: PoolRef = None,
) -> Any:
    def to_map(obj: Any) -> list[Any]:
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            if isinstance(obj, PackedSequence):
                if obj.data.device == target_device:
                    return [obj]
                moved = _cast_tensor_to_device(
                    obj.data,
                    target_device,
                    use_side_stream_for_tensor_copies=use_side_stream_for_tensor_copies,
                    pin_memory=pin_memory,
                    pool=pool,
                )
                if moved is not obj.data:
                    obj.data = moved
                return [obj]

            return [
                _cast_tensor_to_device(
                    obj,
                    target_device,
                    use_side_stream_for_tensor_copies=use_side_stream_for_tensor_copies,
                    pin_memory=pin_memory,
                    pool=pool,
                )
            ]

        from torch.nn.parallel.scatter_gather import _is_namedtuple

        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]

    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]],
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool = True,
    pin_memory: bool = False,
    pool: PoolRef = None,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    moved_inputs = (
        _recursive_to(
            inputs,
            target_device,
            use_side_stream_for_tensor_copies,
            pin_memory,
            pool,
        )
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(
            kwargs,
            target_device,
            use_side_stream_for_tensor_copies,
            pin_memory,
            pool,
        )
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)

def async_offload(
    container: Any,
    target_device: torch.device | str,
    *,
    pool: PoolRef = None,
) -> Any | CopyWork:
    if pool is None:
        pool = 'async_offload'
    return cast_to_device(container, target_device, use_side_stream_for_tensor_copies=True, pin_memory=True, pool=pool, async_op=True)

def async_onload(
    container: Any,
    target_device: torch.device | str,
) -> Any | CopyWork:
    return cast_to_device(container, target_device, use_side_stream_for_tensor_copies=True, async_op=True)