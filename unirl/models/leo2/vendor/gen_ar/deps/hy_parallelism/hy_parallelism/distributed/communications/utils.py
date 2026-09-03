from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union, overload

import torch

__all__ = ["CudaStreamWork", "run_on_async_stream"]

T = TypeVar("T")
U = TypeVar("U")

_async_streams: dict[int, torch.cuda.Stream] = {}


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _get_async_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    stream = _async_streams.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream(device=device_index)
        _async_streams[device_index] = stream
    return stream


@dataclass
class CudaStreamWork(Generic[T]):
    """Deferred CUDA work on a side stream.

    ``value`` is available immediately; call :meth:`wait` before using outputs
    or reusing inputs recorded on the side stream.
    """

    value: T
    _done_event: torch.cuda.Event
    _device: torch.device
    on_wait: Optional[Callable[[T], U]] = None

    def wait(self) -> Union[T, U]:
        current = torch.cuda.current_stream(device=self._device)
        current.wait_event(self._done_event)
        # Output was allocated on the side stream; pin it to the consumer stream
        # so the caching allocator cannot reuse the storage while still in use.
        if isinstance(self.value, torch.Tensor) and self.value.is_cuda:
            self.value.record_stream(current)
        if self.on_wait is not None:
            return self.on_wait(self.value)
        return self.value


@overload
def run_on_async_stream(
    fn: Callable[[], T],
    device: torch.device,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    record_tensors: Optional[Sequence[torch.Tensor]] = None,
    on_wait: None = None,
) -> CudaStreamWork[T]: ...


@overload
def run_on_async_stream(
    fn: Callable[[], T],
    device: torch.device,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    record_tensors: Optional[Sequence[torch.Tensor]] = None,
    on_wait: Callable[[T], U],
) -> CudaStreamWork[T]: ...


def run_on_async_stream(
    fn: Callable[[], T],
    device: torch.device,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    record_tensors: Optional[Sequence[torch.Tensor]] = None,
    on_wait: Optional[Callable[[T], U]] = None,
) -> CudaStreamWork[T]:
    """Run ``fn`` on a side CUDA stream and return a handle to wait for completion.

    If ``on_wait`` is provided, it is called with the result when :meth:`CudaStreamWork.wait`
    is invoked, after the side stream has finished and output tensors are recorded.
    Its return value becomes the return value of :meth:`CudaStreamWork.wait`.
    """
    if device.type != "cuda":
        raise RuntimeError("run_on_async_stream requires CUDA device")
    stream = stream or _get_async_stream(device)
    done_event = torch.cuda.Event()
    stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(stream):
        value = fn()
        done_event.record(stream)
    if record_tensors is not None:
        for tensor in record_tensors:
            if tensor.is_cuda:
                tensor.record_stream(stream)
    return CudaStreamWork(value, done_event, device, on_wait)
