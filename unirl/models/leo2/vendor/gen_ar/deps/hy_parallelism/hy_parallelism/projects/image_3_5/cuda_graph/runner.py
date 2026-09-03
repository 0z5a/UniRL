from __future__ import annotations

import gc
import loguru
import contextlib
from types import MethodType
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from hymm.models.autoregressive.custom_cache import HunyuanStaticCache
from hy_parallelism.common.logging import trace_log

from hy_parallelism.projects.image_3_5.cuda_graph.forward_patch import (
    bind_graph_safe_kv_updates,
    enabled,
    gpu_input_pos,
    graph_safe_decode_forward,
    sync_decode_attn_mask,
    use_fast_forward_context,
)

_WARMUP_STEPS = 3


def sync_committed_length(
    past_key_values: Optional[HunyuanStaticCache],
    input_pos: Optional[torch.Tensor],
) -> None:
    if past_key_values is None or not getattr(past_key_values, "dynamic", False):
        return
    if input_pos is None:
        return
    pos_val = int(input_pos.reshape(-1)[-1].item()) + 1
    if getattr(past_key_values, "_graph_sync_pos", None) == pos_val:
        return
    past_key_values._graph_sync_pos = pos_val
    if pos_val > past_key_values._committed_length:
        past_key_values._committed_length = pos_val


def is_decode_step(
    *,
    input_ids: Optional[torch.Tensor],
    input_pos: Optional[torch.Tensor],
    past_key_values: Optional[HunyuanStaticCache],
) -> bool:
    if input_pos is None or past_key_values is None:
        return False
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] != 1:
        return False
    return True


class DecodeCUDAGraphRunner:
    __slots__ = (
        "_model",
        "_eager_forward",
        "_graph",
        "_static_input_ids",
        "_static_input_pos",
        "_static_output",
        "_fwd_args",
        "_fwd_kwargs",
        "_captured",
    )

    def __init__(self, model, eager_forward: Callable):
        self._model = model
        self._eager_forward = eager_forward
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input_ids: Optional[torch.Tensor] = None
        self._static_input_pos: Optional[torch.Tensor] = None
        self._static_output: Any = None
        self._fwd_args: tuple = ()
        self._fwd_kwargs: dict = {}
        self._captured = False

    def reset(self) -> None:
        self._graph = None
        self._static_input_ids = None
        self._static_input_pos = None
        self._static_output = None
        self._fwd_args = ()
        self._fwd_kwargs = {}
        self._captured = False
        if hasattr(self._model, "_static_decode_attn_mask"):
            del self._model._static_decode_attn_mask
        if hasattr(self._model, "_static_decode_flex_pos"):
            del self._model._static_decode_flex_pos
        gc.collect()

    def _sync_decode_attn_mask(self, kwargs: dict) -> None:
        past_key_values = kwargs.get("past_key_values")
        input_pos = self._static_input_pos
        assert past_key_values is not None and input_pos is not None
        sync_decode_attn_mask(self._model, past_key_values, input_pos)

    def _graph_forward(self, *args, **kwargs):
        return graph_safe_decode_forward(self._model, *args, **kwargs)

    def _build_static_inputs(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        kw = dict(kwargs)
        input_ids = kw.get("input_ids") if not args else args[0]
        input_pos = kw.get("input_pos")
        assert input_ids is not None and input_pos is not None

        self._static_input_ids = torch.empty_like(input_ids)
        self._static_input_ids.copy_(input_ids)
        ip = gpu_input_pos(input_pos)
        self._static_input_pos = torch.empty_like(ip)
        self._static_input_pos.copy_(ip)
        self._sync_decode_attn_mask(kw)

        if args:
            fwd_args = (self._static_input_ids, *args[1:])
        else:
            fwd_args = ()
            kw["input_ids"] = self._static_input_ids
        kw["input_pos"] = self._static_input_pos
        return fwd_args, kw

    def _capture(self, args: tuple, kwargs: dict) -> None:
        fwd_args, fwd_kw = self._build_static_inputs(args, kwargs)
        self._fwd_args = fwd_args
        self._fwd_kwargs = fwd_kw

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(_WARMUP_STEPS):
                self._graph_forward(*fwd_args, **fwd_kw)
        torch.cuda.current_stream().wait_stream(warmup)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_output = self._graph_forward(*fwd_args, **fwd_kw)
        self._graph = graph
        self._captured = True

    def _copy_step_inputs(self, args: tuple, kwargs: dict) -> None:
        input_ids = kwargs.get("input_ids") if not args else args[0]
        input_pos = kwargs.get("input_pos")
        assert self._static_input_ids is not None
        assert self._static_input_pos is not None
        self._static_input_ids.copy_(input_ids)
        self._static_input_pos.copy_(gpu_input_pos(input_pos))
        self._sync_decode_attn_mask(kwargs)

    def __call__(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids") if not args else args[0]
        if not is_decode_step(
            input_ids=input_ids,
            input_pos=kwargs.get("input_pos"),
            past_key_values=kwargs.get("past_key_values"),
        ):
            loguru.logger.warning("Falling back to eager forward.")
            return self._eager_forward(*args, **kwargs)

        sync_committed_length(kwargs.get("past_key_values"), kwargs.get("input_pos"))

        if not self._captured:
            self._capture(args, kwargs)
            if dist.is_initialized() and dist.get_rank() == 0:
                trace_log("[HY35] decode CUDA graph captured")

        self._copy_step_inputs(args, kwargs)
        self._graph.replay()
        return self._static_output


def reset_decode_cuda_graphs(model) -> None:
    runner = getattr(model, "_decode_cuda_graph_runner", None)
    if runner is not None:
        runner.reset()


def _get_or_create_runner(model, eager_forward: Callable) -> DecodeCUDAGraphRunner:
    runner = getattr(model, "_decode_cuda_graph_runner", None)
    if runner is None:
        runner = DecodeCUDAGraphRunner(model, eager_forward)
        model._decode_cuda_graph_runner = runner
    else:
        runner._eager_forward = eager_forward
    return runner


@contextlib.contextmanager
def use_decode_cuda_graph(model):
    if not enabled():
        yield
        return

    original_call_impl = model._call_impl
    runner = _get_or_create_runner(model, model.forward)

    def _graph_call_impl(self, *args, **kwargs):
        return runner(*args, **kwargs)

    # Bind / fast attn patch are only needed for warmup/capture;
    # replay uses kernels already recorded in the graph.
    need_capture_setup = not runner._captured
    if need_capture_setup:
        bind_graph_safe_kv_updates(model)
    model._call_impl = MethodType(_graph_call_impl, model)
    try:
        if need_capture_setup:
            with use_fast_forward_context(model):
                yield
        else:
            yield
    finally:
        model._call_impl = original_call_impl
