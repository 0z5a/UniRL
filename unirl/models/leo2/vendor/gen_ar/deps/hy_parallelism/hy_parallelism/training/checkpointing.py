import contextlib
import torch
from torch.utils.checkpoint import *
from torch.utils.checkpoint import _DEFAULT_DETERMINISM_MODE, _checkpoint_debug_enabled, _get_debug_context_and_cb, _allowed_determinism_checks_to_fns, _infer_device_type, _get_device_module, _is_compiling, _get_autocast_kwargs, _enable_checkpoint_early_stop, _CheckpointFrame, _NoopSaveInputs, _checkpoint_hook, TorchDispatchMode
import warnings
from typing import Any, Tuple, NoReturn, Optional, Callable, ContextManager


# Global config for offload_checkpoint_fn. None = offload all inputs.
_offload_list = None

def set_offload_list(offload_list):
    global _offload_list
    _offload_list = [k + 2 for k in offload_list] # dummy, kwargs

def get_offload_list():
    return _offload_list

def _tensor_to_cpu(tensor, pin_memory=False):
    assert isinstance(tensor, torch.Tensor), f"Expected a tensor, got {type(tensor)}"
    with torch.enable_grad():
        ret = tensor.to('cpu', non_blocking=True)
        assert ret.requires_grad == tensor.requires_grad, f'{ret.requires_grad=} {tensor.requires_grad=}'
    return ret
    packed = torch.empty(
        tensor.size(),
        dtype=tensor.dtype,
        layout=tensor.layout,
        pin_memory=(not tensor.is_sparse),
    )
    packed.copy_(tensor)
    return packed


def _to_cpu(x, pin_memory=False):
    if isinstance(x, torch.Tensor):
        return _tensor_to_cpu(x, pin_memory)
    if isinstance(x, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in x):
        return type(x)(_tensor_to_cpu(t, pin_memory) for t in x)
    raise AssertionError(f"Expected a tensor, got {type(x)}")


def _to_cuda(x):
    if isinstance(x, torch.Tensor):
        with torch.enable_grad():
            return x.cuda(non_blocking=True)
    if isinstance(x, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in x):
        return type(x)(_to_cuda(t) for t in x)
    raise AssertionError(f"Expected a tensor, got {type(x)}")
    return x.cuda()


def get_selective_offload_fn(offload_list=None, pin_memory=False):
    offload_all = offload_list is None
    offload_indices = set() if offload_all else set(offload_list)

    def should_offload(i):
        return offload_all or i in offload_indices

    class OffloadSaveInputs(torch.autograd.Function):
        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(*args):
            return torch.empty((0,))

        @staticmethod
        def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
            tensor_pairs = [
                (i, _to_cpu(o, pin_memory=pin_memory) if should_offload(i) else o)
                for i, o in enumerate(inputs)
                if isinstance(o, torch.Tensor)
            ]
            if tensor_pairs:
                tensor_indices, tensors = zip(*tensor_pairs, strict=False)
            else:
                tensor_indices, tensors = (), ()
            idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}

            args = [
                None if isinstance(o, torch.Tensor) else (_to_cpu(o, pin_memory=pin_memory) if should_offload(i) else o)
                for i, o in enumerate(inputs)
            ]

            def get_args(saved_tensors):
                # restore the placeholders with the original tensors grabbed from
                # ctx.saved_tensors (which may be saved on a parent checkpoint if
                # this checkpoint is nested, and that would trigger a recursive
                # unpack!)
                ret = []
                for i, o in enumerate(args):
                    if i in tensor_indices:
                        o = saved_tensors[idx2saved_idx[i]]
                    if should_offload(i):
                        o = _to_cuda(o)
                    ret.append(o)
                return ret[1:]  # skip dummy

            ctx.get_args = get_args
            ctx.save_for_backward(*tensors)

        @staticmethod
        def backward(ctx, *grad_outputs) -> NoReturn:
            raise AssertionError("Did not expect to backward on this graph")
    
    return OffloadSaveInputs

def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    *args,
    pin_memory: bool = False,
    **kwargs,
):
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn != noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args)
    device_module = _get_device_module(device_type)
    forward_context, recompute_context = context_fn()
    if _is_compiling(fn, args, kwargs) and context_fn != noop_context_fn:
        assert (
            isinstance(forward_context, TorchDispatchMode) and
            isinstance(recompute_context, TorchDispatchMode)
        ), \
            "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` " + \
            "must generate a tuple of two `TorchDispatchMode`s."
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, device_type=device_type
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=device_type, **device_autocast_kwargs
            ) if torch.amp.is_autocast_available(device_type) else contextlib.nullcontext()
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop if _enable_checkpoint_early_stop is not None else early_stop,
        unpack_error_cb,
        metadata_fn
    )
    dummy = torch.empty((0,), requires_grad=True)
    new_frame.input_saver = get_selective_offload_fn(get_offload_list(), pin_memory=pin_memory).apply(dummy, kwargs, *args)
    # new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return

@torch._disable_dynamo
def offload_checkpoint_fn(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    generator_kwargs: Optional[dict] = None,
    **kwargs
):
    if generator_kwargs is None:
        generator_kwargs = {}
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. Starting in PyTorch 2.9, calling checkpoint "
            "without use_reentrant will raise an exception. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    assert not use_reentrant
    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, early_stop, *args, **kwargs,
            **generator_kwargs,
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret