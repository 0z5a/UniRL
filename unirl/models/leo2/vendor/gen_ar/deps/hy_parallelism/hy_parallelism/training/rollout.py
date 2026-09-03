from contextlib import contextmanager
from typing import cast

import torch
import contextlib
import types
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from hy_parallelism.engines.parallel_engine import BaseParallelEngine
from packaging import version

from hy_parallelism.parallel_states import get_parallel_state

from torch import nn
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed.fsdp._fully_shard._fully_shard import _get_post_forward_mesh_info

def set_reshard_after_forward(
    self, reshard_after_forward: bool, recurse: bool = True
) -> None:
    """
    Sets if the module should reshard parameters after forward. This can be
    used to change the ``reshard_after_forward`` FSDP arg at runtime. For
    example, this can be used to set the FSDP root module's value to
    ``True`` (since it is otherwise specially set to ``False``), or it can
    set an FSDP module's value to ``False`` for running evals and set back
    to ``True`` for training.

    Args:
        reshard_after_forward (bool): Whether to reshard parameters after
            forward.
        recurse (bool): Whether to set for all FSDP submodules or just the
            passed-in module.
    """
    self_module = cast(nn.Module, self)
    modules = list(self_module.modules()) if recurse else [self_module]
    for module in modules:
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.post_forward_mesh_info = (
                    _get_post_forward_mesh_info(
                        reshard_after_forward, fsdp_param_group.mesh_info
                    )
                )



@contextmanager
def rollout_context(parallel_engine: BaseParallelEngine):
    r"""Context manager for evaluation-only rollouts without resharding after forward.

    This context disables parameter resharding after forward in all registered FSDP models of the
    provided parallel engine, and runs them in eval mode with unsharded parameters under
    `torch.no_grad()`. After the context exits, it restores the models to their original training mode and
    re-enables resharding.

    This is especially useful for performing efficient rollouts or evaluations with large models wrapped
    by FSDP, where avoiding extraneous communication and unnecessary parameter sharding improves
    performance.

    Args:
        parallel_engine (BaseParallelEngine): The parallel engine containing models
            (typically FSDP-wrapped) whose settings will be modified for the rollout.
    """
    original_training_state = parallel_engine.training

    parallel_engine.eval()
    parallel_engine.unshard()
    for model in parallel_engine.fsdp_models:

        # if pp is enabled, reshard after forward may be already disabled, we should keep it unchanged.
        if not get_parallel_state().pp_enabled:
            model.set_reshard_after_backward(False)
            if hasattr(model, 'set_reshard_after_forward'):
                model.set_reshard_after_forward(False)
            else:
                set_reshard_after_forward(model, False)

    with torch.no_grad():
        yield

    if original_training_state:
        parallel_engine.train()
    for model in parallel_engine.fsdp_models:

        # if pp is enabled, reshard after forward may be already disabled, we should keep it unchanged.
        if not get_parallel_state().pp_enabled:
            model.set_reshard_after_backward(True)
            if hasattr(model, 'set_reshard_after_forward'):
                model.set_reshard_after_forward(True)
            else:
                set_reshard_after_forward(model, True)

        parallel_engine.reshard()



def _fsdp_is_unsharded(m: nn.Module) -> bool:
    if not isinstance(m, FSDPModule):
        return False
    pg = m._get_fsdp_state()._fsdp_param_group
    return pg is not None and pg.is_unsharded


def _forward_only(self, *args, **kwargs):
    from hy_parallelism.tools.profiling import profile_range
    with profile_range('forward_only'):
        return self.forward(*args, **kwargs)

# from typing import Any
# from torch.distributed.fsdp._fully_shard._fsdp_state import TrainingState, _cast_fp_tensor, FSDPState
# from torch.utils._pytree import tree_flatten, tree_map

# original_pre_forward = FSDPState._pre_forward
# original_post_forward = FSDPState._post_forward

# def _fast_pre_forward(
#     self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
# ) -> tuple[tuple[Any, ...], dict[str, Any]]:
#     print('enter fast')
#     import functools
#     # When composing with module-hook-based activation checkpointing, the
#     # the pre-backward hook is responsible for the unshard
#     if self._training_state == TrainingState.PRE_BACKWARD:
#         return args, kwargs
#     self._training_state = TrainingState.FORWARD
#     if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
#         with torch.profiler.record_function("FSDP::cast_forward_inputs"):
#             cast_fn = functools.partial(
#                 _cast_fp_tensor, self._mp_policy.param_dtype
#             )
#             args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
#     return args, kwargs

# def _fast_post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
#     import functools
#     # When composing with module-hook-based activation checkpointing, the
#     # post-backward hook is responsible for the reshard
#     if self._training_state == TrainingState.PRE_BACKWARD:
#         return output
#     self._training_state = TrainingState.IDLE
#     if self._mp_policy.output_dtype is not None:
#         with torch.profiler.record_function("FSDP::cast_forward_outputs"):
#             output = tree_map(
#                 functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
#                 output,
#             )
#     return output


# @contextlib.contextmanager
# def fsdp_skip_hooks_if_unsharded(module: nn.Module):
#     patched = []
#     for m in module.modules():
#         if not isinstance(m, FSDPModule) and m is not module:
#             continue
#         fsdp_state = m._get_fsdp_state()
#         pg = fsdp_state._fsdp_param_group
#         if pg is not None and pg.is_unsharded:
#             patched.append(fsdp_state)
#             print('update pre_forward and post_forward')
#             fsdp_state._pre_forward = _fast_pre_forward.__get__(fsdp_state, FSDPState)
#             fsdp_state._post_forward = _fast_post_forward.__get__(fsdp_state, FSDPState)
#     try:
#         yield
#     finally:
#         for m in patched:
#             fsdp_state._pre_forward = original_pre_forward.__get__(fsdp_state, FSDPState)
#             fsdp_state._post_forward = original_post_forward.__get__(fsdp_state, FSDPState)

@contextmanager
def call_forward_directly(module: nn.Module):
    from types import MethodType
    saved = []
    try:
        for m in module.modules():
            if not isinstance(m, FSDPModule):
                continue
            saved.append((m, m._call_impl))
            def _direct_call_impl(self, *args, **kwargs):
                return self.forward(*args, **kwargs)
            m._call_impl = MethodType(_direct_call_impl, m)
        yield
    finally:
        for m, old_call_impl in reversed(saved):
            m._call_impl = old_call_impl