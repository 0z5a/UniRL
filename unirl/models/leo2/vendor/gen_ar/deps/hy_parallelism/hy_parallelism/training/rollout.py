from contextlib import contextmanager
from typing import cast

import torch

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