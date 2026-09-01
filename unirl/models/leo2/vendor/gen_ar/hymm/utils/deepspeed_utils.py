import itertools
from contextlib import contextmanager

from packaging import version
import deepspeed
from deepspeed import DeepSpeedEngine


def add_hooks(model: DeepSpeedEngine) -> None:
        """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
        if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
            return
        if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
            optimizer_offload = model.optimizer.parameter_offload
        elif model.optimizer is not None:
            optimizer_offload = model.optimizer
        else:
            raise RuntimeError("The model optimizer is None, which is not yet supported.")
        if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
            # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
            optimizer_offload._register_deepspeed_module(optimizer_offload.module)
        else:
            optimizer_offload._register_hooks_recursively(optimizer_offload.module)


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def remove_hooks(model: DeepSpeedEngine) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


# Adapted from: https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/models/utils.py#L186
@contextmanager
def unwrap_model_for_generation_deepspeed(
    model: DeepSpeedEngine,
    gather_deepspeed3_params: bool = True,
):
    """
    Modified context manager for DeepSpeed-only environments
    
    Args:
        model (DeepSpeedEngine): 
            Model to be unwrapped.
        gather_deepspeed3_params (bool):  
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.
    
    Yields:
        torch.nn.Module: Unwrapped model.
    
    Example:
    ```python
    with unwrap_model_for_generation_deepspeed(ds_model) as unwrapped_model:
        outputs = unwrapped_model.generate(inputs)
    ```
    """
    unwrapped_model = model.module if isinstance(model, DeepSpeedEngine) else model
    
    is_zero3 = hasattr(model, 'zero_optimization_stage') and (model.zero_optimization_stage() == 3)
    
    if is_zero3 and gather_deepspeed3_params:
        if not gather_deepspeed3_params:
            yield model.module if isinstance(model, DeepSpeedEngine) else model
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield model.module if isinstance(model, DeepSpeedEngine) else model
                add_hooks(model)
    else:
        yield unwrapped_model
