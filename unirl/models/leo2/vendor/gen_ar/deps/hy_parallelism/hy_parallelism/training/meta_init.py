import torch
from torch import nn
from contextlib import contextmanager
import loguru

old_to = nn.Module.to
old_load_state_dict = nn.Module.load_state_dict
old_cuda = nn.Module.cuda
old_cpu = nn.Module.cpu

def no_op_to(self, device=None, dtype=None, *args, **kwargs):
    if device is not None and device != torch.device('meta'):
        loguru.logger.warning(f'{self.__class__.__name__}.to(device={device}) is called in meta_init context. Skipping...')
    return old_to(self, dtype=dtype)

def no_op_load_state_dict(self, state_dict, strict=True, *args, **kwargs): 
    loguru.logger.opt(depth=1).error(f'{self.__class__.__name__}.load_state_dict(state_dict, strict={strict}) is called in meta_init context. Skipping...')
    raise RuntimeError(f'{self.__class__.__name__}.load_state_dict(state_dict, strict={strict}) is called in meta_init context. Skipping...')

def no_op_cuda(self, **kwargs):
    loguru.logger.opt(depth=1).warning(f'{self.__class__.__name__}.cuda() is called in meta_init context. Skipping...')
    return self

def no_op_cpu(self, **kwargs):
    loguru.logger.opt(depth=1).warning(f'{self.__class__.__name__}.cpu() is called in meta_init context. Skipping...')
    return self


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)



@contextmanager
def meta_init():
    nn.Module.to = no_op_to
    nn.Module.load_state_dict = no_op_load_state_dict
    nn.Module.cuda = no_op_cuda
    nn.Module.cpu = no_op_cpu
    with init_empty_weights():
        yield
    nn.Module.to = old_to
    nn.Module.load_state_dict = old_load_state_dict
    nn.Module.cuda = old_cuda
    nn.Module.cpu = old_cpu