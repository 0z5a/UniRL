import contextlib
import copy
import glob
import importlib
import os
import random
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import default_collate
import torch.distributed as dist
from functools import partial
from torch.optim import AdamW, Adam
from torch.distributed.fsdp import ShardingStrategy
from safetensors.torch import load_file

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
}

NAME_TO_SHARDING_STRATEGY = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
}


def set_manual_seed(global_seed):
    # Seed the RNG for Python
    random.seed(global_seed)
    # Seed the RNG for Numpy
    np.random.seed(global_seed)
    # Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(global_seed)


def set_reproducibility(enable, global_seed=None, benchmark=None):
    if enable:
        # Configure the seed for reproducibility
        set_manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)

    # LSTM and RNN networks are not deterministic


class set_worker_seed_builder:
    def __init__(self, global_rank):
        self.global_rank = global_rank

    def __call__(self, worker_id):
        # Make sure to set torch.manual_seed(seed + dp_rank) before call this function
        # The torch.initial_seed() will be seed + dp_rank.
        set_manual_seed(torch.initial_seed() % (2**32 - 1))


def profiler_context(enable, exp_dir, worker_name):
    if enable:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=1,
                active=3,
                repeat=2,
            ),
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(exp_dir, worker_name=worker_name),
            with_stack=True,
        )
    else:
        # return empty python context manager
        return contextlib.nullcontext()


def all_gather_sum(running_value, device):
    value = torch.tensor(running_value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item()


def get_states_from_safe_tensors(safe_tensor_file_list):
    state_dict = {}
    for safe_tensor_file in safe_tensor_file_list:
        sharded_state = load_file(safe_tensor_file, "cpu")
        state_dict.update(sharded_state)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_k = k.replace("model.", "transformer.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def expand_head_embedding(target_dict, state_dict, expand_keys):
    new_state = copy.deepcopy(state_dict)
    for k in expand_keys:
        current_val = target_dict[k]
        original_val = state_dict[k]
        current_val[:original_val.shape[0]] = original_val
        new_state[k] = current_val
    return new_state


def shrink_head_embedding(target_dict, state_dict, shrink_keys):
    """
    Extract and overwrite the leading portions of large tensors under specified keys in the state_dict onto smaller tensors in the target_dict.
    Used for loading a model with a smaller head embedding size from a checkpoint with a larger head embedding size.
    """
    new_state = copy.deepcopy(state_dict)
    for k in shrink_keys:
        original_val = state_dict[k]
        current_val = target_dict[k]
        current_val = original_val[:current_val.shape[0]]
        new_state[k] = current_val
        
    return new_state


def load_state_dict(model, ckpt_file, load_llm_to_mllm=False, expand_keys=None, load_prepend_key=None, load_prepend_key_dict=None, shrink_keys=None):
    if os.path.isdir(ckpt_file):
        safetensor_files = sorted(glob.glob(os.path.join(ckpt_file, "*.safetensors")))
        if len(safetensor_files):
            state_dict = get_states_from_safe_tensors(safetensor_files)
        else:
            from hymm.parallelism.checkpoint_manager import CheckpointManager, Checkpoint
            CheckpointManager(
                {},
                ckpt_config=Checkpoint(),
                model_parts=[model],
            ).load_from_path(ckpt_file)
            return [], []
    else:
        state_dict = torch.load(ckpt_file, map_location="cpu", mmap=True)
        if "module" in state_dict:
            state_dict = state_dict["module"]
    if load_llm_to_mllm:
        if expand_keys is None:
            expand_keys = ["transformer.wte.weight", "lm_head.weight", "lm_head.bias"]
        state_dict = expand_head_embedding(model.state_dict(), state_dict, expand_keys)
    if load_prepend_key:
        state_dict = {load_prepend_key + k: v for k, v in state_dict.items()}
    if load_prepend_key_dict:
        new_state_dict = {}
        old_name_key_list = load_prepend_key_dict[0]
        new_prepend_key_list = load_prepend_key_dict[1]
        for k, v in state_dict.items():
            new_key = k
            for old_name_key, new_prepend_key in zip(old_name_key_list, new_prepend_key_list):
                if old_name_key in k:
                    new_key = new_key.replace(old_name_key, new_prepend_key)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    if shrink_keys is not None:
        state_dict = shrink_head_embedding(model.state_dict(), state_dict, shrink_keys)
    m, u = model.load_state_dict(state_dict, strict=False, assign=True)
    return m, u


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def build_optimizer(args):
    if args.optimizer_name == "AdamW":
        return partial(AdamW, **args.optimizer_params)
    elif args.optimizer_name == "Adam":
        return partial(Adam, **args.optimizer_params)
    

def move_model_params_and_grads_to(model: torch.nn.Module, device="cpu"):
    model.to(device, non_blocking=True)
    
    grad_states = {}
    for name, param in model.named_parameters():
        grad_states[name] = param.grad is not None
       
    for name, param in model.named_parameters():
        if grad_states[name] and param.grad is not None:
            param.grad.data = param.grad.data.to(device)
    
    return model


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return isinstance(data, tuple) and hasattr(data, "_asdict") and hasattr(data, "_fields")


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)
    

# Copied from: https://github.com/huggingface/accelerate/blob/74e08e5205501826c0d4bae5abb7c25cb3da5e15/src/accelerate/utils/operations.py#L84
def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs (additional keyword arguments, *optional*):
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data


def to_device(data, device):
    if data is None:
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        raise ValueError(f"Unknown data type {type(data)}")


class DummyTensor(dict):
    """
    We define a DummyTensor class for representing a tensor without really building it.
    It can be used for compatibility of attention mask within the AngelPTM.
    """
    def __init__(self, **kwargs):
        assert 'shape' in kwargs
        assert 'dtype' in kwargs
        assert 'device' in kwargs
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

def except_collate_fn(batch, except_keys=None):
    if except_keys is None:
        return default_collate(batch)

    ret = {}
    existing_except_keys = []
    for expect_key in except_keys:
        if expect_key in batch[0]:
            existing_except_keys.append(expect_key)
    
    except_keys = existing_except_keys
    
    for expect_key in except_keys:
        ret[expect_key] = []

    for batch_index in range(len(batch)):
        for except_key in except_keys:
            ret[except_key].append(batch[batch_index][except_key])
            del batch[batch_index][except_key]
    
    ret.update(default_collate(batch))

    return ret


def recursive_info(value):
    if isinstance(value, (list, tuple)):
        return [recursive_info(v) for v in value]
    elif isinstance(value, dict):
        return {k: recursive_info(v) for k, v in value.items()}
    elif isinstance(value, torch.Tensor):
        return f"{value.dtype}, {value.shape}, {value.device}"
    else:
        return value


def print_arguments(func):
    def wrapper(*args, **kwargs):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank == 0:
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            arg_values = args + tuple(kwargs.get(name, None) for name in arg_names[len(args):])
            print(f"Arguments for {func.__name__}:")
            for name, value in zip(arg_names, arg_values):
                if name == "self":
                    continue
                print(f"[rank{rank}] {name}: {recursive_info(value)}")
        return func(*args, **kwargs)

    return wrapper


@torch.no_grad()
def print_stat(name, tensor, rank=None):
    import torch.distributed.tensor
    prefix = os.getenv("DEBUG_PREFIX", "")
    if torch.distributed.is_initialized():
        if rank is not None and torch.distributed.get_rank() != rank:
            return
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            tensor = tensor.to_local().float()
        else:
            tensor = tensor.detach().float()
    else:
        tensor = tensor.detach().float()
        rank = 0
    print(f"[rank {rank}] {prefix}{name}, "
          f"shape={tensor.shape}, "
          f"sum={tensor.sum():.6f}, "
          f"mean={tensor.mean():.6f}, "
          f"std={tensor.std():.6f}, "
          f"min={tensor.min():.6f}, "
          f"max={tensor.max():.6f}", flush=True)


class Timer(object):
    def __init__(self, enabled=False):
        self.timers = {}
        self.enabled = enabled

    def __getitem__(self, key):
        return self.timers[key]

    def start(self, name, sync=False, barrier=False):
        if not self.enabled:
            return
        if name not in self.timers:
            self.timers[name] = {"total_time": 0.0, "count": 0, "sync": sync, "barrier": barrier}
        else:
            if "start_time" in self.timers[name]:
                raise ValueError(f"Timer '{name}' is already running.")
            self.timers[name].update({"sync": sync, "barrier": barrier})
        if sync:
            torch.cuda.synchronize()
        if barrier:
            dist.barrier()
        self.timers[name]["start_time"] = time.time()

    def stop(self, name):
        if not self.enabled:
            return
        if name not in self.timers or "start_time" not in self.timers[name]:
            raise ValueError(f"Timer '{name}' was not started.")
        if self.timers[name]["sync"]:
            torch.cuda.synchronize()
        if self.timers[name]["barrier"]:
            dist.barrier()
        elapsed_time = time.time() - self.timers[name]["start_time"]
        self.timers[name]["elapsed_time"] = elapsed_time
        self.timers[name]["total_time"] += elapsed_time
        self.timers[name]["count"] += 1
        del self.timers[name]["start_time"]

    def elapsed(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers or "elapsed_time" not in self.timers[name]:
            raise ValueError(f"Timer '{name}' has no recorded elapsed time.")
        return self.timers[name]["elapsed_time"]

    def elapsed_total(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' has no recorded times.")
        return self.timers[name]["total_time"]

    def average(self, name):
        if not self.enabled:
            return 0.0
        if name not in self.timers or self.timers[name]["count"] == 0:
            raise ValueError(f"Timer '{name}' has no recorded times.")
        return self.timers[name]["total_time"] / self.timers[name]["count"]


def nanstd(tensor: torch.Tensor) -> float:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return float("nan")
    return finite.std(unbiased=False).item()
