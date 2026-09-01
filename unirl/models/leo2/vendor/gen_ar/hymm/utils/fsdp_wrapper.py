import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import (
    Callable,
    Iterable,
    Optional,
    Union,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig, FullOptimStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, CustomPolicy
from torch.distributed.fsdp._init_utils import (
    ProcessGroupType,
)
from ..trainers.helpers import get_trainable_params
from .monitor import MonitorMaster
import deepspeed


# Mimic deepspeed engine
class FSDPEngine(FullyShardedDataParallel):
    def __init__(
        self,
        args,
        ds_config,
        module: nn.Module,
        optimizer,
        lr_scheduler,
        logger = None,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Union[Callable, ModuleWrapPolicy, CustomPolicy]] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__(
            module,
            process_group,
            sharding_strategy,
            cpu_offload,
            auto_wrap_policy,
            backward_prefetch,
            mixed_precision,
            ignored_modules,
            param_init_fn,
            device_id,
            sync_module_states,
            forward_prefetch,
            limit_all_gathers,
            use_orig_params,
            ignored_states,
            device_mesh,
        )
        self.optimizer = optimizer(get_trainable_params(self.module, args.training_parts))
        self.lr_scheduler = lr_scheduler(self.optimizer)

        monitor_config = deepspeed.runtime.config.get_monitor_config(ds_config)
        self.monitor = MonitorMaster(monitor_config)
        self.args = args
        self.micro_steps = 0
        self.global_rank = dist.get_rank()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        if logger is None:
            from loguru import logger
        self.logger = logger
        self._global_grad_norm = 0.0
    
    def get_global_grad_norm(self) -> float:
        """Return the 2-norm of all gradients. If there is model parallelism,
        the norm will be global.
        The computed norm will be cached and reused until the next step() pass.
        .. note::
            In the presence of model parallelism, this is a collective call
            and acts as a barrier among ``mpu.get_model_parallel_group()``.
        Returns:
            float: norm
        """
        return self._global_grad_norm
    
    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        return (self.micro_steps + 1) % self.args.gradient_accumulation_steps == 0

    def backward(self, loss):
        # TODO: make it support fp16 with scaling
        loss.backward()

    def step(self, lr_kwargs=None):
        if self.is_gradient_accumulation_boundary():
            if self.args.gradient_clipping > 0.0:
                self._global_grad_norm = self.clip_grad_norm_(max_norm=self.args.gradient_clipping).cpu().item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step(**(lr_kwargs or {}))
            self.micro_steps = 0
        self.micro_steps += 1

    def full_state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # cast to cpu to avoid oom
        with FSDP.state_dict_type(
                self,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        from torch.distributed.tensor.parallel.fsdp import DTensor
        cpu_state = {k: (v.full_tensor().cpu() if isinstance(v, DTensor) else v).contiguous() for k, v in cpu_state.items()}
        return cpu_state
