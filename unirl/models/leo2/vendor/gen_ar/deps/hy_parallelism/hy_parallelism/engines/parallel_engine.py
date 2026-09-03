# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from contextlib import contextmanager
import itertools
import copy
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import Type

import loguru
from loguru import logger
import torch
from packaging import version
from torch import isfinite, nn
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    # ScheduleZBVZeroBubble,
)
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from hy_parallelism.checkpoint import checkpoint_manager
from hy_parallelism.checkpoint.checkpoint_manager import (
    Checkpoint,
    CheckpointManager,
    LRSchedulersContainer,
    OptimizersContainer,
    load_pt_or_safetensors,
)
from hy_parallelism.common.logging import debug_log
from hy_parallelism.globabl_states import GlobalStates, get_global_states
from hy_parallelism.parallel_states import ParallelDims, get_parallel_state
from hy_parallelism.pipeline import hy_get_schedule_class as get_schedule_class
from hy_parallelism.pipeline.stage import NoShapeInferenceStage
from hy_parallelism.utils import (
    auto_broadcast,
    clip_grad_norm_by_mesh_,
    format_keys,
    gather_obj,
    isolate_rng,
    log_once,
    print_model_info,
    sync_object_for_parallel_training,
)
from hy_parallelism.tools.profiling import profile_range
from hy_parallelism.training.cast_device import cast_to_device
from hy_parallelism.training.checkpointing import ACTIVATION_POOL_NAME
from hy_parallelism.training.pinned_memory_pool import (
    PinnedMemoryPool,
    get_pinned_memory_pool,
)

DEBUG_MODE = False


class SafeObjectWrapper:
    def __init__(self, obj):
        self.obj = obj


def get_arg_names(func):
    r"""Get the argument names of a function or method.

    Important:
        If the function is a bound method, 'self' will be skipped automatically.
        If the function is a class method, 'self' will NOT BE SKIPPED!!

    Args:
        func (callable): The function or method to inspect.

    Returns:
        list[str]: A list of argument names in the order they appear in the function signature.

    Old implementation:
        import inspect
        arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
    """
    import inspect

    arg_names = inspect.signature(func).parameters.keys()
    return list(arg_names)


def get_optimizer_config(optimizer_cls=torch.optim.AdamW, optimizer_kwargs=None):
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
    return dict(
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
    )


class ModelToLossFn:
    def __init__(self, model, get_m_microbatch_fn=None):
        self.model = model
        # m_microbatch could be changed
        self.get_m_microbatch_fn = get_m_microbatch_fn

        if get_m_microbatch_fn is None:
            assert version.parse(torch.__version__) >= version.parse("2.7.0")

    def __call__(self, output, target):  # output is assume to be the total loss
        # HACK(kevinkhwu): This issue is fixed in PyTorch 2.7.0
        #   Our PyTorch is not updated to the latest version yet.
        #   https://github.com/pytorch/pytorch/pull/144352
        if version.parse(torch.__version__) >= version.parse("2.7.0"):
            log_once("kevinkhwu: The gradient accumulation issue is fixed, consider refactoring the code.", level="WARNING")
            return output
        else:
            # gradients will be accumulated
            return output / self.get_m_microbatch_fn()


def make_zero(tensors):
    if isinstance(tensors, (list, tuple)):
        return sum([make_zero(t) for t in tensors])
    elif isinstance(tensors, torch.Tensor):
        return (tensors - tensors).mean()
    else:
        raise ValueError(f"Unsupported type {type(tensors)}")


def not_implemented(func):
    def not_implemented_func(*args, **kwargs):
        raise NotImplementedError

    setattr(not_implemented_func, "is_implemented", False)
    return not_implemented_func


def is_implemented(func):
    if func is None:
        return False
    if hasattr(func, "is_implemented"):
        return func.is_implemented
    return True


class DeepSpeedInterface(ABC):
    # https://github.com/deepspeedai/DeepSpeed/blob/c2bb53f20fa32d6cbf472c08a42959a287dd9049/deepspeed/runtime/engine.py#L195

    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
        self,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        custom_load_fn=None,
    ):
        """
        Load training checkpoint

        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Checkpoint tag used as a unique identifier for checkpoint, if not provided will attempt to load tag in 'latest' file
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
            load_module_only: Optional. Boolean to load only the model weights from the checkpoint. Ex. warmstarting.
            custom_load_fn: Optional. Custom model load function.

        Returns:
            A tuple of ``load_path`` and ``client_state``.
            *``load_path``: Path of the loaded checkpoint. ``None`` if loading the checkpoint failed.
            *``client_state``: State dictionary used for loading required training states in the client code.

        Important: under ZeRO3, one cannot load checkpoint with ``engine.load_checkpoint()`` right
        after ``engine.save_checkpoint()``. It is because ``engine.module`` is partitioned, and
        ``load_checkpoint()`` wants a pristine model. If insisting to do so, please reinitialize engine
        before ``load_checkpoint()``.

        """
        ...

    @abstractmethod
    def save_checkpoint(
        self,
        save_dir,
        tag=None,
        client_state: dict | None = None,
        save_latest=True,
        exclude_frozen_parameters=False,
        save_all_ranks_training_states=False,
        dcp_save_kwargs: dict | None = None,
    ):
        ...
        """Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                used if not provided. Tag name must be the same across all ranks.
            client_state: Optional. State dictionary used for saving required training states in the client code.
            save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.
            exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.
            save_all_ranks_training_states: Optional. If True, all ranks save training states into
                a subfolder (e.g. training_states/rank0.pt). Default False (only rank 0 saves).
        Important: all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.

        """

    @abstractmethod
    def zero_grad(self): ...
    @abstractmethod
    def step(self, lr_kwargs=None): ...

    @abstractmethod
    def forward(self, *inputs, **kwargs): ...

    @abstractmethod
    def backward(self, loss, retain_graph=None, create_graph=False, scale_wrt_gas=True): ...

    @abstractmethod
    def train(self, mode=True): ...

    @abstractmethod
    def eval(self): ...

    @property
    def module(self):
        return None

    @abstractmethod
    def get_global_grad_norm(self) -> float: ...

    @property
    def monitor(self):  # TODO:
        return None

    @abstractmethod
    def state_dict(self): ...

    @property
    def optimizer(self): ...


class PPModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from torch import distributed as dist

        self.pp_enabled = get_parallel_state().pp_enabled
        if self.pp_enabled:
            self.pp_preprocess = dist.get_rank() == get_parallel_state().pp_mesh.mesh[0]
            self.pp_postprocess = dist.get_rank() == get_parallel_state().pp_mesh.mesh[-1]
        else:
            self.pp_preprocess = True
            self.pp_postprocess = True


class EngineInterface:
    """
    Interface for the parallel engine.
    Should be implemented by the user.
    """

    guard_attr_names = [
        '_muon_split_fn',
        '_muon_merge_fn',
        'is_expert',
        '_param_name',
    ]

    def __init__(self):
        self.setup_global_states(get_global_states())

    @not_implemented
    def apply_fsdp(self, model):
        """
        hy_parallelism Parameter initialization:
        1. recursivlly call `reset_parameters` on all modules.
        2. run `param_init_fn` if provided.

        An example:

        default_fsdp_kwargs = self.default_fsdp_kwargs.copy()

        # PTM MOE implementation requires param_dtype to be float32
        default_fsdp_kwargs['param_dtype'] = torch.float32
        default_fsdp_kwargs['reduce_dtype'] = torch.float32

        # default_fsdp_kwargs['reshard_after_forward_policy'] = 'always'

        apply_fsdp2(
            model,
            blocks=list(model.double_blocks) + list(model.single_blocks),
            **default_fsdp_kwargs,
        )
        return model
        """

    @not_implemented
    def fsdp_blocks(self):
        # Deprecated
        """
        from torch.distributed.fsdp._fully_shard import fully_shard
        from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
        ep_enabled = self.parallel_dims.ep_enabled
        for m in self.fsdp_models:
            for block in list(m.double_blocks) + list(m.single_blocks):
                if block is None:
                    continue
                for fqn, module in block.named_modules():
                    if self.is_moe_router(fqn, module):
                        if isinstance(module, FSDPModule):
                            yield module
                for fqn, module in block.named_modules():
                    if self.is_expert(fqn) and isinstance(module, FSDPModule):
                        yield module
                yield block
            yield m
        """
        return []

    @not_implemented
    def apply_ac(self, model):
        """
        from hy_parallelism.distributed.fsdp_util import apply_fsdp_checkpointing
        no_split_module_type = None

        # When applying pp, double_blocks[0] could be None
        # apply_fsdp_checkpointing(model, no_split_modules=type(model.double_blocks[0]), p=1)
        # Find block types from both double and single blocks
        for block in model.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break

        if no_split_module_type is not None:
            apply_fsdp_checkpointing(
                model,
                no_split_modules=no_split_module_type,
                p=1,
                use_reentrant=False,
                # activation_offloading=self.activation_offloading,
                # activation_offload_list=[0],
            )
        """

    @not_implemented
    def config_forward_args(self):
        """
        self.set_n_pp_args(0)
        self.set_replacable_kwargs([])
        """

    def get_resolution_key(self, *args, **kwargs):
        raise NotImplementedError

    def get_batch_size(self, *args, **kwargs):
        raise NotImplementedError

    @not_implemented
    def mock_pp_forward(self, **kwargs):
        """
        # Avoid shape_inference
        # return x
        """

    @not_implemented
    def pp_friendly_forward(self, *args, **kwargs):
        """
        return self._original_forward(*args, **kwargs)
        """

    @not_implemented
    def pre_process_input(self, *args, **kwargs):
        """
        return args, kwargs
        """

    @not_implemented
    def param_init_fn(self, model, default_generator=None):
        """
        initialize all parameters for model
        required when meta init is enabled
        """

    @not_implemented
    def apply_tp(self, model, enable_tpsp):
        # call distribute_model
        # Please refer to torchtitan.models.deepseek_v3.infra.parallelize when implementing this
        ...

    @not_implemented
    def apply_etp(self, model):
        ...

    @not_implemented
    def apply_sp(self, model, sp_mesh):
        # generally skipped
        ...

    def setup_global_states(self, global_states: GlobalStates):
        """Mainly used to setup MOE implementation type (PTM or Titan)"""
        pass

    @not_implemented
    def apply_ep(self, model):
        # only required for ByteDance VeoOmni MOE
        """
        from torch.distributed.tensor.placement_types import Shard
        assert self.parallel_dims.ep_enabled
        # ptm moe don't use apply_ep
        assert model.moe_config is not None, 'Enabling ep but no moe config'
        if model.moe_config.use_ptm_moe:
            return

        def set_module_from_path(model: nn.Module, path: str, path_new: str, value: any):
            attrs = path.split(".")
            attrs_new = path_new.split(".")
            if len(attrs) == 1:
                setattr(model, attrs_new[0], value)
                if attrs_new[0] != attrs[0]:
                    delattr(model, attrs[0])
            else:
                next_obj = getattr(model, attrs[0])
                set_module_from_path(
                    next_obj,
                    ".".join(attrs[1:]),
                    ".".join(attrs_new[1:]),
                    value
                )

        ep_fqn_list = []
        ep_fqn_ep_list = []
        ep_local_chunk_list = []
        for fqn, param in model.named_parameters():
            if model.is_expert(fqn):
                from torch.distributed.tensor import distribute_tensor
                dtensor = distribute_tensor(
                    param.data,
                    # self.parallel_dims.world_mesh['ep'],
                    self.parallel_dims.ep_mesh,
                    placements=[Shard(0)],
                )
                local_chunk = torch.nn.Parameter(
                    dtensor.to_local(),
                    requires_grad=param.requires_grad
                )
                new_fqn = fqn

                ep_fqn_list.append(fqn)
                ep_fqn_ep_list.append(new_fqn)
                ep_local_chunk_list.append(local_chunk)

        for fqn, fqn_ep, local_chunk in zip(
            ep_fqn_list, ep_fqn_ep_list, ep_local_chunk_list
        ):
            set_module_from_path(model, fqn, fqn_ep, local_chunk)
        """

    @staticmethod
    def is_expert(fqn):
        return (
            ("experts" in fqn and "deepspeed" not in fqn and 'shared_experts' not in fqn)
            or "deepspeed_experts" in fqn
        )

    def is_moe(self, module_full_name, module):
        from hy_parallelism.models.modules.moe import PTMHunYuanMoE, HunYuanMoE, MoE
        if isinstance(module, (PTMHunYuanMoE, HunYuanMoE, MoE)):
            return True
        # TODO: Determine if the module is a MoE module by checking the fqn
        return False

    @not_implemented
    def is_moe_router(self, module_full_name, module):
        from torchtitan.models.moe.moe import TokenChoiceTopKRouter
        if isinstance(module, TokenChoiceTopKRouter):
            return True
        if isinstance(module_full_name, str) and \
            (
                module_full_name.endswith('.router') or
                (module_full_name.endswith('.gate') and 'router' not in module_full_name) # puretorch gate, filter out router.gate (submodule of titan router)
            ):
            return True
        # TODO: Determine if the module is a router module by checking the fqn
        return False

    @not_implemented
    def apply_compile(self, model): ...

    @not_implemented
    def apply_quantization(self, model): ...



class BaseParallelEngine(DeepSpeedInterface, EngineInterface):
    """
    __call__ 设计准则：


    如果开了pp:
        __call__ 返回: cache 好的 loss [closure 返回的，梯度累积之间 mean reduce] 或 ret_val [forward 返回值，无修改] (根据 self.training)
        get_cached_result: loss [closure 返回的，无修改] 或 ret_val [forward 返回值，无修改]
        get_real_ret 返回: cache 好的 loss [closure 返回的，无修改] 或 ret_val [forward 返回值，无修改] (根据 self.training)

    如果没开pp:
        __call__ 返回: cache 好的 loss [closure 返回的，无修改] 或 ret_val [forward 返回值，无修改] (根据 self.training)
        get_cached_result 返回: ...
        get_real_ret 返回: cache 好的 loss [closure 返回的，无修改] 或 ret_val [forward 返回值，无修改] (根据 self.training)
    """

    def is_final_step_rank(self) -> bool:
        """Check if current rank is the final step in pipeline."""
        if self.enable_pp:
            # assert len(self.pp_stages) == 1
            return self.pp_stages[-1].is_last
        return True

    def get_sub_device_mesh(self, mesh, key):
        try:
            return mesh[key]
        except Exception:
            return None

    def _check_scheduler(self) -> None:
        """Verify pipeline schedule configuration."""
        schedule_class = get_schedule_class(self.pipeline_parallel_schedule)
        if len(self.pp_models) > 1:
            assert issubclass(schedule_class, PipelineScheduleMulti)
            assert isinstance(self.pp_schedule, PipelineScheduleMulti)
        else:
            assert issubclass(schedule_class, PipelineScheduleSingle)
            assert isinstance(self.pp_schedule, PipelineScheduleSingle)

    def set_high_prio_micro_batch_size(self, micro_batch_size):
        self.__high_prio_micro_batch_size = micro_batch_size
        self.force_refresh_pipeline_scheduler()

    def set_high_prio_m_microbaches(self, m_microbatches):
        self.__high_prio_micro_batch_size = None
        self._m_microbatch_dict[self.training] = m_microbatches
        self.force_refresh_pipeline_scheduler()

    def force_refresh_pipeline_scheduler(self):
        if self.training:
            self.train(force=True)
        else:
            self.eval(force=True)

    def __init__(
        self,
        model,

        # ---------- Parallelism / Training Settingss ----------
        enable_fsdp=True,
        enable_ddp=False,
        enable_tpsp=False,
        enable_compile=False,
        gradient_accumulation_steps=1,
        gradient_sync_during_accumulation=True, # True saves memory, False saves time
        enable_gradient_checkpointing=False,
        cpu_offload=False,
        activation_offloading=False,
        activation_offloading_pin_memory=True,
        optimizer_offloading=False,
        optimizer_offloading_pin_memory=True,

        # ---------- Mixed Precision/CUDA/Autocast/Quantization ----------
        enable_autocast=True,
        autocast_prec="bf16",
        enable_grad_scaler=False,
        enable_quantization=False,             # Not implemented

        # ---------- Pipeline Parallelism & Scheduling ----------
        pipeline_parallel_schedule=None,       # 1F1B 需要 m_microbatch >= pp_size
        auto_benchmark=False,
        training_benchmark_schedule=("1F1B", "GPipe"),
        benchmark_m_list=None,                 # 需要测试的 m_micro_batch.  [即梯度累积次数]
        max_pp_batchsize=None,                 # 仅在auto_benchmark=True时生效，用于避免m太小时，显存不足。accumulate 模式下必须 1F1B
        batch_size=None,                       # Deprecated, must be None, use micro_batch_size instead
        m_microbatch=None,                     # 需要所有rank的batchsize都能整除这个数. 梯度累积次数
        micro_batch_size=None,                 # 优先级最高！一般不需要设置这个，除非场景。一般只需要设置 m_microbatch 和 batch_size (最终 m 用多少还是看 _m_microbatch_dict)
        collection_to_object=False,            # tuple, list, dict 不能很好地被 pp 处理，有可能出现错误切分，因此需要手动包装一层, 但这种只能处理 static kwargs. stage 之间不允许传这个

        # ---------- Loss & Optimizer & LR Scheduler ----------
        loss_closure=None,
        optimizer_config=None,                 # A dict, can be created by `get_optimizer_config`. See also stateful.py `OptimizersContainer.__init__`
        get_lr_scheduler_func=None,

        # ---------- Model State/Checkpoint ----------
        full_sd=None,                          # 切并行之前读的ckpt, 如果用的是天然切好的模型，那这个将不允许使用
        load_ckpt_path=None,                   # 切并行之后读的ckpt, 调用 load_checkpoint
        ckpt_dir=None,                         # 模型保存的路径, 训练时使用
        keep_latest_k=-1,                      # 保存的时候保留多少个checkpoint，-1 表示保留所有
        initial_training_states: dict | None = None,
        model_stateful_class: Type[Stateful]=None,
        extra_dcp_states=None,

        # ---------- Model Initialization/Meta Param ----------
        initialize_meta_param=True,            # 为了临时适配jianwei的需求加的，如果为 False，则不在切并行之前对meta参数做初始化
        init_meta_stage="post_fsdp",           # ['pre_fsdp' | 'post_fsdp' ]
        call_module_reset_parameters=True,     # 是否递归调用各 module 的 reset_parameters
        dp_replicate_param_handler='sync',     # ['sync' | 'safe_sync' | 'check', 'none', 'noop']

        # ---------- Debug/Check ----------
        check_grad=False,                     # 是否检查梯度是否正确, 并且是否每一个参数都有梯度

        # ---------- Temporary hacks ----------
    ):
        r"""Initialize the ParallelEngine.

        This constructor sets up the parallel engine for distributed training or inference,
        supporting various parallelism strategies (pipeline, tensor, sequence, expert, FSDP, DDP),
        mixed precision, gradient accumulation, and checkpointing.

        Args:
            model (nn.Module): The model to be parallelized and trained or evaluated.
            enable_gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing
                for memory savings.
            cpu_offload (bool, optional): Whether to enable CPU offloading for FSDP.

            # ---------- Mixed Precision/CUDA/Autocast/Quantization ----------
            enable_autocast (bool, optional): Whether to enable automatic mixed precision.
            autocast_prec (str, optional): Precision for autocast. One of 'bf16', 'fp16', 'fp32', etc.
            enable_grad_scaler (bool, optional): Whether to enable gradient scaler for mixed precision training.
            enable_quantization (bool, optional): Whether to enable quantization. Not yet implemented.

            # ---------- Pipeline Parallelism & Scheduling ----------
            pipeline_parallel_schedule (str or tuple, optional): The pipeline parallel schedule to use.
                If None, defaults to 'GPipe'. Note that '1F1B' requires m_microbatch >= pp_size.
            training_benchmark_schedule (tuple, optional): Pipeline parallelism schedules to benchmark.
                Used when auto_benchmark is enabled. Default: ('1F1B', 'GPipe').
            auto_benchmark (bool, optional): Whether to automatically benchmark and select the optimal
                pipeline parallel schedule and microbatch size. If True, will override some other settings.
            max_pp_batchsize (int, optional): Maximum batch size for pipeline parallelism when
                auto_benchmark is enabled. Used to avoid out-of-memory errors when microbatch size is too small.
                In accumulation mode, '1F1B' schedule is required.
            batch_size (int, optional): Deprecated. Must be None. Use micro_batch_size instead.
            m_microbatch (int, optional): Number of microbatches for pipeline parallelism. The batch size
                on all ranks must be divisible by this number. If None, will be set to 1 unless
                auto_benchmark is enabled.
            micro_batch_size (int, optional): The microbatch size. Has the highest priority for determining
                microbatching. Generally not needed unless for special scenarios. The actual number of
                microbatches used is determined by the combination of m_microbatch, micro_batch_size, and
                batch_size, with micro_batch_size having the highest priority.
            collection_to_object (bool, optional): Whether to convert collections (tuple, list, dict) to
                objects for pipeline parallelism. Collections cannot be properly handled by PP and may
                cause incorrect partitioning. Only works with static kwargs. Cannot be passed between stages.

            # ---------- Loss & Optimizer & LR Scheduler ----------
            loss_closure (callable, optional): A custom loss closure function. Can be registered later
                using the `register_loss_closure` method.
            optimizer_config (dict, optional): Configuration for the optimizer(s). Should include
                `optimizer_cls` and `optimizer_kwargs`. An optional `filter_param_func` can be provided.
            get_lr_scheduler_func (callable, optional): Function to create learning rate scheduler(s).

            # ---------- Model State/Checkpoint ----------
            full_sd (dict, optional): Full state_dict loaded before partitioning the model. Should not
                be used if the model is already partitioned during initialization. This is a checkpoint
                loaded before applying parallelism sharding.
            load_ckpt_path (str, optional): Path to a checkpoint to load after partitioning. The engine
                will call load_checkpoint to load this checkpoint after the model is partitioned.
            ckpt_dir (str, optional): Directory path for saving model checkpoints during training.
            keep_latest_k (int, optional): Number of latest checkpoints to keep. Default: -1 (keep all).
            initial_training_states (dict, optional): Initial training state variables.

            # ---------- Parallel/Distributed ----------
            enable_fsdp (bool, optional): Whether to enable Fully Sharded Data Parallel (FSDP).
            enable_ddp (bool, optional): Whether to enable Distributed Data Parallel (DDP).
            gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before
                performing an optimizer step.

            # ---------- Model Initialization/Meta Param ----------
            initialize_meta_param (bool, optional): Whether to initialize meta parameters before applying
                parallelism. If False, meta parameters will not be initialized before sharding. This is
                a temporary workaround for jarvizhang's specific use cases.
            init_meta_stage (str, optional): Stage at which to initialize meta parameters. Options are
                'pre_fsdp' or 'post_fsdp'. 'pre_fsdp' is designed for MoE models where nn.init.kaiming_uniform_
                initialization requires slicing, which DTensor cannot handle. 'post_fsdp' better supports
                meta parameters and avoids memory/GPU memory peaks. Default: 'post_fsdp'.
            dp_replicate_param_handler (str, optional): Handler for parameter synchronization across dp replication.
                Options are 'sync', 'safe_sync', or 'check'. Default: 'sync'.
                'sync': Perform synchronous broadcast.
                'safe_sync': Perform synchronous broadcast with safe checks.
                'check': Perform asynchronous broadcast with safe checks.

            # ---------- Compile/Logger ----------
            logger (logging.Logger, optional): Logger instance to use.
            enable_compile (bool, optional): Whether to enable PyTorch compilation for performance optimization.

        Notes:
            - If both auto_benchmark and micro_batch_size/m_microbatch are not set, auto_benchmark
              will be enabled by default.
            - The actual microbatching used is determined by the combination of m_microbatch,
              micro_batch_size, and batch_size, with micro_batch_size having the highest priority.
            - [Deprecated]: Generally, we need to implement `pp_friendly_forward` before super().__init__().
              The `pp_friendly_forward` handles pp_args preprocessing, etc.
              Example:
              ```
                def __init__(self, model, *args, **kwargs):

                    old_forward = model.forward
                    from functools import wraps

                    def pp_friendly_forward(
                        self,

                        pp_img=None, pp_txt=None, pp_vec=None, pp_text_mask=None,
                        pp_loss_moe_0=None, pp_loss_moe_1=None,
                        pp_token_replace_vec=None, # optional, put in the end

                        hidden_states: torch.Tensor=None,
                        ...

                        # Replace extra_kwargs
                        byt5_text_states=None,
                        byt5_text_mask=None,
                        is_token_replace=False,
                    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
                        extra_kwargs = dict(
                            byt5_text_states=byt5_text_states,
                            byt5_text_mask=byt5_text_mask,
                            is_token_replace=is_token_replace,
                        )

                        return old_forward(
                            hidden_states=hidden_states,
                            ...,

                            extra_kwargs=extra_kwargs,


                            pp_img=pp_img,
                            pp_txt=pp_txt,
                            pp_vec=pp_vec,
                            pp_text_mask=pp_text_mask,
                            pp_loss_moe_0=pp_loss_moe_0,
                            pp_loss_moe_1=pp_loss_moe_1,
                            pp_token_replace_vec=pp_token_replace_vec,
                        )

                    if self.parallel_dims.pp_enabled:
                        model.forward = pp_friendly_forward.__get__(model)

                    super().__init__(model, *args, **kwargs)
            ```

        """
        # Handles mutable default arguments
        extra_dcp_states = {} if extra_dcp_states is None else extra_dcp_states
        initial_training_states = {} if initial_training_states is None else initial_training_states

        self.model = model
        super().__init__()

        parallel_dims = get_parallel_state()

        from hy_parallelism.common.constants import TORCH_DTYPE_MAP as precision_map
        if autocast_prec not in precision_map:
            raise ValueError(f"Invalid autocast precision: {autocast_prec}")

        self.autocast_prec = precision_map[autocast_prec]

        self.enable_pp = parallel_dims.pp_enabled
        self.enable_tp = parallel_dims.tp_enabled
        self.enable_etp = parallel_dims.etp_enabled
        self.enable_sp = parallel_dims.sp_enabled # Legacy, same as cp
        self.enable_cp = parallel_dims.cp_enabled
        self.enable_tpsp = enable_tpsp
        if self.enable_tpsp:
            assert self.enable_tp
            assert not self.enable_cp, 'TPSP with CP is not tested yet.'
        self.enable_ep = parallel_dims.ep_enabled
        self.enable_fsdp = enable_fsdp
        self.enable_ddp = enable_ddp
        self.enable_quantization = enable_quantization
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_compile = enable_compile
        self.parallel_dims = parallel_dims
        self.initialize_meta_param = initialize_meta_param
        self.init_meta_stage = init_meta_stage
        self.call_module_reset_parameters = call_module_reset_parameters
        assert dp_replicate_param_handler.lower() in ['sync', 'safe_sync', 'check', 'none', 'noop'], f"Invalid dp_replicate_param_handler: {dp_replicate_param_handler}"
        self.dp_replicate_param_handler = dp_replicate_param_handler.lower()
        self.model_stateful_class = model_stateful_class
        self.enable_autocast = enable_autocast
        self.extra_dcp_states = extra_dcp_states
        self.benchmark_m_list = benchmark_m_list
        self.check_grad = check_grad


        self.enable_grad_scaler = enable_grad_scaler
        if self.enable_grad_scaler:
            self.grad_scaler = GradScaler(device="cuda")
            if self.autocast_prec in [torch.float32, torch.bfloat16]:
                loguru.logger.warning("Grad scaler is not needed for float32 or bfloat16 precision.")


        if parallel_dims.pp_enabled:
            assert is_implemented(self.config_forward_args)

        self._check_implementation_status()
        assert batch_size is None, "Deprecated argument `batch_size`, use `micro_batch_size` instead."

        if micro_batch_size is not None:
            assert not auto_benchmark and m_microbatch is None
            self.__high_prio_micro_batch_size = micro_batch_size
        else:
            self.__high_prio_micro_batch_size = None

        # fix pipeline_parallel_schedule and m_microbatch
        if pipeline_parallel_schedule is None or m_microbatch is None:
            if not auto_benchmark:
                if micro_batch_size is None and m_microbatch is None:
                    auto_benchmark = True
            if pipeline_parallel_schedule is None:
                pipeline_parallel_schedule = "GPipe"
            if m_microbatch is None:
                m_microbatch = 1


        def cache_result(self, tag, ret_val):
            key = f"_pp_result_cache_list_{tag}"
            # here we use `model` instead of `self` to enable submodules calling cache_result()
            # and save the result in root module
            # FIXME(kevinkhwu): do not save the attributes to 'model', save elsewhere
            if not hasattr(model, key):
                setattr(model, key, [])
                if hasattr(model, "_pp_valid_result_keys"):
                    getattr(model, "_pp_valid_result_keys").add(key)
                else:
                    setattr(model, "_pp_valid_result_keys", {key})
            else:
                getattr(model, "_pp_valid_result_keys").add(key)

            cached_tensor = ret_val
            getattr(model, key).append(cached_tensor)

        # We might need call cache_result in the submodules.
        # Registering cache_result in the submodules.
        # NOTE: In the latest implementation, we cache ret_val, loss, loss_dict in `loss_closure_forward` defined below.
        #   We might not need to manually cache the result in forward pass anymore.
        #   However, we keep this to enabling users to manually retrive intermediate results in forward pass.
        for name, module in model.named_modules():
            module.cache_result = cache_result.__get__(module)
            assert not hasattr(module, "loss_closure"), f"{name} {type(module)} already has a loss_closure ({module.loss_closure})"
            module.loss_closure = self.loss_closure

        self.cache_result = cache_result.__get__(self)

        # which scheduler to use when calling .train()
        # different from self.pipeline_parallel_schedule, which is the scheduler that is currently used
        self.training_pipeline_parallel_schedule = pipeline_parallel_schedule
        self.set_pp_schedule(pipeline_parallel_schedule)

        if auto_benchmark and pipeline_parallel_schedule is not None:
            assert isinstance(training_benchmark_schedule, (tuple, list))
            for schedule in training_benchmark_schedule:
                assert self.is_multi_schedule(schedule) == self.is_multi_schedule(pipeline_parallel_schedule), (
                    "Configuration of `pipeline_parallel_schedule` and `training_benchmark_schedule` is imcompatible."
                )

        self.cpu_offload = cpu_offload
        self.activation_offloading = activation_offloading
        self.activation_offloading_pin_memory = activation_offloading_pin_memory
        self.optimizer_offloading = optimizer_offloading
        self.optimizer_offloading_pin_memory = optimizer_offloading_pin_memory
        self.model_pinned_memory_pool = None
        if self.optimizer_offloading_pin_memory:
            self.optimizer_pinned_memory_pool = PinnedMemoryPool()
        self.full_sd = full_sd
        self.load_ckpt_path = load_ckpt_path

        self.ckpt_dir = ckpt_dir
        self.keep_latest_k = keep_latest_k
        self.max_pp_batchsize = max_pp_batchsize
        self.gradient_accumulation_steps = gradient_accumulation_steps  # gradient accumulation related
        self.gradient_sync_during_accumulation = gradient_sync_during_accumulation
        self.micro_steps = 0  # gradient accumulation related

        if self.gradient_accumulation_steps > 1:
            assert not self.enable_pp, "Gradient accumulation is not supported in pipeline parallel mode yet."

        # states
        self.training = True
        self.reso_stage_manager = {}
        self.batch_object_kwargs_keys = []
        self.sync_input = None

        self._default_fsdp_kwargs = dict(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            default_fsdp_mesh=parallel_dims.default_fsdp_mesh,
            expert_fsdp_mesh=parallel_dims.expert_fsdp_mesh,
            cpu_offload=self.cpu_offload,
        )

        if self.autocast_prec == torch.float32 and enable_autocast:
            raise ValueError("Autocast should be disabled for float32 precision")

        self.__loss_closure = loss_closure

        self.training_states = dict(
            tp=self.parallel_dims.tp,
            ep=self.parallel_dims.ep,
            pp=self.parallel_dims.pp,
            cp=self.parallel_dims.cp,
            step_cnt=0,
        )
        if initial_training_states:
            self.training_states["client_state"] = initial_training_states

        self.optimizer_config = optimizer_config
        self.get_lr_scheduler_func = get_lr_scheduler_func
        self._check_optimizer_config()

        self.pp_mesh = parallel_dims.pp_mesh

        if self.pp_mesh is None:
            assert not self.enable_pp
            self.pp_rank = 0
            self.pp_size = 1
        else:
            self.pp_rank = self.pp_mesh.get_local_rank()
            self.pp_size = self.pp_mesh.size()
        self.tp_mesh = parallel_dims.tp_mesh
        self.sp_mesh = parallel_dims.sp_mesh

        self.device = torch.device("cuda")


        if not self.enable_pp:
            self.pp_models = [model]
        else:
            self.pp_stages, self.pp_models = self.pipeline_manual_split()
            self._check_pp_no_duplicate()

        if is_implemented(self.pp_friendly_forward):
            for m in self.pp_models:
                m._original_forward = m.forward
                m.forward = self.__class__.pp_friendly_forward.__get__(m)

        for m in self.pp_models:
            # Note: `_original_forward_or_pp_friendly_forward` here could potentially be a bound method of pp_friendly_forward or the original forward method
            _original_forward_or_pp_friendly_forward = m.forward

            # inspect will unwrap the bound method to get the signature
            @wraps(m.forward.__func__)
            def loss_closure_forward(model_self, *args, calc_loss=True, **kwargs):  # Similar to register_after_forward_hook
                assert self.enable_pp, "Jan 22 2026: Only pp should call this function."

                """
                下面这一块是原本尝试支持 stage 之前传 object 的，但应用场景比较少，且没开发完，容易出错，暂时不启用
                """
                # from hy_parallelism.utils import batch_obj_to_tensor, obj_to_tensor
                # from hy_parallelism.utils import batch_tensor_to_obj, tensor_to_obj
                # for k in self.batch_object_kwargs_keys:
                #     if k in kwargs:
                #         # kwargs[k] = batch_tensor_to_obj(kwargs[k])
                #         kwargs[k] = tensor_to_obj(kwargs[k][0])


                if self.enable_pp and self.collection_to_object:
                    for k in kwargs:
                        if isinstance(kwargs[k], SafeObjectWrapper):
                            kwargs[k] = kwargs[k].obj

                if is_implemented(self.mock_pp_forward) and self._is_mock_stage:
                    ret = self.mock_pp_forward(*args, **kwargs)
                else:
                    ret = _original_forward_or_pp_friendly_forward(*args, **kwargs)

                # NOTE(IMPORTANT): 如果针对pp改了cache的逻辑，别忘了同步修改非pp下的 get_real_ret 的实现。因为目前这里只有pp才会进来
                model_self.cache_result("ret_val", ret)

                # only the last pp rank should call this
                if self.enable_pp and not model_self.is_last_stage:
                    if isinstance(ret, (list, tuple)):
                        original_type = type(ret)
                        ret = list(ret)
                        for return_idx, val in enumerate(ret):
                            self._check_output(val)
                            if isinstance(val, torch.Tensor):
                                ret[return_idx] = val.contiguous()  # pytorch pp 的 bug， 不 contiguous 会有神奇问题
                        if original_type is tuple:
                            ret = tuple(ret)
                    else:
                        assert isinstance(ret, torch.Tensor)
                        ret = ret.contiguous()  # pytorch pp 的 bug， 不 contiguous 会有神奇问题

                if self.training and (not self.enable_pp or model_self.is_last_stage) and calc_loss:
                    # NOTE(WARNING): 如果是 replace 的情况， kwargs 的 loss 计算不能和 被 replace 的东西相关！不然会出问题
                    #                例如 假如 loss 计算使用 x 和 output 算 MSE，但 x 有可能倍替换了，所以计算出来的 loss 不正确
                    # NOTE(IMPORTANT): 如果针对pp改了cache的逻辑，别忘了同步修改非pp下的 get_real_ret 的实现。因为目前这里只有pp才会进来
                    pp_friendly_loss, loss_closure_out = self.loss_closure(ret, {"args": args, "kwargs": kwargs})
                    loss, loss_dict = loss_closure_out
                    model_self.cache_result("loss", loss)
                    model_self.cache_result("loss_dict", loss_dict)
                    if self.enable_pp:
                        if self.enable_grad_scaler:
                            return self.grad_scaler.scale(pp_friendly_loss)  # 开了pp的话最后一个stage在training时必须返回 loss
                        else:
                            return pp_friendly_loss  # 开了pp的话最后一个stage在training时必须返回 loss
                    else:
                        return loss
                else:
                    # 有三种情况可能会进入这个分支：
                    #   1. 要么 validation (不需要走上面计算loss)
                    #   2. 要么 pp 的中间 stage (非 last stage，所以不计算loss)
                    #   ~3. （开pp或者没开pp) calc_loss=False~
                    if not self.enable_pp:
                        raise RuntimeError("Should never happen")
                        return ret
                    else:
                        if model_self.is_last_stage:
                            if not self.training:
                                bs = self.microbatch_size
                                # pp validation, return anything (real return value is cached)
                                # return torch.tensor(-2.0) # Not [-2.0], Encounter error: `tage 1 forward outputs: Number of values (1) does not match expected number (2)` for bs=2
                                return torch.tensor([-2.0] * bs).cuda(non_blocking=True)  # make pp happy
                            else:
                                raise ValueError(
                                    "Setting calc_loss=False with model.training=True?"
                                    "When enabling pp, This will call backward automatically since trining=True, leading to errors."
                                    "If what you want is to return the forward result instead of the loss, set training=False directly."
                                )
                        else:
                            return ret

            # NOTE: Why we add `self.enable_pp` here:
            #
            #       Background:
            #           Only pp requires calculating loss and caching the results in forward.
            #           However, to keep the API consistency, we need to ensure that the engine returns
            #           loss in training mode and return forward_out in eval mode.
            #           We also want the engine can call `get_cached_result` even pp is not enabled.
            #
            #       The initial implementation:
            #           Before, we wrap the forward with loss_closure_forward.__get__(m) even pp is not enabled.
            #           This make HYImage3.5 fail to call `clear_cached_result` after forward. since they directly
            #           call the original forward method rather than Engine.__call__ in their `GenerationMixin``.
            #
            #       To fix this:
            #            for non-pp case, we keep the forward unchanged and cache the result in __call__.
            #            returning loss in get_real_ret.
            if self.enable_pp:
                m.forward = loss_closure_forward.__get__(m)

        if enable_autocast:
            # TODO(kevinkhwu): owing to PyTorch issue, we can not run schedule.step() under autocast. (no_grad in shape_inference)
            #   A workaround is to enable autocast only in the forward pass.
            #   But gradient checkpointing can lead to dtype mismatch during the forward pass in the backward passes.
            #   so we need to enable autocast for all module (especially gradient checkpointing blocks).
            #   When the PyTorch bug is fixed, wrap the autocast context around the schedule.step()
            #   See also: https://github.com/pytorch/pytorch/issues/158232
            m.forward = wraps(m.forward)(torch.amp.autocast("cuda", dtype=self.autocast_prec)(m.forward))

            """
            AC(FSDP) has a bug. We wrap all submodules' forward with autocast as a workaround.

            Sep 28 2025: This bug can be fixed by implementing checkpointing in another way.
            We thus don't apply autocast to submodules since this makes it hard to disable autocast inside forward call (its submodule will be enabled again)
            """
            # for module in m.modules():
            #     module.forward = wraps(module.forward)(
            #         torch.amp.autocast('cuda', dtype=self.autocast_prec)(module.forward)
            #     )

            # Disable autocast for MoE modules
            for module in m.modules():
                if is_implemented(self.is_moe) and self.is_moe(None, module):
                    module.forward = wraps(module.forward)(torch.amp.autocast("cuda", enabled=False)(module.forward))
                if is_implemented(self.is_moe_router) and self.is_moe_router('', module):
                    module.forward = wraps(module.forward)(torch.amp.autocast("cuda", enabled=False)(module.forward))

        if not self.enable_pp:
            self.post_init()

            assert len(self.pp_models) == 1
            assert len(self.fsdp_models) == 1
            return

        # Auto benchmark related variables
        self.is_first_run = {True: True, False: True}
        self.benchmark_schedule = {True: training_benchmark_schedule, False: [self.get_inference_pipeline_schedule()]}
        self.enable_benchmark = auto_benchmark
        self.enable_autocast = enable_autocast
        self.collection_to_object = collection_to_object

        if self.__high_prio_micro_batch_size is None:
            self._m_microbatch_dict = {True: m_microbatch, False: 1}

        if batch_size is None:
            batch_size = 2**20  # use large enough batchsize to pass all check for build pipeline schedule
        self.set_micro_batch_size(batch_size)

        self.post_init()

        self.loss_fn = ModelToLossFn(self.fsdp_models[-1], get_m_microbatch_fn=lambda: self.m_microbatch)

        assert self.loss_fn is not None

        self.build_pp_schedule()

        self.config_forward_args()

        self._check_scheduler()

        # Remove the reference to 'model' from self to allow for correct object reference counting and proper GPU memory release.
        # This is important when the final model used is a deepcopy, making this reference unnecessary.
        delattr(self, "model")

    def _check_optimizer_config(self):
        if self.optimizer_config is None:
            return

        if not isinstance(self.optimizer_config, dict):
            raise ValueError(f"optimizer_config must be a dict, but got {type(self.optimizer_config)}")

        if self.optimizer_config.get('optimizer_cls') is None or self.optimizer_config.get('optimizer_kwargs') is None:
            if self.optimizer_config.get('optimizer_factory_for_special_param') is None:
                raise ValueError(f"`optimizer_cls` and `optimizer_kwargs` entries are required in `optimizer_config`, but got {self.optimizer_config.keys()}")

    @property
    def default_fsdp_kwargs(self):
        return copy.deepcopy(self._default_fsdp_kwargs)

    def post_setup_global_states(self, global_states: GlobalStates):
        # Avoid importing MOE. Some machine does not have torchtitan installed.
        pass
        # from hy_parallelism.models.modules.moe import MoE
        # for name, module in self.named_modules():
        #     if isinstance(module, MoE):
        #         global_states.use_titan_moe = True

    def _check_implementation_status(self):
        if self.enable_fsdp:
            assert is_implemented(self.apply_fsdp)
        if self.enable_quantization:
            assert is_implemented(self.apply_quantization)

    def state_dict(self):
        """Return the state dictionary of the model for the current pipeline stage.

        This method only handles DTensor, which means it returns only the state of the
        current pipeline stage when pipeline parallel is enabled.
        The returned state dict includes DTensors (distributed tensors) as-is,
        without converting them to regular tensors.

        Note:
            To get the full state dict across all pipeline stages, use `full_state_dict()`
            instead.
        """
        from hy_parallelism.checkpoint.checkpoint_manager import MODEL
        return self.model_checkpoint_manager.states[MODEL].state_dict()

    def full_state_dict(self, lazy=False):
        """Return the complete state dictionary of the model across all pipeline stages.

        This method gathers state dicts from all pipeline stages (when pipeline parallelism
        is enabled) and converts DTensors to regular tensors. All tensors are moved to CPU
        to avoid out-of-memory issues.

        Warning:
            This is an experimental feature and may not work as expected. Use with caution.

        Note:
            For pipeline-parallel training, this method gathers state dicts from all ranks
            in the pipeline parallel group and merges them into a single complete state dict.
            Use `state_dict()` if you only need the state of the current pipeline stage.

        """
        # TODO(kevinkhwu): Experimental feature
        loguru.logger.warning("`Engine.full_state_dict` is an experimental feature.")

        dist.barrier()
        if lazy:
            raise NotImplementedError

        from hy_parallelism.checkpoint.lazy_state_dict import LazyStateDict
        from functools import partial

        ret = {}

        def get_full_tensor(v): return v.full_tensor()

        for k, v in self.state_dict().items():
            if lazy:
                ret[k] = partial(get_full_tensor, v)
            else:
                if isinstance(v, DTensor):
                    v = v.full_tensor().cpu() # cast to cpu to avoid OOM
                ret[k] = v

        if self.parallel_dims.pp_enabled:
            debug_log(f'Gather PP param from {self.parallel_dims.pp_group}.')

            from hy_parallelism import utils

            sds = utils.gather_obj(ret, group=self.parallel_dims.pp_group)
            ret = {}
            for sd in sds:
                ret.update(sd)

        if lazy:
            ret = LazyStateDict(ret)

        return ret

    def load_state_dict(self, state_dict, strict=True, is_non_standard_sharded_sd=False):
        return self.model_checkpoint_manager.load_full_sd(state_dict, strict=strict, is_non_standard_sharded_sd=is_non_standard_sharded_sd)

    @property
    def optimizer(self):
        if not hasattr(self, "optimizer_container"):
            raise RuntimeError("Should parse optimizer_config to pp_engine initialization")
        return self.optimizer_container

    @property
    def lr_scheduler(self):
        if not hasattr(self, "lr_scheduler_container"):
            loguru.logger.critical(f"Should parse optimizer_config to pp_engine initialization")
            raise RuntimeError("Should parse optimizer_config to pp_engine initialization")
        return self.lr_scheduler_container

    def register_fsdp_forward_method(self, method_name: str):
        import torch.distributed.fsdp
        for model in self.fsdp_models:
            torch.distributed.fsdp.register_fsdp_forward_method(model, method_name)

    def register_loss_closure(self, loss_fn):
        r"""Register a custom loss closure function.

        This method registers a loss function that will be called automatically during
        forward passes. When registered, calling ``engine(*args, **kwargs)`` becomes
        equivalent to:

        .. code-block:: python

            def __call__(self, *args, **kwargs):
                model_output = model(*args, **kwargs)
                loss, loss_dict = loss_fn(model_output, {"args": args, "kwargs": kwargs})
                if model.training:
                    return loss
                else:
                    return model_output

        The registered loss function transforms the engine's behavior:
        - In training mode (``model.training=True``), the engine returns the computed loss tensor.
        - In evaluation mode (``model.training=False``), the engine returns the model output.

        The loss dictionary returned by the loss function can be retrieved using:
        ``engine.get_cached_result("loss_dict")`` or ``engine.get_last_call_result("loss")``.

        Args:
            loss_fn (callable): A function that computes the loss. Must accept exactly two
                arguments:
                1. ``model_output``: The return value of the model's forward method.
                2. ``model_input``: A dictionary containing ``{"args": args, "kwargs": kwargs}``
                   where ``args`` and ``kwargs`` are the arguments passed to the engine.

                The function must return a tuple ``(loss, loss_dict)`` where:
                - ``loss`` (Tensor): The computed loss tensor. Should have a batch dimension;
                  if the loss is a scalar (0-dimensional), it will be treated as batch size 1.
                - ``loss_dict`` (dict): A dictionary of loss components or metrics. Can be
                  an empty dictionary if no additional metrics are needed.

        Examples::

            >>> import torch
            >>> import torch.nn as nn
            >>> from hy_parallelism.engines import ParallelEngine
            >>>
            >>> model = nn.Linear(10, 1)
            >>> engine = ParallelEngine(model)
            >>>
            >>> # Define a simple loss closure
            >>> def loss_closure(model_output, model_input):
            ...     loss = model_output.mean()
            ...     return loss, {'loss': loss.item()}
            >>>
            >>> engine.register_loss_closure(loss_closure)
            >>> engine.train()
            >>>
            >>> # Forward pass returns the loss in training mode
            >>> loss = engine(torch.randn(32, 10))
            >>>
            >>> # Retrieve loss dictionary
            >>> loss_dict = engine.get_cached_result("loss_dict")
            >>>
            >>> # In eval mode, returns model output instead
            >>> engine.eval()
            >>> output = engine(torch.randn(32, 10))  # Returns model output, not loss

        .. note::
            The loss function must accept exactly two arguments. If the function signature
            cannot be inspected, a warning may be suppressed, but incorrect usage will
            raise an error during execution.

        .. seealso::
            :meth:`set_loss_function` for an alias of this method.
        """
        try:
            arg_names = get_arg_names(loss_fn)
            assert len(arg_names) >= 2, f"Loss function {loss_fn} should accept two arguments (forward_output, forward_input[dict(args=..., kwargs=...)]), but got {arg_names}"
        except AssertionError:
            raise
        except:
            pass
        self.__loss_closure = loss_fn

    def remove_loss_closure(self): # somehow save memory
        self.__loss_closure = None

    def set_loss_function(self, loss_fn):  # Equivalent to register_loss_closure
        self.register_loss_closure(loss_fn)

    def loss_closure(self, *args, **kwargs):
        if self.__loss_closure is None:
            raise RuntimeError(
                "Loss closure is not registered. "
                "If you are training the model, please register a loss closure function using `register_loss_closure` method. "
                "If you are performing inference, you must call `engine.eval()` before calling forward."
            )

        loss_closure_out = self.__loss_closure(*args, **kwargs)
        (loss, loss_dict) = loss_closure_out
        if self.enable_pp:  # make pp happy
            pp_friendly_loss = loss.contiguous().clone()
            if len(pp_friendly_loss.shape) == 0:
                return pp_friendly_loss[None].contiguous().clone(), loss_closure_out
            else:
                return pp_friendly_loss, loss_closure_out
        else:
            return loss, loss_closure_out

    def create_checkpoint_manager(self, ckpt_dir, keep_latest_k=-1):
        self.MODEL_FOLDER = "weights"
        self.OPTIMIZER_FOLDER = "optimizers"
        self.TRAINING_STATES_FILE = "training_states.pt"

        extra_dcp_states = {}
        from hy_parallelism.checkpoint.stateful import wrap_model_to_stateful, ModelWrapper
        for k, v in self.extra_dcp_states.items():
            if isinstance(v, nn.Module):
                # Assuming other module do not contain old MOE.
                # Only old MOE use EPModelWrapper
                extra_dcp_states[k] = wrap_model_to_stateful([v], ModelWrapper),
            elif isinstance(v, Stateful):
                extra_dcp_states[k] = v
            else:
                raise ValueError(f"Unsupported type: {type(v)} for extra_dcp_states[{k}].")


        self.model_checkpoint_manager = CheckpointManager(
            extra_dcp_states,
            Checkpoint(
                dump_folder=ckpt_dir,
                folder=self.MODEL_FOLDER,
                enable_checkpoint=True,
                keep_latest_k=keep_latest_k,
            ),
            model_parts=self.fsdp_models,
            stateful_class=self.model_stateful_class,
        )
        # optimizer 和 weights 分开存，方便后面储存不够时只删 optimizer
        if hasattr(self, "optimizer_container") and hasattr(self, "lr_scheduler_container"):
            self.optimizer_checkpoint_manager = CheckpointManager(
                {},
                Checkpoint(
                    dump_folder=ckpt_dir,
                    folder=self.OPTIMIZER_FOLDER,
                    enable_checkpoint=True,
                    keep_latest_k=keep_latest_k,
                ),
                model_parts=None,
                optimizers=self.optimizer_container,
                lr_schedulers=self.lr_scheduler_container,
            )

    @property
    def m_microbatch(self):
        r"""Get the number of micro-batches (gradient accumulation steps) for pipeline parallelism.

        Returns the number of micro-batches used for gradient accumulation in pipeline parallelism.
        If pipeline parallelism is not enabled, returns 1. Otherwise, returns the value from
        :attr:`_m_microbatch_dict` based on the current training mode, or calculates it as
        ``batch_size // micro_batch_size`` if :attr:`__high_prio_micro_batch_size` is set.

        Returns:
            int: The number of micro-batches for gradient accumulation.

        .. note::
            When using high priority micro batch size, the batch size must be divisible by
            the micro batch size. Different batch sizes across data parallel groups could
            break expert parallelism (EP).

        .. warning::
            Batch size can be different in different data parallel groups, which could break
            expert parallelism (EP).
            Setting :attr:`__high_prio_micro_batch_size` can easily result in different `m` in
            different DP ranks.  If different DP rank has different `m_microbatch`, EP will hang.
        """
        if not self.enable_pp:
            return 1
        if self.__high_prio_micro_batch_size is None:
            return self._m_microbatch_dict[self.training]
        else:
            assert self.batch_size % self.__high_prio_micro_batch_size == 0, (
                f"batch_size ({self.batch_size}) must be divisible by the given micro batch size ({self.__high_prio_micro_batch_size})"
            )
            # WARNING: batch size can be different in different dp.
            # This could break EP.
            return self.batch_size // self.__high_prio_micro_batch_size

    @classmethod
    def get_inference_pipeline(cls, model, pipeline_parallel_schedule="GPipe", m_microbatch=1, **kwargs):
        engine = cls(
            model=model,
            pipeline_parallel_schedule=pipeline_parallel_schedule,
            m_microbatch=m_microbatch,
            enable_autocast=True,
            **kwargs,
        )
        engine.eval()
        return engine

    def zero_grad(self):
        if not hasattr(self, "optimizer_container"):
            for param in self.parameters():
                param.grad = None
        else:
            self.optimizer_container.zero_grad()

    def barrier_fsdp_groups(self):
        with profile_range("barrier_fsdp_groups"):
            if not self.enable_fsdp:
                return
            # Performing forward and backward requires communication on FSDP groups.
            # If previous communication encounters error, communication will become undefined behavior.
            # We explicitly barrier here to ensure all ranks have completed the previous communication correctly.
            for group in self.parallel_dims.default_fsdp_mesh.get_all_groups():
                dist.barrier(group)
            if self.parallel_dims.ep_enabled:
                for group in self.parallel_dims.expert_fsdp_mesh.get_all_groups():
                    dist.barrier(group)

    def backward(self, loss, retain_graph=None, create_graph=False, scale_wrt_gas=True, backward_fn=None):
        self.barrier_fsdp_groups()
        self.remove_loss_closure()
        self.clear_cached_result()
        if self.enable_pp:
            pass
        else:
            loss = loss / self.gradient_accumulation_steps
            if backward_fn is not None:
                backward_fn(loss)
            else:
                if self.enable_grad_scaler:
                    self.grad_scaler.scale(loss).backward(retain_graph=retain_graph, create_graph=create_graph)
                else:
                    loss.backward(retain_graph=retain_graph, create_graph=create_graph)

        from torch.distributed.fsdp._fully_shard import FSDPModule
        # HACK: Support old PyTorch version
        if self.enable_ep and not hasattr(FSDPModule, 'set_gradient_divide_factor') and self.is_gradient_accumulation_boundary():
            for name, param in self.named_parameters():
                if self.is_expert(name):
                    default_fsdp_mesh = get_parallel_state().default_fsdp_mesh
                    ep_fsdp_mesh = get_parallel_state().expert_fsdp_mesh
                    if param.grad is not None:
                        param.grad.data.mul_(ep_fsdp_mesh.size() / default_fsdp_mesh.size())

        if not retain_graph:
            get_pinned_memory_pool(ACTIVATION_POOL_NAME).reset()

        if self.check_grad:
            non_count = 0
            nan_inf_count = 0
            param_count = 0

            non_param_list = []
            nan_inf_param_list = []
            param_list = []
            for name, param in self.named_parameters():
                param_list.append(name)
                param_count += 1
                if param.grad is None:
                    non_count += 1
                    non_param_list.append(name)
                    # loguru.logger.warning(f"Gradient is None for parameter {name}")
                else:
                    if not torch.isfinite(param.grad).all():
                        nan_inf_count += 1
                        nan_inf_param_list.append(name)
            from hy_parallelism.utils import format_keys
            if non_count > 0:
                loguru.logger.warning(f"{non_count}/{param_count} ({non_count/param_count*100:.2f}%) parameters have None gradient")
                loguru.logger.warning(f"Non-gradient parameters: {format_keys(non_param_list, prefix='Non-gradient parameters')}")
                loguru.logger.debug(f"Has gradient: {format_keys(list(set(param_list) - set(non_param_list)), prefix='Has gradient')}")
            if nan_inf_count > 0:
                loguru.logger.critical(f"{nan_inf_count}/{param_count} ({nan_inf_count/param_count*100:.2f}%) parameters have NaN or Inf gradient")
                loguru.logger.critical(f"NaN or Inf parameters: {format_keys(nan_inf_param_list)}")



    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        return (self.micro_steps + 1) % self.gradient_accumulation_steps == 0

    def _init_missing_grad(self):
        """
        保存checkpoint的时候的 get_optimizer_state_dict 会判断当前模型有没有step过，如果step过就直接保存（那些没梯度的会被漏掉）
        否则，会初始化 optimizer state（为所有参数初始化）。
        resume的时候同理，get_optimizer_state_dict 会给所有参数optimizer state初始化

        但训练场景可能出现某些参数 backward 不会产生梯度，但被加入 optimizer 内，导致保存下来的 optimizer state 不完整。
        而 resume 时初始化的完整 state 无法加载不完整的 ckpt，从而报错 (missing xxx.step 之类)。

        解决办法：保存之前，给所有 optimizer state 做一个检查，如果发现有参数的 grad 为 None，则初始化一个零梯度。
        """
        import warnings
        param_cnt = 0
        missing_grad_cnt = 0

        no_grad_fqns = []
        for name, param in self.named_parameters():
            if param.requires_grad and getattr(param, 'grad', None) is None:
                no_grad_fqns.append(name)

        if len(no_grad_fqns) > 0:
            from hy_parallelism.utils import format_keys
            if dist.get_rank() == 0:
                loguru.logger.warning(f"No gradient parameters: {format_keys(no_grad_fqns, prefix='No gradient parameters')}")

        for param in self.optimizer_container.all_params:
            param_cnt += 1
            if param.requires_grad and getattr(param, 'grad', None) is None:
                missing_grad_cnt += 1
                try:
                    param_name = self._get_param_name_from_param(param)
                    warnings.warn(f"Gradient is None for parameter '{param_name}'; this parameter has requires_grad=True but no gradient was produced. Consider setting requires_grad=False if gradients are not needed.")
                except Exception:
                    warnings.warn("Some parameters with requires_grad=True did not produce a gradient. Consider setting requires_grad=False for this parameter if gradients are not needed.")
                param.grad = torch.zeros_like(param)

        if missing_grad_cnt > 0:
            loguru.logger.warning(f"{missing_grad_cnt}/{param_cnt} ({missing_grad_cnt/param_cnt*100:.2f}%) parameters have None gradient")

    def step(self, lr_kwargs=None, zero_grad_after_step=True):
        r"""Performs a single optimization step and updates the micro-step counter.

        This method must be called after every backward pass to properly update the
        micro-step count, even when gradient accumulation is enabled. At gradient
        accumulation boundaries (determined by :meth:`is_gradient_accumulation_boundary`),
        it performs the actual optimizer step, learning rate scheduler step, and optionally
        zeros gradients. The micro-step counter is incremented on every call.

        Args:
            lr_kwargs (dict, optional): Additional keyword arguments to pass to the
                learning rate scheduler. Default: ``None``
            zero_grad_after_step (bool, optional): If ``True``, zeros gradients after
                performing the optimizer step at gradient accumulation boundaries.
                Default: ``True``

        .. note::
            This method must be called after every backward pass, regardless of whether
            gradient accumulation is enabled. The micro-step counter is always incremented,
            but the optimizer and learning rate scheduler are only stepped at gradient
            accumulation boundaries.

        .. warning::
            Failing to call this method after backward passes will result in incorrect
            micro-step counting, which can break gradient accumulation logic and pipeline
            parallelism scheduling.
        """
        if self.is_gradient_accumulation_boundary():
            if "step_cnt" not in self.training_states:
                self.training_states["step_cnt"] = 0

            # 只需要第一次做即可，保证 optimizer state 正确创建就足够了
            if self.training_states["step_cnt"] == 0:
                self._init_missing_grad()

            if self.optimizer_offloading:
                self.cast_optimizer('cuda')
            if self.enable_grad_scaler:
                self.grad_scaler.step(self.optimizer_container)
                self.grad_scaler.update()
            else:
                self.optimizer_container.step()
            if self.lr_scheduler_container is not None:
                self.lr_scheduler_container.step()

            self.training_states["step_cnt"] += 1

            if zero_grad_after_step:
                self.zero_grad()
            if self.optimizer_offloading:
                self.cast_optimizer('cpu')
        self.micro_steps += 1
        self.clear_cached_result()

    def get_last_lr(self):
        if not hasattr(self, "lr_scheduler_container"):
            raise RuntimeError("Should parse optimizer_config to pp_engine initialization")
        return self.lr_scheduler_container.schedulers[0].get_last_lr()

    def _tag_is_expert_info_to_param_and_grad(self):
        # Only useful for old EP grad clipping or grad norm computation.
        for name, param in self.named_parameters():
            is_expert = self.is_expert(name)
            param.is_expert = is_expert
            if param.grad is not None:
                param.grad.is_expert = is_expert

    @staticmethod
    def _tag_param_name_to_params(module):
        for name, param in module.named_parameters():
            param._param_name = name

    @staticmethod
    def _get_param_name_from_param(param):
        if hasattr(param, "_param_name"):
            return param._param_name
        else:
            raise ValueError(f"Parameter {param} does not have a _param_name attribute. Should be tagged by `_tag_param_name_to_params` first.")

    def clip_grad_norm_(
        self,
        parameters,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach=None,
    ):
        # 之前系只有啓用 ep 或 pp 先會有非標準 DTensor 嘅 model 切分 需要特別實現以正確計算 norm
        # 但係其實仲有一種可能，就係參數系唔同嘅 group, 俾 optimizer 按照 mesh 分組，需要特別實現以正確計算 norm
        # (即系調用 clip by mesh)
        # Otherwise, the original implementation could lead to cross mesh computation.
        if self.enable_pp or self.enable_ep:
        # if True:
            # return clip_grad_norm_(
            #     parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh
            # )
            from hy_parallelism.utils import clip_grad_norm_by_mesh_old_ep_
            from hy_parallelism.utils import clip_grad_norm_by_mesh_

            if self.enable_ep and self.moe_impl == "old":

                self._tag_is_expert_info_to_param_and_grad()
                return clip_grad_norm_by_mesh_old_ep_(
                    self.optimizer_container.param_groups_by_mesh, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh,
                )
            return clip_grad_norm_by_mesh_(
                self.optimizer_container.param_groups_by_mesh, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh
            )
        else:
            grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            return grad_norm

    def set_micro_batch_size(self, bs):
        self.batch_size = bs

        if self.__high_prio_micro_batch_size is not None:
            bs_list = gather_obj(bs)
            assert all([bs_list[0] == k for k in bs_list]), f"Batch size must be the same in all ranks when `micro_batch_size` is set. {bs_list}"

        assert self.batch_size % self.m_microbatch == 0, f"Invalid batch size {self.batch_size} or invalid m:{self.m_microbatch}"
        self.microbatch_size = self.batch_size // self.m_microbatch

        if self.training:
            if "1F1B" in self.pipeline_parallel_schedule:
                assert self.m_microbatch >= self.pp_size

        assert self.microbatch_size > 0
        assert self.is_valid_m(self.m_microbatch, bs), (
            f"Invalid m_microbatch: {self.m_microbatch} for input batch size {bs} and scheduler {self.training_pipeline_parallel_schedule}"
        )

    def set_resolution(self, reso, bs):
        if reso is None:
            return

        self.set_micro_batch_size(bs)

        key = (self.training, reso, bs, self.microbatch_size, self.pipeline_parallel_schedule)

        should_refresh = False
        if key not in self.reso_stage_manager:
            should_refresh = True

        # pp init stage unshard vs next batch unshard
        # ep all2all/allgather vs init stage unshard
        buffer = [None] * dist.get_world_size()
        dist.all_gather_object(buffer, should_refresh)
        should_refresh = any(buffer)

        # schedule._stage_initialized is None: -> schedule._initialize_stage() -> stage._prepare_forward_infra()
        # ->  inputs_meta is None? -> stage._shape_inference()
        # -> Create communication buffer
        def clear_stages_schedule(stages, schedule):
            for stage in stages:
                stage.clear_runtime_states()  # Basic clear
                stage.args_recv_info.clear()  # Clear communication buffer
                stage.grad_recv_info.clear()  # Clear communication buffer

                """
                Keep inputs_meta to avoid running shape inference again
                inputs_meta do not consume memory
                """
                # stage.inputs_meta = None
                # stage.outputs_meta = None
                # stage._outputs_meta = None

            # Make schedule uninitialized, so that communication buffer will be created in the next step
            if hasattr(schedule, "_stages_initialized"):
                schedule._stages_initialized = False
            if hasattr(schedule, "_stage_initialized"):
                schedule._stage_initialized = False

        if should_refresh:
            # Update batch_size attribute of Stage object (according to the batchsize set by `set_micro_batch_size` above)
            # Create uninitialized stages and schedule, which will perform shape inference later.
            self.force_refresh_pipeline_scheduler()
            self.reso_stage_manager[key] = (self.pp_stages, self.pp_schedule)
        else:
            self.pp_stages, self.pp_schedule = self.reso_stage_manager[key]

        for i, (k, v) in enumerate(self.reso_stage_manager.items()):
            stages, schedule = v
            if stages is self.pp_stages:
                assert schedule is self.pp_schedule
                continue
            # Avoid GPU OOM. The cached stages could take up too much GPU memory.
            clear_stages_schedule(stages, schedule)

        if DEBUG_MODE:
            # Ensure all ranks in the pp group have the same pp_schedule.
            from hy_parallelism import bing_utils

            all_pp_schedule = bing_utils.gather_obj(type(self.pp_schedule))
            for k in all_pp_schedule:
                if k != all_pp_schedule[0]:
                    loguru.logger.critical("Please report this bug to kevinkhwu")
                    if torch.distributed.get_rank() == 0:
                        import pdb

                        pdb.set_trace()
                    torch.distributed.barrier()
                    raise RuntimeError("Please report this bug to kevinkhwu")

    def set_pp_schedule(self, pipeline_parallel_schedule):
        self.pipeline_parallel_schedule = pipeline_parallel_schedule
        schedule_class = get_schedule_class(self.pipeline_parallel_schedule)
        # self.pp_style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"
        self.pp_style = "v" if "Bubble" in self.pipeline_parallel_schedule else "loop"

    @staticmethod
    def has_meta(module):
        for param in module.parameters():
            if param.device == torch.device("meta"):
                return True
        for buf in module.buffers():
            if buf.device == torch.device("meta"):
                return True
        return False

    def has_meta_param(self):
        for i, model in enumerate(self.pp_models):
            if self.has_meta(model):
                return True
        return False


    def _maybe_init_meta_param(self, models, call_init_fn=True):
        r"""Initializes model parameters if any parameters are on the 'meta' device, and moves models to the appropriate device.

        If the model does not have any parameters on the 'meta' device and `self.cpu_offload` is False, the model is moved to CUDA.

        Args:
            models (list[nn.Module]): List of models to process. Can be `fsdp_models` or `pp_models` depending on context.
            call_init_fn (bool, optional): Whether to call the user-provided parameter initialization function
                (`self.param_init_fn`) on each model after materialization. Default is True.

        Note:
            - This function isolates the random number generator state to avoid inconsistencies across pipeline parallel ranks.
            - The function assumes that either `self.param_init_fn` is implemented, or a full state dict or checkpoint path is provided.
        """
        if not self.initialize_meta_param:
            return
        # Different pp ranks may have different number of parameters,
        # To avoid inconsistent random states after initialization, we isolate the rng.
        with isolate_rng():
            if hasattr(self, "fsdp_models"):
                assert models is self.fsdp_models or models is self.pp_models
            else:
                assert models is self.pp_models

            self.reset_parameters(models, call_init_fn=call_init_fn)
            for model in models:
                assert not self.has_meta(model), f"Model {model} has meta parameters after initialization."


    def reset_parameters(self, models, call_init_fn=True):

        def mark_as_initialized(model):
            for param in model.parameters():
                param._hy_parallelism_initialized = True

        def get_initialization_status(model):
            initialized_params = []
            uninitialized_params = []
            for name, param in model.named_parameters():
                if not hasattr(param, "_hy_parallelism_initialized") or not param._hy_parallelism_initialized:
                    uninitialized_params.append(name)
                else:
                    initialized_params.append(name)
            return initialized_params, uninitialized_params

        with isolate_rng():
            for i, model in enumerate(models):
                if self.has_meta(model):
                    # TODO: Implement some check to ensure all parameters are properly initialized by either `param_init_fn` or `reset_parameters`.
                    # Warning: If `param_init_fn` or `reset_parameters` do not cover every parameter, some may remain uninitialized after using `to_empty`.
                    full_init = False
                    if full_init:
                        models[i] = model.to_empty(device="cuda" if not self.cpu_offload else "cpu")
                    else:
                        for module in self.recursive_module_generator_buttom_up(models[i]):
                            if self.has_meta(module):
                                module.to_empty(device="cuda" if not self.cpu_offload else "cpu")

                    if self.call_module_reset_parameters:
                        for module_name, module in self.recursive_module_generator_buttom_up(models[i], return_name=True):
                            if hasattr(module, "reset_parameters"):
                                # initialized_params, uninitialized_params = get_initialization_status(module)
                                # if len(initialized_params) > 0:
                                #     loguru.logger.warning(
                                #         f"Detected already-initialized parameters during reset_parameters: {module_name}:{initialized_params}. "
                                #         "This indicates that this module is initialized by its parent module via `reset_parameters`. "
                                #         # "This is likely due to a recursive call to `reset_parameters`.",
                                #     )
                                module.reset_parameters()
                                mark_as_initialized(module)
                            else:
                                assert not self.has_meta(module), f"Module {module_name} does not implement `reset_parameters` and has meta parameters. This is not allowed."
                                mark_as_initialized(module)

                        if not (is_implemented(self.param_init_fn) or self.full_sd is not None or self.load_ckpt_path is not None):
                            msg = "No parameter initialization function provided. Please ensure that this is intended."
                            loguru.logger.warning(msg)

                        initialized_params, uninitialized_params = get_initialization_status(models[i])
                        assert len(uninitialized_params) == 0, f"Some parameters are not properly initialized (by either `param_init_fn` or `reset_parameters`): {uninitialized_params}"

                if call_init_fn and is_implemented(self.param_init_fn):  # and self.full_sd is None and self.load_ckpt_path is None:
                    with torch.no_grad():
                        self.param_init_fn(models[i])

                assert not self.has_meta(models[i])

    @property
    def moe_impl(self):
        """Can be overridden by the user to set the MOE implementation."""
        return 'old'

    def post_init(self):
        if self.full_sd:
            raise DeprecationWarning("`full_sd` is deprecated. Please use `engine.load_checkpoint`/`engine.model_checkpoint_manager.load_full_sd` instead.")
            assert self.init_meta_stage == "pre_fsdp", "pre_fsdp is not supported yet"
            self._maybe_init_meta_param(self.pp_models)
            # if self.has_meta(self.model):
            #     self.model.to_empty(device="cpu")
            self.model.load_state_dict(self.full_sd, strict=True)

        for model in self.pp_models:
            for sub_module in model.modules():
                sub_module.is_expert = self.is_expert
                sub_module.is_moe = self.is_moe
                sub_module.is_moe_router = self.is_moe_router

            model.dummy_tensor = torch.tensor(0.6, device="cuda", requires_grad=True)
            model.dummy_tensor.grad = torch.zeros_like(model.dummy_tensor)

        # If FSDP is enabled, we must replace the fp8Linear before applying FSDP
        if self.enable_fsdp and is_implemented(self.apply_quantization) and self.enable_quantization:
            for model in self.pp_models:
                self.apply_quantization(model)

        if dist.get_rank() == 0:
            for m in self.pp_models:
                print_model_info(m, tag="Before parallelism")
        with isolate_rng():
            self.apply_parallelism()  # already called _maybe_init_meta_param inside

        # HACK(torch2.10):
        # fsdp 嘅 lazy init 會創建 _orig_dtype, 但係繫裹所有參數都唔 requires_grad
        # 則會跳過。導致系 grad reduce 時期，無法正確將 grad cast 到正確嘅 orig_dtype。
        # 導致 grad_dtype error (因為 PyTorch 2.10 新增對 grad_dtype 嘅檢查)
        #
        # Eager FSDP lazy init (no forward) so _orig_dtype is snapshotted while
        # requires_grad=True. Avoids PyTorch 2.10 grad_dtype error when params
        # are frozen before the first forward.
        if self.enable_fsdp:
            from hy_parallelism.distributed.fsdp_util import ensure_requires_grad_and_eager_fsdp_lazy_init
            ensure_requires_grad_and_eager_fsdp_lazy_init(self.fsdp_models)

        if dist.get_rank() == 0:
            for m in self.pp_models:
                print_model_info(m, tag="After parallelism")

        self.post_setup_global_states(get_global_states())

        self.build_optimize_scheduler()
        self.create_checkpoint_manager(self.ckpt_dir, self.keep_latest_k)

        if self.load_ckpt_path:
            self.load_checkpoint(self.load_ckpt_path)

        # If FSDP is not enabled, we should apply quantization after applying parallelism and after loading checkpoint.
        if not self.enable_fsdp and is_implemented(self.apply_quantization) and self.enable_quantization:
            assert self.load_ckpt_path is not None, "Quantization should be applied after loading checkpoint, `load_ckpt_path` must be set."
            for model in self.fsdp_models:
                self.apply_quantization(model)

        for fsdp_part in self.fsdp_models:

            for name, param in itertools.chain(fsdp_part.named_parameters(), fsdp_part.named_buffers()):
                if param.device == torch.device("meta"):
                    if self.initialize_meta_param:
                        msg = f"Parameter '{name}' is still on the 'meta' device after post-initialization. " \
                            f"This likely means its weights were not properly initialized. " \
                            f"Check your model's parameter initialization (`param_init_fn`) logic."
                        loguru.logger.critical(msg)
                        raise RuntimeError(msg)
                    else:
                        msg = f"Parameter '{name}' is still on the 'meta' device after post-initialization. " \
                            f"This likely means its weights were not properly initialized. " \
                            f"Try setting initialize_meta_param=True to initialize the parameters."
                        loguru.logger.warning(msg)
                        raise RuntimeError(msg)


        # Wait for all processes to finish the initialization of the model parameters.
        # Ensure that the following timing statistics are accurate.
        torch.cuda.synchronize()
        dist.barrier()

        if self.enable_fsdp and self.parallel_dims.dp_replicate_enabled \
            and self.dp_replicate_param_handler not in ['none', 'noop'] \
            and self.load_ckpt_path is None: # loading checkpoint after fsdp, does not requires syncing
            self.rank0_log("Synchronizing model parameters across all ranks...")
            for model in self.fsdp_models:
                with torch.no_grad():
                    src = int(self.parallel_dims.dp_replicate_mesh.mesh[0])
                    group = self.parallel_dims.dp_replicate_mesh.get_group()
                    group_ranks = dist.get_process_group_ranks(group)
                    assert src in group_ranks

                    async_broadcast = True # Experimental feature, can be removed in the future

                    for name, param in model.named_parameters():
                        if isinstance(param, DTensor):
                            param_mesh = param.device_mesh

                            if self.dp_replicate_param_handler == 'check':
                                DTensor.from_local(
                                    param.to_local(),
                                    device_mesh=param.device_mesh,
                                    placements=param.placements,
                                    shape=param.shape,
                                    stride=param._spec.stride,
                                    run_check=True,
                                )
                            else:
                                # TODO：似乎所有情況都系用 mesh_dim=0 作為 dp_relicate 嘅 group
                                #     可以考慮精簡代碼
                                if (self.parallel_dims.ep_enabled) and model.is_expert(name):
                                    if param_mesh.ndim == 4:
                                        # [(ep_fsdp_replicate=2, ep_fsdp_shard=1, ep=2, etp=2)]
                                        # broadcast_group = param_mesh[param_mesh.mesh_dim_names[0]].get_group()
                                        broadcast_group = param_mesh.get_group(mesh_dim=0)
                                    elif param_mesh.ndim == 3:
                                        # (ep_fsdp_replicate, ep_fsdp_shard, ep)
                                        # (ep_fsdp_replicate, ep_fsdp_shard, tp)
                                        # TODO: (ep_fsdp_shard, ep, etp) Inpossible, since we assume dp_replicate is enabled.
                                        #       If `enable_expert_fsdp_sharding` is supported in a future version,
                                        #       (ep_fsdp_shard, ep, etp) could be possible and the code below need to be modified.
                                        broadcast_group = param_mesh.get_group(mesh_dim=0)
                                    else:
                                        assert param_mesh.ndim == 2, f'Expert param {name} is not on a proper mesh. Got ({param_mesh}). Check `is_expert`, `apply_fsdp` or `apply_ep`.'
                                        # Could be old MOE implementation with dp_replicate or new MOE implementation without dp_replicate
                                        if self.parallel_dims.expert_fsdp_mesh.ndim < param_mesh.ndim:
                                            # 新 MOE 實現，param 嘅 mesh 自帶 ep 維度
                                            continue
                                        else:
                                            # TODO: Old MOE implementation (ptm / hunyuan torch moe v1)
                                            # raise NotImplementedError
                                            broadcast_group = param_mesh.get_group(mesh_dim=0)
                                else:
                                    if param_mesh.ndim == 2:
                                        # could be [dp_shard, tp] or [dp_replicate, tp]
                                        # but we assume dp_replicate is enabled above, so this must be [dp_replicate, dp_shard]
                                        # We can assert this param is not Linear that wrap with TP hooks
                                        assert not self.parallel_dims.tp_enabled
                                        # FIXME(kevinkhwu): PyTorch 2.6 will fail to slice submesh on submesh.
                                        #     We can try parsing dim or name to `get_group` to fix this, but requiring testing.
                                        broadcast_group = param_mesh.get_group(mesh_dim=0)
                                    elif param_mesh.ndim == 3:
                                        # Could be dp_replicatee, dp_shard, tp
                                        assert self.parallel_dims.tp_enabled
                                        broadcast_group = param_mesh.get_group(mesh_dim=0)
                                    else:
                                        raise RuntimeError(f'DP replicate is enabled, but param {name} is not on a proper mesh. Got ({param_mesh})')

                                if self.dp_replicate_param_handler == 'safe_sync':
                                    # with check, raise error if broadcast is incorrect, this is very slow

                                    broadcast_param = auto_broadcast(
                                        param._local_tensor, group=broadcast_group, group_src=0, async_op=async_broadcast
                                    )
                                    param._local_tensor.data = broadcast_param.data

                                    # check if the broadcast is correct
                                    DTensor.from_local(
                                        param.to_local(),
                                        device_mesh=param.device_mesh,
                                        placements=param.placements,
                                        shape=param.shape,
                                        stride=param._spec.stride,
                                        run_check=True,
                                    )
                                elif self.dp_replicate_param_handler == 'sync':
                                    # without check, 10x faster
                                    dist.broadcast(
                                        param._local_tensor.data,
                                        group_src=0,
                                        group=broadcast_group,
                                        async_op=async_broadcast,
                                    )
                                else:
                                    raise ValueError(f"Invalid dp_replicate_param_handler: {self.dp_replicate_param_handler}")
                        else:
                            raise RuntimeError(f"HSDP requires all parameters to be DTensor")

                    if async_broadcast:
                        torch.cuda.synchronize()

                self.rank0_log("Model parameters synchronized.")

    def build_optimize_scheduler(self):
        if self.optimizer_config:
            self.optimizer_container = OptimizersContainer(self.fsdp_models, **self.optimizer_config)
        if self.get_lr_scheduler_func is not None:
            # loguru.logger.debug(f"Build lr scheduler container")
            self.lr_scheduler_container = LRSchedulersContainer(self.optimizer_container, self.get_lr_scheduler_func)
        else:
            # loguru.logger.critical(f"No lr scheduler function provided")
            self.lr_scheduler_container = None

    def _check_pp_no_duplicate(self):
        """
        Ensure that there are no duplicate parameters across different pipeline (pp) stages.
        If duplicate parameters exist, it indicates that the pipeline partitioning (pp) was not applied correctly.
        """
        if self.enable_pp:
            keys = sum([list(model.state_dict().keys()) for model in self.pp_models], [])
            buffer = [None] * self.parallel_dims.pp
            dist.all_gather_object(buffer, keys, self.parallel_dims.pp_group)
            # check no duplicate key in buffer
            all_keys = sum(buffer, [])
            from collections import Counter

            duplicate_keys = []
            for k, v in Counter(all_keys).items():
                if v > 1:
                    duplicate_keys.append(k)
            if duplicate_keys:
                raise RuntimeError(f"Duplicate keys found: {duplicate_keys}. Check PP.")

    def apply_parallelism(self):
        if self.enable_etp:
            assert is_implemented(self.apply_etp), "Enabling expert tensor parallelism requires `apply_etp` to be implemented in your engine."
            for m in self.pp_models:
                self.apply_etp(m)
        else:
            if self.enable_tp:
                if is_implemented(self.apply_tp):
                    self.make_tp(self.apply_tp)

            if self.enable_ep:
                if is_implemented(self.apply_ep):
                    for m in self.pp_models:
                        self.apply_ep(m)

        if self.enable_sp:
            if is_implemented(self.apply_sp):
                self.make_sp(self.apply_sp)

        # Originally, this was intended to be placed before `apply_ep` to ensure that the model contains unsharded parameters.
        # By doing this, nn.init can easily initialize each expert to different values.
        # However, for the new implementation and ptm implementation partitioned experts during initialization, it's better to place this after `apply_ep`.
        # This is placed before FSDP because DTensor does not support indexing well, which makes index-dependent initialization logic difficult.
        if self.init_meta_stage == "pre_fsdp":
            self._maybe_init_meta_param(self.pp_models, call_init_fn=self.full_sd is None)

        # HACK(kevinkhwu): PyTorch 2.6 bug
        # FSDP[AC[FSDP]] -> Potential Issue: The innermost FSDP may skip pre_forward if it has already entered the PRE_BACKWARD state,
        # resulting in parameters being DTensor instead of Tensor.
        # Explanation:
        #   In _fsdp_state.py, pre_forward will immediately return if the state is PRE_BACKWARD.
        #   After the first recomputation call completes, the innermost FSDP will be in the PRE_BACKWARD state.
        #   However, other backward threads (with different gids) may still attempt to recompute activations (with pre_forward skipped).
        #   Skipping pre_forward leads to parameters remaining as DTensor rather than Tensor.
        #   If the innermost FSDP computation does not involve custom kernels, DTensor is generally safe.
        #   Otherwise, using DTensor may cause unexpected bugs.
        # AC[FSDP] -> skipped dtype casting! https://github.com/pytorch/pytorch/issues/159359  (The cause is the same as above)
        # AC[FSDP[FSDP]] -> skipped dtype casting ! https://github.com/pytorch/pytorch/issues/159359  (The cause is the same as above)
        def apply_ac():
            if self.enable_gradient_checkpointing:
                assert is_implemented(self.apply_ac), "Enabling gradient checkpointing requires `apply_ac` to be implemented in your engine."
                for model in self.pp_models:
                    self.apply_ac(model)

        apply_ac()

        # after ac and before fsdp
        if self.enable_compile:
            if is_implemented(self.apply_compile):
                # if dist.get_rank() == 0:
                #     torch._logging.set_logs(graph_code=True, recompiles=True)
                for model in self.pp_models:
                    self.apply_compile(model)
            else:
                loguru.logger.error("`apply_compile` is not implemented but `enable_compile` is True. Please ensure that this is intended.")

        if self.enable_fsdp:
            if is_implemented(self.apply_fsdp):
                # Deprecated
                # assert is_implemented(self.fsdp_blocks), "`fsdp_blocks` must be implemented if `apply_fsdp` is implemented"
                self.make_fsdp(self.apply_fsdp)
            else:
                self.fsdp_models = self.pp_models
                raise RuntimeError("`apply_fsdp` is not implemented but `enable_fsdp` is True. Please ensure that this is intended.")
        else:
            if self.enable_ddp:
                raise NotImplementedError("DDP is not implemented yet")  # accessing model attributes from DDP is different from FSDP
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.fsdp_models = [DDP(m) for m in self.pp_models]
            else:
                self.fsdp_models = self.pp_models
                loguru.logger.warning("Skipping FSDP on experts because `enable_fsdp` is False. Please ensure that this is intended.")

            self.eval()  # If fsdp is not enabled, we need to eval the model to ensure that the model is in eval mode

        if self.init_meta_stage == "post_fsdp":
            self._maybe_init_meta_param(self.pp_models, call_init_fn=self.full_sd is None)

        if not self.gradient_sync_during_accumulation and self.gradient_accumulation_steps > 1 and self.enable_fsdp:
            for module in self.fsdp_models:
                module.set_requires_gradient_sync(False)

    def make_tp(self, apply_tp):
        for m in self.pp_models:
            apply_tp(m, self.enable_tpsp)

    def make_sp(self, apply_sp):
        for m in self.pp_models:
            apply_sp(m, self.sp_mesh)

    def make_fsdp(self, apply_fsdp):
        self.fsdp_models = []
        for i, m in enumerate(self.pp_models):
            fsdp_model = apply_fsdp(m)

            assert fsdp_model is not None, "apply_fsdp should return an fsdp model"
            if self.enable_pp:
                self.pp_stages[i].submod = fsdp_model
            self.fsdp_models.append(fsdp_model)
            self._tag_param_name_to_params(fsdp_model)


    @staticmethod
    def recursive_get_attr(model, attr):
        splits = attr.split(".")
        for split in splits:
            if split:
                model = getattr(model, split)
        return model

    def stage_ids_this_rank(self, num_stages: int) -> tuple[int]:
        """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
        pp_rank, pp_size = self.pp_rank, self.pp_size
        style = self.pp_style
        assert num_stages % pp_size == 0, f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
        stages_per_rank = num_stages // pp_size
        if style == "loop":
            return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
        elif style == "v":
            assert stages_per_rank == 2, f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1)))
            return stage_v_pairs[pp_rank]

    def is_multi_schedule(self, schedule):
        return issubclass(get_schedule_class(schedule), PipelineScheduleMulti)

    @staticmethod
    def default_loss_fn(output, target):
        ret = output.mean()[None]
        return ret

    def build_pp_schedule(self):
        # assert self.pipeline_parallel_schedule in [
        #     'Interleaved1F1B', 'GPipe', '1F1B', 'ZBVZeroBubble',
        #     # 'PipelineScheduleSingle', 'PipelineScheduleMulti',
        # ], 'Other schedulers are not tested'

        if not self.training:
            assert self.pipeline_parallel_schedule == self.get_inference_pipeline_schedule()

        stages = self.pp_stages

        from torch.distributed.pipelining.schedules import PipelineScheduleMulti

        schedule_class = get_schedule_class(self.pipeline_parallel_schedule)
        looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)

        if self.batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by number of microbatches {self.microbatch_size}. "
                "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
            )

        n_microbatches = self.batch_size // self.microbatch_size
        num_total_stages = len(stages) * self.pp_size
        if n_microbatches < num_total_stages:
            log_once(
                f"Number of microbatches ({n_microbatches}) is less than the total number "
                f"of stages ({num_total_stages}) which may result in a bubble in the pipeline. "
                f"(Theoretical throughput will be n-times lower than without pp (if micro bs=1 can already fully utilize the GPU), where n = 1 + (p - 1) / m = 1 + ({self.pp_size} - 1) / {self.m_microbatch} = {1 + (self.pp_size - 1) / self.m_microbatch})",
                level="WARNING",
            )
        if not looped_schedule:
            assert len(stages) == 1, f"{len(stages)=} {schedule_class}"

        log_once(f"Building pp schedule with {schedule_class} and {n_microbatches=}")
        schedule = schedule_class(
            stages if looped_schedule else stages[0],
            n_microbatches=n_microbatches,
            loss_fn=self.loss_fn if self.training else None,
        )
        if not self.training:
            for stage in self.pp_stages:
                assert not stage.has_backward

        has_first_stage = False
        has_last_stage = False
        for stage in stages:
            if stage.is_first:
                has_first_stage = True
            if stage.is_last:
                has_last_stage = True

        schedule.has_first_stage = has_first_stage
        schedule.has_last_stage = has_last_stage

        self.pp_schedule = schedule

        self._check_scheduler()

        return schedule

    def parameters(self):
        return iter(sum([list(model.parameters()) for model in self.fsdp_models], []))

    def named_parameters(self):
        return iter(sum([list(model.named_parameters()) for model in self.fsdp_models], []))

    def named_buffers(self):
        return iter(sum([list(model.named_buffers()) for model in self.fsdp_models], []))

    def buffers(self):
        return iter(sum([list(model.buffers()) for model in self.fsdp_models], []))

    def _get_fsdp_blocks(self):
        if self.enable_fsdp:
            return self.fsdp_blocks()
        else:
            return []

    def recursive_module_generator_buttom_up(self, model, return_name=False):
        """
        Recursively generate modules in bottom-up order (children before parent).

        Args:
            model: The root module to traverse

        Yields:
            tuple: (name, module) pairs where name is the full path from root
        """
        def _traverse(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                yield from _traverse(child, full_name)

            if return_name:
                yield (prefix, module) if prefix else ("", module)
            else:
                yield module

        yield from _traverse(model)

    def reshard(self):
        if not self.enable_fsdp:
            return
        for model in self.pp_models:
            for name, module in self.recursive_module_generator_buttom_up(model, return_name=True):
                if hasattr(module, "reshard"):
                    debug_log(f'Resharding {name} {type(module)=}')
                    module.reshard()

        # Mistargeted prefetch（例如 unshard 咗但無 wait / 無 forward 到）會留下 _all_gather_result，
        # 入面嘅 all_gather copy_in buffer 會常駐顯存；單純 module.reshard() 對未 copy_out 完嘅
        # prefetch 會直接 skip 變 no-op，唔會 free。呢度要靠 finalize_backward 先真正清走 copy_in。
        for module in self.module.modules():
            if isinstance(module, FSDPModule):
                module._get_fsdp_state()._fsdp_param_group.finalize_backward()


    def unshard(self):
        if not self.enable_fsdp:
            return
        for model in self.pp_models:
            for name, module in self.recursive_module_generator_buttom_up(model, return_name=True):
                if hasattr(module, "unshard"):
                    debug_log(f'Unsharding {name} {type(module)=}')
                    module.unshard()


    def cast_optimizer_new(self, device):
        with profile_range("cast_optimizer"):
            if getattr(self, "optimizer_container", None) is not None:
                for optimizer_pp in self.optimizer_container.optimizers:
                    for optimizer in optimizer_pp:
                        for param, state in optimizer.state.items():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    if torch.device(device).type == "cuda":
                                        state[k] = v.to(device, non_blocking=True)
                                    else:
                                        if self.optimizer_offloading_pin_memory:
                                            if isinstance(v, DTensor):
                                                local_v = v._local_tensor
                                                buf = self.optimizer_pinned_memory_pool.allocate(
                                                    local_v.shape, local_v.dtype
                                                )
                                                buf.copy_(local_v, non_blocking=True)
                                                state[k] = DTensor(
                                                    buf, v._spec, requires_grad=v.requires_grad
                                                )
                                            else:
                                                buf = self.optimizer_pinned_memory_pool.allocate(
                                                    v.shape, v.dtype
                                                )
                                                buf.copy_(v, non_blocking=True)
                                                state[k] = buf
                                        else:
                                            # Initializing pin memory is time-consuming, so we set non_blocking=False
                                            state[k] = v.to(device, non_blocking=False)
                torch.cuda.synchronize()
                # After on-loading, no need to keep the pinned memory pool alive
                if self.optimizer_offloading_pin_memory and torch.device(device).type == "cuda":
                    self.optimizer_pinned_memory_pool.reset()


    def cast_optimizer_old(self, device):
        with profile_range("cast_optimizer"):
            if getattr(self, "optimizer_container", None) is not None:
                for optimizer_pp in self.optimizer_container.optimizers:
                    for optimizer in optimizer_pp:
                        for param, state in optimizer.state.items():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    if torch.device(device).type == "cuda":
                                        state[k] = v.to(device, non_blocking=True)
                                    else:
                                        # Initializing pin memory is time-consuming, so we set non_blocking=False
                                        state[k] = v.to(device, non_blocking=False)
                torch.cuda.synchronize()

    def cast_optimizer(self, device):
        return self.cast_optimizer_new(device)


    @contextmanager
    def attr_safe_context_for_module_apply(self):
        """
        .cuda() 这类操作会调用 nn.Module._apply
        然后调用 torch.utils.swap_tensors
        导致 param 的 attr 被重置
        """
        from collections import defaultdict
        attr_map = defaultdict(dict)
        for m in self.fsdp_models:
            for name, param in m.named_parameters():
                for attr in self.guard_attr_names:
                    if hasattr(param, attr):
                        attr_map[id(param)][attr] = getattr(param, attr)
        yield attr_map
        for m in self.fsdp_models:
            for name, param in m.named_parameters():
                for attr in self.guard_attr_names:
                    if id(param) in attr_map and attr in attr_map[id(param)]:
                        setattr(param, attr, attr_map[id(param)][attr])

    def cpu(self, cast_optimizer=False, pin_memory=False):
        # all_gather copy_in buffer 唔會自動 offload, 繫裹 unshard 嗰時對一啲唔會 Forward 嘅參數 unshard，噉 copy_in buffer 會常駐顯存
        # 因為唔會 forward 代表唔會進行 post_forward, 代表唔會 reshard, 代表唔會 free_unsharded_buffer
        # [Mistargeted prefetch]
        # 要 call finalize_backward 先真正清走 copy_in
        self.reshard()

        with self.attr_safe_context_for_module_apply():
            if pin_memory:
                if self.model_pinned_memory_pool is None:
                    self.model_pinned_memory_pool = PinnedMemoryPool()
                pool = self.model_pinned_memory_pool

                def _to_pinned_cpu(t):
                    if isinstance(t, DTensor):
                        local = cast_to_device(
                            t._local_tensor,
                            "cpu",
                            use_side_stream_for_tensor_copies=True,
                            pin_memory=True,
                            pool=pool,
                        )
                        return DTensor(local, t._spec, requires_grad=t.requires_grad)
                    return cast_to_device(
                        t,
                        "cpu",
                        use_side_stream_for_tensor_copies=True,
                        pin_memory=True,
                        pool=pool,
                    )

                for m in self.fsdp_models:
                    m._apply(_to_pinned_cpu)
                torch.cuda.synchronize()
            else:
                for m in self.fsdp_models:
                    m.cpu()
        if cast_optimizer:
            self.cast_optimizer("cpu")
        return self


    def cuda(self, cast_optimizer=False):
        with self.attr_safe_context_for_module_apply():
            self.reshard()
            for m in self.fsdp_models:
                m.cuda()

        if cast_optimizer:
            self.cast_optimizer("cuda")

        if self.model_pinned_memory_pool is not None:
            self.model_pinned_memory_pool.reset()
        return self

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def __getattr__(self, name):
        guards = ["fsdp_models"]
        if name in guards:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        if hasattr(self, "fsdp_models") and self.fsdp_models:
            try:
                ret = getattr(self.fsdp_models[-1], name)
                debug_log(f"Implicitly geting attribute {name} from fsdp model")
                return ret
            except AttributeError:
                pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def train(self, pipeline_parallel_schedule=None, force=False, mode=True):
        # self.microbatch_size = self.train_microbatch_size
        if not mode:
            return self.eval()

        if not self.enable_fsdp:
            if dist.get_world_size() > 1:
                msg = "FSDP is not enabled. Enabling training can lead to incorrect gradient synchronization."
                loguru.logger.warning(msg)
                raise ValueError(msg)

        for m in self.fsdp_models:
            m.train()

        if self.training and not force:
            return self

        self.training = True

        if self.enable_pp:
            self.set_pp_schedule(self.training_pipeline_parallel_schedule)
            self.refresh_stages()  # The input sizes may be different between train and eval
            self.build_pp_schedule()
        return self

    def eval(self, force=False):
        for m in self.fsdp_models:
            m.eval()
        if self.training or force:
            self.training = False
            if self.enable_pp:
                self.set_pp_schedule(self.get_inference_pipeline_schedule())
                self.refresh_stages()  # The input sizes may be different between train and eval
                self.build_pp_schedule()
        return self

    def get_inference_pipeline_schedule(self):
        if issubclass(get_schedule_class(self.training_pipeline_parallel_schedule), PipelineScheduleSingle):
            return "GPipe"
        else:
            # return self.pipeline_parallel_schedule # not to change
            return "InferencePipelineScheduleMulti"
            # return 'PipelineScheduleMulti'

    def _check_states(self):
        for m in self.fsdp_models:
            assert m.training == self.training
        if self.enable_ep and self.enable_pp:
            ms = gather_obj(self.m_microbatch, group=self.parallel_dims.ep_mesh.get_group())
            assert all([ms[0] == k for k in ms]), f"m_microbatch must be the same in all ep ranks when ep is enabled. (m_microbatches: {ms})"

    def config_forward_args(self):
        self.set_n_pp_args(0)
        self.set_replacable_kwargs([])

    def set_batch_object_kwargs_keys(self, keys, max_len=1000):
        # 如果 dict, list, tuple 一定要作为 static_kwargs 传进pp来，并且不切分batch
        # 那就要包一层 SafeObjectWrapper, 这里的 keys 为需要传进来的 collection 的 key
        # 其实也可以通过设置 collection_to_object=True 来实现
        self.batch_object_kwargs_keys = keys
        self.batch_object_kwargs_max_len = max_len

    def set_n_pp_args(self, n):
        """
        There are two modes for pipeline parallelism (PP) implementation:

        1. Replace mode:
            - The output of a block can directly replace the first few inputs of the next forward call.
            - In this mode, n must be set to 0.
            - ~~All arguments except the replaceable ones should be placed in static_kwargs.~~ (Deprecated)

        2. PP args mode:
            - PP args refer to the arguments at the beginning of the forward method, prefixed with 'pp_'.
            - These arguments are used for communication between pipeline stages.
            - If using PP args mode, replace mode is not supported.
        """
        self.n_pp_args = n

        arg_names = get_arg_names(self.pp_models[0].forward)
        for i in range(n):
            assert arg_names[i].startswith("pp_")
        for i in range(n, len(arg_names)):
            assert not arg_names[i].startswith("pp_")

    def set_static_kwargs_keys(self, keys):
        """
        Note: 'pp_args' and replaceable arguments should not be included here to avoid duplication.

        This function is deprecated. Use set_replacable_kwargs instead.
        """
        raise DeprecationWarning("set_static_kwargs_keys is deprecated. Use set_replacable_kwargs instead.")

    def set_replacable_kwargs(self, keys):
        self.replacable_kwargs_keys = keys

    def set_sync_input(self, enabled):
        self.sync_input = enabled

    def _check_input(self, *args, **kwargs):
        from types import NoneType

        for arg in args:
            if isinstance(arg, (int, float, str, bool, NoneType)):
                continue
            assert isinstance(arg, torch.Tensor), f"Input should be a tensor, got {arg}"
            assert arg.shape[0] == self.batch_size, f"Wrong batch size input, current shape {arg.shape}, expected batch_size {self.batch_size}"
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool, NoneType)):
                continue
            if isinstance(v, torch.Tensor):
                try:
                    assert v.shape[0] == self.batch_size, f"Wrong batch size input for `{k}` , current shape {v.shape}, expected batch_size {self.batch_size}"
                except Exception as e:
                    # NestedTensor still works fine in pp communication, but some ops are not supported.
                    if "NestedTensorImpl" not in str(e):
                        raise
            else:
                # TODO(kevinkhwu): Remove this type checking when PyTorch fixes this issue.
                # dict, list or tuple input will lead to error in pipeline._check_input.
                # Root cause:
                #   1. The output of the previous stage can not be an arbitrary object.
                #      An arbitrary output object will be wrapped into a tuple and lead to error in pipeline._check_input in the next stage.
                #      (_check_input assumes that the length of the tuple is the batch size, but got 1 in this case)
                #      (For dict, len((dict, )) != len(dict)) # this tuple has length 1.
                #      See also: torch/distributed/pipelining/stage.py: _normalize_model_output_as_tuple
                #   2. split_args_kwargs_into_chunks can not properly split list,
                #      each stage will receive the entire list instead of a microbatch chunk.
                #      Thus, parsing list or tuple input requires handling stage index in each stage to retrieve the correct chunk.
                # See also:
                #   - torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks (using torch.tensor_split)
                #   - torch.distributed.pipelining.schedules.py:_split_inputs
                #   - torch.distributed.pipelining.schedules.py:_check_inputs
                assert not isinstance(v, (dict, list, tuple)), f"Kwarg Input `{k}` should be a tensor, got {type(v)}"
                if not isinstance(v, SafeObjectWrapper):
                    log_once(
                        f"Parsing a Ojbect ({type(v)}) as input to pp pipeline. "
                        "Please ensure that this object is properly chunked already or does not require chunking. "
                        "Consider setting collection_to_object=True or set batch_object_kwargs_keys to include the keys of the collection.",
                        level="WARNING",
                    )

    def _check_output(self, output):
        assert not isinstance(output, (dict, list, tuple)), f"Output should be a tensor, got {type(output)}"

    def is_valid_m(self, m, input_batch_size):
        if input_batch_size % m != 0:
            return False
        if (
            "1F1B" in self.pipeline_parallel_schedule or self.is_multi_schedule(self.pipeline_parallel_schedule)
        ) and "Inference" not in self.pipeline_parallel_schedule:
            if m < self.pp_size:
                return False
            if self.max_pp_batchsize is not None and (input_batch_size // m * self.pp_size > self.max_pp_batchsize):
                return False
        if "GPipe" in self.pipeline_parallel_schedule:
            if self.max_pp_batchsize is not None and input_batch_size > self.max_pp_batchsize:
                return False
        return True

    def rank0_log(self, msg, level="INFO"):
        if dist.get_rank() == 0:
            loguru.logger.opt(depth=1).log(level, msg)

    def pre_process_input(self, *args, **kwargs):
        """
        Can be overridden to pre-process input arguments.
        Only effective in pp mode.
        """
        return args, kwargs

    def clear_states_after_auto_benchmark(self):
        self.clear_cached_result()

    def __get_reserved_keys(self):
        reserved_keys = {"sync_input", "calc_loss"}
        if self.parallel_dims.pp_enabled:
            reserved_keys.update({"target", "losses"})
        return reserved_keys

    def filter_reserved_keys_from_kwargs(self, kwargs):
        reserved_keys = self.__get_reserved_keys()
        return {k: v for k, v in kwargs.items() if k not in reserved_keys}

    def __call__(self,
        *args,

        sync_input=None,
        check_input=False,
        check_input_prob=1,
        calc_loss=True,

        # target=None,     # placeholder
        # losses=None,     # placeholder

        **kwargs,
    ):
        r"""Executes the parallel engine with the provided input arguments.

        This method is the main entry point for running the parallel engine. It handles
        synchronization of input data across parallel dimensions, manages cached results,
        and delegates execution to the internal implementation.

        Args:
            args, kwargs: Positional and keyword arguments to be passed to the engine.
            calc_loss (bool, optional): Whether to calculate the loss. See code above for details.
                This will influence the return value of this function.
            sync_pp_result (bool, optional): Whether to synchronize the result across pipeline stages.

        Returns:
            Returns maybe_sync(self.get_real_ret())

            Please refer to `get_real_ret` for more details.

        Note:
            'loss' and 'loss_dict' are cached in the forward pass (unless `calc_loss` is False).
            In pp case, only the last rank has cached results. But all ranks can retrieve the cached results
            by calling `get_cached_result` and set `sync_pp_result` to True.
        """
        self.barrier_fsdp_groups()

        arg_names = get_arg_names(self.fsdp_models[-1].forward)
        reserved_keys = self.__get_reserved_keys()
        overlap = reserved_keys.intersection(arg_names)

        if overlap:
            raise RuntimeError(
                f"The model's `forward` uses reserved keys {overlap} as argument names. "
                f"Please avoid using: {reserved_keys}"
            )

        self.clear_cached_result()
        if sync_input is None:
            if self.sync_input is not None:
                sync_input = self.sync_input
            else:
                sync_input = os.environ.get("HY_PARALLELISM_SYNC_INPUT", "0") == "1"
        if sync_input:
            with profile_range("sync_input"):
                try:
                    args, kwargs = sync_object_for_parallel_training([args, kwargs], parallel_dims=self.parallel_dims)
                except Exception as e:
                    # jvp can not run syncing since to_list is called broadcast_object_list
                    if "Cannot access data pointer of Tensor that doesn't have storage" in str(e):
                        log_once(f"Skip syncing: JVP can not run syncing since to_list is called broadcast_object_list, {e}", level="WARNING")
                    else:
                        raise

        if check_input:
            import random
            if random.random() < check_input_prob:
                sync_object_for_parallel_training([args, kwargs], parallel_dims=self.parallel_dims, debug_with_check=True)

        # if self.activation_offloading:
        #     context = torch.autograd.graph.save_on_cpu(pin_memory=self.activation_offloading_pin_memory)
        # else:
        #     context = nullcontext()
        # with context:
        return self.__call_impl(
            *args,
            sync_input=sync_input,
            calc_loss=calc_loss,

            **kwargs
        )

    def maybe_auto_benchmark_to_choose_optimal_pp_setting(self, *args, **kwargs):
        # benchmark, find the optimal m
        self.is_first_run[self.training] = False

        torch.cuda.synchronize()
        dist.barrier()

        import numpy as np
        from collections import defaultdict, Counter

        self.rank0_log(f"Benchmarking to find the optimal m for pipeline parallelism for {'Training' if self.training else 'Inference'}.")
        input_batch_size = self.get_batch_size(*args, **kwargs)
        assert input_batch_size is not None

        max_runs = 50
        converge_n = 5
        time_dict = defaultdict(list)
        schedules = self.benchmark_schedule[self.training]

        min_m_1f1b = float("inf")

        for s in schedules:
            if not self.training:
                assert s == self.get_inference_pipeline_schedule()
            else:
                self.training_pipeline_parallel_schedule = s
            self.set_pp_schedule(s)
            if self.benchmark_m_list is not None:
                valid_m = list(reversed([i for i in self.benchmark_m_list if self.is_valid_m(i, input_batch_size)]))
            else:
                valid_m = list(reversed([i for i in range(1, input_batch_size + 1) if self.is_valid_m(i, input_batch_size)]))

            if "1F1B" in s and len(valid_m) > 0:
                min_m_1f1b = min(*valid_m, min_m_1f1b)

            self.rank0_log(f"Testing m candidates: {valid_m} for schedule {s}")
            for test_m in valid_m:
                if "GPipe" == s and test_m > min_m_1f1b:
                    loguru.logger.info(f"Skip Gpipe {test_m}")
                    continue

                self._m_microbatch_dict[self.training] = test_m
                self.set_micro_batch_size(input_batch_size)
                converge_count = 0
                for i in range(max_runs):
                    start = time.time()
                    self.__call__(*args, **kwargs)
                    end = time.time()
                    time_cost = end - start
                    self.rank0_log(f"Testing {s, test_m} {i}/{max_runs}: {time_cost=}", level="DEBUG")

                    if i != 0:  # skip warmup
                        time_dict[(s, test_m)].append(time_cost)

                    if dist.get_rank() == 0 and len(time_dict[(s, test_m)]) >= converge_n:
                        if abs(time_cost - np.mean(time_dict[(s, test_m)][converge_n:])) < 2 * np.std(time_dict[(s, test_m)][converge_n:]):
                            converge_count += 1
                        else:
                            converge_count = 0
                    else:
                        converge_count = 0
                    buffer = [converge_count]
                    dist.broadcast_object_list(buffer, src=0)
                    converge_count = buffer[0]

                    if converge_count > converge_n:  # and i > min_runs:
                        self.rank0_log("Time cost converge. break", level="DEBUG")
                        break

                # clear schedule cache
                self.reso_stage_manager.clear()

        if len(time_dict) == 0:
            msg = f"No valid m for {schedules} with input batchsize {input_batch_size}"
            # if self.max_pipeline_microbatchsize:
            #     msg += f' and max_pipeline_microbatchsize {self.max_pipeline_microbatchsize}'
            raise RuntimeError(msg)

        # Remove time_cost outliers by filter out those that are more than 3 stds away from the mean
        time_dict = {k: [x for x in v if abs(x - sum(v) / len(v)) < 3 * np.std(v)] for k, v in time_dict.items()}

        pp_config_pair = min(time_dict.keys(), key=lambda x: np.mean(time_dict[x][-converge_n:]))

        for k, v in time_dict.items():
            self.rank0_log(f"Time cost for {k}: [{', '.join([f'{cost:.2f}' for cost in v])}]")

        buffer = [None] * dist.get_world_size()
        dist.all_gather_object(buffer, pp_config_pair)
        self.rank0_log(f"Optimal config in all ranks: {buffer}")
        cnt = Counter(buffer)
        pp_config_pair = max(cnt.keys(), key=lambda x: cnt[x])

        min_cost_schedule, min_cost_m = pp_config_pair

        self.rank0_log(f"Use {pp_config_pair} as the optimal m for pipeline parallelism during {'Training' if self.training else 'Inference'}.")
        self.rank0_log(f"Fwd{' + Bwd' if self.training else ''} Time cost: {sum(time_dict[pp_config_pair]) / len(time_dict[pp_config_pair]):.2f}")

        if self.training:
            self.training_pipeline_parallel_schedule = min_cost_schedule
        self.set_pp_schedule(min_cost_schedule)
        self._m_microbatch_dict[self.training] = min_cost_m
        self.set_micro_batch_size(input_batch_size)
        if hasattr(self, "optimizer_container"):
            self.zero_grad()
        self.clear_states_after_auto_benchmark()


    def __call_impl(self, *args, **kwargs):
        if self.enable_pp:
            self.clear_cached_result()

            if self.is_first_run[self.training] and self.enable_benchmark:
                self.maybe_auto_benchmark_to_choose_optimal_pp_setting(*args, **kwargs)
                return self.__call__(*args, **kwargs)

            self._check_states()

            bs = self.get_batch_size(*args, **kwargs)
            self.set_resolution(self.get_resolution_key(*args, **kwargs), bs)

            self._check_states()

            """
            __init__ 中可能有几次替换forward；
                1. pp_friendly_forward 替换
                2. loss_closure_forward 替换
                3. autocast 替换
                4. compile 也可能替换？
            get_arg_names 获取参数名字来决定什么会被使用
            所以需要注意前面几次替换forward时是否有 wraps，因为这会影响 get_arg_names 的返回值

            以下代码分 3 步：
            1. 将 args 放进 kwargs
            2. 看 kwargs 中是否有 batch_object_kwargs_keys 中的 key，如果有，则将对应的值 转 tensor
            3. 把 replacable 拎出来，其他放进static_kwargs
            """

            static_kwargs = {}
            if len(args) > 0:
                # import inspect
                # arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
                arg_names = get_arg_names(self.pp_models[0].forward)

                args_to_kwargs = {}
                for idx, k in enumerate(args):
                    arg_name = arg_names[self.n_pp_args + idx]
                    assert arg_name not in kwargs
                    args_to_kwargs[arg_name] = k

                kwargs.update(args_to_kwargs)

                args = []

            # 用户传入的 obj 可能有几种意图
            # 1. list[obj], tuple[obj]:
            #    1. 一个batch的obj，每个sample从list中取 [用户一般不会习惯这么用，暂不支持]
            #    2. 独立obj, 这个batch中的每个sample都一样  (勉强支持一下。。。其实不知道为什么非要传 object, 可能tuple方便?)
            # 2. obj:
            #    1. pp 本身可以处理, 但无法切分里面的tensor, 而且pp无法处理stage之间的 obj
            # 3. dict(tensor):
            #    1. 这种最难搞，这里面的tensor可能要分batch... 不允许这样传参！！！！
            # 4. list(tensor):
            #    1. 这种也很难搞，list 的 tensor 也要分 batch... 不允许这样传参！！！！
            # for k in self.batch_object_kwargs_keys:
            #     if k in kwargs:
            #         obj = kwargs[k]
            #         if isinstance(obj, dict):
            #             raise ValueError(f'Dict args or dict kwargs are not supported. Please use tensor.')
            #         # if not isinstance(obj, (list, tuple)): # 只有这种情况需要特别处理，其他自定义obj pp 本身可以处理
            #             # raise ValueError(f'There is no need to handle {k} specially, it can be handled by pp itself.')

            #         # 如果是 dict, list, tuple, 请确保里面的value也好，内容也好，不能存在 tensor!, 否则 batch 切分的逻辑未知

            #         # 注释掉下面部分，暂不支持上面的1.1情况
            #         # bs = self.get_batch_size(*args, **kwargs)
            #         # if isinstance(obj, (list, tuple)):
            #         #     assert len(obj) == bs, f'Wrong batch size input for `{k}` , current shape {len(obj)}, expected batch_size {bs}'
            #         # else:
            #         #     obj = [obj] * bs
            #         # obj = [obj] * bs
            #         # kwargs[k] = batch_obj_to_tensor(kwargs[k], max_len=self.batch_object_kwargs_max_len)
            #         obj_tensor = obj_to_tensor(obj, max_len=self.batch_object_kwargs_max_len)
            #         kwargs[k] = torch.stack([obj_tensor] * bs)

            if self.collection_to_object:
                for k in kwargs:
                    if isinstance(kwargs[k], (list, tuple, dict)):
                        kwargs[k] = SafeObjectWrapper(kwargs[k])

            args, kwargs = self.pre_process_input(*args, **kwargs)
            self._check_input(*args, **kwargs)

            static_kwargs = {}
            replacable_kwargs = {}
            for k in list(self.filter_reserved_keys_from_kwargs(kwargs).keys()):
                if k not in self.replacable_kwargs_keys:
                    static_kwargs[k] = kwargs.pop(k)
                else:
                    replacable_kwargs[k] = kwargs.pop(k)

            assert len(args) == 0
            replacable_args = []
            arg_names = get_arg_names(self.pp_models[0].forward)
            for i in range(len(arg_names)):
                if arg_names[i] in replacable_kwargs:
                    replacable_args.append(replacable_kwargs.pop(arg_names[i]))
                else:
                    assert len(replacable_kwargs) == 0
                    break

            # 检查一下 pp_models[0].forward, 确保所有参数都有默认值
            # 因为 pp_xxx 要放在最前面，并且用户不会传，所以pp_xxx要有默认值
            # 最前面的都有默认值，那所有参数都有默认值
            if len(self.replacable_kwargs_keys) == 0:
                import inspect

                forward_fn = self.pp_models[0].forward
                sig = inspect.signature(forward_fn)
                for param in list(sig.parameters.values()):
                    if param.default is inspect.Parameter.empty:
                        raise ValueError(
                            f"In pp_args mode, all parameters must have default values. Parameter '{param.name}' in {type(self.pp_models[0]).__name__}.forward must have a default value."
                        )

            return self.forward(replacable_args, static_kwargs=static_kwargs)
        else:
            self.clear_cached_result()
            assert len(self.fsdp_models) == 1
            ret = self.fsdp_models[0](*args, **self.filter_reserved_keys_from_kwargs(kwargs))  # loss_closure_forward will be called and ret_val will be cached
            return self.get_real_ret(ret, *args, **kwargs)

    def get_result_rank(self, stage_idx=-1):
        # assert len(self.pp_models) == 1
        return self.pp_mesh.mesh[stage_idx].item()

    def sync_pp_result(self, result, src_stage_idx=-1):
        if not self.enable_pp:
            return result
        return auto_broadcast(result, src=self.get_result_rank(src_stage_idx), group=self.pp_mesh.get_group())

    def forward(self, replacable_args=None, static_kwargs=None, sync_pp_result=True):
        """
        Forward pass for the parallel engine.

        If pipeline parallelism (enable_pp) is enabled:
            - This method (forward) and __call__ both perform forward and backward passes.
            - The return value is the result of the forward pass, optionally synchronized across pipeline stages.

        If pipeline parallelism is not enabled:
            - forward only performs the forward pass.
            - __call__ performs both forward and backward passes.

        Args:
            *args: Positional arguments for the model's forward method.
            static_kwargs (dict, optional): Static keyword arguments to be passed to the model's forward method.
            sync_pp_result (bool, optional): If True and pipeline parallelism is enabled, synchronize the result across pipeline stages. Default is True.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            The output of the forward pass, optionally synchronized across pipeline stages if pipeline parallelism is enabled.
        """
        ret = self._forward(replacable_args=replacable_args, static_kwargs=static_kwargs)
        if self.enable_pp:
            real_ret = self.get_real_ret(ret)
            if sync_pp_result:
                return self.sync_pp_result(real_ret)
            else:
                return real_ret
        else:
            raise Exception("Should never happen")
            return ret

    def del_cached_result_entry(self, key):
        key = f"_pp_result_cache_list_{key}"
        if not hasattr(self.fsdp_models[-1], "_pp_valid_result_keys"):
            return
        if key in self.fsdp_models[-1]._pp_valid_result_keys:
            delattr(self.fsdp_models[-1], key)
            self.fsdp_models[-1]._pp_valid_result_keys.remove(key)

    def clear_cached_result(self):
        if not hasattr(self.fsdp_models[-1], "_pp_valid_result_keys"):
            return
        for valid_key in self.fsdp_models[-1]._pp_valid_result_keys:
            if hasattr(self.fsdp_models[-1], valid_key):
                getattr(self.fsdp_models[-1], valid_key).clear()

    def get_last_call_result(self, key, sync_pp_result=True, merge_op="cat"):
        return self.get_cached_result(key, sync_pp_result, merge_op)

    def get_cached_result(self, key, sync_pp_result=True, merge_op="cat"):
        r"""Retrieve and merge cached results from the last pipeline stage and broadcast it to all ranks if sync_pp_result is True.

        Args:
            key (str): The tag identifying the cached result to retrieve.
            sync_pp_result (bool, optional): If True and pipeline parallelism is enabled,
                synchronize the merged result across all pipeline stages. Default is True.
                When setting this to False, some ranks may get None.
            merge_op (str, optional): One pp forward may include multiple microbatches.
                This argument specifies the operation to use when merging microbatch results.
                Supported values are 'cat' (concatenate), 'mean', and 'sum'. Default is 'cat'.

        Example:
            >>> ret = self.get_cached_result("loss", merge_op="mean") # return exactly what the loss_closure returns (after mean on the microbatch dimension)
            >>> ret = self.get_cached_result("ret_val") # return exactly what the model.forward / pp_friendly_forward returns
            >>> ret = self.get_cached_result("loss_dict") # return exactly what the loss_closure returns

            If the user caches the 'other' key in the forward pass, it can be retrieved by:
            >>> ret = self.get_cached_result('other')

        Returns:
            The merged and optionally synchronized cached result, or None if the result is not found.
        """
        # Generally, only the last rank has cached results
        # assert self.enable_pp

        if self.enable_pp and "loss" in key and merge_op != "mean":
            loguru.logger.opt(depth=1).warning(f'merge_op should be "mean" for "loss" when pp is enabled, but got {merge_op}')

        key = f"_pp_result_cache_list_{key}"
        if not hasattr(self.fsdp_models[-1], "_pp_valid_result_keys") or key not in self.fsdp_models[-1]._pp_valid_result_keys:
            ret = None
        else:
            ret = getattr(self.fsdp_models[-1], key)
            if not self.enable_pp:
                ret = ret[0]  # 非 pp 只会forward一次，cache的列表只有一个元素
            else:
                ret = self.merge_microbatch_object_list(ret, merge_op=merge_op)
        if self.enable_pp and sync_pp_result:
            ret = self.sync_pp_result(ret)

        return ret

    def _cache_ret_and_loss(self, step_ret, *args, **kwargs):
        for model in self.fsdp_models:
            model.cache_result("ret_val", step_ret)
        if self.training and kwargs.get('calc_loss', True):
            # WARNING: 如果是 replace 的情况， kwargs 的 loss 不能和 被 replace 的东西相关！不然会出问题
            pp_friendly_loss, loss_closure_out = self.loss_closure(step_ret, {"args": args, "kwargs": kwargs})
            loss, loss_dict = loss_closure_out
            for model in self.fsdp_models:
                model.cache_result("loss", loss)
                model.cache_result("loss_dict", loss_dict)

    def get_real_ret(self, step_ret, *args, **kwargs):
        if self.enable_pp:
            # The step_ret obtained by pp_scheduler.step() can be loss (if training) or forward return value (if eval)
            # We return other stuffs instead of the step_ret below.
            if self.training:
                # This is exactly what the loss_closure returns, after applying mean on the microbatch dimension.
                # Since pp handles backward internally, this loss takes no effect.
                # Engine.backward is a no-op.
                return self.get_cached_result("loss", merge_op="mean")
            else:
                # In inference, step_ret and the merged cached return value (self.get_cached_result("ret_val")) (MAY) be the same
                # `ret_val` is exactly what the model.forward / pp_friendly_forward returns
                ret = self.get_cached_result("ret_val")
                if ret is None:
                    raise RuntimeError("Ret val is not cached")
                    return step_ret
                return ret
        else:
            # To align with pp, we cache the ret_val and loss here.
            # For the pp case, ret_val is cached in the new `forward` method defined in `loss_closure_forward`
            self._cache_ret_and_loss(step_ret, *args, **kwargs)

            if self.training:
                # if pp is not enabled, this is EXACTLY what the loss_closure returns
                return self.get_cached_result("loss", merge_op=None)
            else:
                # return self.get_cached_result('ret_val')
                return step_ret

    def custom_merge_return_val(self, ret_val, merge_op=None):
        # fall back
        return ret_val

    def merge_microbatch_object_list(self, ret_val, merge_op="cat"):
        r"""Merge a list of objects (from microbatches) into a single object.

        This function recursively merges a list of objects, typically the outputs from multiple microbatches,
        into a single object. The merging strategy depends on the type of the objects:
        - For tensors, it concatenates or aggregates them along the batch dimension.
        - For lists/tuples, it merges each element recursively.
        - For dicts, it merges each value recursively by key.
        - For scalars (int, str, bool), it checks for consistency across microbatches.
        - For other types, it calls `self.custom_merge_return_val`.

        Args:
            ret_val (list): List of objects to merge, one per microbatch. Each element should be of the same type.
            merge_op (str): Merge operation for tensors. Supported: 'cat' (concatenate), 'mean', 'sum'.

        Returns:
            The merged object, with the same structure as the elements of `ret_val`.

        Raises:
            ValueError: If the types are inconsistent, or if an unsupported type is encountered,
                or if scalar values are inconsistent across microbatches.

        Example:
            >>> ret_val = [torch.randn(1, 2) for _ in range(2)]
            >>> merged = m.merge_microbatch_object_list(ret_val)
            >>> ret_val = [
            ...     (1, torch.randn(1, 2), {'hh': torch.randn(3)}),
            ...     (1, torch.randn(1, 2), {'hh': torch.randn(3)}),
            ... ]
            >>> merged = m.merge_microbatch_object_list(ret_val)
        """
        if merge_op is None:
            return ret_val
        assert isinstance(ret_val, list)
        if len(ret_val) == 0:  # not output rank
            return

        if len(ret_val) == self.m_microbatch + 1:  # drop the shape inference result
            ret_val = ret_val[1:]
        assert len(ret_val) == self.m_microbatch, f"{len(ret_val)=} {self.m_microbatch=}"

        assert all([type(k) is type(ret_val[0]) for k in ret_val])
        if isinstance(ret_val[0], torch.Tensor):
            ret_val = [k for k in ret_val]  # copy
            if len(ret_val[0].shape) == 0:
                ret = torch.stack(ret_val)
            else:
                ret = torch.cat(ret_val, dim=0)
            if merge_op == "cat":
                return ret
            elif merge_op == "mean":
                return ret.mean(0)
            elif merge_op == "sum":
                return ret.sum(0)
            else:
                raise ValueError(f"Invalid merge_op: {merge_op}")
        elif isinstance(ret_val[0], (list, tuple)):
            l = []
            for i in range(len(ret_val[0])):
                l.append(self.merge_microbatch_object_list([k[i] for k in ret_val], merge_op=merge_op))
            return l
        elif isinstance(ret_val[0], dict):
            l = {}
            for k in ret_val[0].keys():
                l[k] = self.merge_microbatch_object_list([m[k] for m in ret_val], merge_op=merge_op)
            return l
        elif isinstance(ret_val[0], (int, str, bool)):
            if all([k == ret_val[0] for k in ret_val]):
                return ret_val[0]
            else:
                raise ValueError(f"Inconsistent return value in different micro batch {ret_val[0]}")
        else:
            if ret_val[0] is None:
                assert all(k is None for k in ret_val)
                return None
            # ret_val_type = type(ret_val[0])
            # from hy_parallelism.utils import gather_obj
            # ret_val_type_across_ranks = gather_obj(ret_val_type)
            # assert all(k == ret_val_type_across_ranks[0] for k in ret_val_type_across_ranks)
            # if ret_val_type == type(None):
            #     return None
            try:
                ret = self.custom_merge_return_val(ret_val, merge_op=merge_op)
                if ret == ret_val:
                    raise ValueError(f"Unsupported type {type(ret_val[0])}")
                else:
                    return ret
            except Exception as e:
                raise ValueError(f"Unsupported type {type(ret_val[0])}") from e

    def _forward(self, replacable_args=None, static_kwargs=None):
        if static_kwargs is None:
            static_kwargs = {}
        if replacable_args is None:
            replacable_args = []

        assert self.enable_pp

        assert "target" not in static_kwargs
        assert "losses" not in static_kwargs
        assert "calc_loss" not in static_kwargs

        pp_schedule = self.pp_schedule
        if self.training:
            targets, losses = (
                (torch.empty([self.batch_size, 1]), []) if pp_schedule.has_last_stage else (None, None)
            )
        else:
            targets, losses = (None, None)

        self.clear_cached_result()
        extra_context = nullcontext()
        if not self.training:
            extra_context = torch.no_grad()

        # HACK(kevinkhwu): This is a workaround for an existing PyTorch bug.
        # Setting autocast to False, since shape inference contains no_grad()
        # With autocast + no_grad + fsdp forward, next backward under autocast will raise error
        # https://github.com/pytorch/pytorch/issues/158232 (Seems to be fixed)
        with torch.amp.autocast("cuda", enabled=False), extra_context:
            if pp_schedule.has_first_stage:
                if self.n_pp_args > 0:
                    assert len(replacable_args) == 0
                    ret = pp_schedule.step(**static_kwargs, target=targets, losses=losses)
                else:
                    assert len(replacable_args) > 0
                    ret = pp_schedule.step(*replacable_args, **static_kwargs, target=targets, losses=losses)
            else:
                ret = pp_schedule.step(**static_kwargs, target=targets, losses=losses)

        if self.training:
            loss = (
                # not to use mean, because gradients are manually reduced in ModelToLossFn
                torch.sum(torch.stack(losses)).to(self.device) if pp_schedule.has_last_stage else torch.tensor([-1.0], device=self.device)
            )
            assert isinstance(self.loss_fn, ModelToLossFn)
            return loss
        else:
            return ret

    def refresh_stages(self):
        if not self.enable_pp:
            return
        new_stages = []
        for i in range(len(self.pp_stages)):
            stage = self.pp_stages[i]
            assert isinstance(stage, NoShapeInferenceStage)
            cls = type(stage)
            new_stage = cls(
                self,
                stage.submod,
                stage_index=stage.stage_index,
                num_stages=stage.num_stages,
                device=stage.device,
                group=stage.group,
                dw_builder=stage.dw_builder,
            )
            new_stages.append(new_stage)
        self.pp_stages = new_stages

    @staticmethod
    def remove_module(model, keys):
        for k in keys:
            setattr(model, k, None)

    def build_stage(
        self,
        stage_idx: int,
        num_stages: int,
        is_first: bool = False,
        is_last: bool = False,
        remove_module_fn=None,
    ):
        if len(self.stage_ids_this_rank(num_stages)) == 1:
            model = self.model
        else:
            model = copy.deepcopy(self.model)

        for module in model.modules():
            module.is_first_stage = is_first
            module.is_last_stage = is_last
            module.stage_idx = stage_idx
            module.num_stages = num_stages

        if remove_module_fn:
            remove_module_fn(model)

        stage = NoShapeInferenceStage(
            self,
            model,
            stage_idx,
            num_stages,
            self.device,
            group=self.pp_mesh.get_group(),
        )
        return stage, model

    def get_forward_func(self):
        return self.forward

    def pipeline_manual_split(self):
        r"""Manually splits the model into pipeline stages for pipeline parallelism.

        This method creates pipeline stages by partitioning the model according to the number of pipeline stages
        specified in the parallel state. For each stage assigned to the current rank, it builds a stage using
        `build_stage`, marking the first and last stages appropriately.

        Example:
        ```python
        def pipeline_manual_split(self):

            num_stages = self.parallel_dims.pp

            stages = []
            models = []
            for stage_idx in self.stage_ids_this_rank(num_stages):

                stage, model_chunk = self.build_stage(
                    stage_idx,
                    num_stages,
                    is_first=stage_idx == 0,
                    is_last=stage_idx == num_stages - 1,
                )

                stages.append(stage)
                models.append(model_chunk)

            return stages, models
        ```
        Example of initializing blocks:
        ```python
        n_layers = mm_double_blocks_depth + mm_single_blocks_depth

        pp_splits = None
        if pp_splits is None:
            assert n_layers % self.parallel_dims.pp_mesh.size() == 0, f"n_layers ({n_layers}) must be divisible by pp_mesh.size() ({self.parallel_dims.pp_mesh.size()})"
            n_each_pp_rank = n_layers // self.parallel_dims.pp_mesh.size()
            pp_splits = ','.join([str(n_each_pp_rank) for _ in range(self.parallel_dims.pp_mesh.size())])

        def is_pp_split(index, n_layers, pp_splits, bias=0):
            if not self.pp_enabled:
                return True
            splits = pp_splits.split(',')
            splits = [int(x) for x in splits]
            pp_rank = get_parallel_state().pp_mesh.get_local_rank()

            assert n_layers == sum(splits), f"n_layers ({n_layers}) and sum(pp_splits) ({sum(splits)}) mismatch"

            start = sum(splits[:pp_rank])
            end = start + splits[pp_rank]

            if index + bias < start or index + bias >= end:
                return False
            return True
        self.double_blocks = nn.ModuleList(
            [
                get_block()
                if is_pp_split(d_block_index, n_layers, bias=0) else None
                for d_block_index in range(mm_double_blocks_depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                get_block()
                if is_pp_split(d_block_index, n_layers, bias=mm_double_blocks_depth) else None
                for d_block_index in range(mm_single_blocks_depth)
            ]
        )
        ```

        Returns:
            tuple[list[PipelineStage], list[torch.nn.Module]]:
                - stages: List of PipelineStage objects for this rank.
                - models: List of model chunks (torch.nn.Module) corresponding to each stage.
        """
        num_stages = self.parallel_dims.pp

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(num_stages):
            stage, model_chunk = self.build_stage(
                stage_idx,
                num_stages,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )

            stages.append(stage)
            models.append(model_chunk)

        return stages, models

    @staticmethod
    def sync_obj_across_nodes(obj):
        from hy_parallelism import utils

        assert torch.cuda.device_count() == 8

        n_node = dist.get_world_size() // 8
        node_mesh = init_device_mesh("cuda", (n_node, 8), mesh_dim_names=("nodes", "gpus"))
        obj = utils.auto_broadcast(obj, group_src=0, group=node_mesh["gpus"].get_group())
        # dist.get_process_group_ranks(node_mesh['gpus'].get_group())
        dist.barrier()
        return obj

    def load_full_sd_by_cvt_to_dcp(self, sd_path, strict=True, enable_sharding=False):
        loguru.logger.warning('This function is deprecated. Please use `engine.load_checkpoint` instead.')
        raise DeprecationWarning('This function is deprecated. Please use `engine.load_checkpoint` instead.')
        if not enable_sharding:
            if dist.get_node_local_rank() == 0:
                td = tempfile.TemporaryDirectory()
                checkpoint_manager.torch_state_dict_to_dcp(dcp_save_path=td.name, sd_path=sd_path)
                loguru.logger.debug(f"Checkpoint conversion (sd -> dcp) is finished. Saving to {td.name} (original {sd_path})")
                sd_path = td.name
            else:
                sd_path = None

            sd_path = self.sync_obj_across_nodes(sd_path)
            # self.load_checkpoint(td.name, load_optimizer_states=False)
        else:
            return self.load_checkpoint(sd_path, strict=strict)
            self.model_checkpoint_manager.load_full_sd(sd_path, strict=strict)

            if False:
                from hy_parallelism.checkpoint.checkpoint_manager import load_pt_or_safetensors
                raise NotImplementedError("Sharding is not supported yet")
                state_dict = load_pt_or_safetensors(sd_path)
                sharded_sd = {
                    k: distribute_tensor(v, self.parallel_dims.gpu_wise_mesh, placements=[Shard(0)])
                    for k, v in state_dict.items()
                }
                td = tempfile.TemporaryDirectory()
                checkpoint_manager.torch_state_dict_to_dcp(dcp_save_path=td.name, sd_input=sharded_sd)
                loguru.logger.debug(f"Checkpoint conversion (sd -> dcp) is finished. Saving to {td.name} (original {sd_path})")


        self.model_checkpoint_manager.load_from_path(sd_path, strict=strict)

    def load_checkpoint(
        self,
        load_dir,
        tag=None,  # unused
        load_module_strict=True,  # unused
        load_optimizer_states=None,  # True False None
        optimizer_strict=None,
        load_lr_scheduler_states=True,
        load_module_only=False,  # unused
        custom_load_fn=None,  # unused
        strict=True,
    ):
        r"""Load model and optimizer checkpoints from a directory or a torch state dict.

        This function supports two types of input:
            1. A torch checkpoint file (state dict). In this case, the checkpoint will be converted to DCP format before loading.
            2. A DCP directory, which must contain 'weights' and optionally 'optimizer' subfolders. If only '.distcp' files are present,
               it is recommended to use `checkpoint_manager.load_ckpt` directly.

        Args:
            load_dir (str or Path): Path to the checkpoint directory or torch checkpoint file.
            tag (str, optional): Optional tag for subdirectory or checkpoint version. (Unused)
            load_module_strict (bool, optional): Unused.
            load_optimizer_states (bool or None, optional): Whether to load optimizer states. None for best effort loading.
            optimizer_strict (bool or None, optional): Whether to strictly enforce optimizer state keys.
                If ``None``, uses ``strict`` for backward compatibility.
            load_lr_scheduler_states (bool, optional): Whether to load learning rate scheduler states. (Currently unused)
            load_module_only (bool, optional): Unused.
            custom_load_fn (callable, optional): Unused.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the model.

        Returns:
            tuple:
                - path (Path): The resolved checkpoint path.
                - training_states (dict): The loaded training states, if available.
        """
        from pathlib import Path

        path = Path(load_dir)
        if tag is not None:
            path = path / tag
        if optimizer_strict is None:
            optimizer_strict = strict
        if path.is_dir():
            weight_path = Path(path) / self.MODEL_FOLDER
            if not weight_path.exists():
                assert not load_optimizer_states
                weight_path = Path(path)
                assert len(list(weight_path.glob("*.distcp"))) > 0
            self.model_checkpoint_manager.load_from_path(str(weight_path), strict=strict)

            if os.path.isdir(os.path.join(path, self.OPTIMIZER_FOLDER)) and load_optimizer_states is not False:
                try:
                    self.optimizer_checkpoint_manager.load_from_path(
                        os.path.join(path, self.OPTIMIZER_FOLDER), strict=optimizer_strict
                    )
                except BaseException as e:
                    if load_optimizer_states is True:
                        raise
                    else:
                        loguru.logger.warning(
                            f"Error encountered while loading optimizer/lr_scheduler states. "
                            f"Will be ignored in best effort loading mode ({load_optimizer_states=}). Error: {e}"
                        )
            else:
                if load_optimizer_states is True:
                    msg = f"The user specifies load_optimizer_states=True, but {os.path.join(path, self.OPTIMIZER_FOLDER)} does not exist"
                    loguru.logger.critical(msg)
                    raise FileNotFoundError(msg)

            base, ext = os.path.splitext(self.TRAINING_STATES_FILE)
            rank_state_path = os.path.join(path, base, f"rank{dist.get_rank()}{ext}")
            state_path = os.path.join(path, self.TRAINING_STATES_FILE)
            if os.path.isfile(rank_state_path):
                self.training_states = torch.load(rank_state_path, weights_only=False)
                loguru.logger.info(f"load training states from {rank_state_path}")
            elif os.path.isfile(state_path):
                self.training_states = torch.load(state_path, weights_only=False)
                loguru.logger.info(f"load training states from {state_path}")

        else:
            self.model_checkpoint_manager.load_full_sd(load_pt_or_safetensors(path), strict=strict)
            assert not load_optimizer_states
        return path, self.training_states

    def save_checkpoint(
        self,
        save_dir=None,
        tag=None,
        client_state: dict | None = None,
        # unused
        save_latest=True,
        exclude_frozen_parameters=False,
        save_optimizer_states=True,
        save_all_ranks_training_states=False,
        dcp_save_kwargs: dict | None = None,
    ):
        r"""Save training checkpoint including model state, optimizer states, and custom client state.

        Saves the current model state, optionally saves optimizer and learning rate scheduler states,
        and can include custom client state. The checkpoint is saved using distributed checkpoint
        format, ensuring proper handling of sharded parameters across parallel ranks.

        Args:
            save_dir (str, optional): Directory for saving the checkpoint. If ``None``, uses the
                default checkpoint directory set during engine initialization (``ckpt_dir``).
                Default: ``None``
            tag (str, optional): Checkpoint tag used as a unique identifier for the checkpoint.
                If ``None``, defaults to ``"latest"``. Tag name must be the same across all ranks.
                Default: ``None``
            client_state (dict, optional): State dictionary containing custom training states to
                save alongside the checkpoint. This is saved separately and can include any
                additional information needed for resuming training. Default: ``None``
            save_latest (bool, optional): Currently unused parameter. Default: ``True``
            exclude_frozen_parameters (bool, optional): Currently unused parameter.
                Default: ``False``
            save_optimizer_states (bool, optional): If ``True``, saves optimizer and learning rate
                scheduler states. Set to ``False`` to save only model parameters.
                Default: ``True``
            save_all_ranks_training_states (bool, optional): If ``True``, all ranks save their
                training states into a subfolder (e.g. ``training_states/rank0.pt``).
                If ``False``, only rank 0 saves ``training_states.pt``. Default: ``False``
            dcp_save_kwargs (dict, optional): Keyword arguments forwarded to
                :func:`torch.distributed.checkpoint.save` via the checkpoint manager.
                Default: ``None``

        .. note::
            All processes must call this method, not just rank 0. Each process needs to save its
            own sharded parameters and optimizer states. Calling this method only on rank 0 will
            cause the process to hang waiting for synchronization with other processes.

        .. warning::
            The method modifies checkpoint manager folder paths temporarily if ``save_dir`` is
            provided. The original paths are restored after saving, but ensure no concurrent
            checkpoint operations occur during this time.

        .. seealso::
            :meth:`load_checkpoint` for loading saved checkpoints.
        """
        # Handles mutable default arguments
        if client_state is None:
            client_state = {}


        if save_dir is not None:
            if self.ckpt_dir is not None:
                model_old_folder = self.model_checkpoint_manager.folder
                if hasattr(self, "optimizer_checkpoint_manager"):
                    optimizer_old_folder = self.optimizer_checkpoint_manager.folder
            self.model_checkpoint_manager.folder = save_dir
            if hasattr(self, "optimizer_checkpoint_manager"):
                self.optimizer_checkpoint_manager.folder = save_dir

        if tag is None:
            tag = "latest"

        self.reshard()
        self.model_checkpoint_manager.save(step_or_tag=tag, dcp_save_kwargs=dcp_save_kwargs)
        if save_optimizer_states:
            if hasattr(self, "optimizer_checkpoint_manager"):
                if self.optimizer_offloading:
                    self.cast_optimizer('cuda')
                self.optimizer_checkpoint_manager.save(step_or_tag=tag, dcp_save_kwargs=dcp_save_kwargs)
                if self.optimizer_offloading:
                    self.cast_optimizer('cpu')

        if client_state is not None:
            saved_state = self.training_states.copy()
            saved_state["client_state"] = client_state
            if save_all_ranks_training_states:
                base, ext = os.path.splitext(self.TRAINING_STATES_FILE)
                state_save_dir = os.path.join(save_dir, str(tag), base)
                Path(state_save_dir).mkdir(parents=True, exist_ok=True)
                state_save_path = os.path.join(state_save_dir, f"rank{dist.get_rank()}{ext}")
                torch.save(saved_state, state_save_path)
                loguru.logger.info(f"client_state saved to {state_save_path}")
            elif dist.get_rank() == 0:
                state_save_path = os.path.join(save_dir, str(tag), self.TRAINING_STATES_FILE)
                Path(state_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(saved_state, state_save_path)
                loguru.logger.info(f"client_state saved to {state_save_path}")

        if save_dir is not None and self.ckpt_dir is not None:
            self.model_checkpoint_manager.folder = model_old_folder
            if hasattr(self, "optimizer_checkpoint_manager"):
                self.optimizer_checkpoint_manager.folder = optimizer_old_folder

    @torch.no_grad()
    def get_global_grad_norm(self) -> float:
        from hy_parallelism.utils import get_global_grad_norm_by_mesh_old_ep
        from hy_parallelism.utils import get_global_grad_norm_by_mesh
        if self.enable_ep and self.moe_impl == "old":
            self._tag_is_expert_info_to_param_and_grad()
            return get_global_grad_norm_by_mesh_old_ep(
                self.optimizer_container.param_groups_by_mesh,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=None,
                pp_mesh=self.pp_mesh
            ).item()
        else:
            return get_global_grad_norm_by_mesh(
                self.optimizer_container.param_groups_by_mesh,
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=None,
                pp_mesh=self.pp_mesh
            ).item()

    def get_fsdp_model(self):
        if len(self.fsdp_models) == 1:
            return self.fsdp_models[0]
        return nn.ModuleList(self.fsdp_models)

    @property
    def module(self):
        if len(self.fsdp_models) == 1:
            return self.fsdp_models[0]
        return nn.ModuleList(self.fsdp_models)

    def local_numel(self):
        return sum(p.numel() for p in self.parameters())


    def load_state_dict_from_engine(self, engine, strict=True):
        from hy_parallelism.parallel_states import device_mesh_context
        with device_mesh_context(self.parallel_dims.mesh_tag):
            if engine.parallel_dims.pp_enabled:
                state_dict = engine.full_state_dict(lazy=False)
            else:
                state_dict = engine.state_dict()
        with device_mesh_context(self.parallel_dims.mesh_tag):
            self.load_state_dict(state_dict, strict=strict)
