from collections import deque
from typing import Iterable

from torch.distributed import init_device_mesh

import copy
import math
import time
# from torch.distributed.pipelining import ScheduleZBVZeroBubble
from abc import abstractmethod
from contextlib import nullcontext
import loguru
import torch
from loguru import logger
from torch import distributed as dist
from torch import nn
from torch.distributed.pipelining import PipelineStage
from hymm.parallelism.pipeline import hy_get_schedule_class as get_schedule_class
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    # ScheduleZBVZeroBubble,
)
from hymm.parallelism.utils import log_once
from torch.distributed.tensor import DTensor

from .. import stateful
from .. import checkpoint_manager
from ..checkpoint_manager import OptimizersContainer, LRSchedulersContainer, CheckpointManager, Checkpoint
from ..stateful import TrainingState
import torch.distributed.checkpoint as dcp

from hymm.parallelism.utils import isolate_rng

import sys, os

from ..parallel_states import get_parallel_state, ParallelDims

DEBUG_MODE = False
# if DEBUG_MODE:
loguru.logger.remove(None)
loguru.logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:^8}</level> | <level><bold>[Rank " + os.environ.get('RANK', '0') + "]: {message}</bold></level> (<cyan>{file}:{line}</cyan>)",)


def get_arg_names(func):
    # will skip `self` if func is a bounded method
    # `self` will not be skipped if func is a class method
    import inspect
    arg_names = inspect.signature(func).parameters.keys()
    return list(arg_names)




class ModelToLossFn:
    def __init__(self, model, get_m_microbatch_fn):

        self.model = model
        # m_microbatch could be changed
        self.get_m_microbatch_fn = get_m_microbatch_fn

    def __call__(self, output, target):  # output is assume to be the total loss
        # kevinkhwu: This issue is fixed in PyTorch 2.7.0
        #   Our PyTorch is not updated to the latest version yet.
        #   https://github.com/pytorch/pytorch/pull/144352
        if torch.__version__ >= (2, 7, 0):
            loguru.logger.debug('kevinkhwu: The gradient accumulation issue is fixed, consider refactoring the code.')
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
    func.enabled = False
    return func

def is_implemented(func):
    if func is None: return False
    if hasattr(func, 'enabled'):
        return func.enabled
    return True

from abc import ABC
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
    def load_checkpoint(self,
                        load_dir,
                        tag=None,
                        load_module_strict=True,
                        load_optimizer_states=True,
                        load_lr_scheduler_states=True,
                        load_module_only=False,
                        custom_load_fn=None):
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
    def save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
        """Save training checkpoint

        Arguments:
            save_dir: Required. Directory for saving the checkpoint
            tag: Optional. Checkpoint tag used as a unique identifier for the checkpoint, global step is
                used if not provided. Tag name must be the same across all ranks.
            client_state: Optional. State dictionary used for saving required training states in the client code.
            save_latest: Optional. Save a file 'latest' pointing to the latest saved checkpoint.
            exclude_frozen_parameters: Optional. Exclude frozen parameters from checkpointed state.
        Important: all processes must call this method and not just the process with rank 0. It is
        because each process needs to save its master weights and scheduler+optimizer states. This
        method will hang waiting to synchronize with other processes if it's called just for the
        process with rank 0.

        """

    @abstractmethod
    def zero_grad(self):
        ...
    @abstractmethod
    def step(self, lr_kwargs=None):
        ...

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        ...

    @abstractmethod
    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        ...

    @abstractmethod
    def train(self, mode=True):
        ...

    @abstractmethod
    def eval(self):
        ...

    @property
    def module(self): return None

    @abstractmethod
    def get_global_grad_norm(self) -> float:
        ...

    @property
    def monitor(self): # TODO:
        return None

    @abstractmethod
    def state_dict(self):
        ...

    # @abstractmethod
    # def full_state_dict(self):
    #     ...

    # @abstractmethod
    @property
    def optimizer(self):
        ...

class ResoStageManager:

    def __init__(self, max_size=4, deque_size=1000):
        from collections import Counter
        self.stage_schedules = {}
        self.counter = Counter()
        self.reso_deque = deque(maxlen=deque_size)
        self.max_size = max_size


    def update(self, resolution_key, stage, scheduler):
        if len(self.reso_deque) == self.reso_deque.maxlen:
            left_most = self.reso_deque[0]
            self.counter[left_most] -= 1
            if self.counter[left_most] == 0:
                del self.counter[left_most]

        self.reso_deque.append(resolution_key)
        self.counter[resolution_key] += 1


        most_common = dict(self.counter.most_common(self.max_size))
        if resolution_key in most_common:
            self.stage_schedules[resolution_key] = (stage, scheduler)
        for key in list(self.stage_schedules.keys()):
            if key not in most_common:
                del self.stage_schedules[key]
        assert len(self.stage_schedules) <= self.max_size


    def clear(self):
        self.stage_schedules.clear()
        self.counter.clear()
        self.reso_deque.clear()

    def __contains__(self, item):
        loguru.logger.info(f'{self.counter=}')
        loguru.logger.info(f'{self.stage_schedules.keys()=}')
        return item in self.stage_schedules


    def __getitem__(self, item):
        return self.stage_schedules[item]





class BaseParallelEngine(DeepSpeedInterface):

    def is_final_step_rank(self) -> bool:
        """Check if current rank is the final step in pipeline."""
        if self.enable_pp:
            # assert len(self.pp_stages) == 1
            return self.pp_stages[-1].is_last
        return True

    def get_sub_device_mesh(self, mesh, key):
        try:
            return mesh[key]
        except:
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

    @not_implemented
    def param_init_fn(self, model, default_generator=None): pass

    @not_implemented
    def apply_tp(self, model, tp_mesh): ...

    @not_implemented
    def apply_sp(self, model, sp_mesh): ...

    @not_implemented
    def apply_ep(self, model): ...

    @not_implemented
    def apply_fsdp(self, model): ...


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
            self, model, loss_closure=None,
            pipeline_parallel_schedule=None,
            training_benchmark_schedule=('1F1B', 'GPipe'),
            auto_benchmark=False,
            max_pp_batchsize=None, # 仅在auto_benchmark=True时生效，用于避免m太小时，显存不足。accumulate 模式下必须 1F1B
            batch_size=None,
            m_microbatch=None,
            micro_batch_size=None, # 优先级最高！
            optimizer_config=None,
            get_lr_scheduler_func=None,
            loss_fn=None,
            pp_friendly_forward=None,
            logger=None,
            enable_fsdp=True,
            cpu_offload=False,
            parallel_dims:ParallelDims=None,
            pp_enable_autocast=True,
            autocast_prec='bf16',
            weight_prec='bf16',
            full_sd=None, # 切并行之前读的ckpt, 如果用的是天然切好的模型，那这个将不允许使用
            load_ckpt_path=None, # 切并行之后读的ckpt, 调用 load_checkpoint
            ckpt_dir=None, # 模型保存的路径, 训练时使用
            keep_latest_k=-1,
            initial_training_states={},
            gradient_accumulation_steps=1,
    ):
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_steps = 0

        if parallel_dims is None:
            loguru.logger.warning('Use default get_parallel_state')
            parallel_dims = get_parallel_state()

        if micro_batch_size:
            assert not auto_benchmark and m_microbatch is None
            self.__high_prio_micro_batch_size = micro_batch_size
        else:
            self.__high_prio_micro_batch_size = None

        self.reso_stage_manager = {}

        if parallel_dims.pp_enabled:
            def cache_result(self, tag, ret_val):
                key = f'_pp_result_cache_list_{tag}'
                # here we use `model` instead of `self` to enable submodules calling cache_result()
                # and save the result in root module
                if not hasattr(model, key):
                    setattr(model, key, [])
                    if hasattr(model, '_pp_valid_result_keys'):
                        getattr(model, '_pp_valid_result_keys').add(key)
                    else:
                        setattr(model, '_pp_valid_result_keys', {key})
                else:
                    getattr(model, '_pp_valid_result_keys').add(key)
                # getattr(model, key).append(ret_val)
                from hymm.parallelism.utils import map_tensor
                getattr(model, key).append(map_tensor(ret_val, lambda x: x.detach()))

            for module in model.modules():
                module.cache_result = cache_result.__get__(model)
                module.loss_closure = self.loss_closure

        # fix pipeline_parallel_schedule and m_microbatch
        if pipeline_parallel_schedule is None or m_microbatch is None:
            if not auto_benchmark:
                if micro_batch_size is None and m_microbatch is None:
                    auto_benchmark = True
            if pipeline_parallel_schedule is None:
                pipeline_parallel_schedule = 'GPipe'
            if m_microbatch is None:
                m_microbatch = 1

        precision_map = {
            'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
            'fp16': torch.float16, 'float16': torch.float16,
            'fp32': torch.float32, 'float32': torch.float32,
        }
        if autocast_prec not in precision_map:
            raise ValueError(f"Invalid autocast precision: {autocast_prec}")


        self.enable_pp = parallel_dims.pp_enabled
        self.enable_tp = parallel_dims.tp_enabled
        self.enable_sp = parallel_dims.sp_enabled
        self.enable_ep = parallel_dims.ep_enabled
        self.enable_fsdp = enable_fsdp




        # which scheduler to use when calling .train()
        # different from self.pipeline_parallel_schedule, which is the scheduler that is currently used
        self.training_pipeline_parallel_schedule = pipeline_parallel_schedule
        self.set_pp_schedule(pipeline_parallel_schedule)

        if auto_benchmark and pipeline_parallel_schedule is not None:
            assert isinstance(training_benchmark_schedule, (tuple, list))
            for schedule in training_benchmark_schedule:
                assert self.is_multi_schedule(schedule) == self.is_multi_schedule(pipeline_parallel_schedule), 'Configuration of `pipeline_parallel_schedule` and `training_benchmark_schedule` is imcompatible.'


        self.is_first_run = {True: True, False: True}
        self.benchmark_schedule = {True: training_benchmark_schedule, False: [self.get_inference_pipeline_schedule()]}
        self.enable_benchmark = auto_benchmark
        self.pp_enable_autocast = pp_enable_autocast
        self.autocast_prec = precision_map[autocast_prec]
        self.cpu_offload = cpu_offload
        self.weight_prec = precision_map[weight_prec]
        self.full_sd = full_sd
        self.load_ckpt_path = load_ckpt_path

        self.ckpt_dir = ckpt_dir
        self.keep_latest_k = keep_latest_k
        self.max_pp_batchsize = max_pp_batchsize
        self.training = True


        if self.autocast_prec == torch.float32 and pp_enable_autocast:
            raise ValueError("Autocast should be disabled for float32 precision")

        self.model = model
        self.__loss_closure = loss_closure

        assert parallel_dims is not None
        self.parallel_dims = parallel_dims

        self.training_states = TrainingState(
            tp=self.parallel_dims.tp,
            ep=self.parallel_dims.ep,
            pp=self.parallel_dims.pp,
            step_cnt=0,
        )
        self.training_states.update(initial_training_states)


        # if optimizer_config is None:
        #     optimizer_config = dict(
        #         optimizer_cls=torch.optim.AdamW,
        #         optimizer_kwargs=dict(
        #             lr=1e-5,
        #             betas=(0.9, 0.999),
        #             weight_decay=0.01,
        #             eps=1e-8,
        #         ))
        self.optimizer_config = optimizer_config
        self.get_lr_scheduler_func = get_lr_scheduler_func




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

        self.device = torch.device('cuda')

        self.logger = logger or loguru.logger

        if not self.enable_pp:
            self.pp_models = [model]

            self.post_init()

            assert len(self.pp_models) == 1
            assert len(self.fsdp_models) == 1
            return


        if self.__high_prio_micro_batch_size is None:
            self._m_microbatch_dict = {True: m_microbatch, False: 1}

        if batch_size is None:
            batch_size = 2**20 # use large enough batchsize to pass all check for build pipeline schedule
        self.set_micro_batch_size(batch_size)


        self.pp_stages, self.pp_models = self.pipeline_manual_split()
        # self.rank0_log('模型 pp 切分完成')
        self._check_pp_no_duplicate()


        for m in self.pp_models:
            if hasattr(m, 'pp_friendly_forward'):
                m.forward = m.pp_friendly_forward
            else:
                if pp_friendly_forward is not None:
                    m.forward = pp_friendly_forward.__get__(m)
                else:
                    ...
                    # raise RuntimeError(
                    #     f'{type(self.model)} does not implement `pp_friendly_forward`. '
                    #     f'The user should parse `pp_friendly_forward` as an argument to ParallelEngine'
                    # )

            if pp_enable_autocast:
                from functools import wraps
                m.forward = wraps(m.forward)(torch.amp.autocast('cuda', dtype=self.autocast_prec)(m.forward))

                # TODO: owing to PyTorch issue, we can not run schedule.step() under autocast. (no_grad in shape_inference)
                #   A workaround is to enable autocast only in the forward pass.
                #   But gradient checkpointing can lead to dtype mismatch during the forward pass in the backward passes.
                #   so we need to enable autocast for all module (especially gradient checkpointing blocks).
                #   When the PyTorch bug is fixed, wrap the autocast context around the schedule.step()
                loguru.logger.info('Enable autocast')
                for module in m.modules():
                    module.forward = wraps(module.forward)(torch.amp.autocast('cuda', dtype=self.autocast_prec)(module.forward))



        self.post_init()


        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            # assert len(self.fsdp_models) == 1
            self.loss_fn = ModelToLossFn(self.fsdp_models[-1], get_m_microbatch_fn=lambda : self.m_microbatch)

        assert self.loss_fn is not None

        self.build_pp_schedule()

        self.config_forward_args()

        self._check_scheduler()

        # offload model to cpu
        # self.model.cpu() # may influence the fsdp model since it is not copied sometimes
        delattr(self, 'model') # todo: maybe unnecessary??




    def state_dict(self):
        return self.model_checkpoint_manager.states['model'].state_dict()

    @property
    def optimizer(self):
        if not hasattr(self, 'optimizer_container'):
            raise RuntimeError('Should parse optimizer_config to pp_engine initialization')
        return self.optimizer_container



    def register_loss_closure(self, loss_fn):
        self.__loss_closure = loss_fn


    def loss_closure(self, *args, **kwargs):
        loss, loss_dict = self.__loss_closure(*args, **kwargs)
        loss = loss.contiguous().clone()
        if len(loss.shape) == 0:
            return loss[None].contiguous().clone(), loss_dict
        else:
            return loss, loss_dict


    def create_checkpoint_manager(self, ckpt_dir, keep_latest_k=-1):
        self.MODEL_FOLDER = 'weights'
        self.OPTIMIZER_FOLDER = 'optimizers'
        self.TRAINING_STATES_FOLDER = 'states'

        self.model_checkpoint_manager = CheckpointManager(
            {},
            Checkpoint(
                dump_folder=ckpt_dir,
                folder=self.MODEL_FOLDER,
                enable_checkpoint=True,
                keep_latest_k=keep_latest_k,
            ),
            model_parts=self.fsdp_models,
        )
        # optimizer 和 weights 分开存，方便后面储存不够时只删 optimizer
        if hasattr(self, 'optimizer_container') and hasattr(self, 'lr_scheduler_container'):
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

        # keys that not appears in self.training_states will not be loaded
        self.training_states_checkpoint_manager = CheckpointManager(
            {'states': self.training_states},
            Checkpoint(
                dump_folder=ckpt_dir,
                folder=self.TRAINING_STATES_FOLDER,
                enable_checkpoint=True,
                keep_latest_k=keep_latest_k,
            ),
            model_parts=None,
        )

    @property
    def m_microbatch(self):
        if self.__high_prio_micro_batch_size is None:
            return self._m_microbatch_dict[self.training]
        else:
            assert self.batch_size % self.__high_prio_micro_batch_size == 0, f'batch_size ({self.batch_size}) must be divisible by the given micro batch size ({self.__high_prio_micro_batch_size})'
            return self.batch_size // self.__high_prio_micro_batch_size

    @classmethod
    def get_inference_pipeline(cls, model, parallel_dims, pipeline_parallel_schedule='GPipe', m_microbatch=1, **kwargs):
        engine = cls(
            model=model,
            parallel_dims=parallel_dims,
            pipeline_parallel_schedule=pipeline_parallel_schedule,
            m_microbatch=m_microbatch,
            pp_enable_autocast=True,
            **kwargs,
        )
        engine.eval()
        return engine

    def zero_grad(self):
        if not hasattr(self, 'optimizer_container'):
            raise RuntimeError('Should parse optimizer_config to pp_engine initialization')
        self.optimizer_container.zero_grad()

    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        if self.enable_pp:
            pass
        else:
            loss.backward()

    def is_gradient_accumulation_boundary(self):
        """
        Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.

        """
        return (self.micro_steps + 1) % self.gradient_accumulation_steps == 0

    def step(self, lr_kwargs=None):
        if self.is_gradient_accumulation_boundary():
            self.optimizer_container.step()
            if self.lr_scheduler_container is not None:
                self.lr_scheduler_container.step()

            if 'step_cnt' in self.training_states:
                self.training_states['step_cnt'] += 1
            else:
                self.training_states['step_cnt'] = 0
            self.zero_grad()
        self.micro_steps += 1


    def get_last_lr(self):
        return self.lr_scheduler_container.schedulers[0].get_last_lr()

    def clip_grad_norm_(self,
                        parameters,
                        max_norm: float,
                        norm_type: float = 2.0,
                        error_if_nonfinite: bool = False,
                        foreach = None,
                        ):
        from ..utils import  clip_grad_norm_
        if self.enable_pp or self.enable_ep:
            return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh=self.parallel_dims.pp_mesh)
        else:
            grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)
            if hasattr(grad_norm, 'full_tensor'):
                grad_norm = grad_norm.full_tensor()
            return grad_norm






    def get_resolution_key(self, *args, **kwargs):
        raise NotImplementedError
    def get_batch_size(self, *args, **kwargs):
        raise NotImplementedError

    def set_micro_batch_size(self, bs):
        self.batch_size = bs

        assert self.batch_size % self.m_microbatch == 0, f'Invalid batch size {self.batch_size} or invalid m:{self.m_microbatch}'
        self.microbatch_size = self.batch_size // self.m_microbatch

        if self.training:

            if '1F1B' in self.pipeline_parallel_schedule:
                assert self.m_microbatch >= self.pp_size
        # else:
        #     self.microbatch_size = bs

        assert self.microbatch_size > 0
        assert self.is_valid_m(self.m_microbatch, bs), f'Invalid m_microbatch: {self.m_microbatch} for input batch size {bs} and scheduler {self.training_pipeline_parallel_schedule}'

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

        def clear_stages_schedule(stages, schedule):
            for stage in stages:
                stage.clear_runtime_states()
                stage.args_recv_info.clear()
                stage.grad_recv_info.clear()
                # stage.inputs_meta = None
                # stage.outputs_meta = None
                # stage._outputs_meta = None
            if hasattr(schedule, '_stages_initialized'):
                schedule._stages_initialized = False
            if hasattr(schedule, '_stage_initialized'):
                schedule._stage_initialized = False


        if should_refresh:
            self.force_refresh_pipeline_scheduler()
            self.reso_stage_manager[key] = (self.pp_stages, self.pp_schedule)
        else:
            self.pp_stages, self.pp_schedule = self.reso_stage_manager[key]

        for i, (k, v) in enumerate(self.reso_stage_manager.items()):
            stages, schedule = v
            if stages is self.pp_stages:
                assert schedule is self.pp_schedule
                continue
            clear_stages_schedule(stages, schedule)

            # loguru.logger.info(f'{i}/{len(self.reso_stage_manager)=} {vars(stages[0])}')
            # loguru.logger.warning(f'{i}/{len(self.reso_stage_manager)=} {vars(schedule)}')


        if DEBUG_MODE:
            all_pp_schedule = bing_utils.gather_obj(type(self.pp_schedule))
            for k in all_pp_schedule:
                if k != all_pp_schedule[0]:
                    loguru.logger.critical('Please report this bug to kevinkhwu')
                    if torch.distributed.get_rank() == 0:
                        import pdb
                        pdb.set_trace()
                    torch.distributed.barrier()
                    raise RuntimeError('Please report this bug to kevinkhwu')

    @abstractmethod
    def config_forward_args(self):
        pass



    def set_pp_schedule(self, pipeline_parallel_schedule):
        self.pipeline_parallel_schedule = pipeline_parallel_schedule
        schedule_class = get_schedule_class(self.pipeline_parallel_schedule)
        # self.pp_style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"
        self.pp_style = "v" if 'Bubble' in self.pipeline_parallel_schedule else "loop"


    @staticmethod
    def has_meta(module):
        for param in module.parameters():
            if param.device == torch.device('meta'):
                return True
        return False

    def to_empty_if_has_meta_and_init_param(self, models, call_init_fn=True):
        with isolate_rng():
            # models can be fsdp_models or pp_models (before applying fsdp). Depending on the implementation
            if hasattr(self, 'fsdp_models'):
                assert models is self.fsdp_models or models is self.pp_models
            else:
                assert models is self.pp_models

            for i, model in enumerate(models):


                if self.has_meta(model):
                    models[i] = model.to_empty(device="cuda" if not self.cpu_offload else 'cpu')

                    for module_name, module in models[i].named_modules():
                        if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()

                    assert is_implemented(self.param_init_fn) or self.full_sd is not None or self.load_ckpt_path is not None
                else:
                    if not self.cpu_offload:
                        models[i].cuda()

                if call_init_fn and is_implemented(self.param_init_fn): # and self.full_sd is None and self.load_ckpt_path is None:
                    with torch.no_grad():
                        self.param_init_fn(models[i])

                # kevinkhwu:
                #   Warning: Magic! don't touch
                #   For unknown reason, when using meta init and enabling gradient checkpointing, the recomputed activations may be in different dtype
                #   (even though all parameters are ensured to be in the same dtype and no parameter is in meta).
                #   Here the magic comes!
                #   We explicitly cast the model to the weight precision (even though they are already there), the issue got fixed.
                model.to(self.weight_prec)
                assert not self.has_meta(model)



    def post_init(self):

        if self.full_sd:
            self.to_empty_if_has_meta_and_init_param(self.fsdp_models)
            # if self.has_meta(self.model):
            #     self.model.to_empty(device="cpu")
            self.model.load_state_dict(self.full_sd, strict=True)

        self.apply_parallelism() # already called to_empty_if_has_meta_and_init_param inside

        self.build_optimize_scheduler()
        self.create_checkpoint_manager(self.ckpt_dir, self.keep_latest_k)

        if self.load_ckpt_path:
            self.load_checkpoint(self.load_ckpt_path)


        for fsdp_part in self.fsdp_models:
            for name, param in fsdp_part.named_parameters():
                if param.device == torch.device('meta'):
                    loguru.logger.critical(f'{name} is still in meta !!!!!!!!!')
                    raise RuntimeError

        if self.parallel_dims.dp_replicate_enabled:
            pass
            # I think we can skip the entire process
            # Since we will load checkpoint later, we are not training from scratch...

            # self.rank0_log("Synchronizing model parameters across all ranks...")
            # for model in self.fsdp_models:
            #     with torch.no_grad():

            #         src = int(self.parallel_dims.dp_replicate_mesh.mesh[0])
            #         group = self.parallel_dims.dp_replicate_mesh.get_group()
            #         group_ranks = dist.get_process_group_ranks(group)
            #         assert src in group_ranks

            #         print('broadcast')
            #         # can we guarantee the pararms are iterated in the same order?
            #         # For ranks in pp: NO, but we don't have to
            #         # For ranks in ep: ep shards cannot cross pp ranks
            #         for name, param in model.named_parameters():
            #             if isinstance(param, DTensor):
            #                 # FIXME: experts should not be broadcasted in initialization
            #                 if model.is_expert(name):
            #                     # What if we skip expert broadcasting
            #                     print('skip expert broadcast')
            #                     continue
            #                     replicate_group = self.parallel_dims.device_mesh_for_ep['ep_fsdp_replicate'].get_group()
            #                 else:
            #                     replicate_group = self.parallel_dims.dp_replicate_mesh.get_group()
            #                 dist.broadcast(
            #                     param._local_tensor,
            #                     group_src=0,
            #                     group=replicate_group,
            #                 )
            #             else:
            #                 raise NotImplementedError
            #                 dist.broadcast(
            #                     param,
            #                     group_src=0,
            #                     group=self.parallel_dims.dp_replicate_mesh.get_group()
            #                 )
            #         print('broadcast finished')

            #     self.rank0_log("Model parameters synchronized.")





    def build_optimize_scheduler(self):
        if self.optimizer_config:
            self.optimizer_container = OptimizersContainer(self.fsdp_models, **self.optimizer_config)
        if self.get_lr_scheduler_func is not None:
            self.lr_scheduler_container = LRSchedulersContainer(self.optimizer_container, self.get_lr_scheduler_func)
        else:
            self.lr_scheduler_container = None

    def _check_pp_no_duplicate(self):
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
                raise RuntimeError(f'Duplicate keys found: {duplicate_keys}. Check PP.')



    def apply_parallelism(self):
        if self.enable_tp:
            assert is_implemented(self.apply_tp)
            self.make_tp(self.apply_tp)

        if self.enable_sp:
            assert is_implemented(self.apply_sp)
            self.make_sp(self.apply_sp)

        if self.enable_ep:
            assert is_implemented(self.apply_ep)
            for m in self.pp_models:
                self.apply_ep(m)

        # 本来想放在 ep 之前，确保初始化的时候是有完整的参数，然后专家之前可以很轻易地保证初始化成不一样的值
        # 但是 ptm 的 ep 天然就切好的，所以还是放在 ep 后，用generator
        # 之所以放在fsdp前，是因为 DTensor 不好 index, 导致依赖index的初始化的逻辑不好写
        self.to_empty_if_has_meta_and_init_param(self.pp_models, call_init_fn=self.full_sd is None)

        if self.enable_fsdp:
            assert is_implemented(self.apply_fsdp)
            self.make_fsdp(self.apply_fsdp)
        else:
            self.fsdp_models = self.pp_models




    def make_tp(self, apply_tp):
        for m in self.pp_models:
            loguru.logger.info(f'apply tp on {type(m)}  {self.tp_mesh=}')
            apply_tp(m, self.tp_mesh)

    def make_sp(self, apply_sp):
        for m in self.pp_models:
            apply_sp(m, self.sp_mesh)

    def make_fsdp(self, apply_fsdp):
        self.fsdp_models = []
        for i, m in enumerate(self.pp_models):
            model_engine = apply_fsdp(m)
            assert model_engine is not None, 'apply_fsdp should return a model engine'
            if self.enable_pp:
                self.pp_stages[i].submod = model_engine
            self.fsdp_models.append(model_engine)


    @staticmethod
    def recursive_get_attr(model, attr):
        splits = attr.split('.')
        for split in splits:
            if split:
                model = getattr(model, split)
        return model

    def stage_ids_this_rank(
            self, num_stages: int
    ) -> tuple[int]:
        """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
        pp_rank, pp_size = self.pp_rank, self.pp_size
        style = self.pp_style
        assert (
                num_stages % pp_size == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
        stages_per_rank = num_stages // pp_size
        if style == "loop":
            return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
        elif style == "v":
            assert (
                    stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
            )
            return stage_v_pairs[pp_rank]

    def generate_split_points(
            self,
            num_layers: int,
            n_split = None,
            input_weight: int = 1,
            output_weight: int = 1,
            layers_per_stage=None,
            layers_preffix='transformer.h.',
    ) -> list[str]:
        """
        Generate a list of split points based on the number of layers and
        pipeline parallel dimension, ensuring the first and last stages have the least layers.

        Args:
            schedule_str (str): The string of the schedule name.
            layers_per_stage (int): The number of layers per stage.
            n_split (int): The pipeline parallel dimension.
            num_layers (int): The number of layers in the model.
            input_output_weight (int): The number of layers to consider the input/output modules in the layer calculation.

        Returns:
            list[str]: A list of split point FQNs.
        """
        assert layers_preffix.endswith('.')
        if n_split is None:
            n_split = self.pp_size
        schedule_str = self.pipeline_parallel_schedule
        if schedule_str:
            schedule_class = get_schedule_class(schedule_str)
            is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
            num_stages_per_rank = 1 if is_single_stage_schedule else 2
        else:
            is_single_stage_schedule = True
            num_stages_per_rank = 1

        if layers_per_stage is not None:
            total_stages = math.ceil(num_layers / layers_per_stage)
            if total_stages % n_split != 0:
                raise ValueError(
                    f"Number of stages ({total_stages}) must be divisible by the pipeline parallel dimension ({n_split})."
                    f"Each rank should have the same number of stages. "
                )
            num_stages_per_rank = total_stages // n_split

            if is_single_stage_schedule and num_stages_per_rank != 1:
                raise ValueError(
                    f"Number of stages per rank ({num_stages_per_rank}) must be 1 for single stage schedules."
                )
            elif not is_single_stage_schedule and num_stages_per_rank < 2:
                raise ValueError(
                    f"Number of stages per rank ({num_stages_per_rank}) must be >= 2 for multi stage schedules."
                )
        else:
            total_stages = n_split * num_stages_per_rank
            if total_stages > num_layers:
                raise ValueError(f"Total stages cannot be greater than the number of layers {total_stages=}  {n_split=} {num_stages_per_rank=}")

        # Calculate effective number of layers including input and output weights
        effective_num_layers = num_layers + input_weight + output_weight
        base_layers_per_stage = effective_num_layers // total_stages

        splits = [""] * (total_stages - 1)
        current_layer_index = 0

        # First stage
        layers_on_first_stage = max(0, base_layers_per_stage - input_weight)
        current_layer_index += layers_on_first_stage
        splits[0] = layers_preffix + str(current_layer_index)

        # Last stage
        layers_on_last_stage = max(0, base_layers_per_stage - output_weight)
        splits[-1] = layers_preffix + str(num_layers - layers_on_last_stage)

        # Middle stages
        remaining_layers = num_layers - layers_on_first_stage - layers_on_last_stage - 1
        middle_stages = len(splits) - 2
        layers_per_middle_stage = remaining_layers // middle_stages
        # split remainder evenly across middle stages
        remainder = remaining_layers % middle_stages

        for i in range(1, middle_stages + 1):
            current_layer_index += layers_per_middle_stage
            if remainder > 0:
                current_layer_index += 1
                remainder -= 1
            splits[i] = layers_preffix + str(current_layer_index)

        logger.info(
            f"No 'pipeline_parallel_split_points' provided so the generated splits are: {splits} "
            "This may be sub-optimal as the number of layers per stage may be unbalanced."
        )
        return splits

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
                f"of stages ({num_total_stages}) which may result in a bubble in the pipeline.",
                level='WARNING',
            )
        if not looped_schedule:
            assert len(stages) == 1, f'{len(stages)=} {schedule_class}'

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
        return sum([list(model.parameters()) for model in self.fsdp_models], [])
    
    def named_parameters(self):
        named_params = []
        for i, model in enumerate(self.fsdp_models):
            for name, param in model.named_parameters():
                assert (
                    name not in named_params
                ), f'{name} is already in named_params'
                named_params.append((name, param))
        return named_params


    def reshard(self):
        raise NotImplementedError

    def cpu(self):
        for m in self.fsdp_models:
            m.cpu()
        return self

    def cuda(self):
        self.reshard()
        for m in self.fsdp_models:
            m.cuda()
        return self

    def train(self, pipeline_parallel_schedule=None, force=False, mode=True):
        # self.microbatch_size = self.train_microbatch_size
        if not mode:
            return self.eval()

        for m in self.fsdp_models:
            m.train()
        if self.training and not force:
            return self

        self.training = True

        if self.enable_pp:
            self.set_pp_schedule(self.training_pipeline_parallel_schedule)

            self.refresh_stages() # train 和 eval 的尺寸可能不一样
            self.build_pp_schedule()
        return self


    def eval(self, force=False):
        for m in self.fsdp_models:
            m.eval()
        if self.training or force:
            self.training = False
            if self.enable_pp:
                self.set_pp_schedule(self.get_inference_pipeline_schedule())

                self.refresh_stages() # train 和 eval 的尺寸可能不一样
                self.build_pp_schedule()
        return self

    def get_inference_pipeline_schedule(self):
        if issubclass(get_schedule_class(self.training_pipeline_parallel_schedule), PipelineScheduleSingle):
            return 'GPipe'
        else:
            # return self.pipeline_parallel_schedule # not to change
            return 'InferencePipelineScheduleMulti'
            # return 'PipelineScheduleMulti'

    def _check_states(self):
        for m in self.fsdp_models:
            assert m.training == self.training

    def set_n_pp_args(self, n):
        self.n_pp_args = n
        self.pp_args = tuple([None] * n)

        import inspect
        # arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
        # for i in range(1, n + 1):

        arg_names = get_arg_names(self.pp_models[0].forward)
        for i in range(n):
            assert arg_names[i].startswith('pp_')


    def set_static_kwargs_keys(self, keys):
        self.static_kwargs_keys = keys

    def gather_attributes(self, attributes):
        ret = []
        for att_key in attributes:
            attr = [getattr(model, att_key) for model in self.fsdp_models]
            buffer = [None] * self.pp_mesh.size()
            assert len(attr) == 1 # Currently we assume that all pp rank have the same number of stages
            dist.all_gather_object(buffer, attr[0], group=self.pp_mesh.get_group())
            ret.append(buffer)
        return ret

    def _check_input(self, *args, **kwargs):
        from types import NoneType
        for arg in args:
            if isinstance(arg, (int, float, str, bool, NoneType)):
                continue
            assert isinstance(arg, torch.Tensor), f'Input should be a tensor, got {arg}'
            assert arg.shape[0] == self.batch_size, f'Wrong batch size input, current shape {arg.shape}, expected batch_size {self.batch_size}'
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool, NoneType)):
                continue
            assert isinstance(v, torch.Tensor), f'Kwarg Input `{k}` should be a tensor, got {v}'
            try:
                assert v.shape[0] == self.batch_size, f'Wrong batch size input for `{k}` , current shape {v.shape}, expected batch_size {self.batch_size}'
            except Exception as e:
                if 'NestedTensorImpl' not in str(e):
                    raise



    def is_valid_m(self, m, input_batch_size):
        if input_batch_size % m != 0:
            return False
        if ('1F1B' in self.pipeline_parallel_schedule or self.is_multi_schedule(self.pipeline_parallel_schedule)) \
                and 'Inference' not in self.pipeline_parallel_schedule:
            if m < self.pp_size:
                return False
            if self.max_pp_batchsize is not None and (input_batch_size // m * self.pp_size > self.max_pp_batchsize):
                return False
        if 'GPipe' in self.pipeline_parallel_schedule:
            if self.max_pp_batchsize is not None and input_batch_size > self.max_pp_batchsize:
                return False
        return True

    def rank0_log(self, msg):
        if dist.get_rank() == 0:
            loguru.logger.info(msg)

    def pre_process_input(self, *args, **kwargs):
        return args, kwargs

    def clear_states_after_auto_benchmark(self):
        self.clear_cached_result()

    def __call__(self, *args, **kwargs):
        # with torch.autocast("cuda", dtype=self.autocast_prec, enabled=self.pp_enable_autocast):
        self.clear_cached_result()
        return self.__call_impl(*args, **kwargs)

    def __call_impl(self, *args, **kwargs):
        if self.enable_pp:
            self.clear_cached_result()

            if self.is_first_run[self.training] and self.enable_benchmark:
                # benchmark, find the optimal m
                self.is_first_run[self.training] = False

                self.rank0_log('Benchmark: synchronize')
                torch.cuda.synchronize()
                self.rank0_log('Benchmark: Barrier')
                dist.barrier()


                import numpy as np
                from collections import defaultdict, Counter

                self.rank0_log(f'Benchmarking to find the optimal m for pipeline parallelism for {"Training" if self.training else "Inference"}.')
                input_batch_size = self.get_batch_size(*args, **kwargs)
                assert input_batch_size is not None

                max_runs = 50
                converge_n = 5
                time_dict = defaultdict(list)
                schedules = self.benchmark_schedule[self.training]
                # self.original_training_schedule = self.training_pipeline_parallel_schedule

                min_m_1f1b = float('inf')

                for s in schedules:
                    if not self.training:
                        assert s == self.get_inference_pipeline_schedule()
                    else:
                        self.training_pipeline_parallel_schedule = s
                    self.set_pp_schedule(s)
                    valid_m = list(reversed([i for i in range(1, input_batch_size + 1) if self.is_valid_m(i, input_batch_size)]))

                    if '1F1B' in s and len(valid_m) > 0:
                        min_m_1f1b = min(*valid_m, min_m_1f1b)

                    self.rank0_log(f'm candidates: {valid_m} for schedule {s}')
                    for test_m in valid_m:

                        if 'GPipe' == s and test_m > min_m_1f1b:
                            loguru.logger.info(f'Skip Gpipe {test_m}')
                            continue

                        self._m_microbatch_dict[self.training] = test_m
                        self.set_micro_batch_size(input_batch_size)
                        converge_count = 0
                        for i in range(max_runs):
                            start = time.time()
                            self.__call__(*args, **kwargs)
                            end = time.time()
                            time_cost = (end - start)
                            self.rank0_log(f'Testing {s, test_m} {i}/{max_runs}: {time_cost=}')


                            if i != 0: # skip warmup
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

                            if converge_count > converge_n: # and i > min_runs:
                                self.rank0_log('Time cost converge. break')
                                break


                        # clear schedule cache
                        self.reso_stage_manager.clear()

                if len(time_dict) == 0:
                    msg = f'No valid m for {schedules} with input batchsize {input_batch_size}'
                    if self.max_pipeline_microbatchsize:
                        msg += f' and max_pipeline_microbatchsize {self.max_pipeline_microbatchsize}'
                    raise RuntimeError(msg)

                # Remove time_cost outliers by filter out those that are more than 3 stds away from the mean
                time_dict = {k: [x for x in v if abs(x - sum(v) / len(v)) < 3 * np.std(v)] for k, v in time_dict.items()}

                pp_config_pair = min(time_dict.keys(), key=lambda x: np.mean(time_dict[x][-converge_n:]))

                for k, v in time_dict.items():
                    self.rank0_log(f'Time cost for {k}: [{", ".join([f"{cost:.2f}" for cost in v])}]')

                buffer = [None] * dist.get_world_size()
                dist.all_gather_object(buffer, pp_config_pair)
                self.rank0_log(f'Optimal config in all ranks: {buffer}')
                cnt = Counter(buffer)
                pp_config_pair = max(cnt.keys(), key=lambda x: cnt[x])

                min_cost_schedule, min_cost_m = pp_config_pair


                self.rank0_log(f'Use {pp_config_pair} as the optimal m for pipeline parallelism during {"Training" if self.training else "Inference"}.')
                self.rank0_log(f'Fwd{" + Bwd" if self.training else ""} Time cost: {sum(time_dict[pp_config_pair]) / len(time_dict[pp_config_pair]):.2f}')

                if self.training:
                    self.training_pipeline_parallel_schedule = min_cost_schedule
                self.set_pp_schedule(min_cost_schedule)
                self._m_microbatch_dict[self.training] = min_cost_m
                self.set_micro_batch_size(input_batch_size)
                if hasattr(self, 'optimizer_container'):
                    self.zero_grad()
                self.clear_states_after_auto_benchmark()
                return self.__call__(*args, **kwargs)


            assert hasattr(self, 'pp_args')
            assert hasattr(self, 'static_kwargs_keys')

            self._check_states()

            bs = self.get_batch_size(*args, **kwargs)
            self.set_resolution(self.get_resolution_key(*args, **kwargs), bs)

            self._check_states()

            static_kwargs = {}
            if len(args) > 0:
                import inspect
                # arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
                arg_names = get_arg_names(self.pp_models[0].forward)

                args_to_kwargs = {}
                for idx, k in enumerate(args):
                    arg_name = arg_names[self.n_pp_args + idx]
                    assert arg_name not in kwargs
                    args_to_kwargs[arg_name] = k

                kwargs.update(args_to_kwargs)

                args = []


            args, kwargs = self.pre_process_input(*args, **kwargs)
            self._check_input(*args, **kwargs)
            for k in self.static_kwargs_keys:
                if k in kwargs:
                    static_kwargs[k] = kwargs.pop(k)
            ret = self.forward(*args, static_kwargs=static_kwargs, **kwargs)
            return ret
        else:
            target = None
            if 'target' in kwargs:
                import inspect
                # arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
                arg_names = get_arg_names(self.pp_models[0].forward)
                if 'target' not in arg_names:
                    target = kwargs.pop('target')

            self.clear_cached_result()
            ret = self.forward(*args, **kwargs)
            ret = self.get_real_ret(ret)
            return ret # when pp is disabled, mimic the original behavior

            # if self.training:
            #     assert target is not None
            #     loss_fn_out = self.loss_fn(ret, target)
            #     if isinstance(self.loss_fn, LossFnWrapper):
            #         return self.loss_fn.loss_fn_ret
            #     return loss_fn_out
            # else:
            #     return ret

    def get_result_rank(self, stage_idx=-1):
        # assert len(self.pp_models) == 1
        return self.pp_mesh.mesh[stage_idx].item()


    def sync_pp_result(self, result, src_stage_idx=-1):
        if not self.enable_pp:
            return result
        from hymm.parallelism.utils import auto_broadcast
        return auto_broadcast(result, src=self.get_result_rank(src_stage_idx), group=self.pp_mesh.get_group())



    def forward(self, *args, static_kwargs=None, sync_pp_result=True, **kwargs):
        """
        如果enable_pp, forward 和 call 就是 forward + backward
        否则，forward 是 forward, call 是 forward + backward
        """
        ret = self._forward(*args, static_kwargs=static_kwargs, **kwargs)
        if self.enable_pp:
            real_ret = self.get_real_ret(ret)
            if sync_pp_result:
                return self.sync_pp_result(real_ret)
            else:
                return real_ret
        else:
            return ret

    def clear_cached_result(self):
        if not hasattr(self.fsdp_models[-1], '_pp_valid_result_keys'):
            return
        for valid_key in self.fsdp_models[-1]._pp_valid_result_keys:
            if hasattr(self.fsdp_models[-1], valid_key):
                getattr(self.fsdp_models[-1], valid_key).clear()

    def get_cached_result(self, key, sync_pp_result=True, merge_op='cat'):
        # Generally, only the last rank has cached results
        assert self.enable_pp
        key = f'_pp_result_cache_list_{key}'
        if not hasattr(self.fsdp_models[-1], '_pp_valid_result_keys') or key not in self.fsdp_models[-1]._pp_valid_result_keys:
            ret = None
        else:
            ret = getattr(self.fsdp_models[-1], key)
            ret = self.merge_microbatch_object_list(ret, merge_op=merge_op)
        if sync_pp_result:
            ret = self.sync_pp_result(ret)
        return ret


    def get_real_ret(self, step_ret):
        # assert self.enable_pp
        if self.enable_pp:
            if self.training:
                # scheduler.step_ret can be loss (if training) or forward return value (if eval)
                return self.get_cached_result('ret_val')
            else:
                # 推理情况 step_ret 和 self.merge_return_val(self.get_cached_return_val()) 有可能是一样的
                ret = self.get_cached_result('ret_val')
                if ret is None:
                    return step_ret
                return ret
        else:
            return step_ret

    def custom_merge_return_val(self, ret_val, merge_op=None):
        return ret_val

    def merge_microbatch_object_list(self, ret_val, merge_op='cat'):
        """
        test:
            ret_val = [torch.randn(1, 2) for _ in range(2)]
            print(ret_val)
            print(m.merge_return_val(ret_val))
            ret_val = [
                (1, torch.randn(1, 2), {'hh': torch.randn(3)}),
                (1, torch.randn(1, 2), {'hh': torch.randn(3)}),
            ]
            print(m.merge_microbatch_object_list(ret_val))
        """
        assert isinstance(ret_val, list)
        if len(ret_val) == 0: # not output rank
            return

        if len(ret_val) == self.m_microbatch + 1: # drop the shape inference result
            ret_val = ret_val[1:]
        assert len(ret_val) == self.m_microbatch, f'{len(ret_val)=} {self.m_microbatch=}'

        assert all([type(k) == type(ret_val[0]) for k in ret_val])
        if isinstance(ret_val[0], torch.Tensor):
            ret_val = [k for k in ret_val] # copy
            if len(ret_val[0].shape) == 0:
                ret = torch.stack(ret_val)
            else:
                ret = torch.cat(ret_val, dim=0)
            if merge_op == 'cat':
                return ret
            elif merge_op == 'mean':
                return ret.mean(0)
            elif merge_op == 'sum':
                return ret.sum(0)
            else:
                raise ValueError(f'Invalid merge_op: {merge_op}')
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
                raise ValueError(f'Inconsistent return value in different micro batch {ret_val[0]}')
        else:
            try:
                ret = self.custom_merge_return_val(ret_val, merge_op=merge_op)
                if ret == ret_val:
                    raise ValueError(f'Unsupported type {type(ret_val[0])}')
                else:
                    return ret
            except Exception as e:
                raise ValueError(f'Unsupported type {type(ret_val[0])}') from e




    def _forward(self, *args, static_kwargs=None, **kwargs):
        if static_kwargs is None:
            static_kwargs = {}



        if not self.enable_pp:
            assert len(self.fsdp_models) == 1
            return self.fsdp_models[0](*args, **kwargs, **static_kwargs)

        assert 'losses' not in kwargs

        assert 'target' not in static_kwargs
        assert 'losses' not in static_kwargs


        pp_schedule = self.pp_schedule
        if self.training:
            if 'target' in kwargs:
                targets, losses = (
                    (kwargs['target'], []) if pp_schedule.has_last_stage else (None, None)
                )
                del kwargs['target']
            else:
                targets, losses = (
                    (torch.empty([self.batch_size, 1]), []) if pp_schedule.has_last_stage else (None, None)
                )
        else:
            targets, losses = (None, None)

        self.clear_cached_result()
        extra_context = nullcontext()
        if not self.training:
            extra_context = torch.no_grad()
        with torch.amp.autocast('cuda', enabled=False), extra_context:
            if pp_schedule.has_first_stage:
                if self.n_pp_args > 0:
                    assert len(args) == 0, 'args should not be parsed in the first stage to avoid duplicate arguments.'
                    ret = pp_schedule.step(*args, **kwargs, **static_kwargs, target=targets, losses=losses)
                else:
                    # TODO: 这种实现很容易漏参数，一不小心传入的参数在后面的stage被扔掉，改用默认值而不报错
                    #   未来 去掉 statickwargs, 设置 replacable 参数 从 kwargs 里丢掉
                    import inspect
                    # arg_names = inspect.getfullargspec(self.pp_models[0].forward)[0]
                    arg_names = get_arg_names(self.pp_models[0].forward)
                    new_args = []
                    for i in range(len(arg_names)):
                        if arg_names[i] in kwargs:
                            new_args.append(kwargs.pop(arg_names[i]))
                        else:
                            # 当有参数 missing 的时候，也不能再传 args 了
                            assert self.static_kwargs_keys
                            break

                    ret = pp_schedule.step(*new_args, **static_kwargs, target=targets, losses=losses)
            else:
                ret = pp_schedule.step(**static_kwargs, target=targets, losses=losses)


        if self.training:

            loss = (
                # not to use mean, because gradients are already scaled in ModelToLossFn
                torch.sum(torch.stack(losses)).to(self.device)
                if pp_schedule.has_last_stage
                else torch.tensor([-1.0], device=self.device)
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
            assert isinstance(stage, PipelineStage)
            cls = type(stage)
            new_stage = cls(
                stage.submod,
                stage_index=stage.stage_index,
                num_stages=stage.num_stages,
                device=stage.device,
                group=stage.group,
                dw_builder=stage.dw_builder
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

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            self.device,
            group=self.pp_mesh.get_group(),
        )
        return stage, model



    def get_forward_func(self):
        return self.forward

    @abstractmethod
    def pipeline_manual_split(self):
        pass


    def load_full_sd_by_cvt_to_dcp(self, sd_path):
        import tempfile
        if dist.get_node_local_rank() == 0:
            td = tempfile.TemporaryDirectory()
            checkpoint_manager.torch_state_dict_to_dcp(
                dcp_save_path=td.name,
                sd_path=sd_path
            )
            loguru.logger.debug(f'Checkpoint conversion (sd -> dcp) is finished. Saving to {td.name} (original {sd_path})')
            sd_path = td.name
        else:
            sd_path = None

        from hymm.parallelism import utils
        n_node = dist.get_world_size() // 8
        node_mesh = init_device_mesh('cuda', (n_node, 8,), mesh_dim_names=('nodes', 'gpus'))
        sd_path = utils.auto_broadcast(sd_path, group_src=0, group=node_mesh['gpus'].get_group())
        bc = utils.auto_broadcast(sd_path, group_src=0, group=node_mesh['gpus'].get_group())
        # dist.get_process_group_ranks(node_mesh['gpus'].get_group())
        dist.barrier()
        # self.load_checkpoint(td.name, load_optimizer_states=False)
        self.model_checkpoint_manager.load_from_path(sd_path)


    def load_checkpoint(self,
                        load_dir,
                        tag=None,  # unused
                        load_module_strict=True,  # unused
                        load_optimizer_states=None, # True False None
                        load_lr_scheduler_states=True,
                        load_module_only=False,  # unused
                        custom_load_fn=None,  # unused
                        ):
        """
        2 种输入:
            1. torch ckpt, 先转dcp再读，适应已经切分好的模型
            2. dcp 目录, 下面必须要有 'weights', 'optimizer' 这种文件夹
                如果下面没有这个，而只有distcp, 则建议用 checkpoint_manager.load_ckpt
        """
        from pathlib import Path
        path = Path(load_dir)
        if tag is not None:
            path = path / tag
        if path.is_dir():
            # self.load_dcp_state_dict(os.path.join(load_dir, self.MODEL_FOLDER))

            weight_path = Path(path) / self.MODEL_FOLDER
            if not weight_path.exists():
                assert not load_optimizer_states
                weight_path = Path(path)
                assert len(list(weight_path.glob('*.distcp'))) > 0
            self.model_checkpoint_manager.load_from_path(str(weight_path))



            if os.path.isdir(os.path.join(path, self.OPTIMIZER_FOLDER)) and load_optimizer_states is not False:
                try:
                    self.optimizer_checkpoint_manager.load_from_path(os.path.join(path, self.OPTIMIZER_FOLDER))
                except:
                    if load_optimizer_states is True:
                        raise
            else:
                if load_optimizer_states is True:
                    msg = f'The user specifies load_optimizer_states=True, but {os.path.join(path, self.OPTIMIZER_FOLDER)} does not exist'
                    loguru.logger.error(msg)
                    raise FileNotFoundError(msg)
            if os.path.isdir(os.path.join(path, self.TRAINING_STATES_FOLDER)):
                self.training_states_checkpoint_manager.load_from_path(os.path.join(path, self.TRAINING_STATES_FOLDER))
        else:
            self.load_full_sd_by_cvt_to_dcp(path)
            assert not load_optimizer_states
        return path, self.training_states

    def save_checkpoint(
            self, save_dir=None, tag=None,  client_state={},
            # unused
            save_latest=True, exclude_frozen_parameters=False,
            save_optimizer_states=True,
    ):
        if save_dir is not None:
            if self.ckpt_dir is not None:
                model_old_folder = self.model_checkpoint_manager.folder
                if hasattr(self, 'optimizer_checkpoint_manager'):
                    optimizer_old_folder = self.optimizer_checkpoint_manager.folder
                states_old_folder = self.training_states_checkpoint_manager.folder
            self.model_checkpoint_manager.folder = save_dir
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.folder = save_dir
            self.training_states_checkpoint_manager.folder = save_dir



        if tag is None:
            tag = 'latest'


        self.reshard()
        self.model_checkpoint_manager.save(step_or_tag=tag)
        if save_optimizer_states:
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.save(step_or_tag=tag)
        if isinstance(client_state, dict):
            old_state = self.training_states.copy()
            self.training_states.update(client_state)
        else:
            raise NotImplementedError
        self.training_states_checkpoint_manager.save(step_or_tag=tag)
        if isinstance(client_state, dict):
            self.training_states.clear()
            self.training_states.update(old_state)


        if save_dir is not None and self.ckpt_dir is not None:
            self.model_checkpoint_manager.folder = model_old_folder
            if hasattr(self, 'optimizer_checkpoint_manager'):
                self.optimizer_checkpoint_manager.folder = optimizer_old_folder
            self.training_states_checkpoint_manager.folder = states_old_folder


    @torch.no_grad()
    def get_global_grad_norm(self) -> float:
        log_once(
            'Using `model_engine.get_global_grad_norm` is not recommended. Here we just follow the ugly Deepspeed API and provide a naive implementation',
            'WARNING',
        )
        parameters = self.parameters()
        norm_type: float = 2.0
        error_if_nonfinite: bool = False
        foreach: bool | None = None

        if self.parallel_dims.ep_enabled:
            foreach = False # Avoid cross mesh computation

            # TODO: full tensor is retrieved in advance to avoid cross mesh computation, this could lead to speed issues
            grads = [p.grad.full_tensor() for p in parameters if p.grad is not None]
        else:
            grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.

            total_norm = total_norm.full_tensor()

        return total_norm.item()

    def get_fsdp_model(self):
        if len(self.fsdp_models) == 1:
            return self.fsdp_models[0]
        return nn.ModuleList(self.fsdp_models)
    @property
    def module(self):
        if len(self.fsdp_models) == 1:
            return self.fsdp_models[0]
        return nn.ModuleList(self.fsdp_models)



class ParallelEngine(BaseParallelEngine):
    def pipeline_manual_split(self):

        layer_prefix="transformer.h."
        assert isinstance(self.recursive_get_attr(self.model, layer_prefix), nn.ModuleList)

        splits = self.generate_split_points(
            len(self.recursive_get_attr(self.model, layer_prefix)),
            layers_preffix=layer_prefix,
        )

        num_stages = len(splits) + 1

        stages = []
        models = []

        for stage_idx in self.stage_ids_this_rank(num_stages):
            start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = self.build_stage(
                splits,
                layer_prefix,
                stage_idx,
                num_stages,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            self.logger.info(
                f"PP rank {self.pp_rank} is building stage_idx {stage_idx} ({num_stages=})"
                f" with start_layer {start_layer}, stop_layer {stop_layer}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models

    def config_forward_args(self):
        """
        pp 有两种实现模式：
            1. replaceable args: 一个block返回的可以直接替换下一次forward的前几个输入
                这个模式下，pp_args为空, 最好把非replaceable args都放在static_kwargs_keys中
            2. pp_args: 一个bloc返回的作为 pp_args, 原本的输入不变
                这个模式下，pp_args不为空, 所有参数通过 kwargs 传递，有没有 static 无所谓
        """
        self.set_static_kwargs_keys([])
        self.set_n_pp_args(0)
