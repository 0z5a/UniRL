# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import signal
import time
import loguru
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.cuda import nvtx
from torch.profiler import record_function
# from torch.autograd.profiler import emit_nvtx
# from torch.autograd.profiler import profile as profile_autograd
# from torch.cuda.profiler import profile as profile_cuda

# from torchtitan.config import Profiling as ProfilingConfig
from dataclasses import dataclass
from loguru import logger


@dataclass
class ProfilingConfig:
    enable_profiling: bool = True
    """Whether to enable pytorch profile"""

    save_traces_folder: str = "profile_traces"
    """Trace files location"""

    profile_freq: int = 1
    """How often to collect profile traces, in iterations"""

    profiler_active: int = 1
    """
    The steps profiler is active for.

    This is used to configure torch.profile.schedule.
    """

    profiler_warmup: int = 0
    """
    The number of warmup steps before the active step in each profiling cycle.

    This is used to configure torch.profile.schedule.
    """

    enable_memory_snapshot: bool = False
    """Whether to dump memory snapshot"""

    save_memory_snapshot_folder: str = "memory_snapshot"
    """Memory snapshot files location"""


# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 10000000
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

def start_memory_snapshot():
    if torch.cuda.is_available():
        from hy_parallelism.tools.profiling import MEMORY_SNAPSHOT_MAX_ENTRIES
        torch.cuda.memory._record_memory_history(
            max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES,
            stacks='python',
        )

def pause_memory_snapshot():
    # torch.cuda.memory._record_memory_history(context=None)
    torch.cuda.memory._record_memory_history(enabled=None)

def trace_handler(prof: torch.profiler.profile):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"memory_ana/autocast_fp16_new{timestamp}"
    # prof.export_chrome_trace(f"{file_name}.json")
    prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")
# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         on_trace_ready=trace_handler,
# ) as prof:

_global_prof = None
_global_context = None
def start_profiling(*args, **kwargs):
    global _global_prof, _global_context
    if _global_prof is not None: # 只能 start 一次。。
        return
    _global_context = maybe_enable_profiling(*args, **kwargs)
    _global_prof = _global_context.__enter__()

def stop_profiling():
    global _global_prof, _global_context
    if _global_context is not None:
        _global_context.__exit__(None, None, None)
    _global_prof = None
    _global_context = None

def is_profiler_enabled() -> bool:
    return _global_prof is not None


def profiler_step(skip_barrier: bool = False):
    if _global_prof is not None:
        torch.cuda.synchronize()
        _global_prof.step()
        if not skip_barrier:
            with profile_range("profiler_step_barrier"):
                torch.cuda.synchronize()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig|None=None,
    *,
    global_step: int = 0,
    base_folder: str = "torch_profile",
    leaf_folder: str = "",
):
    """
    from hy_parallelism.tools.profiling import maybe_enable_profiling, ProfilingConfig
    profiling_config = ProfilingConfig(enable_profiling=True)
    profiler_context = maybe_enable_profiling(
        profiling_config,
    )
    with profiler_context as prof:
        for batch in dataloader:
            train_step()
            prof.step()
    """
    if profiling_config is None:
        profiling_config = ProfilingConfig(enable_profiling=True)

    Path(base_folder).mkdir(parents=True, exist_ok=True)
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
        profile_freq, warmup, active = (
            profiling_config.profile_freq,
            profiling_config.profiler_warmup,
            profiling_config.profiler_active,
        )

        rank = int(os.environ.get('RANK', '0'))

        def trace_handler(prof):
            if prof.step_num == 1:
                loguru.logger.info(f"Profiling step 1, skip dumping traces")
                return
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json.gz")
            # print(prof.key_averages().table(row_limit=100))
            prof.export_chrome_trace(output_file)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds. (file: {Path(output_file).resolve()})"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        # elif torch.xpu.is_available():
        #     gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            # pyrefly: ignore [bad-argument-type]
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            profile_memory=False,
            on_trace_ready=trace_handler,
            record_shapes=False,
            with_stack=True,
            # with_modules=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            # with emit_nvtx():
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None

memory_profiler = None

@contextmanager
def maybe_enable_nsys_profile():
    yield
    # with profile_cuda():
    #     with emit_nvtx():
    #         yield


def memory_snapshot_function(*snapshot_args, **snapshot_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with maybe_enable_memory_snapshot(*snapshot_args, **snapshot_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextlib.contextmanager
def maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig=None,
    *,
    global_step: int = 0,
    base_folder: str = "./memory_snapshot",
    leaf_folder: str = "",
    stacks='python',
):
    if profiling_config is None:
        profiling_config = ProfilingConfig(enable_memory_snapshot=True)
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = os.path.join(
            base_folder, profiling_config.save_memory_snapshot_folder
        )
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get('RANK', '0'))

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES,
                    stacks=stacks,
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name, leaf_folder)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)


                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                print_memmory('Ready to dump memory')


                output_file = os.path.join(
                    curr_snapshot_dir, f"rank{rank}_memory_snapshot.pickle"
                )
                snapshot = torch.cuda.memory._snapshot()
                logger.info(f"Snapshot collected.")
                with open(output_file, "wb") as output:
                    pickle.dump(snapshot, output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds. (file: {Path(output_file).resolve()})"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        global memory_profiler
        profiler = MemoryProfiler(global_step, profiling_config.profile_freq)
        memory_profiler = profiler

        def oom_observer(device, alloc, device_alloc, device_free):
            print("saving allocated state during OOM")
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            pickle.dump(
                torch.cuda.memory._snapshot(),
                open(
                    f"{snapshot_dir}/oom_rank-{torch.distributed.get_rank()}.pickle",
                    "wb",
                ),
            )
        torch._C._cuda_attach_out_of_memory_observer(oom_observer)
        logger.info(f"Attach OOM observer")

        # SIGTERM/SIGINT/SIGABRT may not surface as a catchable Python exception
        # (e.g. external kill, C-level abort), so install handlers that dump the
        # snapshot before the process dies.
        _dump_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGABRT)

        def _signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name}, dumping memory snapshot before exit")
            try:
                profiler.step(exit_ctx=True)
            except Exception as exc:
                logger.error(f"Failed to dump memory snapshot on {sig_name}: {exc}")
            # Restore default handler and re-raise so the process actually terminates;
            # this also bypasses the finally below, avoiding a double dump.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        prev_handlers = {}
        for sig in _dump_signals:
            try:
                prev_handlers[sig] = signal.signal(sig, _signal_handler)
            except (ValueError, OSError) as exc:
                # signal.signal only works in the main thread of the main interpreter
                logger.warning(f"Could not register {signal.Signals(sig).name} handler: {exc}")

        try:
            yield profiler
        # except torch.OutOfMemoryError as e:
        except Exception as e:
            profiler.step(exit_ctx=True)
            raise
        finally:
            for sig, prev_handler in prev_handlers.items():
                signal.signal(sig, prev_handler)
            profiler.step(exit_ctx=True)
    else:
        yield None


@contextlib.contextmanager
def maybe_record_profile_range(name: str):
    # 这个目的是避免 profiler 开始之前，record_function
    if is_profiler_enabled():
        assert isinstance(name, str), f"name must be a string, but got {type(name)}. {name}"
        with record_function(name), nvtx.range(name):
            yield
    else:
        yield


@contextmanager
def profile_range(
    msg: str,
    enable_sync: bool = False,
    barrier: bool | torch.distributed.ProcessGroup = False,
    enable_time: bool = False,
):
    """
    enable_sync=True 确保 同步 cuda streams, 以及同步cuda和cpu
    barrier 同步不同的进程，避免进程快慢干扰正确执行时间
    """
    if not is_profiler_enabled() and not enable_time:
        barrier = False
        enable_sync = False

    if barrier is not None and barrier is not False:
        with maybe_record_profile_range(f'pre_barrier_of_{msg}'):
            if isinstance(barrier, torch.distributed.ProcessGroup):
                torch.distributed.barrier(barrier)
            else:
                torch.distributed.barrier()

    if enable_time:
        from hy_parallelism.bing_utils import Timer
        timer_context = Timer(msg, synchronized_time=enable_sync)
    else:
        timer_context = contextlib.nullcontext()

    if enable_sync:
        with maybe_record_profile_range(f'pre_sync_of_{msg}'):
            torch.cuda.synchronize()
    with maybe_record_profile_range(msg), timer_context:

        yield

        if enable_sync:
            with maybe_record_profile_range(f'post_sync_of_{msg}'):
                torch.cuda.synchronize()

    if barrier is not None and barrier is not False:
        with maybe_record_profile_range(f'post_barrier_of_{msg}'):
            if isinstance(barrier, torch.distributed.ProcessGroup):
                torch.distributed.barrier(barrier)
            else:
                torch.distributed.barrier()

def profile_func(*range_args, **range_kwargs):
    from functools import wraps
    import inspect

    def decorator(func):
        kwargs = dict(range_kwargs)
        if 'msg' not in kwargs:
            if range_args and isinstance(range_args[0], str):
                kwargs['msg'] = range_args[0]
            else:
                kwargs['msg'] = func.__qualname__

        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            with profile_range(**kwargs):
                return func(*args, **kwargs_inner)
        return wrapper

    if len(range_args) == 1 and not range_kwargs and callable(range_args[0]) and not inspect.isclass(range_args[0]):
        return decorator(range_args[0])
    return decorator

def profile_class(*range_args, **range_kwargs):
    from functools import wraps
    import inspect

    def _should_wrap(name, func):
        if name == 'forward':
            return False
        if name.startswith('__'):
            return False
        if inspect.isbuiltin(func):
            return False
        if getattr(func, '__module__', None) == 'builtins':
            return False
        return True

    def _wrap(name, func):
        kwargs = dict(range_kwargs)
        kwargs.setdefault('msg', name)
        return profile_func(**kwargs)(func)

    def _patch_module_instance(module):
        if getattr(module, "_hy_profile_instance_wrapped", False):
            return
        module._hy_profile_instance_wrapped = True
        module_name = getattr(module, "_hy_profile_module_name", module.__class__.__name__)
        kwargs = dict(range_kwargs)
        kwargs.setdefault("msg", module_name)
        if callable(getattr(module, "_call_impl", None)):
            module._call_impl = profile_func(**kwargs)(module._call_impl)
        for name, attr in module.__class__.__dict__.items():
            if not inspect.isfunction(attr) or not _should_wrap(name, attr):
                continue
            setattr(module, name, profile_func(**{**kwargs, "msg": f"{module_name}.{name}"})(getattr(module, name)))

    def decorator(cls):
        if cls.__dict__.get("_hy_profile_class_wrapped", False):
            return cls
        setattr(cls, "_hy_profile_class_wrapped", True)

        if issubclass(cls, torch.nn.Module):
            original_init = cls.__init__

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                for module_name, module in self.named_modules():
                    module._hy_profile_module_name = module_name
                    _patch_module_instance(module)

            cls.__init__ = wrapped_init
            return cls

        for name, attr in list(cls.__dict__.items()):
            if isinstance(attr, staticmethod):
                func = attr.__func__
                if _should_wrap(name, func):
                    setattr(cls, name, staticmethod(_wrap(name, func)))
            elif isinstance(attr, classmethod):
                func = attr.__func__
                if _should_wrap(name, func):
                    setattr(cls, name, classmethod(_wrap(name, func)))
            elif inspect.isfunction(attr) and _should_wrap(name, attr):
                setattr(cls, name, _wrap(name, attr))
        return cls

    if len(range_args) == 1 and inspect.isclass(range_args[0]) and not range_kwargs:
        return decorator(range_args[0])
    return decorator

def profile_range_func_wrapper(*range_args, **range_kwargs):
    return profile_func(*range_args, **range_kwargs)

def range_push(msg: str, enable_sync: bool = False):
    if enable_sync:
        with maybe_record_profile_range(f'pre_sync_of_{msg}'):
            torch.cuda.synchronize()
    if is_profiler_enabled():
        nvtx.range_push(msg)
        record_function_ctx = record_function(msg)
        record_function_list.append((record_function_ctx, msg))
        record_function_ctx.__enter__()
    else:
        record_function_list.append((None, msg))

record_function_list = []
def range_pop(enable_sync: bool = False):
    record_function_ctx, msg = record_function_list.pop()
    if enable_sync:
        with maybe_record_profile_range(f'post_sync_of_{msg}'):
            torch.cuda.synchronize()

    if is_profiler_enabled():
        nvtx.range_pop()
        record_function_ctx.__exit__(None, None, None)

def is_under_nsys_profile():
    import psutil
    try:
        # 获取当前Python进程对象
        current_process = psutil.Process(os.getpid())
        # 获取父进程（nsys profile启动时，父进程是nsys相关进程）
        parent_process = current_process.parent()

        # 检查父进程是否存在且名称/命令行包含nsys
        if parent_process:
            # 检查父进程名称（如nsys、nsys-profile）
            if "nsys" in parent_process.name().lower():
                return True
            # 检查父进程命令行（避免名称匹配不到的情况，如nsys profile xxx）
            cmdline = " ".join(parent_process.cmdline()).lower()
            if "nsys profile" in cmdline:
                return True

        # 辅助检查：NSYS相关环境变量（nsys可能会注入的环境变量）
        nsys_env_vars = [
            "NSYS_PROFILER_CONFIG",
            "NSYS_SESSION_ID",
            "NSYS_OUTPUT_FILENAME",
            "NVIDIA_NSYS_PROFILING_ENABLED"
        ]
        for env_var in nsys_env_vars:
            if env_var in os.environ:
                return True

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # 处理进程信息获取失败的异常（如权限不足）
        pass

    # 未检测到nsys profile环境
    return False


def print_memmory(tag='memory'):
    if not torch.cuda.is_available():
        return
    loguru.logger.info(f'[{tag}]: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}G | reserved={torch.cuda.memory_reserved() / 1024**3:.2f}G | peak={torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024**3:.2f}G')
    loguru.logger.info(f'[{tag}]: {torch.cuda.memory_summary()}')
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    s = torch.cuda.memory_stats()
    loguru.logger.info(f'[{tag}]: cudaMemGetInfo free/total(GB)', free/2**30, total/2**30)
    loguru.logger.info(f'[{tag}]: allocated/reserved(GB)',
        s["allocated_bytes.all.current"]/2**30,
        s["reserved_bytes.all.current"]/2**30)
    loguru.logger.info(f'[{tag}]: inactive_split_mb',
        s.get("inactive_split_bytes.all.current", 0)/2**20)

def memory_tag():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.zeros(10000, 10000).cuda()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def largest_cuda_tensors(n=30):
    import gc
    xs = []
    for o in gc.get_objects():
        try:
            if not torch.is_tensor(o):
                continue
            if not o.is_cuda:
                continue
            xs.append(o)
        except Exception:
            continue
    xs.sort(key=lambda t: t.numel() * t.element_size(), reverse=True)
    print(f"reachable cuda tensors: {len(xs)}")
    for t in xs[:n]:
        print(f"{t.numel()*t.element_size()/1024**3:.3f} GiB  shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")