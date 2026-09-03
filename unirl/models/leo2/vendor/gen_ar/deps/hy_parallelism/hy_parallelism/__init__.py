"""
Environment variables (prefix: HY_PARALLELISM)
================================================

This project reads the following `HY_PARALLELISM_*` environment variables:

- `HY_PARALLELISM_DEBUG` (default: `0`)
  - Enable extra debug logs in logging/checkpoint/MoE paths.
  - In this package init, it also enables memory snapshot profiling hooks.

- `HY_PARALLELISM_ALWAYS_PRINT_EP_MSG` (default: `0`)
  - In DeepSeek MoE expert dispatch debug path, always print expert-token diagnostics
    (otherwise it prints only on abnormal dispatch shapes).

- `HY_PARALLELISM_DISABLE_FORMAT_KEYS` (default: `0`)
  - Disable hierarchical key formatting and print raw key lists directly when reporting
    model-state missing/unexpected keys.

- `HY_PARALLELISM_DEFAULT_LOGGING_COLOR` (default: `0`)
  - Keep default loguru colors (do not apply custom color override for local user setup).

- `HY_PARALLELISM_LOGGING_DISABLE_FILENAME_FLUSH` (default: `0`)
  - Disable right-aligned `(file:line)` flush logic in logger formatting; useful when
    logs are captured by systems such as Ray.

- `HY_PARALLELISM_ENABLE_JVP` (default: `0`)
  - When TorchTitan is available, enables extra MoE JVP-related monkey patch.

- `HY_PARALLELISM_PATCH_TORCH_LOAD` (default: `0`)
  - Enable monkey patch for `torch.load`: resolve input path via
    `bing_utils.get_nonlocal_file(...)` before actual loading.

- `HY_PARALLELISM_SYNC_INPUT` (default: `0`)
  - When set to `1`, enable `sync_input` in `ParallelEngine.__call__` if neither
    an explicit `sync_input` argument nor `set_sync_input` was used.

- `HY_PARALLELISM_PACK_CHECKPOINT_FOR_INFERENCE` (default: `0`)
  - When set to `1`, pack checkpoint for inference.

- `HY_PARALLELISM_FUSED_SORT_CHUNKS` (default: `0`)
  - 不开启 deepep 时， fused sort chunks 的开关

- ENABLE_HY_PARALLELISM_LOGGING
  - 开启 hy_parallelism 的日志格式

- `HY_PARALLELISM_DEEPEP_FAST_EXPERT_SORT`
  - 开启 deepep 时， fast expert sort 的开关

- `HY_PARALLELISM_DEEPEP_ASYNC_FINISH` (default: `1`)
  - 开启 deepep 时， dispatch/combine 的 async_finish 通信-计算重叠

- `NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default: `10240`)
  - 开启 deepep LL 时， num_max_dispatch_tokens_per_rank 的默认值

- `HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM` (default: `0`)
  - 开启 cutlass grouped GEMM 的开关
"""

# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from .version import __version__

import os
# try:
#     if 'CUDA_DEVICE_MAX_CONNECTIONS' not in os.environ or int(os.environ['CUDA_DEVICE_MAX_CONNECTIONS']) <= 1:
#         os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '64'
# except ValueError:
#     os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '64'

from .cpu_affinity import bind_cpu_for_local_rank
import atexit
import torch
import time
import loguru
from packaging import version
from hy_parallelism.utils import get_taiji_user
from .failure_state_hooks import register_failure_state_hooks

if get_taiji_user() == 'kevinkhwu':
    if 'ENABLE_HY_PARALLELISM_LOGGING' not in os.environ:
        os.environ['ENABLE_HY_PARALLELISM_LOGGING'] = '1'

# if version.parse(torch.__version__) <= version.parse('2.9.0'):
#     raise RuntimeError(
#         f'PyTorch version {torch.__version__} is not supported. Please upgrade to PyTorch 2.9.1 or later. '
#         f'You can also consider installing a older version of hy_parallelism via `pip3 install hy-parallelism==0.3.0 --index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --force-reinstall`'
#     )

from . import monkey_patches

__all__ = [
    'checkpoint_manager',
    # 'parallel_engine',
    'parallel_utils',
    'fsdp_util',
    'get_logger',
    'increase_non_torch_allocator_buffer',
    'decrease_non_torch_allocator_buffer',
    'set_non_torch_allocator_buffer',
]



from . import utils as parallel_utils
# from .engines import parallel_engine
from .distributed import fsdp_util
from .checkpoint import checkpoint_manager




def debug_distributed(TORCH_CPP_LOG_LEVEL="INFO", TORCH_DISTRIBUTED_DEBUG="DETAIL"):
    os.environ["TORCH_CPP_LOG_LEVEL"] = TORCH_CPP_LOG_LEVEL
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = TORCH_DISTRIBUTED_DEBUG

try:
    from pathlib import Path
    rank_file_dir = "/tmp/torchrun"
    Path(rank_file_dir).mkdir(parents=True, exist_ok=True)
    (Path(rank_file_dir) / f"{os.environ.get('LOCAL_RANK', 0)}").write_text(f'{os.getpid()}\n')
except Exception:
    pass  # Ignore race condition

if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
    from hy_parallelism.tools.profiling import maybe_enable_memory_snapshot
    context = maybe_enable_memory_snapshot()
    context.__enter__()
    atexit.register(context.__exit__, None, None, None)





from .common.logging import get_logger, configure_logger_level


def _cuda_device(device=None):
    if device is None:
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    return device


def _get_memory_fraction(device=None):
    device = _cuda_device(device)
    getter = getattr(torch.cuda, 'get_per_process_memory_fraction', None)
    if getter is None:
        getter = torch.cuda.memory.get_per_process_memory_fraction
    return getter(device)


def _buffer_ratio(ratio=None, n_gb=None, device=None) -> float:
    if (ratio is None) == (n_gb is None):
        raise ValueError('Exactly one of ratio or n_gb must be provided')
    if ratio is not None:
        return float(ratio)
    device = _cuda_device(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    return float(n_gb) * (1024 ** 3) / total_mem


def increase_non_torch_allocator_buffer(ratio=None, n_gb=None, device=None) -> float:
    r"""Reserve more GPU memory for non-torch allocators by decreasing torch memory fraction.

    Provide exactly one of ``ratio`` (fraction of total GPU memory) or ``n_gb`` (GiB).
    New fraction = current fraction - delta.
    """
    device = _cuda_device(device)
    delta = _buffer_ratio(ratio=ratio, n_gb=n_gb, device=device)
    fraction = max(0.0, min(1.0, _get_memory_fraction(device) - delta))
    torch.cuda.set_per_process_memory_fraction(fraction, device)

    # total_mem = torch.cuda.get_device_properties(device).total_memory
    # loguru.logger.info(f'increase_non_torch_allocator_buffer: {fraction=} {delta=} {ratio=} {n_gb=} {total_mem=}')
    return fraction


def decrease_non_torch_allocator_buffer(ratio=None, n_gb=None, device=None) -> float:
    r"""Reserve less GPU memory for non-torch allocators by increasing torch memory fraction.

    Provide exactly one of ``ratio`` (fraction of total GPU memory) or ``n_gb`` (GiB).
    New fraction = current fraction + delta.
    """
    device = _cuda_device(device)
    delta = _buffer_ratio(ratio=ratio, n_gb=n_gb, device=device)
    fraction = max(0.0, min(1.0, _get_memory_fraction(device) + delta))
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    loguru.logger.info(f'decrease_non_torch_allocator_buffer: {fraction=}')
    return fraction


def set_non_torch_allocator_buffer(ratio=None, n_gb=None, device=None) -> float:
    r"""Set the GPU memory reserved for non-torch allocators.

    Provide exactly one of ``ratio`` (fraction of total GPU memory) or ``n_gb`` (GiB).
    Torch memory fraction is set to ``1 - reserved``.
    """
    device = _cuda_device(device)
    reserved = _buffer_ratio(ratio=ratio, n_gb=n_gb, device=device)
    fraction = max(0.0, min(1.0, 1.0 - reserved))
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return fraction


if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        set_non_torch_allocator_buffer(ratio=0.02)
        # set_non_torch_allocator_buffer(ratio=0.20) # h800 test on h20
        # import gc
        # gc.disable()
    except Exception as e:
        pass
register_failure_state_hooks()
