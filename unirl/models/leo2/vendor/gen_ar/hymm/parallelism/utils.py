# from torch.distributed.pipelining import ScheduleZBVZeroBubble

from functools import lru_cache
import contextlib
from contextlib import contextmanager
import math
import os
from collections.abc import Generator, Iterable
from datetime import timedelta

import loguru
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from loguru import logger
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend

from hymm.parallelism.parallel_states import get_parallel_state


# loguru.logger.remove(None)
# loguru.logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:^8}</level> | <level><bold>[Rank " + os.environ['RANK'] + "]: {message}</bold></level> (<cyan>{file}:{line}</cyan>)",)
def _dist_reduce(
        x: torch.Tensor,
        reduceOp: str,
        mesh: DeviceMesh,
        extra_pg: dist.ProcessGroup | None = None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
        x: torch.Tensor,
        mesh: DeviceMesh,
        extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
        x: torch.Tensor,
        mesh: DeviceMesh,
        extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
        world_mesh: DeviceMesh | None,
        device: torch.device,
        seed: int | None = None,
        deterministic: bool = False,
        distinct_seed_mesh_dim: str = "pp",
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimension denoted by `distinct_seed_mesh_dim`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # Set distinct seed for each rank in mesh dimensions, with dimension name provdied by `distinct_seed_mesh_dim`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # TODO(jianiw): We could further extend this to support mutiple distinct dimensions instead of just one.
    if (
            c10d.get_world_size() > 1
            and distinct_seed_mesh_dim in world_mesh.mesh_dim_names
    ):
        distinct_mesh = world_mesh[distinct_seed_mesh_dim]
        seed += distinct_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(
            f"{distinct_seed_mesh_dim} rank {distinct_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}"
        )
        duplicate_seed_mesh = list(
            filter(
                lambda name: name != distinct_seed_mesh_dim, world_mesh.mesh_dim_names
            )
        )
        duplicate_seed_mesh = (
            world_mesh[duplicate_seed_mesh] if len(duplicate_seed_mesh) else None
        )
    else:
        duplicate_seed_mesh = world_mesh
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)


def create_context_parallel_ctx(
        cp_mesh: DeviceMesh,
        cp_buffers: list[torch.Tensor],
        cp_seq_dims: list[int],
        cp_no_restore_buffers: set[torch.Tensor],
        cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(
        enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Generator[None, None, None]:
    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                if SDPBackend.MATH in ScaledDotProductAttention.backends:
                    ScaledDotProductAttention.backends.remove(SDPBackend.MATH)
                assert (
                    ScaledDotProductAttention.backends
                ), "No valid SDPA backends with CP."
                stack.enter_context(cp_context)

            yield

    return context


def init_distributed(job_config):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    def _get_distributed_backend(job_config):
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map:
            backend = torch.distributed.Backend.default_device_backend_map.get(
                device_type
            )
        if job_config.training.enable_cpu_offload:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
    TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(job_config.comm.trace_buf_size))
    if job_config.comm.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = f"{job_config.job.dump_folder}/comm_trace"
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(job_config),
        timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
    )



# hardcoded BF16 type peak flops for NVIDIA A100, H100, H200, B200 GPU and AMD MI250, MI300X, AMD MI325X and Intel PVC
def get_peak_flops(device_name: str) -> int:
    import subprocess
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 4.5e15
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name:
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12

def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


from torch._utils import _get_available_device_type, _get_device_module

def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()
def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


@torch.no_grad()
def get_grad_norm(
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
        pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    assert len(grads) > 0
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    return total_norm

@torch.no_grad()
def clip_grad_norm_(
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
        pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if get_parallel_state().ep_enabled:
        foreach = False # Avoid cross mesh computation

        # TODO: full tensor is retrieved in advance to avoid cross mesh computation, this could lead to speed issues
        grads = [p.grad.full_tensor() for p in parameters if p.grad is not None]
    else:
        grads = [p.grad for p in parameters if p.grad is not None]


    assert len(grads) > 0
    # TODO: expert parallelism is not considered
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

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def gather_obj(obj, group=None):
    if group is None:
        ws = dist.get_world_size()
    else:
        ws = dist.get_world_size(group)
    lst = [None for _ in range(ws)]
    dist.all_gather_object(lst, obj, group=group)
    return lst


def sync_object_for_parallel_training(object, inplace=False, debug_with_check=False, trace='arg0'):
    parallel_dims = get_parallel_state()
    sync_groups = []
    group_names = []
    if parallel_dims.tp_enabled:
        group_names.append('tp')
        sync_groups.append(parallel_dims.tp_mesh.get_group())
    if parallel_dims.pp_enabled:
        group_names.append('pp')
        sync_groups.append(parallel_dims.pp_mesh.get_group())
    if parallel_dims.sp_enabled:
        group_names.append('sp')
        sync_groups.append(parallel_dims.sp_mesh.get_group())

    if isinstance(object, torch.Tensor):
        assert object.device.type == 'cuda'

        if inplace:
            original_object_id = id(object)
            assert object.is_contiguous()
        else:
            object = object.contiguous()

        if debug_with_check:
            for group_id, group in enumerate(sync_groups):
                shapes = gather_obj(object.shape, group=group)
                dtypes = gather_obj(object.dtype, group=group)
                assert all(x == shapes[0] for x in shapes), f'checking group {group_id} ({group_names[group_id]} group), {trace} has different shapes ({shapes})'
                assert all(y == dtypes[0] for y in dtypes), f'checking group {group_id} ({group_names[group_id]} group), {trace} has different dtypes ({dtypes})'

                buffer = [torch.empty_like(object) for _ in range(group.size())]
                dist.all_gather(buffer, object, group=group)
                assert all(torch.allclose(x, buffer[0]) for x in buffer), f'checking group {group_id} ({group_names[group_id]} group), {trace} not equal'


        for group in sync_groups:
            dist.broadcast(object, group_src=0, group=group)

        if inplace:
            assert id(object) == original_object_id

    elif isinstance(object, (int, float, str)):
        if debug_with_check:
            for group in sync_groups:
                buffer = [None] * group.size()
                dist.all_gather_object(buffer, object, group=group)
                assert all(x == buffer[0] for x in buffer)

        buffer = [object]
        for group in sync_groups:
            dist.broadcast_object_list(buffer, group_src=0, group=group)
        object = buffer[0]
    elif isinstance(object, list) or isinstance(object, tuple):
        object = list(object)
        for i, obj in enumerate(object):
            object[i] = sync_object_for_parallel_training(obj, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'list[{i}].')
    elif isinstance(object, dict):
        for key, value in object.items():
            object[key] = sync_object_for_parallel_training(value, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'dict[{key}].')
    elif isinstance(object, set):
        new_set = set()
        for i, item in enumerate(object):
            new_set.add(sync_object_for_parallel_training(item, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'set[{i}].'))
        object = new_set
    elif object == None:
        return None
    else:
        raise NotImplementedError(f"Unsupported type {type(object)}")
    return object


def map_tensor(obj, func):
    if isinstance(obj, (int, str, bool, float)):
        return obj
    elif isinstance(obj, torch.Tensor):
        # return obj.to(torch.cuda.current_device())
        return func(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = map_tensor(v, func=func)
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj, tuple):
            obj = list(obj)
        for i in range(len(obj)):
            obj[i] = map_tensor(obj[i], func=func)
    elif isinstance(obj, set):
        new_set = set()
        for item in obj:
            new_set.add(map_tensor(item, func=func))
        obj = new_set
    else:
        raise ValueError(f"Unsupported type {type(obj)}")
        # assert_no_tensor(obj, f'Syncing {type(obj)} could lead to potential device mismatch. e.g. getting stuck when trying to conduct gathering or reducing')
    return obj

def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise NotImplementedError

def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        device=device,
        # async_op=async_op,
    )
    buffer = [obj] if is_src(src, group_src, group) else [None]

    # loguru.logger.debug(f'broadcast_object: {buffer=}, {kwargs}')
    dist.broadcast_object_list(buffer, **kwargs)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group, )
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group, )

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, **kwargs)
    return buffer

def auto_broadcast(
        obj,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    # obj: None or list/dict/tensor or basic type
    obj_type = type(obj)
    obj_type = broadcast_object(obj_type, src=src, group_src=group_src, group=group, )

    if obj_type == torch.Tensor:
        return broadcast_tensor(obj, **kwargs)
    elif obj_type == list or obj_type == tuple:
        if is_src(src, group_src, group):
            length = len(obj)
        else:
            length = None
        length = broadcast_object(length, src=src, group_src=group_src, group=group, )
        if not is_src(src, group_src, group):
            obj = [None] * length
        return [auto_broadcast(x, **kwargs) for x in obj]
    elif obj_type == dict:
        if is_src(src, group_src, group):
            keys = list(obj.keys())
        else:
            keys = None
        def get_value(key):
            if is_src(src, group_src, group):
                return obj[key]
            else:
                return None
        keys = broadcast_object(keys, src=src, group_src=group_src, group=group, )
        return {k: auto_broadcast(get_value(k), **kwargs) for k in keys}
    else:
        return broadcast_object(obj, src=src, group_src=group_src, group=group, )


from typing import *
import numpy as np
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state


def _collect_rng_states(include_cuda: bool = True) -> dict[str, Any]:
    r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "python": python_get_rng_state(),
    }

    states["numpy"] = np.random.get_state()
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states


def _set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    r"""Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])

    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextlib.contextmanager
def isolate_rng() -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Example:
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1, device='cuda') for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])
    """
    states = _collect_rng_states()
    yield
    _set_rng_states(states)


@contextmanager
def hook_safe_forward(model, forward_func_name, refresh_fn=None):
    """
    >>> with hook_safe_forward(model, 'infer_forward'):
    >>>     model(...) # will call model.infer_forward(...) instead of model.forward(...)
    """
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, forward_func_name)
    old_forward = model.forward
    model.forward = getattr(model, forward_func_name)
    if refresh_fn:
        refresh_fn()
    yield
    if refresh_fn:
        refresh_fn()
    # ret = model(*args, **kwargs)
    model.forward = old_forward
    # return ret


def register_hook_safe_forward(model, forward_func_name, refresh_fn):
    # this is not reversible
    def rename_func(func_name):
        return func_name + '_hook_safe_backup'

    new_name = rename_func(forward_func_name)
    setattr(model, new_name, getattr(model, forward_func_name))

    def new_forward(self, *args, **kwargs):
        with hook_safe_forward(self, new_name, refresh_fn):
            return self(*args, **kwargs)  # use __call__

    setattr(model, forward_func_name, new_forward)

def obj_to_tensor(obj, max_len=1000):
    import pickle
    ret = torch.tensor(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
    real_length = ret.shape[0]
    if real_length < max_len:
        ret = torch.cat([
            torch.tensor([real_length], dtype=torch.uint8),
            ret,
            torch.zeros(max_len - real_length - 1, dtype=torch.uint8)
        ])
    else:
        raise ValueError(f'obj_to_tensor: obj is too large, real_length={real_length}, max_len={max_len}')
    return ret


def tensor_to_obj(tensor):
    import pickle
    real_length = tensor[0].item()
    tensor = tensor[1:real_length + 1]
    return pickle.loads(tensor.cpu().numpy())

def batch_obj_to_tensor(batch):
    obj_tensors = []
    for sample in batch:
        obj_tensors.append(obj_to_tensor(sample))
    return stack_tensor(obj_tensors)

def batch_tensor_to_obj(tensor):
    return [tensor_to_obj(sample) for sample in tensor]

def stack_tensor(tensors):
    # return torch.nested.nested_tensor(tensors, dtype=tensors[0].dtype)
    return torch.stack(tensors)



def __position(depth=0):
    import sys
    frame = sys._getframe(depth + 1)
    file = frame.f_code.co_filename
    line = frame.f_lineno
    return f'{file}:{line}'
@lru_cache
def __log_once(msg, level, position):
    loguru.logger.opt(depth=2).log(level, msg)

def log_once(msg, level='INFO'):
    __log_once(msg, level, __position(1))

if __name__ == '__main__':
    from hymm.parallelism.utils import isolate_rng
    import torch
    torch.manual_seed(1)
    with isolate_rng():
        print([torch.randn(1, device='cuda') for _ in range(3)])
    print(torch.randn(1, device='cuda'))

    with isolate_rng():
        print([torch.randn(1) for _ in range(3)])
    print(torch.randn(1))

