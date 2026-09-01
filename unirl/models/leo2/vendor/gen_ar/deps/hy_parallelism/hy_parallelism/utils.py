# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

# from torch.distributed.pipelining import ScheduleZBVZeroBubble
from collections import defaultdict
import loguru
from functools import lru_cache
import contextlib
from contextlib import contextmanager
import math
import os
from collections.abc import Generator, Iterable
from datetime import timedelta

from typing import Optional, Any

import loguru
import torch
from torch import nn
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from loguru import logger
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from hy_parallelism.parallel_states import get_parallel_state
from hy_parallelism.distributed.communications import *


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
            # logger.debug(f"Single-process job using seed: {seed}")
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
        # logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

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
def _get_total_norm(
    tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach = None,
) -> torch.Tensor:
    """copied from torch.nn.utils.get_total_norm"""


    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@torch.no_grad()
def _clip_grads_with_norm_(
    parameters,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    """copied from torch.nn.utils.clip_grads_with_norm_"""

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


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
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if "ep" in p.device_mesh.mesh_dim_names:
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, error_if_nonfinite, foreach
    )
    # ep_grads may be an empty list, in which case get_total_norm returns tensor(0.), a non-DTensor
    # This can occur in PP + EP setups where certain PP ranks only own non-EP layers, for instance.
    if isinstance(ep_grads_total_norm, DTensor):
        ep_grads_total_norm = ep_grads_total_norm.full_tensor()

    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    ).full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = (
            ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        )
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm

@torch.no_grad()
def get_global_grad_norm_by_mesh_old_ep(
    param_groups_by_mesh: list[list[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Implementation that avoids cross mesh computation.
    """
    grads_by_meshes = [[] for _ in range(len(param_groups_by_mesh))]
    expert_grads = []
    for i, params in enumerate(param_groups_by_mesh):
        for p in params:
            if p.grad is not None:
                if p.is_expert:
                    expert_grads.append(p.grad)
                else:
                    grads_by_meshes[i].append(p.grad)

    norms = []
    for i, grads in enumerate(grads_by_meshes):
        if len(grads) == 0:
            continue
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        norms.append(total_norm)
    
    if len(expert_grads) > 0: # TODO: 这个 if 在一些特殊情况有可能导致通信不一致，未来fix下
        expert_total_norm = torch.nn.utils.get_total_norm(
            expert_grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(expert_total_norm, DTensor):
            expert_total_norm = expert_total_norm.full_tensor()
        
        if math.isinf(norm_type):
            dist.all_reduce(expert_total_norm, op=dist.ReduceOp.MAX, group=get_parallel_state().ep_mesh.get_group())
        else:
            expert_total_norm **= norm_type
            dist.all_reduce(expert_total_norm, op=dist.ReduceOp.SUM, group=get_parallel_state().ep_mesh.get_group())
            expert_total_norm **= 1.0 / norm_type
        norms.append(expert_total_norm)

    if len(norms) == 0: # Defensive programming
        return torch.tensor(0.0)
    else:
        if math.isinf(norm_type):
            raise NotImplementedError
            # total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
        else:
            total_norm = sum(norm**norm_type for norm in norms)
            total_norm **= 1.0 / norm_type

        
        if get_parallel_state().pp_enabled:
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / norm_type
        return total_norm

@torch.no_grad()
def get_global_grad_norm_by_mesh(
    param_groups_by_mesh: list[list[torch.Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    log_once(
        "Using `model_engine.get_global_grad_norm` is not recommended. Here we just follow the ugly Deepspeed API and provide a naive implementation",
        "WARNING",
    )
    norm_type = 2.0
    error_if_nonfinite = False
    foreach = None

    grads_by_meshes = [[] for _ in range(len(param_groups_by_mesh))]
    for i, params in enumerate(param_groups_by_mesh):
        for p in params:
            if p.grad is not None:
                grads_by_meshes[i].append(p.grad)

    norms = []
    for i, grads in enumerate(grads_by_meshes):
        if len(grads) == 0:
            continue
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        norms.append(total_norm)

    if len(norms) == 0: # Defensive programming
        return torch.tensor(0.0)
    else:
        if math.isinf(norm_type):
            # total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
            raise NotImplementedError
        else:
            total_norm = sum(norm**norm_type for norm in norms)
            total_norm **= 1.0 / norm_type

        if get_parallel_state().pp_enabled:
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / norm_type

        return total_norm

def clip_grad_norm_by_mesh_old_ep_(
    param_groups_by_mesh: list[list[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    total_norm = get_global_grad_norm_by_mesh_old_ep(
        param_groups_by_mesh, norm_type, error_if_nonfinite, foreach, pp_mesh
    )
    for param in param_groups_by_mesh:
        torch.nn.utils.clip_grads_with_norm_(param, max_norm, total_norm, foreach)
    
    return total_norm

@torch.no_grad()
def clip_grad_norm_by_mesh_(
    param_groups_by_mesh: list[list[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Implementation that avoids cross mesh computation.
    """
    grads_by_meshes = [[] for _ in range(len(param_groups_by_mesh))]
    for i, params in enumerate(param_groups_by_mesh):
        for p in params:
            if p.grad is not None:
                grads_by_meshes[i].append(p.grad)

    norms = []
    for i, grads in enumerate(grads_by_meshes):
        if len(grads) == 0:
            continue
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()
        norms.append(total_norm)

    if len(norms) == 0: # Defensive programming
        return torch.tensor(0.0)
    else:
        if math.isinf(norm_type):
            raise NotImplementedError
            # total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
        else:
            total_norm = sum(norm**norm_type for norm in norms)
            total_norm **= 1.0 / norm_type

        
        if get_parallel_state().pp_enabled:
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / norm_type

        for param in param_groups_by_mesh:
            torch.nn.utils.clip_grads_with_norm_(param, max_norm, total_norm, foreach)
        
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

        # TODO: full tensor is retrieved in advance to avoid cross mesh computation, this could lead to speed and memory issues
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



from typing import *
import numpy as np
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state


def _collect_rng_states(include_cuda: bool = True, local_only: bool = False) -> dict[str, Any]:
    r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "python": python_get_rng_state(),
    }

    states["numpy"] = np.random.get_state()
    if include_cuda:
        if local_only:
            states["torch.cuda"] = [torch.cuda.get_rng_state() for _ in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        else:
            states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states


def _set_rng_states(rng_state_dict: dict[str, Any], local_only: bool = False) -> None:
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
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
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
    states = _collect_rng_states(include_cuda=include_cuda)
    try:
        yield
    finally:
        _set_rng_states(states)


@contextmanager
def hook_safe_forward(model, forward_func_name, refresh_fn=None):
    """
    >>> with hook_safe_forward(model, 'infer_forward', refresh_fn=self.force_refresh_pipeline_scheduler):
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

def batch_obj_to_tensor(batch, max_len=1000):
    obj_tensors = []
    for sample in batch:
        obj_tensors.append(obj_to_tensor(sample, max_len=max_len))
    return stack_tensor(obj_tensors)

def batch_tensor_to_obj(tensor):
    return [tensor_to_obj(sample) for sample in tensor]

def stack_tensor(tensors):
    # return torch.nested.nested_tensor(tensors, dtype=tensors[0].dtype)
    return torch.stack(tensors)


def _object_to_tensor(obj, device, group):
    import io
    import pickle
    _pickler = pickle.Pickler


    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size

def _tensor_to_object(tensor, tensor_size, group):
    import pickle
    import io

    _unpickler = pickle.Unpickler
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()



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
    from hy_parallelism.utils import isolate_rng
    import torch
    torch.manual_seed(1)
    with isolate_rng():
        print([torch.randn(1, device='cuda') for _ in range(3)])
    print(torch.randn(1, device='cuda'))

    with isolate_rng():
        print([torch.randn(1) for _ in range(3)])
    print(torch.randn(1))

def get_stack():
    import sys
    frame = sys._getframe()
    s = ''
    while frame:
        s = s + f"File: {frame.f_code.co_filename}, Line: {frame.f_lineno}, Function: {frame.f_code.co_name}" + '\n'
        frame = frame.f_back
    return s

def is_recomputing():  # Only supports reentrant=False
    stack = get_stack()
    return 'recompute_fn' in stack and 'unpack_hook' in stack


def is_pp_SI(): # for debuging gradient checkpoiting
    stack = get_stack()
    return 'shape_inference' in stack

def barrier():
    parallel_dims = get_parallel_state()
    assert parallel_dims is not None
    sync_groups = []
    group_names = []
    if parallel_dims.tp_enabled:
        group_names.append('tp')
        sync_groups.append(parallel_dims.tp_group)
    if parallel_dims.sp_enabled:
        group_names.append('sp')
        sync_groups.append(parallel_dims.sp_group)
    if parallel_dims.ep_enabled:
        group_names.append('ep')
        sync_groups.append(parallel_dims.ep_group)
    for group in sync_groups:
        dist.barrier(group)

def sync_random_states(parallel_dims=None):
    if parallel_dims is None:
        parallel_dims = get_parallel_state()
    rng_states = _collect_rng_states(local_only=True)
    rng_states = sync_object_for_parallel_training(rng_states, parallel_dims=parallel_dims, force_object=True)
    _set_rng_states(rng_states, local_only=True)


def get_global_batch_size(local_batch_size):
    return sum(gather_obj(local_batch_size, get_parallel_state().dp_mesh.get_group()))
    

def split_data_in_data_shared_group(data):
    non_dp_size = get_parallel_state().non_data_parallel_size
    non_dp_rank = get_parallel_state().non_dp_rank
    chunk_list = torch.chunk(data, non_dp_size, dim=0)
    if non_dp_rank < len(chunk_list):
        return chunk_list[non_dp_rank]
    else:
        return None


def gather_data_in_data_shared_group(data):
    non_dp_group = get_parallel_state().non_dp_group
    # TODO: optimize all_gather, support None
    tensor_list = all_gather_tensor(data, group=non_dp_group)
    tensor_list = [k.cuda() for k in tensor_list if k is not None]
    return torch.cat(tensor_list, dim=0).cuda()


def _format_keys(keys, depth=5, prefix=None):
    from collections import defaultdict
    dic = defaultdict(list)
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start
    for k in keys:
        pos = find_nth(k, '.', depth)
        dic[k[:pos]].append(k)
    ret = []
    n = 5
    for k, v in dic.items():
        if len(v) < n:
            ret += v
        else:
            ret += v[:n]
            ret += [k + f': [{len(v) - n} more params]']
    if prefix:
        ret = [f'{prefix}: {s}' for s in ret]
    return '\n'.join(ret)

def get_missing_unexpected_str(missing, unexpected, loaded=None, tag=''):
    if loaded is None:
        loaded = []
    if tag:
        tag = f"{tag}: "
    if loaded:
        loaded_str = f"======== [Loaded keys ({len(loaded)} params)] =======:\n"
        loaded_str += format_keys(loaded, prefix='Loaded')
        loaded_str += "\n"
    else:
        loaded_str = ""
    return (
        f'{loaded_str}'
        f"======== [Missing keys ({len(missing)} params)] =======:\n"
        f"{format_keys(missing, prefix='Missing')}\n"
        f"======== [Unexpected keys ({len(unexpected)} params)] =======:\n"
        f"{format_keys(unexpected, prefix='Unexpected')}"
        f'\n======== [{tag}Total keys num: {len(missing) + len(unexpected) + len(loaded)}, {len(missing)} missing, {len(unexpected)} unexpected' + (f', {len(loaded)} loaded' if loaded else '') + '] ======='

    )

def format_keys(keys, lines=500, prefix=None):
    if os.environ.get('HY_PARALLELISM_DISABLE_FORMAT_KEYS', '0') == '1':
        return '\n'.join(keys)
    depths = list(reversed(range(1, 10)))
    for depth in depths:
        ret = _format_keys(keys, depth=depth, prefix=prefix)
        # print(f"current lines for depth {depth} is {(ret.count('\n'))}")
        if (ret.count('\n')) <= lines:
            return ret
    return _format_keys(keys, depth=5, prefix=prefix)

def get_module_dtype(module):
    """
    Get dtype information from a module wrapped by fully_shard.
    
    Returns:
        dict: A dictionary with:
            - 'is_consistent': bool - True if all parameters have consistent 
              master_dtype and param_dtype across all FSDP modules
            - 'master_dtype': torch.dtype | None - The original dtype before 
              fully_shard wrapping (from _orig_dtype)
            - 'param_dtype': torch.dtype | None - The param_dtype from mp_policy
    """
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
    if not isinstance(module, FSDPModule):
        try:
            first_master_dtype = next(module.parameters()).dtype
        except StopIteration:
            return {
                'is_consistent': True,
                'master_dtype': None,
                'param_dtype': None,
                'is_fsdp': False,
            }
        for param in module.parameters():
            if param.dtype != first_master_dtype:
                return {
                    'is_consistent': False,
                    'master_dtype': first_master_dtype,
                    'param_dtype': param.dtype,
                    'is_fsdp': False,
                }
        return {
            'is_consistent': True,
            'master_dtype': first_master_dtype,
            'param_dtype': first_master_dtype,
            'is_fsdp': False,
        }
    
    # Track the first encountered dtype values
    first_master_dtype = None
    first_param_dtype = None
    found_any = False
    
    def collect_fsdp_dtypes(module):
        """Recursively collect dtype information from all FSDP modules."""
        nonlocal first_master_dtype, first_param_dtype, found_any
        
        # Check if this module has FSDP state
        fsdp_state = _get_module_fsdp_state(module)
        if fsdp_state is not None and fsdp_state._fsdp_param_group is not None:
            param_group = fsdp_state._fsdp_param_group
            try:
                param_group._init_mp_dtypes()
            except Exception as e:
                raise ValueError(f'Error initializing mp_dtypes for {module.__class__.__name__}: {e}')
            master_dtype = param_group._orig_dtype
            if master_dtype is None: # no trainable params
                return None
            # Get param_dtype from mp_policy
            param_dtype = param_group.mp_policy.param_dtype
            
            if not found_any:
                # First FSDP module encountered
                first_master_dtype = master_dtype
                first_param_dtype = param_dtype
                found_any = True
            else:
                # Check consistency with previous modules
                if master_dtype != first_master_dtype or param_dtype != first_param_dtype:
                    # Found inconsistency, return immediately
                    return {
                        'is_consistent': False,
                        'master_dtype': first_master_dtype,
                        'param_dtype': first_param_dtype,
                        'is_fsdp': True,
                    }
                # Also check that master_dtype matches param_dtype
                if first_master_dtype != first_param_dtype:
                    return {
                        'is_consistent': False,
                        'master_dtype': first_master_dtype,
                        'param_dtype': first_param_dtype,
                        'is_fsdp': True,
                    }
        
        # Recursively check submodules
        for submodule in module.modules():
            if submodule is not module:  # Avoid double-checking the current module
                result = collect_fsdp_dtypes(submodule)
                if result is not None:  # Early stop signal
                    return result
        
        return None  # Continue traversal
    
    # Start traversal
    early_stop_result = collect_fsdp_dtypes(module)
    if early_stop_result is not None:
        return early_stop_result
    
    # All modules checked and consistent (or no FSDP modules found)
    if not found_any:
        return {
            'is_consistent': False,
            'master_dtype': None,
            'param_dtype': None,
            'is_fsdp': False,
        }
    
    # All consistent
    return {
        'is_consistent': True,
        'master_dtype': first_master_dtype,
        'param_dtype': first_param_dtype,
        'is_fsdp': True,
    }

def collect_model_dtype(model, master_dtype_map, param_dtype_map, fqn='root'):
    res = get_module_dtype(model)
    if res['is_consistent']:
        master_dtype_map[res['master_dtype']].append(fqn)
        if res['is_fsdp']:
            param_dtype_map[res['param_dtype']].append(fqn)
        return
    else:
        # master_dtype_map[res['master_dtype']].append(fqn)
        if res['is_fsdp']:
            param_dtype_map[res['param_dtype']].append(fqn)
        for name, child in model.named_children():
            collect_model_dtype(child, master_dtype_map, param_dtype_map, fqn=fqn + '.' + name)


def _format_param_count(numel):
    """Format parameter count with appropriate unit (K, M or B)."""
    if numel < 1e3:
        return f"{numel:.0f}"
    elif numel < 1e6:
        return f"{numel/1e3:.2f} K"
    elif numel < 1e9:
        return f"{numel/1e6:.2f} M"
    else:
        return f"{numel/1e9:.2f} B"


def print_model_info(model, tag=""):
    total_numel = 0
    dense_numel = 0
    moe_numel = 0
    trainable_numel = 0
    local_numel = 0
    memory_consumption = 0
    buffer_numel = 0
    for name, param in model.named_parameters():
        total_numel += param.numel()
        if "expert" in name:
            moe_numel += param.numel()
        else:
            dense_numel += param.numel()
        if param.requires_grad:
            trainable_numel += param.numel()
        if isinstance(param, DTensor):
            local_numel += param.to_local().numel()
            memory_consumption += param.to_local().numel() * param.to_local().element_size()
            # print(f"param {name} is {param.to_local().numel() * param.to_local().element_size() / 1024**3:.2f} GB. global {param.numel() * param.element_size() / 1024**3:.2f} GB")
        else:
            local_numel += param.numel()
            memory_consumption += param.numel() * param.element_size()
            # print(f"param {name} is {param.numel() * param.element_size() / 1024**3:.2f} GB")
    
    for name, buf in model.named_buffers():
        buffer_numel += buf.numel()
        memory_consumption += buf.numel() * buf.element_size()
    try:
        master_dtype_map = defaultdict(list)
        param_dtype_map = defaultdict(list)
        collect_model_dtype(model, master_dtype_map, param_dtype_map)

        dtype_info_str = ''
        for dtype, fqns in master_dtype_map.items():
            if dtype is not None:
                dtype_info_str += f"Master dtype: {dtype} is used in {fqns} \n"
        for dtype, fqns in param_dtype_map.items():
            if dtype is not None:
                dtype_info_str += f"Param dtype: {dtype} is used in {fqns} \n"
    except Exception as e:
        dtype_info_str = f'Error collecting model dtype: {e}\n'


    model_info_str = (
        f"\n============================={tag}========================================="[:-len(tag)] + "\n"
        f"Total number of parameters: {_format_param_count(total_numel)} (Incorrect if using old MOE implementation with EP enabled) \n"
        f"Dense number of parameters: {_format_param_count(dense_numel)} \n"
        f"MoE number of parameters: {_format_param_count(moe_numel)} (counted by 'expert' in param_name)\n"
        f"Buffer numel: {_format_param_count(buffer_numel)}\n"
        f"Trainable number of parameters: {_format_param_count(trainable_numel)} \n"
        f"-------------------------------------------------------------------\n"
        f"{dtype_info_str}"
        f"-------------------------------------------------------------------\n"
        f"Local number of parameters: {_format_param_count(local_numel)} \n"
        f"-------------------------------------------------------------------\n"
        f"Param Memory consumption: {memory_consumption / (1024 ** 3):.2f} GB \n"
        f"======================================================================\n"
    )


    loguru.logger.debug(
        model_info_str
    )

def early_binding_closure(closure, locals, late_binding_keys=None, on_snapshot=None, on_restore=None, return_infos=False):
    if closure.__closure__ is None: # may only use global vars
        raise ValueError("Only supports closure input.")
    import inspect
    co_freevars = closure.__code__.co_freevars
    closure_vars = inspect.getclosurevars(closure)
    contents = []

    # loguru.logger.debug(f"co_freevars: {co_freevars}")
    # non_locals = {k:type(v) for k, v in closure_vars.nonlocals.items()}
    # loguru.logger.debug(f"closure_vars.nonlocals: {non_locals}")


    for k in co_freevars:
        if on_snapshot is not None:
            contents.append(on_snapshot(closure_vars.nonlocals[k]))
        else:
            contents.append(closure_vars.nonlocals[k])

    def wrapped_closure(*args, **kwargs):
        for i in range(len(closure.__closure__)):
            if late_binding_keys is not None and co_freevars[i] in late_binding_keys:
                continue
            if on_restore is not None:
                closure.__closure__[i].cell_contents = on_restore(contents[i])
            else:
                closure.__closure__[i].cell_contents = contents[i]

 
        ret = closure(*args, **kwargs)


        if locals is not None:
            new_locals = {}
            for i in range(len(closure.__closure__)):
                k = co_freevars[i]
                new_locals[k] = closure.__closure__[i].cell_contents

            # Find values in new_locals that are different from locals (before update)
            changed_locals = {}
            for k, v in new_locals.items():
                if k not in locals or locals[k] is not v:
                    changed_locals[k] = v

            locals.update(new_locals)

        if return_infos:
            assert locals is not None
            return ret, {
                'changed_locals': changed_locals,
                'co_freevars': co_freevars,
            }
        return ret

    return wrapped_closure


class EarlyBindingClosureList:
    def __init__(self, on_snapshot=None, on_restore=None):
        self.closures = []
        self.on_snapshot = on_snapshot
        self.on_restore = on_restore

    def add(self, closure, locals):
        self.closures.append(early_binding_closure(closure, locals, self.on_snapshot, self.on_restore, return_infos=True))

    def execute(self, i):
        ret, infos = self.closures[i]()
        # 更新后面的closure的binding
        for j in range(i+1, len(self.closures)):
            if hasattr(self.closures[j], '__closure__'):
                changed_locals = infos['changed_locals']
                co_freevars = infos['co_freevars']
                for local_idx, k in enumerate(co_freevars):
                    if k in changed_locals:
                        self.closures[j].__closure__[local_idx].cell_contents = changed_locals[k]
        return ret


def early_binding_closure_with_FunctionType(f):
    import types
    if f.__closure__ is None:
        return f
    new_cells = tuple(types.CellType(c.cell_contents) for c in f.__closure__)
    return types.FunctionType(
        f.__code__,
        f.__globals__,
        f.__name__,
        f.__defaults__,
        # f.__kwdefaults__,
        new_cells,
    )

def context_manager_to_decorator(context_manager):
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with context_manager():
                return func(*args, **kwargs)
        return wrapper
    return decorator

def get_taiji_user() -> str:
    ssh_global_onion_infomation = os.environ.get('SSH_GLOBAL_ONION_INFOMATION', '')
    # login user
    if ssh_global_onion_infomation:
        try:
            return ssh_global_onion_infomation[1:ssh_global_onion_infomation.index(',')]
        except:
            pass
    # machine owner
    return os.environ.get('RTX_NAME', '')


def replace_module(model, is_target_module:Callable, get_alternative:Callable):
    def __replace_module(model, full_name):
        for name, child in model.named_children():
            if is_target_module(full_name, child):
                new_module = get_alternative(full_name, child)
                setattr(model, name, new_module)
            __replace_module(child, full_name + '.' + name)
    __replace_module(model, '')

class GeneralObject(int):
    def __init__(self):
        super().__init__()
        self.list_val = []
        self.dict_val = {}
        self.object_type = None

    def append(self, val):
        if self.object_type is None:
            self.object_type = list
        if self.object_type != list:
            raise TypeError("GeneralObject is not a list")
        self.list_val.append(val)

    def __getitem__(self, key):
        if self.object_type is None:
            self.object_type = dict
        if self.object_type == dict:
            if key not in self.dict_val:
                self.dict_val[key] = GeneralObject()
        return self.dict_val[key]

    def __setitem__(self, key, val):
        if self.object_type is None:
            self.object_type = dict
        if self.object_type == list:
            self.list_val[key] = val
        elif self.object_type == dict:
            self.dict_val[key] = val

    def __contains__(self, key):
        if self.object_type is None:
            return False
        elif self.object_type == list:
            return key in self.list_val
        elif self.object_type == dict:
            return key in self.dict_val

    def __str__(self):
        if self.object_type is None:
            return str(int(self))
        elif self.object_type == list:
            return str(self.list_val)
        elif self.object_type == dict:
            return str(self.dict_val)

    def __repr__(self):
        return self.__str__()

global_state = GeneralObject()