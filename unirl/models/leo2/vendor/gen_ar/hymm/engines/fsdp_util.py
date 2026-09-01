import torch
import torch.nn as nn
from functools import partial
from typing import Any, Callable
from packaging import version

from hy_parallelism.parallel_states import get_parallel_state
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    fully_shard,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX  # noqa

from hymm.core.global_vars import get_logger, get_args


def print_gpu_memory(name):
    torch.distributed.barrier()
    memory = torch.cuda.memory_allocated() / (1024 ** 3)
    memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[rank {torch.distributed.get_rank()}] "
          f"{name:>30s}: Memory allocated: {memory:.2f} GB, Peak memory: {memory_peak:.2f} GB")


def _get_module_materialize_and_init_state(module: nn.Module):
    return getattr(module, "_materialize_and_init_state", None)


@torch.no_grad()
def _materialize_and_init_params(
        root_prefix: str | None,
        root_module: nn.Module,
        cpu_offload: bool = False,
        init_param_fn: Callable = None,
        init_buffer_fn: Callable = None,
):
    modules: list[tuple[str, nn.Module]] = []
    root_modules_set = {root_module}
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(prefix, module: nn.Module) -> None:
        """
        Runs a DFS to collect leaf modules, not recursing into modules with
        ``materialize and init`` already applied. A leaf module is defined as a module
        that does not contain any child modules.
        """
        if (
            module not in root_modules_set
            and _get_module_materialize_and_init_state(module) is not None
        ):
            return  # nested `materialize and init` module
        visited_modules.add(module)
        # Remove the checkpoint wrapper name from the prefix
        if prefix is not None:
            prefix = prefix.replace(_CHECKPOINT_PREFIX, "")
        # Recurse into children
        num_children = 0
        for sub_prefix, submodule in module.named_children():
            num_children += 1
            if prefix is not None:
                sub_prefix = f"{prefix}.{sub_prefix}"
            if submodule not in visited_modules:
                dfs(sub_prefix, submodule)
        # If the layer contains nn.Parameter or buffer, we also consider it as a leaf module
        num_cur_params = len(list(module.parameters(recurse=False)))
        num_cur_buffers = len(list(module.buffers(recurse=False)))
        # Only add leaf modules
        if num_children == 0 or num_cur_params > 0 or num_cur_buffers > 0:
            modules.append((prefix, module))

    for prefix, module in [(root_prefix, root_module)]:
        dfs(prefix, module)

    device = "cpu" if cpu_offload else "cuda"
    for prefix, module in modules:
        if hasattr(module, "reset_parameters"):
            module.to_empty(device=device, recurse=False)
            module.reset_parameters()
            for name, param in module.named_parameters(recurse=False):
                fqn = f"{prefix}.{name}"
                if init_param_fn is not None:
                    init_param_fn(fqn, param)
                # to_empty() creates new param objects, we assign name to param attributes for easy access.
                param._name_hymm_engine_ = f"{prefix}.{name}"
            # Also load persistent buffers (e.g. expert_bias) from checkpoint.
            # Non-persistent buffers are skipped as they are not saved in checkpoints.
            # Unlike parameters, buffers not found in checkpoint are silently skipped
            # because they may be deterministic (e.g. position_ids) and already
            # properly initialized by reset_parameters().
            for name, buf in module.named_buffers(recurse=False):
                if name in getattr(module, "_non_persistent_buffers_set", set()):
                    continue
                fqn = f"{prefix}.{name}"
                if init_buffer_fn is not None:
                    init_buffer_fn(fqn, buf)
        else:
            num_params = len(list(module.parameters()))
            if num_params == 0:
                continue
            raise RuntimeError(
                f"Leaf module {module} contains {num_params} parameters, but does not have `reset_parameters` method. "
                "Please provide an `reset_parameters` method to initialize parameters."
            )

    # Mark the module as materialized and initialized.
    # This set operation may trigger parameter rearrangement if needed.
    setattr(root_module, "_materialize_and_init_state", True)
    return root_module


def load_and_apply_fsdp2(
        model: nn.Module,
        default_fsdp_mesh: DeviceMesh,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        root_param_dtype: torch.dtype | None = None,
        root_reduce_dtype: torch.dtype | None = None,
        block_type: type | tuple[type] = None,
        blocks: list[nn.Module] = None,
        block_names: list[str] = None,
        cpu_offload: bool = False,
        expert_fsdp_mesh: DeviceMesh = None,
        reshard_after_forward_policy: str = "default",
        expert_gate_on_fp32: bool = True,  # deepep and ptm moe implement requires expert gate on fp32
        vit_on_fp32: bool = None,
        init_param_fn: Callable = None,
        init_buffer_fn: Callable = None,
        shard_placement_fn=None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply FSDP2 to.
        default_fsdp_mesh (DeviceMesh): The default FSDP mesh to use.
        param_dtype: The parameter type for MixedPrecisionPolicy.
        reduce_dtype: The reduce type for MixedPrecisionPolicy.
        block_type: The block type for apply FSDP2 to.
        blocks (list): List of blocks to apply FSDP2 to.
        block_names (list): List of block names corresponding to the blocks.
        cpu_offload (bool, optional): Whether to use CPU offloading. Defaults to False.
        expert_fsdp_mesh (DeviceMesh, optional): Expert FSDP mesh. Defaults to None.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
        expert_gate_on_fp32 (bool, optional): Whether to use fp32 for expert gate parameters. Defaults to True.
        init_param_fn (Callable, optional): Initialization function. Defaults to None.
    Returns:
        nn.Module: The model with FSDP2 applied.
    """
    if block_type is not None and blocks is not None:
        raise ValueError("block_type and blocks cannot be both specified.")
    if blocks is None:
        if block_type is not None:
            blocks, block_names = [], []
            for name, module in model.named_modules():
                if isinstance(module, block_type):
                    blocks.append(module)
                    block_names.append(name)
    else:
        if block_names is None:
            raise ValueError("block_names must be specified when blocks is specified.")
        assert len(blocks) == len(block_names), \
            f"blocks and block_names must have the same length, got {len(blocks)} and {len(block_names)}"

    pp_enabled = get_parallel_state().pp_enabled
    ep_enabled = get_parallel_state().ep_enabled
    logger = get_logger()
    args = get_args()

    if hasattr(default_fsdp_mesh, "ndim") and default_fsdp_mesh.ndim == 0:
        raise ValueError("0-dimensional DeviceMesh is not supported")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=args.fsdp_cast_forward_inputs,
        # output_dtype=param_dtype
    )
    fsdp_config: dict[str, Any] = {
        "mesh": default_fsdp_mesh,
        "mp_policy": mp_policy,
    }

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Initialize parameters if they are on 'meta' device
    has_meta = False
    for param in model.parameters():
        if param.device == torch.device('meta'):
            has_meta = True
            break
    if has_meta:
        _materialize_and_init = partial(
            _materialize_and_init_params, cpu_offload=cpu_offload, init_param_fn=init_param_fn,
            init_buffer_fn=init_buffer_fn)
    else:
        _materialize_and_init = lambda prefix, module: module

    for layer_id, (prefix, transformer_block) in enumerate(zip(block_names, blocks)):
        if transformer_block is None:
            continue
        if reshard_after_forward_policy == "always":
            reshard_after_forward = True
        elif reshard_after_forward_policy == "never":
            reshard_after_forward = False
        elif reshard_after_forward_policy == "default":
            if pp_enabled:
                # For PP, do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                reshard_after_forward = False
            else:
                # As an optimization, do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately
                reshard_after_forward = int(layer_id) < len(blocks) - 1
        else:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

        if expert_gate_on_fp32:
            for fqn, module in transformer_block.named_modules():
                if not fqn.endswith(".gate"):
                    continue

                gate_fsdp_config = fsdp_config.copy()
                ep_mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                    # output_dtype=param_dtype
                )
                gate_fsdp_config["mp_policy"] = ep_mp_policy
                fully_shard(
                    _materialize_and_init(f"{prefix}.{fqn}", module),
                    **gate_fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )

        if ep_enabled:
            assert expert_fsdp_mesh is not None, "expert_fsdp_mesh must be provided when ep_enabled is True."
            for fqn, module in transformer_block.named_modules():
                if not model.is_expert(fqn):    # 'experts' in fqn
                    continue

                expert_fsdp_config = fsdp_config.copy()
                expert_fsdp_config.update({"mesh": expert_fsdp_mesh})
                fully_shard(
                    _materialize_and_init(f"{prefix}.{fqn}", module),
                    **expert_fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
                if hasattr(module, 'set_gradient_divide_factor'):
                    module.set_gradient_divide_factor(
                        get_parallel_state().fsdp_gradient_divide_factor,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info(f" --> Applying FSDP to module: {prefix} "
                        f"{'(with loading pretrained weights if provided)' if has_meta else ''}...")
        fully_shard(
            _materialize_and_init(prefix, transformer_block),
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
            shard_placement_fn=shard_placement_fn,
        )

    if vit_on_fp32:
        if torch.distributed.get_rank() == 0:
            logger.info(f" --> Applying FSDP to vit "
                        f"{'(with loading pretrained weights if provided)' if has_meta else ''}...")
        vit_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32,
        )
        vit_fsdp_config = fsdp_config.copy()
        vit_fsdp_config["mp_policy"] = vit_mp_policy
        model_config = model.get_config()
        if model_config.use_vit:
            fully_shard(
                _materialize_and_init("vit", model.vit),
                **vit_fsdp_config,
                reshard_after_forward=False,
            )
        if model_config.use_vit_aligner:
            fully_shard(
                _materialize_and_init("vit_aligner", model.vit_aligner),
                **vit_fsdp_config,
                reshard_after_forward=False,
            )

    if torch.distributed.get_rank() == 0:
        logger.info(f" --> Applying FSDP to main model "
                    f"{'(with loading pretrained weights if provided)' if has_meta else ''}...")
    resolved_root_param_dtype = param_dtype if root_param_dtype is None else root_param_dtype
    resolved_root_reduce_dtype = reduce_dtype if root_reduce_dtype is None else root_reduce_dtype
    root_fsdp_config = fsdp_config.copy()
    root_fsdp_config["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=resolved_root_param_dtype,
        reduce_dtype=resolved_root_reduce_dtype,
        cast_forward_inputs=args.fsdp_cast_forward_inputs,
        # output_dtype=resolved_root_param_dtype
    )
    return fully_shard(
        _materialize_and_init(None, model),
        **root_fsdp_config,
        reshard_after_forward=False,
    )
