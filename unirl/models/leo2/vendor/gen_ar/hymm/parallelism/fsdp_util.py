import loguru
from sympy import use
import torch
import torch.nn as nn
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)

from functools import partial

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import functools


non_reentrant_wrapper = partial(
    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT,
)


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=True,
    )
    return mixed_precision


def get_dit_fsdp_kwargs(
    transformer,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    from hymm.utils.load import get_no_split_modules
    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        from peft.utils.other import fsdp_auto_wrap_policy
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "use_orig_params": True, 
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update(
            {
                "use_orig_params": False,  # Required for LoRA memory savings
                "sync_module_states": True,
            }
        )

    return fsdp_kwargs, no_split_modules


def get_dit_fsdp_kwargs_v2(
    transformer,
    parallel_dims,
    world_mesh,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    from hymm.utils.load import get_no_split_modules
    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        from peft.utils.other import fsdp_auto_wrap_policy
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)


    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD       # zero3 within each node 
            #sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2 # zero2 within each node 
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)
            sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        dp_mesh_dim_names = ("dp_replicate", )
        sharding_strategy = ShardingStrategy.NO_SHARD #DDP
        
    # Check if the DeviceMesh has mesh_dim_names before slicing
    if hasattr(world_mesh, "mesh_dim_names") and world_mesh.mesh_dim_names is not None:
        device_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    else:
        # Fallback: Use the world_mesh as is without slicing
        device_mesh = world_mesh
        print(f"Warning: Cannot slice DeviceMesh without mesh_dim_names. Using unsliced mesh.")

    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        #"device_id": torch.cuda.current_device(),
        "device_mesh": device_mesh,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
        "use_orig_params": True, 
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update(
            {
                "use_orig_params": False,  # Required for LoRA memory savings
                "sync_module_states": True,
            }
        )

    return fsdp_kwargs, no_split_modules


def get_discriminator_fsdp_kwargs(
    parallel_dims,
    world_mesh,
    sharding_strategy,
    cpu_offload=False,
    master_weight_type="fp32",
):
    auto_wrap_policy = None

    # Use existing mixed precision settings

    mixed_precision = get_mixed_precision(master_weight_type)
    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if parallel_dims.dp_shard_enabled or parallel_dims.sp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_sp")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            dp_mesh_dim_names = ("dp_shard_sp",)
            sharding_strategy = ShardingStrategy.FULL_SHARD

    # Check if the DeviceMesh has mesh_dim_names before slicing
    if hasattr(world_mesh, "mesh_dim_names") and world_mesh.mesh_dim_names is not None:
        device_mesh = world_mesh[tuple(dp_mesh_dim_names)]
    else:
        # Fallback: Use the world_mesh as is without slicing
        device_mesh = world_mesh
        print(f"Warning: Cannot slice DeviceMesh without mesh_dim_names. Using unsliced mesh.")

    cpu_offload = (
        torch.distributed.fsdp.CPUOffload(offload_params=True) if cpu_offload else None
    )
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        # "device_id": device_id,
        "device_mesh": device_mesh,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }

    return fsdp_kwargs


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def apply_fsdp2(
    model: nn.Module,
    blocks,
    default_fsdp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    ep_enabled: bool,
    cpu_offload: bool = False,
    expert_fsdp_mesh: DeviceMesh=None,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        default_fsdp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    # Check if dp_mesh is a 0-dimensional mesh and convert to 1D if needed
    if hasattr(default_fsdp_mesh, "ndim") and default_fsdp_mesh.ndim == 0:
        print(f"Warning: Converting 0-dimensional DeviceMesh to 1D mesh for fully_shard compatibility")
        # Create a 1D mesh from the device in the 0D mesh
        device = default_fsdp_mesh.device_type
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Create a 1D mesh with proper dimension name
        mesh_shape = (world_size,)
        mesh_dim_names = ("dp",)
        default_fsdp_mesh = DeviceMesh(device, torch.arange(world_size).reshape(mesh_shape), mesh_dim_names=mesh_dim_names)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype,
        # output_dtype=param_dtype
    )
    fsdp_config = {"mesh": default_fsdp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    for layer_id, transformer_block in enumerate(blocks):
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
        
        if ep_enabled:
            for fqn, module in transformer_block.named_modules():
                if model.is_expert(fqn):
                    expert_fsdp_config = fsdp_config.copy()
                    expert_fsdp_config.update({"mesh": expert_fsdp_mesh})
                    fully_shard(
                        module,
                        **expert_fsdp_config,
                        reshard_after_forward=reshard_after_forward,
                    )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    return fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)