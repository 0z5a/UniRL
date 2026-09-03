# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os
from functools import partial
from collections import defaultdict
from ntpath import realpath
from typing import Callable
from packaging import version

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import Shard

import loguru

from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
    offload_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy
from hy_parallelism.common.logging import debug_log


from hy_parallelism.parallel_states import get_parallel_state
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper



def get_default_op_sac_context_fn(op_sac_save_list=None):
    """Get default context function for operator selective activation checkpointing.

    Args:
        op_sac_save_list (set[torch._ops.OpOverload], optional): The list of ops to save instead
            of recomputing. If None, uses default list.

    Returns:
        callable: A context function that creates selective checkpoint contexts with metadata tracking.
    """
    if op_sac_save_list is None:
        op_sac_save_list = {
            torch.ops.aten.mm.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_attention_math.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops._c10d_functional.all_to_all_single.default,
            # for low precision training, it's useful to always save
            # the result of max, since the absolute maximum is
            # used to compute the scaling factor for quantization.
            torch.ops.aten.max.default,
            torch._higher_order_ops.flex_attention,
        }
        if hasattr(torch._higher_order_ops, 'inductor_compiled_code'):
            op_sac_save_list.add(torch._higher_order_ops.inductor_compiled_code)

        if hasattr(torch.ops, 'deepep'):
            op_sac_save_list.add(torch.ops.deepep.dispatch.default)
            op_sac_save_list.add(torch.ops.deepep.combine.default)

    def _get_custom_policy(meta):
        """Create a custom policy function with metadata tracking.

        Args:
            meta (defaultdict): Dictionary to track metadata across forward/recompute passes.

        Returns:
            callable: Policy function that receives (ctx, func, *args, **kwargs).
        """
        def _custom_policy(ctx, func, *args, **kwargs):
            # Always save CPU offload operations
            if (
                func == torch.ops.aten._to_copy.default
                and len(args) > 0
                and hasattr(args[0], 'device')
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            # Track mm operations separately for forward vs recompute passes
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"

            # Track mm count for alternating pattern
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1

            # Saves output of all compute ops, except every second mm
            # This balances memory savings with recomputation overhead
            to_save = func in op_sac_save_list and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        """Create selective checkpointing contexts with fresh metadata for each checkpoint region.

        Returns:
            Context manager: Selective checkpointing contexts.
        """
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return selective_checkpointing_context_fn

def get_sub_fsdp_modules(block):
    ret = []
    for name, module in block.named_modules():
        if isinstance(module, FSDPModule) and module is not block:
            ret.append(module)
    return ret


def set_forward_prefetch(
    model,
    blocks,
    prefetch_factor=1,
    should_forward_prefetch: Callable[[str, nn.Module], bool] | None = None,
):
    if should_forward_prefetch is None:
        should_forward_prefetch = lambda fqn, module: True

    module_to_fqn = {module: fqn for fqn, module in model.named_modules()}
    transformer_blocks = list(blocks)

    for idx, block in enumerate(transformer_blocks):
        if block is None:
            continue

        real_prefetch_blocks = []
        for module in get_sub_fsdp_modules(block):
            if should_forward_prefetch(module_to_fqn[module], module):
                real_prefetch_blocks.append(module)

        for b in transformer_blocks[idx + 1 : idx + prefetch_factor + 1]:
            if b is None:
                continue
            if should_forward_prefetch(module_to_fqn[b], b):
                real_prefetch_blocks.append(b)

        block.set_modules_to_forward_prefetch(real_prefetch_blocks)

def apply_fsdp_checkpointing(
    model, no_split_modules, p=1, use_reentrant=False, enable_op_sac=False, op_sac_policy_fn=None,
    activation_offloading=False, activation_offload_list=None, activation_offload_pin_memory=True,
    defer_offload=False,
):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """
    Apply activation checkpointing to model
    returns None as model is updated directly

    Selectivity is defined as a percentage p, which means we apply ac
    on p of the total blocks. p is a floating number in the range of
    [0, 1].

    Some examples:
    p = 0: no ac for all blocks. same as `fsdp_activation_checkpointing=False`
    p = 1: apply ac on every block. i.e. "full ac".
    p = 1/2: [ac, no-ac, ac, no-ac, ...]
    p = 1/3: [no-ac, ac, no-ac,   no-ac, ac, no-ac,   ...]
    p = 2/3: [ac, no-ac, ac,    ac, no-ac, ac,    ...]
    Since blocks are homogeneous, we make ac blocks evenly spaced among
    all blocks.

    Args:
        use_reentrant (bool): Whether to use reentrant checkpointing.
            If the module contains custom kernels, we must set this to True.
        enable_op_sac (bool): Whether to use operator selective checkpointing.
            Sometimes makes the computation slower...

    Implementation:
    For a given ac ratio p, we should essentially apply ac on every "1/p"
    blocks. The first ac block can be as early as the 0th block, or as
    late as the "1/p"th block, and we pick the middle one: (0.5p)th block.
    Therefore, we are essentially to apply ac on:
    (0.5/p)th block, (1.5/p)th block, (2.5/p)th block, etc., and of course,
    with these values rounding to integers.
    Since ac is applied recursively, we can simply use the following math
    in the code to apply ac on corresponding blocks.
    """
    loguru.logger.info(f'Applying activation checkpointing. {use_reentrant=} {enable_op_sac=} {op_sac_policy_fn=}')
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

    wrapper_kwargs = {}
    if use_reentrant:
        wrapper_kwargs["checkpoint_impl"] = CheckpointImpl.REENTRANT
    else:
        wrapper_kwargs["checkpoint_impl"] = CheckpointImpl.NO_REENTRANT

    if activation_offloading:
        from hy_parallelism.training.checkpointing import (
            offload_checkpoint_fn,
            set_defer_offload,
        )
        set_defer_offload(defer_offload)
        assert not use_reentrant
        wrapper_kwargs["checkpoint_fn"] = partial(
            offload_checkpoint_fn,
            generator_kwargs={
                "pin_memory": activation_offload_pin_memory,
                "offload_list": activation_offload_list,
            },
            use_reentrant=use_reentrant,
        )

    checkpoint_wrapper_fn = partial(checkpoint_wrapper, **wrapper_kwargs)

    if enable_op_sac:
        assert not use_reentrant, "Reentrant checkpointing is not supported with operator selective checkpointing. Please set `use_reentrant=False` or disable operator selective checkpointing."
        if op_sac_policy_fn is None:
            # Get default context function that creates contexts with metadata tracking
            context_fn = get_default_op_sac_context_fn()
        else:
            # If custom policy function is provided, wrap it properly with metadata tracking
            def _get_custom_policy_with_meta(meta):
                def _custom_policy(ctx, func, *args, **kwargs):
                    return op_sac_policy_fn(ctx, func, *args, **kwargs)
                return _custom_policy

            def selective_checkpointing_context_fn():
                meta = defaultdict(int)
                return create_selective_checkpoint_contexts(_get_custom_policy_with_meta(meta))

            context_fn = selective_checkpointing_context_fn
        checkpoint_wrapper_fn = partial(checkpoint_wrapper_fn, context_fn=context_fn)

    # will fail for args in noop (always keep ref)
    # if activation_offloading and activation_offload_list is None: # full offload, can lead to duplicate offload in checkpointing
    #     base_wrapper_fn = checkpoint_wrapper_fn
    #     from hy_parallelism.training.activation_offload import offload_wrapper, offload_wrapper_nopin
    #     # def checkpoint_wrapper_fn(module):
    #     #     return offload_wrapper(base_wrapper_fn(module))
    #     def checkpoint_wrapper_fn(module):
    #         return offload_wrapper_nopin(base_wrapper_fn(module))


    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
        check_fn=selective_checkpointing,
    )



def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)

def check_uniform_dtype(model):
    from collections import defaultdict
    dtype_mapping = defaultdict(list)
    for name, param in model.named_parameters():
        dtype_mapping[param.dtype].append(name)
    if len(dtype_mapping) > 1:
        # Inconsitent dtype found.
        from hy_parallelism.utils import format_keys
        dict_str = '\n'
        for k, v in dtype_mapping.items():
            dict_str += f"=================  [{k}]  =================\n{format_keys(v)}\n==============================================\n"
        msg = f"FSDP expects uniform original parameter dtype but non-uniform dtype found: {dict_str}"
        loguru.logger.critical(msg)
        raise RuntimeError(msg)

def custom_fully_shard(module, fqn=None, materialize_fn: Callable | None = None, enable_symm_mem_for_comm: bool = False, ring_n: int = 2, **kwargs):
    mesh = kwargs.get('mesh', None)
    if mesh is not None:
        mesh_str = f'mesh={mesh}'
    else:
        mesh_str = ''
    debug_log(f'fuly_shard {repr(fqn)} {type(module)} {mesh_str}')
    if materialize_fn is not None:
        module = materialize_fn(module)
    module = fully_shard(module, **kwargs)
    if enable_symm_mem_for_comm:
        if version.parse(torch.__version__) < version.parse("2.10.0rc6"):
            # Skip enabling symmetric memory if torch < 2.10
            return module

        if os.getenv('NCCL_CTA_POLICY') != '2':
            raise RuntimeError('NCCL_CTA_POLICY must be set to 2.')
        mesh = kwargs['mesh']
        group = mesh.get_group(mesh_dim=-1)
        group_ranks = dist.get_process_group_ranks(group)
        if all((rank // 8 == dist.get_rank() // 8) for rank in group_ranks):
            from hy_parallelism.distributed.communications.symm_mem import set_symm_mem_for_comm
            loguru.logger.info(f'Enabling symmetric memory for {fqn} {type(module)}. {group_ranks=}')
            set_symm_mem_for_comm(module, recursive=False, ring_n=ring_n)
        else:
            pass
            # loguru.logger.warning(f'{fqn} is sharded on {mesh}. Symmetric memory is not enabled for this mesh with ranks {group_ranks}.')
    return module


def eager_fsdp_lazy_init(model: nn.Module) -> None:
    """
    fsdp 嘅 lazy init 會創建 _orig_dtype, 但係繫裹所有參數都唔 requires_grad
    則會跳過。導致系 grad reduce 時期，無法正確將 grad cast 到正確嘅 orig_dtype。
    導致 grad_dtype error (因為 PyTorch 2.10 新增對 grad_dtype 嘅檢查)
    See #32

    Snapshots _orig_dtype / _reduce_dtype on all param groups.
    Safe to call multiple times (root lazy init is idempotent).
    """
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    state = _get_module_fsdp_state(model)
    if state is None:
        raise RuntimeError(
            "eager_fsdp_lazy_init expects an FSDP root module; call fully_shard(model) first."
        )
    state._lazy_init()


def ensure_requires_grad_and_eager_fsdp_lazy_init(models) -> None:
    if isinstance(models, nn.Module):
        models = [models]
    for model in models:
        frozen = [param for param in model.parameters() if not param.requires_grad]
        for param in frozen:
            param.requires_grad_(True)
        try:
            eager_fsdp_lazy_init(model)
        finally:
            for param in frozen:
                param.requires_grad_(False)


class ShardPlacementFnCollection:

    def __init__(self, placement_fns=None):
        if placement_fns is not None:
            self.placement_fns = placement_fns
        else:
            self.placement_fns = []

    @classmethod
    def from_single_placement_fn(cls, placement_fn):
        return cls([placement_fn])

    def add_placement_fn(self, placement_fn):
        self.placement_fns.append(placement_fn)

    def __call__(self, param):
        for placement_fn in self.placement_fns[::-1]: # 后面的优先级更高
            ret = placement_fn(param)
            if ret is not None:
                return ret
        return None



def apply_fsdp2(
    model: nn.Module,
    blocks,
    default_fsdp_mesh: DeviceMesh=None,
    root_param_dtype: torch.dtype | None = None,
    root_reduce_dtype: torch.dtype | None = None,
    param_dtype: torch.dtype=torch.float32,
    reduce_dtype: torch.dtype=torch.float32,
    cpu_offload: bool = False,
    expert_fsdp_mesh: DeviceMesh = None,
    reshard_after_forward_policy: str = "default",
    expert_on_32: bool = False, # ptm moe implement requires expert on fp32
    router_on_32: bool = True, # make gate on fp32 to avoid precision issue
    wrap_expert_with_fsdp: bool | None = None, # 当不开 Ep 时，设置 true 也给 expert 套上 fsdp
    prefetch_factor: int = 1,
    backward_prefetch_factor: int | None = None,
    disable_naive_backward_prefetch: bool = False, # pytorch naive backward prefetch is buggy.
    should_forward_prefetch: Callable[[str, nn.Module], bool] | None = None,
    materialize_fn: Callable | None = None,
    cast_forward_inputs: bool = True,
    # allow_ununiform_dtype: bool = False,
    cast_master_weight_to_param_dtype: bool = False, # Enabling this is error-prone, since casting parent module could affect the child module's dtype
    shard_placement_fn_collection: Callable | ShardPlacementFnCollection | None = None,
    enable_symm_mem_for_comm: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.
        cast_forward_inputs (bool, optional): If True, cast floating-point forward inputs to
            ``param_dtype`` per ``MixedPrecisionPolicy``. Defaults to True.
    """
    ring_n = prefetch_factor + 1
    # 對於 leo 模型，暫時只需要關閉 default reshard_after_forward 行為
    # 避免最後一層連續 all_gather 即可
    # 同一個 shape 嘅 tensor 唔會存在兩份
    ring_n = 0
    if enable_symm_mem_for_comm and ring_n < prefetch_factor + 1:
        assert reshard_after_forward_policy in ['always', 'never']

    # if allow_ununiform_dtype:
    #     model = model.to(torch.float32)

    if backward_prefetch_factor is None:
        backward_prefetch_factor = 0

    pp_enabled = get_parallel_state().pp_enabled
    ep_enabled = get_parallel_state().ep_enabled

    if default_fsdp_mesh is not None and hasattr(default_fsdp_mesh, "ndim") and default_fsdp_mesh.ndim == 0:
        raise ValueError("0-dimensional DeviceMesh is not supported")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
        # output_dtype=param_dtype
    )
    fsdp_config = {"mesh": default_fsdp_mesh, "mp_policy": mp_policy, "shard_placement_fn": shard_placement_fn_collection}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    resolved_root_param_dtype = param_dtype if root_param_dtype is None else root_param_dtype
    resolved_root_reduce_dtype = reduce_dtype if root_reduce_dtype is None else root_reduce_dtype

    layer_param_dtype = fsdp_config["mp_policy"].param_dtype

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
            reshard_after_forward = True
    else:
        raise ValueError(
            f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
        )


    if router_on_32 and layer_param_dtype != torch.float32:
        for fqn, module in model.named_modules():
            if not hasattr(model, 'is_moe_router'):
                raise ValueError(f'{type(model)} must implement `is_moe_router` method. You can implement it in the ParallelEngine or the model itself.')
            if model.is_moe_router(fqn, module):
                router_fsdp_config = fsdp_config.copy()
                if router_on_32:
                    router_fsdp_config["mp_policy"] = MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32,
                        output_dtype=torch.float32
                    )
                # router_fsdp_config["mesh"] = get_parallel_state().router_fsdp_mesh
                if cast_master_weight_to_param_dtype:
                    module = module.to(router_fsdp_config["mp_policy"].param_dtype)

                # loguru.logger.debug(f'Router on 32, apply fsdp to {fqn}')
                custom_fully_shard(
                    module,
                    fqn=fqn,
                    **router_fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    materialize_fn=materialize_fn,
                    enable_symm_mem_for_comm=enable_symm_mem_for_comm,
                    ring_n=ring_n,
                )
    expert_fsdp = (
        ep_enabled or (expert_on_32 and layer_param_dtype != torch.float32) or (wrap_expert_with_fsdp is True)
    )
    if expert_fsdp:
        for fqn, module in model.named_modules():
            if not hasattr(model, 'is_expert'):
                raise ValueError('model must implement `is_expert` method. You can implement it in the ParallelEngine or the model itself.')
            if model.is_expert(fqn):
                expert_fsdp_config = fsdp_config.copy()
                if expert_on_32:
                    ep_mp_policy = MixedPrecisionPolicy(
                        param_dtype=torch.float32, reduce_dtype=torch.float32,
                    )
                    expert_fsdp_config["mp_policy"] = ep_mp_policy

                if get_parallel_state().enable_expert_fsdp_sharding:
                    expert_fsdp_config['shard_placement_fn'] = lambda param: Shard(1)
                if ep_enabled:
                    assert expert_fsdp_mesh is not None, 'Expert FSDP mesh is not set'
                    expert_fsdp_config.update({"mesh": expert_fsdp_mesh})

                if cast_master_weight_to_param_dtype:
                    module = module.to(expert_fsdp_config["mp_policy"].param_dtype)
                custom_fully_shard(
                    module,
                    fqn=fqn,
                    **expert_fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    materialize_fn=materialize_fn,
                    enable_symm_mem_for_comm=enable_symm_mem_for_comm,
                    ring_n=ring_n,
                )

                # NOTE: # Although the FSDP sharding of experts is done on a mesh of
                #       a different size than other parameters, the gradient division
                #       factor should be consistent with data.
                if version.parse(torch.__version__) >= version.parse('2.9.1') or hasattr(module, 'set_gradient_divide_factor'):
                    module.set_gradient_divide_factor(
                        get_parallel_state().fsdp_gradient_divide_factor,
                    )
                else:
                    pass
                    # loguru.logger.warning(
                    #     f'Your PyTorch version is too old ({torch.__version__}), which does not support `set_gradient_divide_factor`. '
                    #     'Upgrade PyTorch to 2.9.1 or later is recommended.'
                    # )

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

        if cast_master_weight_to_param_dtype:
            transformer_block = transformer_block.to(fsdp_config["mp_policy"].param_dtype)
        custom_fully_shard(
            transformer_block,
            fqn=f'layer[{layer_id}]',
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
            materialize_fn=materialize_fn,
            enable_symm_mem_for_comm=enable_symm_mem_for_comm,
            ring_n=ring_n,
        )


    if prefetch_factor > 0:
        set_forward_prefetch(model, blocks, prefetch_factor, should_forward_prefetch)

    # if disable_naive_backward_prefetch and backward_prefetch_factor in [None, 0]:
    if disable_naive_backward_prefetch:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_modules_to_backward_prefetch([module])

    if backward_prefetch_factor is not None and backward_prefetch_factor > 0:
        reversed_transformer_blocks = list(reversed(blocks))

        for idx, block in enumerate(reversed_transformer_blocks):
            # real_prefetch_blocks = get_sub_fsdp_modules(block)
            real_prefetch_blocks = []

            prefetch_blocks = reversed_transformer_blocks[idx+1:idx+backward_prefetch_factor+1]
            for b in prefetch_blocks:
                real_prefetch_blocks.append(b)
                # real_prefetch_blocks.extend(get_sub_fsdp_modules(b))
            block.set_modules_to_backward_prefetch(
                real_prefetch_blocks
            )


    root_fsdp_config = fsdp_config.copy()
    root_fsdp_config["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=resolved_root_param_dtype,
        reduce_dtype=resolved_root_reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )
    model = custom_fully_shard(
        model,
        fqn='root module',
        **root_fsdp_config,
        reshard_after_forward=(reshard_after_forward_policy == 'always'),
        materialize_fn=materialize_fn,
        enable_symm_mem_for_comm=enable_symm_mem_for_comm,
        ring_n=ring_n,
    )

    return model


def apply_compile(model: nn.Module, blocks,
    components: list[str],
    backend: str = "inductor",
    ep_enabled: bool = False
):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)

    from torchtitan.models.moe import moe as moe_module

    for layer_id, transformer_block in enumerate(blocks):
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    moe_module._run_experts_grouped_mm = torch.compile(
        moe_module._run_experts_grouped_mm,
        backend=backend,
        fullgraph=True,
    )

    if ep_enabled:
        compiled_fn = moe_module._run_experts_grouped_mm

        def _run_experts_grouped_mm_dynamic(
            w1: torch.Tensor,
            w2: torch.Tensor,
            w3: torch.Tensor,
            x: torch.Tensor,
            num_tokens_per_expert: torch.Tensor,
        ) -> torch.Tensor:
            # dynamic number of tokens in expert parallel
            torch._dynamo.mark_dynamic(x, 0)
            return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

        moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    loguru.logger.info("Compiling each TransformerBlock with torch.compile")

def get_fsdp_named_parameters(model: nn.Module):
    from torch.distributed.checkpoint.state_dict import _get_fqns
    for name, param in model.named_parameters():
        fqns = _get_fqns(model, name)
        if fqns is not None:
            assert len(fqns) == 1, (name, fqns)
            yield next(iter(fqns)), param

def get_fsdp_named_buffers(model: nn.Module):
    from torch.distributed.checkpoint.state_dict import _get_fqns
    for name, buffer in model.named_buffers():
        fqns = _get_fqns(model, name)
        if fqns is not None:
            assert len(fqns) == 1, (name, fqns)
            yield next(iter(fqns)), buffer


def iter_sharded_param_tensors(root):
    # 配合 move_sharded_pair 使用
    from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
    for module in root.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = module._get_fsdp_state()._fsdp_param_group
        if group is None:
            continue
        for p in group.fsdp_params:
            # if p.sharded_state != ShardedState.UNSHARDED:
            #     loguru.logger.debug(
            #         f"{p._param_fqn}: expected UNSHARDED, got {p.sharded_state}. Skip iterating it."
            #     )
            #     continue
            yield p


@torch.no_grad()
def move_sharded_param(fsdp_param, device, *, pin_memory=False):
    # 應用場景系當 model 已經保持 unshard 嘅時候，希望保持 unshard 推理，但係唔想保留 fsdp 於半嘅 hsarded 顯存
    device = torch.device(device)
    data = fsdp_param._sharded_param_data
    if data.device == device:
        return
    local = fsdp_param.sharded_param._local_tensor
    size, stride, offset = local.size(), local.stride(), local.storage_offset()
    flat = data.to(device, non_blocking=True)
    if pin_memory and device.type == "cpu" and not flat.is_pinned():
        flat = flat.pin_memory()
    # local_tensor.set_(flat.as_strided(size, stride, offset))
    # 不要 set_，直接换引用，避免 cross-device
    fsdp_param._sharded_param_data = flat
    fsdp_param.sharded_param._local_tensor = flat.as_strided(size, stride, offset)
