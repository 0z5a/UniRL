# Copied from https://github.com/pytorch/torchtitan/blob/9f211ec199bc887901b874edd6af5a20527a4175/torchtitan/components/optimizer.py#L363

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.tensor import Replicate

from hy_parallelism.parallel_states import ParallelDims, get_parallel_state
from hy_parallelism.checkpoint.stateful import OptimizersContainer

def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
    for model_part in model_parts:
        # pyrefly: ignore [not-callable]
        for transformer_block in model_part.layers.values():
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe_enabled:
                # Assumption: load_balance_coeff is set universally on all moe blocks.
                # pyrefly: ignore [missing-attribute]
                return bool(transformer_block.moe.load_balance_coeff)
    return False

# for MoE auxiliary-loss-free load balancing
def _is_recomputation_enabled(module):
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

def _update_expert_bias(
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
):
    # loss_mesh = parallel_dims.get_optional_mesh("loss")
    # loss_mesh = parallel_dims.get_mesh("loss")
    loss_mesh = parallel_dims._global_meshes["loss"]


    # TODO: Currently this sync is blocking (thus exposed) and happens on the
    # default compute stream. Need to assess if this is OK performance-wise.
    tokens_per_expert_list = []
    for model_part in model_parts:
        # pyrefly: ignore [not-callable]
        for module in model_part.modules():
            from hy_parallelism.models.modules.moe import MoE
            if not isinstance(module, MoE) or module.load_balance_coeff is None:
                continue
            # pyrefly: ignore [missing-attribute]
            tokens_per_expert = module.tokens_per_expert
            # if _is_recomputation_enabled(transformer_block):
            #     # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
            #     # This does not affect to expert choice, but affects the experts usage metrics.
            #     # We divide by 2 to correct for this double-counting due to recomputation
            #     # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
            #     tokens_per_expert = tokens_per_expert // 2
            tokens_per_expert_list.append(tokens_per_expert)

    tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

    if loss_mesh is not None:
        if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
            tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                placements=[Replicate()]
                * tokens_per_expert_by_layer.device_mesh.ndim
            )
        else:
            # Perform single all-reduce to get global statistics across all processes
            pg = loss_mesh.get_group()
            torch.distributed.all_reduce(
                tokens_per_expert_by_layer,
                group=pg,
                op=torch.distributed.ReduceOp.SUM,
            )

    with torch.no_grad():
        for model_part in model_parts:
            # pyrefly: ignore [not-callable]
            for module in model_part.modules():
                from hy_parallelism.models.modules.moe import MoE
                if not isinstance(module, MoE) or module.load_balance_coeff is None:
                    continue
                moe = module
                tokens_per_expert = module.tokens_per_expert
                expert_bias_delta = module.load_balance_coeff * torch.sign(
                    tokens_per_expert.mean() - tokens_per_expert
                )
                expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                module.expert_bias.add_(expert_bias_delta)
                module.tokens_per_expert.zero_()

                # update the expert bias
                # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                expert_bias_delta = moe.load_balance_coeff * torch.sign(
                    tokens_per_expert.mean() - tokens_per_expert
                )
                expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                moe.expert_bias.add_(expert_bias_delta)
                moe.tokens_per_expert.zero_()

