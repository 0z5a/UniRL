import math
from typing import List, Optional, Union

import torch
import torch.distributed as dist

from ..core.parallel_states import ParallelState
from ..core.global_vars import get_parallel_state


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly.
    
    This function handles models wrapped by DDP, FSDP, or other wrappers.
    """
    if isinstance(model, list):
        raise RuntimeError("get_attr_wrapped_model given a list of models")

    if allow_none:
        def condition(model, attr):
            return not hasattr(model, attr)
    else:
        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_updated_expert_bias(
    tokens_per_expert, 
    expert_bias, 
    expert_bias_update_rate, 
    enable_zero_mean_update=False
):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_update_rate (float): The update rate for the expert bias.
        enable_zero_mean_update (bool): Whether to enable zero-mean update for expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        # Try to get group from ParallelState
        if dist.is_initialized():
            group = get_parallel_state().get_tensor_and_data_parallel_group(with_context_parallel=True)
            dist.all_reduce(tokens_per_expert, group=group)
        
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        if enable_zero_mean_update:
            updated_expert_bias = (
                expert_bias
                + (torch.sign(offset) - torch.sign(offset).mean(dim=-1, keepdim=True))
                * expert_bias_update_rate
            )
        else:
            updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


def update_router_expert_bias(
    model: Union[torch.nn.Module, List[torch.nn.Module]], 
    expert_bias_update_rate: float, 
    enable_expert_bias_zero_mean_update: bool = False,
    mot_und_frozen: bool = False,
    mot_gen_frozen: bool = False,
):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks.
    
    Args:
        model: Model or list of model chunks.
        expert_bias_update_rate: The update rate for the expert bias.
        enable_expert_bias_zero_mean_update: Whether to enable zero-mean update.
        mot_und_frozen: Whether the MoT und branch is frozen.
        mot_gen_frozen: Whether the MoT gen branch is frozen.
    """
    tokens_per_expert_list = []
    expert_bias_list = []
    tokens_reset_only_list = []
    for model_chunk in model:
        for name, module in get_attr_wrapped_model(model_chunk, 'named_modules')():
            if hasattr(module, 'expert_bias'):
                if mot_und_frozen and not ("mlp_mot_gen" in name.split(".")):
                    tokens_reset_only_list.append(module.local_tokens_per_expert)  # und：不更新 bias，但要清零计数
                    continue
                if mot_gen_frozen and ("mlp_mot_gen" in name.split(".")):
                    tokens_reset_only_list.append(module.local_tokens_per_expert)  # gen：不更新 bias，但要清零计数
                    continue
                tokens_per_expert_list.append(module.local_tokens_per_expert)      # 更新 bias
                expert_bias_list.append(module.expert_bias)
    # For hybrid models with both MoE and Dense layers, this list can be empty.
    if len(expert_bias_list) == 0:
        return
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)

    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert,
        stacked_expert_bias,
        expert_bias_update_rate,
        enable_expert_bias_zero_mean_update,
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
        tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)

    # Reset tokens_per_expert for und branch if mot_und_frozen is True or gen branch if mot_gen_frozen is True
    for tokens_per_expert in tokens_reset_only_list:
        tokens_per_expert.zero_()


# Alias for backward compatibility
_update_router_expert_bias = update_router_expert_bias