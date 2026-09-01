import copy
from loguru import logger
import loguru
import torch
from torch import distributed as dist
from torch.optim import Adam, AdamW
from hy_parallelism.optimizers.muon.torch_muon import Muon
from hy_parallelism.optimizers.muon.dist_muon import get_dist_muon_for_engine

from hymm.utils.helpers import multi_pattern_match


def build_optimizer_factory(
    *,
    optimizer_name: str,
    optimizer_cls,
    base_opt_kwargs: dict,
    muon_opt_kwargs: dict,
    adamw_opt_kwargs: dict,
    special_adamw_params: list,
    special_weight_decay_params: list,
    no_weight_decay_params: list,
    muon_params: list,
    adamw_params: list,
    enable_special_weight_decay: bool = False,
):
    # TODO(kevinkhwu): Remove this deprecated function in a future version

    def optimizer_factory(param_name, param):
        # if param is not required, return None
        if not param.requires_grad:
            return None

        # if optimizer is not muon, return None
        if "muon" in optimizer_name.lower():
            # if param is a matrix parameter and not special adamw params, use muon
            if param.ndim >= 2 and not multi_pattern_match(param_name, special_adamw_params):
                selected_optimizer_cls = get_dist_muon_for_engine
                selected_optimizer_kwargs = muon_opt_kwargs
                muon_params.append(param_name)
            else:
                selected_optimizer_cls = AdamW
                selected_optimizer_kwargs = adamw_opt_kwargs
                adamw_params.append(param_name)
        else:
            selected_optimizer_cls = optimizer_cls
            selected_optimizer_kwargs = base_opt_kwargs
        
        # 向前兼容，保证之前的adam代码能复现效果
        if enable_special_weight_decay:
            # 以下逻辑是为与Megatron-LM的no_weight_decay默认设置逻辑一致
            # megatron代码里将bais、layernorm/rmsnorm、embedding、output_layer等参数设置为no-weight-decay
            # Check if parameter matches extra substrings for weight decay
            match_extra_substrings = multi_pattern_match(param_name, special_weight_decay_params)
            no_wd = not match_extra_substrings and (
                param_name.endswith(".bias")
                or len(param.shape) == 1
                or "embedding" in param_name
                or "embed_tokens" in param_name
            )
            if no_wd:
                no_wd_opt_kwargs = selected_optimizer_kwargs.copy()
                no_wd_opt_kwargs["weight_decay"] = 0.0
                no_weight_decay_params.append(param_name)
                return selected_optimizer_cls, no_wd_opt_kwargs

        return selected_optimizer_cls, selected_optimizer_kwargs

    return optimizer_factory


def build_optimizer_factory_from_args(*, args):
    # TODO(kevinkhwu): Remove this deprecated function in a future version

    optimizer_factory_map = {
        "adam": AdamW,
        "muon": get_dist_muon_for_engine,
        "dist_muon": get_dist_muon_for_engine,
    }
    optimizer_name = getattr(args, "optimizer", None)
    assert optimizer_name in optimizer_factory_map, (
        f"Invalid optimizer name: {optimizer_name}; available optimizers: {list(optimizer_factory_map.keys())}"
    )
    optimizer_cls = optimizer_factory_map[optimizer_name]

    # set no weight decay params
    base_opt_kwargs = args.optimizer_params.copy()
    # transform adamw optimizer params
    adamw_opt_kwargs = base_opt_kwargs.copy()
    adamw_opt_kwargs.pop("momentum", None)
    # transform muon optimizer params
    if "muon" in optimizer_name.lower():
        if "adamw_betas" not in base_opt_kwargs and "betas" in base_opt_kwargs:
            base_opt_kwargs["adamw_betas"] = base_opt_kwargs.pop("betas")
        if "adamw_eps" not in base_opt_kwargs and "eps" in base_opt_kwargs:
            base_opt_kwargs["adamw_eps"] = base_opt_kwargs.pop("eps")
    muon_opt_kwargs = base_opt_kwargs.copy()
    
    special_weight_decay_params = getattr(args, "special_weight_decay_params", [])
    special_adamw_params = getattr(args, "special_adamw_params", ["embed_tokens", "lm_head"])
    enable_special_weight_decay = getattr(args, "enable_special_weight_decay", False)

    no_weight_decay_params = []
    muon_params = []
    adamw_params = []

    optimizer_factory = build_optimizer_factory(
        optimizer_name=optimizer_name,
        optimizer_cls=optimizer_cls,
        base_opt_kwargs=base_opt_kwargs,
        muon_opt_kwargs=muon_opt_kwargs,
        adamw_opt_kwargs=adamw_opt_kwargs,
        special_adamw_params=special_adamw_params,
        special_weight_decay_params=special_weight_decay_params,
        no_weight_decay_params=no_weight_decay_params,
        muon_params=muon_params,
        adamw_params=adamw_params,
        enable_special_weight_decay=enable_special_weight_decay,
    )

    return optimizer_factory, optimizer_cls, base_opt_kwargs, no_weight_decay_params, muon_params, adamw_params


def build_param_optimizer_map(
    model, 
    default_adam_kwargs: dict, 
    default_muon_kwargs: dict,
    force_adamw_params: list = ["embed_tokens", "lm_head"],
    special_weight_decay_params: list = [],
):

    param_optimizer_map = {}
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # MOE layers could have 3D parameters.
        # Parameters with higher dimensions could be conv layers.
        # Flattening Conv with FSDP is kinda tricky in hy_parallelism, because we need to 
        # distinguish between Conv and MOE layers. 
        # Use AdamW for Conv layers for now.
        if param.ndim >= 2 and not multi_pattern_match(param_name, force_adamw_params):
            selected_optimizer_cls = Muon
            selected_optimizer_kwargs = default_muon_kwargs
        else:
            selected_optimizer_cls = AdamW
            selected_optimizer_kwargs = default_adam_kwargs

        # 以下逻辑是为与Megatron-LM的no_weight_decay默认设置逻辑一致
        # megatron代码里将bias、layernorm/rmsnorm、embedding、output_layer等参数设置为no-weight-decay
        match_extra_substrings = multi_pattern_match(param_name, special_weight_decay_params)
        no_wd = not match_extra_substrings and (
            param_name.endswith(".bias")
            or len(param.shape) == 1
            or "embedding" in param_name
            or "embed_tokens" in param_name
        )
        if no_wd:
            no_wd_opt_kwargs = selected_optimizer_kwargs.copy()
            no_wd_opt_kwargs["weight_decay"] = 0.0
            param_optimizer_map[param_name] = (selected_optimizer_cls, no_wd_opt_kwargs)
        else:
            param_optimizer_map[param_name] = (selected_optimizer_cls, selected_optimizer_kwargs)

    return param_optimizer_map

def get_optimizer_factory_from_param_optimizer_mapping(param_optimizer_mapping):
    def factory(name, param):
        return param_optimizer_mapping.get(name, None)
    return factory

def reverse_param_optimizer_mapping(param_optimizer_mapping):
    from collections import defaultdict
    def dict_to_str(d): # Dirty hack for unhashable dict
        import json
        return json.dumps(d, sort_keys=True)
    hashable_param_optimizer_mapping = defaultdict(list)
    for param_name, (optimizer_cls, optimizer_kwargs) in param_optimizer_mapping.items():
        hashable_param_optimizer_mapping[(optimizer_cls, dict_to_str(optimizer_kwargs))].append(param_name)
    return hashable_param_optimizer_mapping

def pre_optimizer_hook(model):
    from hy_parallelism.distributed.fsdp_util import get_fsdp_named_parameters
    for param_name, param in get_fsdp_named_parameters(model):
        if 'gate_and_up_proj.weight' in param_name:
            param._muon_split_fn = lambda x: torch.chunk(x, 2, dim=0)
            param._muon_merge_fn = lambda tensors: torch.cat(tensors, dim=0)
        elif 'mod_proj' in param_name and param_name.endswith('.weight'):
            # TODO: Fix hardcode factor 6
            param._muon_split_fn = lambda x: torch.chunk(x, 6, dim=0) 
            param._muon_merge_fn = lambda tensors: torch.cat(tensors, dim=0)
        elif param.ndim == 3:
            param._muon_split_fn = lambda x: list(x)
            param._muon_merge_fn = lambda tensors: torch.stack(tensors)
        elif param.ndim == 4:
            from functools import partial
            def merge_fn(x, shape):
                return x[0].view(shape)
            original_shape = param.shape
            partial_merge_fn = partial(merge_fn, shape=original_shape)
            param._muon_split_fn = lambda x: [x.view(x.size(0), -1)]
            param._muon_merge_fn = partial_merge_fn
        elif param_name.endswith("qkv_proj.weight"):
            param._muon_split_fn = lambda x: torch.split(x, [model.config.num_attention_heads, model.config.num_kv_heads, model.config.num_kv_heads], dim=0)
            param._muon_merge_fn = lambda tensors: torch.cat(tensors, dim=0)


def build_optimizer_factory_from_model_args(*, model, args):
    # Adam supported kwargs
    # params: ParamsT,
    # lr: Union[float, Tensor] = 1e-3,
    # betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
    # eps: float = 1e-8,
    # weight_decay: float = 1e-2,
    # amsgrad: bool = False,
    # *,
    # maximize: bool = False,
    # foreach: Optional[bool] = None,
    # capturable: bool = False,
    # differentiable: bool = False,
    # fused: Optional[bool] = None,


    # Muon supported kwargs
    # lr: float = 1e-3,
    # weight_decay: float = 0.1,
    # momentum: float = 0.95,
    # nesterov: bool = True,
    # ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    # eps: float = EPS,
    # ns_steps: int = DEFAULT_NS_STEPS,
    # adjust_lr_fn: str | None = None,

    adam_supported_kwargs = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'}
    muon_supported_kwargs = {'lr', 'weight_decay', 'momentum', 'nesterov', 'ns_coefficients', 'eps', 'ns_steps', 'adjust_lr_fn', 'sum_decay_momentum'}

    
    adam_default_kwargs = args.optimizer_params.copy()
    for key in copy.deepcopy(adam_default_kwargs):
        if key not in adam_supported_kwargs:
            adam_default_kwargs.pop(key)
            logger.debug(f"{key} is not supported by Adam optimizer. It will be ignored.")


    muon_default_kwargs = args.optimizer_params.copy()
    for key in copy.deepcopy(muon_default_kwargs):
        if key not in muon_supported_kwargs:
            muon_default_kwargs.pop(key)
            logger.debug(f"{key} is not supported by Muon optimizer. It will be ignored.")
    

    special_weight_decay_params = getattr(args, "special_weight_decay_params", [])
    force_adamw_params = getattr(args, "special_adamw_params", ["embed_tokens", "lm_head"])

    param_optimizer_map = build_param_optimizer_map(
        model, 
        default_adam_kwargs=adam_default_kwargs, 
        default_muon_kwargs=muon_default_kwargs, 
        force_adamw_params=force_adamw_params, 
        special_weight_decay_params=special_weight_decay_params
    )

    return get_optimizer_factory_from_param_optimizer_mapping(param_optimizer_map), reverse_param_optimizer_mapping(param_optimizer_map)