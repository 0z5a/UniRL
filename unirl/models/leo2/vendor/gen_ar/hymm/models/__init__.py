import importlib
from typing import Union, Dict, Any, TYPE_CHECKING
from easydict import EasyDict
import torch

from .basic import BaseConfig

if TYPE_CHECKING:
    from .multimodal.hunyuan_multimodal_state import HunyuanMultimodalState
    from .multimodal.hunyuan_multimodal_hf import HunyuanMultimodalHF
    from .diffusion.leo_hf import LeoModelHF

# For runtime lazy imports
_LAZY_IMPORTS = {
    "Arrangement": (".arrangement", "Arrangement"),
    "load_vae": (".autoencoders", "load_vae"),
    "EMA": (".ema", "EMA"),
    "DistributedEMA": (".ema", "DistributedEMA"),
    "TPDistributedEMA": (".ema", "TPDistributedEMA"),
    "TokenizerWrapper": (".tokenizers", "TokenizerWrapper"),
    "AudioTokenizerWrapper": (".tokenizers", "AudioTokenizerWrapper"),
}

MODEL_TYPE = Union[torch.nn.Module, "HunyuanMultimodalState", "HunyuanMultimodalHF", "LeoModelHF"]

# structure --> model name prefix
name_to_structure = {
    # Diffusion model zoo
    'diffusion': ['DiT', 'PixArt', 'Lumina', 'Aries', 'SDXL-EMU2', 'VQ-SDXL-EMU2', 'TV2A-Aries', 'leo'],

    # Autoregressive model zoo
    'autoregressive': ['phi', 'hunyuan', 'qwen', 'deepseek', 'janus'],

    # MLM model zoo
    'masked_modeling': ['maskgit-256-reimpl', 'maskgit-512', 'maskgit-256-400m'],

    # Multimodal model zoo
    'multimodal': ['hunyuan']
}


def convert_model_name_to_structure(model_name):
    if "." in model_name:
        model_structure, model_name = model_name.split(".")
        return model_structure
    for model_structure, model_names in name_to_structure.items():
        for name in model_names:
            if model_name.startswith(name):
                return model_structure
    return None


def build_model(
        args,
        pretrained_ckpt=None,
        logger=None,
        dtype=None,
        device=None,
        **kwargs,
) -> tuple[MODEL_TYPE, Union[Dict[str, Any], EasyDict, BaseConfig]]:
    # Support cpu, cuda, meta devices
    factor_kwargs = {"device": device, "dtype": dtype}

    if logger is None:
        from loguru import logger

    structure = convert_model_name_to_structure(args.model_name)
    if structure is None:
        raise NotImplementedError(f"model structure not implemented for {args.model_name}. "
                                  f"Please check the `name_to_structure` dict in models/__init__.py")

    models_spec = importlib.import_module(f"hymm.models.{structure}")
    logger.info(f"Building model from {structure} for {args.model_name}")

    if device == 'meta':
        context = torch.device('meta')
    else:
        from contextlib import nullcontext
        context = nullcontext()

    with context:
        model, model_config = models_spec.build_model(
            args, pretrained_ckpt=pretrained_ckpt, logger=logger, **kwargs, **factor_kwargs)

    if getattr(args, "use_hf", None):
        # When using HF models, the device is already set when building the model.
        pass
    elif device != 'meta':
        # Although we have pass factor_kwargs to build_model, some models may still
        # not support it (e.g., HF models). So we still need to move the model here.
        # model = model.to(**factor_kwargs)
        #
        # Now we assume all models are moved to the target device in build_model.
        pass
    else:
        # For meta device, the caller is responsible for initializing the parameters later.
        pass

    return model, model_config


# Cache
_imported_objects: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        if name not in _imported_objects:
            module_path, object_name = _LAZY_IMPORTS[name]
            module = importlib.import_module(module_path, package=__package__)
            _imported_objects[name] = getattr(module, object_name)
        return _imported_objects[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys()) + [
    "build_model",
]
