import importlib
from pathlib import Path
from typing import Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast
    from .custom_audio_tokenizer import AudioTokenizerWrapper
    from .tokenizer_wrapper import TokenizerWrapper
    from .tokenization_hunyuan_multimodal import HunyuanMultimodalTokenizerFast


def load_tokenizer(
        tokenizer_name: str,
        tokenizer_class: str,
) -> "Union[TokenizerWrapper, HunyuanMultimodalTokenizerFast]":
    assert '.' in tokenizer_class, (
        f"Invalid tokenizer class: {tokenizer_class}. A valid tokenizer name should be in the form of "
        f"<module_name>.<tokenizer_cls>."
    )

    # Check if is a transformers tokenizer
    if tokenizer_class.startswith("transformers."):
        module_name, tokenizer_cls = tokenizer_class.rsplit('.', 1)
        module_spec = importlib.import_module(module_name)
        TokenizerSpec = getattr(module_spec, tokenizer_cls)     # noqa

    # Get submodules in hymm.models.tokenizers directory
    else:
        local_submodules = set([x.stem for x in Path(__file__).parent.glob("*") if x.stem != "__init__"])

        module_name, tokenizer_cls = tokenizer_class.rsplit('.', 1)
        if module_name.split(".")[0] in local_submodules:
            module_spec = importlib.import_module(f"hymm.models.tokenizers.{module_name}")
        else:
            module_spec = importlib.import_module(module_name)
        TokenizerSpec = getattr(module_spec, tokenizer_cls)     # noqa

    from ...constants import TOKENIZER_PATH
    if tokenizer_name in TOKENIZER_PATH:
        tokenizer_name = TOKENIZER_PATH[tokenizer_name]

    return TokenizerSpec.from_pretrained(tokenizer_name)


# For runtime lazy imports
_LAZY_IMPORTS = {
    "TokenizerWrapper": (".tokenizer_wrapper", "TokenizerWrapper"),
    "AudioTokenizerWrapper": (".custom_audio_tokenizer", "AudioTokenizerWrapper"),
}


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
    "load_tokenizer",
]
