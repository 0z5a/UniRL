import importlib.metadata
from packaging import version
from functools import lru_cache

@lru_cache
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = True) -> bool:
    """
    dev version of pytorch will raise error in `torch.load` when ussing transformers
    if `accept_dev=False`, this patch will set `accept_dev=True`
    """
    from transformers.utils.import_utils import _is_package_available
    if not _is_package_available("torch"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("torch")).base_version) >= version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("torch")) >= version.parse(library_version)



def patch_transformers_is_torch_greater_or_equal():
    try:
        from transformers.utils import import_utils
    except:
        return
    if hasattr(import_utils, 'is_torch_greater_or_equal'):
        import_utils.is_torch_greater_or_equal = is_torch_greater_or_equal


