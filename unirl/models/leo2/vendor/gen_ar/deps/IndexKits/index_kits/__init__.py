import importlib
from typing import Any, TYPE_CHECKING

from .version import __version__


# For type checking only
if TYPE_CHECKING:
    from .indexer import ArrowIndexV2
    from .bucket import MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2
    from .builder.base import IndexV2Builder
    from .builder.multireso import build_multi_resolution_bucket_fast
    from .common import load_index, show_index_info
    from .resolution import Resolution, ResolutionGroup, DurationGroup, DurationAndResolution, DurationAndResolutionGroup
    from .utils import arrow_mapper
    from .arrow_tools import get_table, df_to_table, pydict_to_table


# For runtime lazy imports
_LAZY_IMPORTS = {
    # indexers
    "ArrowIndexV2": (".indexer", "ArrowIndexV2"),
    "MultiIndexV2": (".bucket", "MultiIndexV2"),
    "MultiResolutionBucketIndexV2": (".bucket", "MultiResolutionBucketIndexV2"),
    "MultiMultiResolutionBucketIndexV2": (".bucket", "MultiMultiResolutionBucketIndexV2"),
    # builders
    "IndexV2Builder": (".builder.base", "IndexV2Builder"),
    "build_multi_resolution_bucket_fast": (".builder.multireso", "build_multi_resolution_bucket_fast"),
    # common tools
    "load_index": (".common", "load_index"),
    "show_index_info": (".common", "show_index_info"),
    # resolution
    "Resolution": (".resolution", "Resolution"),
    "ResolutionGroup": (".resolution", "ResolutionGroup"),
    "DurationGroup": (".resolution", "DurationGroup"),
    "DurationAndResolution": (".resolution", "DurationAndResolution"),
    "DurationAndResolutionGroup": (".resolution", "DurationAndResolutionGroup"),
    # utils
    "arrow_mapper": (".utils", "arrow_mapper"),
    # arrow tools
    "get_table": (".arrow_tools", "get_table"),
    "df_to_table": (".arrow_tools", "df_to_table"),
    "pydict_to_table": (".arrow_tools", "pydict_to_table"),
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
    "__version__",
]
