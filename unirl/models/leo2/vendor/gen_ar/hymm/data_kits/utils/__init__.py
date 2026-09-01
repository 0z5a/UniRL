import importlib
from typing import Any, TYPE_CHECKING


# For type checking only
if TYPE_CHECKING:
    from .audio_utils import AudioMixin
    from .audio_caption_utils import AudioCaptionMixin
    from .data_utils import DataMixin
    from .caption_utils import ImageCaptionMixin
    from .image_utils import ImageMixin
    from .text_utils import TextMixin
    from .video_utils import VideoMixin
    from .video_caption_utils import VideoCaptionMixin
    from .index_utils import IndexColumn, IndexDataset, resample_for_errors
    from .multimodal_states import (
        MultimodalTasksState, DistributedSamplingState, prepare_distributed_sampling,
    )


# For runtime lazy imports
_LAZY_IMPORTS = {
    # indexers
    "AudioMixin": (".audio_utils", "AudioMixin"),
    "AudioCaptionMixin": (".audio_caption_utils", "AudioCaptionMixin"),
    "DataMixin": (".data_utils", "DataMixin"),
    "ImageCaptionMixin": (".caption_utils", "ImageCaptionMixin"),
    "ImageMixin": (".image_utils", "ImageMixin"),
    "TextMixin": (".text_utils", "TextMixin"),
    "VideoMixin": (".video_utils", "VideoMixin"),
    "VideoCaptionMixin": (".video_caption_utils", "VideoCaptionMixin"),
    "MultimodalTasksState": (".multimodal_states", "MultimodalTasksState"),
    "DistributedSamplingState": (".multimodal_states", "DistributedSamplingState"),
    "prepare_distributed_sampling": (".multimodal_states", "prepare_distributed_sampling"),
    "IndexColumn": (".index_utils", "IndexColumn"),
    "IndexDataset": (".index_utils", "IndexDataset"),
    "resample_for_errors": (".index_utils", "resample_for_errors"),
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


__all__ = list(_LAZY_IMPORTS.keys())
