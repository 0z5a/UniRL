from .caption_base import CaptionOut
from ..utils.text_utils import has_repeat


class CaptionAug:
    def __init__(
            self,
            caption_sample_ratio=None,
            logger=None,
            filter_repeat=True,
    ):
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.caption_sample_ratio = caption_sample_ratio    # not used
        self.filter_repeat = filter_repeat

    def caption_aug(self, raw_caption, lang, return_key=False, **kwargs) -> CaptionOut:
        if self.filter_repeat and has_repeat(raw_caption):
            raise ValueError(f"caption has repeat: {raw_caption}")
        if return_key:
            return CaptionOut(caption=raw_caption, lang=lang)
        else:
            return CaptionOut(caption=None, lang=None)
