import sys
from typing import TYPE_CHECKING, Union

from loguru import logger
if TYPE_CHECKING:
    from index_kits import ArrowIndexV2, MultiIndexV2

from .data_utils import DataMixin
from ..caption_strategy.caption_base import CaptionOut
from ..caption_strategy.manager import MultiCaptionManager


class AudioCaptionMixin(DataMixin):
    """
    A mixin class for audio captioning strategies.

    Notice: Make sure all the method and attribute names contain 'audio' to avoid name conflicts with other mixins,
            unless they are compatible with both image, video and audio modalities.
    """
    log_prefix: str
    task_kwargs: dict
    index_kwargs: dict
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"
    audio_caption_manager: MultiCaptionManager

    def setup_audio_caption(self, args):
        # task kwargs: prompt_token_length, recaption_token_length, reasoning_token_length, cot_kwargs
        # cot_kwargs: recaption_cot_prob, recaption_cot, recaption_cot_kwargs, reasoning_cot_prob
        _ = args
        # Audio Caption configs and manager
        self.audio_prompt_token_length = self.task_kwargs.get('audio_prompt_token_length', 256)
        # Notice that audio_prompt_token_length only used for t2a. For t2va with separated video and audio captions,
        # we still use video_prompt_token_length to control the total prompt token length.
        self.audio_caption_manager = MultiCaptionManager(
            resource=self.task_kwargs.get('audio_caption_resource'),
            dataset=self,
        )

        # Total text length
        self.audio_text_token_length = self.task_kwargs.get('all_text_token_length', self.audio_prompt_token_length)

    def get_audio_caption(
            self,
            ind: int,
    ) -> tuple[CaptionOut, bool]:
        """ Get audio caption for a given index. """
        try:
            out = self.audio_caption_manager.get_caption(ind, return_dict=True)
        except Exception as e:
            line_no = sys._getframe().f_lineno
            logger.error(f"L{line_no} <- get_audio_caption | {e.__class__.__name__}: {str(e)} (index={ind})")
            out = CaptionOut(caption=None, lang=None)

        # Check status
        assert isinstance(out, CaptionOut), \
            f"audio_caption_manager.get_caption() must return a CaptionOut object, but got {type(out)}"
        success = bool(out.caption)

        return out, success
