import sys
import random
from typing import TYPE_CHECKING, Union, Optional

from loguru import logger
if TYPE_CHECKING:
    from index_kits import ArrowIndexV2, MultiIndexV2

from ...utils.video_base import VideoTensor
from .data_utils import DataMixin
from ..caption_strategy.caption_base import CaptionOut
from ..caption_strategy.manager import MultiCaptionManager
from ..caption_strategy.prompt_patcher import apply_prompt_patchers


class VideoCaptionMixin(DataMixin):
    """
    A mixin class for video captioning strategies.

    Notice: Make sure all the method and attribute names contain 'video' to avoid name conflicts with other mixins,
            unless they are compatible with both image and video modalities.
    """
    log_prefix: str
    task_kwargs: dict
    index_kwargs: dict
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"
    video_caption_manager: MultiCaptionManager
    recaption_cot: Optional[str] = None

    def setup_video_caption(self, args):
        # task kwargs: prompt_token_length, recaption_token_length, reasoning_token_length, cot_kwargs
        # cot_kwargs: recaption_cot_prob, recaption_cot, recaption_cot_kwargs, reasoning_cot_prob
        _ = args
        # Video Caption configs and manager
        self.video_prompt_token_length = self.task_kwargs.get('prompt_token_length', 256)
        self.video_patcher_names = self.task_kwargs.get('patcher_names', [])
        caption_resource = self.task_kwargs.get('video_caption_resource')
        self.video_caption_manager = MultiCaptionManager(
            resource=caption_resource,
            dataset=self,
        )

        # Whether to predict user prompt tokens
        self.video_predict_prompt = self.task_kwargs.get('predict_prompt', False)
        # Instruction candidates type (for instruction tuning).
        instruct_type = self.task_kwargs.get("instruct_type", "none").lower()
        self.video_instruct_type, *extra = instruct_type.split('@')
        self.video_instruct_prob = float(extra[0]) if extra else 1.0
        assert self.video_instruct_type in {"fixed_set", "none"}, \
            f"Unsupported instruct_type: {self.video_instruct_type}"

        # Video Caption CoT configs
        self.video_recaption_token_length = self.task_kwargs.get('recaption_token_length', 0)
        self.video_reasoning_token_length = self.task_kwargs.get('reasoning_token_length', 0)

        self.video_cot_kwargs = self.task_kwargs.get("cot_kwargs", {})
        self.video_recaption_cot_prob = self.video_cot_kwargs.get("recaption_cot_prob", 0)
        if self.video_recaption_cot_prob > 0:
            self.require_configs(self.video_cot_kwargs, "recaption_cot", "recaption cot")
            self.video_recaption_cot = self.video_cot_kwargs["recaption_cot"]
            assert isinstance(self.video_recaption_cot, str), "caption_cot must be a string pattern."
            assert self.video_recaption_token_length > 0, \
                f"When recaption_cot_prob > 0, recaption_token_length should be greater than 0."

        self.video_reasoning_cot_prob = self.video_cot_kwargs.get('reasoning_cot_prob', 0)
        if self.video_reasoning_cot_prob > 0:
            self.require_configs(self, (
                ["think_en_col", "think_zh_col"], ["think_en_key", "think_zh_key"],
            ), "t2v reasoning cot")
            assert self.video_reasoning_token_length > 0, \
                f"When reasoning_cot_prob > 0, reasoning_token_length should be greater than 0."

        # Total text length
        self.video_text_token_length = self.task_kwargs.get(
            'all_text_token_length',
            self.video_prompt_token_length + self.video_recaption_token_length + self.video_reasoning_token_length
        )

    def get_video_instruct(self, candidates):
        if self.video_instruct_type == "none":
            return ""
        # fixed_set
        selected = random.choice(candidates).strip() + " "
        if self.video_instruct_prob < 1.0 and random.random() > self.video_instruct_prob:
            return ""
        return selected

    def get_video_caption(
            self,
            src: int | dict,
            video: Optional[VideoTensor] = None,
    ) -> tuple[CaptionOut | list[CaptionOut], bool]:
        """ Get video caption for a given index. """
        try:
            if self.video_recaption_cot_prob > 0 and random.random() < self.video_recaption_cot_prob:
                recaption_cot = self.recaption_cot
            else:
                recaption_cot = None
            out = self.video_caption_manager.get_caption(src, return_dict=True, pattern=recaption_cot)
        except Exception as e:
            line_no = sys._getframe().f_lineno
            logger.error(f"L{line_no} <- get_video_caption | {e.__class__.__name__}: {str(e)} (src={src})")
            out = CaptionOut(caption=None, lang=None)

        # Apply prompt patchers
        if self.video_patcher_names:
            lang = out[0].lang if isinstance(out, (list, tuple)) else out.lang
            out = apply_prompt_patchers(
                out,
                self.video_patcher_names,
                lang=lang,
                dataset=self,
                index=src,
                # kwargs
                video=video,
            )

        # Check status
        if isinstance(out, (list, tuple)):
            success = all(o.caption for o in out)
        else:
            success = bool(out.caption)

        return out, success
