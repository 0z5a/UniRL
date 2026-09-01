import sys
import random
from typing import TYPE_CHECKING, Union, Optional

from loguru import logger
if TYPE_CHECKING:
    from index_kits import ArrowIndexV2, MultiIndexV2

from ...utils.image_base import ImageTensor
from .data_utils import DataMixin
from ..caption_strategy.caption_base import CaptionOut
from ..caption_strategy.manager import MultiCaptionManager
from ..caption_strategy.prompt_patcher import apply_prompt_patchers


class ImageCaptionMixin(DataMixin):
    """ A mixin class for image captioning strategies. """
    log_prefix: str
    task_kwargs: dict
    index_kwargs: dict
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"
    image_caption_manager: MultiCaptionManager
    recaption_cot: Optional[str] = None

    def setup_image_caption(self, args):
        # task kwargs: single_text_max_length, prompt_token_length, patcher_names, recaption_token_length, reasoning_token_length, cot_kwargs
        # cot_kwargs: recaption_cot_prob, recaption_cot, recaption_cot_kwargs, reasoning_cot_prob
        _ = args
        # use single_text_max_length to skip extremely long text to avoid tokenization timeout
        self.single_text_max_length = self.task_kwargs.get("single_text_max_length", 0)
        # Image Caption configs and manager
        self.image_prompt_token_length = self.task_kwargs.get('prompt_token_length', 256)
        self.patcher_names = self.task_kwargs.get('patcher_names', [])
        self.image_caption_manager = MultiCaptionManager(
            resource=self.task_kwargs.get('image_caption_resource'),
            dataset=self,
        )

        # Whether to predict user prompt tokens
        self.predict_prompt = self.task_kwargs.get('predict_prompt', False)
        # Instruction candidates type (for instruction tuning).
        instruct_type = self.task_kwargs.get("instruct_type", "none").lower()
        self.instruct_type, *extra = instruct_type.split('@')
        self.instruct_prob = float(extra[0]) if extra else 1.0
        assert self.instruct_type in {"fixed_set", "none"}, \
            f"Unsupported t2i_instruct_type: {self.instruct_type}"

        # Image Caption CoT configs
        self.image_recaption_token_length = self.task_kwargs.get('recaption_token_length', 0)
        self.image_reasoning_token_length = self.task_kwargs.get('reasoning_token_length', 0)

        self.cot_kwargs = self.task_kwargs.get("cot_kwargs", {})
        self.recaption_cot_prob = self.cot_kwargs.get("recaption_cot_prob", 0)
        if self.recaption_cot_prob > 0:
            self.require_configs(self.cot_kwargs, "recaption_cot", "recaption cot")
            self.recaption_cot = self.cot_kwargs["recaption_cot"]
            assert isinstance(self.recaption_cot, str), "caption_cot must be a string pattern."
            assert self.image_recaption_token_length > 0, \
                f"When recaption_cot_prob > 0, recaption_token_length should be greater than 0."

        self.reasoning_cot_prob = self.cot_kwargs.get('reasoning_cot_prob', 0)
        if self.reasoning_cot_prob > 0:
            assert self.require_configs(self.index_columns, ["think_en_col", "think_zh_col"], "t2i reasoning cot", do_assert=False) or self.require_configs(self, ["think_en_key", "think_zh_key"], "t2i reasoning cot", do_assert=False)
            assert self.image_reasoning_token_length > 0, \
                f"When reasoning_cot_prob > 0, reasoning_token_length should be greater than 0."

        # Total text length
        self.image_text_token_length = self.image_prompt_token_length + self.image_recaption_token_length + self.image_reasoning_token_length

    def get_instruct(self, candidates):
        if self.instruct_type == "none":
            return ""
        # fixed_set
        selected = random.choice(candidates).strip() + " "
        if self.instruct_prob < 1.0 and random.random() > self.instruct_prob:
            return ""
        return selected

    def get_image_caption(self, ind, image: ImageTensor | None = None) -> tuple[CaptionOut | list[CaptionOut], bool]:
        """ Get image caption for a given index. """
        try:
            if self.recaption_cot_prob > 0 and random.random() < self.recaption_cot_prob:
                recaption_cot = self.recaption_cot
            else:
                recaption_cot = None
            out = self.image_caption_manager.get_caption(ind, return_dict=True, pattern=recaption_cot)
        except Exception as e:
            line_no = sys._getframe().f_lineno
            logger.error(f"L{line_no} <- get_image_caption | {e.__class__.__name__}: {str(e)} (index={ind})")
            out = CaptionOut(caption=None, lang=None)

        # Apply prompt patchers
        if self.patcher_names and ((isinstance(out, (list, tuple)) and all(o.caption is not None and o.lang is not None for o in out)) or (isinstance(out, CaptionOut) and out.caption is not None and out.lang is not None)):
            lang = out[0].lang if isinstance(out, (list, tuple)) else out.lang
            out = apply_prompt_patchers(
                out,
                self.patcher_names,
                lang=lang,
                dataset=self,
                index=ind,
                # kwargs
                image=image,
            )

        # Check status
        if isinstance(out, (list, tuple)):
            success = all(o.caption for o in out)
        else:
            success = bool(out.caption)

        return out, success
