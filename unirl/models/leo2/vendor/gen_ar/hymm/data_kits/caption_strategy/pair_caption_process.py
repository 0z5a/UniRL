# =======================================================
# Caption process script for instruction-based captions
# =======================================================

import json
from .caption_process_v3 import SEPARATORS, CaptionAug as BaseCaptionAug


class CaptionAug(BaseCaptionAug):

    def __init__(self,
                 caption_sample_ratio=None,
                 logger=None,
                 force_use_style_tag=False,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
            logger : logger object
            force_use_style_tag : 如果为 True, 且存在有效的 style tag, 则忽略 style tag 的概率, 一定会添加到 caption 中.
                摄影风格除外 (仍然遵循概率采样).
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.caption_sample_ratio = caption_sample_ratio
        if isinstance(caption_sample_ratio, str):
            self.caption_sample_ratio = json.loads(caption_sample_ratio)

        self.force_use_style_tag = force_use_style_tag

        # Predefined keys - 使用instruction相关的键
        self.caption_keys = {"instruction", "instruction_short", "instruction_medium", "instruction_long", "instruction_template", "instruction_think", "instruction_change_short", "instruction_change_medium", "instruction_change_long", "instruction_rewritten", "instruction_relation_gemini"}
        self.tag_keys = {'instruction_keep'}
        self.predefined_keys = self.caption_keys.union(self.tag_keys)

        # 检查 short_caption 和 medium_caption 是否被使用, 防止apply_caption_strategy被触发
        assert "short_caption" not in self.caption_keys and "medium_caption" not in self.caption_keys, "short_caption and medium_caption are not allowed in pair_caption_process"

        # User-defined caption sample ratio(csr) keys
        csr_keys = set([key for key, value in self.caption_sample_ratio.items() if value > 0])
        if csr_keys - self.predefined_keys:
            raise NotImplementedError(f"Unexpected keys in caption_sample_ratio: {csr_keys - self.predefined_keys}")
        self.valid_caption_keys = csr_keys.intersection(self.caption_keys)
        self.valid_tag_keys = csr_keys.intersection(self.tag_keys)

        self.logger = logger
        self.logger.info(
            "CaptionAug using caption sample ratio: {}".format(json.dumps(self.caption_sample_ratio))
        )
        self.logger.info(f"short_caption and medium_caption are not allowed in pair_caption_process")
        
    def apply_tag_strategy(self, tag_candidates, lang, caption, caption_key):
        # 进这个函数的 tag 都是要在当前样本中保留的.
        # 这个函数负责处理 tag 的格式和内容, 并返回一个列表, 列表中的元素是要添加到 caption 中的 tag 字符串.
        output = []
        for key, value in tag_candidates.items():

            if key == "instruction_keep" and caption_key == 'instruction_change_long':
                output.append(value.rstrip(SEPARATORS[lang]["period"]) + SEPARATORS[lang]["period"])


            else:
                # 其他没有特殊处理, 直接返回
                output.append(value)
        return output