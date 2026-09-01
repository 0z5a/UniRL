# =======================================================
# Caption process script used for caption_v3_zh and
# caption_v3_en
# =======================================================

import json
import random
import re

from .tag_templates import style_templates
from .caption_base import CaptionOut

SEPARATORS = {
    "zh": {"comma": "，", "period": "。"},
    "en": {"comma": ", ", "period": ". "}
}


class CaptionAug:

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

        # Predefined keys
        self.caption_keys = {"short_caption", "medium_caption", "long_caption", "long_long_caption"}
        self.tag_keys = {"background", "shot_type", "style", "light", "atmosphere", "composition", "IP", "motion"}
        self.predefined_keys = self.caption_keys.union(self.tag_keys)

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

    @staticmethod
    def safe_load_string(text):
        """加载结构化caption数据, json or xml格式"""
        if isinstance(text, dict):
            return text
        try:
            text = json.loads(text)
        except Exception as e:
            raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        if type(text) is str:
            try:
                text = eval(text)
                if type(text) is str:
                    text = text.replace("'", "\"")
                    text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got eval str: {}".format(str(e), text))

        if not isinstance(text, dict):
            print('load caption error', text)

        return text

    @staticmethod
    def strip_zh(text):
        text = text.replace("。，", "，").replace("，。", "。").replace("。。", "。").replace("，，", "，")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip("，。 ")
        return text

    @staticmethod
    def strip_en(text):
        text = text.replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip(",. ")
        return text

    def apply_caption_strategy(self, caption_key_candidates, caption_dict, lang):
        output_keys = caption_key_candidates
        full_caption = ' '.join([caption_dict[key] for key in caption_key_candidates])
        full_caption = full_caption.lower()

        # 拼接图, 去掉 short_caption 和 medium_caption
        if (
                lang == "en" and ("collage" in full_caption or "triptych" in full_caption or "composite" in full_caption)
        ) or (
                lang == "zh" and ("拼贴" in full_caption or "三联画" in full_caption or "复合" in full_caption or "四格" in full_caption or "拼接" in full_caption or "并排" in full_caption)
        ):
            output_keys = [x for x in output_keys if x not in ["short_caption", "medium_caption"]]
            # 如果应用了规则后 output_keys 为空, 则回退
            if len(output_keys) == 0:
                output_keys = caption_key_candidates

        # OCR, 去掉 short_caption 和 medium_caption
        # if (
        #     lang == "en" and re.findall(r'"([^"]*)"', full_caption)
        # ) or (
        #     lang == "zh" and re.findall(r'“([^”]*)”', full_caption)
        # ):
        #     output_keys = [x for x in output_keys if x not in ["short_caption", "medium_caption"]]

        return output_keys

    def apply_tag_strategy(self, tag_candidates, lang, caption, caption_key):
        # 进这个函数的 tag 都是要在当前样本中保留的.
        # 这个函数负责处理 tag 的格式和内容, 并返回一个列表, 列表中的元素是要添加到 caption 中的 tag 字符串.
        output = []
        for key, value in tag_candidates.items():
            if key == "background":
                # 背景是一个句子, 完整保留即可. 如果已经有背景, 则不再添加
                if (lang == "en" and "background" in caption) or (
                        lang == "zh" and "背景" in caption):
                    continue
                output.append(value)

            elif key == "style":
                # ==================================================================
                # 下面的逻辑先注释掉了, 因为 caption 中的 "风格" 关键词可能并不是描述图片整体
                # 的风格, 容易导致误判: 特别是在 recaption 的时候会导致风格不一致.
                # ------------------------------------------------------------------
                # # 如果 caption 中已经有 风格/style 关键词, 则不再添加.
                # if (lang == "en" and "style" in caption) or (
                #         lang == "zh" and "风格" in caption):
                #     continue
                # 风格是一个词, 按一半一半的概率套一个模板变成一句话
                if random.random() < 0.5:
                    # 直接使用风格词
                    output.append(value + SEPARATORS[lang]["period"])
                else:
                    # 套模板
                    template = random.choice(style_templates[lang])
                    if lang == 'zh' and value.endswith('风格'):
                        value = value[:-2]
                    elif lang == 'en' and value.lower().endswith('style'):
                        value = value[:-5]
                    output.append(template.format(value.strip()))

            elif key == "shot_type":
                # shot_type 是一个词, 直接使用, 暂时没想好有什么模板可以套
                # 如果 style 是摄影类, 才加这个 tag.
                output.append(value.rstrip(SEPARATORS[lang]["period"]) + SEPARATORS[lang]["period"])

            elif key == "IP":
                # IP 也是一个词, 直接使用
                output.append(value.rstrip(SEPARATORS[lang]["period"]) + SEPARATORS[lang]["period"])

            else:
                # 其他没有特殊处理, 直接返回
                output.append(value)
        return output

    @staticmethod
    def has_photography_style(in_style, lang):
        photography_style = dict(
            en=["Photography Style", "Cinematography Style", "Photographic Style", "Camera Style"],
            zh=["摄影风格", "摄像风格", "摄影与摄像风格"],
        )
        if in_style.strip() in photography_style[lang]:
            return True
        return False

    def random_compose(self, caption_dict, lang, return_key=False, caption_keys=None, tag_keys=None,
                       ignore_tag_prob=False, caption_keys_prob=None):
        """

        Parameters
        ----------
        caption_dict: dict
        lang: str
            Language of the caption, either "zh" or "en".
        return_key: bool, optional
            Whether to return the key of the selected caption.
        caption_keys: list, optional
            Allow user to specify which caption keys to use.
        tag_keys: list, optional
            Allow user to specify which tag keys to use.
        ignore_tag_prob: bool, optional
            Only valid when tag_keys is not None. If True, ignore the caption_sample_ratio for tag keys,
            and use all available tags specified by tag_keys.
        caption_keys_prob: dict, optional
            Allow user to specify the sampling probability of each caption key. (not tag keys)
        """
        assert lang in {"zh", "en"}, f"Unsupported language: {lang}"
        if caption_keys is not None:
            assert isinstance(caption_keys, list), f"caption_keys must be a list: {caption_keys}, got {type(caption_keys)}"
            valid_caption_keys = caption_keys
        else:
            valid_caption_keys = self.valid_caption_keys
        if tag_keys is not None:
            assert isinstance(tag_keys, list), f"tag_keys must be a list: {tag_keys}, got {type(tag_keys)}"
            valid_tag_keys = tag_keys
        else:
            valid_tag_keys = self.valid_tag_keys

        caption_key_candidates = []
        tag_candidates = {}
        for key in caption_dict:
            if key in valid_caption_keys:
                if caption_dict[key] and caption_dict[key] != "" and caption_dict[key].lower() != "none" and caption_dict[key] != "无":
                    caption_key_candidates.append(key)
            elif key in valid_tag_keys:
                if ignore_tag_prob or random.random() < self.caption_sample_ratio[key] or (
                    # 开启 force_use_style_tag 时, 非摄影风格的 style tag 总是被添加到 caption 中
                    key == "style" and self.force_use_style_tag and not self.has_photography_style(caption_dict[key], lang)
                ):
                    tag = caption_dict[key]
                    if tag and tag != "" and tag.lower() != "none" and tag != "无":
                        tag_candidates[key] = tag.strip()

        # Apply caption key selection strategy
        caption_key_candidates = self.apply_caption_strategy(caption_key_candidates, caption_dict, lang)

        # from all_captions, select by sample ratio weights
        selected_key = random.choices(
            caption_key_candidates,
            weights=[self.caption_sample_ratio[key] for key in caption_key_candidates] if caption_keys_prob is None else [caption_keys_prob[key] for key in caption_key_candidates],
        )[0]
        caption = caption_dict[selected_key].strip()

        if len(tag_candidates) > 0:
            tag_candidates_lst = self.apply_tag_strategy(tag_candidates, lang, caption, selected_key)
            random.shuffle(tag_candidates_lst)
        else:
            tag_candidates_lst = []

        if lang == "zh":
            caption = self.strip_zh(' '.join([caption] + tag_candidates_lst))
        elif lang == "en":
            caption = self.strip_en(' '.join([caption] + tag_candidates_lst))
        else:
            raise NotImplementedError(f"Unsupported language: {lang}")

        # Remove meaningless characters
        caption = caption.strip()
        caption = caption.replace("\\N", "").strip("，,")

        if return_key:
            return CaptionOut(
                caption=caption, lang=lang, key=selected_key, tag_keys=list(tag_candidates.keys()),
            )
        return caption

    def get_multiple_captions(self, pattern, caption_dict, lang):
        # pattern example: 'short_caption/medium_caption | long_caption/long_long_caption'
        assert '|' in pattern, f"pair_pattern must contain '|', got {pattern}"
        caption_key_groups = pattern.split('|')

        for _ in range(10):
            first_candidate_keys = [x.strip() for x in caption_key_groups[0].split('/')]
            if "@" in first_candidate_keys[0]:
                caption_keys_prob = {x.split('@')[0]: float(x.split('@')[1]) for x in first_candidate_keys}
                first_candidate_keys = [x.split('@')[0] for x in first_candidate_keys]
            else:
                caption_keys_prob = None
            first_out = self.random_compose(
                caption_dict, lang, return_key=True, caption_keys=first_candidate_keys, caption_keys_prob=caption_keys_prob,
            )
            outs = (first_out,)

            if len(caption_key_groups) > 1:
                # If there is a second group, we will compose a second caption, and reuse tags (ignore_tag_prob=True)
                # of the first caption.
                for caption_key_group in caption_key_groups[1:]:
                    candidate_keys = [x.strip() for x in caption_key_group.split('/')]
                    if "@" in candidate_keys[0]:
                        caption_keys_prob = {x.split('@')[0]: float(x.split('@')[1]) for x in candidate_keys}
                        candidate_keys = [x.split('@')[0] for x in candidate_keys]
                    else:
                        caption_keys_prob = None
                    out = self.random_compose(
                        caption_dict, lang, return_key=True, caption_keys=candidate_keys, tag_keys=first_out.tag_keys,
                        ignore_tag_prob=True, caption_keys_prob=caption_keys_prob,
                    )
                    outs += (out,)

            selected_keys = [out.key for out in outs]
            if len(set(selected_keys)) == len(selected_keys):
                break
            else:
                continue

        return outs

    def caption_aug(self, raw_string, lang, **kwargs):
        caption_dict = self.safe_load_string(raw_string)
        if kwargs.get("return_key"):
            if (pattern := kwargs.get("pattern")) is not None:
                return self.get_multiple_captions(pattern, caption_dict, lang)
            else:
                return self.random_compose(caption_dict, lang, return_key=True)
        else:
            caption = self.random_compose(caption_dict, lang)
            return caption
