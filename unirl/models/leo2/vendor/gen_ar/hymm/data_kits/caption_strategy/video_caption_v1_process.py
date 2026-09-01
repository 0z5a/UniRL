
# =======================================================
# Caption process script used for new_caption_v1_zh and
# new_caption_v1_en 
# new caption structure:
# {
#     "style_features": str,
#     "content_summary": str,
#     "background_audio": str,
#     "shots": [
#         {
#             "time_range": [start, end],
#             "static_description": str,
#             "dynamic_description": str,
#         }
#     ],
#     "tags": {...}
# }
# caption assembly:
# [style_features?] → content_summary → background_audio
#                       → [shot_1 → shot_2 → ...]?
# each shot internally:
# [time_range?], static_description . dynamic_description
# shots rules:
#   - include with probability caption_sample_ratio["shots"], default 1.0
#   - when not included, all shot-level text is omitted
# time_range rules:
#   - len(shots) >= 2 → always include time_range for every shot
#   - len(shots) == 1 → include with probability caption_sample_ratio["time_range"]
# =======================================================

import json
import random
import re

from .caption_base import CaptionOut
from ..utils.text_utils import has_repeat


SEPARATORS = {
    "zh": {"comma": "，", "period": "。"},
    "en": {"comma": ", ", "period": ". "}
}


class CaptionAug:

    def __init__(self,
                 caption_sample_ratio=None,
                 ocr_only_long_caption=False,
                 num_replace_rate=0,
                 random_caption_tag_order=False,
                 filter_repeat=True,
                 background_audio_filter_meaningless_string=False,
                 logger=None,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        # background_audio 字段的后处理参数。
        #   - background_audio_filter_meaningless_string: 若 background_audio（去掉尾部标点/
        #     空白后）严格等于 _BG_AUDIO_MEANINGLESS 中某一项，则丢弃该字段，不拼进最终 caption。
        #     注意只做严格匹配——短语后无标点，或紧跟 "." / ";" / "," 等结尾标点都会被识别；
        #     但短语后面若还接了真实描述，则不算无意义、原样保留。
        self.background_audio_filter_meaningless_string = background_audio_filter_meaningless_string

        self.caption_sample_ratio = caption_sample_ratio
        if isinstance(caption_sample_ratio, str):
            self.caption_sample_ratio = json.loads(caption_sample_ratio)
        # content_summary / background_audio / static_description / dynamic_description
        # 是"必取"字段，对应概率只能是 1.0；缺省时自动填 1.0，显式配成非 1.0 直接报错。
        _MANDATORY_KEYS = (
            "content_summary",
            # "background_audio",
            "static_description",
            "dynamic_description",
        )
        if isinstance(self.caption_sample_ratio, dict):
            for _k in _MANDATORY_KEYS:
                if _k in self.caption_sample_ratio and self.caption_sample_ratio[_k] != 1.0:
                    raise ValueError(
                        f"caption_sample_ratio['{_k}'] must be 1.0 (mandatory field), "
                        f"got {self.caption_sample_ratio[_k]!r}"
                    )
                self.caption_sample_ratio.setdefault(_k, 1.0)
            self.caption_sample_ratio.setdefault("background_audio", 0.0)
            self.caption_sample_ratio.setdefault("shots", 1.0)
            # shot_ordinal 是控制字段：以此概率将镜头前缀的时间戳替换成
            # “第一个镜头 / 镜头一 / shot 1”这类序号标签。缺省 0.0 表示关闭。
            self.caption_sample_ratio.setdefault("shot_ordinal", 0.0)

        # Predefined keys
        self.caption_keys = {"style_features", "content_summary", "background_audio"}
        # shots是控制字段，用于控制shots的采样概率
        self.shot_control_keys = {"shots"}
        # shot_ordinal 是控制字段：以此概率将镜头前缀的时间戳替换为序号标签
        self.shot_keys = {"time_range", "shot_ordinal", "static_description", "dynamic_description"}
        # tag_keys暂时不需要
        self.tag_keys = {"ip_tag", "audio_tag", "music_tag", "language_dialect_tag", "visual_realism", "overlay_tag"}
        self.predefined_keys = self.caption_keys.union(self.shot_control_keys, self.shot_keys, self.tag_keys)

        # User-defined caption sample ratio(csr) keys
        csr_keys = set([key for key, value in self.caption_sample_ratio.items() if value > 0])
        if csr_keys - self.predefined_keys:
            raise NotImplementedError(f"Unexpected keys in caption_sample_ratio: {csr_keys - self.predefined_keys}")
        self.valid_caption_keys = csr_keys.intersection(self.caption_keys)
        self.valid_shot_keys = csr_keys.intersection(self.shot_keys)
        self.valid_tag_keys = csr_keys.intersection(self.tag_keys)

        self.logger = logger
        self.logger.info(
            "CaptionAug using caption sample ratio: {}".format(json.dumps(self.caption_sample_ratio))
        )
        self.ocr_only_long_caption = ocr_only_long_caption
        self.num_replace_rate = num_replace_rate
        self.random_caption_tag_order = random_caption_tag_order
        self.filter_repeat = filter_repeat

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

        if "caption" in text:
            if isinstance(text["caption"], str):
                text = json.loads(text["caption"])
            else:
                text = text["caption"]

        # reload if still string
        if isinstance(text, str):
            try:
                text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        return text

    def random_replace_num(self, long_short_caption):
        """是否随机替换数字"""
        if self.num_replace_rate == 0 or self.ocr_in_caption(long_short_caption):
            return long_short_caption

        num_tag_map = {
            # "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",

            # "二": "2",
            # "三": "3",
            # "四": "4",
            # "五": "5",
            # "六": "6",
            # "七": "7",
            # "八": "8",
            # "九": "9",
            # "十": "10",
        }

        # 使用正则表达式匹配单词边界
        def replace_num(match):
            word = match.group(0).lower()
            if word in num_tag_map:
                # 对每个匹配到的数字按概率决定是否替换
                if random.random() < self.num_replace_rate:
                    return num_tag_map[word]
            return match.group(0)

        # 使用正则表达式替换，\b表示单词边界，这里中文匹配不生效
        pattern = r'\b(' + '|'.join(num_tag_map.keys()) + r')\b'
        new_caption = re.sub(pattern, replace_num, long_short_caption, flags=re.IGNORECASE)
        return new_caption

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

    def ocr_in_caption(self, caption):

        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, caption)
        result = [match[0] or match[1] for match in matches]

        if result:
            return True
        return False

    # ------------------------------------------------------------------
    # background_audio processing
    # ------------------------------------------------------------------
    # 视为"无意义"的 background_audio：整段（去掉尾部标点/空白后）严格等于列表中某一项时，
    # 命中后整段字段丢弃、不拼进最终 caption。
    # 注意：只做严格匹配——短语后面无标点，或紧跟 "." / ";" / "," 等结尾标点都会被识别；
    # 但如果短语后面还接了真实描述（如 "..., faint outdoor sounds"），则不算无意义、原样保留。
    # (None / 字段缺失当作空串处理)
    _BG_AUDIO_MEANINGLESS = (
        "",
        "无",
        "无背景音乐",
        "None",
        "<null_或_无background_audio>",
        "No discernible background music",
        "No discernible background audio",
        "No significant background music",
        "No significant background audio",
        "No background music",
        "No ambient sound",
        "No ambient sound throughout",
        "No audio track",
        "No audio track throughout",
        "No background ambient sound",
        "No background ambient sound throughout",
        "No background audio",
        "No background audio throughout",
        "No background audio track",
        "No background audio track throughout",
        "No background audio tracks",
        "No background music or ambient noise",
        "No background music or ambient sound",
        "No background music or ambient sound effects",
        "No background music or ambient sound effects throughout",
        "No background music or ambient sound throughout",
        "No background music or noticeable ambient noise",
        "No background music or noticeable ambient noise throughout",
        "No background music or noticeable ambient sound",
        "No background music or noticeable ambient sound effects",
        "No background music or noticeable ambient sound effects throughout",
        "No background music or noticeable ambient sound throughout",
        "No background music or significant ambient noise",
        "No background music or significant ambient sound",
        "No background music or significant ambient sound effects",
        "No background music or significant ambient sound effects throughout",
        "No background music or sound effects",
        "No background music or sound effects throughout",
        "No background music throughout",
        "No background noise",
        "No background noise throughout",
        "No background sound effects",
        "No background sound throughout",
        "No continuous background audio",
        "No continuous background audio throughout",
        "No continuous background audio track",
        "No continuous background audio track throughout",
        "No continuous background music",
        "No continuous background music throughout",
        "No continuous background music track",
        "No continuous background track",
        "No dialogue",
        "No dialogue throughout",
        "No dialogue track",
        "No discernible ambient noise",
        "No discernible ambient sound",
        "No discernible ambient sound throughout",
        "No discernible ambient sounds",
        "No discernible background audio",
        "No discernible background audio throughout",
        "No discernible background audio track",
        "No discernible background audio track throughout",
        "No discernible background music or ambient noise",
        "No discernible background music or ambient noise throughout",
        "No discernible background music or ambient sound",
        "No discernible background music or ambient sound effects",
        "No discernible background music or ambient sound throughout",
        "No discernible background music or continuous ambient sound",
        "No discernible background music or sound effects",
        "No discernible background music throughout",
        "No discernible background noise",
        "No discernible background noise throughout",
        "No discernible background sound",
        "No discernible sound",
        "No discernible sound throughout",
        "No distinct ambient sound",
        "No distinct ambient sound throughout",
        "No distinct ambient sounds",
        "No distinct ambient sounds throughout",
        "No distinct background audio",
        "No distinct background audio throughout",
        "No distinct background audio track",
        "No distinct background music",
        "No distinct background music or ambient sound",
        "No instrumental accompaniment",
        "No other significant background sounds",
        "No prominent background audio",
        "No prominent background audio throughout",
        "No prominent background audio track",
        "No prominent background audio track throughout",
        "No prominent background music",
        "No prominent background music throughout",
        "No prominent background noise",
        "No significant ambient background audio throughout",
        "No significant ambient background noise",
        "No significant ambient background noise throughout",
        "No significant ambient noise",
        "No significant ambient noise is present",
        "No significant ambient noise throughout",
        "No significant ambient sound",
        "No significant ambient sound throughout",
        "No significant ambient sounds",
        "No significant ambient sounds throughout",
        "No significant background ambient sound",
        "No significant background ambient sound throughout",
        "No significant background audio throughout",
        "No significant background audio track",
        "No significant background audio track throughout",
        "No significant background audio tracks",
        "No significant background noise",
        "No significant background noise is present",
        "No significant background noise is present throughout",
        "No significant background noise throughout",
        "No significant background sound",
        "No significant background sound effects",
        "No significant background sound effects throughout",
        "No significant background sound throughout",
        "No significant background sounds",
        "No significant background sounds throughout",
        "No significant continuous background audio throughout",
        "No significant continuous background noise throughout",
        "No significant reverb",
    )

    # 匹配前归一化使用的标点符号集合：去掉字符串尾部的标点与空白。
    _PUNCT_CHARS = r",\.;:!?，。；：！？、"
    _TRAILING_PUNCT_RE = re.compile(r"[\s" + _PUNCT_CHARS + r"]+$")

    _BG_AUDIO_MEANINGLESS_SET = frozenset(_BG_AUDIO_MEANINGLESS)

    def _process_background_audio(self, value):
        """按 background_audio_filter_meaningless_string 处理 background_audio 字段。

        Returns
        -------
        str or None
            返回处理后的字符串；若该字段被判定为"无意义"应整段丢弃，返回 None。
        """
        if value is None:
            text = ""
        else:
            text = str(value).strip()

        # background_audio_filter_meaningless_string: 去掉尾部标点/空白后严格匹配（短语后无
        # 标点，或紧跟 . ; , 等都可识别）；命中则丢弃整段。
        if self.background_audio_filter_meaningless_string:
            core = self._TRAILING_PUNCT_RE.sub("", text)
            if core in self._BG_AUDIO_MEANINGLESS_SET:
                return None

        return text

    # ------------------------------------------------------------------
    # time_range formatting
    # ------------------------------------------------------------------
    # Per-language candidate units for time values.
    _TIME_UNITS = {
        "zh": ("s", "秒"),
        "en": ("s",),
    }

    @staticmethod
    def _format_seconds(value, unit="s"):
        """Render a single time value with the given suffix.

        Preserves the original decimal precision of the input. 0 -> 0s / 0秒
        """
        if value is None:
            return ""
        s = str(value).strip()
        try:
            f = float(s)
        except (TypeError, ValueError):
            return f"{s}{unit}"
        if f == 0:
            return f"0{unit}"
        return f"{s}{unit}"

    def format_time_range(self, time_range, lang):
        """Sample a natural-language rendering of a [start, end] pair."""
        if not isinstance(time_range, (list, tuple)) or len(time_range) < 2:
            return ""

        unit = random.choice(self._TIME_UNITS.get(lang, ("s",)))
        start_s = self._format_seconds(time_range[0], unit=unit)
        end_s = self._format_seconds(time_range[1], unit=unit)
        if not start_s or not end_s:
            return ""

        if lang == "zh":
            styles = [
                f"[{start_s}~{end_s}]",
                f"[{start_s}-{end_s}]",
                f"({start_s}~{end_s})",
                f"[时间：{start_s}-{end_s}]",
                f"[时段：{start_s}-{end_s}]",
                f"[从{start_s}到{end_s}]",
                f"[{start_s}至{end_s}]",
            ]
        else:
            styles = [
                f"[{start_s}~{end_s}]",
                f"[{start_s}-{end_s}]",
                f"({start_s}~{end_s})",
                f"[time: {start_s}-{end_s}]",
                f"[period: {start_s}-{end_s}]",
                f"[from {start_s} to {end_s}]",
                f"[between {start_s} and {end_s}]",
            ]
        return random.choice(styles)

    # Chinese numerals / English ordinals used by shot-ordinal prefixes.
    _ZH_NUMERALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
    _EN_ORDINALS = (
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth",
    )

    def format_shot_ordinal(self, idx, lang):
        """Sample a natural-language ordinal prefix for the ``idx``-th shot (0-based)."""
        n = idx + 1
        if lang == "zh":
            zh_num = self._ZH_NUMERALS[idx] if idx < len(self._ZH_NUMERALS) else str(n)
            styles = [
                f"[镜头{n}]",
                f"[镜头{zh_num}]",
                f"镜头{n}",
                f"镜头{zh_num}",
                f"镜头{n}：",
                f"镜头{zh_num}：",
                f"[第{n}个镜头]",
                f"[第{zh_num}个镜头]",
                f"第{n}个镜头",
                f"第{zh_num}个镜头",
                f"第{n}个镜头：",
                f"第{zh_num}个镜头：",
                f"[第{zh_num}幕]",
                f"第{zh_num}幕：",
            ]
        else:
            ordinal = self._EN_ORDINALS[idx] if idx < len(self._EN_ORDINALS) else f"{n}th"
            styles = [
                f"[Shot {n}]",
                f"[Shot{n}]",
                f"Shot {n}",
                f"Shot{n}",
                f"Shot {n}: ",
                f"[The {ordinal} shot]",
                f"The {ordinal} shot",
                f"The {ordinal} shot: ",
                f"[Clip {n}]",
                f"Clip {n}",
                f"Clip {n}: ",
            ]
        return random.choice(styles)

    @staticmethod
    def _lower_first_word(text):
        """Lowercase the first word's initial letter (English), keep ALL-CAPS acronyms."""
        if not text:
            return text
        first = text.split()[0] if text.split() else ""
        if len(first) > 1 and first.isupper():
            return text  # 保留 NASA/UFO 这类缩写
        return text[0].lower() + text[1:]

    # ------------------------------------------------------------------
    # Main compose
    # ------------------------------------------------------------------
    def random_compose(self, caption_dict, lang, return_key=False,
                       caption_keys=None, tag_keys=None, shot_keys=None,
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
        shot_keys: list, optional
            Allow user to specify which shot keys to use.
        ignore_tag_prob: bool, optional
            Only valid when tag_keys is not None. If True, ignore the caption_sample_ratio for tag keys,
            and use all available tags specified by tag_keys.
        caption_keys_prob: dict, optional
            Allow user to specify the sampling probability of each caption key. (not tag keys)
        """
        assert lang in {"zh", "en"}, f"Unsupported language: {lang}"
        sep = SEPARATORS[lang]

        if caption_keys is not None:
            assert isinstance(caption_keys, list), \
                f"caption_keys must be a list: {caption_keys}, got {type(caption_keys)}"
            valid_caption_keys = caption_keys
        else:
            valid_caption_keys = self.valid_caption_keys
        if tag_keys is not None:
            assert isinstance(tag_keys, list), \
                f"tag_keys must be a list: {tag_keys}, got {type(tag_keys)}"
            valid_tag_keys = tag_keys
        else:
            valid_tag_keys = self.valid_tag_keys
        if shot_keys is not None:
            assert isinstance(shot_keys, list), \
                f"shot_keys must be a list: {shot_keys}, got {type(shot_keys)}"
            valid_shot_keys = shot_keys
        else:
            valid_shot_keys = self.valid_shot_keys

        def _prob(key):
            if caption_keys_prob is not None and key in caption_keys_prob:
                return caption_keys_prob[key]
            return self.caption_sample_ratio.get(key, 0.0)

        # tags暂时不需要，先组装类似global caption，然后组装shots_text
        global_caption_values = {}
        tag_candidates = {}
        for key in caption_dict:
            if key in valid_caption_keys:
                value = caption_dict[key]
                if key == "background_audio":
                    # 命中"无意义"字符串会返回 None，被下面的过滤直接跳过（不拼进 caption）。
                    value = self._process_background_audio(value)
                if value and value != "" and value.lower() != "none" and value != "无":
                    if random.random() < _prob(key):
                        global_caption_values[key] = value.strip()
            elif key in valid_tag_keys:
                if random.random() < self.caption_sample_ratio[key]:
                    tag = caption_dict[key].strip()
                    if tag != "" and tag.lower() != "none" and tag != "无":
                        tag_candidates[key] = tag.strip()

        # 单镜头下，time_range是可选的（可以指定概率）
        # 多镜头，time_range是必须的，即使指定选择time_range的概率不是1，也要包含time_range
        shots = caption_dict.get("shots") or []
        if len(shots) == 0:
            raise ValueError("shots is required but empty.")
        
        shots_text = ""
        include_shots = random.random() < _prob("shots")
        if include_shots:
            num_time_ranges = sum(
                1 for s in shots if isinstance(s, dict) and s.get("time_range")
            )
            is_multi_shot = num_time_ranges > 1
            include_time_range = (
                is_multi_shot
                or (
                    "time_range" in valid_shot_keys
                    and random.random() < _prob("time_range")
                )
            )
            # 以一定概率把镜头前缀的时间戳替换成序号标签（第一个镜头/镜头一/shot 1）。
            use_shot_ordinal = (
                "shot_ordinal" in valid_shot_keys
                and random.random() < _prob("shot_ordinal")
            )

            shot_parts = []
            shot_ordinal_idx = 0
            for shot in shots:
                if not isinstance(shot, dict):
                    continue

                descs = []
                static_desc = (shot.get("static_description") or "").strip()
                if static_desc:
                    descs.append(static_desc)
                dynamic_desc = (shot.get("dynamic_description") or "").strip()
                if dynamic_desc:
                    descs.append(dynamic_desc)

                if not descs:
                    continue  # skip degenerate shot with no description

                prefix_str = ""
                if use_shot_ordinal:
                    prefix_str = self.format_shot_ordinal(shot_ordinal_idx, lang)
                elif include_time_range:
                    prefix_str = self.format_time_range(shot.get("time_range"), lang)
                
                if prefix_str:
                    desc_text = sep["period"].join(descs)
                    # 英文序号前缀后，描述首词小写（Shot 1, the camera...）。
                    if lang == "en" and use_shot_ordinal:
                        desc_text = self._lower_first_word(desc_text)
                    # 前缀若已以冒号结尾（如“镜头1：/ Shot 1:”），直接接描述，不再补逗号。
                    connector = "" if prefix_str.rstrip().endswith((":", "：")) else sep["comma"]
                    shot_text = prefix_str + connector + desc_text
                else:
                    shot_text = sep["period"].join(descs)

                shot_parts.append(shot_text)
                shot_ordinal_idx += 1

            if not shot_parts:
                raise ValueError("No valid shot descriptions in shots.")

            shots_text = sep["period"].join(shot_parts)

        caption_order = ("style_features", "content_summary", "background_audio")
        ordered = [global_caption_values[k] for k in caption_order if k in global_caption_values]
        if shots_text:
            ordered.append(shots_text)

        caption = sep["period"].join(ordered)
        caption = self.strip_zh(caption) if lang == "zh" else self.strip_en(caption)
        caption = caption.strip().replace("\\N", "").strip("，,")

        if self.filter_repeat and has_repeat(caption):
            raise ValueError(f"caption has repeat: {caption}")

        if return_key:
            # key之前是selected_key，现在改成new_caption_v1
            return CaptionOut(
                caption=caption,
                lang=lang,
                key="new_caption_v1",
                tag_keys=[],
            )
        return caption

    def caption_aug(self, raw_string: str | dict, lang, **kwargs):
        caption_dict = self.safe_load_string(raw_string)

        if kwargs.get("return_key"):
            out = self.random_compose(caption_dict, lang, return_key=True)
            out.caption = self.random_replace_num(out.caption)
            return out

        caption = self.random_compose(caption_dict, lang)
        return self.random_replace_num(caption)
