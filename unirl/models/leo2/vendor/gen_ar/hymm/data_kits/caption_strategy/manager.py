import json
import random
from typing import Union, List
import re

from . import load_caption_processor
from .caption_base import CaptionOut
from .tag_templates import style_v2_zh2en
from .ocr_templates import OCR_TEMPLATES_ZH, OCR_TEMPLATES_EN
from ..utils.index_utils import IndexColumn


def apply_ocr_template(text, ocr_words, lang):
    if all([word in text for word in ocr_words]):
        return text # 已经出现在short/medium caption中，不需要再添加

    if lang == 'zh':
        prompt = random.choice(OCR_TEMPLATES_ZH)
        split_seq = random.choice([",", ", ", "，"])
        ocr_words = [f'“{word}”' for word in ocr_words]
    elif lang == 'en':
        prompt = random.choice(OCR_TEMPLATES_EN)
        split_seq = random.choice([",", ", "])
        ocr_words = [f'"{word}"' for word in ocr_words]
    else:
        raise ValueError(f"Invalid language: {lang}")

    ocr_texts = split_seq.join(ocr_words)
    prompt = prompt.format(text=text, ocr_words=ocr_texts)
    return prompt


def get_ocr_words(text, lang=None):
    # 从 long_long_caption 中提取 ocr 文字, return a list of ocr words
    pattern = r'[“"]([^“”"]+)[”"]'
    matches = re.findall(pattern, text)
    return matches


class GroupCaption(object):
    def __init__(self, resource, dataset, log_error=True):
        self.dataset = dataset

        src = resource["src"]
        prob_dict = json.loads(src)
        # `message` datasets keep the caption inline in the message dict, so `src` keys name
        # message fields and have no Arrow column to register.
        message_format = (getattr(self.dataset, "task_kwargs", None) or {}).get("data_format") == "message"
        # Make sure the key of probs are all registered columns
        self.cols = []
        for key in prob_dict.keys():
            if hasattr(self.dataset, "index_columns") and key in self.dataset.index_columns:
                self.cols.append(self.dataset.index_columns[key])
            elif hasattr(self.dataset, key):
                self.cols.append(getattr(self.dataset, key))
            elif message_format:
                self.cols.append(IndexColumn(key))
            else:
                raise ValueError(f"Caption column {key} is not registered as IndexColumn.")
        self.probs = list(prob_dict.values())
        self.caption_aug = self.load_caption_processor(resource)
        self.extra_processor = resource.get("extra_processor", [])
        self.data_format = resource.get("data_format", "json") # "json" or "plain"
        assert self.data_format in {"json", "plain"}, f"Unsupported data_format: {self.data_format}"
        self.log_error = log_error

        if "fix_style" in self.extra_processor:
            self.name2col = {}
            for col in self.cols:
                if "_en" in col.key:
                    self.name2col["en"] = col
                elif "_zh" in col.key or "_cn" in col.key:
                    self.name2col["zh"] = col
            assert len(self.name2col) == 2, \
                f"fix_style requires both en and zh captions, got {self.cols}"

        elif "filter_shot" in self.extra_processor or "inject_style_v2" in self.extra_processor:
            # 确保有 style_v2_col 属性
            error_msg = "`filter_shot/inject_style_v2` processor requires `style_v2_col` IndexColumn, but not defined."
            if hasattr(self.dataset, "index_columns") and "style_v2_col" not in self.dataset.index_columns:
                raise ValueError(error_msg)
            elif not hasattr(self.dataset, "index_columns") and not hasattr(self.dataset, "extra_style_v2_col"):
                raise ValueError(error_msg)

        elif "inject_camera" in self.extra_processor:
            # 确保有 camera_movement_col 和 camera_movement_special_col 属性
            error_msg = ("`inject_camera` processor requires `camera_movement_col` and `camera_movement_special_col` "
                         "IndexColumns, but not defined.")
            assert hasattr(self.dataset, "index_columns"), \
                "Dataset must have `index_columns` attribute to use `inject_camera` processor."
            if ("camera_movement_col" not in self.dataset.index_columns
                    or "camera_movement_special_col" not in self.dataset.index_columns):
                raise ValueError(error_msg)

    def __repr__(self):
        return f"GroupCaption(cols={self.cols}, probs={self.probs}, extra_processor={self.extra_processor})"

    def load_caption_processor(self, resource):
        # specific processor
        if 'caption_processor' in resource:
            if resource.get('caption_sample_ratio') is not None:
                if isinstance(resource['caption_sample_ratio'], str):
                    caption_sample_ratio = json.loads(resource['caption_sample_ratio'])
                elif isinstance(resource['caption_sample_ratio'], dict):
                    caption_sample_ratio = resource['caption_sample_ratio']
                else:
                    raise TypeError(
                        f"`caption_sample_ratio` must be a JSON string or dict, got {type(resource['caption_sample_ratio'])}.")
            else:
                caption_sample_ratio = None
            caption_aug = load_caption_processor(
                name=resource['caption_processor'],
                caption_sample_ratio=caption_sample_ratio,
                logger=self.dataset.logger,
                kwargs=resource.get('caption_processor_kwargs'),
            )
        else:
            caption_aug = None
        return caption_aug

    def fix_style(self, index):
        # 修复 caption v3 的风格:
        #   1. 英文的摄影风格不如中文的准, 所以如果英文 style 中有 photo 关键词, 但是中文 style 中没有 摄影 关键词,
        #      则将英文 style 置空.
        try:
            cap_en = self.dataset.index_manager.get_attribute(index, **self.name2col["en"])
            cap_en = json.loads(cap_en)
            cap_zh = self.dataset.index_manager.get_attribute(index, **self.name2col["zh"])
            cap_zh = json.loads(cap_zh)
            style_en = cap_en['style']
            style_zh = cap_zh['style']
            if 'photo' in style_en.lower() and '摄影' not in style_zh:
                cap_en['style'] = ''
        except Exception as e:
            self.dataset.logger.error(f"{e.__class__.__name__}: {e}")
            return dict(en=None, zh=None)
        return dict(en=cap_en, zh=cap_zh)

    def filter_shot(self, index, lang, caption):
        # 过滤 caption 的 shot
        #   1. 过滤景别, 让景别 tag 只在摄影类的数据中出现.
        #   2. 需要 style_v2 字段
        try:
            if hasattr(self.dataset, "index_columns") and "style_v2_col" in self.dataset.index_columns:
                style_v2_col = self.dataset.index_columns["style_v2_col"]
            else:
                style_v2_col = self.dataset.extra_style_v2_col
            style = self.dataset.index_manager.get_attribute(index, **style_v2_col)
            if "摄影" not in str(style):  # 强制把 style 转成 str, 可以 cover 住 None 和 list 类型.
                caption["shot_type"] = ""
        except (KeyError, FileNotFoundError):
            pass
        except Exception as e:
            self.dataset.logger.error(f"(index={index}) {e.__class__.__name__}: {e}")
        return caption

    def inject_style_v2(self, index, lang, caption):
        # 把 style_v2 字段注入到 caption 中. 注意 style_v2 本身是中文.
        try:
            if hasattr(self.dataset, "index_columns") and "style_v2_col" in self.dataset.index_columns:
                style_v2_col = self.dataset.index_columns["style_v2_col"]
            else:
                style_v2_col = self.dataset.extra_style_v2_col
            style = self.dataset.index_manager.get_attribute(index, **style_v2_col)
            if style: # not None and not empty string
                if style in style_v2_zh2en:
                    value = style_v2_zh2en[style]
                    # 如果 style 中有 / , 则认为是多选项, 需要随机选择一个.
                    if isinstance(value, dict):
                        candidates = value[lang]
                    else:
                        candidates = [style] if lang == "zh" else value
                    sel_style = random.choice(candidates) if len(candidates) > 1 else candidates[0]
                    caption["style"] = sel_style
        except (KeyError, FileNotFoundError):
            pass
        except Exception as e:
            self.dataset.logger.error(f"(index={index}) {e.__class__.__name__}: {e}")
        return caption
    
    def inject_ocr(self, index, lang, caption):
        try:
            long_long_caption = caption['long_long_caption']
            ocr_words = get_ocr_words(long_long_caption, lang)

            short_caption = apply_ocr_template(caption['short_caption'], ocr_words, lang)
            medium_caption = apply_ocr_template(caption['medium_caption'], ocr_words, lang)

            caption['short_caption'] = short_caption
            caption['medium_caption'] = medium_caption
        except (KeyError, FileNotFoundError):
            pass
        except Exception as e:
            self.dataset.logger.error(f"(index={index}) {e.__class__.__name__}: {e}")

        return caption

    def _select_col(self, index):
        """Select a column for reading caption data.

        When ``select_nonempty`` is in extra_processor, iterate through all
        columns (in shuffled order) and return the first one whose data is
        non-empty.  Otherwise fall back to weighted random selection.
        """
        if "select_nonempty" in self.extra_processor:
            cols_shuffled = list(self.cols)
            random.shuffle(cols_shuffled)
            for col in cols_shuffled:
                try:
                    data = self.dataset.index_manager.get_attribute(index, **col)
                    if data:
                        return col
                except Exception:
                    continue
        if len(self.cols) > 0:
            return random.choices(self.cols, weights=self.probs, k=1)[0]
        return self.cols[0]

    def inject_camera(self, index, lang, caption):
        try:
            columns = self.dataset.index_manager.get_columns(index)
            camera_movement_col = self.dataset.index_columns["camera_movement_col"]
            camera_movement = self.dataset.index_manager.get_attribute(index, **camera_movement_col) \
                if camera_movement_col.key in columns else None
            camera_movement_special_col = self.dataset.index_columns["camera_movement_special_col"]
            camera_movement_special = self.dataset.index_manager.get_attribute(index, **camera_movement_special_col) \
                if camera_movement_special_col.key in columns else None
            caption['_inject_camera_'] = dict(
                camera_movement=camera_movement,
                camera_movement_special=camera_movement_special,
            )
        except (KeyError, FileNotFoundError):
            pass
        except Exception as e:
            self.dataset.logger.error(f"(index={index}) {e.__class__.__name__}: {e}")
        return caption

    def get_caption(self, index, return_dict=False, pattern=None) -> Union[str, CaptionOut, List[CaptionOut]]:
        sel_col = self._select_col(index)
        lang = "en" if "_en" in sel_col.key else "zh"

        out = CaptionOut(lang=lang)
        return_key = return_dict

        try:
            if "fix_style" in self.extra_processor:
                raw_caption_data = self.fix_style(index)[lang]
            else:
                raw_caption_data = self.dataset.index_manager.get_attribute(index, **sel_col)
                if self.data_format == "json":
                    raw_caption_data = json.loads(raw_caption_data)
            # raw_caption_data: either a dict or a string.

            # 这里还是 caption_dict, 可以设置一些 tag 的过滤规则 (filter_*), 避免错误 tag 拼到 caption 上.
            for filter_key in self.extra_processor:
                if filter_key.startswith("filter_"):
                    raw_caption_data = getattr(self, filter_key)(index, lang, raw_caption_data)

            # 这里还是 caption_dict, 可以设置一些注入规则 (inject_*), 代替 caption 中的字段.
            for inject_key in self.extra_processor:
                if inject_key.startswith("inject_"):
                    raw_caption_data = getattr(self, inject_key)(index, lang, raw_caption_data)

            if raw_caption_data:
                if return_key:
                    out = self.caption_aug.caption_aug(raw_caption_data, lang, return_key=True, pattern=pattern)
                else:
                    out.caption = self.caption_aug.caption_aug(raw_caption_data, lang)
            else:
                out.caption = None
        except (KeyError, FileNotFoundError):
            out.caption = None
        except Exception as e:
            if self.log_error:
                self.dataset.logger.error(f"(index={index}, sel_col={sel_col}) {e.__class__.__name__}: {e}")
            out.caption = None

        if return_dict:
            return out

        return out.caption
    
    def get_pair_caption(self, caption_dict, return_dict=False, pattern=None) -> Union[str, CaptionOut, List[CaptionOut]]:
        valid_cols, valid_probs = [], []
        for col, prob in zip(self.cols, self.probs):
            if col.key in caption_dict and caption_dict[col.key]:
                valid_cols.append(col)
                valid_probs.append(prob)
        if len(self.cols) > 0 and len(valid_cols) > 0 and sum(valid_probs) > 0:
            sel_col = random.choices(valid_cols, weights=valid_probs, k=1)[0]
        else:
            sel_col = self.cols[0]
        lang = "en" if "_en" in sel_col.key else "zh"

        out = CaptionOut(lang=lang)
        return_key = return_dict

        try:
            caption_dict = caption_dict[sel_col.key]
            if caption_dict:
                if return_key:
                    out = self.caption_aug.caption_aug(caption_dict, lang, return_key=True, pattern=pattern)
                else:
                    out.caption = self.caption_aug.caption_aug(caption_dict, lang)
            else:
                out.caption = None
        except (KeyError, FileNotFoundError):
            out.caption = None
        except Exception as e:
            if self.log_error:
                # `caption_dict` holds the raw caption by now; keep it out of the log.
                self.dataset.logger.error(f"(sel_col={sel_col.key}) {e.__class__.__name__}: {str(e)[:200]}")
            out.caption = None

        if return_dict:
            if isinstance(out, (tuple, list)):
                for ot in out:
                    ot.sel_col = sel_col.key
            else:
                out.sel_col = sel_col.key
            return out

        return out.caption


class MultiCaptionManager(object):
    def __init__(self, resource, dataset, caption_processor=None, caption_sample_ratio=None):
        if resource is None:
            self.enabled = False
        else:
            self.enabled = True
            self.dataset = dataset
            self.groups = []

            if isinstance(resource, str):
                self.groups.append(GroupCaption({
                    "src": resource,
                    "caption_processor": caption_processor,
                    "caption_sample_ratio": caption_sample_ratio,
                }, dataset))
            elif isinstance(resource, list):
                # If image_caption_col_probs is a list, we assume it is a list of candidate caption columns
                # and their probabilities. We will sequentially try to use the first available valid caption.
                # If multiple caption candidates are provided, we will not log errors anymore and the user
                # should make sure at least one of them is valid.
                log_error = len(resource) == 1
                for item in resource:
                    assert isinstance(item, dict), \
                        f"Each item in image_caption_col_probs must be a dict, got {item}"
                    # key eg: caption_v3
                    key = list(item.keys())[0]
                    # stripped value eg: {"src": "{"image_caption_col": 0.5, "image_caption_col_2": 0.5}", "caption_processor": "caption_process_v3", "caption_sample_ratio": "{"long_long_caption":0.3,"long_caption":0.35,"medium_caption":0.25,"short_caption":0.1,"background":0.1,"style":0.7}", "extra_processor": ["fix_style"]}
                    value = self.dataset.strip_leading_tag(item[key], required=True, tag=key)
                    self.groups.append(GroupCaption(value, dataset, log_error=log_error))
            else:
                raise TypeError(f"`image_caption_col_probs` must be a JSON string, got {type(resource)}.")

            if len(self.groups) == 0:
                raise ValueError("`image_caption_col_probs` must have at least one group.")

    def __repr__(self):
        return f"MultiCaptionManager(groups={self.groups})"

    def get_caption(
            self, index, return_dict=False, pattern=None, real_index=None
    ) -> Union[CaptionOut, tuple[CaptionOut], str]:
        if pattern:
            assert return_dict, f"`return_dict` is required when `pattern` is specified."
        text = None
        caption_out = None
        for group in self.groups:
            if isinstance(index, dict) or isinstance(index, str):
                caption_out = group.get_pair_caption(index, return_dict=return_dict, pattern=pattern)
            else:
                caption_out = group.get_caption(index, return_dict=return_dict, pattern=pattern)

            if return_dict:
                if isinstance(caption_out, (tuple, list)):
                    text = [out.caption for out in caption_out]
                    if any([t is None for t in text]):
                        text = None
                else:
                    text = caption_out.caption
            else:
                text = caption_out

            # Detect invalid text that contains non-utf-8 characters, which may cause tokenizer error
            try:
                if isinstance(text, str):
                    _ = text.encode("utf-8")
                elif isinstance(text, list):
                    for t in text:
                        _ = t.encode("utf-8")
            except UnicodeEncodeError as e:
                self.dataset.logger.error(f"({index=}, {real_index=}) {e.__class__.__name__}: {e}")
                text = None

            if text is not None:
                break

        if text is None:
            caption_out = CaptionOut(caption=None, lang=None)
            # For message datasets `index` is the whole message dict; `real_index` identifies the sample.
            index_repr = "<message>" if isinstance(index, dict) else index
            self.dataset.logger.error(f"(index={index_repr}, {real_index=}) No valid caption found.")

        return caption_out
