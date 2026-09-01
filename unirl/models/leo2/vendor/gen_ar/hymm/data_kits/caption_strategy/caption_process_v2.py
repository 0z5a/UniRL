# =======================================================
# Caption process script used for caption_v2
#
# =======================================================

import json
import random
import string
import re


# parse json to caption for QWEN
class CaptionAug:
    """
    对caption进行数据增广,包括:
    1. 配置对IP数据替换的比例, 默认配置为全部替换(长短caption)
    2. 配置比例删除文字，去除可能存在错误的文字(长caption)
    3. 对结构化caption进行随机组合。背景, 镜头, 风格, 景别, 氛围, 运镜(视频)+[长文本]/[短文本], 字幕, 高质, 影视类别

    """

    def __init__(self,
                 caption_sample_ratio=None,
                 ip_remove_rate=0,
                 text_remove_rate=0,
                 ocr_only_long_caption=False,
                 long_short_caption_mutex=True,
                 ip_tag_mapping_file=None,
                 logger=None,
                 use_recaption_template=False,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
            ip_remove_rate (float, optional): IP数据替换的比例
            text_remove_rate (float, optional): 文字替换的比例
            long_short_caption_mutex : 长短caption互斥
            ip_tag_mapping_file: IP字段映射名称
            ocr_only_long_caption: 当long caption中包含OCR时，只使用long caption
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.ip_tag_map = json.loads(open(ip_tag_mapping_file, "r").read()) if ip_tag_mapping_file else {}
        if len(self.ip_tag_map) > 0:
            self.use_ip_tag_map = True
        else:
            self.use_ip_tag_map = False
        self.ip_remove_rate = ip_remove_rate
        self.text_remove_rate = text_remove_rate
        self.caption_sample_ratio = self.normalize_dict_keys(caption_sample_ratio)

        # Predefined keys
        self.caption_keys = {"long caption", "short caption", "medium caption"}
        self.tag_keys = {"background", "shot type", "style", "light", "atmosphere", "camera movement",
                         "high quality", "movie category", 'composition', "IP"}
        self.predefined_keys = self.caption_keys.union(self.tag_keys)
        # We treat both short and medium caption as short caption
        self.short_keys = {"short caption", "medium caption"}

        # User-defined caption sample ratio(csr) keys
        csr_keys = set(self.caption_sample_ratio.keys())
        if csr_keys - self.predefined_keys:
            raise NotImplementedError(f"Unexpected keys in caption_sample_ratio: {csr_keys - self.predefined_keys}")
        self.valid_caption_keys = csr_keys.intersection(self.caption_keys)
        self.valid_tag_keys = csr_keys.intersection(self.tag_keys)
        self.valid_short_keys = csr_keys.intersection(self.short_keys)

        self.long_short_caption_mutex = long_short_caption_mutex
        self.logger = logger
        self.ocr_only_long_caption = ocr_only_long_caption

        self.logger.info(
            "CaptionAug using caption sample ratio: {} ip_remove_rate:{} text_remove_rate:{} ocr_only_long_caption:{}".format(
                json.dumps(self.caption_sample_ratio, indent=4), ip_remove_rate, text_remove_rate,
                ocr_only_long_caption))

        self.use_recaption_template = use_recaption_template

    @staticmethod
    def safe_load_string(text):
        """加载结构化caption数据, json or xml格式"""
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

    def random_drop_ip(self, long_short_caption):
        """是否随机丢弃对IP的描述"""
        if self.ip_remove_rate == 0 or random.random() > self.ip_remove_rate:
            return long_short_caption

        for key in self.ip_tag_map:
            if key in long_short_caption:
                long_short_caption = long_short_caption.replace(key, self.ip_tag_map[key])
        return long_short_caption

    def random_drop_word(self, long_caption):
        """是否随机丢弃文字部分描述"""
        if self.text_remove_rate == 0 or random.random() > self.text_remove_rate:
            return long_caption

        """有一些句号在引号中间，在引号外面的句号才split"""
        def split_by_unquoted_periods(text):
            result = []
            current_segment = ""
            in_quotes = False

            for char in text:
                if char == '"':
                    in_quotes = not in_quotes

                if char == '.' and not in_quotes:
                    result.append(current_segment)
                    current_segment = ""
                else:
                    current_segment += char

            if current_segment:
                result.append(current_segment)

            return result

        def find_ocr_and_drop(text):
            items = split_by_unquoted_periods(text)
            if len(items) == 1:
                return text

            valid_items = []
            for item in items:
                if "\"" in item and any([_x in item.lower() for _x in
                                         ["reads", "reading", "written", "title", "writes", "text", "words",
                                          "chinese characters"]]):  # 包含引号并且包含某些特殊字符串
                    continue
                valid_items.append(item)
            if len(valid_items) == 0:
                return text
            result = ".".join(valid_items)
            if result and result[-1] not in string.punctuation:
                result += '.'
            return result

        return find_ocr_and_drop(long_caption)

    def get_source_tag_mapping(self):
        source_rename_mapping = {
            "caimai_jilupian": "CMJLP",
            "1400T_video": "1400T",
            "1400T_video_anping_cos": "1400TVAC",
            "artlist_video": "ARTLIST",
            "699pic_video": "699PIC",
            "4kvideo": "4KVID",
            "highquality93w": "HQ93",
            "xpc": "XPC",
            "nature": "XPC",
            "artificial_object": "XPC",
            "artificial_scene": "XPC",
            "people": "XPC",
            "plant": "XPC",
            "transport": "XPC",
            "render": "XPC",
            "animal": "XPC",
            "logo": "XPC",
            "storyblocks_video": "STBLK",
            "pixabay_video": "PIXABAY",
            "videvo_video": "VIDEVO",
        }

        source_tag_mapping = {
            "caimai_jilupian": "Documentary",
            "1400T_video": "Cinematic",
            "1400T_video_anping_cos": "Cinematic",
            "artlist_video": "High-quality, HDR, High detailed",
            "699pic_video": "High-quality",
            "4kvideo": "4k, High-quality",
            "highquality93w": "High-quality",
            "xpc": "High-quality",
            "nature": "High-quality",
            "artificial_object": "High-quality",
            "artificial_scene": "High-quality",
            "people": "High-quality",
            "plant": "High-quality",
            "transport": "High-quality",
            "render": "High-quality",
            "animal": "High-quality",
            "logo": "High-quality",
            "storyblocks_video": "High-quality",
            "pixabay_video": "High-quality",
            "videvo_video": "High-quality",
        }

        return source_rename_mapping, source_tag_mapping

    def get_hq_text(self, source, stage=None):
        """
        high quality tag
        """
        # 仅对stage 540p, 720p生效
        if stage not in ["540p", "720p"]:
            return None
        # special case for "highquality93w"
        if source.startswith("highquality93w"):
            source = "highquality93w"

        _, source_tag_mapping = self.get_source_tag_mapping()
        for source_name, hq_tag in source_tag_mapping.items():
            if source_name == source:
                tags = hq_tag.split(", ")
                random.shuffle(tags)
                hq_tag = ", ".join(tags)
                return hq_tag
        return None

    def get_hq_source_tag(self, source):
        # special case for "highquality93w"
        if source.startswith("highquality93w"):
            source = "highquality93w"

        source_rename_mapping, _ = self.get_source_tag_mapping()
        all_hq_sources = set(source_rename_mapping.keys())
        if source in all_hq_sources:
            return f"Video Source: [{source_rename_mapping[source]}]"
        return None

    def ocr_in_caption(self, caption):
        if re.findall(r'"([^"]*)"', caption):
            return True
        return False

    def random_compose(self, caption_dict):
        """随机组合caption"""
        tag_list = []
        long_short_caption = []
        all_captions = []
        hq_source_tag = None
        for key in caption_dict:
            """
            key的几种类别：
            1. short_caption/medium_caption/long_caption 类
            2. tag 类
            """
            if key in self.valid_caption_keys:
                all_captions.append(caption_dict[key])
                if random.random() < self.caption_sample_ratio[key]:
                    long_short_caption.append(caption_dict[key])
            elif key in self.valid_tag_keys:
                if random.random() < self.caption_sample_ratio[key]:
                    tag = caption_dict[key].strip()
                    if tag == "" or tag.lower() == "none":
                        continue
                    tag_list.append(tag)
            else:
                if key not in ["caption sample frame mode"]:
                    self.logger.warning(f"Unexpected key in caption_dict: {key}")

        # 采用 recaption 模板方式构造 prompt
        if self.use_recaption_template:
            return self.apply_recaption_template(caption_dict, tag_list)

        has_long_caption = 'long caption' in caption_dict
        long_caption = caption_dict.get('long caption')
        if self.ocr_only_long_caption and has_long_caption and self.ocr_in_caption(long_caption):
            long_short_caption = [long_caption]

        else:
            # from all_captions, select by sample ratio weights
            caption_keys = sorted(list(self.valid_caption_keys.intersection(caption_dict.keys())))
            caption_weights = {k: self.caption_sample_ratio[k] for k in caption_keys}
            selected_key = random.choices(
                list(caption_weights.keys()),
                weights=list(caption_weights.values())
            )[0]
            long_short_caption = [caption_dict[selected_key]]

        # 随机替换关键词顺序
        random.shuffle(tag_list)
        random.shuffle(long_short_caption)

        # 随机替换关键词与文本描述的顺序
        final_string = [", ".join(tag_list), ", ".join(long_short_caption)]
        final_string = [x for x in final_string if len(x)]
        random.shuffle(final_string)
        if hq_source_tag:
            final_string.insert(0, hq_source_tag)
        final_string = ". ".join(final_string).replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,",
                                                                                                                ",")
        final_string = re.sub(r'\s+', ' ', final_string).strip()
        return final_string

    def strip(self, text):
        text = text.replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
        text = re.sub(r'\s+,', ' ', text).strip()
        return text

    def apply_recaption_template(self, caption_dict, tag_list):
        """
        A `recaption` template means a recaption template like:
            <short caption> <recaption> <long caption> </recaption>

        This template is inspired by the <think></think> template introduced by DeepSeek-R1.
        This template is used to primarily demonstrate the test-time scaling ability of Transfusion models.
        """
        long_caption = caption_dict.get('long caption', '')

        # In the `recaption` template, we treat both the short & medium caption as the short caption
        short_caption_keys = sorted(list(self.valid_short_keys.intersection(caption_dict.keys())))
        if len(short_caption_keys) > 0:
            short_caption_weights = {k: self.caption_sample_ratio[k] for k in short_caption_keys}
            selected_short_key = random.choices(
                list(short_caption_weights.keys()),
                weights=list(short_caption_weights.values())
            )[0]
            short_caption = caption_dict[selected_short_key]
        else:
            short_caption = ''

        # 随机替换关键词顺序
        random.shuffle(tag_list)

        # 随机替换关键词与文本描述的顺序
        if random.random() > 0.5:
            short_caption = ". ".join([short_caption] + tag_list)
            long_caption = ". ".join([long_caption] + tag_list)
        else:
            short_caption = ". ".join(tag_list + [short_caption])
            long_caption = ". ".join(tag_list + [long_caption])

        final_short_caption = self.strip(short_caption)
        final_long_caption = self.strip(long_caption)

        final_string = [final_short_caption, "<recaption>", final_long_caption, "</recaption>"]
        return final_string

    @staticmethod
    def normalize_dict_keys(d):
        if d is None:
            return {}
        new_dict = {}
        for key in list(d.keys()):
            if "_" in key:
                new_dict[key.replace("_", " ")] = d[key]
            else:
                new_dict[key] = d[key]
        return new_dict

    def replace_style(self, caption_dict, style_op):
        style_in_caption = caption_dict['style'].strip()
        try:
            style_op_2nd_level = str(style_op).strip().strip(',').split(',')[-1]  # str() for nan
            if style_in_caption.lower() in ['realistic photography'] and style_op_2nd_level in ['油画']:
                caption_dict['style'] = random.choice(['oil painting', 'oil painting style'])
        except Exception as e:
            self.logger.error(f'encounterd parsing error:{e}')
        return caption_dict

    def caption_aug(self, raw_string, style_op=None, **kwargs):
        caption_dict = self.safe_load_string(raw_string)
        caption_dict = self.normalize_dict_keys(caption_dict)
        if style_op is not None and style_op.strip() != "":
            self.replace_style(caption_dict, style_op)
        if "short caption" in caption_dict and self.use_ip_tag_map:
            caption_dict["short caption"] = self.random_drop_ip(caption_dict["short caption"])
        if "long caption" in caption_dict:
            if self.use_ip_tag_map:
                caption_dict["long caption"] = self.random_drop_ip(caption_dict["long caption"])
            caption_dict["long caption"] = self.random_drop_word(caption_dict["long caption"])
        caption = self.random_compose(caption_dict)
        return caption
