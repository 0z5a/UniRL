import json
import random


# parse json to caption for QWEN
class CaptionAug:
    """
    对caption进行数据增广,包括:
    1. 配置对IP数据替换的比例, 默认配置为全部替换(长短caption)
    2. 配置比例删除文字，去除可能存在错误的文字(长caption)
    3. 对结构化caption进行随机组合。背景, 镜头, 风格, 景别, 氛围, 运镜(视频)+[长文本]/[短文本]

    """

    def __init__(
        self,
        caption_sample_ratio=None,
        ip_remove_rate=1.0,
        text_remove_rate=1.0,
        long_short_caption_mutex=True,
        ip_tag_mapping_file=None,
        logger=None,
    ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
            ip_remove_rate (float, optional): IP数据替换的比例
            text_remove_rate (float, optional): 文字替换的比例
            long_short_caption_mutex : 长短caption互斥
            ip_tag_mapping_file: IP字段映射名称
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
        self.caption_sample_ratio = caption_sample_ratio
        self.caption_keys = {
            "long caption",
            "short caption",
            "background",
            "shot type",
            "style",
            "light",
            "atmosphere",
            "camera movement",
        }
        self.long_short_caption_mutex = long_short_caption_mutex
        self.logger = logger

        # default
        # self.caption_sample_ratio = {key: 0.8 for key in self.caption_keys}
        # self.caption_sample_ratio["long caption"] = 0.5
        # self.caption_sample_ratio["short caption"] = 0.5

        self.logger.info(
            "CaptionAug using caption sample ratio: {}".format(json.dumps(self.caption_sample_ratio, indent=4))
        )

        for key in self.caption_sample_ratio:
            if key not in self.caption_keys:
                raise NotImplementedError("caption sample ratio only support keys: {}".format(self.caption_keys))

    def safe_load_string(self, text):
        """加载结构化caption数据, json or xml格式"""
        try:
            text = json.loads(text)
        except Exception as e:
            raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        if type(text) is str:
            try:
                text = eval(text)
                if type(text) is str:
                    text = text.replace("'", '"')
                    text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got eval str: {}".format(str(e), text))

        if type(text) != dict:
            print("load caption error", text)

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

        def find_ocr_and_drop(text):
            items = text.split(".")
            if len(items) == 1:
                return text

            valid_items = []
            for item in items:
                if '"' in item and any(
                    [_x in item for _x in ["reads", "writes", "text", "words", "Chinese characters"]]
                ):  # 包含引号并且包含某些特殊字符串
                    continue
                valid_items.append(item)
            if len(valid_items) == 0:
                return text
            return ".".join(valid_items)

        return find_ocr_and_drop(long_caption)

    def random_compose(self, caption_dict):
        """随机组合caption"""
        tag_list = []
        long_short_caption = []
        for key in caption_dict:
            if "caption" not in key and random.random() < self.caption_sample_ratio[key]:
                tag_list.append(caption_dict[key])
            if "caption" in key and random.random() < self.caption_sample_ratio[key]:
                long_short_caption.append(caption_dict[key])

        if self.long_short_caption_mutex:
            renorm_probs = [self.caption_sample_ratio["short caption"], self.caption_sample_ratio["long caption"]]
            renorm_probs = [x * 1.0 / sum(renorm_probs) for x in renorm_probs]
            if random.random() < renorm_probs[0]:
                long_short_caption = [caption_dict["short caption"]]
            else:
                long_short_caption = [caption_dict["long caption"]]

        elif len(long_short_caption) == 0:
            long_short_caption = random.choice(
                [
                    [caption_dict["short caption"]],
                    [caption_dict["long caption"]],
                    [caption_dict["short caption"], caption_dict["long caption"]],
                ]
            )
        # 随机替换关键词顺序
        random.shuffle(tag_list)
        random.shuffle(long_short_caption)

        # 随机替换关键词与文本描述的顺序
        final_string = [",".join(tag_list), ",".join(long_short_caption)]
        final_string = [x for x in final_string if len(x)]
        random.shuffle(final_string)
        final_string = ",".join(final_string).replace(".,", ",")
        return final_string

    def caption_aug(self, raw_string, **kwargs):
        caption_dict = self.safe_load_string(raw_string)
        if "short caption" in caption_dict and self.use_ip_tag_map:
            caption_dict["short caption"] = self.random_drop_ip(caption_dict["short caption"])
        if "long caption" in caption_dict:
            if self.use_ip_tag_map:
                caption_dict["long caption"] = self.random_drop_ip(caption_dict["long caption"])
            caption_dict["long caption"] = self.random_drop_word(caption_dict["long caption"])
        caption = self.random_compose(caption_dict)
        return caption
