import random
import json
from collections import defaultdict
from functools import partial

import torch
from PIL import Image
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits
from index_kits import arrow_mapper

from hymm.utils.helpers import to_2tuple, default
from hymm.data_kits.caption_strategy import load_caption_processor
from .index_dataset import IndexDataset


class SiglipTextImageArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        training_image_size=256,
        index_file=None,
        raise_text_error=False,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
    ):
        super().__init__(
            index_file,
            multireso,
            index_kwargs.get("batch_size", 1),
            index_kwargs.get("world_size", 1),
            logger,
            debug,
        )
        self.args = args
        self.raise_text_error = raise_text_error
        self.training_image_size = to_2tuple(training_image_size)

        if index_kwargs is None:
            index_kwargs = {}

        self.image_key = index_kwargs.get("image_key", args.get("image_key", "image"))
        self.image_key_shadow = None
        self.image_key_arrow_suffix = None
        if '@' in self.image_key:
            self.image_key, self.image_key_arrow_suffix = self.image_key.split('@')
            self.image_key_shadow = self.image_key

        self.image_caption_rate = index_kwargs.get("image_caption_rate", 0.0)

        self.image_text_col = index_kwargs.get("image_text_col", None)
        if self.image_caption_rate < 1.0 and self.image_text_col is None:
            raise ValueError("Missing `image_text_col` in `index_kwargs`")
        if '@' in self.image_text_col:
            self.image_text_col, self.image_text_arrow_suffix = self.image_text_col.split('@')
            self.image_text_shadow = 'text'
        else:
            self.image_text_arrow_suffix = None
            self.image_text_shadow = None

        self.image_caption_col = index_kwargs.get("image_caption_col", None)
        if self.image_caption_rate > 0.0 and self.image_caption_col is None:
            raise ValueError("Missing `image_caption_col` in `index_kwargs`")
        if self.image_caption_col is not None and '@' in self.image_caption_col:
            self.image_caption_col, self.image_caption_arrow_suffix = self.image_caption_col.split('@')
            self.image_caption_shadow = 'caption'
        else:
            self.image_caption_arrow_suffix = None
            self.image_caption_shadow = None

        self.caption_sample_ratio = index_kwargs.get("caption_sample_ratio", None)
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=default(args.caption_processor, 'caption_process'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=args.get('caption_processor_kwargs'),
            )
        else:
            self.use_structural_caption = False

        # 是否在特定场景下使用 `general_style` 列
        self.use_general_style = args.use_general_style
        # 是否优先使用 source_text (应用 image_caption_ratio 的概率)
        self.try_first_use_source_text = args.try_first_use_source_text

        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        shadow_file_fn = {'clip_score': partial(arrow_mapper, suffix='_clip_score')}  # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        if self.image_caption_rate < 1.0 and self.image_text_arrow_suffix is not None:
            shadow_file_fn.update({self.image_text_shadow: partial(arrow_mapper, suffix=self.image_text_arrow_suffix)})
        if self.image_caption_rate > 0.0 and self.image_caption_arrow_suffix is not None:
            shadow_file_fn.update(
                {self.image_caption_shadow: partial(arrow_mapper, suffix=self.image_caption_arrow_suffix)})
        if self.image_key_shadow:
            shadow_file_fn.update({self.image_key_shadow: partial(arrow_mapper, suffix=self.image_key_arrow_suffix)})

        index_load_kwargs["shadow_file_fn"] = shadow_file_fn

        self.logger.info(f"    (Text-Image) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    (Text-Image) Using {self.index_manager}")
        self.logger.info(f"    Text col: {self.image_text_col} (suffix={self.image_text_arrow_suffix})")
        self.logger.info(f"    Caption col: {self.image_caption_col} (suffix={self.image_caption_arrow_suffix})")
        self.logger.info(f"    Caption rate: {self.image_caption_rate}")
        self.logger.info(f"    Caption sample ratio: {self.caption_sample_ratio}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        pass

    def get_raw_image(self, index, image_key="image", shadow=None):
        try:
            if image_key == "image":
                ret = self.get_image_from_arrow(index, shadow=shadow)
            elif image_key == "url_cos":
                ret = self.get_image_from_url_cos(index, shadow=shadow)
            elif image_key == "cache_image":
                ret = Image.open(self.index_manager.get_attribute(index, 'cache_image', shadow=shadow)).convert("RGB")
            else:
                raise ValueError(f"Unknown image_key: {image_key}")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"({image_key=}, {index=}) {type(e)}: {e}. Fallback to gray image.")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_text(self, ind):
        try:
            if self.try_first_use_source_text:
                source = self.index_manager.get_attribute(ind, 'source')
                try:
                    clip_score = self.index_manager.get_attribute(ind, 'clip_score', shadow='clip_score')
                except:
                    clip_score = None
                if (
                        (source == 'laion2b_en_sd2.1base' or (clip_score is not None and clip_score >= 0.18))
                        and (self.image_caption_rate == 0 or random.random() >= self.image_caption_rate)
                ):
                    text = self.index_manager.get_attribute(ind, self.image_text_col, shadow=self.image_text_shadow)
                else:
                    text = self.index_manager.get_attribute(ind, self.image_caption_col, shadow=self.image_caption_shadow)
                    if self.use_general_style:
                        style_op = self.index_manager.get_attribute(ind, 'general_style')
                    else:
                        style_op = None
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text, style_op=style_op)

            else:
                if self.image_caption_rate > 0.0 and random.random() < self.image_caption_rate:
                    text = self.index_manager.get_attribute(ind, self.image_caption_col,
                                                            shadow=self.image_caption_shadow)
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text)
                else:
                    text = self.index_manager.get_attribute(ind, self.image_text_col, shadow=self.image_text_shadow)

        except Exception as e:
            if self.raise_text_error:
                print(f"#{torch.distributed.get_rank()}: indices {ind}")
                raise e
            self.handle_exception_message(self.get_text, e)
            text = ""
        if text is None:
            text = ""

        if isinstance(text, str):
            text = str(text).strip()
            # Remove meaningless characters
            text = text.replace("\\N", "").strip("，,")

        return text

    def __getitem__(self, index):
        text = self.get_text(index)

        image, image_flag = self.get_raw_image(index, image_key=self.image_key, shadow=self.image_key_shadow)
        if image_flag == "gray":
            text = "A gray image"
        return {
            "text": text,
            "image": image,
        }

    @staticmethod
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        return {
            "image": images,
            "text": texts,
        }