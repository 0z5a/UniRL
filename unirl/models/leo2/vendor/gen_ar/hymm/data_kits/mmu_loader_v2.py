import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from typing import List, Dict
from urllib.parse import unquote
from functools import partial
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import transforms
from loguru import logger as all_logger

from .caption_strategy import load_caption_processor
from .caption_strategy.manager import MultiCaptionManager
from .caption_strategy.caption_process_v2 import CaptionAug
from .image_dataset import ImageDataset
from .index_dataset import IndexColumn, resample_on_gray
from .instruction_template import image_captioning_instructions_dict
from ..constants import VISION_ENCODER_META_INFO, VAE_META_INFO
from ..models.tokenizers import TokenizerWrapper
from ..models.visual_encoders import load_vision_model_processor
from ..utils.helpers import default, to_2tuple
from ..utils.import_utils import require_version
from ..utils.resolution import ResolutionGroup
from ..models.tokenizers.conversation import get_conversation_template
from ..data_kits.system_prompt import vanilla_system_prompts, unified_system_prompts

require_version("index-kits", "0.5.0", "MultiModalUnderstandingArrowStream")


@dataclass
class MMUData:
    messages: List[Dict]
    image_tensor: torch.Tensor
    kwargs: Dict
    vae_image_list: List
    image_flag: str
    index: int

    def unbind(self):
        return self.messages, self.image_tensor, self.kwargs, self.vae_image_list, self.image_flag, self.index


class MultiModalUnderstandingArrowStream(ImageDataset):
    def __init__(
            self,
            args,
            dataset_tag=None,
            tokenizer_name=None,
            task_kwargs=None,
            index_kwargs=None,
            post_kwargs=None,
            logger=None,
            tokenizer=None,
            pad_color=(127, 127, 127),
            dummy_number=0,
            conv_format="hunyuan-gemini-alpha",
            template="pretrain",
    ):
        self.dataset_tag = dataset_tag or "mmu"
        # strip leading `mmu_` or `mmu_interleave_` for task_kwargs and index_kwargs
        self.task_kwargs = self.strip_leading_tag(default(task_kwargs, {}))
        self.index_kwargs = self.strip_leading_tag(default(index_kwargs, {}))

        _base_size = self.task_kwargs.get('vae_base_size', 256)
        super().__init__(index_file=self.index_kwargs['index_file'], base_size=_base_size, logger=logger, args=args)
        self.logger.info(f"    (MMU-{dataset_tag}) {self.task_kwargs=}")
        self.logger.info(f"    (MMU-{dataset_tag}) {self.index_kwargs=}")
        self.dummy_number = dummy_number
        self.max_text_token_length = self.task_kwargs.get('text_token_length')
        self.max_token_length = self.task_kwargs['token_length'] + 1 - self.dummy_number
        self.drop_last = self.task_kwargs.get("drop_last", False)
        self.medium_caption_as = self.task_kwargs.get("medium_caption_as", "long")
        assert self.medium_caption_as in ["long", "short"], \
            f"medium_caption_as should be either 'long' or 'short', but got {self.medium_caption_as}"
        self.logger.info(f"    (MMU-{dataset_tag}) {self.dummy_number=}, {self.max_token_length=}")

        if args.vision_model_type == "siglip2-so400m-patch16-naflex":
            self.vision_encoder_processor = load_vision_model_processor(args.vision_model_type)
            self.image_token_length = args.vision_encoder_max_num_patches
            self.vision_encoder_processor = partial(self.vision_encoder_processor, max_num_patches=self.image_token_length)
            self.image_size = to_2tuple(self.task_kwargs['image_size'])
        else:
            self.vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
            self.downsample_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
            if self.task_kwargs.get('image_size') is None:
                self.image_size = to_2tuple(self.vision_encoder_meta_info["image_size"])
            else:
                self.image_size = to_2tuple(self.task_kwargs['image_size'])
            self.image_token_length = math.prod(self.image_size) // math.prod(self.downsample_factor)
            # TODO: should use a base size instead of image_size[0]
            self.reso_group = ResolutionGroup(self.image_size[0])
            h, w = self.image_size
            self.base_size, self.ratio_idx = self.reso_group.get_base_size_and_ratio_index(w, h)
        self.logger.info(f"    (MMU-{dataset_tag}) Using {self.image_token_length=}, {self.drop_last=}, {self.max_token_length=}")

        # Prepare index manager
        self.setup_index_manager(f"MMU-{dataset_tag}")

        # Check required columns
        if self.cos_base is not None and any("{bucket}" in target for target in self.cos_base_targets):
            assert hasattr(self, "extra_url_cos_col"), \
                "When using cos_base with `{bucket}` template, `extra_url_cos_col` should be provided in index_kwargs."

        self.short_caption_rate = self.task_kwargs.get("short_caption_rate")
        self.long_caption_rate = self.task_kwargs.get("long_caption_rate")

        # Image transform
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.use_front_boi_token = self.args.use_front_boi_token
        self.add_image_shape_token = self.args.get('add_image_shape_token', False)
        self.pad_color = pad_color
        self.use_und_token = self.args.get('use_und_token', False)

        if 'switch_image_text' in self.task_kwargs:
            self.logger.warning("switch_image_text is deprecated. Please use use_front_src_image instead.")
        if 'switch_image_text' in self.task_kwargs and 'use_front_src_image' in self.args:
            assert self.task_kwargs['switch_image_text'] ^ self.args.use_front_src_image, "switch_image_text and use_front_src_image should be mutually exclusive"
            self.use_front_src_image = self.args.use_front_src_image
        elif 'switch_image_text' in self.task_kwargs:
            self.use_front_src_image = not self.task_kwargs['switch_image_text']
        else:
            self.use_front_src_image = self.args.get('use_front_src_image', False)

        self.use_joint_image_feature = self.args.get('use_joint_image_feature', False)
        if self.use_joint_image_feature:
            self.add_timestep_token = self.args.get('add_timestep_token', False)
            self.vae_reso_group = ResolutionGroup(_base_size)
            self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
            self.vae_downsample_factor = self.vae_meta_info["downsample_factor"]
            self.vae_patch_size = self.args.patch_size
            self.vae_h_factor = self.vae_downsample_factor[0] * self.vae_patch_size
            self.vae_w_factor = self.vae_downsample_factor[1] * self.vae_patch_size

        if not self.use_joint_image_feature and self.add_image_shape_token and self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            raise NotImplementedError("siglip2-so400m-patch16-naflex does not match our image shape token.")

        # Text tokenizer
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Sequence pack related and affected
        self.sequence_pack = self.task_kwargs.get('sequence_pack', False)
        if self.sequence_pack:
            # if sequence pack is enabled, we will use the sequence-wise dummy_number to pad the packed sequence,
            # instead of dummy_number to pad the sample. `block_size` will be used as the maximum sequence length
            # for all the datasets.
            self.max_sequence_length = self.args.block_size + 1 - self.dummy_number

        # Template
        self.template = template
        assert template in ["pretrain", "instruct"], f"Unsupported template: {template}"
        if template == "instruct":
            assert conv_format, f"conv_format should be provided for instruct template."
            self.conv_format = conv_format
            self.default_conv = get_conversation_template(self.conv_format)
            self.roles = self.default_conv.roles
            # {"User": 3, "Assistant": 3, "System": 0}
            self.role_prefix_offset = {
                role: len(self.tokenizer.encode_text(self.default_conv.get_role_prefix(role)))
                for role in self.roles
            }
            self.role_prefix_offset["System"] = 0

        self.system_prompt_candidates_type = self.task_kwargs.get("system_prompt_candidates_type", "off")
        assert self.system_prompt_candidates_type in {"fixed_set", "off"} or self.system_prompt_candidates_type.startswith("mix_up"), \
            f"Unsupported system_prompt_candidates_type: {self.system_prompt_candidates_type}"
        self.use_vanilla_system_prompt = self.task_kwargs.get("use_vanilla_system_prompt", False)

        #   system prompt length is bounded by `system_prompt_token_length`
        self.system_prompt_token_length = self.task_kwargs.get('system_prompt_token_length', 0)

        # Image caption processor
        #    for 老格式. 新的 CaptionManager 内部会自动管理 caption_processer, 不再需要这里的.
        if self.task_kwargs.get('caption_processor') is not None:
            self.caption_processor = load_caption_processor(
                name=self.task_kwargs.get('caption_processor'),
                caption_sample_ratio=json.loads(self.task_kwargs['caption_sample_ratio']),
                logger=self.logger,
                kwargs=self.task_kwargs.get('caption_processor_kwargs'),
            )
        self.ignore_user_instruct = self.task_kwargs.get('ignore_user_instruct', False)
        # 控制多个 caption 版本
        image_caption_col_probs = self.task_kwargs.get('image_caption_col_probs', None)
        if isinstance(image_caption_col_probs, list):
            # CaptionManager 格式
            self.multi_caption_manager = MultiCaptionManager(
                resource=image_caption_col_probs,
                dataset=self,
                # backward compatibility
                caption_processor=self.task_kwargs.get('caption_processor'),
                caption_sample_ratio=self.task_kwargs.get('caption_sample_ratio'),
            )
        else:
            self.multi_caption_manager = None
            # 兼容老格式
            self.multi_caption_cols, self.multi_caption_probs = self.parse_caption_col_probs(
                self.task_kwargs.get('image_caption_col_probs', None)
            )
            self.caption_sample_ratio = self.task_kwargs.get('caption_sample_ratio', {})
            if isinstance(self.caption_sample_ratio, str):
                self.caption_sample_ratio = json.loads(self.caption_sample_ratio)

        # OCR processor
        self.quad_pattern = re.compile(r"<quad>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</quad>")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        # After init, we set the logger to all_logger to print warnings and errors of all ranks
        self.logger = all_logger

    def parse_columns_and_register_shadow(self, index_kwargs):
        self.image_col = IndexColumn(index_kwargs.get("image_key"), self, self.logger)
        self.message_col = IndexColumn(index_kwargs.get("message_col"), self, self.logger)
        self.register_attributes(index_kwargs)

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_system_prompt(self, system_prompt_candidates):
        if self.system_prompt_candidates_type == "off":
            return None
        elif self.system_prompt_candidates_type == "fixed_set":
            return random.choice(system_prompt_candidates).strip()
        elif self.system_prompt_candidates_type.startswith("mix_up"):
            mix_up_ratio = float(self.system_prompt_candidates_type.split("@")[-1])
            if random.random() < mix_up_ratio:
                return random.choice(system_prompt_candidates).strip()
            else:
                return None
        else:
            raise ValueError(f"Unsupported system_prompt_candidates_type: {self.system_prompt_candidates_type}")

    def parse_caption_col_probs(self, src):
        if src is None:
            return None, None
        if isinstance(src, str):
            prob_dict = json.loads(src)
            # Make sure the key of probs are all registered columns
            for key in prob_dict.keys():
                if not hasattr(self, key) or not isinstance(getattr(self, key), IndexColumn):
                    raise ValueError(f"Caption column {key} is not registered as IndexColumn.")
            cols = [getattr(self, key) for key in prob_dict.keys()]
            probs = list(prob_dict.values())
        else:
            raise TypeError(f"`image_caption_col_probs` must be a JSON string, got {type(src)}.")
        return cols, probs

    def get_vae_image(self, image, random_crop=True):
        origin_size = image.size  # (w_ori, h_ori)

        # for mmu, we use online reso_group to get the target size
        target_size = self.vae_reso_group.get_target_size(*origin_size)

        vae_image_tensor, vae_kwargs, (h, w), (tk_height, tk_width) = self.vae_process_image(
            image, target_size, random_crop=random_crop, return_resized=True,
        )

        return vae_image_tensor, vae_kwargs, (h, w), (tk_height, tk_width)

    @staticmethod
    def _get_resized_shape(image, target_size):
        # 返回 resize 后, crop 前的尺寸.
        tw, th = target_size
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            resized_height = th
            resized_width = int(round(th / h * w))
        else:
            resized_width = tw
            resized_height = int(round(tw / w * h))
        return {"resized_width": resized_width, "resized_height": resized_height}

    def vae_process_image(self, image, target_size, random_crop=True, return_resized=False):
        origin_size = image.size  # (w_ori, h_ori)
        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random" if random_crop else "center", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        if return_resized:
            kwargs.update(self._get_resized_shape(image, target_size))

        h, w = image_tensor.shape[1], image_tensor.shape[2]
        assert (h % self.vae_h_factor == 0 and w % self.vae_w_factor == 0), \
            (f"Image size should be divisible by downsample_factor * patch_size, "
             f"but got ({h} x {w}) with downsample_factor={self.vae_downsample_factor} and patch_size={self.vae_patch_size}")
        tk_height = h // self.vae_h_factor
        tk_width = w // self.vae_w_factor

        return image_tensor, kwargs, (h, w), (tk_height, tk_width)

    def preprocess_image(self, image):
        if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            inputs = self.vision_encoder_processor(image)
            image_tensor = inputs["pixel_values"].squeeze(0)   # seq_len x dim 
            spatial_shapes = inputs["spatial_shapes"]  # 1 x 2, don't squeeze to align with outputs of get_interleave_data
            pixel_attention_mask = inputs["pixel_attention_mask"]  # 1 x seq_len, don't squeeze to align with outputs of get_interleave_data

            kwargs = {
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            }
            return image_tensor, kwargs

        else:
            origin_size = image.size  # (w_ori, h_ori)

            image, (pad_left, pad_top) = self.index_manager.resize_and_pad(
                image, self.image_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
            )

            image_tensor = self.pil_image_to_tensor(image)

            kwargs = {
                "origin_size": origin_size,
                "target_size": self.image_size,
                "pad_coords_xy": (pad_left, pad_top),
            }
            return image_tensor, kwargs

    def get_any_image(self, src, real_index=None, apply_exif=True):
        image, image_flag = self.get_raw_image(src, real_index=real_index, apply_exif=apply_exif, **self.image_col)
        image_tensor, kwargs = self.preprocess_image(image)

        if self.use_joint_image_feature:
            vae_image_tensor, vae_kwargs, (h, w), (tk_height, tk_width) = self.get_vae_image(image)
            return image_tensor, image_flag, kwargs, vae_image_tensor, vae_kwargs, (h, w), (tk_height, tk_width)

        return image_tensor, image_flag, kwargs

    @staticmethod
    def vae_coord_tokens(x0, y0, x1, y1, vae_kwargs, return_token=False):
        # (x0, y0, x1, y1) 应该是介于 [0, 1000] 的归一化整数坐标: 即 [0, 1] 的归一化坐标 * 1000 并取整.
        resized_width = vae_kwargs["resized_width"]
        resized_height = vae_kwargs["resized_height"]
        crop_left = vae_kwargs["crop_coords_xy"][0]
        crop_top = vae_kwargs["crop_coords_xy"][1]
        target_width = vae_kwargs["target_size"][0]
        target_height = vae_kwargs["target_size"][1]
        # 计算 resize + crop 后的归一化整数坐标
        t_x0 = max(int(((x0 / 1000 * resized_width - crop_left) / target_width) * 1000), 0)
        t_y0 = max(int(((y0 / 1000 * resized_height - crop_top) / target_height) * 1000), 0)
        t_x1 = min(int(((x1 / 1000 * resized_width - crop_left) / target_width) * 1000), 999)
        t_y1 = min(int(((y1 / 1000 * resized_height - crop_top) / target_height) * 1000), 999)
        if return_token:
            return f"<pos_x_{t_x0}><pos_y_{t_y0}><pos_x_{t_x1}><pos_y_{t_y1}>"
        return t_x0, t_y0, t_x1, t_y1

    def format_ocr_data(self, ocr_str, vae_kwargs):
        # We always fall back to the original ocr_str if any error occurs.
        formatted_str = ocr_str
        try:
            matched = []
            for match in self.quad_pattern.finditer(ocr_str):
                matched.append({
                    "start": match.start(),
                    "end": match.end(),
                    "coords": tuple(map(int, match.groups())),   # [x0, y0, x1, y1]
                })
            if len(matched) > 0:
                cat_str = ""
                last_pos = 0
                for m in matched:
                    cat_str += ocr_str[last_pos:m["start"]]
                    coord_tokens = self.vae_coord_tokens(*m["coords"], vae_kwargs, return_token=True)
                    cat_str += f"<quad>{coord_tokens}</quad>"
                    last_pos = m["end"]
                cat_str += ocr_str[last_pos:]
                formatted_str = cat_str
        except Exception as e:
            self.logger.error(f"{e.__class__.__name__}: {e}")
        return formatted_str

    def format_message_list(self, message_list, return_type="list", add_system=True):
        conversation = get_conversation_template(self.conv_format)
        conversation.system_message = ""  # No system message
        for msg in message_list:
            # User message
            conversation.add_message(conversation.roles[0], msg["User"].strip())
            # Assistant message
            assistant_msg = '<answer>' + msg["Assistant"].strip() + '</answer>'
            conversation.add_message(conversation.roles[1], assistant_msg)
        text = conversation.get_prompt(return_type=return_type, add_system=add_system)
        return text

    def get_mmu_image_caption(self, index):
        image, image_flag = self.get_raw_image(index, apply_exif=True, **self.image_col)
        image_tensor, kwargs = self.preprocess_image(image)
        if image_flag == "gray":
            text = "A gray image."
        else:
            try:
                text = self.index_manager.get_attribute(index, "caption_v2")
                text = self.caption_processor.caption_aug(text)
            except Exception as e:
                self.handle_exception_message(self.get_mmu_image_caption, e)
                text = ""
        if text is None:
            text = "This is an image."
            self.logger.warning(f"(mmu, {index=}) text is None. Fallback to a fixed string '{text}'.")

        # message_list = [{"user": "Describe the image.", "assistant": text}]
        return image_tensor, text, kwargs

    def get_caption_old(self, index):
        caption_candidates = []
        caption_probs = []
        lang = "en"
        for _ in range(5):
            try:
                if self.multi_caption_cols is not None:
                    sel_col = random.choices(
                        self.multi_caption_cols, weights=self.multi_caption_probs, k=1
                    )[0]
                    text = self.index_manager.get_attribute(index, **sel_col)
                    lang = sel_col.key.split('_')[-1]  # e.g., caption_v3_zh -> zh, caption_v3_en -> en
                else:
                    text = self.index_manager.get_attribute(index, "caption_v2")
                _caption_dict = CaptionAug.safe_load_string(text)
                for key in self.caption_sample_ratio:
                    if key in _caption_dict and _caption_dict[key] != 'None' and _caption_dict[key] != '无':
                        caption_candidates.append((key, _caption_dict[key]))
                        caption_probs.append(self.caption_sample_ratio[key])
                if len(caption_candidates) == 0:
                    raise ValueError(f"All the long/medium/short captions are empty. {index=}")
                break
            except Exception as e:
                self.handle_exception_message(self.get_mmu_image_caption_v2, e)
                new_index = self.index_manager.random_dindex(index)
                self.logger.error(f"Error with index={index}, trying new index={new_index}")
                index = new_index
        if len(caption_candidates) == 0:
            self.logger.error(f"All the long/medium/short captions are empty. {index=}")
        # Process caption
        key, caption = random.choices(caption_candidates, weights=caption_probs)[0]
        short_keys = ["short_caption"] + (["medium_caption"] if self.medium_caption_as == "short" else [])
        if key in short_keys:
            user = random.choice(image_captioning_instructions_dict["short"][lang])
        else:
            user = random.choice(image_captioning_instructions_dict["long"][lang])
        if lang == "en":
            user = user.rstrip() + " "
        return user, caption

    def get_caption_from_manager(self, index):
        cap_out = self.multi_caption_manager.get_caption(index, return_dict=True)
        lang = cap_out.lang

        short_keys = ["short_caption"] + (["medium_caption"] if self.medium_caption_as == "short" else [])
        if cap_out.key in short_keys:
            user = random.choice(image_captioning_instructions_dict["short"][lang])
        else:
            user = random.choice(image_captioning_instructions_dict["long"][lang])

        return user, cap_out.caption

    def get_mmu_image_caption_v2(self, index):
        # v2 是为了区别于老的 get_mmu_image_caption, 并不是特指 caption_v2.
        if self.multi_caption_manager is not None:
            user, caption = self.get_caption_from_manager(index)
        else:
            user, caption = self.get_caption_old(index)
        if caption is None:
            caption = "This is an image."
            valid_caption = False
        else:
            valid_caption = True

        messages = [
            dict(type="cond_image"),
            dict(type="cond_text", text=user, kwargs=dict(ignore=self.ignore_user_instruct)),
            dict(type="gen_text", text=caption),
        ]

        # Process image
        image_tensor, image_flag, kwargs, *extra_list = self.get_any_image(index, real_index=index)
        if not valid_caption:
            image_flag = "gray"     # set `gray` flag to activate resample_on_gray

        return messages, image_tensor, kwargs, image_flag, *extra_list

    def get_interleave_data(self, index):
        # assert self.template == "pretrain", "Only support pretrain template for interleave data"

        messages = self.index_manager.get_attribute(index, **self.message_col)
        images = []
        vae_image_list = []
        kwargs_list = []
        image_flags = []
        for msg in messages:
            if msg['type'] in ['cond_image', 'image']:  # leave 'image' here for backward compatibility
                image_tensor, image_flag, kwargs, *extra_list = self.get_any_image(msg, real_index=index)
                image_flags.append(image_flag)
                images.append(image_tensor)
                vae_image_list.append(extra_list)
                kwargs_list.append(kwargs)
            elif msg['type'] in ['cond_text', 'gen_text']:
                # 兼容 "text" in msg 的老格式, 后续再统一.
                # Process ocr data, always use the previous image as reference
                for key in ["text_en", "text_zh"]:
                    if key not in msg:
                        continue
                    if (text := msg[key]) and len(vae_image_list) > 0:
                        if "<ref>" in text and "</ref>" in text and "<quad>" in text and "</quad>" in text:
                            vae_kwargs = vae_image_list[-1][1]
                            msg[key] = self.format_ocr_data(text, vae_kwargs)

        images = torch.stack(images)

        image_flag = "normal" if all(flag == "normal" for flag in image_flags) else "gray"

        if self.use_joint_image_feature:
            if len(images) == 1:
                return messages, images, kwargs_list[0], image_flag, *vae_image_list[0]
            else:
                raise NotImplementedError("Joint image feature is not supported for interleave data with multiple images.")

        return messages, images, {}, image_flag

    def build_pretrain_template(self, messages, **kwargs):
        # Build mmu QA template.
        #
        # Example (Q: question text token, A: answer text token, P: picture token):
        #     Q  Q  Q  P  P  A  A  A  Q  Q  Q  P  P  A  A  A
        image_kwargs = dict(
            add_iw_ih_token=self.add_iw_ih_token, use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token
        )
        if self.use_joint_image_feature:
            image_kwargs['add_timestep_token'] = self.add_timestep_token
            assert "vae_image_list" in kwargs, "vae_image_list is required for joint image feature"
            vae_image_list = kwargs["vae_image_list"]
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(kwargs["vae_image_list"][2][1], kwargs["vae_image_list"][2][0])
            image_kwargs['base_size'] = base_size
            image_kwargs['ratio_idx'] = ratio_idx
            vae_token_height, vae_token_width = vae_image_list[3][0], vae_image_list[3][1]
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                assert "spatial_shapes" in kwargs, "spatial_shapes is required for siglip2-so400m-patch16-naflex"
                spatial_shapes = kwargs["spatial_shapes"]
                token_height = spatial_shapes[0][0].item()
                token_width = spatial_shapes[0][1].item()
            else:
                token_height = self.image_size[0] // self.downsample_factor[0]
                token_width = self.image_size[1] // self.downsample_factor[1]
            image_kwargs.update(dict(token_height=[vae_token_height, token_height],
                                     token_width=[vae_token_width, token_width]))
            vae_image_token_length = vae_image_list[3][0] * vae_image_list[3][1]
        else:
            image_kwargs['base_size'] = self.base_size
            image_kwargs['ratio_idx'] = self.ratio_idx
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                assert "spatial_shapes" in kwargs, "spatial_shapes is required for siglip2-so400m-patch16-naflex"
                spatial_shapes = kwargs["spatial_shapes"]
                token_height = spatial_shapes[0][0]
                token_width = spatial_shapes[0][1]
            else:
                token_height = self.image_size[0] // self.downsample_factor[0]
                token_width = self.image_size[1] // self.downsample_factor[1]
            image_kwargs.update(dict(token_height=token_height,
                                     token_width=token_width))

        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0) + (1 if self.use_joint_image_feature and self.add_timestep_token else 0)

        # 目前 mmu 数据还存在两种格式: 文本字段为 text 和 文本字段为 text_zh/text_en. 现在临时兼容两种. 后续统一成后者.
        lang = random.choice(["en", "zh"])

        templates = []
        sections = []
        for msg in messages:
            if msg['type'] == 'cond_text' or msg['type'] == 'gen_text':
                templates.append('text')
                kwargs = msg.get('kwargs', {})
                kwargs["ignore"] = msg['type'] == 'cond_text'
                if "text" in msg:
                    sections.append(dict(
                        type="text",
                        text=msg['text'].replace('<image>', '').replace('<img>', ''),
                        **kwargs,
                    ))
                else:
                    text = msg[f"text_{lang}"] if (f"text_{lang}" in msg and msg[f"text_{lang}"]) else (
                        msg['text_zh'] if lang == "en" else msg['text_en']
                    )
                    sections.append(dict(
                        type="text",
                        text=text.replace('<image>', '').replace('<img>', ''),
                        **kwargs,
                    ))
            elif msg['type'] == 'cond_image':
                if self.use_joint_image_feature:
                    templates.append('joint_image')
                    sections.append(dict(type="joint_image", token_length=[vae_image_token_length, self.image_token_length], **image_kwargs))
                else:
                    templates.append('und_image')
                    sections.append(dict(type="und_image", token_length=self.image_token_length, **image_kwargs))

        # Switch und_image with following one text
        if not self.use_front_src_image:
            i = 0
            while i < len(sections) - 1:
                if sections[i]['type'] in ['und_image', 'joint_image'] and sections[i + 1]['type'] == 'text':
                    # switch with next text and forward 2 sections to avoid repeated switch
                    sections[i], sections[i + 1] = sections[i + 1], sections[i]
                    templates[i], templates[i + 1] = templates[i + 1], templates[i]
                    i += 2
                else:
                    i += 1

        # Add empty text to enable <eos> token in the text mask
        templates.append('text')
        sections.append(dict(type='text', text='', end_offset=1))

        # We avoid the text before the first image is too long to squeeze the image tokens out of
        # the max length. If there are multiple prefix text sections, we evenly split the max length.
        num_prefix_text_sections = 0
        for section in sections:
            if section['type'] == 'text':
                num_prefix_text_sections += 1
            else:
                break
        if num_prefix_text_sections > 0:
            extra_length = (
                    2 +     # <bos> and <eos> tokens
                    2 +     # <und_boi> and <und_eoi> or <boi> and <eoi> tokens
                    (2 if self.add_iw_ih_token else 0) +
                    (2 if self.add_image_shape_token else 0) +
                    (1 if self.use_joint_image_feature and self.add_timestep_token else 0) +    # <timestep>
                    (1 if self.use_joint_image_feature else 0)    # <joint_img_sep>
            )
            prefix_text_max_length = self.max_token_length - self.image_token_length - (vae_image_token_length if self.use_joint_image_feature else 0) - extra_length
            per_text_max_length = prefix_text_max_length // num_prefix_text_sections
            for i in range(len(sections)):
                if sections[i]['type'] == 'text':
                    sections[i]['max_length'] = per_text_max_length
                else:
                    break

        if not self.use_front_src_image:
            # Insert an empty text section before image to include num_image_prefix tokens to the text mask
            new_templates, new_sections = [], []
            for t, s in zip(templates, sections):
                if 'image' in t:
                    new_templates.append('text')
                    new_sections.append(dict(type='text', text='', end_offset=num_image_prefix))
                new_templates.append(t)
                new_sections.append(s)

            # Pack the sequence
            return '-'.join(new_templates), new_sections
        else:
            return '-'.join(templates), sections
    
    def get_text_from_message(self, msg, lang):
        if "text" in msg:
            return msg['text'].replace('<image>', '').replace('<img>', '').strip()
        else:
            text = msg[f"text_{lang}"] if (f"text_{lang}" in msg and msg[f"text_{lang}"]) else (
                msg['text_zh'] if lang == "en" else msg['text_en']
            )
            return text.replace('<image>', '').replace('<img>', '').strip()
    
    def convert_new_messages_to_old_messages(self, messages):
        record_msgs = []
        i = 0
        lang = random.choice(["en", "zh"])
        while i < len(messages):
            single_msg = {}
            if messages[i]['type'] == 'cond_text':
                single_msg['User'] = self.get_text_from_message(messages[i], lang)
                while i + 1 < len(messages) and messages[i + 1]['type'] == 'cond_text':
                    single_msg['User'] += " " + self.get_text_from_message(messages[i + 1], lang)
                    i += 1
                i += 1
                # fix bug: 如果cond_text后面没有gen_text，则Assistant为空, 加上i<len(messages)判断,否则会有数组越界风险
                if i < len(messages) and messages[i]['type'] == 'gen_text':
                    single_msg['Assistant'] = self.get_text_from_message(messages[i], lang)
                else:
                    single_msg['Assistant'] = ""
                while i + 1 < len(messages) and messages[i + 1]['type'] == 'gen_text':
                    single_msg['Assistant'] += " " + self.get_text_from_message(messages[i + 1], lang)
                    i += 1
            elif i < len(messages) and messages[i]['type'] == 'gen_text':
                single_msg['User'] = ""
                single_msg['Assistant'] = self.get_text_from_message(messages[i], lang) 
                while i + 1 < len(messages) and messages[i + 1]['type'] == 'gen_text':
                    single_msg['Assistant'] += " " + self.get_text_from_message(messages[i + 1], lang)
                    i += 1  
            i += 1
            if len(single_msg) > 0:
                record_msgs.append(single_msg)
        
        return record_msgs

    def build_instruct_template(self, messages, **kwargs):
        image_kwargs = dict(
            add_iw_ih_token=self.add_iw_ih_token, use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token
        )
        if self.use_joint_image_feature:
            image_kwargs['add_timestep_token'] = self.add_timestep_token
            assert "vae_image_list" in kwargs, "vae_image_list is required for joint image feature"
            vae_image_list = kwargs["vae_image_list"]
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(vae_image_list[2][1], vae_image_list[2][0])
            image_kwargs['base_size'] = base_size
            image_kwargs['ratio_idx'] = ratio_idx
            vae_token_height, vae_token_width = vae_image_list[3][0], vae_image_list[3][1]
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                assert "spatial_shapes" in kwargs, "spatial_shapes is required for siglip2-so400m-patch16-naflex"
                spatial_shapes = kwargs["spatial_shapes"]
                token_height = spatial_shapes[0][0]
                token_width = spatial_shapes[0][1]
            else:
                token_height = self.image_size[0] // self.downsample_factor[0]
                token_width = self.image_size[1] // self.downsample_factor[1]
            image_kwargs.update(dict(token_height=[vae_token_height, token_height],
                                     token_width=[vae_token_width, token_width]))
            vae_image_token_length = vae_image_list[3][0] * vae_image_list[3][1]
        else:
            image_kwargs['base_size'] = self.base_size
            image_kwargs['ratio_idx'] = self.ratio_idx
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                assert "spatial_shapes" in kwargs, "spatial_shapes is required for siglip2-so400m-patch16-naflex"
                spatial_shapes = kwargs["spatial_shapes"]
                token_height = spatial_shapes[0][0]
                token_width = spatial_shapes[0][1]
            else:
                token_height = self.image_size[0] // self.downsample_factor[0]
                token_width = self.image_size[1] // self.downsample_factor[1]
            image_kwargs.update(dict(token_height=token_height,
                                     token_width=token_width))

        assert isinstance(messages, list), "Only support list message for instruct template"
        extra_num_tokens = (
                1 +  # <bos>
                2 +  # <und_boi> + <und_eoi> or <boi> and <eoi> tokens
                (2 if self.add_iw_ih_token else 0) +
                (2 if self.add_image_shape_token else 0) +
                (1 if self.use_joint_image_feature else 0) +  # <joint_img_sep>
                9 +     # "User: " + "\n\n" + "Assistant: <answer>" + </answer>
                128     # assistant answers, at least
                # <eos> is not included because it will be stripped in the shift of next-token-prediction
        )

        # 将新的messages格式转换为旧的messages格式
        messages = self.convert_new_messages_to_old_messages(messages)
        
        
        if self.use_vanilla_system_prompt:
            system_prompt = self.get_system_prompt(vanilla_system_prompts["en"]) # for mmu data, use vanilla system prompt
        else:
            system_prompt = self.get_system_prompt(unified_system_prompts["en_unified"]) # for mmu editing recaption, use unified system prompt
            
        if system_prompt == "":
            sections = []
        else:
            sections = [
                dict(type="text", text=system_prompt, ignore=True, max_length=self.system_prompt_token_length - 1),
                dict(type="text", text=self.default_conv.sep, ignore=True), # "\n\n" 1 token
            ]

        answer_prefix_token = '<answer>' if '<answer>' not in messages[0]['Assistant'] else ''
        answer_suffix_token = '</answer>' if '<answer>' not in messages[0]['Assistant'] else ''
        if not self.use_front_src_image:
            # We want the image prefix special tokens <und_boi> to be learned.
            # It is implemented by adding text mask end offsets to the previous text sections.
            if self.use_joint_image_feature:
                raise ValueError(f"Joint image feature is not supported for instruct template with use_front_src_image=False. ")
            # User: xxx\n\nAssistant: <answer><und_boi>[image]<und_eoi>xxx</answer><eos>
            num_image_prefix = 1
            
            sections.extend([
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True),
                dict(type="text", text=messages[0]['User'], max_length=self.max_text_token_length - extra_num_tokens, ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True),
                dict(type="text", text='', end_offset=num_image_prefix, ignore=False), # <und_boi> token
                dict(type="und_image", token_length=self.image_token_length, **image_kwargs),
                dict(type="text", text=f"{self.roles[1]}: " + answer_prefix_token, ignore=True),
                dict(type="text", text=messages[0]['Assistant'] + answer_suffix_token, ignore=False),  # 1+x token
            ])

        else:
            # User: <und_boi>[image]<und_eoi> xxx\n\nAssistant: <answer>xxx</answer><eos>
            sections.extend([
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True),
                dict(type="und_image", token_length=self.image_token_length, **image_kwargs) if not self.use_joint_image_feature else dict(type="joint_image", token_length=[vae_image_token_length, self.image_token_length], **image_kwargs),
                dict(type="text", text=messages[0]['User'], max_length=self.max_text_token_length - extra_num_tokens, ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True),
                dict(type="text", text=f"{self.roles[1]}: " + answer_prefix_token, ignore=True),
                dict(type="text", text=messages[0]['Assistant'] + answer_suffix_token, ignore=False),  # 1+x token
            ])

        for message in messages[1:]:
            answer_prefix_token = '<answer>' if '<answer>' not in message['Assistant'] else ''
            answer_suffix_token = '</answer>' if '<answer>' not in message['Assistant'] else ''
            sections.extend([
                dict(type="text", text=self.tokenizer.tokenizer.eos_token, ignore=False),
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True),
                dict(type="text", text=message['User'], ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True),
                dict(type="text", text=f"{self.roles[1]}: " + answer_prefix_token, ignore=True),
                dict(type="text", text=message['Assistant'] + answer_suffix_token, ignore=False),  # 1+x token
            ])
        # If the last text does not end with <eos> or <boi>, then include possible <eos> token
        if not sections[-1]['text'].endswith(self.tokenizer.tokenizer.eos_token) and not sections[-1]['text'].endswith('<boi>'):
            sections[-1]['end_offset'] = 1
        return None, sections

    def get_rope_image_info(self, sections, output):
        if self.args.rope_type == "2d":
            image_slices = output.all_image_slices
            image_shapes = []
            for section in sections:
                if 'image' in section['type']:
                    if isinstance(section['token_height'], list):
                        assert len(section['token_height']) == len(section['token_height']), \
                            f"token_height and token_width should have the same length, but got {len(section['token_height'])} and {len(section['token_width'])}"
                        image_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    else:
                        image_shapes.append((section['token_height'], section['token_width']))
            assert len(image_slices) == len(image_shapes), (
                f"Size miss matching: Image slices({len(image_slices)}) != image shapes({len(image_shapes)})"
            )
            return list(zip(image_slices, image_shapes))
        return None

    @resample_on_gray
    def get_mmu_data(self, index):
        columns = self.index_manager.get_columns(index)
        if self.message_col.key in columns:
            messages, image_tensor, kwargs, image_flag, *vae_image_list = self.get_interleave_data(index)
        else:
            # T2I dataset used as image captioning
            messages, image_tensor, kwargs, image_flag, *vae_image_list = self.get_mmu_image_caption_v2(index)
            image_tensor = image_tensor.unsqueeze(0)    # Align with outputs of get_interleave_data

        # At least [cond_image, cond_text, gen_text] are required.
        if len(messages) < 3:
            image_flag = "gray"

        return MMUData(
            messages=messages,
            image_tensor=image_tensor,
            kwargs=kwargs,
            vae_image_list=vae_image_list,
            image_flag=image_flag,
            index=index,
        )

    def __getitem__(self, index):
        data = self.get_mmu_data(index)
        messages, image_tensor, kwargs, vae_image_list, image_flag, index = data.unbind()

        if self.template == "pretrain":
            template, sections = self.build_pretrain_template(messages, vae_image_list=vae_image_list, **kwargs)
        else:
            template, sections = self.build_instruct_template(messages, vae_image_list=vae_image_list, **kwargs)

        max_token_length = self.max_sequence_length if self.sequence_pack else self.max_token_length
        output = self.tokenizer.encode_general(
            template=template,
            sections=sections,
            max_token_length=max_token_length,
            add_pad=False if self.sequence_pack else 'auto',
            drop_last=self.drop_last,
        )
        target_token = output.tokens.clone()
        target_token[output.text_mask == 0.0] = -100
        if output.und_image_mask is None:
            self.logger.error(f"Missing und image: {index=}, {self.index_manager.get_data(index, [], return_meta=True)}")
        
        src_image = None
        src_image_mask = None
        if self.use_joint_image_feature:
            src_image = vae_image_list[0]
            src_image_mask = output.src_image_mask

        # When drop_last enabled, some images may be dropped. We count the actual number
        # of und images from the slices.
        n_und_images = len(output.und_image_slices)
        if image_tensor.shape[0] > n_und_images:
            image_tensor = image_tensor[:n_und_images]

        # Prepare attention mask
        if self.task_kwargs.get('attn_type', 'auto') == 'auto':
            n_tokens = output.tokens.shape[0] - 1 + self.dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            for sli in output.und_image_slices + output.joint_image_slices:
                attention_mask[sli, sli] = True
            attention_mask = attention_mask.unsqueeze(0)    # head dim
        else:
            attention_mask = None

        # 2d rope
        rope_image_info = self.get_rope_image_info(sections, output)

        ret = {
            "data_type": "mmu",                         # data_type determines the loss type.
            "dtype": self.dataset_tag,                  # The prefix determines the prepare fn.
            "und_images": image_tensor,                 # (1, H, W)
            "n_samples": 1,                             # ()
            "tokens": output.tokens,                    # (L), L = text_token_length + 1 + image_token_length
            "target_tokens": target_token,              # (L)
            "und_image_mask": output.und_image_mask,    # (L)
            "text_mask": output.text_mask,              # (L)
        }
        if attention_mask is not None:
            ret["attention_mask"] = attention_mask
        elif self.use_joint_image_feature:
            ret["joint_image_slices"] = output.joint_image_slices
        else:
            ret["src_image_slices"] = output.src_image_slices
            ret["und_image_slices"] = output.und_image_slices
        if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            ret["vision_encoder_kwargs"] = {
                "spatial_shapes": kwargs["spatial_shapes"],
                "attention_mask": kwargs["pixel_attention_mask"],
            }
        if output.iw_ih_scatter_index is not None:
            # here use resized image size as scatter_src of iw and ih
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            iw_ih_scatter_src = [w, h] * (len(image_tensor) if image_tensor.ndim == 4 else 1)
            iw_ih_scatter_src = torch.tensor(iw_ih_scatter_src, dtype=torch.long)
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,      # (2n)
                "iw_ih_scatter_src": iw_ih_scatter_src,              # (2n)
            })
        if output.timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": output.timestep_scatter_index,   # (1)
            })
        if src_image is not None:
            ret["src_image"] = src_image
            ret["src_image_mask"] = src_image_mask
        if rope_image_info is not None:
            ret.update({
                "rope_image_info": rope_image_info,  # (2)
            })

        return ret

    def collate_fn(self, batch):
        if self.sequence_pack:
            return batch

        data_type = [item["data_type"] for item in batch]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])

        und_images = [item["und_images"] for item in batch]
        can_stack = all(im.shape == und_images[0].shape for im in und_images)
        if can_stack:
            und_images = torch.stack(und_images)

        if "src_image" in batch[0]:
            src_images = [item["src_image"] for item in batch]
            if src_images[0] is None:
                src_images = None
            else:
                try:
                    src_images = torch.stack(src_images)
                except Exception as e:
                    # here, src_images is a list of a list of tensors, the length of the first list is batch_size,
                    # the length of the second list is the number of source images of the i-th sample,
                    # and each tensor is c x h x w
                    pass
        else:
            src_images = None

        tokens = torch.stack([item["tokens"] for item in batch])
        target_tokens = torch.stack([item["target_tokens"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        src_image_mask = torch.stack([item["src_image_mask"] for item in batch]) if src_images is not None else None
        und_image_mask = torch.stack([item["und_image_mask"] for item in batch])
        
        if "iw_ih_scatter_index" in batch[0]:
            iw_ih_scatter_index = [item["iw_ih_scatter_index"] for item in batch]
            iw_ih_scatter_src = [item["iw_ih_scatter_src"] for item in batch]
            if can_stack:
                iw_ih_scatter_index = torch.stack(iw_ih_scatter_index)
                iw_ih_scatter_src = torch.stack(iw_ih_scatter_src)
        else:
            iw_ih_scatter_index = None
            iw_ih_scatter_src = None

        if "timestep_scatter_index" in batch[0]:
            timestep_scatter_index = [item["timestep_scatter_index"] for item in batch]
            if can_stack:
                timestep_scatter_index = torch.stack(timestep_scatter_index)
        else:
            timestep_scatter_index = None

        attention_mask = torch.stack([item["attention_mask"] for item in batch]) if "attention_mask" in batch[0] else None
        src_image_slices = [item["src_image_slices"] for item in batch] if "src_image_slices" in batch[0] else None
        und_image_slices = [item["und_image_slices"] for item in batch] if "und_image_slices" in batch[0] else None
        joint_image_slices = [item["joint_image_slices"] for item in batch] if "joint_image_slices" in batch[0] else None
        rope_image_info = [item["rope_image_info"] for item in batch] if "rope_image_info" in batch[0] else None

        vision_encoder_kwargs = None
        if "vision_encoder_kwargs" in batch[0]:
            # TODO(ckczzjzhang): now we only support n = 1, n > 1 should be supported
            vision_encoder_kwargs = {}
            vision_encoder_kwargs["spatial_shapes"] = torch.stack([item["vision_encoder_kwargs"]["spatial_shapes"] for item in batch])  # batch_size x n x seq_len x dim
            vision_encoder_kwargs["attention_mask"] = torch.stack([item["vision_encoder_kwargs"]["attention_mask"] for item in batch])  # batch_size x n x seq_len

        ret = {
            "data_type": data_type,
            "dtype": dtype,
            "src_images": src_images,
            "und_images": und_images,
            "n_samples": n_samples,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "und_image_mask": und_image_mask,
            "attention_mask": attention_mask,
            "src_image_slices": src_image_slices,
            "und_image_slices": und_image_slices,
            "joint_image_slices": joint_image_slices,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "timestep_scatter_index": timestep_scatter_index,
            "vision_encoder_kwargs": vision_encoder_kwargs,
            "rope_image_info": rope_image_info,
        }
        ret = {key: value for key, value in ret.items() if value is not None}

        return ret

    def seq_collate_fn(self, items):
        """ sequence collate function. It is used to combine multiple (non-batched) samples into one sample. """
        assert len(items) > 0
        assert self.max_sequence_length is not None, "max_sequence_length should be set for sequence packing."
        max_length = self.max_sequence_length
        first = items[0]
        seq_pad_value = {
            "tokens": self.tokenizer.pad_token,
            "target_tokens": -100,
            "text_mask": 0.0,
            "src_image_mask": False,
            "und_image_mask": False,
        }

        lengths = [item["tokens"].shape[0] for item in items]
        offsets = [0] + list(itertools.accumulate(lengths))
        assert offsets[-1] <= max_length, \
            f"Total length {offsets[-1]} exceeds max_length {max_length}. The lengths are {lengths}."

        new_item = {"offsets": torch.tensor(offsets)[None]}
        # Let batch size = 1
        for key, value in first.items():
            # ================ No sequence & no position keys ================
            # Simple constant values
            if key in {"data_type", "dtype"}:
                new_item[key] = [value]     # noqa
            # Accumulated values
            elif key in {"n_samples"}:
                new_item[key] = torch.tensor([sum(item[key] for item in items)])    # noqa
            # Image tensors
            elif key in {"src_image"}:
                image_list = []
                for item in items:
                    if isinstance(item[key], list):
                        for im in item[key]:
                            if im.ndim == 3:
                                image_list.append(im)
                            elif im.ndim == 4:
                                image_list.extend(im)
                            else:
                                raise ValueError(f"Unsupported image tensor shape: {im.shape} for key {key}")
                    elif isinstance(item[key], torch.Tensor):
                        if item[key].ndim == 3:
                            image_list.append(item[key])
                        elif item[key].ndim == 4:
                            image_list.extend(item[key])
                        else:
                            raise ValueError(f"Unsupported image tensor shape: {item[key].shape} for key {key}")
                    else:
                        raise ValueError(f"Unsupported image type: {type(item[key])} for key {key}")
                new_item["src_images"] = [image_list]  # list of 3-D Tensor     # noqa
            elif key in {"und_images"}:
                new_item[key] = torch.cat([item[key] for item in items])[None]  # 4-D Tensor  # noqa
            elif key in {"vision_encoder_kwargs"}:
                new_item[key] = {   # noqa
                    "spatial_shapes": torch.cat([item[key]["spatial_shapes"] for item in items])[None],   # Σn_j x 2
                    "attention_mask": torch.cat([item[key]["attention_mask"] for item in items])[None],     # Σn_j x seq_len(1024)
                }
            # ================ No sequence & positional keys ================
            # Slices list
            elif key in {"src_image_slices", "und_image_slices", "joint_image_slices"}:
                shifted = []
                for item, offset in zip(items, offsets):
                    for sli in item[key]:
                        shifted.append(slice(sli.start + offset, sli.stop + offset))
                new_item[key] = [shifted]   # noqa
            # Rope image info
            elif key in {"rope_image_info"}:
                shifted = []
                for item, offset in zip(items, offsets):
                    for sli, shape in item[key]:
                        shifted.append((slice(sli.start + offset, sli.stop + offset), shape))
                new_item[key] = [shifted]   # noqa
            # Scatter index
            elif key in {"timestep_scatter_index"}:
                src_ts_indices = []
                for item, offset in zip(items, offsets):
                    src_ts_indices.append(item[key] + offset)
                new_item[key] = torch.cat(src_ts_indices)[None]  # noqa
            # ================ Sequence & no positional keys ================
            elif key in seq_pad_value:
                cat_list = [item[key] for item in items]
                pad_length = max_length - sum(len(t) for t in cat_list)
                new_item[key] = torch.cat(  # noqa
                    cat_list + [torch.full((pad_length,), seq_pad_value[key], dtype=cat_list[0].dtype)]
                )[None]
            # ================ Not implemented keys ================
            elif key in {
                "iw_ih_scatter_index", "iw_ih_scatter_src", "token_bbox_mask", "attention_mask",
                "face_image_mask", "src_face_embedding",
            }:
                raise NotImplementedError()
            else:
                raise ValueError(f"Unsupported key: {key}")

        return new_item
