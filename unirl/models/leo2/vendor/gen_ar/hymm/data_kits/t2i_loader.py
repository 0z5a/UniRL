import io
import itertools
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict
from functools import partial
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

try:
    import pycocotools.mask as mask_util
except ImportError:
    warnings.warn("pycocotools is not installed. Some functions may not work properly.")
    mask_util = None
try:
    from insightface.app import FaceAnalysis # type: ignore
except ImportError:
    FaceAnalysis = None
from scipy import ndimage
from loguru import logger as all_logger

from .instruction_template import style_transfer_instructions_zh
from .caption_strategy import load_caption_processor
from .caption_strategy.manager import MultiCaptionManager
from .caption_strategy.prompt_patcher import apply_prompt_patchers
from .grounding_data_processors import CocoDataProcessor, GritDataProcessor, rle_to_mask
from .image_dataset import ImageDataset
from .index_dataset import IndexColumn, resample_on_gray
from .lama_mask import MixedMaskGenerator
from ..constants import VAE_META_INFO, VISION_ENCODER_META_INFO
from ..models.tokenizers.tokenizer_wrapper import TokenizerWrapper
from ..models.tokenizers.conversation import get_conversation_template
from ..models.visual_encoders import load_vision_model_processor
from ..utils.helpers import default, to_2tuple, count_zh_en_words
from ..utils.resolution import ResolutionGroup
from ..utils.image_base import ImageInfo, ImageTensor, JointImage
from ..data_kits.instruction_template import (
    text2image_instructions,
    inpainting_instructions,
    editing_instructions,
    controlnet_instructions,
    subject_driven_instructions,
    face_id_instructions,
    controlnet_condition_instructions,
    image_style_transfer_instructions,
    realistic_style_tag,
)
from ..data_kits.system_prompt import t2i_system_prompts, ti2i_system_prompts, unified_system_prompts
from processors.image_kits import read_binary_image


@dataclass
class TextImageData:
    prompt: str
    tgt_image: ImageTensor
    # Optional
    src_images: List[ImageTensor] = field(default_factory=list)
    max_num_srcs: int = 0
    cond_type_list: List[str] = field(default_factory=list)
    recaption: Optional[str] = None
    reasoning: Optional[str] = None
    face_embeddings: Optional[List[ImageTensor]] = field(default_factory=list)
    token_bbox_mask: Optional[torch.Tensor] = None
    und_images: Optional[List[ImageTensor]] = field(default_factory=list)
    joint_images: Optional[List[JointImage]] = field(default_factory=list)
    disallow_patchers: List[str] = field(default_factory=list)
    # Sample meta data
    image_flag: Optional[str] = None
    index: Optional[int] = None
    ref_image: Optional[ImageTensor] = None

    def __post_init__(self):
        assert len(self.src_images) <= self.max_num_srcs, \
            f"src_images({len(self.src_images)}) > max_num_srcs({self.max_num_srcs})"

        counts = defaultdict(int)
        for cond_type in self.cond_type_list:
            counts[cond_type] += 1
            # joint_image share the tensor containers of src_images and und_images
            if cond_type == "joint_image":
                counts["src_image"] += 1
                counts["und_image"] += 1
        for key, images in [
            ("src_image", self.src_images), ("und_image", self.und_images), ("face", self.face_embeddings)
        ]:
            count = counts.get(key, 0)
            assert count == len(images), f"cond_type count({count}) != {key}({len(images)})"

        # create joint_images if use_joint_image_feature is True
        if any(cond_type == "joint_image" for cond_type in self.cond_type_list):
            assert all(cond_type == "joint_image" for cond_type in self.cond_type_list), \
                f"All cond_type should be 'joint_image' if use_joint_image_feature is True, got {self.cond_type_list}"
            assert len(self.src_images) == len(self.und_images), \
                f"src_images({len(self.src_images)}) != und_images({len(self.und_images)})"
            self.joint_images = [JointImage(a, b) for a, b in zip(self.src_images, self.und_images)]

    @property
    def num_special_tokens(self):
        total = self.tgt_image.i.num_special_tokens
        if len(self.joint_images) > 0:
            # joint image is exclusive to src_image and und_image
            for joint_image in self.joint_images:
                total += joint_image.i.num_special_tokens
        else:
            for src_image in self.src_images:
                total += src_image.i.num_special_tokens  # for <joint_image> token
            for und_image in self.und_images:
                total += und_image.i.num_special_tokens
        for face_embedding in self.face_embeddings:
            total += face_embedding.i.num_special_tokens
        return total

    def get_cond_sections(self):
        sections = []
        index = defaultdict(int)
        cond2images = {
            "src_image": self.src_images,
            "und_image": self.und_images,
            "face": self.face_embeddings,
            "joint_image": self.joint_images,
        }
        for cond_type in self.cond_type_list:
            image: Union[ImageTensor, JointImage] = cond2images[cond_type][index[cond_type]]
            sections.append(dict(type=cond_type, **image.i.meta_info))
            index[cond_type] += 1
        return sections

    def get_image_length(self, dataset):
        total = dataset.image_token_length  # tgt_image
        if len(self.src_images):
            total += dataset.image_token_length * len(self.src_images)  # src_images
        if len(self.und_images):
            total += dataset.vision_encoder_token_length * len(self.und_images)  # und_images
        if len(self.face_embeddings):
            total += dataset.face_token_length * len(self.face_embeddings)  # face
        return total

    @property
    def iw_ih_scatter_src(self):
        wh_list = []
        for src_image in self.src_images:
            wh_list.extend([src_image.i.w, src_image.i.h])
        wh_list.extend([self.tgt_image.i.w, self.tgt_image.i.h])
        return torch.tensor(wh_list, dtype=torch.long)


class TextImageArrowStream(ImageDataset):
    """
    所有数据集公用的参数，通过 args 获取。
    数据集 specific 的参数，通过 task_kwargs 传入
    跟 index_manager 相关的参数，通过 index_kwargs 传入
    """
    def __init__(
        self,
        args,
        dataset_tag=None,
        tokenizer_name=None,
        task_kwargs=None,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        dummy_number=0,
        conv_format="hunyuan-gemini-alpha",
        template="pretrain",
        instruction_candidates=None,
        attn_mask_seq_m1=True,      # create attention mask with sequence length -1
        pad_color=(127, 127, 127),
        all_logger_after_init=True,
    ):
        self.dataset_tag = dataset_tag or "t2i"

        self.task_kwargs = default(task_kwargs, {})
        self.index_kwargs = default(index_kwargs, {})
        
        self.task_kwargs = self.strip_leading_tag(self.task_kwargs, required=False)
        self.index_kwargs = self.strip_leading_tag(self.index_kwargs, required=False)

        # Assign class attribute for easy access
        ImageInfo.args = args

        _base_size = self.task_kwargs.get('training_image_size', 256)
        super().__init__(
            index_file=self.index_kwargs['index_file'],
            multireso=self.index_kwargs.get('multireso', False),
            batch_size=self.index_kwargs.get("batch_size", 1),
            world_size=self.index_kwargs.get("world_size", 1),
            base_size=_base_size,
            logger=logger,
            debug=debug,
            args=args,
        )
        self.logger.info(f"    (T2I-{dataset_tag}) {task_kwargs=}")
        self.logger.info(f"    (T2I-{dataset_tag}) {index_kwargs=}")
        self.reso_group = ResolutionGroup(_base_size, extra_resolutions=self.args.get('extra_resolutions', None))
        self.vae_reso_group = self.reso_group   # Attribute `vae_reso_group` will be used for vae-prerun
        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.add_timestep_token = self.args.add_timestep_token
        self.use_front_boi_token = self.args.use_front_boi_token
        self.add_image_shape_token = self.args.get('add_image_shape_token', False)  # img_ratio_* and img_size_* tokens
        self.use_front_src_image = self.args.get('use_front_src_image', False)
        self.use_joint_image_feature = self.args.get('use_joint_image_feature', False)
        self.cond_type = "joint_image" if self.use_joint_image_feature else "src_image"
        self.dummy_number = dummy_number
        self.logger.info(f"    (T2I-{dataset_tag}) {self.base_size=}, {self.dummy_number=}")

        # unconditions
        self.uncond_p = self.task_kwargs.get('uncond_p', 0.0)
        self.uncond_p_src = self.task_kwargs.get('uncond_p_src', 0.0)
        self.multi_uncond_strategy = self.task_kwargs.get('multi_uncond_strategy', 'independent')
        self.ignore_text_ntp = self.task_kwargs.get('ignore_text_ntp', False)

        # VAE
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = self.args.patch_size
        self.h_factor = self.downsample_factor[0] * self.patch_size
        self.w_factor = self.downsample_factor[1] * self.patch_size

        # Vision Encoder
        self.has_vision_encoder = hasattr(args, "vision_model_type")
        if self.has_vision_encoder:
            self.vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
            if args.vision_model_type == "siglip2-so400m-patch16-naflex":
                self.vision_encoder_processor = load_vision_model_processor(args.vision_model_type)
                self.vision_encoder_h_factor, self.vision_encoder_w_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
            else:
                self.pad_color = pad_color
                self.vision_encoder_h_factor, self.vision_encoder_w_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
                self.vision_encoder_base_size: int = default(
                    self.task_kwargs.get('vision_encoder_image_size'), self.vision_encoder_meta_info["image_size"]
                )
                self.vision_encoder_image_size = to_2tuple(self.vision_encoder_base_size)
                self.vision_encoder_downsample_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
                self.vision_encoder_h_factor = self.vision_encoder_downsample_factor[0]
                self.vision_encoder_w_factor = self.vision_encoder_downsample_factor[1]

        self.logger.info(f"    (T2I-{dataset_tag}) {self.has_vision_encoder=}")

        # Prepare index manager
        self.setup_index_manager(f"T2I-{dataset_tag}")

        # Check required columns
        if self.cos_base is not None and any("{bucket}" in target for target in self.cos_base_targets):
            assert hasattr(self, "extra_url_cos_col"), \
                "When using cos_base with `{bucket}` template, `extra_url_cos_col` should be provided in index_kwargs."

            # Prepare image transformations
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

        self.logger.info(f"Image transform: {self.pil_image_to_tensor}")

        # Handle caption processor
        self.caption_sample_ratio = self.task_kwargs.get("caption_sample_ratio")
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=self.task_kwargs.get('caption_processor'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=self.task_kwargs.get('caption_processor_kwargs'),
            )
        else:
            self.use_structural_caption = False
        self.logger.info(f"Caption sample ratio: {self.caption_sample_ratio}")
        # 是否在特定场景下使用 `general_style` 列
        self.use_general_style = self.task_kwargs.get('use_general_style')
        # 是否优先使用 source_text (应用 image_caption_ratio 的概率)
        self.try_first_use_source_text = self.task_kwargs.get('try_first_use_source_text')
        # 控制多个 caption 版本
        image_caption_col_probs = self.task_kwargs.get('image_caption_col_probs', None)
        self.multi_caption_manager = MultiCaptionManager(
            resource=image_caption_col_probs,
            dataset=self,
            # backward compatibility
            caption_processor=self.task_kwargs.get('caption_processor'),
            caption_sample_ratio=self.task_kwargs.get('caption_sample_ratio'),
        )
        self.caption_cot = self.task_kwargs.get('caption_cot', None)
        self.caption_cot_prob = self.task_kwargs.get('caption_cot_prob', None)
        if self.caption_cot is not None and self.caption_cot_prob is None:
            raise ValueError(f"When using caption_cot, caption_cot_prob should be provided.")

        self.patcher_names = self.task_kwargs.get('patcher_names', [])

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)

        # Max token length for prompt section
        #   When cot is disabled (no reasoning, no recaption), text_token_length is used to bound the length of prompt.
        #   When cot is enabled, text_token_length is used to bound the length of recaption. In this case, prompt
        #   max length is calculated by `text_cot_token_length - text_token_length`.
        self.text_token_length = self.task_kwargs.get('text_token_length', 256)
        #   deprecated, use `text_cot_token_length` instead
        self.description_token_length = self.task_kwargs.get('description_token_length', 0)
        #   For cot, prompt+reason+recaption token length is bounded by `text_cot_token_length`
        self.text_cot_token_length = self.task_kwargs.get('text_cot_token_length', 0)
        #   reasoning length is bounded by `text_reason_token_length`
        self.text_reason_token_length = self.task_kwargs.get('text_reason_token_length', 0)
        #   description_token_length is deprecated, but we keep it for backward compatibility.
        if self.text_cot_token_length > 0:
            assert self.description_token_length == 0, \
                f"`description_token_length` is deprecated, please use `text_cot_token_length` instead."

            assert self.text_cot_token_length >= (self.text_token_length + self.text_reason_token_length), \
                f"`text_token_length` + `text_reason_token_length` should be less than or equal to `text_cot_token_length`, " \
                f"but got {self.text_token_length} + {self.text_reason_token_length} > {self.text_cot_token_length}"

        #   system prompt length is bounded by `system_prompt_token_length`
        self.system_prompt_token_length = self.task_kwargs.get('system_prompt_token_length', 0)

        self.all_text_max_length = self.system_prompt_token_length + max(self.text_token_length + self.description_token_length, self.text_cot_token_length)

        self.reasoning_cot_prob = self.task_kwargs.get('reasoning_cot_prob', 0)
        if self.reasoning_cot_prob > 0:
            assert self.text_reason_token_length > 0, \
                f"When reasoning_cot_prob > 0, text_reason_token_length should be provided with a positive value."
            assert (hasattr(self, "extra_think_en_col") and hasattr(self, "extra_think_zh_col")) \
                or (hasattr(self, "extra_think_en_key") and hasattr(self, "extra_think_zh_key")), \
                f"When reasoning_cot_prob > 0, extra_think_en_col and extra_think_zh_col or extra_think_en_key and extra_think_zh_key should be provided in index_kwargs."

        # Max token length for image section
        self.image_token_length = self.task_kwargs.get(
            'image_token_length', _base_size ** 2 // (self.h_factor * self.w_factor)
        )
        if self.has_vision_encoder:
            if args.vision_model_type == "siglip2-so400m-patch16-naflex":
                self.vision_encoder_token_length = args.vision_encoder_max_num_patches
                self.vision_encoder_processor = partial(self.vision_encoder_processor, max_num_patches=self.vision_encoder_token_length)
            else:
                self.vision_encoder_token_length = self.task_kwargs.get(
                    'vision_encoder_token_length',
                    self.vision_encoder_base_size ** 2 // (self.vision_encoder_h_factor * self.vision_encoder_w_factor)
                )
        self.face_token_length = self.task_kwargs.get("qformer_token_length", 16)

        # Sequence pack related and affected
        self.sequence_pack = self.task_kwargs.get('sequence_pack', False)
        if self.sequence_pack:
            # if sequence pack is enabled, attention mask sequence length -1 is disabled in __getitem__,
            # and performed in seq_collate_fn instead.
            self.attn_mask_seq_m1 = False
            # if sequence pack is enabled, we will use the sequence-wise dummy_number to pad the packed sequence,
            # instead of dummy_number to pad the sample. `block_size` will be used as the maximum sequence length
            # for all the datasets.
            self.max_sequence_length = self.args.block_size + 1 - self.dummy_number
        else:
            self.attn_mask_seq_m1 = attn_mask_seq_m1

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
            self.instruction_candidates = instruction_candidates

        # Instruction candidates type (for instruction tuning). When using self.instruction_candidates_type,
        # self.instruction_candidates will not be used and be ignored.
        self.instruction_candidates_type = self.task_kwargs.get("instruction_candidates_type", "fixed_set")
        assert self.instruction_candidates_type in {"fixed_set", "off", "merged"} or self.instruction_candidates_type.startswith("mix_up"), \
            f"Unsupported instruction_candidates_type: {self.instruction_candidates_type}"

        self.system_prompt_candidates_type = self.task_kwargs.get("system_prompt_candidates_type", "off")
        assert self.system_prompt_candidates_type in {"fixed_set", "off"} or self.system_prompt_candidates_type.startswith("mix_up"), \
            f"Unsupported system_prompt_candidates_type: {self.system_prompt_candidates_type}"

        self.use_unified_system_prompt = self.task_kwargs.get("use_unified_system_prompt", False)

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

        self.all_logger_after_init = all_logger_after_init
        if all_logger_after_init:
            # After init, we set the logger to all_logger to print warnings and errors of all ranks
            self.logger = all_logger

    def __post_init__(self, **kwargs):
        pass

    def parse_columns_and_register_shadow(self, index_kwargs):
        # 这四个别动, 为了向前兼容.
        self.image_col = IndexColumn(index_kwargs.get("image_col"), self, self.logger)
        self.image_col_2 = IndexColumn(index_kwargs.get("image_col_2"), self, self.logger)
        self.image_text_col = IndexColumn(index_kwargs.get("image_text_col"), self, self.logger)
        self.clip_score_col = IndexColumn(index_kwargs.get("clip_score_col"), self, self.logger)
        self.register_attributes(index_kwargs)

    def handle_exception_message(self, func, e, line_no, index=None):
        if self.args.raise_data_error:
            raise e
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"L{line_no} <- {func.__name__} | {e.__class__.__name__}: {message}. (index={index})")

    @staticmethod
    def get_uncond_flags(probs, strategy="independent"):
        if len(probs) == 1:
            flags = [probs[0] > 0 and random.random() < probs[0]]
        elif len(probs) == 2:
            if strategy == "independent":
                flags = []
                for prob in probs:
                    flags.append((prob > 0) and random.random() < prob)
            elif strategy == "dependent":
                prob_a, prob_b = probs
                assert prob_a + prob_b <= 1.0, f"{prob_a} + {prob_b} > 1.0"
                r = random.random()
                if r < prob_a:
                    flags = [True, False]
                elif r < prob_a + prob_b:
                    flags = [False, True]
                elif r < 2 * (prob_a + prob_b):
                    flags = [True, True]
                else:
                    flags = [False, False]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
        else:
            raise ValueError(f"Unsupported number of probs: {len(probs)}")
        return flags

    def get_instruction(self, instruction_candidates, with_space=False):
        if self.instruction_candidates_type == "fixed_set":
            return random.choice(instruction_candidates).strip() + (" " if with_space else "")
        elif self.instruction_candidates_type in ["off", "merged"]:
            return ""
        elif self.instruction_candidates_type.startswith("mix_up"):
            mix_up_ratio = float(self.instruction_candidates_type.split("@")[-1])
            if random.random() < mix_up_ratio:
                return random.choice(instruction_candidates).strip() + (" " if with_space else "")
            else: 
                return ""
        else:
            raise ValueError(f"Unsupported instruction_candidates_type: {self.instruction_candidates_type}")

    def get_system_prompt(self, system_prompt_candidates):
        if self.system_prompt_candidates_type == "off":
            return ""
        elif self.system_prompt_candidates_type == "fixed_set":
            return random.choice(system_prompt_candidates).strip()
        elif self.system_prompt_candidates_type.startswith("mix_up"):
            mix_up_ratio = float(self.system_prompt_candidates_type.split("@")[-1])
            if random.random() < mix_up_ratio:
                return random.choice(system_prompt_candidates).strip()
            else:
                return ""
        else:
            raise ValueError(f"Unsupported system_prompt_candidates_type: {self.system_prompt_candidates_type}")

    def get_text(self, ind, return_lang=False):
        image_caption_rate = self.task_kwargs['image_caption_rate']
        lang = None
        try:
            if self.try_first_use_source_text:
                source = self.index_manager.get_attribute(ind, 'source')
                try:
                    clip_score = self.index_manager.get_attribute(ind, **self.clip_score_col)
                except: # noqa
                    clip_score = None
                if (
                        (source == 'laion2b_en_sd2.1base' or (clip_score is not None and clip_score >= 0.18))
                        and (image_caption_rate == 0 or random.random() >= image_caption_rate)
                ):
                    text = self.index_manager.get_attribute(ind, **self.image_text_col)
                else:
                    text = self.index_manager.get_attribute(ind, **self.image_caption_col)  # noqa
                    if self.use_general_style:
                        style_op = self.index_manager.get_attribute(ind, 'general_style')
                    else:
                        style_op = None
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text, style_op=style_op)

            elif self.multi_caption_manager.enabled:
                if self.caption_cot:
                    caption_cot = self.caption_cot if random.random() < self.caption_cot_prob else None
                else:
                    caption_cot = None
                out = self.multi_caption_manager.get_caption(ind, return_dict=True, pattern=caption_cot)
                if isinstance(out, (tuple, list)):
                    # Make sure both captions are valid
                    if out[0].caption and out[1].caption:
                        text = (out[0].caption, out[1].caption)
                    else:
                        text = None
                    lang = out[0].lang
                else:
                    text = out.caption
                    lang = out.lang
            else:
                columns = self.index_manager.get_columns(ind)
                # First class: shadow caption (image_caption_col_3)
                col_3 = self.image_caption_col_3    # noqa
                if (col_3.key is not None
                        and Path(self.index_manager.get_arrow_file(ind, shadow=col_3.shadow)).exists()
                        and col_3.key in self.index_manager.get_columns(ind, shadow=col_3.shadow)):
                    # simple shadow caption
                    text = self.index_manager.get_attribute(ind, **self.image_caption_col_3)    # noqa

                # Second class: in-arrow caption (image_caption_col, image_caption_col_2)
                elif self.image_caption_col['column'] in columns:   # noqa
                    if image_caption_rate > 0.0 and random.random() < image_caption_rate:
                        text = self.index_manager.get_attribute(ind, **self.image_caption_col)  # noqa
                        style_op = None
                        if self.use_general_style and 'general_style' in columns:
                            style_op = self.index_manager.get_attribute(ind, 'general_style')
                        if self.use_structural_caption:
                            text = self.caption_aug.caption_aug(text, style_op=style_op)
                    else:
                        text = self.index_manager.get_attribute(ind, **self.image_text_col)
                elif self.image_caption_col_2['column'] in columns: # noqa
                    # simple caption
                    text = self.index_manager.get_attribute(ind, **self.image_caption_col_2)    # noqa
                else:
                    raise ValueError(
                        f"{self.image_caption_col} and {self.image_caption_col_2} are not found "   # noqa
                        f"in index columns: {columns}"
                    )

        except Exception as e:
            self.handle_exception_message(self.get_text, e, line_no=sys._getframe().f_lineno, index=ind)
            text = ""
        if text is None:
            text = ""

        if isinstance(text, str):
            text = str(text).strip()
            # Remove meaningless characters
            text = text.replace("\\N", "").strip("，,")
        elif isinstance(text, (tuple, list)):
            is_tuple = isinstance(text, tuple)
            text = [t.replace("\\N", "").strip("，,") for t in text]
            if is_tuple:
                text = tuple(text)
        
        # 返回语言
        if return_lang:
            return text, lang

        return text

    def as_image_tensor(self, image, image_type="vae", **kwargs) -> ImageTensor:
        if isinstance(image, Image.Image):
            tensor = self.pil_image_to_tensor(image)
        else:
            tensor = image
        if image_type == "vae":
            assert tensor.ndim == 3 or tensor.ndim == 4
            h, w = tensor.shape[-2], tensor.shape[-1]
            assert (h % self.h_factor == 0 and w % self.w_factor == 0), \
                (f"Image size should be divisible by downsample_factor * patch_size, "
                 f"but got ({h} x {w}) with {self.downsample_factor=} and {self.patch_size=}")
            tk_height = h // self.h_factor
            tk_width = w // self.w_factor
            base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(w, h)
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=w, image_height=h, token_width=tk_width, token_height=tk_height,
                base_size=base_size, ratio_index=ratio_idx,
            )
        elif image_type == "vision_encoder":
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                spatial_shapes = kwargs["spatial_shapes"]  # 2  (h, w)
                pixel_attention_mask = kwargs["pixel_attention_mask"]  # seq_len
                tensor.i = ImageInfo(
                    image_type=image_type,
                    image_width=spatial_shapes[1].item() * self.vision_encoder_w_factor,
                    image_height=spatial_shapes[0].item() * self.vision_encoder_h_factor,
                    token_width=spatial_shapes[1].item(),
                    token_height=spatial_shapes[0].item(),
                    image_token_length=self.vision_encoder_token_length,
                )
                tensor.vision_encoder_kwargs = {
                    "spatial_shapes": spatial_shapes,
                    "pixel_attention_mask": pixel_attention_mask,
                }
            else:
                assert tensor.ndim == 3 or tensor.ndim == 4
                h, w = tensor.shape[-2], tensor.shape[-1]
                assert (h % self.vision_encoder_h_factor == 0 and w % self.vision_encoder_w_factor == 0), \
                    (f"Image size should be divisible by downsample_factor * patch_size, "
                     f"but got ({h} x {w}) with {self.vision_encoder_downsample_factor=}")
                tk_height = h // self.vision_encoder_h_factor
                tk_width = w // self.vision_encoder_w_factor
                tensor.i = ImageInfo(
                    image_type=image_type,
                    image_width=w, image_height=h, token_width=tk_width, token_height=tk_height,
                )
                tensor.vision_encoder_kwargs = {}
        elif image_type == "face":
            tensor.i = ImageInfo(
                image_type=image_type,
                image_token_length=self.face_token_length,
            )
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        return tensor

    def vae_process_image(self, image, target_size, random_crop=False) -> ImageTensor:
        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, _ = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random" if random_crop else "center", resample=Image.Resampling.BICUBIC
        )
        return self.as_image_tensor(image, image_type="vae")

    def vision_encoder_process_image(self, image) -> ImageTensor:
        if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            inputs = self.vision_encoder_processor(image)
            image = inputs["pixel_values"].squeeze(0)   # seq_len x dim
            pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)   # seq_len
            spatial_shapes = inputs["spatial_shapes"].squeeze(0)   # 2  (h, w)
            return self.as_image_tensor(image, image_type="vision_encoder", pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
        else:
            image, _ = self.index_manager.resize_and_pad(
                image, self.vision_encoder_image_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
            )
            return self.as_image_tensor(image, image_type="vision_encoder")

    @staticmethod
    def resize_degradation(source_image):
        
        new_method = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        method = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])

        source_w, source_h = source_image.size 
        
        long_size = random.choice([256, 384, 512, 768])

        if source_w > source_h:
            new_source_w = long_size
            new_source_h = int(source_h * (long_size / source_w))
        else:
            new_source_h = long_size
            new_source_w = int(source_w * (long_size / source_h))

        resized_source_image = source_image.resize((new_source_w, new_source_h), new_method).resize((source_w, source_h), method)
        return resized_source_image

    def get_image_with_size(
            self,
            src: int | np.integer | dict[str, str | None],
            random_crop: bool = False,
            target_size_type: str = "index",
            return_vision_encoder_image: bool = False,
            real_index: int = None,
            apply_exif: bool = True,
            apply_resize_degradation: bool = False,
            **image_col,
    ) -> Tuple[ImageTensor, Optional[ImageTensor], str]:
        """ For various image generation tasks, dynamic image sizes """
        image, image_flag = self.get_raw_image(src, real_index=real_index, apply_exif=apply_exif, **image_col)
        if apply_resize_degradation:
            image = self.resize_degradation(image)
        origin_size = image.size  # (w_ori, h_ori)

        if self.multireso:
            if target_size_type == "index":
                index = src
                target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
            else:   # target_size_type == "image"
                target_size = self.reso_group.get_target_size(*origin_size)
        elif target_size_type == "image":
            target_size = self.reso_group.get_target_size(*origin_size)
        else:
            target_size = self.base_width, self.base_height

        image_tensor = self.vae_process_image(image, target_size, random_crop=random_crop)

        if return_vision_encoder_image:
            vision_encoder_image_tensor = self.vision_encoder_process_image(image)
            return image_tensor, vision_encoder_image_tensor, image_flag

        return image_tensor, None, image_flag

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
    def get_t2i_data(self, index):
        columns = self.index_manager.get_columns(index)
        # Get image
        image_col = self.image_col if 'cache_image' in columns else self.image_col_2
        # Only t2i task use random crop. For other editing tasks, make sure random_crop as False to use center crop.
        # T2i multireso bucket relies on the height/width property in data-pipeline which may not correctly handled
        # exif. So we disable exif here.
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=True, apply_exif=False, **image_col)

        # Get text
        prompt, lang = self.get_text(index, return_lang=True)
        if (isinstance(prompt, tuple) and prompt[0] == "") or prompt == "":
            image_flag = "gray"     # set `gray` flag to activate resample_on_gray

        # Apply prompt patcher
        if self.patcher_names and len(self.patcher_names) > 0 and lang is not None:
            prompt = apply_prompt_patchers(prompt, self.patcher_names, lang, index_manager=self.index_manager, index=index, ratio_index=tgt_image.i.ratio_index)

        # -- for recaption
        if isinstance(prompt, tuple):
            prompt, recaption = prompt
        else:
            recaption = None
        if self.template == "instruct" and lang is not None:
            instruction = self.get_instruction(text2image_instructions[lang], with_space=True)
            prompt = f"{instruction}{prompt}"

        # -- for reasoning
        if recaption is not None and self.reasoning_cot_prob > 0 and random.random() < self.reasoning_cot_prob:
            # self.extra_think_col should already registered in self.register_extra_cols()
            reasoning = self.index_manager.get_attribute(index, **getattr(self, f"extra_think_{lang}_col"))     # reasoning text
            # Remove leading and trailing <think> tags if they exist. These tags will be added in the template for
            # detailed control.
            if reasoning is not None and reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>"):]
            if reasoning is not None and reasoning.endswith("</think>"):
                reasoning = reasoning[:-len("</think>")]
        else:
            reasoning = None

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            recaption=recaption, reasoning=reasoning,
            image_flag=image_flag,
            index=index,
        )

    def __getitem__(self, index):
        data = self.get_t2i_data(index)
        index = data.index

        # 1 is <bos>.
        # <eos> is not included because it will be stripped in the shift of next-token-prediction
        extra_num_tokens = 1 + data.num_special_tokens + self.dummy_number
        # uncondition
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))
        # We want the image prefix special tokens <boi>, <img_size_*>, and <img_ratio_*> to be learned.
        # It is implemented by adding text mask end offsets to the prefix text sections.
        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0)

        if self.template == "pretrain":
            # t2i: xxxx<boi>[image]<eoi>
            # t2i with recaption: xxxx<recaption>yyyy</recaption><boi>[image]<eoi>
            # t2i with reasoning and recaption: xxxx<think>zzzz</think><recaption>yyyy</recaption><boi>[image]<eoi>
            if data.recaption:
                # If it has recaption, text length is limited by text_cot_token_length - text_token_length
                prompt_max_length = self.text_cot_token_length - self.text_token_length - self.text_reason_token_length
            else:
                # If no recaption, we assume no reasoning.
                prompt_max_length = self.text_token_length - extra_num_tokens
            prompt_section = [
                dict(type="text", text=data.prompt, max_length=prompt_max_length,
                     **uncond_kwargs, ignore=do_uncond or self.ignore_text_ntp),
            ]
            gen_section = [
                dict(type="text", text='', ignore=False, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text='', ignore=False, end_offset=1)    # include <eos> token, 1 token
            ]
            cot_section = []
            if data.reasoning:  # if not None and not empty
                cot_section.extend([
                    dict(type="text", text="<think>", ignore=True),  # start token is ignored to serve as a switch
                    dict(type="text", text=data.reasoning, ignore=do_uncond,
                         max_length=self.text_reason_token_length - 2, **uncond_kwargs),
                    dict(type="text", text="</think>", ignore=do_uncond),
                ])
            if data.recaption:  # if not None and not empty
                recaption_max_length = self.text_token_length - extra_num_tokens - 2
                cot_section.extend([
                    dict(type="text", text="<recaption>", ignore=True),   # start token is ignored to serve as a switch
                    dict(type="text", text=data.recaption, ignore=do_uncond,
                         max_length=recaption_max_length, **uncond_kwargs),
                    dict(type="text", text="</recaption>", ignore=do_uncond),
                ])

            sections = prompt_section + cot_section + gen_section

        elif self.template == "instruct":
            # User: xxxx\n\nAssistant: <answer><boi>[image]<eoi></answer>
            # User: xxxx\n\nAssistant: <recaption>yyyy</recaption><answer><boi>[image]<eoi></answer>
            # User: xxxx\n\nAssistant: <think>zzzz</think><recaption>yyyy</recaption><answer><boi>[image]<eoi></answer>
            extra_num_tokens += 9  # "User: " + "\n\n" + "Assistant: <answer>" + </answer>
            if data.reasoning:
                extra_num_tokens += 2  # <think> + </think>
            if data.recaption:
                extra_num_tokens += 2  # <recaption> + </recaption>

            if data.recaption:
                # If it has recaption, text length is limited by text_cot_token_length - text_token_length - text_reason_token_length
                prompt_max_length = self.text_cot_token_length - self.text_token_length - self.text_reason_token_length - extra_num_tokens
            else:
                # If no recaption, we assume no reasoning.
                prompt_max_length = self.text_token_length - extra_num_tokens

            if self.use_unified_system_prompt:
                system_prompt = self.get_system_prompt(unified_system_prompts["en_unified"])
            else:
                if data.recaption:
                    if data.reasoning:
                        system_prompt = self.get_system_prompt(t2i_system_prompts["en_think_recaption"])
                    else:
                        system_prompt = self.get_system_prompt(t2i_system_prompts["en_recaption"])
                else:
                    system_prompt = self.get_system_prompt(t2i_system_prompts["en_vanilla"])
                        
            if system_prompt == "":
                system_prompt_section = []
            else:
                assert self.system_prompt_token_length > 0, "system_prompt_token_length should be greater than 0"
                system_prompt_section = [
                    dict(type="text", text=system_prompt.strip("\n "), ignore=True, max_length=self.system_prompt_token_length - 1),
                    dict(type="text", text=self.default_conv.sep, ignore=True), # "\n\n" 1 token
                ]

            user_prefix_section = [
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True), # "User: " 3 tokens
            ]
            prompt_and_bot_prefix_section = [
                dict(type="text", text=data.prompt, max_length=prompt_max_length,
                     **uncond_kwargs, ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True), # "\n\n" 1 token
                dict(type="text", text=f"{self.roles[1]}: ", ignore=True), # "Assistant: " 3 tokens
            ]
            gen_section = [
                dict(type="text", text="<answer>", ignore=True),
                dict(type="text", text='', ignore=True if do_uncond else False, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text="</answer>", ignore=False),
                dict(type="text", text='', ignore=False, end_offset=1)
            ]
            cot_section = []
            if data.reasoning:  # if not None and not empty
                cot_section.extend([
                    dict(type="text", text="<think>", ignore=True),  # start token is ignored to serve as a switch
                    dict(type="text", text=data.reasoning, ignore=do_uncond,
                         max_length=self.text_reason_token_length, **uncond_kwargs),
                    dict(type="text", text="</think>", ignore=do_uncond),
                ])
            if data.recaption:  # if not None and not empty
                cot_section.extend([
                    dict(type="text", text="<recaption>", ignore=True),
                    dict(type="text", text=data.recaption, ignore=do_uncond,
                         max_length=self.text_token_length, **uncond_kwargs),
                    dict(type="text", text="</recaption>", ignore=do_uncond),
                ])

            sections = system_prompt_section + user_prefix_section + prompt_and_bot_prefix_section + cot_section + gen_section
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        # Build template and encode tokens
        max_token_length = self.max_sequence_length if self.sequence_pack else \
            (self.all_text_max_length + self.image_token_length + 1 - self.dummy_number)
        try:
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_pad=False if self.sequence_pack else 'auto',
            )
        except AssertionError as e:
            self.logger.error(
                f"Error in encoding sections (index={index}): {max_token_length=}, {self.sequence_pack=}, {sections=}"
            )
            raise e
        target_token = output.tokens.clone()
        target_token[output.text_mask == 0.0] = -100

        # Prepare attention mask
        if self.task_kwargs.get('attn_type', 'auto') == 'auto':
            n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + self.dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            for image_slice in output.gen_image_slices:
                attention_mask[image_slice, image_slice] = True
            attention_mask = attention_mask.unsqueeze(0)    # head dim
        else:
            attention_mask = None

        # 2d rope
        rope_image_info = self.get_rope_image_info(sections, output)

        ret = {
            "data_type": "t2i",
            "dtype": self.dataset_tag,
            "image": data.tgt_image,                # (3, H~, W~)
            "n_samples": 1,                         # ()
            "tokens": output.tokens,                # (L), L = text_token_length + 1 + image_token_length
            "target_tokens": target_token,          # (L)
            "text_mask": output.text_mask,          # (L)
            "image_mask": output.gen_image_mask,    # (L)
        }
        if attention_mask is not None:
            ret["attention_mask"] = attention_mask
        else:
            ret["gen_image_slices"] = output.gen_image_slices
        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,  # (2)
                "iw_ih_scatter_src": data.iw_ih_scatter_src,        # (2)
            })
        if output.timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": output.timestep_scatter_index,   # (1)
            })
        if rope_image_info is not None:
            ret.update({
                "rope_image_info": rope_image_info,  # (2)
            })

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
            "image_mask": False,
            "und_image_mask": False,
        }

        lengths = [item["tokens"].shape[0] for item in items]
        offsets = [0] + list(itertools.accumulate(lengths))
        assert offsets[-1] <= max_length, \
            f"Total length {offsets[-1]} exceeds max_length {max_length}. The lengths are {lengths}."

        def num_images(item, key):
            if key not in item:
                return 0
            return len(item[key]) if isinstance(item[key], list) else (
                item[key].shape[0] if item[key].ndim == 4 else 1
            )

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
            elif key in {"image", "src_image"}:
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
                new_item["src_images" if key == "src_image" else key] = [image_list]  # [list of 3-D Tensor]     # noqa
            elif key in {"und_images"}:
                new_item[key] = torch.cat([item[key] for item in items])[None]  # 4-D Tensor  # noqa
            elif key in {"vision_encoder_kwargs"}:
                new_item[key] = {   # noqa
                    "spatial_shapes": torch.cat([item[key]["spatial_shapes"] for item in items])[None],   # Σn_j x 2
                    "attention_mask": torch.cat([item[key]["pixel_attention_mask"] for item in items])[None],     # Σn_j x seq_len(1024)
                }
            # ================ No sequence & positional keys ================
            # Slices list
            elif key in {"src_image_slices", "und_image_slices", "joint_image_slices", "gen_image_slices"}:
                shifted = []
                for item, offset in zip(items, offsets):
                    for sli in item[key]:
                        shifted.append(slice(sli.start + offset, sli.stop + offset))
                new_item[key] = [shifted]   # noqa
            # Rope image info
            elif key in {"rope_image_info"}:
                shifted = []
                num_overlapped_tokens = 0
                for item, offset in zip(items, offsets):
                    # Here we need minus the number of overlap tokens from offset for interleave sequence.
                    offset -= num_overlapped_tokens
                    for sli, shape in item[key]:
                        shifted.append((slice(sli.start + offset, sli.stop + offset), shape))
                    # accumulate the number of overlap tokens
                    num_overlapped_tokens += item.get("n_overlapped_tokens", 0)
                new_item[key] = [shifted]   # noqa
            # Scatter index
            elif key in {"timestep_scatter_index"}:
                src_ts_indices = []
                tgt_ts_indices = []
                for item, offset in zip(items, offsets):
                    n_src = num_images(item, "src_image")
                    n_tgt = num_images(item, "image")
                    assert n_src + n_tgt == item[key].shape[0], \
                        f"Scatter index length mismatch: {n_src} + {n_tgt} != {item[key].shape[0]}"
                    src_ts_indices.append(item[key][:n_src] + offset)
                    tgt_ts_indices.append(item[key][n_src:] + offset)
                new_item[key] = torch.cat(src_ts_indices + tgt_ts_indices)[None]  # noqa
            # ================ Sequence & no positional keys ================
            elif key in seq_pad_value:
                cat_list = [item[key] for item in items]
                pad_length = max_length - sum(len(t) for t in cat_list)
                new_item[key] = torch.cat(  # noqa
                    cat_list + [torch.full((pad_length,), seq_pad_value[key], dtype=cat_list[0].dtype)]
                )[None]
            # ================ Not implemented keys ================
            elif key in {"n_overlapped_tokens"}:
                pass
            elif key in {
                "iw_ih_scatter_index", "iw_ih_scatter_src", "token_bbox_mask", "attention_mask",
                "face_image_mask", "src_face_embedding",
            }:
                raise NotImplementedError()
            else:
                raise ValueError(f"Unsupported key: {key}")

        return new_item

    def collate_fn(self, batch, force=False):
        if self.sequence_pack and not force:
            return batch

        data_type = [item["data_type"] for item in batch]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])

        if "src_image" in batch[0]:
            src_images = [item["src_image"] for item in batch]
            if src_images[0] is None:
                src_images = None
            else:
                try:
                    src_images = torch.stack(src_images)
                except Exception as e:  # noqa
                    # here, src_images is a list of a list of tensors, the length of the first list is batch_size,
                    # the length of the second list is the number of source images of the i-th sample,
                    # and each tensor is c x h x w
                    pass
        else:
            src_images = None

        und_images = None
        if "und_images" in batch[0]:
            und_images = [item["und_images"] for item in batch]
            if und_images[0] is None:
                und_images = None
            else:
                try:
                    und_images = torch.stack(und_images)
                except Exception as e:  # noqa
                    # here, und_images is a list of tensors, the length of the list is batch_size,
                    # and each tensor is n_{i} x c x h x w or n_{i} x seq_len x dim
                    pass

        vision_encoder_kwargs = None
        if "vision_encoder_kwargs" in batch[0]:
            vision_encoder_kwargs = {}
            try:
                # batch_size x 2 or batch_size x n x 2 (n > 1)
                vision_encoder_kwargs["spatial_shapes"] = torch.stack([item["vision_encoder_kwargs"]["spatial_shapes"] for item in batch])
            except Exception as e:  # noqa
                # list of n_{i} x 2, where n_{i} may be different for each sample
                # in this case, und_images must be a list of tensors
                vision_encoder_kwargs["spatial_shapes"] = [item["vision_encoder_kwargs"]["spatial_shapes"] for item in batch]
            try:
                # batch_size x seq_len or batch_size x n x seq_len
                vision_encoder_kwargs["attention_mask"] = torch.stack([item["vision_encoder_kwargs"]["pixel_attention_mask"] for item in batch])
            except Exception as e:  # noqa
                # list of n_{i} x seq_len, where n_{i} may be different for each sample
                # in this case, und_images must be a list of tensors
                vision_encoder_kwargs["attention_mask"] = [item["vision_encoder_kwargs"]["pixel_attention_mask"] for item in batch]

        can_stack = isinstance(src_images, torch.Tensor) or src_images is None

        if self.sequence_pack == "reso_bucket":
            # If sequence_pack enabled, we merge the batch and n_i dimensions into one batch dimension
            # for accelerated vae encoding. One can split it using batch["n_samples"].
            image = [im for item in batch for im in item["image"]]
            image = torch.stack(image)
            can_stack = False
        else:
            image = [item["image"] for item in batch]
            if all(isinstance(x, torch.Tensor) for x in image) and all(x.shape == image[0].shape for x in image):
                image = torch.stack(image)
            else:
                can_stack = False

        if "src_face_embedding" in batch[0].keys():
            src_face_embedding = torch.stack([item["src_face_embedding"] for item in batch])
        else:
            src_face_embedding = None
        if "token_bbox_mask" in batch[0].keys():
            token_bbox_mask = torch.stack([item["token_bbox_mask"] for item in batch])
        else:
            token_bbox_mask = None

        tokens = torch.stack([item["tokens"] for item in batch])
        target_tokens = torch.stack([item["target_tokens"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        image_mask = torch.stack([item["image_mask"] for item in batch])
        src_image_mask = torch.stack([item["src_image_mask"] for item in batch]) if src_images is not None else None
        
        # Add und_image_mask, face_image_mask to support multiple conditions for face_id task
        if "und_image_mask" in batch[0]:
            und_image_mask = torch.stack([item["und_image_mask"] for item in batch])
        else:
            und_image_mask = None

        if "face_image_mask" in batch[0]:
            face_image_mask = torch.stack([item["face_image_mask"] for item in batch])
        else:
            face_image_mask = None

        if "iw_ih_scatter_index" in batch[0]:
            iw_ih_scatter_index = [item["iw_ih_scatter_index"] for item in batch]
            iw_ih_scatter_src = [item["iw_ih_scatter_src"] for item in batch]
            if self.sequence_pack:
                raise NotImplementedError()
            elif can_stack:
                iw_ih_scatter_index = torch.stack(iw_ih_scatter_index)
                iw_ih_scatter_src = torch.stack(iw_ih_scatter_src)
        else:
            iw_ih_scatter_index = None
            iw_ih_scatter_src = None

        if "timestep_scatter_index" in batch[0]:
            timestep_scatter_index = [item["timestep_scatter_index"] for item in batch]
            if self.sequence_pack == "reso_bucket":
                timestep_scatter_index = torch.cat(timestep_scatter_index)
            elif can_stack:
                timestep_scatter_index = torch.stack(timestep_scatter_index)
        else:
            timestep_scatter_index = None

        attention_mask = torch.stack([item["attention_mask"] for item in batch]) if "attention_mask" in batch[0] else None

        gen_image_slices = [item["gen_image_slices"] for item in batch] if "gen_image_slices" in batch[0] else None
        src_image_slices = [item["src_image_slices"] for item in batch] if "src_image_slices" in batch[0] else None
        und_image_slices = [item["und_image_slices"] for item in batch] if "und_image_slices" in batch[0] else None
        joint_image_slices = [item["joint_image_slices"] for item in batch] if "joint_image_slices" in batch[0] else None
        face_image_slices = [item["face_image_slices"] for item in batch] if "face_image_slices" in batch[0] else None
        rope_image_info = [item["rope_image_info"] for item in batch] if "rope_image_info" in batch[0] else None
        offsets = [torch.tensor(item["offsets"]) for item in batch] if "offsets" in batch[0] else None

        ret = {
            "data_type": data_type,
            "dtype": dtype,
            "n_samples": n_samples,
            "src_images": src_images,
            "und_images": und_images,
            "image": image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "und_image_mask": und_image_mask,
            "face_image_mask": face_image_mask,
            "image_mask": image_mask,
            # Conditional values
            "attention_mask": attention_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,  # (2)
            "iw_ih_scatter_src": iw_ih_scatter_src,      # (2)
            "timestep_scatter_index": timestep_scatter_index,   # (1)
            "src_face_embedding": src_face_embedding,
            "gen_image_slices": gen_image_slices,
            "src_image_slices": src_image_slices,
            "und_image_slices": und_image_slices,
            "joint_image_slices": joint_image_slices,
            "face_image_slices": face_image_slices,
            "token_bbox_mask": token_bbox_mask,
            "rope_image_info": rope_image_info,
            # kwargs for vision encoder
            "vision_encoder_kwargs": vision_encoder_kwargs,
            # for sequence pack
            "offsets": offsets,
        }
        ret = {key: value for key, value in ret.items() if value is not None}

        return ret


class TextImageToImageArrowStream(TextImageArrowStream):
    def __init__(
            self,
            args,
            dataset_tag,
            tokenizer_name=None,
            task_kwargs=None,
            index_kwargs=None,
            post_kwargs=None,
            debug=False,
            logger=None,
            dummy_number=0,
            conv_format="hunyuan-gemini-alpha",
            template="pretrain",
            instruction_candidates=None,
            attn_mask_seq_m1=True,      # create attention mask with sequence length -1
            all_logger_after_init=True,
    ):
        # 存储数据集标签标识
        self.dataset_tag = dataset_tag

        # 去除前缀
        task_kwargs_spec = self.strip_leading_tag(task_kwargs)

        # 初始化index_kwargs_spec字典，包含通用索引参数
        index_kwargs_spec = dict(
            # Common index kwargs
            batch_size=index_kwargs.pop(f"batch_size", 1),
            world_size=index_kwargs.pop(f"world_size", 1),
        )
        index_kwargs_spec.update(self.strip_leading_tag(index_kwargs))

        super().__init__(
            args=args,
            dataset_tag=dataset_tag,
            tokenizer_name=tokenizer_name,
            task_kwargs=task_kwargs_spec,
            index_kwargs=index_kwargs_spec,
            post_kwargs=post_kwargs,
            debug=debug,
            logger=logger,
            dummy_number=dummy_number,
            conv_format=conv_format,
            template=template,
            instruction_candidates=instruction_candidates,
            attn_mask_seq_m1=attn_mask_seq_m1,
            all_logger_after_init=False,
        )

        # 从task_kwargs中获取无条件概率源参数，默认值为0.0
        self.uncond_p_src = self.task_kwargs.get('uncond_p_src', 0.0)

        # 如果task_kwargs中包含mask生成器概率配置，则创建混合掩码生成器
        if "mask_generator_probs" in self.task_kwargs:
            self.mask_provider = MixedMaskGenerator(mask_generator_probs=self.task_kwargs["mask_generator_probs"])

        # prepare grounding data processor
        # ---- n_select: int, the number of masks to sample each time, default is None (sample masks randomly)
        # ---- sample_mask_rate: float, the rate of sampling masks, default is 0.9, otherwise sample box-type masks
        # ---- box_score_threshold: float, the threshold of box score, default is 0.65
        # ---- pred_mask_type: str, the type of grounding mask, can be "masked_image" or "pure_mask"

        # 图像编辑任务
        if "editing" in self.dataset_tag:
            self.grounding_with_caption = self.task_kwargs.get(f"grounding_with_caption", False)
            self.grounding_pred_mask_type = self.task_kwargs.get(f"grounding_pred_mask_type", "masked_image")

            # 初始化COCO数据处理器，用于处理COCO数据集
            self.coco_data_processor = CocoDataProcessor(
                index_manager=self.index_manager,
                sample_mask_rate=self.task_kwargs.get(f"grounding_coco_sample_mask_rate", 0.9),
                n_select=self.task_kwargs.get(f"grounding_n_select", None),
            )

            # 初始化GRIT数据处理器，用于处理GRIT数据集
            self.grit_data_processor = GritDataProcessor(
                index_manager=self.index_manager,
                box_score_threshold=self.task_kwargs.get(f"grounding_grit_box_score_threshold", 0.65),
                n_select=self.task_kwargs.get(f"grounding_n_select", None),
            )

        # 创建用于图像膨胀操作的结构元素（2D，连通性为2）
        self.dilation_struct = ndimage.generate_binary_structure(rank=2, connectivity=2)

        # Photomaker
        if "caption_sample_ratio_photomaker" in self.task_kwargs:
            self.caption_sample_ratio_photomaker = self.task_kwargs["caption_sample_ratio_photomaker"]
            self.caption_sample_ratio_photomaker = json.loads(self.caption_sample_ratio_photomaker)
            self.caption_aug_photomaker = load_caption_processor(
                name=self.task_kwargs.get('caption_processor_photomaker'),
                caption_sample_ratio=self.caption_sample_ratio_photomaker,
                logger=self.logger,
                kwargs=self.task_kwargs.get('caption_processor_kwargs_photomaker', None),
            )
            self.logger.info(f"Caption sample ratio (Photomaker): {self.caption_sample_ratio_photomaker}")

        # 人脸相关处理
        if "face_id" in self.dataset_tag:
            self.face_uncond_p = self.task_kwargs.get("face_uncond_p", 0.0)

            # 检查是否需要人脸分析（基于裁剪、局部损失权重或条件类型配置）
            if self.task_kwargs.get("crop_face", False) is True \
                    or "local_loss_weight" in self.task_kwargs \
                    or "face_embedding" in self.task_kwargs.get("condition_type", "clip"):
                self.face_analysis = self.get_face_analysis(self.logger)
            else:
                self.face_analysis = None
                self.logger.info("Face analysis is not used")

        # 设置日志配置标志
        self.all_logger_after_init = all_logger_after_init

        # 如果需要在初始化后设置全局日志器，则进行设置
        if all_logger_after_init:
            self.logger = all_logger

    def parse_columns_and_register_shadow(self, index_kwargs):
        """ Parse data columns for different tasks and register to self.shadow_file_fn """
        self.src_image_col = IndexColumn(index_kwargs.get("src_image_col"), self, self.logger)
        self.tgt_image_col = IndexColumn(index_kwargs.get("tgt_image_col"), self, self.logger)
        self.image_caption_col = IndexColumn(index_kwargs.get("image_caption_col"), self, self.logger)
        self.image_caption_col_2 = IndexColumn(index_kwargs.get("image_caption_col_2"), self, self.logger)
        # Editing_extra src/tgt image
        self.extra_src_image_col = IndexColumn(index_kwargs.get("extra_src_image_col"), self, self.logger)
        self.extra_tgt_image_col = IndexColumn(index_kwargs.get("extra_tgt_image_col"), self, self.logger)
        # Face ID
        self.face_bbox_col = IndexColumn(index_kwargs.get("face_bbox_col"), self, self.logger)
        self.face_embed_col = IndexColumn(index_kwargs.get("face_embed_col"), self, self.logger)
        self.face_photomaker_filtered_columns_col = IndexColumn(index_kwargs.get("face_photomaker_filtered_columns_col"), self, self.logger)
        self.merged_instruction = IndexColumn(index_kwargs.get("merged_instruction"), self, self.logger)
        # Inpainting V2
        self.inpainting_image_col = IndexColumn(index_kwargs.get("inpainting_image_col"), self, self.logger)
        self.inpainting_caption_col = IndexColumn(index_kwargs.get("inpainting_caption_col"), self, self.logger)
        # Controlnet
        self._controlnet_src_cols = index_kwargs.get("controlnet_src_cols", [])
        self._controlnet_src_cols_set = set(self._controlnet_src_cols)
        self.controlnet_src_cols_dict = {
            src_col: IndexColumn(src_col, self, self.logger)
            for src_col in self._controlnet_src_cols
        }
        self.controlnet_caption_col = IndexColumn(index_kwargs.get("controlnet_caption_col"), self, self.logger)
        self.controlnet_caption_v2_col = IndexColumn(index_kwargs.get("controlnet_caption_v2_col"), self, self.logger)
        self.controlnet_pred_condition_ratio = index_kwargs.get("controlnet_pred_condition_ratio", None)
        
        self.editing_online_cot_image_caption_col = IndexColumn("vlm_prompt_en", self, self.logger)

    def get_simple_prompt(self, index, **prompt_col):
        if len(prompt_col) == 0:
            prompt_col = self.image_caption_col
        try:
            instruction = self.index_manager.get_attribute(index, **prompt_col)
            if isinstance(instruction, list):
                instruction = random.choice(instruction)
        except Exception as e:
            self.handle_exception_message(self.get_simple_prompt, e, line_no=sys._getframe().f_lineno)
            instruction = ""
        instruction = str(instruction).strip()

        # Remove meaningless characters
        instruction = instruction.replace("\\N", "").strip("，,")
        return instruction

    def get_caption_prompt(self, index, column=None, shadow=None, caption_aug=None, patch_exception=True):
        image_caption_col = self.image_caption_col if column is None else dict(column=column, shadow=shadow)
        try:
            content_prompt = self.index_manager.get_attribute(index, **image_caption_col)
            if self.use_structural_caption:
                if caption_aug is None:
                    caption_aug = self.caption_aug
                content_prompt = caption_aug.caption_aug(content_prompt)
        except Exception as e:
            if patch_exception:
                self.handle_exception_message(self.get_caption_prompt, e, line_no=sys._getframe().f_lineno)
                content_prompt = ""
            else:
                return None
        content_prompt = str(content_prompt).strip()

        # Remove meaningless characters
        content_prompt = content_prompt.replace("\\N", "").strip("，,")

        return content_prompt

    # ========================================
    #           Inpainting
    # ========================================
    def get_inpainting_data(self, index):
        max_num_srcs = 1

        # Get image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.tgt_image_col)
        mask = self.mask_provider(h=tgt_image.i.h, w=tgt_image.i.w)      # (1, h, w)
        masked_src_image = self.as_image_tensor(tgt_image * (1 - mask), image_type="vae")     # mask with gray

        # Get text
        prompt = self.get_caption_prompt(index)
        # -- for cot
        if isinstance(prompt, tuple):
            prompt, recaption = prompt
        else:
            recaption = None
        if self.template == "instruct":
            instruction = self.get_instruction(inpainting_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[masked_src_image], max_num_srcs=max_num_srcs, cond_type_list=["src_image"],
            recaption=recaption,
        )

    def get_inpainting_v2_data(self, index):
        # This method is for Gemini Beta
        max_num_srcs = 1

        # Get image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.inpainting_image_col)
        h, w = tgt_image.i.h, tgt_image.i.w
        # TODO: make sure masks are valid and the mask has the same size as image
        try:
            if random.random() > self.task_kwargs['inpainting_lama_ratio']:
                # pre-calculated rle mask
                mask_json = json.loads(self.index_manager.get_attribute(index, 'mask_json'))
                num_masks = len(mask_json['annotations'])
                # random select a mask
                mask_idx = random.randint(0, num_masks - 1)
                # convert rle to 0-1 2D mask(ndarray uint8)
                mask = rle_to_mask(mask_json['annotations'][mask_idx]['segmentation'])
                # dilation to avoid boundary leakage
                mask = ndimage.binary_dilation(mask, structure=self.dilation_struct, iterations=8).astype(np.float32)   # noqa
                mask = torch.from_numpy(mask)[None]     # (1, h, w)
                if mask.shape[1:] != tgt_image.shape[1:]:
                    mask, _ = self.tensor_resize_and_crop(mask, w, h, mode="nearest", crop_type="center")
            else:
                # lama random mask
                mask = self.mask_provider(h=h, w=w)     # (1, h, w)
        except Exception as e:
            self.handle_exception_message(self.get_inpainting_v2_data, e, line_no=sys._getframe().f_lineno)
            # Fallback to lama random mask
            mask = self.mask_provider(h=h, w=w)     # (1, h, w)
        masked_src_image = self.as_image_tensor(tgt_image * (1 - mask), image_type="vae")     # mask with gray

        und_images = []
        if self.use_joint_image_feature:
            masked_src_pil_image = self.tensor_to_pil_image(masked_src_image)
            und_image = self.vision_encoder_process_image(masked_src_pil_image)
            und_images.append(und_image)

        # Get text
        prompt = self.get_caption_prompt(index, **self.inpainting_caption_col)
        if self.template == "instruct":
            instruction = self.get_instruction(inpainting_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[masked_src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
        )

    # ========================================
    #           Editing
    # ========================================
    def get_editing_data(self, index):
        max_num_srcs = 1
        # Get source and target images
        src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature, **self.src_image_col)
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.tgt_image_col)
        und_images = [src_vision_encoder_image] if self.use_joint_image_feature else []

        # Get text
        prompt = self.get_simple_prompt(index)
        if self.template == "instruct":
            instruction = self.get_instruction(editing_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
        )

    # ========================================
    #           CoT Editing
    # ========================================
    def get_cot_editing_data(self, index):
        max_num_srcs = 1
        # Get source and target images
        src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature, **self.src_image_col)
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.tgt_image_col)
        und_images = [src_vision_encoder_image] if self.use_joint_image_feature else []

        # Get text
        prompt = self.get_simple_prompt(index)
        if self.template == "instruct":
            instruction = self.get_instruction(editing_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"
        description = self.get_simple_prompt(index, **self.editing_online_cot_image_caption_col)

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
            recaption=description,
        )
    
    # ========================================================================
    #                           Editing Extra
    #  e.g. editing_caimai, which use 'src_cache_image' and 'tgt_cache_image'
    # ========================================================================
    def get_editing_extra_data(self, index):
        max_num_srcs = 1
        # Get source and target images
        src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature, **self.extra_src_image_col)
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.extra_tgt_image_col)
        und_images = [src_vision_encoder_image] if self.use_joint_image_feature else []

        # Get text
        prompt = self.get_simple_prompt(index)
        if self.template == "instruct":
            instruction = self.get_instruction(editing_instructions, with_space=True)
            prompt = f"{instruction}{prompt}"

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
        )

    # ========================================
    #           Controlnet
    # ========================================
    def get_controlnet_data(self, index):
        max_num_srcs = 1
        do_swap = self.controlnet_pred_condition_ratio is not None and random.random() < self.controlnet_pred_condition_ratio
        # Get target image
        tgt_image, tgt_vision_encoder_image, tgt_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature and do_swap, **self.tgt_image_col)
        # Random select a source image
        columns = set(list(self.index_manager.get_columns(index)))
        valid_src_cols = list(self._controlnet_src_cols_set & columns)
        src_col = random.choice(valid_src_cols)
        src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature and not do_swap, **self.controlnet_src_cols_dict[src_col])

        # Get text
        prompt = None
        if "tgt_caption_v2" in self.index_manager.get_columns(index):
            prompt = self.get_caption_prompt(index, **self.controlnet_caption_v2_col, patch_exception=False)
        if prompt is None:
            prompt = self.get_simple_prompt(index, **self.controlnet_caption_col)
        
        if self.template == "pretrain":
            prompt = prompt
        elif self.template == "instruct":
            instruction = self.get_instruction(controlnet_instructions, with_space=True)
            if '{}' in instruction:
                instruction = instruction.format(src_col.replace('_', ' '))
            prompt = f"{instruction}{prompt}"
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        # Reverse the source and target images to predict the condition image
        if do_swap:
            prompt = random.choice(controlnet_condition_instructions).format(src_col.replace('_', ' '))
            src_image, tgt_image = tgt_image, src_image
            und_images = [tgt_vision_encoder_image] if self.use_joint_image_feature else []
        else:
            und_images = [src_vision_encoder_image] if self.use_joint_image_feature else []

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
        )

    # ========================================
    #           Grounding
    # ========================================
    def get_grounding_image_or_mask_with_size(
            self,
            data_processor,
            index,
            random_crop=False,
            return_type: str = "normal_image",
            mask: Optional[Image.Image] = None,
            image: Optional[Image.Image] = None,
    ):
        """
        Get the image or mask with size for grounding task. 

        return_type: str. Can be "pure_mask", "masked_image" or "normal_image".
            - If `return_type` is "pure_mask", the mask is required and should be a PIL.Image.Image (RGB mode).
            - If `return_type` is "masked_image", both the image and mask are required and should be a PIL.Image.Image (RGB mode).
            - If `return_type` is "normal_image", return the normal image with the index only.
        """
        assert return_type in ["pure_mask", "masked_image", "normal_image"], \
            f"return_type is expected to be one of ['pure_mask', 'masked_image', 'normal_image'], but got {return_type}"
        
        if return_type == "pure_mask":
            assert isinstance(mask, Image.Image), f"mask is expected to be a PIL.Image.Image, but got {type(mask)}"
            image_flag = "normal"
            image = mask
        elif return_type == "masked_image":
            # `image_and_mask` is a list of two PIL.Image.Image, the first is the image, the second is the mask
            assert image is not None and mask is not None, f"image and mask are required when return_type is 'masked_image'"
            assert isinstance(image, Image.Image) and isinstance(mask, Image.Image), f"image and mask are expected to be PIL.Image.Image, but got {type(image)} and {type(mask)}"
            image_flag = "normal"
            # Apply mask to image, to generate the masked image
            image = data_processor.mask_apply_to_image(image, mask)
        else:
            try:
                img_bytes = self.index_manager.get_attribute(index, "image")
                img_bytes = img_bytes if isinstance(img_bytes, bytes) else img_bytes["bytes"]
                image = read_binary_image(img_bytes, apply_exif=True)
                image_flag = "normal"
            except Exception as e:
                # PIL.UnidentifiedImageError: cannot identify image file
                self.logger.error(f"{type(e)}: {e}")
                image = Image.new("RGB", self.base_size, (128, 128, 128))
                image_flag = "gray"

        origin_size = image.size  # (w_ori, h_ori)
        origin_image = image.copy()

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.base_width, self.base_height

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random" if random_crop else "center", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.as_image_tensor(image, image_type="vae")

        kwargs = {
            "image": origin_image,
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs

    def get_grounding_data(self, index):
        if self.use_joint_image_feature:
            raise NotImplementedError(f"Joint image feature is not supported for {self.get_subject_driven_data.__name__} method.")
        dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
        if 'grounding_coco' in dataset_tag:
            data_processor = self.coco_data_processor
        elif 'grounding_grit' in dataset_tag:
            data_processor = self.grit_data_processor
        else:
            raise ValueError(f"Invalid dataset_tag: {dataset_tag} for GroundingDataset")
        
        # Get source image
        # `random_crop` of both the source image and the target image MUST be False, to ensure the cropped area of the source image and the target image are the same
        src_image, src_kwargs = self.get_grounding_image_or_mask_with_size(data_processor, index, random_crop=False)

        # Get instruction and mask using grounding data processor
        origin_w, origin_h = src_kwargs['origin_size']
        grounding_result = data_processor(index, image_height=origin_h, image_width=origin_w, return_caption=self.grounding_with_caption)
        # ---- system_prompt: str, e.g. "Identify and mask the objects as specified in the prompt."
        # ---- instruction: str, e.g. "a car, a tree"
        # ---- mask: PIL.Image.Image, (H, W, 3)
        # ---- caption (optional): str, the caption of the sampled masks, e.g. "a car, a tree"
        system_prompt = grounding_result['system_prompt']
        instruction = grounding_result['instruction']
        mask = grounding_result['mask']

        if self.grounding_with_caption:
            caption = grounding_result['caption']
            system_prompt = f"{caption}\n{system_prompt}"

        max_num_srcs = 1
        # Get target images
        tgt_image, tgt_kwargs = self.get_grounding_image_or_mask_with_size(
            data_processor, index, random_crop=False, return_type=self.grounding_pred_mask_type, image=src_kwargs['image'], mask=mask
        )
        assert tgt_image.size() == src_image.size(), f"tgt_image.size={tgt_image.size()} != src_image.size={src_image.size()}"

        # Get text
        instruct_and_text = f"{system_prompt} {instruction}"

        return TextImageData(
            prompt=instruct_and_text, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
        )

    # ========================================
    #           Subject Driven
    # ========================================
    def parse_annotations(self, index, image_t, max_num_refs):
        origin_h, origin_w = image_t.shape[1], image_t.shape[2]
        # image_t means image_tensor with shape [3, H, W] and dynamic range [-1, 1]
        tags = self.index_manager.get_attribute(index, "tags")
        # [{
        #     "class_name": "xxx",
        #     "bbox": [x1, y1, x2, y2],
        #     "segmentation": {
        #         "size": [1024, 1024],
        #         "counts": "xxx"
        #     },
        #     "score": [0.98]
        # }, {...}, ...]
        annotations = json.loads(self.index_manager.get_attribute(index, "annotations"))

        valid_annotations = []
        # Find those annotations with tags. Each tag only match the first annotation.
        for tag in tags:
            for anno in annotations:
                if tag in anno["class_name"]:
                    valid_annotations.append(anno)
                    break
        # If no matched annotation, use the first annotation (already sorted by some rules).
        # See __data/instruct/subject_driven/process_data.py:sort_anno_by_hole_ratio() for details.
        if not valid_annotations:
            # Assume annotations is not empty
            valid_annotations.append(annotations[0])

        # Convert annotations raw data to masked image
        ref_objects = []
        for anno in valid_annotations[:max_num_refs]:  # No more than 3 reference objects
            # Get mask
            mask = torch.from_numpy(mask_util.decode(anno["segmentation"]))   # [h, w] the same shape with the target image
            mask = F.interpolate(mask[None, None].float(), image_t.shape[-2:], mode="nearest")[0, 0]
            if mask.max() == 0:
                if hasattr(self.index_manager, 'ind_mapper'):
                    in_json_index = self.index_manager.ind_mapper[index]
                else:
                    in_json_index = self.index_manager.indices[index]
                print(f"Warning: mask.max() == 0 in {in_json_index=}")
                continue
            # Find the tight bounding box and crop the image
            xrng = torch.where(mask.max(dim=0)[0] > 0)[0]
            yrng = torch.where(mask.max(dim=1)[0] > 0)[0]
            x1, x2 = xrng[0], xrng[-1] + 1
            y1, y2 = yrng[0], yrng[-1] + 1
            sub_image = image_t[:, y1:y2, x1:x2]
            sub_mask = mask[None, y1:y2, x1:x2]
            ref_obj = torch.clamp(sub_image + (1 - sub_mask) * 2, -1, 1)   # Add 2 to make the background white
            # Extend the sub_image as a square image
            if y2 - y1 > x2 - x1:
                x_pad = (y2 - y1) - (x2 - x1)
                y_pad = 0
            else:
                y_pad = (x2 - x1) - (y2 - y1)
                x_pad = 0
            # Extend the sub_image at least 256 pixels in each direction
            x_pad += max(384 - (x2 - x1) - x_pad, 0)
            y_pad += max(384 - (y2 - y1) - y_pad, 0)
            if x_pad or y_pad:
                pad_counts = (
                    x_pad // 2,
                    x_pad - x_pad // 2,
                    y_pad // 2,
                    y_pad - y_pad // 2,
                )
                ref_obj = F.pad(ref_obj, pad_counts, mode="constant", value=1)  # white background
            # # Resize the ref_obj to align with the multiple of 16 and no more than image_size^2 pixels
            # h, w = ref_obj.shape[-2:]
            # aligned_h = (h + 15) // 16 * 16
            # aligned_w = (w + 15) // 16 * 16
            # if aligned_h * aligned_w > self.training_image_size ** 2:
            #     aligned_h, aligned_w = self.get_target_size(h, w)
            ref_obj = F.interpolate(
                # ref_obj[None], (aligned_h, aligned_w), mode="bilinear", align_corners=False
                ref_obj[None], (origin_h, origin_w), mode="bilinear", align_corners=False
            )[0]
            ref_objects.append(ref_obj)

        # If no reference object, use a white image
        if len(ref_objects) == 0:
            ref_objects = [torch.zeros(3, origin_h, origin_w)]

        return ref_objects

    def get_subject_driven_data(self, index):
        if self.use_joint_image_feature:
            raise NotImplementedError(f"Joint image feature is not supported for {self.get_subject_driven_data.__name__} method.")
        # Get target image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.tgt_image_col)

        # Get source images
        # The src_images should be extracted before resize, but the images of this dataset are all 1024x1024, and they would not be resized.
        max_num_srcs = self.task_kwargs.get('max_num_refs', 3)
        src_images = self.parse_annotations(index, tgt_image, max_num_srcs)

        # Get text
        if self.template == "pretrain":
            prompt = self.get_simple_prompt(index)
        elif self.template == "instruct":
            instruction = self.get_instruction(subject_driven_instructions, with_space=True)
            text = self.get_simple_prompt(index)
            prompt = f"{instruction}{text}"
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=src_images, max_num_srcs=max_num_srcs, cond_type_list=["src_image"] * len(src_images),
        )

    def get_subject_driven_v2_data(self, index):
        max_num_srcs = self.task_kwargs.get('subject_driven_max_num_refs', 3)
        # Get target image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, **self.tgt_image_col)

        # Get source images
        src_counts = self.index_manager.get_attribute(index, "count")
        selected_indices = random.sample(range(src_counts), k=min(src_counts, max_num_srcs))
        src_images, und_images = [], []
        for i in selected_indices:
            src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
                index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature,
                target_size_type="image", column=f"src_img_bytes_{i}"
            )
            und_images_ = [src_vision_encoder_image] if self.use_joint_image_feature else []
            src_images.append(src_image)
            und_images.extend(und_images_)

        # Get text
        dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
        if dataset_tag == "subject_driven_xhs_single_role":
            caption_col = self.image_caption_col_2
        else:
            caption_col = self.image_caption_col
        if self.template == "pretrain":
            prompt = self.get_simple_prompt(index, **caption_col)
        elif self.template == "instruct":
            instruction = self.get_instruction(subject_driven_instructions, with_space=True)
            text = self.get_simple_prompt(index, **caption_col)
            prompt = f"{instruction}{text}"
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=src_images, max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type] * len(src_images),
            und_images=und_images,
        )

    def get_text_style_transfer_data(self, index):
        # Randomly swap content and style images
        do_swap = random.random() < 0.2

        max_num_srcs = 1
        # Get source image
        src_image, src_vision_encoder_image, src_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature and not do_swap, column="src_img_bytes_0")
        # Get a random target image ranges in [0, 5]
        rand_i = random.randint(0, 5)
        tgt_image, tgt_vision_encoder_image, tgt_image_flag = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature and do_swap, column=f"tgt_img_bytes_{rand_i}")
        
        if do_swap:
            src_image, tgt_image = tgt_image, src_image
            und_images = [tgt_vision_encoder_image] if self.use_joint_image_feature else []
        else:
            und_images = [src_vision_encoder_image] if self.use_joint_image_feature else []

        # Get text
        tag = self.index_manager.get_attribute(index, f"tgt_img_style_{rand_i}")
        if do_swap:
            tag = random.choice(realistic_style_tag)
        if self.instruction_candidates_type == "fixed_set":
            prompt = random.choice(style_transfer_instructions_zh).format(tag.rstrip('风格'))
        elif self.instruction_candidates_type == "off":
            prompt = tag
        else:
            raise ValueError(f"Unsupported instruction_candidates_type: {self.instruction_candidates_type}")

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=[src_image], max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type],
            und_images=und_images,
        )

    def get_img_style_transfer_data(self, index):
        max_num_srcs = 2
        # Get source image
        src_image_1, src_vision_encoder_image_1, image_flag_1 = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature, column="source_content_img_bytes")
        src_image_2, src_vision_encoder_image_2, image_flag_2 = self.get_image_with_size(
            index, random_crop=False, return_vision_encoder_image=self.use_joint_image_feature, column="source_style_img_bytes")
        und_image_1 = [src_vision_encoder_image_1] if self.use_joint_image_feature else []
        und_image_2 = [src_vision_encoder_image_2] if self.use_joint_image_feature else []
        
        # Randomly swap content and style images
        swap_source_content_and_style = random.random() < 0.5
        if swap_source_content_and_style:
            src_image_1, src_image_2 = src_image_2, src_image_1
            und_image_1, und_image_2 = und_image_2, und_image_1
        src_images = [src_image_1, src_image_2]
        und_images = []
        und_images.extend(und_image_1)
        und_images.extend(und_image_2)

        # Get target image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, column="target_img_bytes")

        # Get text
        prompt = random.choice(image_style_transfer_instructions)
        if swap_source_content_and_style:
            prompt = prompt.replace("<CONTENT>", "second").replace("<STYLE>", "first")
        else:
            prompt = prompt.replace("<CONTENT>", "first").replace("<STYLE>", "second")

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=src_images, max_num_srcs=max_num_srcs, cond_type_list=[self.cond_type] * len(src_images),
            und_images=und_images,
        )

    # ========================================
    #           Face ID Preserve
    # ========================================
    
    def get_tgt_src_image_sample_list(self, index, count):
        # build a list of column to be sampled. For example:
        # [tgt_img_bytes, src_img_bytes_0, src_img_bytes_1, ..., src_img_bytes_{count-1}]
        # [tgt_img_bytes, src_img_bytes]
        return_list = ["tgt_img_bytes"]
        if "src_img_bytes_0" not in self.index_manager.get_columns(index):
            return_list.append("src_img_bytes")
        else:
            for i in range(count-1):
                return_list.append(f"src_img_bytes_{i}")
        return return_list
    
    def get_tgt_src_caption_sample_list(self, index, count):
        # build a list of column to be sampled. For example:
        # [tgt_simple_caption, src_caption_v2_0, src_caption_v2_1, ..., src_caption_v2_{count-1}]
        # [tgt_caption_v2, src_caption_v2]
        if self.index_manager.get_attribute(index, "dataset_tag") == "subject_driven_xhs_single_role":
            return ["tgt_simple_caption"]
        else:
            return_list = ["tgt_caption_v2"]
            if "src_caption_v2_0" not in self.index_manager.get_columns(index):
                return_list.append("src_caption_v2")
            else:
                for i in range(count-1):
                    return_list.append(f"src_caption_v2_{i}")
            return return_list

    @staticmethod
    def get_face_analysis(logger=None):
        insightface_path = VISION_ENCODER_META_INFO["insightface"]["path"]
        name = 'buffalo_l' # From large to small, support antelopev2, buffalo_l, buffalo_sc
        allowed_modules = ['detection', "recognition"]
        if logger is not None:
            logger.info(f"Loading face analysis, name: {name}, allowed_modules: {allowed_modules} from {insightface_path}")
        face_analysis = FaceAnalysis(name=name, root=insightface_path, allowed_modules=allowed_modules, providers=['CPUExecutionProvider'])
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        return face_analysis

    @staticmethod
    def pil_image_to_numpy_int(self, image):
        """
            convert PIL image to numpy array (h, w, c), [0, 255], and convert to uint8
        """
        image_np = np.array(image)
        image_np = image_np.astype(np.uint8)
        return image_np

    @staticmethod
    def filter_face_info(self, face_info, h, w):
        """
            filter a list of face info to 
        """
        dummy_face_embedding = torch.zeros([512], dtype=torch.float32)
        if len(face_info) >= 1:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            face_bbox = face_info['bbox']
            face_embedding = face_info['embedding']
            face_embedding = torch.from_numpy(face_embedding).float()

            # float to int
            face_bbox = [int(x) for x in face_bbox]
            # clip the face bbox to the image size
            face_bbox[0] = max(0, face_bbox[0])
            face_bbox[1] = max(0, face_bbox[1])
            face_bbox[2] = min(w, face_bbox[2])
            face_bbox[3] = min(h, face_bbox[3])
            if face_bbox[2] - face_bbox[0] < 32 or face_bbox[3] - face_bbox[1] < 32:
                if os.environ.get("DEBUG"):
                    print(f"WARNING: face bbox is {face_bbox}; face area {face_bbox[2] - face_bbox[0]} x {face_bbox[3] - face_bbox[1]} is too small, skip face crop")
                return None, dummy_face_embedding
            return face_bbox, face_embedding
        else:
            if os.environ.get("DEBUG"):
                print(f"WARNING: no face detected, skip face crop")
            return None, dummy_face_embedding

    def crop_face_area(self, pil_image, face_analysis=None):
        """
            If image is PIL image, convert to numpy array first
            Get face info from face_analysis
            crop the face area from image
            Return the cropped face image as PIL image
        """
        if face_analysis is None:
            face_analysis = self.face_analysis
            
        image_np = self.pil_image_to_numpy_int(pil_image)
        face_info = face_analysis.get(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        h, w = image_np.shape[:2]
        filter_face_box, face_embedding = self.filter_face_info(face_info, h, w)
        # filter_face_box is left, top, right, bottom
        # crop the face area in pil_image using filter_face_box
        if filter_face_box is not None:
            cropped_face_pil_image = pil_image.crop(filter_face_box)
            # print(f"[crop_face_area], original shape pil_image: {pil_image.size}; cropped_face_pil_image.size: {cropped_face_pil_image.size}")
        else:
            cropped_face_pil_image = pil_image
        return cropped_face_pil_image

    def get_bbox_from_torch_tensor(self, image_tensor, face_analysis=None):
        """
            Convert torch tensor (C, H, W) [-1, 1] to numpy (H, W, C) [0, 255]
            Return the bbox of the face in the image, left, top, right, bottom; and the cropped tensor
        """
        if face_analysis is None:
            face_analysis = self.face_analysis
        image_np = image_tensor.cpu().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np + 1) * 127.5
        face_info = face_analysis.get(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        h, w = image_np.shape[:2]
        filter_face_box, face_embedding = self.filter_face_info(face_info, h, w)

        return filter_face_box, self.as_image_tensor(face_embedding, image_type="face")

    def get_face_id_preserve_data(self, index):
        # condition_type can be one of `clip`, `face_embedding`, `face_embedding_src_image`,
        # `src_image_sep_face_embedding`.
        # - clip: using siglip2 as image encoder
        # - face_embedding: using insightface as image encoder
        # - face_embedding_src_image: using both insightface and vae as image encoder
        # - src_image_sep_face_embedding: using both insightface and vae as image encoder. The difference between
        #     this and face_embedding_src_image is that `src_image_sep_face_embedding` puts the src_image in user
        #     inputs, and puts face embedding right after assistant <answer> token, and predicts the <bof> token,
        #     while `face_embedding_src_image` puts both src_image and face embedding in user inputs.
        condition_type = self.task_kwargs.get("condition_type", "clip")
        if "src_image" in condition_type or "joint_image" in condition_type:
            # Number of max src images vae latents is 1
            max_num_srcs = 1
        else:
            # For clip and face_embedding, there is no src image
            max_num_srcs = 0
        # we assume there must be a tgt_img_bytes column, and there should be one src_img_bytes column or src_img_bytes_0 column
        # We do not use same src and tgt images in this function to avoid overfitting caused by siglip feature

        arrow_columns = self.index_manager.get_columns(index)
        assert "tgt_img_bytes" in arrow_columns, "tgt_img_bytes column is required"
        assert "src_img_bytes_0" in arrow_columns or "src_img_bytes" in arrow_columns, "src_img_bytes_0 or src_img_bytes column is required"
        
        # Get image counts that includes both source and target images
        if "count" in arrow_columns:
            count = self.index_manager.get_attribute(index, "count") + 1
        else:
            count = 2
        
        # assume the order is tgt, src_0, src_1, ...
        dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")

        # Get image col names
        tgt_src_image_sample_list = self.get_tgt_src_image_sample_list(index, count)

        if dataset_tag == "subject_driven_xhs_single_role":
            # only target image is used for caption in this dataset
            tgt_i = 0
            # only face embedding allows src and tgt to be the same image
            if condition_type == "face_embedding":
                src_i = random.sample(range(0, count), 1)[0]
            else:
                src_i = random.sample(range(1, count), 1)[0]
        else:
            if condition_type == "face_embedding":
                src_i = random.sample(range(0, count), 1)[0]
                tgt_i = random.sample(range(0, count), 1)[0]
            else:
                # Random sample two indices from range(count) as source and target images
                if count >= 2:
                    # In dataset id_preserve_photomaker, each ID may have multiple images, and we sample from the filtered columns if face_photomaker_filtered_columns_col is provided
                    if dataset_tag == "id_preserve_photomaker" and self.index_kwargs.get("face_photomaker_filtered_columns_col", None) is not None:
                        # when we choose photomaker dataset, get {src_i, tgt_i} from filtered columns
                        face_photomaker_filtered_columns_col = self.index_manager.get_attribute(index, **self.face_photomaker_filtered_columns_col)
                        sampled_columns = random.sample(face_photomaker_filtered_columns_col, 1)[0] # example: [src_img_bytes_0, tgt_img_bytes]
                        # inverse sampled_columns with 50% probability
                        if random.random() < 0.5:
                            sampled_columns = sampled_columns[::-1]
                        # map sampled_columns to src_i, tgt_i in tgt_src_image_sample_list
                        src_column = sampled_columns[0]
                        tgt_column = sampled_columns[1]
                        src_i = tgt_src_image_sample_list.index(src_column)
                        tgt_i = tgt_src_image_sample_list.index(tgt_column)
                    else:
                        src_i, tgt_i = random.sample(range(count), 2)
                else: # count == 1 is not used, may be used in the future
                    src_i, tgt_i = 0, 0

        image_src_i_name = tgt_src_image_sample_list[src_i]
        image_tgt_i_name = tgt_src_image_sample_list[tgt_i]

        def get_image(return_vision_encoder_image=False):
            vae_image, vision_encoder_image, image_flag = self.get_image_with_size(
                index, random_crop=False, return_vision_encoder_image=return_vision_encoder_image,
                column=image_src_i_name,
            )
            if vision_encoder_image is not None:
                return vae_image, vision_encoder_image
            return vae_image

        def get_face_embedding():
            return self.get_bbox_from_torch_tensor(src_images[0])[1]

        # Get images
        src_images = []
        face_embeddings = []
        und_images = []
        if condition_type == "clip":
            und_images.append(get_image(return_vision_encoder_image=True)[1])
            cond_type_list = ["und_image"]
        elif condition_type == "face_embedding":
            face_embeddings.append(get_face_embedding())
            cond_type_list = ["face"]
        elif condition_type == "face_embedding_src_image":
            face_embeddings.append(get_face_embedding())
            src_images.append(get_image())
            cond_type_list = ["face", "src_image"]
        elif condition_type == "src_image_sep_face_embedding":
            src_images.append(get_image())
            face_embeddings.append(get_face_embedding())
            cond_type_list = ["src_image", "face"]
        elif condition_type == "joint_image":
            src_image, und_image = get_image(return_vision_encoder_image=True)
            src_images.append(src_image)
            und_images.append(und_image)
            cond_type_list = ["joint_image"]
        else:
            raise ValueError(f"Unsupported condition_type: {condition_type}")

        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, column=image_tgt_i_name)

        if "local_loss_weight" in self.task_kwargs:
            image_bbox_mask = torch.zeros(1, tgt_image.i.h, tgt_image.i.w)
            tgt_image_bbox, tgt_image_face_embedding = self.get_bbox_from_torch_tensor(tgt_image)
            # tgt_image_bbox is left, top, right, bottom
            if tgt_image_bbox is not None:
                image_bbox_mask[:, tgt_image_bbox[1]:tgt_image_bbox[3], tgt_image_bbox[0]:tgt_image_bbox[2]] = 1.0
            token_bbox_mask = F.interpolate(
                image_bbox_mask[None, ...],
                size=(tgt_image.i.tk_h, tgt_image.i.tk_w), mode='bilinear', align_corners=False
            )[0]
        else:
            token_bbox_mask = None

        # Get prompt column names
        tgt_src_caption_sample_list = self.get_tgt_src_caption_sample_list(index, count)
        if self.instruction_candidates_type == "merged":
            caption_col = dict(
                column=f"{tgt_src_caption_sample_list[tgt_i]}_instruction_caption_merge",
                shadow="merged_instruction",    # self.merged_instruction
            )
        else:
            caption_col = dict(column=tgt_src_caption_sample_list[tgt_i])

        # Get prompt
        if dataset_tag == "subject_driven_xhs_single_role":
            text = self.get_simple_prompt(index, **caption_col)
        elif dataset_tag == "id_preserve_photomaker":
            text = self.get_caption_prompt(index, **caption_col, caption_aug=self.caption_aug_photomaker)
        else:
            text = self.get_caption_prompt(index, **caption_col)
        if self.template == "pretrain":
            prompt = text
        elif self.template == "instruct":
            instruction = self.get_instruction(face_id_instructions, with_space=True)
            prompt = f"{instruction}{text}"
        else:
            raise ValueError(f"Unsupported template: {self.template}")
        # Replace chinese ‘’ to english '' 
        prompt = prompt.replace("‘", "'").replace("’", "'")

        return TextImageData(
            prompt=prompt, tgt_image=tgt_image,
            src_images=src_images, max_num_srcs=max_num_srcs, cond_type_list=cond_type_list,
            und_images=und_images,
            face_embeddings=face_embeddings, token_bbox_mask=token_bbox_mask,
        )

    def __getitem__(self, index):
        # <-------- start of Gemini Beta data loaders ---------->
        if self.dataset_tag == "editing":
            # ================= Universal editing task (one src, one target) =================
            dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
            if dataset_tag.startswith("editing_caimai"):
                getter = self.get_editing_extra_data
            elif dataset_tag.startswith("editing"):
                getter = self.get_editing_data
            elif dataset_tag.startswith("grounding"):
                getter = self.get_grounding_data
            elif dataset_tag.startswith("inpainting"):
                getter = self.get_inpainting_v2_data
            elif dataset_tag.startswith("controlnet"):
                getter = self.get_controlnet_data
            elif dataset_tag.startswith("style_transfer"):
                getter = self.get_text_style_transfer_data
            else:
                raise ValueError(f"Unsupported dataset tag: {dataset_tag}. {index=}")
        elif self.dataset_tag == "editcot":
            dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
            if dataset_tag.startswith("editing"):
                getter = self.get_cot_editing_data
            else:
                raise ValueError(f"Unsupported dataset tag: {dataset_tag}. {index=}")
        elif self.dataset_tag == "subject_driven":
            # ================= Universal subject driven task (multiple ragged srcs, one target) =================
            columns = self.index_manager.get_columns(index)
            if "dataset_tag" in columns:
                dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
                if dataset_tag.startswith("subject_driven"):
                    getter = self.get_subject_driven_v2_data
                elif dataset_tag.startswith("img_style_transfer"):
                    getter = self.get_img_style_transfer_data
                else:
                    raise ValueError(f"Unsupported dataset tag: {dataset_tag}. {index=}")
            else:
                # Fallback to old subject driven data loader
                getter = self.get_subject_driven_data
        elif "face_id" in self.dataset_tag:
            getter = self.get_face_id_preserve_data
            # <-------- end of Gemini Beta data loaders ---------->

            # ===================== Old tasks =====================
        elif "inpainting" in self.dataset_tag:
            getter = self.get_inpainting_data
        elif "editing" in self.dataset_tag:
            getter = self.get_editing_data
        elif "grounding" in self.dataset_tag:
            getter = self.get_grounding_data
        else:
            raise ValueError(f"Unsupported dataset tag: {self.dataset_tag}")

        data = getter(index)

        # Deal with single/multiple unconditions
        if self.uncond_p_src > 0:
            do_uncond, do_uncond_src = self.get_uncond_flags([self.uncond_p, self.uncond_p_src], strategy=self.multi_uncond_strategy)
            do_uncond_face = False
        elif len(data.face_embeddings) > 0:
            do_uncond, do_uncond_face = self.get_uncond_flags([self.uncond_p, self.face_uncond_p], strategy=self.multi_uncond_strategy)
            do_uncond_src = False
        else:
            do_uncond, = self.get_uncond_flags([self.uncond_p], strategy=self.multi_uncond_strategy)
            do_uncond_src = False
            do_uncond_face = False

        # src uncond
        if do_uncond_src:
            data.src_images = [self.as_image_tensor(torch.zeros_like(src_image), image_type="vae") for src_image in data.src_images]
            if self.use_joint_image_feature:
                data.und_images = [self.as_image_tensor(torch.zeros_like(und_image), image_type="vision_encoder", **und_image.vision_encoder_kwargs) for und_image in data.und_images]
                data.joint_images = [JointImage(a, b) for a, b in zip(data.src_images, data.und_images)]
        # face uncond
        if do_uncond_face:
            data.face_embeddings = [self.as_image_tensor(torch.zeros_like(face), image_type="face") for face in data.face_embeddings]
        # text uncond
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))

        # We want the image prefix special tokens <boi>, <img_size_*>, and <img_ratio_*> to be learned.
        # It is implemented by adding text mask end offsets to the prefix text sections.
        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0)
        # 1 for <bos>. <eos> is not included because it will be stripped in the shift of next-token-prediction
        extra_num_tokens = 1 + data.num_special_tokens + self.dummy_number

        if self.template == "pretrain":
            prompt_section = [
                dict(type="text", text=data.prompt, max_length=self.text_token_length - extra_num_tokens,
                     **uncond_kwargs, ignore=do_uncond),
            ]
            gen_section = [
                dict(type='text', text='', ignore=do_uncond, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text='', end_offset=1)  # include <eos> token
            ]
            # Support multi condition type in cond_section here
            cond_section = data.get_cond_sections()
            sep_face_cond = ("face_id" in self.dataset_tag and
                             self.task_kwargs.get("condition_type", "clip") == "src_image_sep_face_embedding")
            if sep_face_cond:
                raise NotImplementedError("src_image_sep_face_embedding is not yet implemented in pretrain")

            # use_front_src_image controls whether to put the source image in the front of the prompt
            # or not. If True, the sequence will be
            #     <src_boi>[image]<src_eoi>xxxx<boi>[image]<eoi>
            # which is a natural order. If False, the sequence will be
            #     xxxx<src_boi>[image]<src_eoi><boi>[image]<eoi>
            # which is a reverse order. The latter is used for the case that image generation tasks and
            # image understanding tasks use different image encoders (e.g., vae and siglip2). In this case,
            # given an image, we must determine which image encoder to used. Putting the src_image after the
            # prompt, we can ask the model to predict either <src_boi> or <und_boi> to determine the image
            # encoder.
            if self.use_front_src_image:
                # <src_boi>[image]<src_eoi>xxxx<boi>[image]<eoi>
                sections = cond_section + prompt_section + gen_section
            else:
                raise NotImplementedError("Post src image is not supported anymore.")
            max_token_length = self.text_token_length + data.get_image_length(self) + 1 - self.dummy_number

        elif self.template == "instruct":
            extra_num_tokens += 9   # "User: " + "\n\n" + "Assistant: <answer>" + "</answer>"
            user_prefix_section = [
                dict(type="text", text=f"{self.roles[0]}: ", ignore=True),
            ]
            prompt_and_bot_prefix_section = [
                dict(type="text", text=data.prompt, max_length=self.text_token_length - extra_num_tokens,
                     **uncond_kwargs, ignore=True),
                dict(type="text", text=self.default_conv.sep, ignore=True),
                dict(type="text", text=f"{self.roles[1]}: <answer>", ignore=True),
            ]
            gen_section = [
                dict(type="text", text='', ignore=do_uncond, end_offset=num_image_prefix),
                dict(type="gen_image", **data.tgt_image.i.meta_info),
                dict(type="text", text="</answer>", ignore=False, end_offset=1),
            ]

            cond_section = data.get_cond_sections()
            sep_face_cond = ("face_id" in self.dataset_tag and
                             self.task_kwargs.get("condition_type", "clip") == "src_image_sep_face_embedding")
            if sep_face_cond:
                # In this situation, one condition in user section, one condition in assistant section.
                # TODO: -1 is hard code. Replace with a parameter returned from get_face_id_preserve_data
                user_cond_section, bot_cond_section = cond_section[:-1], cond_section[-1:]
                # predict <bof> token
                bot_cond_section = [
                                       dict(type="text", text='', ignore=False, end_offset=1),
                                   ] + bot_cond_section
            else:
                user_cond_section = cond_section
                bot_cond_section = []

            if data.recaption:
                cot_section = [
                    dict(type="text", text="<recaption>", ignore=True),
                    dict(type="text", text=data.recaption, ignore=do_uncond,
                         max_length=self.description_token_length - 2, **uncond_kwargs),
                    dict(type="text", text="</recaption>", ignore=do_uncond),
                ]
                max_token_length = (self.text_token_length + data.get_image_length(self) + 1 - self.dummy_number +
                                    self.description_token_length)
            else:
                cot_section = []
                max_token_length = self.text_token_length + data.get_image_length(self) + 1 - self.dummy_number

            # See the docstring in the pretrain branch.
            if self.use_front_src_image:
                # User: <src_boi>{image}<src_eoi>xxxx\n\nAssistant: <answer><boi>{image}<eoi></answer>
                # ^^^^^^-------------------------^^^^^^^^^^^^^^^^^^^^^^^^^^^--------------------------
                sections = (user_prefix_section + user_cond_section + prompt_and_bot_prefix_section +
                            cot_section + bot_cond_section + gen_section)
            else:
                # User: xxxx\n\nAssistant: <answer><src_boi>{image}<src_eoi><boi>{image}<eoi></answer>
                # ^^^^^^---------------------------^^^^^^^^^^^^^^^^^^^^^^^^^--------------------------
                sections = user_prefix_section + prompt_and_bot_prefix_section + [
                    # predict the first src_boi
                    dict(type="text", text='', ignore=do_uncond, end_offset=1),
                ] + cond_section + gen_section

            if self.dataset_tag == "editcot":
                max_token_length += self.description_token_length

        else:
            raise ValueError(f"Unsupported template tag: {self.template}")

        try:
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
            )
        except AssertionError as e:
            self.logger.error(f"Error in encoding sections (index={index}): {max_token_length=}, {sections}")
            raise e

        target_tokens = output.tokens.clone()
        target_tokens[output.text_mask == 0.0] = -100

        # Attention mask
        if self.task_kwargs.get('attn_type', 'auto') == 'auto':
            n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + self.dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            #   assume src_image_slices and und_image_slices are not exist at the same time
            for image_slice in output.joint_image_slices + output.src_image_slices + output.und_image_slices + output.face_image_slices + output.gen_image_slices:
                attention_mask[image_slice, image_slice] = True
            attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = None

        # 2d rope
        rope_image_info = self.get_rope_image_info(sections, output)

        # condition image
        src_images = data.src_images
        if len(src_images) == 1:
            src_images = src_images[0]
        elif len(src_images) == 0:
            src_images = None
        und_images = data.und_images
        if len(und_images) == 1:
            # c x h x w or seq_len x dim
            und_images = und_images[0]
            # attention_mask: seq_len, spatial_shapes: 2
            vision_encoder_kwargs = und_images.vision_encoder_kwargs
            if vision_encoder_kwargs == {}:
                vision_encoder_kwargs = None
        elif len(und_images) == 0:
            und_images = None
            vision_encoder_kwargs = None
        else:
            vision_encoder_kwargs = defaultdict(list)
            for und_image in und_images:
                for k, v in und_image.vision_encoder_kwargs.items():
                    vision_encoder_kwargs[k].append(v)
            # attention_mask: n x seq_len, spatial_shapes n x 2
            vision_encoder_kwargs = {k: torch.stack(v) for k, v in vision_encoder_kwargs.items()}
            if vision_encoder_kwargs == {}:
                vision_encoder_kwargs = None
            # n x c x h x w or n x seq_len x dim
            # stack und_imagas after vision_encoder_kwargs to avoid lost of vision_encoder_kwargs attr
            und_images = torch.stack(und_images)

        ret = {
            "data_type": "ti2i",                    # data_type determines the loss type.
            "dtype": self.dataset_tag,              # The prefix determines the prepare fn.
            "n_samples": 1,
            "src_image": src_images,
            "image": data.tgt_image,
            "tokens": output.tokens,
            "target_tokens": target_tokens,
            "text_mask": output.text_mask,
            "src_image_mask": output.src_image_mask,
            "image_mask": output.gen_image_mask,
        }
        if und_images is not None:
            ret["und_images"] = und_images
            if vision_encoder_kwargs is not None:
                ret["vision_encoder_kwargs"] = vision_encoder_kwargs
        if attention_mask is not None:
            ret["attention_mask"] = attention_mask
        else:
            ret["face_image_slices"] = output.face_image_slices
            ret["src_image_slices"] = output.src_image_slices
            ret["und_image_slices"] = output.und_image_slices
            ret["joint_image_slices"] = output.joint_image_slices
            ret["gen_image_slices"] = output.gen_image_slices
        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,  # (2)
                "iw_ih_scatter_src": data.iw_ih_scatter_src,             # (2)
            })
        if output.timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": output.timestep_scatter_index,   # (1)
            })
        if data.token_bbox_mask is not None:
            ret["token_bbox_mask"] = data.token_bbox_mask
        if len(data.face_embeddings) > 0:
            ret["src_face_embedding"] = data.face_embeddings[0]
        if output.und_image_mask is not None:
            ret["und_image_mask"] = output.und_image_mask
        if output.face_image_mask is not None:
            ret["face_image_mask"] = output.face_image_mask
        if rope_image_info is not None:
            ret["rope_image_info"] = rope_image_info
        return ret

# deprecated for face_id_clip
class FaceIDArrowStream(TextImageToImageArrowStream):
    def __post_init__(self, **kwargs):
        self.face_uncond_p = self.task_kwargs['face_uncond_p']
        self.face_index_unique = self.task_kwargs['face_index_unique']
        self.face_token_length = self.task_kwargs['resampler_token_length']
        self.src_condition_type = self.task_kwargs['src_condition_type']
        # After init, we set the logger to all_logger to print warnings and errors of all ranks
        self.logger = all_logger

    def get_id_face_analysis(self, ind):
        try:
            bbox = self.index_manager.get_attribute(ind, **self.face_bbox_col)
            embedding = self.index_manager.get_attribute(ind, **self.face_embed_col)
            bbox_np = np.frombuffer(bbox, dtype=np.float32)
            embedding_np = np.frombuffer(embedding, dtype=np.float32)
            embedding_np = np.copy(embedding_np)
            bbox_np = bbox_np.reshape(-1, 4)
            embedding_np = embedding_np.reshape(-1, 512)
            assert bbox_np.shape[0] == embedding_np.shape[0], \
                f"bbox and embedding shape mismatch: {bbox_np.shape[0]} != {embedding_np.shape[0]}"
        except Exception as e:
            self.handle_exception_message(self.get_id_face_analysis, e, line_no=sys._getframe().f_lineno)
            bbox_np = np.zeros((2, 4), dtype=np.float32)
            embedding_np = np.zeros((2, 512), dtype=np.float32)

        return bbox_np, embedding_np

    def get_valid_bbox_index(self, index, bbox):
        """
        bbox: [num_bbox, 4], check width_height_positive and left_top_right_bottom_in_image, return the index of the valid bbox
        each row in bbox is left, top, right, bottom
        bbox == 0,0,0,0 means no face or multiple faces detected
        Args:
            bbox (_type_): [num_bbox, 4]
        """
        count = bbox.shape[0]
        assert count == self.index_manager.get_attribute(index, "count"), \
            (f'bbox.shape[0] should be equal with count index_manager, '
             f'but  {count}!={self.index_manager.get_attribute(index, f"count")}')

        image_width_np_array = np.array(
            [self.index_manager.get_attribute(index, f"width_{image_count_i}") for image_count_i in range(count)])
        image_height_np_array = np.array(
            [self.index_manager.get_attribute(index, f"height_{image_count_i}") for image_count_i in range(count)])
        bbox_left_top_right_bottom_in_image_index = (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (
                    bbox[:, 2] <= image_width_np_array) & (bbox[:, 3] <= image_height_np_array)

        # Check if there are any invalid bboxes and clip them
        if not bbox_left_top_right_bottom_in_image_index.all():
            invalid_indices = np.where(~bbox_left_top_right_bottom_in_image_index)[0]
            bbox_clip = bbox.copy()
            # Clip the bbox values
            bbox_clip[:, 0] = np.clip(bbox[:, 0], 0, image_width_np_array)  # left
            bbox_clip[:, 1] = np.clip(bbox[:, 1], 0, image_height_np_array)  # top
            bbox_clip[:, 2] = np.clip(bbox[:, 2], 0, image_width_np_array)  # right
            bbox_clip[:, 3] = np.clip(bbox[:, 3], 0, image_height_np_array)  # bottom

            self.logger.warning(f"Found invalid bboxes at indices {invalid_indices}. \
                                Original bbox values: {bbox[invalid_indices]}; \
                                Clipped bbox values: {bbox_clip[invalid_indices]}")
            bbox = bbox_clip

        bbox_width_height_positive_index = (bbox[:, 0] < bbox[:, 2]) & (bbox[:, 1] < bbox[:, 3])
        valid_bbox_index = np.where(bbox_width_height_positive_index)[0]
        # convert valid_bbox_index to list
        valid_bbox_index = valid_bbox_index.tolist()
        # if no valid bbox, loggger.error and return None
        if len(valid_bbox_index) == 0:
            self.logger.error(f"No valid bbox found in {count} images of index {index}, clipped bbox: {bbox}")
        return valid_bbox_index, bbox

    def get_face_id_data(self, index):
        assert self.src_condition_type == "face_embed", \
            f"Only `face_embed` src_condition_type is implemented, but got {self.src_condition_type}"
        # Parse bboxes and select two reasonable ones randomly
        bbox, embedding = self.get_id_face_analysis(index)
        valid_bbox_index, bbox = self.get_valid_bbox_index(index, bbox)
        # (TODO) filter invalid id using index_manager filter, not in dataloader
        valid_bbox_index_sample = valid_bbox_index if len(valid_bbox_index) >= 2 else range(bbox.shape[0])
        if self.face_index_unique:
            # Choose 2 different images from count using random.sample
            src_index, tgt_index = random.sample(valid_bbox_index_sample, 2)
        else:
            src_index, tgt_index = random.choices(valid_bbox_index_sample, k=2)

        # Get source image
        face_embedding = torch.from_numpy(embedding[src_index])
        if random.random() < self.face_uncond_p:
            face_embedding = torch.zeros_like(face_embedding)
        face_embedding = self.as_image_tensor(face_embedding, image_type="face")
        # Get target image
        tgt_image, _, image_flag = self.get_image_with_size(index, random_crop=False, column=f"image_{tgt_index}")

        # Get text
        instruction = self.get_instruction(index, with_space=True)
        text = self.get_caption_prompt(index, column=f"caption_{tgt_index}")
        instruct_and_text = f"{instruction}{text}"

        return TextImageData(
            prompt=instruct_and_text, tgt_image=tgt_image,
            cond_type_list=["face"],
            face_embeddings=[face_embedding],
        )

    def __getitem__(self, index):
        data = self.get_face_id_data(index)
        assert len(data.face_embeddings) == 1, "Only one face is supported for now."

        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))

        # User: xxx\n\nAssistant: <answer><bof>{face}<eof><boi>{image}<eoi></answer>
        user_prefix_section = [
            dict(type="text", text=f"{self.roles[0]}: ", ignore=True),                                  # 1+3 tokens
        ]
        prompt_and_box_prefix_section = [
            dict(type="text", text=data.prompt, max_length=self.text_token_length,
                 **uncond_kwargs, ignore=True),
            dict(type="text", text=self.default_conv.sep, ignore=True),                                 # 1 token
            dict(type="text", text=f"{self.roles[1]}: <answer>", ignore=True),                          # 4 tokens
        ]
        gen_section = [
            dict(type="text", text='', ignore=do_uncond, end_offset=1),
            dict(type="gen_image", **data.tgt_image.i.meta_info),                                       # 5 tokens
            dict(type="text", text="</answer>", ignore=False, end_offset=1),                            # 1+1 tokens
        ]
        cond_section = data.get_cond_sections()

        sections = user_prefix_section + prompt_and_box_prefix_section + cond_section + gen_section
        max_token_length = self.text_token_length + data.get_image_length(self) + 1 - self.dummy_number

        try:
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
            )
        except AssertionError as e:
            self.logger.error(f"Error in encoding sections (index={index}): {max_token_length=}, {sections}")
            raise e

        target_tokens = output.tokens.clone()
        target_tokens[output.text_mask == 0.0] = -100

        # Attention mask
        n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + self.dummy_number
        attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        for image_slice in output.face_image_slices + output.gen_image_slices:
            attention_mask[image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(0)

        ret = {
            "data_type": "faceid",          # data_type determines the loss type.
            "dtype": self.dataset_tag,      # The prefix determines the prepare fn.
            "n_samples": 1,
            "src_image": None,
            "image": data.tgt_image,
            "tokens": output.tokens,
            "target_tokens": target_tokens,
            "text_mask": output.text_mask,
            "src_image_mask": output.face_image_mask,
            "image_mask": output.gen_image_mask,
            "attention_mask": attention_mask,
            "src_face_embedding": data.face_embeddings[0],
        }
        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,  # (2)
                "iw_ih_scatter_src": data.iw_ih_scatter_src,             # (2)
            })
        if output.timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": output.timestep_scatter_index,   # (1)
            })

        return ret


@dataclass
class InterleaveData:
    messages: List[Dict]
    images: List[JointImage]
    cond_images: List[JointImage]
    image_flag: str
    index: int

    def unbind(self):
        return self.messages, self.images, self.cond_images, self.image_flag, self.index


class TextImageInterleaveArrowStream(TextImageToImageArrowStream):
    """ 新的数据格式支持四种类型:
    [
        {
            'type': 'cond_text' | 'gen_text' | 'gen_image' | 'cond_image',
            'cache_image': None or local path,
            'url_cos': None or url,
            'caption_v3_zh': None or caption_v3_zh (JSON string),
            'caption_v3_en': None or caption_v3_en (JSON string),
            'text_zh': text_zh (instruction_zh),
            'text_en': text_en (instruction_en),
            'caption_type': 'caption_v3' | 'instruction' | 'video_caption_v1' | etc.,
        },
        ...
    ]
    """
    def __post_init__(self, **kwargs):
        # self.resolutions = ResolutionGroup(base_size=self.base_size[0], step=self.base_size[0] // 16)
        self.interleave_max_length = self.task_kwargs["max_length"] + 1 - self.dummy_number
        self.drop_last = self.task_kwargs["drop_last"]

        # 强制将第一张图设置为gen_image
        self.gen_first_img = self.task_kwargs.get("gen_first_img", False)
        # 强制将第一张图设置为cond_image
        self.cond_first_img = self.task_kwargs.get("cond_first_img", False)

        self.add_img_ratio = self.task_kwargs.get("add_img_ratio", 0.0)
            
        # 文本和图像caption选择比例配置
        # {"zh": 0.5, "en": 0.5}; 默认只采样英文
        self.sample_lang = json.loads(self.task_kwargs.get("sample_lang", '{"en": 1.0}'))

        # {"text": 0.3, "caption": 0.3, "think": 0.3}
        sample_instruction_ratio = json.loads(self.task_kwargs.get("sample_instruction_ratio", '{"text": 1.0}'))
        self.text_ratio = sample_instruction_ratio.get('text', 1.0)
        self.cap_ratio = sample_instruction_ratio.get('caption', 0.0)
        self.think_ratio = sample_instruction_ratio.get('think', 0.0)
        self.reversed_ratio = sample_instruction_ratio.get('reversed_text', 0.0)
        self.extra_ratio = sample_instruction_ratio.get('extra_info', 0.0)
        self.i2i_ratio = sample_instruction_ratio.get('i2i_caption', 0.0)
        self.cot_ratio = sample_instruction_ratio.get('cot_recaption', 0.0)

        # '{"long caption":0.4,"short caption":0.25,"background":0.08,"shot_type":0.08,"style":0.08,"light":0.08,"atmosphere":0.08,"camera movement":0.08}'
        default_sample_img_cap_ratio = '{"long_long_caption":0.3,"long_caption":0.35,"medium_caption":0.25,"short_caption":0.1,"background":0.1,"style":0.7}'
        self.sample_img_cap_ratio = json.loads(self.task_kwargs.get("sample_img_cap_ratio", default_sample_img_cap_ratio))

        # 0: thinK_0, 1: thinK_1
        self.think_type = self.task_kwargs.get("think_type", None)

        # pure_text, pure_caption, think_caption, mixup
        self.inter_type = self.task_kwargs.get("inter_type", "pure_text")

        if "@" in self.inter_type:
            self.inter_type, self.data_key = self.inter_type.split("@")
        else:
            self.data_key = None

        # extra_info_sample: {"instruction_en": 0.5, "instruction_zh": 0.5}
        default_extra_info_sample = '{"instruction_en": 0.5, "instruction_zh": 0.5}'
        self.extra_info_sample = json.loads(self.task_kwargs.get("extra_info_sample", default_extra_info_sample))
                
        # 图像caption处理器
        if self.task_kwargs.get('caption_processor') is not None:
            self.img_caption_aug = load_caption_processor(
                name=self.task_kwargs.get('caption_processor'),
                caption_sample_ratio=self.sample_img_cap_ratio,
                logger=self.logger,
                kwargs=self.task_kwargs.get('caption_processor_kwargs'),
            )
        self.logger.info(f"Caption sample ratio (Img Caption): {self.sample_img_cap_ratio}")
        self.logger.info(f"Text sample ratio (Cap/Text/Tink): {sample_instruction_ratio}")

        self.cond_resize_degradation_prob = self.task_kwargs.get("cond_resize_degradation_prob", 0.0)

        # After init, we set the logger to all_logger to print warnings and errors of all ranks
        self.logger = all_logger
    
    def parse_columns_and_register_shadow(self, index_kwargs):
        """ Parse data columns for different tasks and register to self.shadow_file_fn """
        self.message_col = IndexColumn(index_kwargs.get("message_col"), self, self.logger)
        self.register_attributes(index_kwargs)

    def get_think_template(self, key, lang="en"):
        """
        think 模版， 根据think_type选择
        think_type: 0: thinK_0, 1: thinK_1
        """
        self.think_template_zh = {
            "thinK_0": '''<think>用户现在想生成一张图片，根据用户的指令，我先生成图片的描述：{}。现在根据描述生成图片</think>''',
            "think_1": '''<think>用户提供了{}张图片，图片内容如下: {}。然后用户的指令是: {}</think>''',
        }

        self.think_template_en = {
            "thinK_0": '''<think>The user now wants to generate a picture. According to the user's instructions, I first generate a description of the picture: {}. Now generate the picture according to the description </think>''',
            "think_1": '''<think>The user provided {} image that describes: {}. Then the user's instruction is: {} </think>''',
        }

        return self.think_template_zh[key] if lang == "zh" else self.think_template_en[key]
    
    def find_think_type1_index_in_messages(self, messages, selected_language):
        """
        seq: 输入的序列，比如 ['cond_img', 'cond_img', 'cond_text', 'gen_image', ...]
        返回: {index: (type, n_cond, new_caption, text, gen_img_caption)}, selected_language
        """
        results = {}
        i = 0
        n = len(messages)
        while i < n-1:
            # 找到cond_text
            if messages[i]['type'] == 'cond_text' and "gen" in messages[i+1]['type']:
                text, actual_language = self.get_text_from_text_message(messages[i], selected_language)
                selected_language = actual_language
                # 向前找连续的cond_img
                j = i - 1
                cond_captions = []
                while j >= 0 and messages[j]['type'] == 'cond_image':
                    # 获取cond_img的caption
                    caption, actual_language = self.get_caption_from_image_message(messages[j], preferred_lang=selected_language)
                    cond_captions.insert(0, caption)  # 保持顺序
                    j -= 1
                
                if messages[i+1]['type'] == 'gen_image':
                    # 获取生成图像的caption
                    gen_img_caption, _ = self.get_caption_from_image_message(messages[i+1], preferred_lang=selected_language)
                else:
                    gen_img_caption = None

                think_caption = ""
                for idx, cap in enumerate(cond_captions):
                    if len(cond_captions) > 1:
                        prefix = f"第{idx+1}张图片描述了：" if selected_language == "zh" else f"The contains of image {idx+1}: "
                    else:
                        prefix = f"该图片描述了：" if selected_language == "zh" else f"This image describes: "
                    think_caption = think_caption + f"{prefix}\n{cap}\n"
                if cond_captions:  # 至少有一个cond_img
                    results[i] = (messages[i+1]['type'], len(cond_captions), think_caption, text, gen_img_caption)
            i += 1
        return results, selected_language

    def find_think_type0_index_in_messages(self, messages, selected_language):
        """
        seq: 输入的序列，比如 ...., 'cond_text', 'gen_image', ...]
        返回: {index: (type, None, None, text, gen_img_caption)}, selected_language
        """
        results = {}
        i = 0
        n = len(messages)
        while i < n-1:
            # 找到cond_text
            if messages[i]['type'] == 'cond_text' and "gen_image" in messages[i+1]['type']:
                text, actual_language = self.get_text_from_text_message(messages[i], selected_language)
                selected_language = actual_language
                gen_img_caption, _ = self.get_caption_from_image_message(messages[i+1], preferred_lang=selected_language)
                results[i] = (messages[i+1]['type'], None, None, text, gen_img_caption)
            i += 1
        return results, selected_language

    @staticmethod
    def get_text_from_text_message(message, selected_language, next_message=None):
        """根据预选的语言获取对应的文本，如果缺失则使用其他语言并返回实际使用的语言
        
        Args:
            message: 当前消息
            selected_language: 预选的语言 ('zh' 或 'en')
            next_message: 下一个消息（此参数保留但不使用）
            
        Returns:
            tuple: (text, actual_language) 文本内容和实际使用的语言
        """
        _ = next_message
        # 定义文本键和对应的语言映射
        text_mapping = {
            'text_en': 'en',
            'text_zh': 'zh'
        }
        
        # 优先选择指定语言的文本
        preferred_key = f'text_{selected_language}'
        if message.get(preferred_key):
            return message[preferred_key], selected_language
        
        # 如果没有指定语言的文本，尝试选择其他语言
        alternative_language = 'zh' if selected_language == 'en' else 'en'
        alternative_key = f'text_{alternative_language}'
        if message.get(alternative_key):
            return message[alternative_key], alternative_language
        
        # 如果都没有，收集所有可用文本
        available_texts = [
            (key, message[key]) 
            for key in text_mapping.keys() 
            if message.get(key)
        ]
        
        # 边界情况：无可用文本
        if not available_texts:
            return "", selected_language
        
        # 返回任意一个可用文本及其对应的语言
        selected_key, selected_text = available_texts[0]
        actual_language = text_mapping[selected_key]
        return selected_text, actual_language

    def get_caption_from_image_message(self, message, preferred_lang=None):
        """从gen_image类型的消息中获取caption
        
        Args:
            message: 消息内容
            preferred_lang: 首选语言 ('zh' 或 'en')，如果为None则随机选择
            
        Returns:
            str: 获取的caption文本
        """
        available_captions = []
        
        # 添加caption_v3_en和caption_v3_zh
        for caption_key in ['caption_v3_en', 'caption_v3_zh']:
            caption_content = message.get(caption_key)
            if caption_content:
                try:
                    if isinstance(caption_content, str):
                        # 尝试解析JSON
                        caption_data = json.loads(caption_content)
                        # 使用合适的caption长度
                        lang = caption_key.split('_')[-1]  # 'en' or 'zh'
                        text = self.img_caption_aug.caption_aug(caption_data, lang)
                        available_captions.append((caption_key, text, lang))
                    else:
                        lang = caption_key.split('_')[-1]
                        available_captions.append((caption_key, str(caption_content), lang))
                except (json.JSONDecodeError, TypeError):
                    # 如果不是JSON，直接使用原文
                    lang = caption_key.split('_')[-1]
                    available_captions.append((caption_key, str(caption_content), lang))
        
        if not available_captions:
            return "", None
        
        # 如果指定了首选语言，优先选择该语言的caption
        if preferred_lang:
            preferred_captions = [cap for cap in available_captions if cap[2] == preferred_lang]
            if preferred_captions:
                return preferred_captions[0][1], preferred_lang
        
        # 如果没有指定语言或没有对应语言的caption，随机选择
        selected_caption = random.choice(available_captions)
        lang_cls = count_zh_en_words(selected_caption[1])
        lang = "zh" if lang_cls['zh_ratio'] > lang_cls['en_ratio'] else "en"

        return selected_caption[1], lang
    
    def get_inpainting_data(self, index, messages, selected_language):     # noqa
        """
        正常messages序列，保持原始数据不变
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if "text" in msg['type']:
                text, actual_language = self.get_caption_from_image_message(msg, selected_language)
                selected_language = actual_language
                record_msgs.append(dict(type=msg['type'], text=text))
            # 处理图片消息
            elif "image" in msg['type']:
                record_msgs.append(messages[i])
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_inpainting_data> : {msg['type']}")
        
        return record_msgs

    def get_pure_text_data(self, index, messages, selected_language):
        """
        正常messages序列，保持原始数据不变
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if "text" in msg['type']:
                text, actual_language = self.get_text_from_text_message(msg, selected_language)
                selected_language = actual_language
                record_msgs.append(dict(type=msg['type'], text=text))
            # 处理生成图片消息
            elif "image" in msg['type']:
                # 处理第一张图片的特殊情况
                if i == 0 and msg['type'] == "gen_image":
                    text, actual_language = self.get_caption_from_image_message(messages[i], preferred_lang=selected_language)
                    selected_language = actual_language if actual_language is not None else selected_language
                    record_msgs.append(dict(type='cond_text', text=text))
                record_msgs.append(messages[i])
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_pure_text_data> : {msg['type']}")
        
        return record_msgs
    
    def get_reversed_text_data(self, index, messages, selected_language):
        """
        首先检查该序列是否满足，图文图文图文图..., 满足实现翻转，
        翻转messages序列，
        同时图片的状态需要改变，原则是交换相邻图片的状态
        instruction也需要使用reversed_en---text_en, reversed_zh-->text_zh
        只适合部分场景，比如编辑pair对，单纯的interleave数据；多轮对话场景不适用
        只支持图文图编辑pair对数据和图文图文图interleave数据，其他类型数据暂不支持；
        复杂类型的数据应该在数据生成的时候进行处理，而不是在数据加载的时候进行处理；
        """
        # 快速检查序列是否满足图文图文图这样的序列
        # 要求：第一个为图片，之后交替出现文本和图片，且类型只允许gen_image/cond_image与cond_text
        # 检查第一个必须是图片类型,最后一张图片必须是gen_image，此处严格限制以防止不可控情况的发生
        if messages[0]['type'] not in ["gen_image", "cond_image"] or messages[-1]['type'] != "gen_image":
            # raise ValueError("messages序列不满足图文交替，第一个元素不是图片类型。")
            print(f"messages序列不满足图文交替，第一个元素不是图片类型。")
            return self.get_pure_text_data(index, messages, selected_language)
        # 检查交替
        for idx, msg in enumerate(messages):
            if idx % 2 == 0:
                # 偶数位必须是图片
                if msg['type'] not in ["gen_image", "cond_image"]:
                    # raise ValueError(f"messages序列不满足图文交替，第{idx}个元素不是图片类型。")
                    print(f"messages序列不满足图文交替，第{idx}个元素不是图片类型。")
                    return self.get_pure_text_data(index, messages, selected_language)
            else:
                # 奇数位必须是文本
                if msg['type'] not in ["cond_text"]:
                    # raise ValueError(f"messages序列不满足图文交替，第{idx}个元素不是文本类型。")
                    print(f"messages序列不满足图文交替，第{idx}个元素不是文本类型。")
                    return self.get_pure_text_data(index, messages, selected_language)
        
        # # 检查通过翻转messages序列，并切换图片消息的type，调整instruction
        # 翻转后第一张图片和最后一张图片交换状态
        # 然后相邻图片交换状态，直到最后一张图片
        reversed_msgs = []
        pre_img_type = messages[0]['type']
        for idx, msg in enumerate(reversed(messages)):
            # 复制msg，避免直接修改原始messages
            msg_copy = msg.copy()
            if msg_copy['type'] in ["gen_image", "cond_image"]:
                # 修改图片状态，交换相邻图片的状态
                img_type = msg_copy['type']
                msg_copy['type'] = pre_img_type
                pre_img_type = img_type
            # 处理instruction字段
            if msg_copy['type'] == 'cond_text':
                assert "reversed_en" in msg and "reversed_zh" in msg, "reversed_en or reversed_zh not in msg <get_reversed_text_data>"
                # 使用reversed_en和reversed_zh替换text_en和text_zh，这样是最小化改动代码
                msg_copy["text_en"] = msg_copy["reversed_en"]
                msg_copy["text_zh"] = msg_copy["reversed_zh"]
            reversed_msgs.append(msg_copy)
        
        # 处理文本消息
        record_msgs = self.get_pure_text_data(index, reversed_msgs, selected_language)

        return record_msgs
    
    def get_extra_info_data(self, index, messages, selected_language):
        """
        从messages中获取额外信息，比如图片的caption，原始的编辑指令，其他文本信息
        extra_info_sample: {"instruction_en": 0.5, "instruction_zh": 0.5}
        是一个字典，key是messages["extra_info"]中的key，value是采样概率
        不支持反向指令，只支持正向指令；反向指令的逻辑过于复杂，需要额外处理；
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if "text" in msg['type']:
                # 加载extra_info
                if "extra_info" in msg:
                    extra_info = json.loads(msg["extra_info"])
                else:
                    extra_info = {}
                
                candidates = []
                weights = []
                for key, value in self.extra_info_sample.items():
                    if "@" in key:
                        k1, k2 = key.split("@")
                        try:
                            text = extra_info[k1][k2]
                        except: # noqa
                            self.logger.warning(f"key: {k1}@{k2} not in extra_info")
                            text = ""
                    else:
                        text = extra_info.get(key, "")

                    if value > 0.0 and text and text not in ["", "NONE", "None", "none", "NULL", "Null", "null"]:
                        candidates.append(text)
                        weights.append(value)
                    
                if len(candidates) > 0:
                    text = random.choices(candidates, weights=weights, k=1)[0]
                    lang_cls = count_zh_en_words(text)
                    selected_language = "zh" if lang_cls['zh_ratio'] > lang_cls['en_ratio'] else "en"
                else:
                    text, actual_language = self.get_text_from_text_message(msg, selected_language)
                    selected_language = actual_language
                record_msgs.append(dict(type=msg['type'], text=text))
            # 处理生成图片消息
            elif "image" in msg['type']:
                # 处理第一张图片的特殊情况
                if i == 0 and msg['type'] == "gen_image":
                    text, actual_language = self.get_caption_from_image_message(messages[i], preferred_lang=selected_language)
                    selected_language = actual_language if actual_language is not None else selected_language
                    record_msgs.append(dict(type='cond_text', text=text))
                record_msgs.append(messages[i])
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_extra_info_data> : {msg['type']}")
        
        return record_msgs
    
    def get_cot_recaption_data(self, index, messages, lang):
        """
        获取caption_cot数据
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if "text" in msg['type']:
                # 加载text信息
                if self.data_key and self.data_key in msg:
                    cot_info = json.loads(msg[self.data_key])
                else:
                    cot_info = msg
                
                if self.caption_cot:
                    caption_cot = self.caption_cot if random.random() < self.caption_cot_prob else None
                else:
                    caption_cot = None
                out = self.multi_caption_manager.get_caption(cot_info, return_dict=True, pattern=caption_cot,
                                                             real_index=index)
                if isinstance(out, (tuple, list)):
                    # Make sure both captions are valid
                    if out[0].caption and out[1].caption:
                        text = (out[0].caption, out[1].caption)
                    else:
                        text = None
                    lang = out[0].lang
                else:
                    text = out.caption
                    lang = out.lang
                
                if isinstance(text, str):
                    text = str(text).strip()
                    # Remove meaningless characters
                    text = text.replace("\\N", "").strip("，,")
                elif isinstance(text, (tuple, list)):
                    is_tuple = isinstance(text, tuple)
                    text = [t.replace("\\N", "").strip("，,") for t in text]
                    if is_tuple:
                        text = tuple(text)
                
                # -- for recaption
                if isinstance(text, tuple):
                    text, recaption = text
                else:
                    recaption = None

                if not text:
                    text = ''
                    self.logger.warning(f"[{index=}] some warning for text in <get_cot_recaption_data> is None")
                    bad_sample = True
                else:
                    bad_sample = False
                
                record_msgs.append(dict(type="cond_text", text=text, bad_sample=bad_sample))
                # -- for reasoning
                if recaption is not None and self.reasoning_cot_prob > 0 and random.random() < self.reasoning_cot_prob:
                    # self.extra_think_col should already registered in self.register_extra_cols()
                    think_key = getattr(self, f"extra_think_{lang}_key").key     # reasoning text
                    reasoning = cot_info[out[0].sel_col][think_key]
                    # Remove leading and trailing <think> tags if they exist. These tags will be added in the template for
                    # detailed control.
                    reasoning = re.sub(r"<think>(.*)</think>", r"\1", reasoning, flags=re.DOTALL)
                    record_msgs.append(dict(type="gen_text", text="<think>{}</think>".format(reasoning)))
                # -- for recaption
                if recaption is not None:
                    record_msgs.append(dict(type="gen_text", text="<recaption>{}</recaption>".format(recaption)))
                
            # 处理生成图片消息
            elif "image" in msg['type']:
                # 处理第一张图片的特殊情况
                if i == 0 and msg['type'] == "gen_image":
                    text, actual_language = self.get_caption_from_image_message(messages[i], preferred_lang=lang)
                    lang = actual_language if actual_language is not None else lang
                    record_msgs.append(dict(type='cond_text', text=text))
                record_msgs.append(messages[i])
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_cot_recaption_data> : {msg['type']}")
        
        return record_msgs
    
    def get_pure_caption_data(self, index, messages, selected_language):
        """
        在每张图前面插入caption，组成cond_text-->gen_image-->cond_text-->gen_image-->...
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if "image" in msg['type']:
                text, actual_language = self.get_caption_from_image_message(msg, preferred_lang=selected_language)
                selected_language = actual_language if actual_language is not None else selected_language
                if self.template == "instruct" and random.random() < self.add_img_ratio:
                    text = text + "{}"
                record_msgs.append(dict(type='cond_text', text=text))
                # 构造一个新的 gen_image 消息
                msg['type'] = 'gen_image'
                record_msgs.append(msg)
            elif "text" in msg['type']:
                pass
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_caption_data> : {msg['type']}")
                
        return record_msgs
    
    def get_i2i_caption_data(self, index, messages, selected_language):
        """
        在每两张图之间，组成cond_image-->cond_text-->gen_image-->cond_text-->gen_image-->...
        """
        record_msgs = []
        # 处理所有文本消息
        for i, msg in enumerate(messages):
            # 处理文本消息
            if i < len(messages) - 1 and msg['type'] == "cond_text" and messages[i + 1]['type'] == "gen_image":
                text, actual_language = self.get_caption_from_image_message(messages[i + 1], preferred_lang=selected_language)
                selected_language = actual_language if actual_language is not None else selected_language
                record_msgs.append(dict(type='cond_text', text=text))
            elif "image" in msg['type']:
                record_msgs.append(msg)
            else:
                raise ValueError(f"[{index=}] Unsupported message type in func: <get_i2i_caption_data> : {msg['type']}")
                
        return record_msgs

    def get_think_caption_data(self, index, messages, selected_language):
        """
        通过在cond_image后面插入caption，形成简单的gen_cot模式
        通过在gēn_image之前插入caption，形成简单的cond_cot模式
        "thinK_0": <think>用户现在想生成一张图片，根据用户的指令，我先生成图片的描述：{}。现在根据描述生成图片</think>,
        "think_1": <think>用户提供了{}张图片，图片内容如下: {}, 然后用户的指令是: {}</think>,
        """
        if self.think_type == 0:
            think_index, selected_language = self.find_think_type0_index_in_messages(messages, selected_language)
            think_template = self.get_think_template("thinK_0", lang=selected_language)
            
            record_msgs = []
            for i, msg in enumerate(messages):
                if i in think_index:
                    _, _, _, text, gen_caption = think_index[i]
                    record_msgs.append(dict(type=msg['type'], text=text))
                    record_msgs.append(dict(type="gen_text", text=think_template.format(gen_caption)))
                elif "text" in msg['type']:
                    text, actual_language = self.get_text_from_text_message(msg, selected_language)
                    selected_language = actual_language
                    record_msgs.append(dict(type=msg['type'], text=text))
                elif "image" in msg['type']:
                    if i == 0 and msg['type'] == "gen_image":
                        text, actual_language = self.get_caption_from_image_message(messages[i], preferred_lang=selected_language)
                        selected_language = actual_language if actual_language is not None else selected_language
                        record_msgs.append(dict(type='cond_text', text=text))
                    record_msgs.append(messages[i])
                else:
                    raise ValueError(f"[{index=}] Unsupported message type in func: <get_think_data> : {msg['type']}")
        
        elif self.think_type == 1:
            think_index, selected_language = self.find_think_type1_index_in_messages(messages, selected_language)
            think_template = self.get_think_template("think_1", lang=selected_language)
            
            record_msgs = []
            for i, msg in enumerate(messages):
                if i in think_index:
                    _, n_cond, tink_text, text, _ = think_index[i]
                    record_msgs.append(dict(type=msg['type'], text=text))
                    record_msgs.append(dict(type="gen_text", text=think_template.format(n_cond, tink_text, text)))
                elif "text" in msg['type']:
                    text, actual_language = self.get_text_from_text_message(msg, selected_language)
                    selected_language = actual_language
                    record_msgs.append(dict(type=msg['type'], text=text))
                elif "image" in msg['type']:
                    if i == 0 and msg['type'] == "gen_image":
                        text, actual_language = self.get_caption_from_image_message(messages[0], preferred_lang=selected_language)
                        selected_language = actual_language if actual_language is not None else selected_language
                        record_msgs.append(dict(type='cond_text', text=text))
                    record_msgs.append(messages[i])
                else:
                    raise ValueError(f"[{index=}] Unsupported message type in func: <get_think_data> : {msg['type']}")

        else:
            raise ValueError(f"Unsupported think_type: {self.think_type}")
    
        return record_msgs

    def choose_language_by_prob(self):
        """
        根据配置的权重，随机选择语言
        """
        # 在一开始就选择语言，确保整个template中语言一致
        language_weights = []
        available_languages = []
        
        for lang in ['zh', 'en']:
            weight = self.sample_lang.get(lang, 0.0)
            if weight > 0.0:
                available_languages.append(lang)
                language_weights.append(weight)
        
        # 如果没有配置任何语言权重，默认使用英文    
        if not available_languages:
            selected_language = "en"
        elif len(available_languages) == 1:
            selected_language = available_languages[0]
        else:
            # 根据权重随机选择语言
            selected_language = random.choices(available_languages, weights=language_weights, k=1)[0]

        return selected_language
    
    def get_img_meta_info(self, index, messages, image_key):
        """
        获取图片的meta信息
        """
        images = []
        cond_images = []
        image_flags = []
        for msg in messages:
            if "inpainting" in self.dataset_tag and msg['type'] == "cond_image":
                # src_image: tensor -1 ~ 1
                src_image, _, image_flag = self.get_image_with_size(
                    msg, random_crop=False, target_size_type="image",
                    return_vision_encoder_image=False, column=image_key, real_index=index,
                )
                h, w = src_image.i.h, src_image.i.w
                # lama random mask
                mask = self.mask_provider(h=h, w=w)     # (1, h, w)

                masked_src_image = self.as_image_tensor(src_image * (1 - mask), image_type="vae")     # mask with gray

                image_tensor = masked_src_image

                if self.use_joint_image_feature:
                    masked_src_pil_image = self.tensor_to_pil_image(masked_src_image)
                    vision_encoder_tensor = self.vision_encoder_process_image(masked_src_pil_image)
                else:
                    vision_encoder_tensor = None

                image_flags.append(image_flag)
                if vision_encoder_tensor is not None:
                    image_tensor = JointImage(image_tensor, vision_encoder_tensor)
                # `image` will be treated both as target image and following condition image.
                # `src_image` will be treated as condition image only.
                cond_images.append(image_tensor)
                # Add a reference to the image_tensor for building template
                msg['image_tensor'] = image_tensor
            elif msg['type'] in ['cond_image', 'gen_image']:
                apply_resize_degradation = True if msg['type'] == 'cond_image' and self.cond_resize_degradation_prob > 0 and random.random() < self.cond_resize_degradation_prob else False
                image_tensor, vision_encoder_tensor, image_flag = self.get_image_with_size(
                    msg, random_crop=False, target_size_type="image",
                    return_vision_encoder_image=self.use_joint_image_feature, column=image_key, real_index=index,
                    apply_resize_degradation=apply_resize_degradation,
                )
                image_flags.append(image_flag)
                if vision_encoder_tensor is not None:
                    image_tensor = JointImage(image_tensor, vision_encoder_tensor)
                # `image` will be treated both as target image and following condition image.
                # `src_image` will be treated as condition image only.
                cond_images.append(image_tensor)
                if msg['type'] == 'gen_image':
                    images.append(image_tensor)
                # Add a reference to the image_tensor for building template
                msg['image_tensor'] = image_tensor
        image_flag = "normal" if all([x == "normal" for x in image_flags]) else "gray"
        return messages, images, cond_images, image_flag

    def process_messages_with_patchers(self, messages, index):
        
        # 如果没有 patcher 需要应用，直接返回
        if not self.patcher_names:
            return

        def _find_next_image_tensor(start_index, messages):
            """从指定索引开始，查找并返回下一个 'gen_image' 消息中的 image_tensor。"""
            for i in range(start_index + 1, len(messages)):
                if messages[i]['type'] == 'gen_image':
                    return messages[i]['image_tensor']
            return None

        # --- 1. 收集需要处理的 prompts ---
        prompts_to_patch = defaultdict(list)
        for i, msg in enumerate(messages):
            # 筛选出需要处理的文本消息
            is_cond_text = msg['type'] == 'cond_text'
            is_recaption_text = msg['type'] == 'gen_text' and '<recaption>' in msg['text']
            
            if not (is_cond_text or is_recaption_text):
                continue

            # 查找与此文本关联的下一个生成的图像
            target_image = _find_next_image_tensor(i, messages)
            if not target_image:
                continue

            # 提取 prompt 并记录是否是 recaption
            if is_recaption_text:
                prompt = re.sub(r"<recaption>(.*)</recaption>", r"\1", msg['text'], flags=re.DOTALL)
                was_recaption = True
            else:
                prompt = msg['text']
                was_recaption = False
                
            # 判断语言
            lang_stats = count_zh_en_words(prompt)
            lang = "zh" if lang_stats['zh_ratio'] > lang_stats['en_ratio'] else "en"
            
            # 按 (图像比例, 语言) 分组
            key = (target_image.vae_image.i.ratio_index, lang)
            prompts_to_patch[key].append({
                "index": i,
                "prompt": prompt,
                "was_recaption": was_recaption
            })

        # --- 2. 批量处理收集到的 prompts ---
        if not prompts_to_patch:
            return

        patched_results = []
        for (ratio_index, lang), prompt_data_list in prompts_to_patch.items():
            # 提取原始 prompts 列表
            original_prompts = [data['prompt'] for data in prompt_data_list]
            
            # 调用核心处理函数
            new_prompts = apply_prompt_patchers(
                original_prompts, self.patcher_names, lang,
                index_manager=self.index_manager, index=index, ratio_index=ratio_index
            )
            
            # 将处理结果与原始信息合并
            for data, new_prompt in zip(prompt_data_list, new_prompts):
                data['new_prompt'] = new_prompt
                patched_results.append(data)

        # --- 3. 更新原始 messages 列表 ---
        for result in patched_results:
            msg_index = result['index']
            new_prompt = result['new_prompt']
            
            if result['was_recaption']:
                messages[msg_index]['text'] = f"<recaption>{new_prompt}</recaption>"
            else:
                messages[msg_index]['text'] = new_prompt

    @resample_on_gray
    def get_interleave_data(self, index):
        # Get messages
        messages = self.index_manager.get_attribute(index, **self.message_col)
        image_key = self.index_kwargs.get('image_key', 'cache_image')

        # 将第一张图的属性设置成gen_image，这样可以在前面拼接上caption,第一轮执行文生图任务
        if self.gen_first_img and "image" in messages[0]['type']:
            messages[0]['type'] = 'gen_image'
        
        if self.cond_first_img and "image" in messages[0]['type']:
            messages[0]['type'] = 'cond_image'
        
        # process messages
        messages = self.process_messages(index, messages)

        # encode the image and get the meta info
        messages, images, cond_images, image_flag = self.get_img_meta_info(index, messages, image_key)
        self.process_messages_with_patchers(messages, index)

        # If there is bad sample, we trigger gray resample
        has_bad_sample = False
        for msg in messages:
            if msg.get('bad_sample', False):
                has_bad_sample = True
                break
        if has_bad_sample:
            image_flag = "gray"

        return InterleaveData(
            messages=messages, images=images, cond_images=cond_images,
            image_flag=image_flag, index=index,
        )

    def process_messages(self, index, messages):
        """
        根据配置的inter_type，处理messages
        """
        selected_language = self.choose_language_by_prob()

        type_func_map = {
            "inpainting": self.get_inpainting_data,
            "pure_text": self.get_pure_text_data,
            "pure_caption": self.get_pure_caption_data,
            "i2i_caption": self.get_i2i_caption_data,
            "think_caption": self.get_think_caption_data,
            "reversed_text": self.get_reversed_text_data,
            "extra_info": self.get_extra_info_data,
            "cot_recaption": self.get_cot_recaption_data,
        }

        if self.inter_type in type_func_map:
            return type_func_map[self.inter_type](index, messages, selected_language)  # noqa
        elif self.inter_type == "mixup":
            # 按照概率采样
            data_types = [
                (self.get_pure_text_data, self.text_ratio),
                (self.get_pure_caption_data, self.cap_ratio),
                (self.get_i2i_caption_data, self.i2i_ratio),
                (self.get_think_caption_data, self.think_ratio),
                (self.get_reversed_text_data, self.reversed_ratio),
                (self.get_extra_info_data, self.extra_ratio),
                (self.get_cot_recaption_data, self.cot_ratio),
            ]
            
            # 根据概率权重选择数据处理方法
            funcs, weights = zip(*data_types)
            chosen_func = random.choices(funcs, weights=weights, k=1)[0]
            return chosen_func(index, messages, selected_language)
        else:
            raise ValueError(f"Unsupported inter_type: {self.inter_type}")

    def build_pretrain_template(self, record_msgs):
        # Build text-image interleave template. Note that the last clean image is removed to
        # save computation. The attention mask will be adjusted accordingly.
        # For text in interleave data, we use uniform text drop, which means if meets an uncond batch,
        # we will replace all the text tokens with <cfg> tokens.
        #
        # Example (T: text token, N: noised image token, C: clean image token):
        #   T  T  T  N  N  T  T  T  C  C  N  N  T  T  T  C  C  N  N

        # uncondition
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))

        # We want the image prefix special tokens <boi>, <img_size_*>, and <img_ratio_*> to be learned.
        # It is implemented by adding text mask end offsets to the prefix text sections.
        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0)

        # (boi + eoi + img_size_* + img_ratio_* + timestep) * 2 + joint
        extra_num_tokens = (2 + (2 if self.add_image_shape_token else 0) + (1 if self.add_timestep_token else 0)) * 2 + 1
        extra_once = 1 + self.dummy_number      # used only once, bos + dummy

        if "<recaption>" in str(record_msgs):
            # If it has recaption, text length is limited by text_cot_token_length - text_token_length - text_reason_token_length
            prompt_max_length = self.text_cot_token_length - self.text_token_length - self.text_reason_token_length
        else:
            # If no recaption, we assume no reasoning.
            prompt_max_length = self.text_token_length

        sections = [] # 构建sections
        for msg in record_msgs:
            if msg['type'] == 'cond_text':
                sections.append(dict(type='text', text=msg['text'], max_length=prompt_max_length - extra_num_tokens - extra_once, **uncond_kwargs, ignore=True,))
                extra_once = 0
            elif msg['type'] == 'gen_text':
                msg['text'] = msg['text'].strip()
                if "<recaption>" in msg['text'] and "</recaption>" in msg['text']:
                    recaption = re.sub(r"<recaption>(.*)</recaption>", r"\1", msg['text'], flags=re.DOTALL)
                    sections.extend([
                            dict(type="text", text="<recaption>", ignore=True),   # start token is ignored to serve as a switch
                            dict(type="text", text=recaption, ignore=do_uncond,
                                 max_length=self.text_token_length - 2, **uncond_kwargs),
                            dict(type="text", text="</recaption>", ignore=do_uncond),
                        ])
                elif "<think>" in msg['text'] and "</think>" in msg['text']:
                    reasoning = re.sub(r"<think>(.*)</think>", r"\1", msg['text'], flags=re.DOTALL)
                    sections.extend([
                        dict(type="text", text="<think>", ignore=True),  # start token is ignored to serve as a switch
                        dict(type="text", text=reasoning, ignore=do_uncond,
                             max_length=self.text_reason_token_length - 2, **uncond_kwargs),
                        dict(type="text", text="</think>", ignore=do_uncond),
                    ])
                else:
                    sections.append(dict(type="text", text=msg['text'], max_length=prompt_max_length - extra_num_tokens - extra_once, **uncond_kwargs))
                # sections.append(dict(type='text', text=msg['text'], max_length=self.text_token_length - extra_num_tokens - extra_once, **uncond_kwargs, ignore=do_uncond,))
                extra_once = 0
            elif msg['type'] == 'gen_image':
                # gen_image always has a successive joint_image/src_image.
                if self.use_joint_image_feature:
                    sections.extend([
                        dict(type='gen_image', **msg['image_tensor'].vae_image.i.meta_info),
                        dict(type='joint_image', **msg['image_tensor'].i.meta_info),
                    ])
                else:
                    sections.extend([
                        dict(type='gen_image', **msg['image_tensor'].i.meta_info),
                        dict(type='src_image', **msg['image_tensor'].i.meta_info),
                    ])
            elif msg['type'] == 'cond_image':
                # 条件图像，类似于src_image的处理
                assert self.use_joint_image_feature, "cond_image should only be used with use_joint_image_feature=True"
                sections.extend([
                    dict(type='joint_image', **msg['image_tensor'].i.meta_info),
                ])
            else:
                raise ValueError(f"Unsupported message type3: {msg['type']}")
            
        if not self.use_front_src_image and not self.use_joint_image_feature:
            # Switch src_image with all the following text until the next image
            for i in range(len(sections) - 1):
                if sections[i]['type'] == 'src_image' and sections[i + 1]['type'] == 'text':
                    sections[i], sections[i + 1] = sections[i + 1], sections[i]

        # For all types of images, only gen_image has losses and is allowed to be the last section.
        if 'image' in sections[-1]['type'] and sections[-1]['type'] != 'gen_image':
            sections.pop()
        # Add empty text to enable <eos> token in the text mask
        sections.append(dict(type='text', text='', end_offset=1))
        # Insert an empty text section before image to include num_image_prefix tokens to the text mask
        new_sections = []
        for s in sections:
            if (
                ('image' in s['type'] and not self.use_front_src_image and not self.use_joint_image_feature)
                or (s['type'] == 'gen_image')
            ):
                new_sections.append(dict(type='text', text='', ignore=do_uncond, end_offset=num_image_prefix))
            new_sections.append(s)

        return new_sections

    def build_instruct_template(self, record_msgs):
        """
        构建多轮对话式的instruct模板，支持User/Assistant轮流，图片/文本交错。
        """
        sections = []
        
        # 1. uncond
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))
        # We want the image prefix special tokens <boi>, <img_size_*>, and <img_ratio_*> to be learned.
        # It is implemented by adding text mask end offsets to the prefix text sections.
        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0)

        # Control the length of each text, minus the length of the extra tokens
        # (boi + eoi + img_size_* + img_ratio_* + timestep) * 2 + joint
        extra_num_tokens = (2 + (2 if self.add_image_shape_token else 0) + (1 if self.add_timestep_token else 0)) * 2 + 1
        extra_once = 1 + self.dummy_number      # used only once, bos + dummy

        record_msgs_str = str(record_msgs)

        if "<recaption>" in record_msgs_str:
            # If it has recaption, text length is limited by text_cot_token_length - text_token_length - text_reason_token_length
            prompt_max_length = self.text_cot_token_length - self.text_token_length - self.text_reason_token_length
        else:
            # If no recaption, we assume no reasoning.
            prompt_max_length = self.text_token_length

        if self.use_unified_system_prompt:
            system_prompt = self.get_system_prompt(unified_system_prompts["en_unified"])
        else:
            if "<recaption>" in record_msgs_str:
                if "<think>" in record_msgs_str:
                    system_prompt = self.get_system_prompt(ti2i_system_prompts["en_think_recaption"])
                else:
                    system_prompt = self.get_system_prompt(ti2i_system_prompts["en_recaption"])
            else:
                system_prompt = self.get_system_prompt(ti2i_system_prompts["en_vanilla"])

        if system_prompt != "":
            assert self.system_prompt_token_length > 0, "system_prompt_token_length should be greater than 0"
            sections.extend([
                dict(type="text", text=system_prompt.strip("\n "), ignore=True, max_length=self.system_prompt_token_length - 1),
                dict(type="text", text=self.default_conv.sep, ignore=True), # "\n\n" 1 token
            ])

        # "User: " + "\n\n" + "Assistant: <answer>" + </answer><eos>共10个token
        extra_user_num_tokens = 5 + extra_num_tokens
        extra_assistant_num_tokens = 5 + extra_num_tokens

        # 2. 构建sections
        role_user = self.roles[0]
        role_assistant = self.roles[1]

        # 3. 构造 instruction_msgs
        sections.extend([dict(type="text", text=f"{role_user}: ", ignore=True)])
        role_flag = role_user
        for idx, msg in enumerate(record_msgs):
            if "cond" in msg['type']:
                if role_flag != role_user:
                    sections.extend([dict(type="text", text=f"</answer>{self.tokenizer.tokenizer.eos_token}", ignore=False)])
                    sections.extend([dict(type="text", text=f"{role_user}: ", ignore=True)])
                    role_flag = role_user
                # 处理用户输入的文本及图片
                if msg['type'] == 'cond_text':
                    # 当下一轮不是gen_text时，需要max_length减去extra_user_num_tokens + extra_assistant_num_tokens，
                    # 不然会少减去几个token，极端情况会少减去几个token，导致最后一张图片被drop，建议序列长度也留有轻微余量
                    if idx < len(record_msgs) - 1 and record_msgs[idx + 1]['type'] == 'gen_text':
                        extra_tokens = extra_user_num_tokens 
                    else:
                        extra_tokens = extra_user_num_tokens + extra_assistant_num_tokens
                    sections.append(dict(type="text", text=msg['text'], max_length=prompt_max_length - extra_tokens - extra_once, **uncond_kwargs, ignore=True))
                    extra_once = 0
                elif msg['type'] == 'cond_image':
                    # 条件图像，类似于src_image的处理
                    assert self.use_joint_image_feature, "in <build_instruct_template> cond_image should only be used with use_joint_image_feature=True"
                    sections.extend([dict(type='joint_image', **msg['image_tensor'].i.meta_info),])
                else:
                    raise ValueError(f"Unsupported message type in instruct template: {msg['type']}")
            elif "gen" in msg['type']:
                if role_flag != role_assistant:
                    sections.extend([dict(type="text", text=self.default_conv.sep, ignore=True)])
                    sections.extend([dict(type="text", text=f"{role_assistant}: ", ignore=True)])
                    role_flag = role_assistant
                # 处理模型输出的文本及图片
                if msg['type'] == 'gen_text':
                    msg['text'] = msg['text'].strip()
                    if "<recaption>" in msg['text'] and "</recaption>" in msg['text']:
                        recaption = re.sub(r"<recaption>(.*)</recaption>", r"\1", msg['text'], flags=re.DOTALL)
                        sections.extend([
                                dict(type="text", text="<recaption>", ignore=True),   # start token is ignored to serve as a switch
                                dict(type="text", text=recaption, ignore=do_uncond,
                                     max_length=self.text_token_length - 2, **uncond_kwargs),
                                dict(type="text", text="</recaption>", ignore=do_uncond),
                            ])
                    elif "<think>" in msg['text'] and "</think>" in msg['text']:
                        reasoning = re.sub(r"<think>(.*)</think>", r"\1", msg['text'], flags=re.DOTALL)
                        sections.extend([
                            dict(type="text", text="<think>", ignore=True),  # start token is ignored to serve as a switch
                            dict(type="text", text=reasoning, ignore=do_uncond,
                                 max_length=self.text_reason_token_length - 2, **uncond_kwargs),
                            dict(type="text", text="</think>", ignore=do_uncond),
                        ])
                    else:
                        sections.append(dict(type="text", text=msg['text'], max_length=prompt_max_length - extra_assistant_num_tokens - extra_once, **uncond_kwargs))       
                elif msg['type'] == 'gen_image':
                    # gen_image always has a successive joint_image/src_image.
                    if self.use_joint_image_feature:
                        sections.extend([
                            dict(type="text", text="<answer>", ignore=True),
                            dict(type="text", text='', ignore=do_uncond, end_offset=num_image_prefix),
                            dict(type='gen_image', **msg['image_tensor'].vae_image.i.meta_info),
                            dict(type='joint_image', **msg['image_tensor'].i.meta_info),
                        ])
                    else:
                        sections.extend([
                            dict(type="text", text='', ignore=do_uncond, end_offset=num_image_prefix),
                            dict(type='gen_image', **msg['image_tensor'].i.meta_info),
                            dict(type='src_image', **msg['image_tensor'].i.meta_info),
                        ])
            else:
                raise ValueError(f"Unsupported message type in instruct template: {msg['type']}")
        
        # 4. For all types of images, only gen_image has losses and is allowed to be the last section.
        if 'image' in sections[-1]['type'] and sections[-1]['type'] != 'gen_image':
            sections.pop()
        
        # 5. 为最后一轮对话添加</answer><eos>
        if role_flag == role_assistant:
            sections.extend([dict(type="text", text=f"</answer>{self.tokenizer.tokenizer.eos_token}", ignore=False)])
        
        # Add empty text to enable <eos> token in the text mask
        sections.append(dict(type='text', text='', end_offset=1))

        return sections

    def get_rope_image_info(self, sections, output):
        # Interleave data has multiple gen images, and successive sections shouldn't attend to previous gen images.
        # Thus, we logically shift the successive sections by the length of previous gen image tokens to define
        # the token positions for 2d RoPE. For example,
        # The original sequence and positions:
        #     A dog is running . <boi> <img> <eoi> <boi> <img> <joint> <img> <eoi> Change the dog to  a cat  .
        #     0  1   2    3    4   5     6     7     8     9      10     11    12    13    14  15 16 17 18  19
        # The shifted sequence and positions (simulate the inference):
        #     A dog is running . <boi> <img> <eoi>
        #     0  1   2    3    4   5     6     7
        #                        <boi> <img> <joint> <img> <eoi> Change the dog to  a cat  .
        #                          5     6     7       8     9     10    11  12 13 14  15 16
        # Flatted shifted sequence and positions:
        #     A dog is running . <boi> <img> <eoi> <boi> <img> <joint> <img> <eoi> Change the dog to  a cat  .
        #     0  1   2    3    4   5     6     7     5     6     7       8     9     10    11  12 13 14  15 16

        num_image_prefix = 1 + (2 if self.add_image_shape_token else 0) + (1 if self.add_timestep_token else 0)
        num_image_suffix = 1

        if self.args.rope_type == "2d":
            image_slices = output.all_image_slices
            image_idx = 0
            offset = 0
            num_overlapped_tokens = 0
            image_shapes = []
            shifted_image_slices = []
            last_is_gen = False
            for section in sections:
                if image_idx >= len(image_slices):
                    break
                if section['type'] == 'gen_image':
                    image_shapes.append((section['token_height'], section['token_width']))
                    # shift image slice
                    sli = image_slices[image_idx]
                    shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                    image_idx += 1
                    offset += section['token_length'] + num_image_prefix + num_image_suffix
                    last_is_gen = True

                elif section['type'] in ['joint_image', 'src_image']:
                    # We have an important assumption here that, if a gen_image is not the last section,
                    # it must be followed by a joint_image section immediately.
                    # That means [gen_image][other sections...][joint_image] is not allowed.
                    if last_is_gen:
                        # Append a text section to shift <boi><size><ratio><timestep> tokens
                        image_shapes.append((None, None))
                        sli = image_slices[image_idx]
                        prefix_start = sli.start - num_image_prefix
                        prefix_end = sli.start
                        shifted_image_slices.append(slice(prefix_start - offset, prefix_end - offset))
                        num_overlapped_tokens = offset

                    if isinstance(section['token_height'], list):
                        assert len(section['token_height']) == len(section['token_width']), \
                            f"token_height and token_width should have the same length, but got {len(section['token_height'])} and {len(section['token_width'])}"
                        # shift image slice
                        for i in range(len(section['token_height'])):
                            image_shapes.append((section['token_height'][i], section['token_width'][i]))
                            sli = image_slices[image_idx + i]
                            shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                            if last_is_gen and i == 0:
                                # Append a text section to shift <eoi> token
                                image_shapes.append((None, None))
                                suffix_start = sli.stop
                                suffix_end = sli.stop + num_image_suffix
                                shifted_image_slices.append(slice(suffix_start - offset, suffix_end - offset))
                        image_idx += len(section['token_height'])
                    else:
                        image_shapes.append((section['token_height'], section['token_width']))
                        # shift image slice
                        sli = image_slices[image_idx]
                        shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                        if last_is_gen:
                            # Append a text section to shift <eoi> token
                            image_shapes.append((None, None))
                            suffix_start = sli.stop
                            suffix_end = sli.stop + num_image_suffix
                            shifted_image_slices.append(slice(suffix_start - offset, suffix_end - offset))
                        image_idx += 1
                    last_is_gen = False

            assert len(shifted_image_slices) == len(image_shapes), (
                f"Size miss matched: shifted image slices({len(shifted_image_slices)}) != image shapes({len(image_shapes)})"
            )
            return list(zip(shifted_image_slices, image_shapes)), num_overlapped_tokens
        return None, 0

    def __getitem__(self, index):
        data = self.get_interleave_data(index)
        messages, gen_images, cond_images, image_flag, index = data.unbind()

        # print(f"record_msgs: {messages}"
        if self.template == "pretrain":
            sections = self.build_pretrain_template(messages)
        elif self.template == "instruct":
            sections = self.build_instruct_template(messages)
        else:
            raise NotImplementedError(f"Unsupported template: {self.template}")

        # Here, the length of output <= length of the sequence in sections, and there is a drop last situation.
        max_token_length = self.max_sequence_length if self.sequence_pack else self.interleave_max_length
        try:
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_pad=False if self.sequence_pack else 'auto',
                drop_last=self.drop_last,
            )
        except AssertionError as e:
            self.logger.error(
                f"Error in encoding sections (index={index}): "
                f"{self.interleave_max_length=}, {self.drop_last=}, {sections}"
            )
            raise e
        
        if self.use_joint_image_feature:
            n_tgt_images = len(output.gen_image_slices)
            n_joint_images = len(output.joint_image_slices)
            tgt_images = [x.vae_image for x in gen_images[:n_tgt_images]]
            vae_images = [x.vae_image for x in cond_images[:n_joint_images]]
            vit_images = [x.vision_image for x in cond_images[:n_joint_images]]

            assert len(tgt_images) >= 1, \
                f"(index={index}) At least one target image is required, but got {len(tgt_images)}. \n {sections}"
            iw_ih_scatter_src = None
        else:
            if not self.drop_last:
                assert len(output.gen_image_slices) == len(output.src_image_slices) + 1, \
                    f"The number of gen_images should be one more than the number of cond_images, " \
                    f"but got {len(output.gen_image_slices)} and {len(output.src_image_slices)}"
            # When drop_last enabled, some images may be dropped. We count the actual number
            # of src and tgt images from the slices.
            n_src_images = len(output.src_image_slices)
            n_tgt_images = len(output.gen_image_slices)
            assert n_src_images == n_tgt_images or n_src_images + 1 == n_tgt_images, \
                f"The number of gen_images should be equal to or one more than the number of cond_images, " \
                f"but got {n_src_images} and {n_tgt_images}"
            vae_images, tgt_images = cond_images[:n_src_images], gen_images[:n_tgt_images]

            iw_ih_srcs = []
            for section in sections:
                if 'image' in section['type']:
                    iw_ih_srcs.extend([section['image_width'], section['image_height']])
            iw_ih_scatter_src = torch.tensor(iw_ih_srcs, dtype=torch.long)

            vit_images = []

        # Format cond_images
        if len(vae_images) == 0:
            vae_images = None
        if len(vit_images) == 0:
            vit_images = None

        # vit_images have some special kwargs for vision encoder.
        if vit_images is not None:
            vision_encoder_kwargs = defaultdict(list)
            for und_image in vit_images:
                for k, v in und_image.vision_encoder_kwargs.items():
                    vision_encoder_kwargs[k].append(v)
            # attention_mask: n x seq_len, spatial_shapes n x 2
            vision_encoder_kwargs = {k: torch.stack(v) for k, v in vision_encoder_kwargs.items()}
            if vision_encoder_kwargs == {}:
                vision_encoder_kwargs = None
        else:
            vision_encoder_kwargs = None

        # Try to stack images
        if all(img.shape == tgt_images[0].shape for img in tgt_images):
            tgt_images = torch.stack(tgt_images)
        if vae_images is not None and all(img.shape == vae_images[0].shape for img in vae_images):
            vae_images = torch.stack(vae_images)
        if vit_images is not None and all(img.shape == vit_images[0].shape for img in vit_images):
            vit_images = torch.stack(vit_images)

        target_tokens = output.tokens.clone()
        # If the ending images are dropped, the dummy text previous of the image still move end_offset, which
        # causes that target tokens will contain the successive pad tokens. So we always set pad tokens to -100.
        pad_token_mask = target_tokens == self.tokenizer.pad_token
        output.text_mask[pad_token_mask] = 0.0
        target_tokens[output.text_mask == 0.0] = -100

        # ------------------------------------------------------------------------------------------------
        # Interleave attention mask specification.
        # For text-image interleave data, we assume there are multiple images and text interleaved,
        # and we will calculate loss on all the images. In this case, within the transfusion model,
        # when an image serves as the prefix, it should be a clean image; when it serves as the target,
        # it should be a noised image. Therefore, each image in the sequence is represented by two blocks
        # of successive image tokens. In attention mask, all the tokens should not attend to the noised
        # image tokens except for the noised image tokens themselves.
        # Note that we don't assume the text always exists between the previous image and the next image,
        # which means the (N N C C N N) pattern is allowed.
        #
        # Example (T: text token, N: noised image token, C: clean image token):
        #   T  T  T  N  N  T  T  T  C  C  N  N  T  T  T  C  C  N  N
        # T x
        # T x  x
        # T x  x  x
        # N x  x  x  x  x
        # N x  x  x  x  x
        # T x  x  x        x
        # T x  x  x        x  x
        # T x  x  x        x  x  x
        # C x  x  x        x  x  x  x  x
        # C x  x  x        x  x  x  x  x
        # N x  x  x        x  x  x  x  x  x  x
        # N x  x  x        x  x  x  x  x  x  x
        # T x  x  x        x  x  x  x  x        x
        # T x  x  x        x  x  x  x  x        x  x
        # T x  x  x        x  x  x  x  x        x  x  x
        # C x  x  x        x  x  x  x  x        x  x  x  x  x
        # C x  x  x        x  x  x  x  x        x  x  x  x  x
        # N x  x  x        x  x  x  x  x        x  x  x  x  x  x  x
        # N x  x  x        x  x  x  x  x        x  x  x  x  x  x  x
        # ------------------------------------------------------------------------------------------------
        if self.task_kwargs.get('attn_type', 'auto') == 'auto':
            num_image_prefix = 1 + (2 if self.add_image_shape_token else 0) + (1 if self.add_timestep_token else 0)
            num_image_suffix = 1
            n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + self.dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            for image_slice in output.gen_image_slices:
                attention_mask[image_slice, image_slice] = True
                hole_slice = slice(image_slice.start - num_image_prefix, image_slice.stop + num_image_suffix)
                attention_mask[hole_slice.stop:, hole_slice] = False  # Make a hole under noise-kv
            cond_slices = output.joint_image_slices if self.use_joint_image_feature else output.src_image_slices
            for image_slice in cond_slices:
                attention_mask[image_slice, image_slice] = True
            attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = None

        # 2d rope
        rope_image_info, n_overlapped_tokens = self.get_rope_image_info(sections, output)

        ret = {
            "data_type": "ti2i",
            "dtype": self.dataset_tag,
            "n_samples": 1,
            "src_image": vae_images,    # name it src_image for compatibility
            "image": tgt_images,
            "tokens": output.tokens,
            "target_tokens": target_tokens,
            "text_mask": output.text_mask,
            "src_image_mask": output.src_image_mask,
            "image_mask": output.gen_image_mask,
        }
        
        if vit_images is not None:
            ret["und_images"] = vit_images  # name it und_images for compatibility
            # Only pass und_image_mask if it's not None
            if output.und_image_mask is not None:
                ret["und_image_mask"] = output.und_image_mask
            if vision_encoder_kwargs is not None:
                ret["vision_encoder_kwargs"] = vision_encoder_kwargs
        if attention_mask is not None:
            ret["attention_mask"] = attention_mask
        else:
            ret["src_image_slices"] = output.src_image_slices
            ret["gen_image_slices"] = output.gen_image_slices
            ret["joint_image_slices"] = output.joint_image_slices
            ret["und_image_slices"] = output.und_image_slices
        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,  # (2n)
                "iw_ih_scatter_src": iw_ih_scatter_src,             # (2n)
            })
        if output.timestep_scatter_index is not None:
            # For interleaved data, the timestep_scatter_index needs to be rearranged from the default
            # interleaved order (as the same in template and sections) to block order with the source
            # timesteps in front of the target timesteps, as the timestep_scatter_src is always built
            # by cat([src_t, t], dim=1) or cat(src_t[i], t[i], dim=1).
            # For example, if the index is [a1, b1, a2, b2, a3], where a1, a2, a3 are the timestep index of
            # target images and b1, b2 are those of source images, the timestep_scatter_index should be
            # rearranged to [b1, b2, a1, a2, a3].
            if output.timestep_scatter_index.shape[0] > 1:
                if output.cond_timestep_scatter_index is not None and output.gen_timestep_scatter_index is not None:
                    timestep_scatter_index = torch.cat([
                        output.cond_timestep_scatter_index,     # source images
                        output.gen_timestep_scatter_index,      # target images
                    ])
                elif output.cond_timestep_scatter_index is not None:
                    timestep_scatter_index = output.cond_timestep_scatter_index
                else:  
                    timestep_scatter_index = output.gen_timestep_scatter_index
            else:
                timestep_scatter_index = output.timestep_scatter_index
            ret.update({
                "timestep_scatter_index": timestep_scatter_index,   # (n)
            })
    
        if rope_image_info is not None:
            ret.update({
                "rope_image_info": rope_image_info,
                # only used in sequence pack
                "n_overlapped_tokens": n_overlapped_tokens,
            })

        return ret
