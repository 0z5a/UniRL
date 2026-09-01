from typing import List, Tuple, Union
import os
from functools import partial
import io
import warnings
from collections import defaultdict
import json
import random
from PIL import Image
import cv2

import numpy as np
import einops
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
try:
    import pycocotools.mask as mask_util
except ImportError:
    mask_util = None

from index_kits import arrow_mapper
from ..ar.mask_schedulers import create_attention_mask_general
from hymm.constants import VAE_META_INFO
from hymm.models import TokenizerWrapper
from hymm.utils.helpers import to_2tuple, default
from hymm.data_kits.caption_strategy import load_caption_processor
from hymm.models.tokenizers.conversation import get_conversation_template
from .index_dataset import IndexDataset
from .lama_mask import MixedMaskGenerator
from hymm.data_kits.instruction_template import inpainting_instructions, editing_instructions, subject_driven_instructions, face_id_instructions, text2image_instructions


# text2image used for instruction tuning
class Text2ImageGeminiAlphaArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        dataset_tag=None,
    ):
        super().__init__(
            index_file,
            multireso,
            index_kwargs.get("batch_size", 1),
            index_kwargs.get("world_size", 1),
            logger,
            debug,
        )
        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")
        
        self.args = args
        self.training_image_size = to_2tuple(training_image_size)

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.add_timestep_token = self.args.add_timestep_token
        self.use_front_boi_token = self.args.use_front_boi_token

        if index_kwargs is None:
            index_kwargs = {}
        
        shadow_file_fn = {}

        self.image_key = index_kwargs.get(f"{dataset_tag}_image_key", None)
        if self.image_key is None:
            raise ValueError(f"Missing `{dataset_tag}_image_key` in `index_kwargs`")

        if '@' in self.image_key:
            self.image_key, image_key_arrow_suffix = self.image_key.split('@')
            self.image_key_shadow = 'image'
            shadow_file_fn[self.image_key_shadow] = partial(arrow_mapper, suffix=f"_{image_key_arrow_suffix}")
        else:
            self.image_key_shadow = None

        self.image_caption_rate = index_kwargs.get(f"{dataset_tag}_image_caption_rate", 0.0)

        self.image_text_col = index_kwargs.get(f"{dataset_tag}_image_text_col", None)
        if self.image_caption_rate < 1.0 and self.image_text_col is None:
            raise ValueError(f"Missing `{dataset_tag}_image_text_col` in `index_kwargs`")
        if '@' in self.image_text_col:
            self.image_text_col, self.image_text_arrow_suffix = self.image_text_col.split('@')
            self.image_text_shadow = 'text'
        elif (image_text_arrow_suffix := index_kwargs.get(f"{dataset_tag}_image_text_arrow_suffix")) is not None:
            # for backward compatibility
            self.image_text_arrow_suffix = image_text_arrow_suffix
            self.image_text_shadow = 'text'
            warnings.warn(
                f"`{dataset_tag}_image_text_arrow_suffix` is deprecated, please use "
                f"`--{dataset_tag}_image_text_col <{dataset_tag}_image_text_col>@<arrow_suffix>` instead.", DeprecationWarning
            )
        else:
            self.image_text_arrow_suffix = None
            self.image_text_shadow = None

        self.image_caption_col = index_kwargs.get(f"{dataset_tag}_image_caption_col", None)
        if self.image_caption_rate > 0.0 and self.image_caption_col is None:
            raise ValueError(f"Missing `{dataset_tag}_image_caption_col` in `index_kwargs`")
        if '@' in self.image_caption_col:
            self.image_caption_col, self.image_caption_arrow_suffix = self.image_caption_col.split('@')
            self.image_caption_shadow = 'caption'
        elif (image_caption_arrow_suffix := index_kwargs.get(f"{dataset_tag}_image_caption_arrow_suffix")) is not None:
            # for backward compatibility
            self.image_caption_arrow_suffix = image_caption_arrow_suffix
            self.image_caption_shadow = 'caption'
            warnings.warn(
                f"`{dataset_tag}_image_caption_arrow_suffix` is deprecated, please use "
                f"`--{dataset_tag}_image_caption_col <{dataset_tag}_image_caption_col>@<arrow_suffix>` instead.", DeprecationWarning
            )
        else:
            self.image_caption_arrow_suffix = None
            self.image_caption_shadow = None

        self.caption_sample_ratio = index_kwargs.get(f"{dataset_tag}_caption_sample_ratio", None)
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=index_kwargs.get(f"{dataset_tag}_caption_processor", 'caption_process'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=index_kwargs.get(f"{dataset_tag}_caption_processor_kwargs", None),
            )
        else:
            self.use_structural_caption = False

        # 是否在特定场景下使用 `general_style` 列
        self.use_general_style = index_kwargs.get(f"{dataset_tag}_use_general_style")
        # 是否优先使用 source_text (应用 image_caption_ratio 的概率)
        self.try_first_use_source_text = index_kwargs.get(f"{dataset_tag}_try_first_use_source_text")


        self.pred_text_boi_eos = index_kwargs.get(f"{dataset_tag}_pred_text_boi_eos", None)
        if self.pred_text_boi_eos is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_text_boi_eos` in `index_kwargs`")
        self.pred_boi_mode = index_kwargs.get(f"{dataset_tag}_pred_boi_mode", None)
        if self.pred_boi_mode is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_boi_mode` in `index_kwargs`")

        self.uncond_p = index_kwargs.get(f"{dataset_tag}_uncond_p", 0.0)

        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        shadow_file_fn.update({'clip_score': partial(arrow_mapper, suffix='_clip_score')})
        if self.image_caption_rate < 1.0 and self.image_text_arrow_suffix is not None:
            shadow_file_fn.update({self.image_text_shadow: partial(arrow_mapper, suffix=self.image_text_arrow_suffix)})
        if self.image_caption_rate > 0.0 and self.image_caption_arrow_suffix is not None:
            shadow_file_fn.update(
                {self.image_caption_shadow: partial(arrow_mapper, suffix=self.image_caption_arrow_suffix)})

        index_load_kwargs["shadow_file_fn"] = shadow_file_fn
        self.index_manager = self.load_index(**index_load_kwargs)

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

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.text_token_length = text_token_length
        self.image_token_length = image_token_length
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self):
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = self.args.patch_size

    def get_raw_image(self, index):
        try:
            if self.image_key == "image":
                ret = self.get_image_from_arrow(index, column=self.image_key, shadow=self.image_key_shadow)
            elif self.image_key == "url_cos":
                ret = self.get_image_from_url_cos(index, column=self.image_key, shadow=self.image_key_shadow)
            else:
                raise ValueError(f"Unknown image_key: {self.image_key}")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"({self.image_key=}, {index=}) {type(e)}: {e}. Fallback to gray image.")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_image_with_size(self, index):
        image, image_flag = self.get_raw_image(index)

        origin_size = image.size  # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs, image_flag

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_instruction(self, ind):
        return text2image_instructions[(np.random.randint(0, 123456) + ind * 123) % len(text2image_instructions)]

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
        """
        Get image and text from a given index

        Args:
            index (int): Index of the dataset

        Returns:
            image (torch.FloatTensor): Image tensor with shape (3, H, W)
            text (str): Original text
            kwargs (dict): Additional information
                origin_size (torch.LongTensor): Original size of the image (W, H)
                target_size (torch.LongTensor): Target size of the image (W, H)
                crop_coords_xy (torch.LongTensor): Crop coordinates (x, y)
                index (torch.LongTensor): Index of the dataset
        """

        # Get text
        text = self.get_text(index)

        # Get image
        image, kwargs, image_flag = self.get_image_with_size(index)
        if image_flag == "gray":
            text = "A gray image"
        
        instruction = self.get_instruction(index)
        instruction_list = [
            "User: ",
            instruction.strip(),
            " " + text.strip(),
            "\n\n",
            "Assistant: ",
        ]
        uncond_enabled = [
            False,
            False,
            True,
            False,
            False,
        ]

        h, w = image.shape[1], image.shape[2]

        assert h % (self.downsample_factor[0] * self.patch_size) == 0 and w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({h} x {w}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"

        tk_height = h // (self.downsample_factor[0] * self.patch_size)
        tk_width = w // (self.downsample_factor[1] * self.patch_size)
        actual_image_token_length = tk_height * tk_width

        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, image_mask = self.tokenizer.encode_transfusion(
            *instruction_list,
            image_token_length=actual_image_token_length,
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_enabled=uncond_enabled,
            uncond_p=self.uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
            pred_boi_mode=self.pred_boi_mode,
        )
        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100

        # here use resized image size as scatter_src of iw and ih
        image_token_shape_wh = torch.tensor([w, h], dtype=torch.long)

        # Prepare attention mask
        n_tokens = tokens.shape[0] - 1
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        image_mask_expand = image_mask.bool()[:-1]
        image_mask_1 = image_mask_expand.view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(0, 1)
        attention_mask = causal_mask | (image_mask_1 & image_mask_2)
        attention_mask = attention_mask.unsqueeze(0)    # head dim

        # Prepare kwargs
        kwargs["index"] = index
        kwargs["text"] = text if isinstance(text, str) else "".join(text)

        ret = {
            "data_type": "t2i",
            "dtype": "t2i",
            "n_samples": 1,
            "image": image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "image_mask": image_mask,
            "attention_mask": attention_mask,
            "freqs_cos": None,
            "freqs_sin": None,
        }

        if iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": iw_ih_scatter_index,     # (2)
                "iw_ih_scatter_src": image_token_shape_wh,      # (2)
            })
        if timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": timestep_scatter_index,   # (1)
            })

        return ret

    def collate_fn(self, batch):
        data_type = batch[0]["data_type"]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])
        
        image = torch.stack([item["image"] for item in batch], dim=0)
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        target_tokens = torch.stack([item["target_tokens"] for item in batch], dim=0)
        text_mask = torch.stack([item["text_mask"] for item in batch], dim=0)
        image_mask = torch.stack([item["image_mask"] for item in batch], dim=0)
        attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)

        iw_ih_scatter_index = torch.stack([item["iw_ih_scatter_index"] for item in batch], dim=0)
        iw_ih_scatter_src = torch.stack([item["iw_ih_scatter_src"] for item in batch], dim=0)
        timestep_scatter_index = torch.stack([item["timestep_scatter_index"] for item in batch], dim=0)

        return {
            "data_type": data_type,
            "dtype": dtype,
            "n_samples": n_samples,
            "image": image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "image_mask": image_mask,
            "attention_mask": attention_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "timestep_scatter_index": timestep_scatter_index,
        }


class TextArrowStream(IndexDataset):
    def __init__(
            self,
            args,
            index_file,
            t2t_text_token_length,
            tokenizer_name,
            post_kwargs=None,
            logger=None,
            tokenizer=None,
            index_kwargs=None,
            template="pretrain",
            conv_format="hunyuan-gemini-alpha",
            dataset_tag=None,
    ):
        super().__init__(index_file, logger=logger)
        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")

        self.args = args
        self.t2t_text_token_length = t2t_text_token_length

        if index_kwargs is None:
            index_kwargs = {}
        
        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
        )

        self.logger.info(f"    (LM) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    (LM) Using {self.index_manager}")

        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer: TokenizerWrapper = tokenizer
        self.eos_token = self.tokenizer.tokenizer.eos_token
        assert isinstance(self.eos_token, str), \
            f"eos_token should be a string, but got {self.eos_token}({type(self.eos_token)})"

        assert template in ["pretrain", "instruct"], f"Unsupported template: {template}"
        if template == "instruct":
            assert conv_format, f"conv_format should be provided for instruct template."
        self.template = template
        self.conv_format = conv_format
        self.default_conv = get_conversation_template(self.conv_format)
        roles = self.default_conv.roles
        # {"User": 3, "Assistant": 3, "System": 0}
        self.role_offset = {
            role: len(self.tokenizer.encode_text(self.default_conv.get_role_prefix(role)))
            for role in roles
        }
        self.role_offset["System"] = 0

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        pass

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_pretrain_text(self, ind) -> str:
        if "caption_v2" in self.index_manager.get_columns(ind):
            try:
                caption = self.index_manager.get_attribute(ind, "caption_v2")
                text = ''.join(self.caption_processor.caption_aug(caption))
            except Exception as e:
                self.handle_exception_message(self.get_pretrain_text, e)
                text = ""
        else:
            text = self.index_manager.get_attribute(ind, "text")
            text = str(text).strip()
        return text

    def format_message_list(self, message_list, return_type="list", system_prompt=None):
        conversation = get_conversation_template(self.conv_format)
        conversation.system_message = "" if system_prompt is None else system_prompt
        for msg in message_list:
            # User message
            conversation.add_message(conversation.roles[0], msg["User"].strip())
            # Assistant message, reasoning msg.
            assistant_msg = '<answer>' + msg["Assistant"] + '</answer>'
            if 'reasoning' in msg:
                assistant_msg = '<think>' + msg["reasoning"] + '</think>' + assistant_msg
            conversation.add_message(conversation.roles[1], assistant_msg)
        text = conversation.get_prompt(return_type=return_type, add_system=system_prompt is not None)
        return text, conversation

    def get_instruct_text(self, ind) -> Tuple[List[str], List[Tuple[bool, int]]]:
        messages = None
        for _ in range(5):
            try:
                messages = self.index_manager.get_attribute(ind, "conversations")
                has_system_prompt_col = "document" in self.index_manager.get_columns(ind)
                system_prompt = self.index_manager.get_attribute(ind, "document") if has_system_prompt_col else None
                break
            except Exception as e:
                self.handle_exception_message(self.get_instruct_text, e)
                new_ind = self.index_manager.random_dindex(ind)
                print(f"Error with index={ind}, trying new index={new_ind}")
                ind = new_ind
        if messages is None:
            raise ValueError(f"Failed to get messages for index={ind}")

        # [("role", "role: message"), ...]
        role_texts, conv = self.format_message_list(
            messages, return_type="list", system_prompt=system_prompt)
        # Extract messages as a List[str]
        texts = [text for role, text in role_texts]
        # Only compute loss on the assistant's messages. offset=3+1 for skipping the `Assistant: <answer>` prefix.
        # [(is_assistant_message, offset), ...]
        text_mask_sections = [(role == conv.roles[1], self.role_offset[role] + 1) for role, text in role_texts]
        return texts, text_mask_sections

    def get_text(self, indices) -> Tuple[List[str], Union[None, List[Tuple[bool, int]]]]:
        if self.template == "pretrain":
            text_list = [self.get_pretrain_text(ind) for ind in indices]
            text = self.eos_token.join(text_list)
            return [text], None
        elif self.template == "instruct":
            assert len(indices) == 1, f"Only one index is supported for instruct template, but got {len(indices)}"
            return self.get_instruct_text(indices[0])
        else:
            raise NotImplementedError(f"Unsupported template: {self.template}")

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        texts, text_mask_sections = self.get_text(indices)
        tokens, _, text_mask = self.tokenizer.encode_lm(
            *texts,
            max_token_length=self.t2t_text_token_length,
            return_text_mask=True,
            text_mask_sections=text_mask_sections,
        )
        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100

        ret = {
            "data_type": "lm",
            "dtype": self.dataset_tag,
            "n_samples": len(indices),
            "text": texts if isinstance(texts, str) else ''.join(texts),
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
        }
        return ret

    def collate_fn(self, batch):
        data_type = batch[0]["data_type"]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])
        
        text = [item["text"] for item in batch]
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        target_tokens = torch.stack([item["target_tokens"] for item in batch], dim=0)
        text_mask = torch.stack([item["text_mask"] for item in batch], dim=0)

        return {
            "data_type": data_type,
            "dtype": dtype,
            "n_samples": n_samples,
            "text": text,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
        }

class InstructionTuningGeminiAlphaArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        shadow_file_fn=None,
    ):
        super().__init__(
            index_file,
            multireso,
            index_kwargs.get("batch_size", 1),
            index_kwargs.get("world_size", 1),
            logger,
            debug
        )
        self.args = args
        self.training_image_size = to_2tuple(training_image_size)

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.add_timestep_token = self.args.add_timestep_token
        self.use_front_boi_token = self.args.use_front_boi_token

        self.index_kwargs = index_kwargs
        self.shadow_file_fn = shadow_file_fn or {}
        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
            shadow_file_fn=self.shadow_file_fn,
        )

        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    Using {self.index_manager}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.text_token_length = text_token_length
        self.image_token_length = image_token_length
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.trans_type = self.vae_meta_info["trans_type"]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = self.args.patch_size

        if self.trans_type == "-11":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ]
            )
        elif self.trans_type == "01":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                ]
            )
        else:
            raise ValueError("Invalid trans_type: {}".format(self.trans_type))

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_binary_image(self, index, column, shadow=None):
        try:
            img_bytes = self.index_manager.get_attribute(index, column, shadow=shadow)
            image_bytes = io.BytesIO(img_bytes)
            image_bytes.seek(0)
            ret = Image.open(image_bytes).convert("RGB")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"{type(e)}: {e}")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_cos_url_image(self, index, column, shadow=None):
        try:
            ret = self.get_image_from_url_cos(index, column=column, cut_param=False, shadow=shadow)
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"{type(e)}: {e}")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_image_with_size(self, index, column, shadow=None):
        if "image" in column or "bytes" in column or "binary" in column:
            image, image_flag = self.get_binary_image(index, column=column, shadow=shadow)
        elif "url_cos" in column or "path" in column:
            image, image_flag = self.get_cos_url_image(index, column=column, shadow=shadow)
        else:
            raise ValueError(f"Invalid column: {column}")

        origin_size = image.size  # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs, image_flag


    @staticmethod
    def collate_fn(batch):
        data_type = batch[0]["data_type"]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])
        src_images = [item["src_image"] for item in batch]
        if src_images[0] is None:
            src_images = None
        else:
            try:
                src_images = torch.stack(src_images)
            except Exception as e:
                # here, src_images is a list of a list of tensors, the length of the first list is batch_size, the length of the second list is the number of source images of the i-th sample, and each tensor is c x h x w
                pass

        can_stack = isinstance(src_images, torch.Tensor) or src_images is None
        

        tgt_image = torch.stack([item["tgt_image"] for item in batch])
        tokens = torch.stack([item["tokens"] for item in batch])
        target_tokens = torch.stack([item["target_tokens"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        src_image_mask = torch.stack([item["src_image_mask"] for item in batch])
        tgt_image_mask = torch.stack([item["tgt_image_mask"] for item in batch])

        iw_ih_scatter_index = [item["iw_ih_scatter_index"] for item in batch]
        if can_stack:
            iw_ih_scatter_index = torch.stack(iw_ih_scatter_index)

        iw_ih_scatter_src = [item["iw_ih_scatter_src"] for item in batch]
        if can_stack:
            iw_ih_scatter_src = torch.stack(iw_ih_scatter_src)

        timestep_scatter_index = [item["timestep_scatter_index"] for item in batch]
        if can_stack:
            timestep_scatter_index = torch.stack(timestep_scatter_index)

        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        freqs_cos = torch.stack([item["freqs_cos"] for item in batch]) if batch[0].get("freqs_cos") is not None else None
        freqs_sin = torch.stack([item["freqs_sin"] for item in batch]) if batch[0].get("freqs_sin") is not None else None

        if "src_face_embedding" in batch[0].keys():
            src_face_embedding = torch.stack([item["src_face_embedding"] for item in batch])
        else:
            src_face_embedding = None

        return {
            "data_type": data_type,
            "dtype": dtype,
            "n_samples": n_samples,
            "src_images": src_images,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "timestep_scatter_index": timestep_scatter_index,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
            "src_face_embedding": src_face_embedding,
        }



class InpaintingGeminiAlphaArrowStream(InstructionTuningGeminiAlphaArrowStream):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        dataset_tag=None,
    ):
        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")

        shadow_file_fn = {}

        self.img_col = index_kwargs.get(f"{dataset_tag}_img_col", None)
        if self.img_col is None:
            raise ValueError(f"Missing `{dataset_tag}_img_col` in `index_kwargs`")
        if '@' in self.img_col:
            self.img_col, img_col_arrow_suffix = self.img_col.split('@')
            self.img_col_shadow = 'image'
            shadow_file_fn[self.img_col_shadow] = partial(arrow_mapper, suffix=f"_{img_col_arrow_suffix}")
        else:
            self.img_col_shadow = None

        super().__init__(
            args,
            index_file,
            training_image_size,
            image_token_length,
            text_token_length,
            tokenizer_name,
            multireso,
            index_kwargs,
            post_kwargs,
            debug,
            logger,
            tokenizer,
            shadow_file_fn,
        )

        self.uncond_p = index_kwargs.get(f"{dataset_tag}_uncond_p", 0.0)

        self.caption_rate = index_kwargs.get(f"{dataset_tag}_caption_rate", 0.0)
        
        self.text_col = index_kwargs.get(f"{dataset_tag}_text_col", None)
        if self.caption_rate < 1.0 and self.text_col is None:
            raise ValueError(f"Missing `{dataset_tag}_text_col` in `index_kwargs`")

        self.caption_col = index_kwargs.get(f"{dataset_tag}_caption_col", None)
        if self.caption_rate > 0.0 and self.caption_col is None:
            raise ValueError(f"Missing `{dataset_tag}_caption_col` in `index_kwargs`")
        
        self.caption_sample_ratio = index_kwargs.get(f"{dataset_tag}_caption_sample_ratio", None)
        if self.caption_sample_ratio is not None:
            self.caption_processor = index_kwargs.get(f"{dataset_tag}_caption_processor", None)
            self.caption_processor_kwargs = index_kwargs.get(f"{dataset_tag}_caption_processor_kwargs", None)
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=default(self.caption_processor, 'caption_process'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=self.caption_processor_kwargs,
            )
        else:
            self.use_structural_caption = False
        
        self.pred_text_boi_eos = self.index_kwargs.get(f"{dataset_tag}_pred_text_boi_eos", None)
        if self.pred_text_boi_eos is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_text_boi_eos` in `index_kwargs`")
        self.pred_boi_mode = self.index_kwargs.get(f"{dataset_tag}_pred_boi_mode", None)
        if self.pred_boi_mode is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_boi_mode` in `index_kwargs`")

        self.mask_generator_probs = index_kwargs.get(f"{dataset_tag}_mask_generator_probs", None)
        if self.mask_generator_probs is None:
            raise ValueError(f"Missing `{dataset_tag}_mask_generator_probs` in `index_kwargs`")
        self.mask_provider = MixedMaskGenerator(mask_generator_probs=self.mask_generator_probs)

    def get_instruction(self, ind):
        return inpainting_instructions[(np.random.randint(0, 123456) + ind * 123) % len(inpainting_instructions)]

    def get_content_prompt(self, ind):
        try:
            if self.caption_rate > 0.0 and random.random() < self.caption_rate:
                content_prompt = self.index_manager.get_attribute(ind, self.caption_col)
                if self.use_structural_caption:
                    content_prompt = self.caption_aug.caption_aug(content_prompt)
            else:
                content_prompt = self.index_manager.get_attribute(ind, self.text_col)
        except Exception as e:
            self.handle_exception_message(self.get_content_prompt, e)
            content_prompt = ""
        content_prompt = str(content_prompt).strip()

        # Remove meaningless characters
        content_prompt = content_prompt.replace("\\N", "").strip("，,")

        return content_prompt

    def __getitem__(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        content_prompt = self.get_content_prompt(index)

        instruction_list = [
            "User: ",
            instruction.strip(),
            " " + content_prompt.strip(),
            "\n\n",
            "Assistant: ",
        ]
        uncond_enabled = [
            False,
            False,
            True,
            False,
            False,
        ]

        src_image, _, src_image_flag = self.get_image_with_size(index, self.img_col, shadow=self.img_col_shadow)
        # TODO(ckczzjzhang) handle exceptions when token_flag is gray

        src_h, src_w = src_image.shape[1], src_image.shape[2]
        assert src_h % (self.downsample_factor[0] * self.patch_size) == 0 and src_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({src_h} x {src_h}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_src_image_token_length = (src_h // (self.downsample_factor[0] * self.patch_size)) * (src_w // (self.downsample_factor[1] * self.patch_size))

        # (1, h, w)
        mask = self.mask_provider(h=src_h, w=src_w)
        # mask with gray
        masked_src_image = src_image * (1.0 - mask)
        tgt_image = src_image

        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion(
            *instruction_list,
            image_token_length=actual_src_image_token_length,
            src_image_token_lengths=[actual_src_image_token_length],
            max_text_token_length=self.text_token_length + 1, 
            max_image_token_length=self.image_token_length,
            uncond_enabled=uncond_enabled,
            uncond_p=self.uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
            pred_boi_mode=self.pred_boi_mode,
        )
        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100


        # 1 for shift
        n_tokens = tokens.shape[0] - 1
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        image_mask_1 = src_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(0, 1)
        image_mask_3 = tgt_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_4 = image_mask_3.transpose(0, 1)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (image_mask_3.bool() & image_mask_4.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(0)

        return {
            "data_type": "ti2i",
            "dtype": self.dataset_tag,
            "n_samples": 1,
            "src_image": masked_src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.tensor([src_w, src_h, src_w, src_h], dtype=torch.long),
            "timestep_scatter_index": timestep_scatter_index,
            "attention_mask": attention_mask,
        }
        

class EditingGeminiAlphaArrowStream(InstructionTuningGeminiAlphaArrowStream):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        dataset_tag=None,
    ):
        super().__init__(
            args,
            index_file,
            training_image_size,
            image_token_length,
            text_token_length,
            tokenizer_name,
            multireso,
            index_kwargs,
            post_kwargs,
            debug,
            logger,
            tokenizer,
        )

        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")

        self.uncond_p = index_kwargs.get(f"{dataset_tag}_uncond_p", 0.0)

        self.instruction_col = index_kwargs.get(f"{dataset_tag}_instruction_col", None)
        if self.instruction_col is None:
            raise ValueError(f"Missing `{dataset_tag}_instruction_col` in `index_kwargs`")

        self.eiditing_src_img_col = self.index_kwargs.get(f"{dataset_tag}_src_img_col", None)
        if self.eiditing_src_img_col is None:
            raise ValueError(f"Missing `{dataset_tag}_src_img_col` in `index_kwargs`")
        self.eiditing_tgt_img_col = self.index_kwargs.get(f"{dataset_tag}_tgt_img_col", None)
        if self.eiditing_tgt_img_col is None:
            raise ValueError(f"Missing `{dataset_tag}_tgt_img_col` in `index_kwargs`")

        self.pred_text_boi_eos = self.index_kwargs.get(f"{dataset_tag}_pred_text_boi_eos", None)
        if self.pred_text_boi_eos is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_text_boi_eos` in `index_kwargs`")
        self.pred_boi_mode = self.index_kwargs.get(f"{dataset_tag}_pred_boi_mode", None)
        if self.pred_boi_mode is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_boi_mode` in `index_kwargs`")

    def get_instruction(self, ind):
        return editing_instructions[(np.random.randint(0, 123456) + ind * 123) % len(editing_instructions)]

    def get_instruction(self, ind):
        try:
            instruction = self.index_manager.get_attribute(ind, column=self.instruction_col)
            if isinstance(instruction, list):
                instruction = random.choice(instruction)
        except Exception as e:
            self.handle_exception_message(self.get_instruction, e)
            instruction = ""
        instruction = str(instruction).strip()

        # Remove meaningless characters
        instruction = instruction.replace("\\N", "").strip("，,")

        return instruction

    def __getitem__(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        instruction = self.get_instruction(index)

        instruction_list = [
            "User: ",
            instruction.strip(),
            " " + instruction.strip(),
            "\n\n",
            "Assistant: ",
        ]
        uncond_enabled = [
            False,
            False,
            True,
            False,
            False,
        ]

        src_image, _, src_image_flag = self.get_image_with_size(index, self.eiditing_src_img_col)
        tgt_image, _, tgt_image_flag = self.get_image_with_size(index, self.eiditing_tgt_img_col)

        # TODO(ckczzjzhang) handle exceptions when image_flag is gray

        src_h, src_w = src_image.shape[1], src_image.shape[2]
        assert src_h % (self.downsample_factor[0] * self.patch_size) == 0 and src_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({src_h} x {src_h}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_src_image_token_length = (src_h // (self.downsample_factor[0] * self.patch_size)) * (src_w // (self.downsample_factor[1] * self.patch_size))

        tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]
        assert tgt_h % (self.downsample_factor[0] * self.patch_size) == 0 and tgt_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({tgt_h} x {tgt_w}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_tgt_image_token_length = (tgt_h // (self.downsample_factor[0] * self.patch_size)) * (tgt_w // (self.downsample_factor[1] * self.patch_size))

        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion(
            *instruction_list,
            image_token_length=actual_tgt_image_token_length,
            src_image_token_lengths=[actual_src_image_token_length],
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_enabled=uncond_enabled,
            uncond_p=self.uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
            pred_boi_mode=self.pred_boi_mode,
        )
        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100

        # 1 for shift
        n_tokens = tokens.shape[0] - 1
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        image_mask_1 = src_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(0, 1)
        image_mask_3 = tgt_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_4 = image_mask_3.transpose(0, 1)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (image_mask_3.bool() & image_mask_4.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(0)

        return {
            "data_type": "ti2i",
            "dtype": self.dataset_tag,
            "n_samples": 1,
            "src_image": src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.tensor([src_w, src_h, tgt_w, tgt_h], dtype=torch.long),
            "timestep_scatter_index": timestep_scatter_index,
            "attention_mask": attention_mask,
        }


class SubjectDrivenGeminiAlphaArrowStream(InstructionTuningGeminiAlphaArrowStream):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        dataset_tag=None,
    ):
        super().__init__(
            args,
            index_file,
            training_image_size,
            image_token_length,
            text_token_length,
            tokenizer_name,
            multireso,
            index_kwargs,
            post_kwargs,
            debug,
            logger,
            tokenizer,
        )

        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")

        self.img_col = index_kwargs.get(f"{dataset_tag}_img_col", None)
        if self.img_col is None:
            raise ValueError(f"Missing `{dataset_tag}_img_col` in `index_kwargs`")

        self.content_prompt_col = index_kwargs.get(f"{dataset_tag}_content_prompt_col", None)
        if self.content_prompt_col is None:
            raise ValueError(f"Missing `{dataset_tag}_content_prompt_col` in `index_kwargs`")

        self.uncond_p = self.index_kwargs.get(f"{dataset_tag}_uncond_p", 0.0)

        self.pred_text_boi_eos = self.index_kwargs.get(f"{dataset_tag}_pred_text_boi_eos", None)
        if self.pred_text_boi_eos is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_text_boi_eos` in `index_kwargs`")
        self.pred_boi_mode = self.index_kwargs.get(f"{dataset_tag}_pred_boi_mode", None)
        if self.pred_boi_mode is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_boi_mode` in `index_kwargs`")

    def get_instruction(self, ind):
        return subject_driven_instructions[(np.random.randint(0, 123456) + ind * 123) % len(subject_driven_instructions)]

    def get_content_prompt(self, ind):
        try:
            content_prompt = self.index_manager.get_attribute(ind, column=self.content_prompt_col)
        except Exception as e:
            self.handle_exception_message(self.get_content_prompt, e)
            content_prompt = ""
        content_prompt = str(content_prompt).strip()

        # Remove meaningless characters
        content_prompt = content_prompt.replace("\\N", "").strip("，,")

        return content_prompt

    def parse_annotations(self, index, image_t):
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
        for anno in valid_annotations[:3]:  # No more than 3 reference objects
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

    def __getitem__(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        content_prompt = self.get_content_prompt(index)

        instruction_list = [
            "User: ",
            instruction.strip(),
            " " + content_prompt.strip(),
            "\n\n",
            "Assistant: ",
        ]
        uncond_enabled = [
            False,
            False,
            True,
            False,
            False,
        ]

        # Get target image
        image, _, image_flag = self.get_image_with_size(index, self.img_col)
        # TODO(ckczzjzhang) handle exceptions when image_flag is gray

        # The ref_objects should be extracted before resize, but the images of this dataset are all 1024x1024, and they would not be resized.
        ref_objects = self.parse_annotations(index, image)

        tgt_h, tgt_w = image.shape[1], image.shape[2]
        assert tgt_h % (self.downsample_factor[0] * self.patch_size) == 0 and tgt_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({tgt_h} x {tgt_w}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_tgt_image_token_length = (tgt_h // (self.downsample_factor[0] * self.patch_size)) * (tgt_w // (self.downsample_factor[1] * self.patch_size))

        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion(
            *instruction_list,
            image_token_length=actual_tgt_image_token_length,
            src_image_token_lengths=[
                (ref_obj.size(1) // (self.downsample_factor[0] * self.patch_size)) * (ref_obj.size(2) // (self.downsample_factor[1] * self.patch_size))
                for ref_obj in ref_objects
            ],
            max_text_token_length=self.text_token_length + 1,
            # max_image_token_length=self.image_token_length,
            max_total_token_length=self.text_token_length + 1 + self.image_token_length * 4,
            uncond_enabled=uncond_enabled,
            uncond_p=self.uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
            pred_boi_mode=self.pred_boi_mode,
        )
        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100

        start_pos = torch.where(tokens == self.tokenizer.special_token_map["<img>"])[0][0].item()

        src_token_slices = []
        for ref_obj in ref_objects:
            th_ = ref_obj.size(1) // (self.downsample_factor[0] * self.patch_size)
            tw_ = ref_obj.size(2) // (self.downsample_factor[1] * self.patch_size)
            src_token_slices.append(slice(start_pos, start_pos + th_ * tw_))
            start_pos += th_ * tw_ + 2 + (2 if self.add_iw_ih_token else 0) + (1 if self.add_timestep_token else 0)
        tgt_token_slice = slice(start_pos, start_pos + actual_tgt_image_token_length)
        attention_mask = create_attention_mask_general(
            sequence=tokens[None, :-1],  # 1 for shift
            pad_id=None,
            slice_s_image_ranges=src_token_slices + [tgt_token_slice],
            mask_pad=False,
            return_inverse_mask=False,
        )[0]

        iw_ih_scatter_src = []
        for ref_obj in ref_objects:
            iw_ih_scatter_src.append(ref_obj.size(2))
            iw_ih_scatter_src.append(ref_obj.size(1))
        iw_ih_scatter_src.append(tgt_w)
        iw_ih_scatter_src.append(tgt_h)
        iw_ih_scatter_src = torch.tensor(iw_ih_scatter_src, dtype=torch.long)

        return {
            "data_type": "ti2i",
            "dtype": self.dataset_tag,
            "n_samples": 1,
            "src_image": ref_objects,
            "tgt_image": image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "timestep_scatter_index": timestep_scatter_index,
            "attention_mask": attention_mask,
        }


class FaceIDPreserveGeminiAlphaArrowStream(InstructionTuningGeminiAlphaArrowStream):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=1024,
        image_token_length=1024,
        text_token_length=256,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
        dataset_tag=None,
    ):
        self.dataset_tag = dataset_tag
        if self.dataset_tag is None:
            raise ValueError(f"Missing dataset_tag")

        self.face_analysis_arrow_suffix = index_kwargs.get(f"{dataset_tag}_face_analysis_arrow_suffix", None)
        if self.face_analysis_arrow_suffix is None:
            raise ValueError(f"Missing `{dataset_tag}_face_analysis_arrow_suffix` in `index_kwargs`")

        shadow_file_fn = {"face_analysis": partial(arrow_mapper, suffix=self.face_analysis_arrow_suffix)}

        super().__init__(
            args,
            index_file,
            training_image_size,
            image_token_length,
            text_token_length,
            tokenizer_name,
            multireso,
            index_kwargs,
            post_kwargs,
            debug,
            logger,
            tokenizer,
            shadow_file_fn,
        )

        self.instruction_uncond_p = index_kwargs.get(f"{dataset_tag}_instruction_uncond_p", 0.0)
        self.face_embedding_uncond_p = index_kwargs.get(f"{dataset_tag}_face_embedding_uncond_p", 0.0)

        # self.instruction_col = index_kwargs.get(f"{dataset_tag}_instruction_col", None)
        # if self.instruction_col is None:
        #     raise ValueError(f"Missing `{dataset_tag}_instruction_col` in `index_kwargs`")
     
        self.caption_sample_ratio = index_kwargs.get(f"{dataset_tag}_caption_sample_ratio", None)
        if self.caption_sample_ratio is not None:
            self.caption_processor = index_kwargs.get(f"{dataset_tag}_caption_processor", None)
            self.caption_processor_kwargs = index_kwargs.get(f"{dataset_tag}_caption_processor_kwargs", None)
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=default(self.caption_processor, 'caption_process'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=self.caption_processor_kwargs,
            )
        else:
            self.use_structural_caption = False


        self.src_condition_type = index_kwargs.get(f"{dataset_tag}_src_condition_type", ["vae"])
        if isinstance(self.src_condition_type, str):
            self.src_condition_type = self.src_condition_type.split("_cat_")
        else: 
            assert isinstance(self.src_condition_type, list), \
                f"src_condition_type should be a list,e.g, ['vae'], ['face_embed', 'clip'], but got {type(self.src_condition_type)}"
        self.image_token_length_clip = index_kwargs.get(f"{dataset_tag}_image_token_length_clip", None)

        self.face_index_unique = index_kwargs.get(f"{dataset_tag}_face_index_unique", True)
        self.face_bof_eof = index_kwargs.get(f"{dataset_tag}_face_bof_eof", False)
        self.resampler_token_length = index_kwargs.get(f"{dataset_tag}_resampler_token_length", None)

        self.pred_text_boi_eos = self.index_kwargs.get(f"{dataset_tag}_pred_text_boi_eos", None)
        if self.pred_text_boi_eos is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_text_boi_eos` in `index_kwargs`")
        self.pred_boi_mode = self.index_kwargs.get(f"{dataset_tag}_pred_boi_mode", None)
        if self.pred_boi_mode is None:
            raise ValueError(f"Missing `{dataset_tag}_pred_boi_mode` in `index_kwargs`")

    def get_instruction(self, ind):
        return face_id_instructions[(np.random.randint(0, 123456) + ind * 123) % len(face_id_instructions)]

    def get_content_prompt(self, ind, caption_col):
        try:
            content_prompt = self.index_manager.get_attribute(ind, caption_col)
            if self.use_structural_caption:
                content_prompt = self.caption_aug.caption_aug(content_prompt)
        except Exception as e:
            self.handle_exception_message(self.get_content_prompt, e)
            content_prompt = ""
        content_prompt = str(content_prompt).strip()

        # Remove meaningless characters
        content_prompt = content_prompt.replace("\\N", "").strip("，,")

        return content_prompt

    @staticmethod
    def get_face_info(face_analysis, face_image_torch, logger=None):
        """
            get the face info from the face image
        """
        face_image_np = face_image_torch.cpu().numpy()
        face_image_np = einops.rearrange(face_image_np, 'c h w -> h w c')
        face_image_np = (face_image_np + 1) * 127.5
        face_image_np = face_image_np.astype(np.uint8) # h w c uint8 range: 0, 255
        face_analysis_input = cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR)
        face_info = face_analysis.get(face_analysis_input)
        return face_info

    @staticmethod
    def crop_face_image_torch(face_analysis, face_image_torch, max_side=[256, 256], logger=None, src_condition_type=["vae"], return_bbox=False):
        """
            crop the face area from image torch tensor, resize to max_side=256, and return the face image torch tensor

        Args:
            face_image_torch (torch tensor): [3, sh, sw] float32 range: -1, 1
            max_side (list of int tw, th, optional): For tuple [ (tw, sw), (th, sh)], resize side with smaller t/s to corresponding axis, pad larger ratio axis to max_side
        Returns:
            torch tensor: [3, tw, th] float32 range: -1, 1
        """
        face_info = FaceIDPreserveGeminiAlphaArrowStream.get_face_info(face_analysis, face_image_torch, logger)
        

        # Handle multiple faces, small face, and negative face_bbox and no face

        if len(face_info) >= 1:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            face_embedding = face_info.get('embedding', None)
            if isinstance(face_embedding, np.ndarray):
                face_embedding = torch.from_numpy(face_embedding)
            face_bbox = face_info['bbox']
            # if area of face is too small, skip
            # TODO: 10% probability to get negative face_bbox coordinates; can be improved later
            if face_bbox[2] - face_bbox[0] < 10 or face_bbox[3] - face_bbox[1] < 10:
                face_image_crop_torch = face_image_torch
            else:
                face_image_crop_torch = face_image_torch[:, int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
            
            n, h, w = face_image_crop_torch.shape
            if h < 10 or w < 10:
                logger.info(f"WARNING: face bbox is {face_bbox}")
                logger.info(f"WARNING: face area {face_image_crop_torch.shape} is too small, skip face crop; set as original image {face_image_torch.shape}")
                face_image_crop_torch = face_image_torch
            
        else:
            logger.info(f"WARNING: no face detected, skip face crop")
            face_image_crop_torch = face_image_torch
            face_embedding = None
            face_bbox = None
        if "face_embed" in src_condition_type:
            if face_embedding is None:
                logger.info(f"WARNING: {src_condition_type=}, but face embedding is None, set as zero")
                face_embedding = torch.zeros([512], dtype=torch.float32)
                face_bbox = torch.zeros([4], dtype=torch.float32)
        try:
            face_image_crop_resize_torch = FaceIDPreserveGeminiAlphaArrowStream.resize_and_pad(face_image_crop_torch[None, ...], max_side=max_side, pad_to_max_side=True, logger=logger)[0, ...]
        except Exception as e:
            logger.info(f"Error: input face_image_torch shape: {face_image_torch.shape}; face_image_crop_torch shape: {face_image_crop_torch.shape}")
            raise e
        if not return_bbox:
            return face_image_crop_resize_torch, face_embedding
        else:
            return face_image_crop_resize_torch, face_embedding, face_bbox

    @staticmethod
    def resize_and_pad( input_image, max_side=[256, 256], size=None, 
                pad_to_max_side=True, base_pixel_number=1, logger=None):
        """
        resize torch tensor [n, c, h, w] range(-1, 1), resize long side to max_side, pad short side to max_side
        Args:
            input_image (torch tensor): [n, c, h, w] float number
            max_side (list of int tw, th, optional): For tuple [ (tw, sw), (th, sh)], resize side with smaller t/s to corresponding axis, pad larger ratio axis to max_side
            pad_to_max_side (bool, optional): pad short side to max_side
            base_pixel_number (int, optional): resize to base_pixel_number
        Returns:
            torch tensor: [n, c, max_side, w* (max_side/h)] or [n, c, h * (max_side/w), max_side] -1, 1
            
        """
        tw, th = max_side
        n, c, h, w = input_image.shape
        if size is not None:
            h_resize_new, w_resize_new = size
        else:
            # Calculate the ratio to resize the image so the larger side is max_side
            ratio = min(tw / w, th / h)
            w, h = round(ratio * w), round(ratio * h)
            w_resize_new = (w // base_pixel_number) * base_pixel_number
            h_resize_new = (h // base_pixel_number) * base_pixel_number
        
        input_image = F.interpolate(input_image, size=(h_resize_new, w_resize_new), mode='bicubic')
        input_image = input_image.clamp(-1, 1)

        if pad_to_max_side:
            res = torch.zeros([n, c, th, tw], dtype=torch.float32)
            offset_x = (tw - w_resize_new) // 2
            offset_y = (th - h_resize_new) // 2
            res[:, :, offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = input_image
            input_image = res
        return input_image

    def get_image_with_size_resize_and_pad(self, index, column, crop_bbox=None):
        image, image_flag = self.get_binary_image(index, column=column)
        if crop_bbox is not None:
            image = image.crop(crop_bbox)
            crop_coords_xy = (crop_bbox[0], crop_bbox[1])
        else:
            crop_coords_xy = (0,0)
        origin_size = image.size  # (w_ori, h_ori)
        target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can reuse resize_and_pad in index kit
        image_tensor = self.pil_image_to_tensor(image)
        image_tensor = self.resize_and_pad(
            image_tensor[None, ...], 
            max_side=target_size, 
            pad_to_max_side=True, 
            logger=self.logger
         )[0, ...]

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": crop_coords_xy,
        }
        return image_tensor, kwargs, image_flag

    def get_src_tgt_index_from_count(self, ind, unique=True, valid_bbox_index=None):
        if valid_bbox_index is None:
            count = self.index_manager.get_attribute(ind, "count")
            valid_bbox_index = range(count)
        if unique:
            # Choose 2 different images from count using random.sample
            indices = random.sample(valid_bbox_index, 2)
        else:
            indices = random.choices(valid_bbox_index, k=2)
        return indices

    def get_id_face_analysis(self, ind):
        try:
            bbox = self.index_manager.get_attribute(ind, "bbox", shadow="face_analysis")
            embedding = self.index_manager.get_attribute(ind, "embedding", shadow="face_analysis")
            bbox_np = np.frombuffer(bbox, dtype=np.float32)
            embedding_np = np.frombuffer(embedding, dtype=np.float32)
            embedding_np = np.copy(embedding_np)
            bbox_np = bbox_np.reshape(-1, 4)
            embedding_np = embedding_np.reshape(-1, 512)
            assert bbox_np.shape[0] == embedding_np.shape[0], f"bbox and embedding shape mismatch: {bbox_np.shape[0]} != {embedding_np.shape[0]}"
        except Exception as e:
            self.handle_exception_message(self.get_id_face_analysis, e)
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
        assert count == self.index_manager.get_attribute(index, f"count"), f'bbox.shape[0] should be equal with count index_manager, but  {count}!={self.index_manager.get_attribute(index, f"count")}'
        
        image_width_np_array = np.array([self.index_manager.get_attribute(index, f"width_{image_count_i}") for image_count_i in range(count)])
        image_height_np_array = np.array([self.index_manager.get_attribute(index, f"height_{image_count_i}") for image_count_i in range(count)])
        bbox_left_top_right_bottom_in_image_index = (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 2] <= image_width_np_array) & (bbox[:, 3] <= image_height_np_array)

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

    def get_id_src_embedding_tgt_image(self, index):
        """
            (1) get bbox from     self.index_manager.get_attribute(index, column);
            (2) if not all bbox are zero: remove zero bbox in the bbox list, which means no / multiple faces detected;
            (3) sample src_index, tgt_index from the left of the bbox list;
            (4) use src_index face embedding as src_face_embedding; resize and pad face in src_index image using the src_index bbox;
            (5) use tgt_index image as tgt_image;
            (6) return the data_item;
        """

        bbox, embedding = self.get_id_face_analysis(index)
        valid_bbox_index, bbox = self.get_valid_bbox_index(index, bbox)

        # (TODO) filter invalid id using index_manager filter, not in dataloader
        valid_bbox_index_sample = valid_bbox_index if len(valid_bbox_index) >= 2 else range(bbox.shape[0])
        src_index, tgt_index = self.get_src_tgt_index_from_count(index, unique=self.face_index_unique, valid_bbox_index=valid_bbox_index_sample)

        content_prompt = self.get_content_prompt(index, f"caption_{tgt_index}")
        src_face_embedding = embedding[src_index]

        src_image, src_kwargs, src_image_flag = self.get_image_with_size_resize_and_pad(
            index, f"image_{src_index}", crop_bbox=bbox[src_index] if len(valid_bbox_index) >= 2 else None)
        tgt_image, tgt_kwargs, tgt_image_flag = self.get_image_with_size(index, f"image_{tgt_index}")

        return src_image, src_face_embedding, tgt_image, content_prompt

    def __getitem__(self, index):
        if self.face_analysis_arrow_suffix is None:
            # Calculate face embedding online
            if self.face_analysis is None and self.args.get("face_crop", False) is True:
                ASSETS_BASE = os.getenv("ASSETS_BASE", "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets").rstrip('/')
                insightface_path = f"{ASSETS_BASE}/others/insightface"
                name = 'buffalo_l' # From large to small, support antelopev2, buffalo_l, buffalo_sc
                allowed_modules = ['detection']
                if "face_embed" in self.src_condition_type:
                    allowed_modules.append('recognition')
                self.logger.info(f"Loading face analysis, name: {name}, allowed_modules: {allowed_modules}, providers CPUExecutionProvider, from {insightface_path}")
                self.face_analysis = FaceAnalysis(name=name, root=insightface_path, allowed_modules=allowed_modules, providers=['CPUExecutionProvider'])
                self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    
            # Get instruction
            src_index, tgt_index = self.get_src_tgt_index_from_count(index, unique=self.args.get("face_index_unique", True))
            id_src_img_col = f"image_{src_index}"
            id_tgt_img_col = f"image_{tgt_index}"

            content_prompt = self.get_content_prompt(index, f"caption_{tgt_index}")
            # TODO (chenyangqi) a better strategy is to first crop the face in original image, then resize to max_side
            src_image, src_kwargs, src_image_flag = self.get_image_with_size(index, id_src_img_col)
            tgt_image, tgt_kwargs, tgt_image_flag = self.get_image_with_size(index, id_tgt_img_col)

            if self.args.get("face_crop", False) is True:
                src_image, src_face_embedding = self.crop_face_image_torch(
                    self.face_analysis, src_image, 
                    max_side=src_kwargs.get('target_size', [256, 256]), 
                    logger=self.logger, 
                    src_condition_type=self.src_condition_type
                )
        else:
            arrow_data_list_index = self.get_id_src_embedding_tgt_image(index)
            src_image, src_face_embedding, tgt_image, content_prompt = arrow_data_list_index

        instruction = self.get_instruction(index)
        instruction_list = [
            "User: ",
            instruction.strip(),
            " " + content_prompt.strip(),
            "\n\n",
            "Assistant: ",
        ]
        uncond_enabled = [
            False,
            False,
            True,
            False,
            False,
        ]

        src_h, src_w = src_image.shape[1], src_image.shape[2]
        tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]
        actual_tgt_image_token_length = self.tokenizer.get_actual_image_token_length(tgt_image, self.vae_meta_info, patch_size=self.patch_size)

        actual_src_image_token_length_vae = self.tokenizer.get_actual_image_token_length(src_image, self.vae_meta_info, patch_size=self.patch_size)
        if "clip" in self.src_condition_type:
            assert self.clip_meta_info is not None, "clip_meta_info is None, but src_condition_type contains clip"
            actual_src_image_token_length_clip = self.tokenizer.get_actual_image_token_length(src_image, self.clip_meta_info, patch_size=self.args.get("patch_size_clip", 1))
        else:
            actual_src_image_token_length_clip = None

        # Loop over self.src_condition_type, and get the actual_src_image_token_length and max_image_token_length
        src_condition_lengths_dict = self.tokenizer.prepare_src_condition_lengths(
            actual_src_image_token_length_vae,
            self.image_token_length,
            actual_src_image_token_length_clip,
            self.image_token_length_clip,
            self.src_condition_type,
            self.face_bof_eof,
            self.resampler_token_length,
        )

        # for target image
        src_condition_lengths_dict["max_image_token_length_list"].append(self.image_token_length)

        
        do_uncond_drop_face = (self.face_embedding_uncond_p is not None) and (random.random() < self.face_embedding_uncond_p)
        if do_uncond_drop_face:
            if src_face_embedding is not None:
                src_face_embedding = np.zeros_like(src_face_embedding)
            if src_image is not None:
                src_image = torch.zeros_like(src_image)
        
        # faceid also
        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion_faceid(
            *instruction_list,
            image_token_length=actual_tgt_image_token_length,
            src_image_token_lengths=src_condition_lengths_dict["actual_src_image_token_length_list"],
            src_face_token_lengths=src_condition_lengths_dict["actual_src_face_token_length_list"],
            max_text_token_length=self.text_token_length + 1, 
            max_image_token_length=src_condition_lengths_dict["max_image_token_length_list"],
            max_face_token_length=src_condition_lengths_dict["max_face_token_length_list"],
            uncond_enabled=uncond_enabled,
            uncond_p=self.instruction_uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
            pred_boi_mode=self.pred_boi_mode,
        )

        target_tokens = tokens.clone()
        target_tokens[text_mask == 0.0] = -100

        # 1 for shift
        n_tokens = tokens.shape[0] - 1
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        image_mask_1 = src_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_2 = image_mask_1.transpose(0, 1)
        image_mask_3 = tgt_image_mask[:-1].view(1, n_tokens).repeat(n_tokens, 1)
        image_mask_4 = image_mask_3.transpose(0, 1)
        attention_mask = causal_mask | (image_mask_1.bool() & image_mask_2.bool()) | (image_mask_3.bool() & image_mask_4.bool())
        # unsqueeze for attention head dim
        attention_mask = attention_mask.unsqueeze(0)

        actual_src_image_token_length_list = src_condition_lengths_dict["actual_src_image_token_length_list"]
        if len(actual_src_image_token_length_list) == 0:
            # in this case, there is no src_image
            # src_image_mask will be treated as und_image_masks in MultimodalTransfusion
            src_image = None
            iw_ih_scatter_src = torch.tensor([tgt_w, tgt_h], dtype=torch.long)
        else:
            # this case is similar to inpainting and editing
            # src_image_mask will be treated as src_image_mask in MultimodalTransfusion
            iw_ih_scatter_src = torch.tensor([src_w, src_h]* len(actual_src_image_token_length_list) + [tgt_w, tgt_h], dtype=torch.long)

        return {
            "data_type": "faceid",
            "dtype": self.dataset_tag,
            "n_samples": 1,
            "src_image": src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "timestep_scatter_index": timestep_scatter_index,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "attention_mask": attention_mask,
            "src_face_embedding": torch.from_numpy(src_face_embedding),
        }
