import json
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from functools import partial

import torch
from PIL import Image
from torchvision.transforms import transforms
from index_kits import arrow_mapper

from .caption_strategy import load_caption_processor
from .caption_strategy.caption_process_v2 import CaptionAug
from .index_dataset import IndexDataset
from .instruction_template import image_captioning_short_instructions, image_captioning_long_instructions
from ..constants import VISION_ENCODER_META_INFO
from ..models.tokenizers import TokenizerWrapper
from ..utils.helpers import default, to_2tuple
from ..utils.import_utils import require_version
from ..models.tokenizers.conversation import get_conversation_template

require_version("index-kits", "0.5.0", "MultiModalUnderstandingArrowStream")


class MultiModalUnderstandingArrowStream(IndexDataset):
    def __init__(
            self,
            args,
            index_file,
            max_token_length,
            tokenizer_name,
            text_token_length=None,
            image_size=None,
            post_kwargs=None,
            logger=None,
            tokenizer=None,
            index_kwargs=None,
            pad_color=(127, 127, 127),
            dummy_number=0,
            conv_format="hunyuan-gemini-alpha",
            template="pretrain",
    ):
        super().__init__(index_file, logger=logger)
        self.args = args
        self.max_text_token_length = text_token_length
        self.max_token_length = max_token_length
        self.dummy_number = dummy_number

        self.vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vision_model_type]
        self.downsample_factor = to_2tuple(self.vision_encoder_meta_info["downsample_factor"])
        if image_size is None:
            self.image_size = to_2tuple(self.vision_encoder_meta_info["image_size"])
        else:
            self.image_size = to_2tuple(image_size)
        self.image_token_length = math.prod(self.image_size) // math.prod(self.downsample_factor)
        self.logger.info(f"    (MMU) Using downsample_factor: {self.downsample_factor}, image size: {self.image_size}, "
                         f"image token length: {self.image_token_length}")

        # Prepare index manager
        if index_kwargs is None:
            index_kwargs = {}
        shadow_file_fn = {}

        self.image_key = index_kwargs.get("mmu_image_key", "url_cos")
        self.image_key_shadow = None
        self.image_key_arrow_suffix = None
        if '@' in self.image_key:
            self.image_key, self.image_key_arrow_suffix = self.image_key.split('@')
            self.image_key_shadow = self.image_key
        shadow_file_fn.update({
            self.image_key_shadow: partial(arrow_mapper, suffix=self.image_key_arrow_suffix),
        })

        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs, key="mmu_ceph_base"),
            shadow_file_fn=shadow_file_fn,
            sample_strategy=index_kwargs.get("mmu_index_strategy", "uniform"),
            probability=index_kwargs.get("mmu_index_probability", None),
        )
        self.logger.info(f"    (MMU) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    (MMU) Using {self.index_manager}")

        self.mmu_short_caption_rate = index_kwargs.get("mmu_short_caption_rate")
        self.mmu_long_caption_rate = index_kwargs.get("mmu_long_caption_rate")

        # Image transform
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.use_front_boi_token = self.args.use_front_boi_token
        self.add_image_shape_token = self.args.get('add_image_shape_token')
        self.pad_color = pad_color
        self.use_und_token = self.args.get('use_und_token', False)

        # Text tokenizer
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Template
        assert template in ["pretrain", "instruct"], f"Unsupported template: {template}"
        if template == "instruct":
            assert conv_format, f"conv_format should be provided for instruct template."
        self.template = template
        self.conv_format = conv_format
        self.default_conv = get_conversation_template(self.conv_format)
        self.roles = self.default_conv.roles
        # {"User": 3, "Assistant": 3, "System": 0}
        self.role_prefix_offset = {
            role: len(self.tokenizer.encode_text(self.default_conv.get_role_prefix(role)))
            for role in self.roles
        }
        self.role_prefix_offset["System"] = 0

        # Image caption processor
        self.caption_processor = load_caption_processor(
            name=default(args.mmu_caption_processor, 'caption_process'),
            caption_sample_ratio=json.loads(args.mmu_caption_sample_ratio),
            logger=self.logger,
            kwargs=args.get('mmu_caption_processor_kwargs'),
        ) if hasattr(args, 'mmu_caption_processor') else None

        # All the data getter
        self.data_getter = dict(
            vl_caudron=self.get_mmu_cauldron,
            vl_llava_onevision=self.get_mmu_llava_onevision,
            Cambrian10M=self.get_mmu_cambrian10m_and_mammoth_vl_instruct_12m,
            MAmmoTHVLInstruct12M=self.get_mmu_cambrian10m_and_mammoth_vl_instruct_12m,
            caption_v2_ancestor=self.get_mmu_caption_v2_ancestor,
        )

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def __post_init__(self, **kwargs):
        pass

    def get_raw_image(self, index, image_key="image", is_list=False, return_first=None, shadow=None):
        try:
            if image_key in ["image", "images"]:
                ret = self.get_image_from_arrow(index, column=image_key, is_list=is_list, return_first=return_first, shadow=shadow)
            elif image_key == "url_cos":
                ret = self.get_image_from_url_cos(index, shadow=shadow)
            else:
                raise ValueError(f"Unknown image_key: {image_key}")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"({image_key=}, {index=}) {type(e)}: {e}. Fallback to gray image.")
            ret = Image.new("RGB", (self.image_size[0], self.image_size[1]), (127, 127, 127))
            image_flag = "gray"
        return ret, image_flag

    def preprocess_image(self, image, target_size):
        origin_size = image.size  # (w_ori, h_ori)

        image, (pad_left, pad_top) = self.index_manager.resize_and_pad(
            image, target_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "pad_coords_xy": (pad_left, pad_top),
        }
        return image_tensor, kwargs

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
        image, image_flag = self.get_raw_image(index, image_key=self.image_key, shadow=self.image_key_shadow)
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)
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

    def get_mmu_image_caption_long_short(self, index):
        long_caption, short_caption = "", ""
        for _ in range(5):
            try:
                text = self.index_manager.get_attribute(index, "caption_v2")
                caption_dict = CaptionAug.safe_load_string(text)
                caption_dict = CaptionAug.normalize_dict_keys(caption_dict)
                long_caption = caption_dict.get("long caption", "")
                short_caption = caption_dict.get("short caption", "")
                if not long_caption and not short_caption:
                    raise ValueError(f"Both long and short captions are empty. {index=}")
                break
            except Exception as e:
                self.handle_exception_message(self.get_mmu_image_caption, e)
                new_index = self.index_manager.random_dindex(index)
                print(f"Error with index={index}, trying new index={new_index}")
                index = new_index
        if not long_caption and not short_caption:
            raise ValueError(f"Both long and short captions are empty. {index=}")

        # Process caption
        if long_caption and short_caption:
            caption = random.choices(
                [long_caption, short_caption], weights=[self.mmu_long_caption_rate, self.mmu_short_caption_rate]
            )[0]
        elif long_caption:
            caption = long_caption
        else:
            caption = short_caption

        if caption == long_caption:
            user = random.choice(image_captioning_long_instructions)
        else:
            user = random.choice(image_captioning_short_instructions)

        message_list = [{
            "User": user,
            "Assistant": caption,
        }]

        # Process image
        image, image_flag = self.get_raw_image(index, image_key=self.image_key, shadow=self.image_key_shadow)
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)

        # message_list = [{"user": "Describe the image.", "assistant": text}]
        return image_tensor, message_list, kwargs

    def get_mmu_caption_v2_ancestor(self, index):
        image, _ = self.get_raw_image(index, image_key="image")
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)

        # [
        #     {
        #         'Assistant': '{
        #           "short_caption": "A person holding a snowboard and wearing...",
        #           "medium_caption": "A light-skinned individual with long...",
        #           "long_caption": "A young woman, smiling at the camera...",
        #           "background": "The background consists of dense green foliage.",
        #           "shot_type": "Medium shot",
        #           "style": "Realistic Photography Style",
        #           "light": "The lighting is soft and diffused, likely from an overcast sky.",
        #           "atmosphere": "Joyful",
        #           "composition": "None",
        #           "IP": "None"
        #         }',
        #         'User': '<image>Describe this image and output the results in json format. The results need to include: short_caption, medium_caption, long_caption, background, shot_type, style, light, atmosphere, composition, IP (Precisely quote clear and prominent text in the image; For illegible or obscured text, simply state 'text is present' at the location.)'
        #     }
        # ]
        message_list: List[Dict] = self.index_manager.get_attribute(index, "conversations")
        message_list[0]["User"] = message_list[0]["User"].replace('<image>', '')  # 移除 <image> 占位符
        return image_tensor, message_list, kwargs

    def get_mmu_cauldron(self, index):
        image, _ = self.get_raw_image(index, image_key="images", is_list=True, return_first=True)
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)

        # [
        # 	{
        # 		"user": "Question: How many actions are depicted in the diagram?\nChoices:\nA. 6.\nB. 4.\nC. 8.\nD. 7.\nAnswer with the letter.",
        # 		"assistant": "Answer: D",
        # 		"source": "TQA"
        # 	}
        # ]
        message_list: List[Dict] = self.index_manager.get_attribute(index, "texts")
        message_list = [{"User": message["user"], "Assistant": message["assistant"]} for message in message_list]
        return image_tensor, message_list, kwargs

    def get_mmu_llava_onevision(self, index):
        image, _ = self.get_raw_image(index, image_key="image")
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)

        # [
        # 	{'from': 'human', 'value': '<image>Hint: Please answer the question and provide the final answer at the end. Question: Subtract all balls. How many objects are left?'},
        #   {'from': 'gpt', 'value': ...},
        # ]
        message_list: List[Dict] = self.index_manager.get_attribute(index, "conversations")
        new_message_list = []
        for i in range(0, len(message_list), 2):
            new_message_list.append({
                "User": message_list[i]["value"],
                "Assistant": message_list[i + 1]["value"],
            })
        return image_tensor, new_message_list, kwargs

    def get_mmu_cambrian10m_and_mammoth_vl_instruct_12m(self, index):
        image, _ = self.get_raw_image(index, image_key="images", is_list=True, return_first=True)
        image_tensor, kwargs = self.preprocess_image(image, self.image_size)

        # 格式与 llava_onevision 相同
        message_list: List[Dict] = self.index_manager.get_attribute(index, "texts")
        new_message_list = []
        for i in range(0, len(message_list), 2):
            new_message_list.append({
                "User": message_list[i]["value"].replace('<image>', ''),    # 移除 <image> 占位符
                "Assistant": message_list[i + 1]["value"],
            })
        return image_tensor, new_message_list, kwargs

    def __getitem__(self, index):
        if "dataset_tag" in self.index_manager.get_columns(index):
            dataset_tag = self.index_manager.get_attribute(index, "dataset_tag")
            getter = self.data_getter[dataset_tag]
        else:
            # T2I dataset used for MMU
            if self.template == "pretrain":
                getter = self.get_mmu_image_caption
            elif self.template == "instruct":
                getter = self.get_mmu_image_caption_long_short
            else:
                raise ValueError(f"Unknown template: {self.template}")

        # Get image and text
        image_tensor, message, kwargs = getter(index)

        image_kwargs = dict(
            add_iw_ih_token=self.add_iw_ih_token, use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token,
        )

        if self.template == "pretrain":
            assert isinstance(message, str), "Only support single message for pretrain template"
            texts = [message]

            itype = "und_image" if self.use_und_token else "image"
            template = f"{itype}-text"
            sections = [
                dict(type=itype, token_length=self.image_token_length, **image_kwargs),
                dict(type="text", text=texts[0], end_offset=1),     # include possible <eos> token
            ]
            max_token_length = self.max_token_length + 1

        else:
            assert isinstance(message, list), "Only support list message for instruct template"
            # [("role", "role: message"), ...]
            role_texts: List[Tuple[str, str]] = self.format_message_list(message, return_type="list", add_system=False)
            # Extract messages as a List[str]
            texts = [text for _, text in role_texts]
            # User: xxx\n\nAssistant: <answer><und_boi>[image]<und_eoi>xxx</answer><eos>
            template = "text-text-und_image-text" + "-text" * (len(texts) - 2)
            assistant_offset = self.role_prefix_offset[self.roles[1]] + 1
            sections = [
                dict(type="text", text=texts[0], ignore=True, max_length=self.max_text_token_length),       # 1+x tokens
                dict(type="text", text=texts[1].split('<answer>')[0] + '<answer>', ignore=False,            # 4 tokens
                     start_offset=assistant_offset, end_offset=1),  # Move slice to the only <und_boi> token
                dict(type="und_image", token_length=self.image_token_length, **image_kwargs),
                dict(type="text", text=texts[1].split('<answer>')[-1], ignore=False),                       # 1+x token
            ]
            for i, text in enumerate(texts[2:], start=2):
                sections.append(dict(type="text", text=text, ignore=i % 2 == 0,
                                     start_offset=0 if i % 2 == 0 else assistant_offset))
            sections[-1]['end_offset'] = 1     # include possible <eos> token
            max_token_length = self.max_token_length + 1

        output = self.tokenizer.encode_general(
            template=template,
            sections=sections,
            max_token_length=max_token_length,
        )
        target_token = output.tokens.clone()
        target_token[output.text_mask == 0.0] = -100

        if self.template == "pretrain" and not self.use_und_token:
            und_image_mask = output.gen_image_mask
            und_image_slices = output.gen_image_slices
        else:
            und_image_mask = output.und_image_mask
            und_image_slices = output.und_image_slices

        # here use resized image size as scatter_src of iw and ih
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        image_token_shape_wh = torch.tensor([w, h], dtype=torch.long)

        # Prepare kwargs
        kwargs.update(dict(
            index=index,
            text="".join(texts),
        ))

        # Prepare attention mask
        n_tokens = output.tokens.shape[0] - 1 + self.dummy_number
        attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
        for sli in und_image_slices:
            attention_mask[sli, sli] = True
        attention_mask = attention_mask.unsqueeze(0)    # head dim

        ret = {
            "dtype": "mmu",
            "image": image_tensor,                      # (3, H~, W~)
            "n_samples": 1,                             # ()
            "tokens": output.tokens,                    # (L), L = text_token_length + 1 + image_token_length
            "target_tokens": target_token,              # (L)
            "und_image_mask": und_image_mask,    # (L)
            "text_mask": output.text_mask,              # (L)
            "attention_mask": attention_mask,
            "kwargs": {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        }

        if output.iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": output.iw_ih_scatter_index,      # (2)
                "iw_ih_scatter_src": image_token_shape_wh,              # (2)
            })

        return ret
