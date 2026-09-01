import random
from typing import List, Optional, Union, Dict, Any
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from diffusers.utils import BaseOutput

from .conversation import get_conversation_template
from .tokenization_hunyuan_multimodal import HunyuanMultimodalTokenizerFast
from ...constants import TOKENIZER_PATH
from ...utils.helpers import default
from ...utils.image_base import ImageInfo, JointImageInfo, JointImage


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class TokenizerEncodeOutput(BaseOutput):
    tokens: torch.Tensor = None
    iw_ih_scatter_index: Optional[torch.Tensor] = None
    timestep_scatter_index: Optional[torch.Tensor] = None
    timestep_r_scatter_index: Optional[torch.Tensor] = None
    guidance_scatter_index: Optional[torch.Tensor] = None
    text_slices: Optional[List[slice]] = None
    src_image_slices: Optional[List[slice]] = None
    vae_image_slices: Optional[List[slice]] = None
    gen_image_slices: Optional[List[slice]] = None
    und_image_slices: Optional[List[slice]] = None
    vit_image_slices: Optional[List[slice]] = None
    joint_image_slices: Optional[List[slice]] = None
    face_image_slices: Optional[List[slice]] = None
    text_mask: Optional[torch.Tensor] = None
    src_image_mask: Optional[torch.Tensor] = None
    vae_image_mask: Optional[torch.Tensor] = None
    gen_image_mask: Optional[torch.Tensor] = None
    und_image_mask: Optional[torch.Tensor] = None
    vit_image_mask: Optional[torch.Tensor] = None
    face_image_mask: Optional[torch.Tensor] = None
    real_pos: Optional[torch.Tensor] = None
    all_image_slices: Optional[List[slice]] = None
    cond_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_timestep_r_scatter_index: Optional[torch.Tensor] = None
    und_token_indices: Optional[torch.Tensor] = None
    gen_token_indices: Optional[torch.Tensor] = None
    gen_video_slices: Optional[list[slice]] = None
    vae_video_slices: Optional[list[slice]] = None
    vit_video_slices: Optional[list[slice]] = None
    vit_video_context_slices: Optional[list[slice]] = None
    joint_video_slices: Optional[list[slice]] = None
    all_video_slices: Optional[list[slice]] = None
    gen_video_mask: Optional[torch.Tensor] = None
    vae_video_mask: Optional[torch.Tensor] = None
    vit_video_mask: Optional[torch.Tensor] = None
    cond_video_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_video_timestep_scatter_index: Optional[torch.Tensor] = None


class TokenizerWrapper(HunyuanMultimodalTokenizerFast):
    def __init__(self, tokenizer: str, logger=None, patch_special_tokens_dict=None):
        self.logger = logger
        if self.logger is None:
            from loguru import logger
            self.logger = logger
        special_tokens_dict = {
            # !!! Notice !!! Order of special tokens is important. Don't change it.
            "additional_special_tokens": [
                "<boi>",
                "<eoi>",
                "<boa>",
                "<eoa>",
                "<bov>",
                "<eov>",
                "<img>",
                "<audio>",
                "<video>",
                "<pad>",
                "<cfg>",
                "<iw>",
                "<ih>",
                "<bot>",
                "<eot>",
                "<text>",
                "<mask>",
                "<timestep>",
                "<recaption>",
                "</recaption>",
                "<bof>",
                "<eof>",
                "<face>",
                "<think>",
                "</think>",
                "<answer>",
                "</answer>",
                "<und_boi>",
                "<und_eoi>",
                "<src_boi>",
                "<src_eoi>",
                "<gen_boi>",  # Used for Janus-CoT
                "<gen_eoi>",  # Used for Janus-CoT
                "<gen_img>",  # Used for Janus-CoT
            ] + [
                # We keep 10 tokens for image size
                f"<img_size_{sz}>" for sz in [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
            ] + [
                # ResolutionGroup(base_size=1024, step=64) can generate 33 different image ratios from 1:4 to 4:1.
                f"<img_ratio_{i}>" for i in range(33)
                # Now we support two types of image shape tokens: (<iw>, <ih>) and (<img_ratio_*> and <img_size_*>).
                # - The former are commonly defined by iw_ih_scatter_index (i.e., a placeholder) for image width
                #   and height, which are generally stored in iw_ih_scatter_src. In Transfusion class, we will encode
                #   the iw_ih_scatter_src into two embeddings and then scatter them into the iw_ih_scatter_index.
                # - The latter are pure special tokens to indicate image height/width ratio and image base size.
                #   The ratio is directly defined by index and can be look up in the resolution group.
            ] + [
                "<joint_img_sep>",
            ] + [
                # <ref> token pair is used for the ocr text, and <quad> token pair is used for the bounding box.
                "<ref>", "</ref>", "<quad>", "</quad>",
            ] + [
                # We add 1000 position tokens for x and y coordinates. Each of them represent the float value with
                # three decimal places of a relative position in the range of [0, 1].
                # For example, (0.512, 0.123) will be represented as (<pos_y_123>, <pos_x_512).
                f"<pos_x_{x}>" for x in range(1000)
            ] + [
                f"<pos_y_{y}>" for y in range(1000)
            ] + [
                "<guidance>",
            ] + [
                f'<relation_{i}>' for i in range(10)
            ] + [
                f'</relation_{i}>' for i in range(10)
            ] + [
                f"<img_ratio_{i}>" for i in range(33, 37)  # for 4:3, 16:9, 3:4, 9:16  # 130103 - 130106
            ] + [
                "<timestep_r>",
            ]
        }
        if tokenizer in TOKENIZER_PATH:
            tokenizer = TOKENIZER_PATH[tokenizer]
        logger.info(f"Loading tokenizer from: {tokenizer}")
        # kwargs of AutoTokenizer.from_pretrained will be passed to Tokenizer.__init__()
        # hy_special_tokens_dict is a special kwargs only used by HYTokenizer.__init__() to align with other transformers tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, hy_special_tokens_dict=special_tokens_dict)
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self._sp_dict = self.special_token_map = {}
        self.bos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.eos_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.bos_token is None:
            logger.warning(f"No `bos_token` found in the {self.tokenizer.__class__.__name__}. Try to fallback to <eos>.")
            if self.eos_token is not None:
                self.bos_token = self.eos_token
                assert isinstance(self.eos_token, int), f"Invalid `eos_token` type: {type(self.eos_token)}."
            else:
                raise ValueError(f"No `eos_token` found in the {self.tokenizer.__class__.__name__}.")
        for ss in special_tokens_dict["additional_special_tokens"]:
            self.special_token_map[ss] = self.tokenizer.convert_tokens_to_ids(ss)
        self.stmap = self.special_token_map
        # Define short name
        self.pad_token = self.special_token_map["<pad>"] # an exception for bc

        self._get_common_used_special_tokens(patch_special_tokens_dict)

        self.answer_token = "<answer>"
        self.end_of_answer_token = "</answer>"
        self.think_token = "<think>"
        self.end_of_think_token = "</think>"
        self.recaption_token = "<recaption>"
        self.end_of_recaption_token = "</recaption>"
        self.ref_token = "<ref>"
        self.end_of_ref_token = "</ref>"
        self.quad_token = "<quad>"
        self.end_of_quad_token = "</quad>"

        self.answer_token_id = self._sp_dict["<answer>"]
        self.end_of_answer_token_id = self._sp_dict["</answer>"]
        self.think_token_id = self._sp_dict["<think>"]
        self.end_of_think_token_id = self._sp_dict["</think>"]
        self.recaption_token_id = self._sp_dict["<recaption>"]
        self.end_of_recaption_token_id = self._sp_dict["</recaption>"]
        self.ref_token_id = self._sp_dict["<ref>"]
        self.end_of_ref_token_id = self._sp_dict["</ref>"]
        self.quad_token_id = self._sp_dict["<quad>"]
        self.end_of_quad_token_id = self._sp_dict["</quad>"]

        self.start_ratio_token_id = self.tokenizer.convert_tokens_to_ids("<img_ratio_0>")
        self.end_ratio_token_id = self.tokenizer.convert_tokens_to_ids("<img_ratio_32>")
        self.ratio_token_other_slices = [
            (self.tokenizer.convert_tokens_to_ids("<img_ratio_33>"), self.tokenizer.convert_tokens_to_ids("<img_ratio_36>") + 1)
        ]

    @staticmethod
    def from_pretrained(tokenizer: str, **kwargs):
        return TokenizerWrapper(tokenizer, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def __repr__(self):
        return f"TokenizerWrapper(tokenizer={self.tokenizer.__class__.__name__})"

    def size_token(self, size: int):
        return f"<img_size_{size}>"

    def ratio_token(self, ratio_idx: int):
        return f"<img_ratio_{ratio_idx}>"

    def get_all_ratio_token_ids(self):
        return [self.ratio_token_id(i) for i in range(37)]

    def get_img_token(self):
        return "<img>"

    def relation_token(self, relation_idx: int, end: bool = False):
        assert len(self._sp_dict) > 0, "Relation tokens are not defined in the tokenizer."
        if end:
            return f"</{relation_idx}>"
        return f"<relation_{relation_idx}>"

    def x_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position x tokens are not defined in the tokenizer."
        return f"<pos_x_{pos}>"

    def y_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position y tokens are not defined in the tokenizer."
        return f"<pos_y_{pos}>"

    def z_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position z tokens are not defined in the tokenizer."
        return f"<pos_z_{pos}>"

    def get_special_token_id(self, default_token_tag_in_map=None, special_token_tag=None):
        if special_token_tag is not None:
            return self.tokenizer.convert_tokens_to_ids(special_token_tag)
        
        assert default_token_tag_in_map is not None
        return self.special_token_map[default_token_tag_in_map]
    
    def _get_common_used_special_tokens(self, patch_special_tokens_dict=None):
        """patch_special_tokens_dict is used for Janus-series models."""
        # Define default special tokens
        special_tokens = {
            "pad": "<pad>",
            "cfg": "<cfg>",
            "boi": "<boi>",
            "eoi": "<eoi>",
            "img": "<img>",
            "gen_boi": "<gen_boi>",
            "gen_eoi": "<gen_eoi>",
            "gen_img": "<gen_img>",
        }
        for key, token in special_tokens.items():
            if not hasattr(self, f"{key}_token"):
                setattr(self, f"{key}_token", token)

        # Initialize special token IDs
        self.special_token_ids = {
            key: self.get_special_token_id(token)
            for key, token in special_tokens.items()
        }

        # Update special token IDs if patch_special_tokens_dict is provided
        if patch_special_tokens_dict:
            for key, value in patch_special_tokens_dict.items():
                if key in self.special_token_ids:
                    self.special_token_ids[key] = self.get_special_token_id(special_tokens[key], value)
                else:
                    raise KeyError(f"{key} is not supported.")

        for key, token_id in self.special_token_ids.items():
            setattr(self, f"{key}_token_id", token_id)

    def pad(self, tensor_list, dim=0, pad_val=None):
        if pad_val is None:
            pad_val = self.special_token_map["<pad>"]
        max_len = max([t.shape[dim] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            if t.shape[dim] < max_len:
                assert pad_val is not False, "Not allowed pad."
                t = F.pad(t, (0, max_len - t.shape[dim]), value=pad_val)
            padded_tensor_list.append(t)
        return padded_tensor_list

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def encode_text(
            self,
            *texts,
            uncond_enabled: Optional[Union[bool, List[bool]]] = None,
            uncond_p: Optional[float] = None,
            max_length: Optional[int] = None,
            pad: Optional[str] = None,
            return_lengths: bool = False,
    ):
        """
        Encode text and image for AR-like model training of the text-to-image/instruction tuning tasks.
        Support encode multiple texts at once. Each text can be separately conditioned or unconditioned
        based on the uncond_flags and a uniform uncond_p.
        **<bos> token is always prepended to the text tokens.**

        Parameters
        ----------
        texts: str or List[str]
            List of texts to be encoded.
        uncond_enabled: bool or List[bool]
            List of flags to indicate whether the text should be unconditioned.
            If False, the text will never be unconditioned.
            If True, the text will be unconditioned with uncond_p.
        uncond_p: float
            Probability to the unconditional text. Only works when uncond_enabled is True.
        max_length: int
            Maximum length of the encoded text.
        pad: Optional[str]
            Padding method. Can be 'left' or 'right'.
        return_lengths: bool
            Whether to return the length of each encoded text.
        """
        if pad is not None:
            assert max_length is not None, "max_length should be provided when pad is not None."

        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        if len(uncond_enabled) != len(texts):
            print(uncond_enabled, texts)
        assert len(uncond_enabled) == len(texts), (
            f"Length of uncond_flags should be equal to the number of texts, "
            f"but got {len(uncond_enabled)} and {len(texts)}."
        )

        # Prepare text/uncond tokens
        # TODO: If len(texts) > 1, such as instruction + prompt in inpainting, we need to determine how to do uncond.
        # Now all texts will be cond or uncond at the same time.
        do_uncond_drop = (uncond_p is not None) and (random.random() < uncond_p)
        text_tokens, lengths = [], []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            # If reach the max_length and there still have unencoded texts, give a warning message and break the loop.
            if max_length is not None and cum_length >= max_length:
                self.logger.warning(
                    f"Text length exceeds the max_length({max_length}). The remaining texts will be ignored: "
                    f"{text[:80]}..."
                )
                break
            # Set add_special_tokens=False to avoid adding <bos> token in some LLMs.
            if isinstance(text, str):
                text_token = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                text_token = text
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            # Cutoff the text by max_length if necessary
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[:max_length - cum_length]
            text_tokens.extend(text_token)
            lengths.append(len(text_token))
            cum_length += len(text_token)

        # Prepend/Append <pad> tokens if applicable
        if pad is not None and (pad_length := max_length - len(text_tokens)) > 0:
            if pad == 'left':
                text_tokens = [self.pad_token_id] * pad_length + text_tokens
            elif pad == 'right':
                text_tokens = text_tokens + [self.pad_token_id] * pad_length
            else:
                raise ValueError(f"Unsupported padding method: {pad}.")

        if return_lengths:
            return text_tokens, lengths
        return text_tokens

    @staticmethod
    def _check_key_number_matched(keys, data):
        # Assert keys and token_source are matched
        assert set(keys) == set(data.keys()), (
            f"Keys in the template and token source should be matched, but got {keys} and {list(data.keys())}."
        )
        key_counts = {k: 0 for k in keys}
        for key in keys:
            key_counts[key] += 1
        for key, count in key_counts.items():
            assert len(data[key]) == count, (
                f"Number of `{key}` in the token source should be matched with the template, but got "
                f"{data[key]}({len(data[key])}) and {count}."
            )

    def _add_image_meta_info_token(self, token_seq, token_count, extra_token_pos, add_iw_ih_token, add_timestep_token,
                                   add_image_shape_token=False, base_size=None, ratio_idx=None, image_type=None, add_guidance_token=False, add_timestep_r_token=False):
        if add_iw_ih_token:
            token_seq.extend([self.special_token_map["<iw>"], self.special_token_map["<ih>"]])
            extra_token_pos['iw_ih'].extend([token_count, token_count + 1])
            token_count += 2
        if add_image_shape_token:
            token_seq.extend([
                self.special_token_map[f"<img_size_{base_size}>"],
                self.special_token_map[f"<img_ratio_{ratio_idx}>"]
            ])
            token_count += 2
        if add_timestep_token:
            token_seq.extend([self.special_token_map["<timestep>"]])
            extra_token_pos['timestep'].append(token_count)
            if image_type is not None:
                if image_type in ["genimage", "image", "gen_image"]:
                    extra_token_pos['gen_timestep'].append(token_count)
                elif image_type in ["src_image", "und_image", "joint_image"]:
                    extra_token_pos['cond_timestep'].append(token_count)
                else:
                    raise ValueError(f"Unsupported image type: {image_type}.")
            token_count += 1
        if add_guidance_token:
            token_seq.extend([self.special_token_map["<guidance>"]])
            extra_token_pos['guidance'].append(token_count)
            token_count += 1
        if add_timestep_r_token:
            token_seq.extend([self.special_token_map["<timestep_r>"]])
            extra_token_pos['timestep_r'].append(token_count)
            token_count += 1
        return token_count

    @staticmethod
    def _shorten_text(text):
        import re
        text = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", text)
        text = re.sub(r"(<pad>)+", lambda m: f"[<pad>]{{{len(m.group(0)) // 5}}}", text)
        return text

    def encode_sequence(
            self,
            template: str,
            token_source: Dict[str, List],
            total_length=None,
            add_iw_ih_token=False,
            add_timestep_token=False,
            add_timestep_r_token=False,
            add_guidance_token=False,
            last_key_only_prefix=False,
            add_eos=True,
            use_front_boi_token=False,
            add_pad=True,
            add_bos=True,
            drop_last: Union[str, bool] = 'auto',
            add_image_shape_token=False,
            und_token_type: list[str] = [],
            gen_token_type: list[str] = [],
    ):
        """
        Encode a sequence based on the template (e.g., `text-image` for t2i, `text-image-image` for instruction tuning)
        and token source.

        Parameters
        ----------
        template: str
            Template of the sequence. E.g., "text-image" means the sequence is composed of text and an image.
            "text-image-image" means the sequence is composed of text and two images.
            "text-face-image" means the sequence is composed of text, face embedding and one image.
        token_source: Dict[str, List]
            Token source for each key in the template, in order.
            - text: List[List[int]]. Each List[int] is a sequence of tokenized text tokens.
            - image: List[int]. Each int means the number of image tokens.
            - face: List[int]. Each int means the number of face embedding tokens. Typically, the number of face embedding tokens is 1.
            - und_image: List[int]. Each int means the number of und_image tokens.
            - src_image: List[int]. Each int means the number of src_image tokens.
        total_length: int
            Total length of the encoded sequence, include padding tokens.
        add_iw_ih_token: bool
            Whether to add iw and ih tokens before the image tokens. (Right before the <timestep> token)
        add_timestep_token: bool
            Whether to add timestep token before the image tokens.
            (Right after the <iw><ih> token or <img_ratio_*><img_size_*> tokens)
        last_key_only_prefix: bool
            Whether to only use the modal prefix in the last key.
        add_eos: bool or 'auto'
            Whether to add eos token at the end of the sequence. If True, always add eos token. If 'auto',
            add eos token only when the total_length is not reached and the last token is not <eos>.
        use_front_boi_token: bool:
            Whether to put the <boi> token at the front of iw, ih and timestep tokens.
        add_pad: bool or 'auto'
            Whether to add padding tokens to the sequence. If True and total_length is not reached, add padding tokens.
        add_bos: bool
            Whether to add bos token at the beginning of the sequence.
        drop_last: bool or 'auto'
            - If auto, drop last tokens exceeding the total_length if the total_length is provided. If cut point is
                in the middle of the image tokens, an error will raised.
            - If True, drop last tokens exceeding the total_length. If cut point is in the middle of the image tokens,
                all the successive image tokens will be dropped.
            - If False, keep the last tokens exceeding the total_length, even if the total_length is reached.
        add_image_shape_token: bool
            Whether to add image shape token before the image tokens. (Right before the <timestep> token)

        Returns
        -------
        token_seq: list
            Encoded token sequence.
        extra_token_pos: dict
            Positions of extra tokens. E.g., iw_ih, timestep.
        """
        if last_key_only_prefix:
            assert add_eos is not True, "add_eos should not be True when last_key_only_prefix is True."
        if drop_last is True and total_length is None:
            raise ValueError("total_length should be provided when drop_last is True.")

        keys = template.split('-')
        modal_length = len(keys)
        index_indicator = {k: 0 for k in token_source}
        for k, v in token_source.items():
            assert isinstance(v, (list, tuple)), (
                f"Value of `{k}` in the token source should be a list or tuple, but got {type(v)}."
            )
        self._check_key_number_matched(keys, token_source)

        token_seq = []
        token_count = 0
        extra_token_pos = defaultdict(list)
        und_token_indices = [] # default: bos, text, boi, vit, eoi, joint_image_sep, eoi, eos, pad
        gen_token_indices = [] # default: timestep, guidance, image_shape, vae
        if add_bos:
            token_seq.append(self.bos_token)
            token_count += 1
        # If drop_last is True, we check the token_count on the fly and exit the loop if the total_length is reached.
        # This check is only applied to the block tokens. Block tokens mean the tokens that are unsplittable, like
        # image tokens, face tokens. Text tokens are splittable, so we don't need to check the token_count for text.
        # If the loop is broken by drop_last, we don't add the eos token at the end because the sequence is not complete.
        drop_last_break = False
        for i, key in enumerate(keys):
            source = token_source[key][index_indicator[key]]
            if key == "text":
                token_seq.extend(source)     # text token sequence
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)

            elif key == "genimage":  # for Janus-CoT
                if isinstance(source, int):
                    source = {'length': source}
                extra_count = 2 + (
                    2 if source.get('iw_ih', add_iw_ih_token) else 0) + (
                    1 if source.get('timestep', add_timestep_token) else 0) + (
                    2 if source.get('image_shape', add_image_shape_token) else 0
                )
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                if source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.gen_boi_token_id)  # Use patched boi for Janus, otherwise useing default <boi>
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_iw_ih_token=source.get('iw_ih', add_iw_ih_token),
                    add_timestep_token=source.get('timestep', add_timestep_token),
                    add_image_shape_token=source.get('image_shape', add_image_shape_token),
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    image_type=key,
                )
                if last_key_only_prefix and i == modal_length - 1:
                    pass   # for AR inference
                else:
                    token_seq.extend(
                        [self.gen_boi_token_id] +
                        [self.gen_img_token_id] * source['length'] +     # token number
                        [self.gen_eoi_token_id]
                    )
                    token_count += 2 + source['length']

            elif key in ["image", "gen_image"]:     # for Transfusion
                if isinstance(source, int):
                    source = {'length': source}
                extra_count = 2 + (
                    2 if source.get('iw_ih', add_iw_ih_token) else 0) + (
                    1 if source.get('timestep', add_timestep_token) else 0) + (
                    1 if source.get('timestep_r', add_timestep_r_token) else 0) + (
                    1 if source.get('guidance', add_guidance_token) else 0) + (
                    2 if source.get('image_shape', add_image_shape_token) else 0
                )
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                if source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise useing default <boi>
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_iw_ih_token=source.get('iw_ih', add_iw_ih_token),
                    add_timestep_token=source.get('timestep', add_timestep_token),
                    add_timestep_r_token=source.get('timestep_r', add_timestep_r_token),
                    add_guidance_token=source.get('guidance', add_guidance_token),
                    add_image_shape_token=source.get('image_shape', add_image_shape_token),
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    image_type=key,
                )
                if not source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                if last_key_only_prefix and i == modal_length - 1:
                    pass       # for AR inference
                else:
                    token_seq.extend(
                        [self.img_token_id] * source['length'] +     # token number
                        [self.eoi_token_id]
                    )
                    extra_token_pos["<img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source['length']
                    extra_token_pos["<img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)
                    extra_token_pos["eoi"].append(token_count)
                    token_count += 1    # <eoi>
            
            elif key == "src_image":
                if isinstance(source, int):
                    source = {'length': source}
                extra_count = 2 + (
                    2 if source.get('iw_ih', add_iw_ih_token) else 0) + (
                    1 if source.get('timestep', add_timestep_token) else 0) + (
                    2 if source.get('image_shape', add_image_shape_token) else 0
                )
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                if source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.special_token_map["<src_boi>"])
                    extra_token_pos["src_boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_iw_ih_token=source.get('iw_ih', add_iw_ih_token),
                    add_timestep_token=source.get('timestep', add_timestep_token),
                    add_image_shape_token=source.get('image_shape', add_image_shape_token),
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    image_type=key,
                )
                if not source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.special_token_map["<src_boi>"])
                    extra_token_pos["src_boi"].append(token_count)
                    token_count += 1
                if last_key_only_prefix and i == modal_length - 1:
                    pass       # for AR inference
                else:
                    token_seq.extend(
                        [self.special_token_map["<img>"]] * source['length'] +     # token number
                        [self.special_token_map["<src_eoi>"]]
                    )
                    extra_token_pos["<src_img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source['length']
                    extra_token_pos["<src_img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)
                    extra_token_pos["src_eoi"].append(token_count)
                    token_count += 1    # <eoi>

            elif key == "und_image":
                assert not add_timestep_token, "add_timestep_token is not supported for und_image."
                assert not last_key_only_prefix, "last_key_only_prefix is not supported for und_image."
                if isinstance(source, int):
                    source = {'length': source}
                extra_count = 2 + (
                    2 if source.get('iw_ih', add_iw_ih_token) else 0) + (
                    2 if source.get('image_shape', add_image_shape_token) else 0
                )
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                if source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.special_token_map["<und_boi>"])
                    extra_token_pos["und_boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_iw_ih_token=source.get('iw_ih', add_iw_ih_token),
                    add_timestep_token=source.get('timestep', False),
                    add_image_shape_token=source.get('image_shape', add_image_shape_token),
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    image_type=key,
                )
                if not source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.special_token_map["<und_boi>"])
                    extra_token_pos["und_boi"].append(token_count)
                    token_count += 1
                token_seq.extend(
                    [self.special_token_map["<img>"]] * source['length'] +     # token number
                    [self.special_token_map["<und_eoi>"]]
                )
                extra_token_pos["<und_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<und_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["und_eoi"].append(token_count)
                token_count += 1    # <eoi>

            elif key == "joint_image":
                assert isinstance(source['length'], list) and len(source['length']) == 2, "joint_image length should be a list of two integers"
                extra_count = 2 + 1 + ( # boi, eoi, joint_img_sep
                    2 if source.get('iw_ih', add_iw_ih_token) else 0) + (
                    1 if source.get('timestep', add_timestep_token) else 0) + (
                    2 if source.get('image_shape', add_image_shape_token) else 0
                )
                if drop_last is True and token_count + extra_count + sum(source['length']) > total_length:
                    drop_last_break = True
                    break
                if source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise useing default <boi>
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                token_count = self._add_image_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_iw_ih_token=source.get('iw_ih', add_iw_ih_token),
                    add_timestep_token=source.get('timestep', add_timestep_token),
                    add_image_shape_token=source.get('image_shape', add_image_shape_token),
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    image_type=key,
                )
                if not source.get('front_boi', use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                if last_key_only_prefix and i == modal_length - 1:
                    pass       # for AR inference
                else:
                    token_seq.extend(
                        [self.img_token_id] * source['length'][0]
                    )
                    extra_token_pos["<src_img>_start"].append(token_count)
                    extra_token_pos["<joint_img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source['length'][0]
                    extra_token_pos["<src_img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)

                    token_seq.extend(
                        [self.special_token_map["<joint_img_sep>"]]
                    )
                    extra_token_pos["joint_img_sep"].append(token_count)
                    token_count += 1

                    token_seq.extend(
                        [self.img_token_id] * source['length'][1]
                    )
                    extra_token_pos["<und_img>_start"].append(token_count)
                    extra_token_pos["<all_img>_start"].append(token_count)
                    token_count += source['length'][1]
                    extra_token_pos["<und_img>_end"].append(token_count - 1)
                    extra_token_pos["<joint_img>_end"].append(token_count - 1)
                    extra_token_pos["<all_img>_end"].append(token_count - 1)

                    token_seq.extend(
                        [self.eoi_token_id]
                    )
                    extra_token_pos["eoi"].append(token_count)
                    token_count += 1    # <eoi>

            elif key == "face":
                if isinstance(source, int):
                    source = {'length': source}
                if drop_last is True and token_count + 2 + source['length'] > total_length:
                    drop_last_break = True
                    break
                token_seq.extend([self.special_token_map["<bof>"]] +
                                 [self.special_token_map["<face>"]] * source['length'] +
                                 [self.special_token_map["<eof>"]]
                                 )
                extra_token_pos["<face>_start"].append(token_count + 1)
                token_count += 2 + source['length']
                extra_token_pos["<face>_end"].append(token_count - 2)

            else:
                raise ValueError(f"Not supported key: {key}")
            index_indicator[key] += 1

        if add_eos is True and not drop_last_break:
            # Typically used for t2i task.
            token_seq.append(self.eos_token)
            extra_token_pos["eos"].append(token_count)
            token_count += 1
        elif add_eos == 'auto' and not drop_last_break:
            # Typically used for lm and mmu task.
            if token_seq[-1] != self.eos_token and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token)
                extra_token_pos["eos"].append(token_count)
                token_count += 1

        if total_length:
            # Check token count and clip sequence if necessary
            if token_count > total_length and drop_last:
                # Assert clip position is not in the middle of the block-wise tokens (gen_image,
                # src_image, und_image, face)
                for start_key, end_key in [
                        ("<img>_start", "<img>_end"), ("<src_img>_start", "<src_img>_end"),
                        ("<und_img>_start", "<und_img>_end"), ("<face>_start", "<face>_end")
                ]:
                    if start_key in extra_token_pos and end_key in extra_token_pos:
                        assert all(
                            (start > total_length or end + 1 < total_length)
                            for start, end in zip(extra_token_pos[start_key], extra_token_pos[end_key])
                        ), ("Clip position should not be in the middle of the image tokens.\n"
                            f"The total_length is set to {total_length}, which is smaller than the sequence length.\n"
                            f"Below is the sequence:\n{self._shorten_text(self.tokenizer.decode(token_seq))}")
                token_seq = token_seq[:total_length]

            # Pad the sequence if necessary
            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra_token_pos["first_pad"].append(token_count)

        return token_seq, extra_token_pos, und_token_indices, gen_token_indices

    def batch_gen_infer(
            self,
            infer_fn,
            prompt_list: list,
            negative_prompt_list: list = None,
            infer_fn_kwargs_list: List[Dict[str, int]] = None,
            do_classifier_free_guidance=False,
            condition_repeat_times: int = 1,
            uncondition_repeat_times: int = 1,
            und_token_type: Optional[list] = None,
            gen_token_type: Optional[list] = None,
            audio_token_type: Optional[list] = None,
    ):
        """
        Batch inference for the AR-like model training of the text-to-image/instruction tuning tasks.

        Parameters
        ----------
        infer_fn: callable
            Inference function to encode the prompt.
        prompt_list: list
            List of prompts. Each element can be a single prompt or a list of prompts passed to the infer_fn.
        negative_prompt_list: list
            List of negative prompts. Only used when do_classifier_free_guidance is True. If None, will use <cfg> token sequence as negative prompt.
        infer_fn_kwargs_list: List[Dict[str, int]]
            List of keyword arguments for the infer_fn.
        do_classifier_free_guidance: bool
            Whether to do classifier-free guidance.
        condition_repeat_times and uncondition_repeat_times: int
            Support multi-condition and multi-uncondition. e.g, [pred_cond, pred_uncond_text, pred_uncond_text_uncond_face]
        """
        if infer_fn_kwargs_list is None:
            infer_fn_kwargs_list = [{} for _ in prompt_list]

        # [n_output, bsz]
        cond_results_list = None
        uncond_results_list = None
        output_type_list = []

        for prompt_idx, (prompt, infer_fn_kwargs) in enumerate(zip(prompt_list, infer_fn_kwargs_list)):
            if not isinstance(prompt, (list, tuple)):
                prompt = [prompt]
            cond_kwargs = {"uncond_p": 0.0} if do_classifier_free_guidance else {}
            results = infer_fn(
                *prompt,
                **infer_fn_kwargs,
                **cond_kwargs,
            )
            output_type_list.append((type(results), len(results) if isinstance(results, (list, tuple)) else 1))
            if isinstance(results, dict):
                raise ValueError("Make batch on dict is not supported. Please return list or tuple for infer_fn.")
            if not isinstance(results, (list, tuple)):
                results = (results,)
            if cond_results_list is None:
                cond_results_list = [[] for _ in results]
                uncond_results_list = [[] for _ in results]
            for i, result in enumerate(results):
                cond_results_list[i].append(result)

            if do_classifier_free_guidance:
                if negative_prompt_list is None:
                    uncond_kwargs = {"uncond_p": 1.0}
                    uncond_results = infer_fn(
                        *prompt,
                        **infer_fn_kwargs,
                        **uncond_kwargs,
                    )
                else:
                    negative_prompt = negative_prompt_list[prompt_idx]
                    if not isinstance(negative_prompt, (list, tuple)):
                        negative_prompt = [negative_prompt]
                    uncond_results = infer_fn(
                        *negative_prompt,
                        **infer_fn_kwargs,
                    )
                if isinstance(uncond_results, TokenizerEncodeOutput):
                    uncond_results_list.append(uncond_results)
                else:
                    for i, result in enumerate(uncond_results):
                        uncond_results_list[i].append(result)

        assert all(output_type_list[0] == n for n in output_type_list), \
            f"Number of outputs should be equal for all samples, but got {output_type_list}."
        output_type, output_num = output_type_list[0]

        def make_batch(batch_cond_item, batch_uncond_item):
            # Process each output item to make batch
            first = batch_cond_item[0]     # The first element in the batch
            if isinstance(first, torch.Tensor):
                stacked_item = torch.stack(self.pad(
                    batch_cond_item * condition_repeat_times + batch_uncond_item * uncondition_repeat_times,
                ))

            elif first is None:
                assert all(item is None for item in batch_cond_item + batch_uncond_item), \
                    (f"The first cond item is None, but some items are not None:\n\n"
                     f"condition: {batch_cond_item}\n\n"
                     f"uncondition: {batch_uncond_item}")
                stacked_item = None

            elif isinstance(first, (list, tuple)):
                # If the output item is a list or tuple, we treat it as a whole, and won't make nested batch any more.
                stacked_item = batch_cond_item * condition_repeat_times + batch_uncond_item * uncondition_repeat_times

            elif isinstance(first, TokenizerEncodeOutput):
                stacked_item = {}
                # Traverse not-None attributes
                for key in list(first.keys()):
                    merged_list = [cond_item[key] for cond_item in batch_cond_item] * condition_repeat_times + \
                        [uncond_item[key] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    if isinstance(first[key], torch.Tensor):
                        if 'mask' in key:
                            pad_val = 0.0
                        elif key == 'tokens':
                            pad_val = self.special_token_map["<pad>"]
                        else:
                            pad_val = False     # Should not pad for other tensors
                        stacked_item[key] = torch.stack(self.pad(merged_list, pad_val=pad_val), dim=0)
                    elif isinstance(first[key], list):
                        stacked_item[key] = merged_list
                    elif first[key] is None:
                        pass
                    else:
                        raise ValueError(f"Unsupported type of {key}: {type(first[key])}.")
                stacked_item = TokenizerEncodeOutput(stacked_item)

            else:
                raise TypeError(f"Making batch on type {type(first)} is not supported.")

            return stacked_item

        stacked_outputs = []
        for cond_results, uncond_results in zip(cond_results_list, uncond_results_list):
            stacked_outputs.append(make_batch(cond_results, uncond_results))

        if output_type == list:
            return stacked_outputs
        elif output_type == tuple:
            return tuple(stacked_outputs)
        elif output_num == 1:
            return stacked_outputs[0]
        else:
            raise ValueError(f"Unsupported output type: {output_type}.")

    def encode_ar(
            self,
            *prompts: str,
            max_text_token_length: int = 256,
            max_image_token_length: int = 1024,
            uncond_enabled: Optional[Union[bool, List[bool]]] = None,
            uncond_p: float = 0.0,
            add_iw_ih_token: bool = False,
            use_front_boi_token: bool = False,
            image_token: Optional[torch.Tensor] = None,
            image_token_length: Optional[int] = None,
            src_image_token_lst: Optional[List[torch.Tensor]] = None,
            last_key_only_prefix: bool = False,
    ):
        """
        Example patterns:
              text to image: <bos> <text>   <boi> <iw> <ih> <img> <eoi>   <eos>
        text+image to image: <bos> <text>   <boi> <iw> <ih> <img> <eoi>   <boi> <iw> <ih> <img> <eoi>   <eos>
        """

        # ==== Prepare text tokens ====
        src_image_token_lst = ensure_list(src_image_token_lst)
        n_img = len(src_image_token_lst) + 1
        max_length_left_for_text = (
            max_text_token_length
            - 2 # 2 for <bos> and <eos>
            - n_img * (2 + (2 if add_iw_ih_token else 0))   # 2 for <boi> and <eoi>, 2 for <iw> and <ih>
        )
        text_token = self.encode_text(*prompts, uncond_enabled=uncond_enabled, uncond_p=uncond_p, max_length=max_length_left_for_text)

        # ==== Prepare image tokens ====
        src_image_token_lengths = [_src_token.shape[0] for _src_token in src_image_token_lst]
        if image_token_length is None:
            image_token_length = max_image_token_length if image_token is None else image_token.shape[0]

        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template="text" + "-image" * n_img,
            token_source=dict(text=[text_token], image=src_image_token_lengths + [image_token_length]),
            total_length=max_text_token_length + max_image_token_length * n_img,
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
            last_key_only_prefix=last_key_only_prefix,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        img_extra_len = 2 if add_iw_ih_token else 0

        # ==== Prepare masks and prefilled image tokens ====
        text_loss_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        text_loss_mask[1: len(text_token)+1] = 1.0 # 1 for <bos>

        start_pos = 1 + len(text_token) + img_extra_len + 1  # 1 for <bos> and 1 for <boi>
        # Source image tokens and masks, if available
        if src_image_token_lengths:
            src_image_loss_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            for i, src_image_token in enumerate(src_image_token_lst):
                src_image_slice = slice(start_pos, start_pos + src_image_token.shape[0])
                full_seq_token_tensor[src_image_slice] = src_image_token
                src_image_loss_mask[src_image_slice] = 1.0
                start_pos = src_image_slice.stop + img_extra_len + 2  # 2 for <eoi> of last image, <boi> of next image
        # Target image token and mask
        tgt_image_slice = slice(start_pos, start_pos + image_token_length)
        if image_token is not None:
            full_seq_token_tensor[tgt_image_slice] = image_token
        tgt_image_loss_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        tgt_image_loss_mask[tgt_image_slice] = 1.0

        if src_image_token_lengths:
            return full_seq_token_tensor, iw_ih_scatter_index, text_loss_mask, src_image_loss_mask, tgt_image_loss_mask

        return full_seq_token_tensor, iw_ih_scatter_index, text_loss_mask, tgt_image_loss_mask
    
    def encode_ar_janus_cot(
        self,
        prompts: str,
        max_n_images: int = 3,
        max_text_token_length: int = 256,
        max_image_token_length: int = 576,
        uncond_enabled: Optional[Union[bool, List[bool]]] = None,
        uncond_p: float = 0.0,
        add_iw_ih_token: bool = False,
        image_token: Optional[torch.Tensor] = None,
        image_token_length: Optional[int] = None,
        last_key_only_prefix: bool = False,
        enable_think_mode: bool = False,
    ):
        """
        Example patterns:
            If not enable_think_mode:
                '<bos> <text>  <gen_boi> <gen_img> <gen_eoi>  <boi> <img> <eoi>  <text> <eos>'
                where the <bos> <boi> <eoi> <img> <gen_eoi> <pad> <cfg> and the first <gen_boi> do not calculate loss
            Otherwise:
                '<bos> <text>
                <think>  <gen_boi> <gen_img> <gen_eoi>  <boi> <img> <eoi>  <text> </think>
                <answer>  <gen_boi> <gen_img> <gen_eoi>  </answer>
                <eos>'
                where the <bos> <boi> <eoi> <img> <gen_eoi> <pad> <cfg> do not calculate loss
        """

        add_iw_ih_token = False
        n_imgs = prompts.count("<image_placeholder>")
        multi_text = prompts.split("<image_placeholder>")
        assert n_imgs <= max_n_images, f"Number of images({n_imgs}) exceeds max_n_images({max_n_images})."

        # ==== Prepare text tokens ====
        max_length_left_for_text = (
            max_text_token_length
            - 2  # <bos> and <eos>
            - (4 if enable_think_mode else 0)  # <think>, </think>, <answer>, </answer>
            - n_imgs * (2 + (2 if add_iw_ih_token else 0)) * 2  # <boi>, <eoi>, <iw>, <ih> for both under- and gen-
            - (2 if enable_think_mode else 0)  # Extra <gen_boi> and <gen_eoi> after <answer>
        )

        multi_text_token = []
        for i, txt in enumerate(multi_text):
            # NOTE: Unconditional sampling ONLY for the input prompt
            if i > 0:
                uncond_enabled = False
            text_token = self.encode_text(
                txt,
                uncond_enabled=uncond_enabled,
                uncond_p=uncond_p,
                max_length=max_length_left_for_text,
            )
            max_length_left_for_text = max(0, max_length_left_for_text - len(text_token))
            if text_token:
                multi_text_token.append(text_token)
            if max_length_left_for_text == 0:
                break
        assert len(multi_text_token) >= n_imgs, "Not enough text tokens for the number of images."

        # ==== Prepare template and token source ====
        if image_token_length is None:
            image_token_length = max_image_token_length if image_token is None else image_token.shape[0]

        template_parts = []
        token_source = {"text": [], "image": [], "genimage": []}

        for i, text_segment in enumerate(multi_text_token):
            template_parts.append("text")
            token_source["text"].append(text_segment)
            if i == 0 and enable_think_mode:
                token_source["text"].append([self.special_token_map["<think>"]])
                template_parts.append("text")
            if i < n_imgs:
                template_parts.extend(["genimage", "image"])
                token_source["genimage"].append(image_token_length)
                token_source["image"].append(image_token_length)

        if enable_think_mode:
            token_source["text"].extend([
                [self.special_token_map["</think>"]],
                [self.special_token_map["<answer>"]],
                [self.special_token_map["</answer>"]],
            ])
            token_source["genimage"].append(image_token_length)
            template_parts.extend(["text", "text", "genimage", "text"])

        template = "-".join(template_parts)

        # ==== Prepare image tokens ====
        total_length = max_text_token_length + max_image_token_length * max_n_images * 2
        if enable_think_mode:
            total_length += max_image_token_length

        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template=template,
            token_source=token_source,
            total_length=total_length,
            add_iw_ih_token=add_iw_ih_token,
            last_key_only_prefix=last_key_only_prefix,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)

        # ==== Prepare masks ====
        no_loss_tokens = torch.tensor([
            self.bos_token,
            self.boi_token_id,
            self.eoi_token_id,
            self.img_token_id,
            self.gen_eoi_token_id,
            self.pad_token_id,
            self.cfg_token_id,
        ])
        no_loss_mask = torch.isin(full_seq_token_tensor, no_loss_tokens)

        # Mask tokens before the first <gen_boi> token, i.e., mask the system prompt
        gen_boi_positions = full_seq_token_tensor == self.gen_boi_token_id
        before_first_gen_boi_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.bool)
        if gen_boi_positions.any():
            first_gen_boi_idx = torch.nonzero(gen_boi_positions, as_tuple=True)[0][0]
            before_first_gen_boi_mask[:(first_gen_boi_idx + 1)] = True
        no_loss_mask |= before_first_gen_boi_mask

        gen_img_mask = full_seq_token_tensor == self.gen_img_token_id
        text_loss_mask = (~no_loss_mask & ~gen_img_mask).float()
        image_loss_mask = gen_img_mask.float()

        return full_seq_token_tensor, text_loss_mask, image_loss_mask

    def encode_ar_t2i_infer(self, text: str, max_length=256, uncond_p=0.0, add_iw_ih_token=False, use_front_boi_token=False, template=None):
        """ Infer mode only returns <bos> text <boi> """
        text_token = self.encode_text(
            text,
            uncond_p=uncond_p,
            max_length=max_length,
        )

        template = "text-image" if template is None else template
        text_key, image_key = template.split("-")
        assert text_key == "text", f"Invalid template: {template}."

        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template=template,
            token_source={text_key: [text_token], image_key: [0]},
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
            last_key_only_prefix=True,
            add_eos=False,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)

        return full_seq_token_tensor, iw_ih_scatter_index, real_pos
    
    def encode_ar_ti2t_infer(
        self,
        text: str,
        imgs_token_length: List[int],
        max_length: int = 256,
        add_iw_ih_token: bool = False,
        uncond_p: float = 0.0,
        img_token_tag: Optional[str] = None,
    ):
        """
        Infer mode for Janus/Janus-CoT multimodal understanding. Returns tokens like:
        '<bos> text1 <boi> <img> <eoi> text2'

        Args:
            text (str): Input text containing image placeholders.
            imgs_token_length (List[int]): List of token lengths for each image.
            max_length (int): Maximum length for text tokens.
            add_iw_ih_token (bool): Whether to add image width/height tokens.
            uncond_p (float): Probability for unconditional sampling.
            img_token_tag (str): Placeholder tag for images in the text.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
                - Full sequence token tensor.
                - Scatter index for image width/height tokens (if applicable).
                - Real position tensor.
        """
        # Set default image token tag if not provided
        img_token_tag = "<img>" if img_token_tag is None else img_token_tag

        # Split text by image token tag and encode each segment
        multi_text = text.split(img_token_tag)
        multi_text_token = [
            self.encode_text(txt, max_length=max_length) for txt in multi_text
        ]

        # Validate the number of image placeholders
        num_images = text.count(img_token_tag)
        assert num_images == len(imgs_token_length), (
            f"Number of image tags ({num_images}) must match the length of imgs_token_length ({len(imgs_token_length)})."
        )

        # Prepare template and token source
        template_parts = []
        token_source = {"text": [], "image": []}

        for i, text_segment in enumerate(multi_text_token):
            template_parts.append("text")
            token_source["text"].append(text_segment)
            if i < num_images:
                template_parts.append("image")
                token_source["image"].append(imgs_token_length[i])

        template = "-".join(template_parts)

        # Encode the sequence
        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template=template,
            token_source=token_source,
            add_iw_ih_token=add_iw_ih_token,
            last_key_only_prefix=False,
            add_eos=False,
        )

        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        iw_ih_scatter_index = (
            torch.tensor(extra_token_pos["iw_ih"], dtype=torch.long)
            if add_iw_ih_token else None
        )
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)

        return full_seq_token_tensor, iw_ih_scatter_index, real_pos
    
    def encode_ar_text_only_infer(self, text: str, max_length=256, uncond_p=0.0, extra_special_tokens: List = None):
        """Infer mode for text-only input. Returns: <bos> text *extra_special_tokens*
           extra_special_tokens: A list of additional special tokens to append to the text tokens, **not affected by uncond_p**"
        """
        text_token = self.encode_text(
            text,
            uncond_p=uncond_p,
            max_length=max_length,
        )

        template = "text"
        token_source = {"text": [text_token]}
        if extra_special_tokens is not None:
            template += "-text"
            token_source["text"].append(extra_special_tokens)

        full_seq_token, _, _, _ = self.encode_sequence(
            template=template,
            token_source=token_source,
            add_eos=False,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)

        return full_seq_token_tensor, None, real_pos

    def encode_ar_editing_infer(self, instruction: str, src_img_token: torch.Tensor, max_length=256, uncond_p=0.0,
                                add_iw_ih_token=False, use_front_boi_token=False):
        text_token = self.encode_text(instruction, uncond_p=uncond_p, max_length=max_length)

        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template="text-image-image",
            token_source=dict(text=[text_token], image=[src_img_token.shape[0], 0]),
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
            last_key_only_prefix=True,
            add_eos=False,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)
        img_extra_len = 2 if add_iw_ih_token else 0

        src_image_start = 1 + len(text_token) + img_extra_len + 1  # 1 for <bos> and 1 for <boi>
        src_image_slice = slice(src_image_start, src_image_start + src_img_token.shape[0])
        full_seq_token_tensor[src_image_slice] = src_img_token

        return full_seq_token_tensor, iw_ih_scatter_index, real_pos

    def encode_ar_inpainting_infer(self, instruction: str, prompt: str, src_img_token: torch.Tensor, mask: torch.Tensor,
                                   max_length=256, uncond_enabled=None, uncond_p=0.0, add_iw_ih_token=False, use_front_boi_token=False):
        text_token = self.encode_text(instruction, prompt, uncond_enabled=uncond_enabled, uncond_p=uncond_p, max_length=max_length)

        mask = mask.reshape(-1)
        masked_src_img_token = torch.where(mask == 1.0, self.special_token_map["<mask>"], src_img_token)

        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template="text-image-image",
            token_source=dict(text=[text_token], image=[masked_src_img_token.shape[0], 0]),
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
            last_key_only_prefix=True,
            add_eos=False,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)
        img_extra_len = 2 if add_iw_ih_token else 0

        src_image_start = 1 + len(text_token) + img_extra_len + 1  # 1 for <bos> and 1 for <boi>
        src_image_slice = slice(src_image_start, src_image_start + masked_src_img_token.shape[0])
        full_seq_token_tensor[src_image_slice] = masked_src_img_token

        return full_seq_token_tensor, iw_ih_scatter_index, real_pos

    def encode_mlm_t2i(
            self,
            text: str,
            image_token_len: int,
            max_text_token_length: int,
            text_uncond_p: float = 0.0,
            dtype=torch.long,
    ):
        # Prepare text tokens: 4 contains <bos>, <boi>, <eoi>, <eos>
        text_tokens = self.encode_text(text, uncond_p=text_uncond_p, max_length=max_text_token_length - 4, pad='left')

        # Combine text and image tokens
        whole_tokens = torch.tensor(
            self.encode_sequence('text-image', dict(text=[text_tokens], image=[image_token_len]))[0],
            dtype=dtype,
        )
        return whole_tokens

    def encode_mlm_editing(
            self,
            *prompts: str,
            src_masked_shifted_image_tokens,
            tgt_masked_shifted_image_tokens,
            src_image_token_len: int,
            tgt_image_token_len: int,
            max_text_token_length: int,
            max_total_token_length: int,
            uncond_enabled: list = None,
            instruction_uncond_p: float = 0.0,
            dtype=torch.long,
            return_labels=True,
            loss_predict=None,
    ):
        # Prepare text tokens: 6 contains <bos>, 2 <boi>, 2 <eoi>, <eos>
        text_tokens = self.encode_text(
            *prompts, uncond_enabled=uncond_enabled,
            uncond_p=instruction_uncond_p, max_length=max_text_token_length - 6, pad='left')

        # Combine text and image tokens
        whole_tokens = torch.tensor(
            self.encode_sequence(
                'text-image-image',
                dict(text=[text_tokens], image=[src_image_token_len, tgt_image_token_len]),
                total_length=max_total_token_length
            )[0], dtype=dtype,
        )

        if return_labels:
            # Predict text tokens(next token prediction) and image tokens(mask prediction).
            if loss_predict.startswith('text__image'):
                whole_labels = whole_tokens.clone()
                whole_labels[
                    (whole_labels == self.special_token_map['<pad>']) | (whole_labels == self.special_token_map['<cfg>']) |
                    (whole_labels == self.bos_token) | (whole_labels == self.special_token_map['<boi>'])
                ] = -100
            elif loss_predict.startswith('target_only'):
                whole_labels = torch.full_like(whole_tokens, -100)
            else:
                raise ValueError(f"Unsupported loss_predict: {loss_predict}")

        # Fill with image tokens
        # Prepare whole tokens
        src_start = max_text_token_length - 4
        src_token_slice = slice(src_start, src_start + src_masked_shifted_image_tokens.size(0))
        whole_tokens[src_token_slice] = src_masked_shifted_image_tokens
        tgt_start = src_token_slice.stop + 2  # <eoi> of src image and <boi> of tgt image
        tgt_token_slice = slice(tgt_start, tgt_start + tgt_masked_shifted_image_tokens.size(0))
        whole_tokens[tgt_token_slice] = tgt_masked_shifted_image_tokens

        if return_labels:
            return whole_tokens, whole_labels, src_token_slice, tgt_token_slice
        else:
            return whole_tokens

    @staticmethod
    def get_actual_image_token_length(image, vae_meta_info, patch_size=1):
        """
            Get the actual image token length for the image
        Args:
            image (torch tensor): [3, h, w] float number 
            vae_meta_info (dict): the meta info of the vae; self.clip_meta_info or self.vae_meta_info
            patch_size (int, optional): the patch size of the image feature encoded by the vae
        Returns:
            int: the actual image token length
        """
        if vae_meta_info.get("fixed_resolution", None) is None:
            h, w = image.shape[1], image.shape[2]
            assert h % (vae_meta_info['downsample_factor'][0] * patch_size) == 0 and w % (vae_meta_info['downsample_factor'][1] * patch_size) == 0, \
            f"Image size should be divisible by downsample_factor * patch_size, but got ({h} x {w}) with downsample_factor={vae_meta_info['downsample_factor']} and patch_size={patch_size}"
            actual_image_token_length = (h // (vae_meta_info['downsample_factor'][0] * patch_size)) * (w // (vae_meta_info['downsample_factor'][1] * patch_size))
        else:
            fixed_h, fixed_w = vae_meta_info['fixed_resolution']
            assert fixed_h % patch_size == 0 and fixed_w % patch_size == 0, \
            f"Image size should be divisible by patch_size, but got ({fixed_h} x {fixed_w}) with patch_size={patch_size}"
            actual_image_token_length = (fixed_h // patch_size) * (fixed_w // patch_size)

        return actual_image_token_length

    def prepare_src_condition_lengths(
            self,
            actual_src_image_token_length_vae,
            image_token_length,
            actual_src_image_token_length_clip,
            image_token_length_clip,
            src_condition_type=["vae"],
            face_bof_eof=False,
            resampler_token_length=None
    ):
        """
        Prepare source condition lengths for both training and inference.
        
        Parameters
        ----------
        src_condition_type : list
            List of source condition types (e.g. ["vae"], ["face_embed", "clip"])
        face_bof_eof : bool
            Whether to use face beginning/end of frame tokens
            
        Returns
        -------
        dict
            Dictionary containing token length information
        """
        actual_src_image_token_length_list = []
        actual_src_face_token_length_list = []
        max_image_token_length_list = []
        max_face_token_length_list = []

        for src_condition_type_item in src_condition_type:
            if src_condition_type_item == "vae":
                src_condition_token_length = actual_src_image_token_length_vae
                max_condition_token_length = image_token_length
            elif src_condition_type_item == "clip":
                src_condition_token_length = actual_src_image_token_length_clip
                max_condition_token_length = image_token_length_clip
            elif src_condition_type_item == "face_embed":
                if resampler_token_length is not None:
                    src_condition_token_length = resampler_token_length
                    max_condition_token_length = resampler_token_length
                else:
                    src_condition_token_length = 1
                    max_condition_token_length = 1
            else:
                raise ValueError(f"Invalid src_condition_type: {src_condition_type_item}")

            if not face_bof_eof:
                actual_src_image_token_length_list.append(src_condition_token_length)
                max_image_token_length_list.append(max_condition_token_length)
            else:
                actual_src_face_token_length_list.append(src_condition_token_length)
                max_face_token_length_list.append(max_condition_token_length)

        return {
            "actual_src_image_token_length_list": actual_src_image_token_length_list,
            "actual_src_face_token_length_list": actual_src_face_token_length_list,
            "max_image_token_length_list": max_image_token_length_list,
            "max_face_token_length_list": max_face_token_length_list
        }

    def encode_transfusion(
        self,
        *prompts: str,
        image_token_length: int,
        src_image_token_lengths: Optional[List[int]] = None,
        max_text_token_length=256,
        max_image_token_length=256,
        max_total_token_length=None,
        uncond_enabled: Optional[Union[bool, List[bool]]] = None,
        uncond_p=0.0,
        add_iw_ih_token=False,
        add_timestep_token=False,
        use_front_boi_token=False,
        pred_text_boi_eos=(True, True, True),
        pred_boi_mode="all",
        add_image_shape_token=False,
        image_sections=None,
    ):
        src_image_token_lengths = ensure_list(src_image_token_lengths)  # t2i for [], instruction tuning for [len_1, ...]
        n_img = len(src_image_token_lengths) + 1
        max_length_left_for_text = (
            max_text_token_length
            - 2     # 2 for <bos> and <eos>
            - n_img * (
                2 +
                (2 if add_iw_ih_token else 0) +
                (1 if add_timestep_token else 0) +
                (2 if add_image_shape_token else 0)
            )
        )
        text_token = self.encode_text(*prompts, uncond_enabled=uncond_enabled, uncond_p=uncond_p, max_length=max_length_left_for_text)

        if image_sections is not None:
            # When using image_sections, src_image_token_lengths and image_token_length are ignored
            image_source = image_sections
        else:
            image_source = src_image_token_lengths + [image_token_length]

        # Combine text and image tokens
        full_token_seq, extra_token_pos, _, _ = self.encode_sequence(
            template='text' + '-image' * n_img,
            token_source=dict(text=[text_token], image=image_source),
            total_length=max_total_token_length if max_total_token_length is not None else (max_text_token_length + max_image_token_length * n_img),
            add_iw_ih_token=add_iw_ih_token,
            add_timestep_token=add_timestep_token,
            use_front_boi_token=use_front_boi_token,
            add_image_shape_token=add_image_shape_token,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        timestep_scatter_index = torch.tensor(extra_token_pos['timestep'], dtype=torch.long) if add_timestep_token else None

        # Build text mask
        pred_text, pred_boi, pred_eos = pred_text_boi_eos
        text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        if pred_text:
            text_mask[1: len(text_token) + 1] = (
                (torch.tensor(text_token, dtype=torch.long) != self.special_token_map["<cfg>"]).float())   # skip <bos>
        if pred_boi:
            src_gen_bois = extra_token_pos.get('src_boi', []) + extra_token_pos.get('boi', [])
            if pred_boi_mode == "all":
                for boi_pos in src_gen_bois:
                    text_mask[boi_pos] = 1.0
            elif pred_boi_mode == "first":
                text_mask[src_gen_bois[0]] = 1.0
            else:
                raise ValueError(f"Invalid pred_boi_mode: {pred_boi_mode}")
        if pred_eos:
            for eos_pos in extra_token_pos['eos']:
                text_mask[eos_pos] = 1.0

        # Source image masks, if available
        if src_image_token_lengths:
            src_image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            for img_start, img_end in zip(extra_token_pos['<img>_start'][:-1], extra_token_pos['<img>_end'][:-1]):
                src_image_mask[img_start: img_end + 1] = 1.0
        # Target image mask
        image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        image_slice = slice(extra_token_pos['<img>_start'][-1], extra_token_pos['<img>_end'][-1] + 1)
        image_mask[image_slice] = 1.0

        if src_image_token_lengths:
            # For text+n*image to image (editing)
            return full_seq_token_tensor, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, image_mask

        # For text to image
        return full_seq_token_tensor, iw_ih_scatter_index, timestep_scatter_index, text_mask, image_mask

    def encode_transfusion_faceid(
        self,
        *prompts: str,
        image_token_length: int,
        src_image_token_lengths: Optional[List[int]] = None,
        src_face_token_lengths: Optional[List[int]] = None,
        max_text_token_length=256,
        max_image_token_length=256,
        max_face_token_length=1,
        uncond_enabled: Optional[Union[bool, List[bool]]] = None,
        uncond_p=0.0,
        add_iw_ih_token=False,
        add_timestep_token=False,
        use_front_boi_token=False,
        pred_text_boi_eos=(True, True, True),
        pred_boi_mode="all",
    ):
        src_image_token_lengths = ensure_list(src_image_token_lengths)  # t2i for [], instruction tuning for [len_1, ...]
        src_face_token_lengths = ensure_list(src_face_token_lengths)  
        n_img = len(src_image_token_lengths) + 1
        n_face = len(src_face_token_lengths)
        # max_text_token_length - number of special tokens to get the max length for language tokens
        max_length_left_for_text = (
            max_text_token_length
            - 2 # 2 for <bos> and <eos>
            - n_img * (2 + (2 if add_iw_ih_token else 0) + (1 if add_timestep_token else 0)) # 2 for <boi> and <eoi>, 2 for <iw> and <ih>, 1 for <timestep>
            - n_face * (2) # 2 for <bof> and <eof>
        )
        # pad or slice to get a sequence of max_length_left_for_text tokens
        text_token = self.encode_text(*prompts, uncond_enabled=uncond_enabled, uncond_p=uncond_p, max_length=max_length_left_for_text)

        if type(max_image_token_length) == int:
            max_image_token_length_sum = max_image_token_length * n_img
        elif type(max_image_token_length) == list:
            # if max_image_token_length is provided as a list of max token length for each image, it should be the sum of all image token lengths
            assert len(max_image_token_length) == n_img, f"max_image_token_length must be a list of length n_img, but got {len(max_image_token_length)}"
            max_image_token_length_sum = sum(max_image_token_length)
        else:
            raise ValueError(f"max_image_token_length must be int or list, but got {type(max_image_token_length)}")
        
        if type(max_face_token_length) == int:
            max_face_token_length_sum = max_face_token_length * n_face
        elif type(max_face_token_length) == list:
            max_face_token_length_sum = sum(max_face_token_length)
        else:
            raise ValueError(f"max_face_token_length must be int or list, but got {type(max_face_token_length)}")
        # Combine text and image tokens
        token_source_dict = dict(
            text=[text_token],
            image=src_image_token_lengths + [image_token_length],
        )
        if src_face_token_lengths: 
            token_source_dict['face'] = src_face_token_lengths
        full_token_seq, extra_token_pos, _, _ = self.encode_sequence(
            template='text' + '-face' * n_face + '-image' * n_img,
            token_source=token_source_dict,
            total_length=max_text_token_length + max_image_token_length_sum + max_face_token_length_sum,
            add_iw_ih_token=add_iw_ih_token,
            add_timestep_token=add_timestep_token,
            use_front_boi_token=use_front_boi_token,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        timestep_scatter_index = torch.tensor(extra_token_pos['timestep'], dtype=torch.long) if add_timestep_token else None

        # Build text mask
        pred_text, pred_boi, pred_eos = pred_text_boi_eos
        text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        if pred_text:
            text_mask[1: len(text_token) + 1] = (
                (torch.tensor(text_token, dtype=torch.long) != self.special_token_map["<cfg>"]).float())   # skip <bos>
        if pred_boi:
            src_gen_bois = extra_token_pos.get('src_boi', []) + extra_token_pos.get('boi', [])
            if pred_boi_mode == "all":
                for boi_pos in src_gen_bois:
                    text_mask[boi_pos] = 1.0
            elif pred_boi_mode == "first":
                text_mask[src_gen_bois[0]] = 1.0
            else:
                raise ValueError(f"Invalid pred_boi_mode: {pred_boi_mode}")
        if pred_eos:
            for eos_pos in extra_token_pos['eos']:
                text_mask[eos_pos] = 1.0

        img_extra_len = (2 if add_iw_ih_token else 0) + (1 if add_timestep_token else 0)
        start_pos = 1 + len(text_token)  + 1  # 1 for <bos> and 1 for <boi> / <bof>
        src_image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        # Source face masks, if available
        if src_face_token_lengths:
            for token_len in src_face_token_lengths:
                src_image_mask[start_pos:start_pos + token_len] = 1.0
                start_pos += token_len + 2 # 2 for <bof>, <eof>, and number of <face>
        start_pos = start_pos + img_extra_len
        # Source image masks, if available
        if src_image_token_lengths:
            for token_len in src_image_token_lengths:
                src_image_mask[start_pos:start_pos + token_len] = 1.0
                start_pos += token_len + img_extra_len + 2
        # Target image mask
        image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        image_mask[start_pos:start_pos + image_token_length] = 1.0
        
        if src_image_token_lengths or src_face_token_lengths:
            # src_image_token_lengths is not (None or empty list), but src_face_token_lengths is (None or empty list)
            # For text+n*image to image (editing)
            return full_seq_token_tensor, iw_ih_scatter_index, timestep_scatter_index, text_mask, src_image_mask, image_mask
        else:
            # For text to image
            return full_seq_token_tensor, iw_ih_scatter_index, timestep_scatter_index, text_mask, image_mask

    def encode_transfusion_mmu(
        self,
        *texts: str,
        image_token_lengths: Union[int, List[int]],
        max_token_length=None,
        add_iw_ih_token=False,
        use_front_boi_token=False,
        add_eos=True,
        text_mask_sections=None,
        use_und_token=False,
    ):
        """
        Encode text and image for multimodal understanding task.

        Parameters
        ----------
        texts: list of str
            List of prompts. Can be simple question, or multi-turn conversations.
        image_token_lengths: int or List[int]
            Number of tokens for each image. If a list, it should have the same length as the number of images.
        max_token_length: int
            Maximum length of the encoded sequence.
        add_iw_ih_token: bool
            Whether to add iw and ih tokens before the image tokens.
        use_front_boi_token: bool
            Whether to put the <boi> token at the front of iw, ih and timestep tokens.
        add_eos: bool
            Whether to add eos token at the end of the sequence.
        text_mask_sections: List[bool]
            List of flags to indicate whether the prompts should be masked as true. If None, all text tokens are
            masked as true. It is useful when the prompts are multi-turn conversations and one wants to ignore
            the user prompts and only predict the assistant prompts.
        use_und_token: bool
            Whether to use the <und_boi> <und_eoi> tokens for the mmu image tokens.
        """
        if isinstance(image_token_lengths, int):
            image_token_lengths = [image_token_lengths]
        n_img = len(image_token_lengths)
        img_extra_len = 2 + (2 if add_iw_ih_token else 0)
        if max_token_length is not None:
            text_max_length = max_token_length - n_img * img_extra_len - sum(image_token_lengths) - 1  # 1 for <bos>
        else:
            text_max_length = None
        text_token, text_lengths = self.encode_text(*texts, max_length=text_max_length, return_lengths=True)

        if text_mask_sections is not None:
            assert isinstance(text_mask_sections, list), f"Expected list, but got {type(text_mask_sections)}."
            assert len(text_mask_sections) >= len(text_lengths), (
                f"Length of text_mask_sections should be equal or longer than the number of texts, "
                f"but got {len(text_mask_sections)} and {len(text_lengths)}."
            )
            text_mask_sections = text_mask_sections[:len(text_lengths)]

        # Combine text and image tokens
        prefix = 'und_' if use_und_token else ''
        full_token_seq, extra_token_pos, _, _ = self.encode_sequence(
            template=f'{prefix}image-' * n_img + 'text',
            token_source={
                "text": [text_token],
                f"{prefix}image": image_token_lengths
            },
            total_length=max_token_length,
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
            add_eos=add_eos,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) if add_iw_ih_token else None
        image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos[f'<{prefix}img>_start'], extra_token_pos[f'<{prefix}img>_end'])
        ]

        image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        for image_slice in image_slices:
            image_mask[image_slice] = 1.0

        # 1 for <bos>
        start_pos = 1 + sum([img_extra_len + image_token_len for image_token_len in image_token_lengths])
        text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        if text_mask_sections is not None:
            for text_len, mask_section in zip(text_lengths, text_mask_sections):
                if mask_section:
                    text_mask[start_pos:start_pos + text_len] = 1.0
                start_pos += text_len
            if start_pos < max_token_length:     # <eos> token
                text_mask[start_pos] = 1.0
        else:
            has_eos = 1 if 'eos' in extra_token_pos and len(extra_token_pos['eos']) > 0 else 0
            text_mask[start_pos:start_pos + len(text_token) + has_eos] = 1.0

        # real_pos is the first position of the <pad> token
        real_pos = torch.tensor([extra_token_pos.get('first_pad', full_seq_token_tensor.shape[0])], dtype=torch.long)

        return full_seq_token_tensor, iw_ih_scatter_index, image_slices, text_mask, image_mask, real_pos

    @staticmethod
    def parse_extra_token_pos(extra_token_pos, prefix, tokens, rng=None):
        if rng is None:
            rng = slice(None)
        image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos[f'<{prefix}>_start'][rng], extra_token_pos[f'<{prefix}>_end'][rng])
        ] if f'<{prefix}>_start' in extra_token_pos and f'<{prefix}>_end' in extra_token_pos else []
        if image_slices:
            image_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for image_slice in image_slices:
                image_mask[image_slice] = True
        else:
            image_mask = None
        return image_slices, image_mask

    def encode_general(
        self,
        template: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        max_token_length: Optional[int] = None,
        add_eos='auto',
        use_text_mask=True,
        add_pad='auto',
        add_bos=True,
        drop_last='auto',
        und_token_type: list[str] = [],
        gen_token_type: list[str] = [],
        audio_token_type: list[str] = [],
    ):
        if sections is None:
            raise ValueError("sections must be provided.")
        if template is None:
            template = '-'.join([section['type'] for section in sections])

        sections = deepcopy(sections)
        token_source = defaultdict(list)
        text_mask_specs = []
        for section in sections:
            if section['type'] == 'text':
                text = self.encode_text(
                    section['text'] if 'text' in section else section['tokens'],
                    uncond_enabled=section.get('uncond_enabled'),
                    uncond_p=section.get('uncond_p'),
                    max_length=section.get('max_length'),
                )
                token_source['text'].append(text)
                text_mask_specs.append(dict(
                    ignore=section.get('ignore', False),
                    start_offset=section.get('start_offset', 0),
                    end_offset=section.get('end_offset', 0),
                ))
            elif section['type'] in ['image', 'gen_image']:
                # Replace gen_image with image for backward compatibility
                template = template.replace('gen_image', 'image')
                token_source['image'].append(dict(
                    length=section['token_length'],
                    iw_ih=section.get('add_iw_ih_token', False),
                    timestep=section.get('add_timestep_token', False),
                    timestep_r=section.get('add_timestep_r_token', False),
                    guidance=section.get('add_guidance_token', False),
                    front_boi=section.get('use_front_boi_token', False),
                    image_shape=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                ))
            elif section['type'] == 'src_image':
                token_source['src_image'].append(dict(
                    length=section['token_length'],
                    iw_ih=section.get('add_iw_ih_token', False),
                    timestep=section.get('add_timestep_token', False),
                    front_boi=section.get('use_front_boi_token', False),
                    image_shape=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                ))
            elif section['type'] == 'und_image':
                token_source['und_image'].append(dict(
                    length=section['token_length'],
                    iw_ih=section.get('add_iw_ih_token', False),
                    timestep=section.get('add_timestep_token', False),
                    front_boi=section.get('use_front_boi_token', False),
                    image_shape=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                ))
            elif section['type'] in ['joint_image', 'cond_joint_image']:
                template = template.replace('cond_joint_image', 'joint_image')  # bc
                token_source['joint_image'].append(dict(
                    length=section['token_length'],
                    iw_ih=section.get('add_iw_ih_token', False),
                    timestep=section.get('add_timestep_token', False),
                    front_boi=section.get('use_front_boi_token', False),
                    image_shape=section.get('add_image_shape_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                ))
            elif section['type'] == 'face':
                token_source['face'].append(dict(
                    length=section['token_length'],
                ))
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        # Combine text and image tokens
        full_token_seq, extra_token_pos, und_token_indices, gen_token_indices = self.encode_sequence(
            template=template,
            token_source=dict(token_source),
            total_length=max_token_length,
            add_eos=add_eos,
            add_pad=add_pad,
            add_bos=add_bos,
            drop_last=drop_last,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)

        iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long) \
            if 'iw_ih' in extra_token_pos else None
        timestep_scatter_index = torch.tensor(extra_token_pos['timestep'], dtype=torch.long) \
            if 'timestep' in extra_token_pos else None
        guidance_scatter_index = torch.tensor(extra_token_pos['guidance'], dtype=torch.long) \
            if 'guidance' in extra_token_pos else None
        timestep_r_scatter_index = torch.tensor(extra_token_pos['timestep_r'], dtype=torch.long) \
            if 'timestep_r' in extra_token_pos else None        
        cond_timestep_scatter_index = torch.tensor(extra_token_pos['cond_timestep'], dtype=torch.long) \
            if 'cond_timestep' in extra_token_pos else None
        gen_timestep_scatter_index = torch.tensor(extra_token_pos['gen_timestep'], dtype=torch.long) \
            if 'gen_timestep' in extra_token_pos else None

        # Gen image mask
        src_image_slices, src_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'src_img', full_seq_token_tensor)
        gen_image_slices, gen_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'img', full_seq_token_tensor)
        # Und image mask
        und_image_slices, und_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'und_img', full_seq_token_tensor)
        # Joint image
        joint_image_slices, _ = self.parse_extra_token_pos(
            extra_token_pos, 'joint_img', full_seq_token_tensor)
        # Face image mask
        face_image_slices, face_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'face', full_seq_token_tensor)
        # All image slices (src_image, gen_image, und_image)
        all_image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_img>_start'], extra_token_pos['<all_img>_end'])
        ] if '<all_img>_start' in extra_token_pos and '<all_img>_end' in extra_token_pos else []

        # Text mask
        text_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<text>_start'], extra_token_pos['<text>_end'])
        ] if '<text>_start' in extra_token_pos and '<text>_end' in extra_token_pos else []
        assert len(text_slices) <= len(text_mask_specs), \
            (f"Number of text slices ({len(text_slices)}) should be less than or equal to "
             f"number of text mask specs ({len(text_mask_specs)})")
        if use_text_mask:
            text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            for text_slice, mask_spec in zip(text_slices, text_mask_specs):
                if not mask_spec['ignore']:
                    real_slice = slice(
                        text_slice.start + mask_spec['start_offset'],
                        text_slice.stop + mask_spec['end_offset']
                    )
                    text_mask[real_slice] = 1.0
        else:
            text_mask = None

        # real_pos is the first position of the <pad> token
        real_pos = torch.tensor(extra_token_pos.get('first_pad', [full_seq_token_tensor.shape[0]]), dtype=torch.long)

        return TokenizerEncodeOutput(
            tokens=full_seq_token_tensor,
            iw_ih_scatter_index=iw_ih_scatter_index,
            timestep_scatter_index=timestep_scatter_index,
            timestep_r_scatter_index=timestep_r_scatter_index,
            gen_timestep_r_scatter_index=timestep_r_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            text_slices=text_slices,
            src_image_slices=src_image_slices,
            vae_image_slices=src_image_slices,
            gen_image_slices=gen_image_slices,
            und_image_slices=und_image_slices,
            vit_image_slices=und_image_slices,
            joint_image_slices=joint_image_slices,
            face_image_slices=face_image_slices,
            text_mask=text_mask,
            src_image_mask=src_image_mask,
            vae_image_mask=src_image_mask,
            gen_image_mask=gen_image_mask,
            und_image_mask=und_image_mask,
            vit_image_mask=und_image_mask,
            face_image_mask=face_image_mask,
            real_pos=real_pos,
            all_image_slices=all_image_slices,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
        )

    def encode_lm(
            self,
            *texts,
            max_token_length=None,
            add_eos='auto',
            return_text_mask=True,
            text_mask_sections=None,
    ):
        text_max_length = max_token_length - 1 if max_token_length is not None else None
        text_token, text_lengths = self.encode_text(*texts, max_length=text_max_length, return_lengths=True)

        if return_text_mask and text_mask_sections is not None:
            assert isinstance(text_mask_sections, list), f"Expected list, but got {type(text_mask_sections)}."
            assert len(text_mask_sections) >= len(text_lengths), (
                f"Length of text_mask_sections should be equal or longer than the number of texts, "
                f"but got {len(text_mask_sections)} and {len(text_lengths)}."
            )
            text_mask_sections = text_mask_sections[:len(text_lengths)]

        # Get text tokens
        full_token_seq, extra_token_pos, _, _ = self.encode_sequence(
            template='text',
            token_source=dict(text=[text_token]),
            total_length=max_token_length,
            add_eos=add_eos,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        real_pos = torch.tensor([full_seq_token_tensor.shape[0]], dtype=torch.long)

        if return_text_mask:
            text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
            start_pos = 1
            if text_mask_sections is not None:
                for text_len, (mask_section, offset) in zip(text_lengths, text_mask_sections):
                    if mask_section:
                        text_mask[start_pos + offset:start_pos + text_len] = 1.0
                    start_pos += text_len
                # When use text_mask_sections, the belief that <eos> is already included in `text_token`.
            else:
                end_pos = extra_token_pos.get('first_pad', [full_seq_token_tensor.shape[0]])[0]
                text_mask[start_pos:end_pos] = 1.0
            return full_seq_token_tensor, real_pos, text_mask

        return full_seq_token_tensor, real_pos

    def encode_transfusion_editing2(
            self,
            *prompts: str,
            src_image_token_lengths: List[int],
            tgt_image_token_length: int,
            max_text_token_length: int,
            max_total_token_length: int,
            uncond_enabled: list = None,
            uncond_p: float = 0.0,
            dtype=torch.long,
            add_iw_ih_token=False,
            use_front_boi_token=False,
    ):
        """ For subject-driven instruction tuning task.
        A left padding variant of encode_transfusion() before refactor """
        n_img = len(src_image_token_lengths) + 1

        max_length_left_for_text = (
            max_text_token_length
            - 2 # 2 for <bos> and <eos>
            - n_img * (2 + (2 if add_iw_ih_token else 0)) # 2 for <boi> and <eoi>, 2 for <iw> and <ih>
        )

        text_token = self.encode_text(
            *prompts, uncond_enabled=uncond_enabled,
            uncond_p=uncond_p,
            max_length=max_length_left_for_text,
        )

        # Combine text and image tokens
        full_seq_token, extra_token_pos, _, _ = self.encode_sequence(
            template='text' + '-image' * n_img,
            token_source=dict(text=[text_token], image=src_image_token_lengths + [tgt_image_token_length]),
            total_length=max_total_token_length,
            add_iw_ih_token=add_iw_ih_token,
            use_front_boi_token=use_front_boi_token,
        )
        full_seq_token_tensor = torch.tensor(full_seq_token, dtype=dtype)
        if add_iw_ih_token:
            iw_ih_scatter_index = torch.tensor(extra_token_pos['iw_ih'], dtype=torch.long)
            iw_ih_tokens = 2
        else:
            iw_ih_scatter_index = None
            iw_ih_tokens = 0

        # Calculate text mask, src image mask, tgt image mask
        text_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        text_mask[1: len(text_token)+1] = 1.0 # 1 for <bos>

        src_image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        start_pos = 1 + len(text_token) + iw_ih_tokens + 1 # 1 for <bos> and 1 for <boi>
        for token_len in src_image_token_lengths:
            src_image_mask[start_pos:start_pos + token_len] = 1.0
            start_pos += token_len + 2 + iw_ih_tokens  # 2 for <eoi> of last image and <boi> of next image

        tgt_image_mask = torch.zeros_like(full_seq_token_tensor, dtype=torch.float32)
        tgt_image_mask[start_pos:start_pos + tgt_image_token_length] = 1.0

        return full_seq_token_tensor, iw_ih_scatter_index, text_mask, src_image_mask, tgt_image_mask

    def encode_visual_audio_sequence(self, audio_token, text, cfg_enabled=False, max_len=1025):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        sequence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > 750:
            audio_token = audio_token[:750]
        token = self.tokenizer.encode(text)
        if cfg_enabled:
            token = [self.special_token_map["<cfg>"]] * len(token)
        if len(token) > 227:
            token = token[:227]
        else:
            token = token + [self.special_token_map["<pad>"]] * (227 - len(token))
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * 40
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + token  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_audio_sequence_no_text(self, audio_token, max_len=1025, max_audio_len=750, max_clip_frame=40):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > max_audio_len:
            audio_token = audio_token[:max_audio_len]
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_sequence_no_text_for_infer(self, max_clip_frame=40):
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<boa>"]]
        )
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_t5_audio_sequence(self, audio_token, max_len=1025, max_audio_len=750, max_clip_frame=40, max_t5_len=227):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        audio_token = audio_token.tolist()
        if len(audio_token) > max_audio_len:
            audio_token = audio_token[:max_audio_len]
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + audio_token
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor

    def encode_visual_t5_audio_placeholder(self, max_len=1025, max_audio_len=750, max_clip_frame=40, max_t5_len=227):
        """ audio sequence format: <bos> <bov> clip <eov> <bot> T5 <eot> <boa> audio_token <eoa> <eos>
        squence length = 1 + 1 + 40(clip) + 1 +1 + 227(t5) +1 +1 + 750(audio) +1 +1
        """
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
            + [self.special_token_map["<audio>"]] * max_audio_len
            + [self.special_token_map["<eoa>"], self.eos_token]
        )
        if len(full_seq_token) < max_len:
            full_seq_token += [self.special_token_map["<pad>"]] * (max_len - len(full_seq_token))
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor
    
    def encode_visual_t5_for_audio_infer(self, max_clip_frame=40, max_t5_len=227):
        full_seq_token = (
            [self.bos_token, self.special_token_map["<bov>"]]
            + [self.special_token_map["<video>"]] * max_clip_frame
            + [self.special_token_map["<eov>"], self.special_token_map["<bot>"]]
            + [self.special_token_map["<text>"]] * max_t5_len  #
            + [self.special_token_map["<eot>"], self.special_token_map["<boa>"]]
        )
        full_seq_tensor = torch.tensor(full_seq_token).long()
        return full_seq_tensor
