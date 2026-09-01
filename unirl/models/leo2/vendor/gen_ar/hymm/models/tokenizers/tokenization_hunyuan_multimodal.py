import random
from collections import defaultdict
from typing import Optional, Any, Union, Literal
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from diffusers.utils import BaseOutput

from hymm.utils.helpers import default
from hymm.utils.image_base import ImageInfo, JointImageInfo, CondImage
from hymm.utils.video_base import VideoInfo
from hymm.utils.audio_base import AudioInfo
from .conversation import Conversation


class TokenizerEncodeOutput(BaseOutput):
    tokens: torch.Tensor = None
    # ======== slices ========
    # -- text --
    text_slices: Optional[list[slice]] = None
    # -- image --
    gen_image_slices: Optional[list[slice]] = None
    vae_image_slices: Optional[list[slice]] = None
    vit_image_slices: Optional[list[slice]] = None
    joint_image_slices: Optional[list[slice]] = None
    all_image_slices: Optional[list[slice]] = None
    # -- video --
    gen_video_slices: Optional[list[slice]] = None
    vae_video_slices: Optional[list[slice]] = None
    vit_video_slices: Optional[list[slice]] = None
    vit_video_context_slices: Optional[list[slice]] = None
    joint_video_slices: Optional[list[slice]] = None
    all_video_slices: Optional[list[slice]] = None
    # -- audio --
    gen_audio_slices: Optional[list[slice]] = None
    vae_audio_slices: Optional[list[slice]] = None
    vit_audio_slices: Optional[list[slice]] = None
    joint_audio_slices: Optional[list[slice]] = None
    all_audio_slices: Optional[list[slice]] = None
    # -- All media --
    all_media_slices: Optional[list[slice]] = None

    # ======== mask ========
    # -- text --
    text_mask: Optional[torch.Tensor] = None
    # -- image --
    gen_image_mask: Optional[torch.Tensor] = None
    vae_image_mask: Optional[torch.Tensor] = None
    vit_image_mask: Optional[torch.Tensor] = None
    # -- video --
    gen_video_mask: Optional[torch.Tensor] = None
    vae_video_mask: Optional[torch.Tensor] = None
    vit_video_mask: Optional[torch.Tensor] = None
    # -- audio --
    gen_audio_mask: Optional[torch.Tensor] = None
    vae_audio_mask: Optional[torch.Tensor] = None
    vit_audio_mask: Optional[torch.Tensor] = None

    # ======== position related ========
    real_pos: Optional[torch.Tensor] = None
    und_token_indices: Optional[torch.Tensor] = None
    gen_token_indices: Optional[torch.Tensor] = None
    audio_token_indices: Optional[torch.Tensor] = None
    # -- image --
    guidance_scatter_index: Optional[torch.Tensor] = None
    cond_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_timestep_r_scatter_index: Optional[torch.Tensor] = None
    # -- video --
    cond_video_timestep_scatter_index: Optional[torch.Tensor] = None
    gen_video_timestep_scatter_index: Optional[torch.Tensor] = None
    # -- audio --
    gen_audio_timestep_scatter_index: Optional[torch.Tensor] = None


class HunyuanMultimodalTokenizerFast(PreTrainedTokenizerFast):
    """
    Tokenizer for Hunyuan Multimodal models, utilizing a fast tokenizer backend.
    This tokenizer extends the PreTrainedTokenizerFast from Hugging Face Transformers
    for multimodal tasks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A convenience mapping for special tokens
        special_tokens = self.special_tokens_map.get('additional_special_tokens', [])
        if len(special_tokens) > 0:
            special_token_ids = self.convert_tokens_to_ids(special_tokens)
            self._sp_dict = dict(zip(special_tokens, special_token_ids))
        else:
            self._sp_dict = dict()

        # used by AutoTokenizer to construct the tokenizer
        # additional_special_tokens may not be written into special_tokens_map
        # but added_tokens_decoder still has id->string mapping.
        if len(self._sp_dict) == 0 and getattr(self, "added_tokens_decoder", None):
            self._sp_dict = {str(tok): idx for idx, tok in self.added_tokens_decoder.items()}
        # Assign commonly used special tokens to attributes for easy access.
        self.setup_special_tokens()

    def setup_special_tokens(self):
        # Define names for commonly used special tokens
        predefined_name_mapping = {
            "slowly_think": "<｜hy_place▁holder▁no▁153｜>",  # 长链
            "quickly_think": "<｜hy_place▁holder▁no▁154｜>",  # 短链
            "think": "<｜hy_place▁holder▁no▁149｜>",  # 思考内容
            "end_of_think": "<｜hy_place▁holder▁no▁150｜>",  # 思考内容结束
            "answer": "<｜hy_place▁holder▁no▁151｜>",  # 答案内容
            "end_of_answer": "<｜hy_place▁holder▁no▁152｜>",  # 答案内容结束
            "boi": "<｜hy_place▁holder▁no▁100｜>",  # 图片开始
            "eoi": "<｜hy_place▁holder▁no▁101｜>",  # 图片结束
            "img": "<｜hy_place▁holder▁no▁102｜>",
        }  
        for name, mapping in predefined_name_mapping.items():
            setattr(self, f"{name}_token", mapping)
            setattr(self, f"{name}_token_id", self.convert_tokens_to_ids(mapping))

        if len(self._sp_dict) > 0:
            name_mapping = dict(
                boa_token="<｜boa｜>",
                eoa_token="<｜eoa｜>",
                bov_token="<｜bov｜>",
                eov_token="<｜eov｜>",
                audio_token="<｜audio｜>",
                video_token="<｜video｜>",
                cfg_token="<｜cfg｜>",
                timestep_token="<｜timestep｜>",
                timestep_r_token="<｜timestep_r｜>",
                guidance_token="<｜guidance｜>",
                joint_img_sep_token="<｜joint_img_sep｜>",
                # for extended cot types
                recaption_token="<｜recaption｜>",
                end_of_recaption_token="<｜end_of_recaption｜>",
                # for grounding
                ref_token="<｜ref｜>",
                end_of_ref_token="<｜end_of_ref｜>",
                quad_token="<｜quad｜>",
                end_of_quad_token="<｜end_of_quad｜>",
            )
            for name, token in name_mapping.items():
                if token in self._sp_dict:
                    setattr(self, name, token)
                    setattr(self, f"{name}_id", self._sp_dict[token])

    def size_token(self, size: int):
        assert len(self._sp_dict) > 0, "Size tokens are not defined in the tokenizer."
        return f"<｜img_size_{size}｜>"

    def size_token_id(self, size: int):
        return self._sp_dict[self.size_token(size)]

    def ratio_token(self, ratio_idx: int):
        assert len(self._sp_dict) > 0, "Ratio tokens are not defined in the tokenizer."
        return f"<｜img_ratio_{ratio_idx}｜>"

    def ratio_token_id(self, ratio_idx: int):
        return self._sp_dict[self.ratio_token(ratio_idx)]

    def tw_th_token(self, k: int):
        """k: token length per dimension. For add_tw_th_token."""
        assert len(self._sp_dict) > 0, "tw_th tokens are not defined in the tokenizer."
        return f"<｜img_tw_th_{k}｜>"

    def tw_th_token_id(self, k: int):
        return self._sp_dict[self.tw_th_token(k)]

    def duration_token(self, duration_idx: int):
        duration_token_repr = f"<｜duration_{duration_idx}｜>"
        assert duration_token_repr in self._sp_dict, "Duration tokens are not defined in the tokenizer."
        return duration_token_repr

    def duration_token_id(self, duration_idx: int):
        return self._sp_dict[self.duration_token(duration_idx)]

    def relation_token(self, relation_idx: int, end: bool = False):
        assert len(self._sp_dict) > 0, "Relation tokens are not defined in the tokenizer."
        if end:
            return f"<｜end_of_relation_{relation_idx}｜>"
        return f"<｜relation_{relation_idx}｜>"

    def relation_token_id(self, relation_idx: int, end: bool = False):
        return self._sp_dict[self.relation_token(relation_idx, end)]

    def x_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position x tokens are not defined in the tokenizer."
        return f"<｜pos_x_{pos}｜>"

    def x_token_id(self, pos: int):
        return self._sp_dict[self.x_token(pos)]

    def y_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position y tokens are not defined in the tokenizer."
        return f"<｜pos_y_{pos}｜>"

    def y_token_id(self, pos: int):
        return self._sp_dict[self.y_token(pos)]

    def z_token(self, pos: int):
        assert len(self._sp_dict) > 0, "Position z tokens are not defined in the tokenizer."
        return f"<｜pos_z_{pos}｜>"

    def z_token_id(self, pos: int):
        return self._sp_dict[self.z_token(pos)]

    def get_img_token(self):
        if hasattr(self, "img_token"):
            return self.img_token
        else:
            return self.convert_ids_to_tokens(len(self) - 1)

    def encode_text(
            self,
            *texts,
            uncond_enabled: Optional[bool | list[bool]] = None,
            uncond_p: Optional[float] = None,
            max_length: Optional[int] = None,
            pad: Optional[str] = None,
            return_lengths: bool = False,
            uncond_length: Literal["equal", "single"] = "equal",
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
        uncond_length: str
            Method to determine the length of the unconditioned text. Can be 'equal' or 'single'.
            If 'equal', the length of the unconditioned tokens will be the same as the conditioned tokens.
            If 'single', one cfg_token will be used for the entire unconditioned text.
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
                print(
                    f"Warning: Text length exceeds the max_length({max_length}). The remaining texts will be ignored: "
                    f"{text[:80]}..."
                )
                break
            # Set add_special_tokens=False to avoid adding <bos> token in some LLMs.
            if isinstance(text, str):
                text_token = self.encode(text, add_special_tokens=False)
            else:
                text_token = text
            if uncond_flag and do_uncond_drop:
                if uncond_length == "equal":
                    text_token = [self.cfg_token_id] * len(text_token)
                elif uncond_length == "single":
                    text_token = [self.cfg_token_id]
                else:
                    raise ValueError(f"Unsupported uncond_length method: {uncond_length}.")
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

    def _add_meta_info_token(
            self,
            token_seq,
            token_count,
            extra_token_pos,
            add_timestep_token: bool = False,
            add_timestep_r_token: bool = False,
            add_image_shape_token: bool = False,
            add_tw_th_token: bool = False,
            add_guidance_token: bool = False,
            add_video_shape_token: bool = False,
            base_size=None,
            ratio_idx=None,
            token_height=None,
            token_width=None,
            duration_idx=None,
            media_type=None,
            und_token_type: list[str] = None,
            gen_token_type: list[str] = None,
            und_token_indices: list[int] = None,
            gen_token_indices: list[int] = None,
            audio_token_type: list[str] = None,
            audio_token_indices: list[int] = None,
            token_count_start: int = 0,
    ):
        add_mot_indices = partial(
            self.process_mot_indices,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
            und_token_type=default(und_token_type, []),
            gen_token_type=default(gen_token_type, []),
            audio_token_indices=audio_token_indices,
            audio_token_type=default(audio_token_type, []),
        )
        if add_tw_th_token:
            h_token = token_height[0] if isinstance(token_height, list) else token_height
            w_token = token_width[0] if isinstance(token_width, list) else token_width
            token_seq.extend([self.tw_th_token_id(h_token), self.tw_th_token_id(w_token)])
            token_count += 2
            add_mot_indices(token_type="vae_info", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        elif add_image_shape_token:
            token_seq.extend([self.size_token_id(base_size), self.ratio_token_id(ratio_idx)])
            token_count += 2
            add_mot_indices(token_type="vae_info", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        if add_video_shape_token:
            token_seq.extend([
                self.duration_token_id(duration_idx), self.size_token_id(base_size), self.ratio_token_id(ratio_idx)
            ])
            token_count += 3
            token_count_start = token_count
        if add_timestep_token:
            token_seq.extend([self.timestep_token_id])
            extra_token_pos['timestep'].append(token_count)
            if media_type is not None:
                if media_type == "gen_image":
                    extra_token_pos['gen_timestep'].append(token_count)
                elif media_type in ["cond_joint_image", "cond_vae_image"]:
                    extra_token_pos['cond_timestep'].append(token_count)
                elif media_type == "gen_video":
                    extra_token_pos['gen_video_timestep'].append(token_count)
                elif media_type == "gen_audio":
                    extra_token_pos['gen_audio_timestep'].append(token_count)
                else:
                    raise ValueError(f"Unsupported image type: {media_type}.")
            token_count += 1
            add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        # guidance token is front of timestep_r token
        if add_guidance_token:
            token_seq.extend([self.guidance_token_id])
            extra_token_pos['guidance'].append(token_count)
            token_count += 1
            add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        if add_timestep_r_token:
            token_seq.extend([self.timestep_r_token_id])
            extra_token_pos['gen_timestep_r'].append(token_count)
            token_count += 1
            add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
            token_count_start = token_count
        return token_count, token_count_start

    def _shorten_text(self, text):
        import re

        if hasattr(self, "cfg_token"):
            cfg_token = re.escape(self.cfg_token)
            text = re.sub(f"({cfg_token})+", lambda m: f"[{self.cfg_token}]{{{len(re.escape(m.group(0))) // len(cfg_token)}}}", text)

        if hasattr(self, "img_token"):
            img_token = re.escape(self.img_token)
            text = re.sub(f"({img_token})+", lambda m: f"[{self.img_token}]{{{len(re.escape(m.group(0))) // len(img_token)}}}", text)

        if hasattr(self, "video_token"):
            video_token = re.escape(self.video_token)
            text = re.sub(f"({video_token})+", lambda m: f"[{self.video_token}]{{{len(re.escape(m.group(0))) // len(video_token)}}}", text)

        if hasattr(self, "audio_token"):
            audio_token = re.escape(self.audio_token)
            text = re.sub(f"({audio_token})+", lambda m: f"[{self.audio_token}]{{{len(re.escape(m.group(0))) // len(audio_token)}}}", text)

        pad_token = re.escape(self.pad_token)
        text = re.sub(f"({pad_token})+", lambda m: f"[{self.pad_token}]{{{len(re.escape(m.group(0))) // len(pad_token)}}}", text)

        return text

    @staticmethod
    def process_mot_indices(
            token_type: str,
            token_indices: list[int],
            und_token_indices: list[int],
            gen_token_indices: list[int],
            und_token_type: list[str],
            gen_token_type: list[str],
            audio_token_indices: list[int] = None,     # The third stream
            audio_token_type: list[str] = None,
    ):
        if token_type in und_token_type:
            und_token_indices.extend(token_indices)
        elif token_type in gen_token_type:
            gen_token_indices.extend(token_indices)
        elif audio_token_type is not None and token_type in audio_token_type:
            assert audio_token_indices is not None, \
                "audio_token_indices should be provided when audio_token_type is not None."
            audio_token_indices.extend(token_indices)

    def encode_sequence(
            self,
            template: str,
            token_source: dict[str, list[list[int] | dict[str, Any]]],
            total_length=None,
            add_eos=True,
            add_pad=True,
            add_bos=True,
            drop_last: str | bool = 'auto',
            add_image_shape_token=False,
            add_tw_th_token=False,
            add_video_shape_token=False,
            und_token_type: Optional[list[str]] = None,
            gen_token_type: Optional[list[str]] = None,
            audio_token_type: Optional[list[str]] = None,
    ):
        """
        Encode a sequence of tokens based on the provided token source.

        Args:
            template: str
                Template of the sequence. E.g., "text-image" means the sequence is composed of text and an image.
                "text-image-image" means the sequence is composed of text and two images.
            token_source (dict[str, list[list[int] | dict[str, Any]]]): Token source for each key in the template, in order.
                - text: List[List[int]]. Each List[int] is a sequence of tokenized text tokens.
                    - start_offset: int. Optional. Offset the text mask correspondingly.
                    - end_offset: int. Optional. Offset the text mask correspondingly. It is a trick to label
                                  the following adjacent tokens as learnable tokens (i.e., not ignored), which is
                                  useful for media-related tokens(e.g., boi, size_token, ratio_token).
                - gen_image: dict. Required keys: 'length' (int). Optional keys: 'timestep' (bool), 'guidance' (bool),
                             'image_shape' (bool), 'base_size' (int), 'ratio_idx' (int).
                - cond_joint_image: dict. The same as gen_image, but 'length' is a list of two integers.
                - cond_vae_image: dict. The same as gen_image.
                - cond_vit_image: dict. The same as gen_image.
                - gen_video: dict. Required keys: 'length' (int). Optional keys: 'timestep' (bool), 'guidance' (bool),
                             'video_shape' (bool), 'base_size' (int), 'duration_idx' (int), 'ratio_idx' (int).
            total_length: int
                Total length of the encoded sequence, include padding tokens.
            add_eos: bool or 'auto'
                Whether to add eos token at the end of the sequence. If True, always add eos token. If 'auto',
                add eos token only when the total_length is not reached and the last token is not <eos>.
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
            add_video_shape_token: bool
                Whether to add video shape token before the video tokens. (Right before the <timestep> token)

        Returns
        -------
        token_seq: list
            Encoded token sequence.
        extra_token_pos: dict
            Positions of extra tokens. E.g., iw_ih, timestep.
        und_token_indices: list
            Indices of understanding tokens in the sequence.
        gen_token_indices: list
            Indices of vae generated tokens in the sequence.
        """
        if drop_last is True and total_length is None:
            raise ValueError("total_length should be provided when drop_last is True.")

        keys = template.split('-')
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
        gen_token_indices = [] # default: timestep, guidance, shape, vae
        audio_token_indices = [] # default: audio_timestep, audio_shape, audio
        add_mot_indices = partial(
            self.process_mot_indices,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
            und_token_type=default(und_token_type, []),
            gen_token_type=default(gen_token_type, []),
            audio_token_indices=audio_token_indices,
            audio_token_type=default(audio_token_type, []),
        )
        if add_bos and self.bos_token_id is not None:   # Some tokenizer (e.g., Qwen2) has no bos token (None value).
            token_seq.append(self.bos_token_id)
            add_mot_indices(token_type="bos", token_indices=[token_count])
            token_count += 1
        # If drop_last is True, we check the token_count on the fly and exit the loop if the total_length is reached.
        # This check is only applied to the block tokens. Block tokens mean the tokens that are unsplittable, like
        # image tokens, face tokens. Text tokens are splittable, so we don't need to check the token_count for text.
        # If the loop is broken by drop_last, we don't add the eos token at the end because the sequence is not complete.
        drop_last_break = False
        for i, key in enumerate(keys):
            source = token_source[key][index_indicator[key]]
            token_count_start = token_count
            
            if key == "text":
                token_seq.extend(source)  # text token sequence
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)
                add_mot_indices(token_type="text", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

            elif key == "gen_image":
                gen_image_template = source.get("gen_image_template", "default")
                extra_count = \
                    (2 if gen_image_template == "default" else 0) \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (1 if source['add_guidance_token'] else 0) \
                    + (1 if source['add_timestep_r_token'] else 0) \
                    + (2 if source['add_image_shape_token'] else 0) \
                    + (2 if source['add_tw_th_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                # boi token
                if gen_image_template == "default":
                    token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise using default <boi>
                    extra_token_pos["boi"].append(token_count)
                    add_mot_indices(token_type="boi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                # meta tokens
                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_guidance_token=source['add_guidance_token'],
                    add_timestep_r_token=source['add_timestep_r_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    add_tw_th_token=source['add_tw_th_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length'] # token number
                )
                extra_token_pos["<img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)

                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                # eoi token
                if gen_image_template == "default":
                    token_seq.extend([self.eoi_token_id])
                    extra_token_pos["eoi"].append(token_count)
                    add_mot_indices(token_type="eoi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

            elif key == "cond_joint_image":
                assert isinstance(source['length'], list) and len(
                    source['length']) == 2, "cond_joint_image length should be a list of two integers"
                # 2 + 1 means: boi, eoi, joint_img_sep
                extra_count = \
                    2 + 1 \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (2 if source['add_image_shape_token'] or source['add_tw_th_token'] else 0)
                if drop_last is True and token_count + extra_count + sum(source['length']) > total_length:
                    drop_last_break = True
                    break
                token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise using default <boi>
                extra_token_pos["boi"].append(token_count)
                token_count += 1
                add_mot_indices(token_type="boi", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    add_tw_th_token=source['add_tw_th_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length'][0]
                )
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<joint_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length'][0]
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)
                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                token_seq.extend([self.joint_img_sep_token_id])
                extra_token_pos["joint_img_sep"].append(token_count)
                add_mot_indices(token_type="joint_image_sep", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                token_seq.extend(
                    [self.img_token_id] * source['length'][1]
                )
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length'][1]
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<joint_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)

                add_mot_indices(token_type="vit", token_indices=list(range(token_count_start, token_count)))

                token_seq.extend(
                    [self.eoi_token_id]
                )
                extra_token_pos["eoi"].append(token_count)
                add_mot_indices(token_type="eoi", token_indices=[token_count])
                token_count += 1  # <eoi>
                token_count_start = token_count
                
            elif key == "cond_vae_image":
                gen_image_template = source.get("gen_image_template", "default")
                # 2 means: boi, eoi
                extra_count = \
                    (2 if gen_image_template == "default" else 0) \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (2 if source['add_image_shape_token'] or source['add_tw_th_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                # boi token
                if gen_image_template == "default":
                    token_seq.append(self.boi_token_id)  # Use patched boi for Janus, otherwise using default <boi>
                    extra_token_pos["boi"].append(token_count)
                    add_mot_indices(token_type="boi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_image_shape_token=source['add_image_shape_token'],
                    add_tw_th_token=source['add_tw_th_token'],
                    base_size=source.get('base_size'),
                    ratio_idx=source.get('ratio_idx'),
                    token_height=source.get('token_height'),
                    token_width=source.get('token_width'),
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )

                token_seq.extend(
                    [self.img_token_id] * source['length']
                )
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)
                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))

                if gen_image_template == "default":
                    token_seq.extend([self.eoi_token_id])
                    extra_token_pos["eoi"].append(token_count)
                    add_mot_indices(token_type="eoi", token_indices=[token_count])
                    token_count += 1  # <eoi>
                    token_count_start = token_count

            elif key == "cond_vit_image":
                # 2 means: boi, eoi
                extra_count = 2
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                if hasattr(self, "boi_token_id"):
                    token_seq.append(self.boi_token_id)
                    add_mot_indices(token_type="boi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                if hasattr(self, "img_token_id"):
                    token_seq.extend([self.img_token_id] * source['length'])
                else:
                    # If not img_token_id defined, but we still need to fill the image tokens,
                    # we use the last token id representing the image token.
                    token_seq.extend([len(self) - 1] * source['length'])
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)
                add_mot_indices(token_type="vit", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                if hasattr(self, "eoi_token_id"):
                    token_seq.append(self.eoi_token_id)
                    extra_token_pos["eoi"].append(token_count)
                    add_mot_indices(token_type="eoi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

            elif key == "gen_video":
                gen_template = source["gen_template"]
                # 2 means bov and eov
                extra_count = \
                    (2 if gen_template == "default" else 0) \
                    + (1 if source['add_timestep_token'] else 0) \
                    + (1 if source['add_guidance_token'] else 0) \
                    + (1 if source['add_timestep_r_token'] else 0) \
                    + (3 if source['add_video_shape_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                # bov token
                if gen_template == "default":
                    token_seq.append(self.bov_token_id)
                    extra_token_pos["bov"].append(token_count)
                    add_mot_indices(token_type="bov", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    add_guidance_token=source['add_guidance_token'],
                    add_timestep_r_token=source['add_timestep_r_token'],
                    add_video_shape_token=source['add_video_shape_token'],
                    base_size=source['base_size'],
                    ratio_idx=source['ratio_idx'],
                    duration_idx=source['duration_idx'],
                    media_type=key,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    token_count_start=token_count_start,
                )
                token_seq.extend(
                    [self.video_token_id] * source['length'] # token number
                )
                extra_token_pos["<video>_start"].append(token_count)
                extra_token_pos["<all_video>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<video>_end"].append(token_count - 1)
                extra_token_pos["<all_video>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)

                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                # eov token
                if gen_template == "default":
                    token_seq.extend([self.eov_token_id])
                    extra_token_pos["eov"].append(token_count)
                    add_mot_indices(token_type="eov", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

            elif key == "cond_vae_video":
                # Source-video VAE latent condition
                extra_count = 0
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break
                token_seq.extend([self.video_token_id] * source['length'])
                extra_token_pos["<vae_video>_start"].append(token_count)
                extra_token_pos["<all_video>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vae_video>_end"].append(token_count - 1)
                extra_token_pos["<all_video>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)
                add_mot_indices(token_type="vae", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

            elif key == "cond_vit_video":
                # Native Qwen video layout per temporal patch:
                # <timestamp> + <vision_start> + <video_pad> * length + <vision_end>.
                if source.get("timestamp") is None:
                    raise ValueError(
                        "cond_vit_video requires a timestamp for every temporal tubelet."
                    )
                timestamp_tokens = self.encode(
                    f"<{float(source['timestamp']):.1f} seconds>",
                    add_special_tokens=False,
                )
                # 每个temporal patch需要两部分token
                # <timestamp>和<boi>, <eoi>
                extra_count = 2 + len(timestamp_tokens)  # timestamp, boi, eoi
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                context_start = token_count
                token_seq.extend(timestamp_tokens)
                extra_token_pos["<text>_start"].append(token_count)
                token_count += len(timestamp_tokens)
                extra_token_pos["<text>_end"].append(token_count - 1)
                add_mot_indices(
                    token_type="text",
                    token_indices=list(range(token_count_start, token_count)),
                )
                token_count_start = token_count

                token_seq.append(self.boi_token_id)
                add_mot_indices(token_type="boi", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

                if not hasattr(self, "video_token_id"):
                    raise ValueError("Native Qwen video conditioning requires `video_token_id`.")
                token_seq.extend([self.video_token_id] * source['length'])
                extra_token_pos["<vit_video>_start"].append(token_count)
                extra_token_pos["<all_video>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<vit_video>_end"].append(token_count - 1)
                extra_token_pos["<all_video>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)
                add_mot_indices(token_type="vit", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                if hasattr(self, "eoi_token_id"):
                    token_seq.append(self.eoi_token_id)
                    extra_token_pos["eoi"].append(token_count)
                    add_mot_indices(token_type="eoi", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                extra_token_pos["<vit_video_context>_start"].append(context_start)
                extra_token_pos["<vit_video_context>_end"].append(token_count - 1)

            elif key == "gen_audio":
                gen_template = source["gen_template"]
                # 2 means boa and eoa
                extra_count = \
                    (2 if gen_template == "default" else 0) \
                    + (1 if source['add_timestep_token'] else 0)
                if drop_last is True and token_count + extra_count + source['length'] > total_length:
                    drop_last_break = True
                    break

                if gen_template == "default":
                    token_seq.append(self.boa_token_id)
                    extra_token_pos["boa"].append(token_count)
                    add_mot_indices(token_type="boa", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

                token_count, token_count_start = self._add_meta_info_token(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source['add_timestep_token'],
                    media_type=key,
                    audio_token_type=audio_token_type,
                    audio_token_indices=audio_token_indices,
                    token_count_start=token_count_start,
                )
                token_seq.extend(
                    [self.audio_token_id] * source['length']  # token number
                )
                extra_token_pos["<audio>_start"].append(token_count)
                extra_token_pos["<all_audio>_start"].append(token_count)
                extra_token_pos["<all_media>_start"].append(token_count)
                token_count += source['length']
                extra_token_pos["<audio>_end"].append(token_count - 1)
                extra_token_pos["<all_audio>_end"].append(token_count - 1)
                extra_token_pos["<all_media>_end"].append(token_count - 1)

                add_mot_indices(token_type="audio", token_indices=list(range(token_count_start, token_count)))
                token_count_start = token_count

                if gen_template == "default":
                    token_seq.extend([self.eoa_token_id])
                    extra_token_pos["eoa"].append(token_count)
                    add_mot_indices(token_type="eoa", token_indices=[token_count])
                    token_count += 1
                    token_count_start = token_count

            else:
                raise ValueError(f"Not supported key: {key}")
            index_indicator[key] += 1

        if add_eos is True and not drop_last_break:
            # Typically used for t2i task.
            token_seq.append(self.eos_token_id)
            extra_token_pos["eos"].append(token_count)
            add_mot_indices(token_type="eos", token_indices=[token_count])
            token_count += 1
            token_count_start = token_count
        elif add_eos == 'auto' and not drop_last_break:
            # Typically used for lm and mmu task.
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra_token_pos["eos"].append(token_count)
                add_mot_indices(token_type="eos", token_indices=[token_count])
                token_count += 1
                token_count_start = token_count

        if total_length:
            # Check token count and clip sequence if necessary
            if token_count > total_length and drop_last:
                # Assert clip position is not in the middle of the block-wise tokens (gen_image,
                # src_image, und_image, face)
                for start_key, end_key in [
                    ("<img>_start", "<img>_end"), ("<vae_img>_start", "<vae_img>_end"),
                    ("<vit_img>_start", "<vit_img>_end"), ("<joint>_start", "<joint>_end"),
                    ("<video>_start", "<video>_end"), ("<audio>_start", "<audio>_end"),
                ]:
                    if start_key in extra_token_pos and end_key in extra_token_pos:
                        assert all(
                            (start > total_length or end + 1 < total_length)
                            for start, end in zip(extra_token_pos[start_key], extra_token_pos[end_key])
                        ), ("Clip position should not be in the middle of the media tokens.\n"
                            f"{total_length=}, {extra_token_pos[start_key]=}, {extra_token_pos[end_key]=}\n"
                            f"Below is the text:\n{self._shorten_text(self.decode(token_seq))}")
                token_seq = token_seq[:total_length]
                und_token_indices = [idx for idx in und_token_indices if idx < total_length] 
                gen_token_indices = [idx for idx in gen_token_indices if idx < total_length]
                audio_token_indices = [idx for idx in audio_token_indices if idx < total_length]

            # Pad the sequence if necessary
            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra_token_pos["first_pad"].append(token_count)
                add_mot_indices(token_type="pad", token_indices=list(range(token_count, token_count + pad_num)))

        if len(und_token_indices) > 0 and len(gen_token_indices) > 0:
            assert und_token_indices[-1] < len(token_seq) and gen_token_indices[-1] < len(token_seq), \
                f"{und_token_indices[-1]=}, {gen_token_indices[-1]=}, {len(token_seq)=}"
        if len(audio_token_indices) > 0:
            assert audio_token_indices[-1] < len(token_seq), f"{audio_token_indices[-1]=}, {len(token_seq)=}"

        return token_seq, extra_token_pos, und_token_indices, gen_token_indices, audio_token_indices

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
            sections: Optional[list[dict[str, Any]]] = None,
            max_token_length: Optional[int] = None,
            add_eos: bool | str = 'auto',
            use_text_mask: bool = True,
            add_pad: bool | str = 'auto',
            add_bos: bool = True,
            drop_last: bool | str = 'auto',
            und_token_type: Optional[list[str]] = None,
            gen_token_type: Optional[list[str]] = None,
            audio_token_type: Optional[list[str]] = None,
            disable_ignore: bool = False,
    ):
        if sections is None:
            raise ValueError("sections must be provided.")
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
                    uncond_length=section.get('uncond_length', 'equal'),
                )
                token_source['text'].append(text)
                text_mask_specs.append(dict(
                    ignore=section.get('ignore', False),
                    start_offset=section.get('start_offset', 0),
                    end_offset=section.get('end_offset', 0),
                ))
            elif section['type'] == 'gen_image':
                token_source['gen_image'].append(dict(
                    length=section['token_length'],
                    gen_image_template=section.get('gen_image_template', "default"),
                    add_timestep_token=section.get('add_timestep_token', False),
                    add_guidance_token=section.get('add_guidance_token', False),
                    add_timestep_r_token=section.get('add_timestep_r_token', False),
                    add_image_shape_token=section.get('add_image_shape_token', False),
                    add_tw_th_token=section.get('add_tw_th_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                    token_height=section.get('token_height'),
                    token_width=section.get('token_width'),
                ))
            elif section['type'] in ['cond_joint_image', 'cond_vae_image', 'cond_vit_image']:
                token_source[section['type']].append(dict(
                    length=section['token_length'],
                    gen_image_template=section.get('gen_image_template', "default"),
                    add_timestep_token=section.get('add_timestep_token', False),
                    add_image_shape_token=section.get('add_image_shape_token', False),
                    add_tw_th_token=section.get('add_tw_th_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                    token_height=section.get('token_height'),
                    token_width=section.get('token_width'),
                ))
            elif section['type'] == 'gen_video':
                token_source['gen_video'].append(dict(
                    length=section['token_length'],
                    gen_template=section['gen_template'],
                    add_timestep_token=section['add_timestep_token'],
                    add_guidance_token=section['add_guidance_token'],
                    add_timestep_r_token=section.get('add_timestep_r_token', False),
                    add_video_shape_token=section['add_video_shape_token'],
                    base_size=section['base_size'],
                    duration_idx=section['duration_idx'],
                    ratio_idx=section['ratio_idx'],
                    token_height=section['token_height'],
                    token_width=section['token_width'],
                    token_duration=section['token_duration'],
                ))
            elif section['type'] == 'cond_vae_video':
                source = dict(
                    length=section['token_length'],
                    token_height=section['token_height'],
                    token_width=section['token_width'],
                    token_duration=section['token_duration'],
                )
                token_source['cond_vae_video'].append(source)
            elif section['type'] == 'cond_vit_video':
                if section.get('timestamp') is None:
                    raise ValueError(
                        "cond_vit_video sections require timestamp metadata."
                    )
                source = dict(
                    length=section['token_length'],
                    token_height=section['token_height'],
                    token_width=section['token_width'],
                    token_duration=section['token_duration'],
                    timestamp=float(section['timestamp']),
                    temporal_index=int(section.get('temporal_index', 0)),
                    temporal_length=int(section.get('temporal_length', 1)),
                    video_id=int(section.get('video_id', 0)),
                )
                # The timestamp is ordinary text and must participate in text encoding/scatter.
                text_mask_specs.append(dict(ignore=False, start_offset=0, end_offset=0))
                token_source['cond_vit_video'].append(source)
            elif section['type'] == 'gen_audio':
                token_source['gen_audio'].append(dict(
                    length=section['token_length'],
                    gen_template=section['gen_template'],
                    add_timestep_token=section['add_timestep_token'],
                ))
            elif section['type'] == 'cond_vit_image':
                token_source['cond_vit_image'].append(dict(
                    length=section['token_length'],
                    add_timestep_token=section.get('add_timestep_token', False),
                    add_image_shape_token=section.get('add_image_shape_token', False),
                    add_tw_th_token=section.get('add_tw_th_token', False),
                    base_size=section.get('base_size'),
                    ratio_idx=section.get('ratio_idx'),
                    token_height=section.get('token_height'),
                    token_width=section.get('token_width'),
                ))
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        # Combine text and image tokens
        (
            full_token_seq, extra_token_pos, und_token_indices, gen_token_indices, audio_token_indices
        ) = self.encode_sequence(
            template=template,
            token_source=dict(token_source),
            total_length=max_token_length,
            add_eos=add_eos,
            add_pad=add_pad,
            add_bos=add_bos,
            drop_last=drop_last,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
            audio_token_type=audio_token_type,
        )
        full_seq_token_tensor = torch.tensor(full_token_seq, dtype=torch.long)
        und_token_indices = torch.tensor(und_token_indices, dtype=torch.long)
        gen_token_indices = torch.tensor(gen_token_indices, dtype=torch.long)
        audio_token_indices = torch.tensor(audio_token_indices, dtype=torch.long)

        # ================ Media special embedding indices ================
        # -- image --
        guidance_scatter_index = torch.tensor(extra_token_pos['guidance'], dtype=torch.long) \
            if 'guidance' in extra_token_pos else None
        cond_timestep_scatter_index = torch.tensor(extra_token_pos['cond_timestep'], dtype=torch.long) \
            if 'cond_timestep' in extra_token_pos else None
        gen_timestep_scatter_index = torch.tensor(extra_token_pos['gen_timestep'], dtype=torch.long) \
            if 'gen_timestep' in extra_token_pos else None
        gen_timestep_r_scatter_index = torch.tensor(extra_token_pos['gen_timestep_r'], dtype=torch.long) \
            if 'gen_timestep_r' in extra_token_pos else None

        # -- video --
        cond_video_timestep_scatter_index = torch.tensor(extra_token_pos['cond_video_timestep'], dtype=torch.long) \
            if 'cond_video_timestep' in extra_token_pos else None
        gen_video_timestep_scatter_index = torch.tensor(extra_token_pos['gen_video_timestep'], dtype=torch.long) \
            if 'gen_video_timestep' in extra_token_pos else None

        # -- audio --
        gen_audio_timestep_scatter_index = torch.tensor(extra_token_pos['gen_audio_timestep'], dtype=torch.long) \
            if 'gen_audio_timestep' in extra_token_pos else None

        # ================ Media slices ================
        # -- image --
        gen_image_slices, gen_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'img', full_seq_token_tensor)
        vae_image_slices, vae_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vae_img', full_seq_token_tensor)
        vit_image_slices, vit_image_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vit_img', full_seq_token_tensor)
        joint_image_slices, _ = self.parse_extra_token_pos(
            extra_token_pos, 'joint_img', full_seq_token_tensor)
        # All image slices (src_image, gen_image, und_image)
        all_image_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_img>_start'], extra_token_pos['<all_img>_end'])
        ] if '<all_img>_start' in extra_token_pos and '<all_img>_end' in extra_token_pos else []

        # -- video --
        gen_video_slices, gen_video_mask = self.parse_extra_token_pos(
            extra_token_pos, 'video', full_seq_token_tensor)
        # r2v source-video conditions
        vae_video_slices, vae_video_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vae_video', full_seq_token_tensor)
        vit_video_slices, vit_video_mask = self.parse_extra_token_pos(
            extra_token_pos, 'vit_video', full_seq_token_tensor)
        vit_video_context_slices, _ = self.parse_extra_token_pos(
            extra_token_pos, 'vit_video_context', full_seq_token_tensor)
        # All video slices
        all_video_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_video>_start'], extra_token_pos['<all_video>_end'])
        ] if '<all_video>_start' in extra_token_pos and '<all_video>_end' in extra_token_pos else []

        # -- audio --
        gen_audio_slices, gen_audio_mask = self.parse_extra_token_pos(
            extra_token_pos, 'audio', full_seq_token_tensor)
        # All audio slices
        all_audio_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_audio>_start'], extra_token_pos['<all_audio>_end'])
        ] if '<all_audio>_start' in extra_token_pos and '<all_audio>_end' in extra_token_pos else []

        # All media slices
        all_media_slices = [
            slice(start, end + 1)
            for start, end in zip(extra_token_pos['<all_media>_start'], extra_token_pos['<all_media>_end'])
        ]

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
                if not mask_spec['ignore'] or disable_ignore:
                    real_slice = slice(
                        text_slice.start + mask_spec['start_offset'],
                        text_slice.stop + mask_spec['end_offset']
                    )
                    text_mask[real_slice] = 1.0
        else:
            text_mask = None

        # real_pos is the first position of the <pad> token
        real_pos = torch.tensor(extra_token_pos.get('first_pad', [full_seq_token_tensor.shape[0]]), dtype=torch.long)

        if len(default(und_token_type, [])) == 0 and len(default(gen_token_type, [])) == 0 and \
                len(default(audio_token_type, [])) == 0:
            und_token_indices = None
            gen_token_indices = None
            audio_token_indices = None

        return TokenizerEncodeOutput(
            tokens=full_seq_token_tensor,

            # ==== slices ====
            # -- text --
            text_slices=text_slices,
            # -- image --
            gen_image_slices=gen_image_slices,
            vae_image_slices=vae_image_slices,
            vit_image_slices=vit_image_slices,
            joint_image_slices=joint_image_slices,
            all_image_slices=all_image_slices,
            # -- video --
            gen_video_slices=gen_video_slices,
            vae_video_slices=vae_video_slices,
            vit_video_slices=vit_video_slices,
            vit_video_context_slices=vit_video_context_slices,
            all_video_slices=all_video_slices,
            # -- audio --
            gen_audio_slices=gen_audio_slices,
            all_audio_slices=all_audio_slices,
            # -- All media --
            all_media_slices=all_media_slices,

            # ==== mask ====
            # -- text --
            text_mask=text_mask,
            # -- image --
            gen_image_mask=gen_image_mask,
            vae_image_mask=vae_image_mask,
            vit_image_mask=vit_image_mask,
            # -- video --
            gen_video_mask=gen_video_mask,
            vae_video_mask=vae_video_mask,
            vit_video_mask=vit_video_mask,
            # -- audio --
            gen_audio_mask=gen_audio_mask,

            # ==== position related ====
            real_pos=real_pos,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
            audio_token_indices=audio_token_indices,
            # -- image --
            guidance_scatter_index=guidance_scatter_index,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            gen_timestep_r_scatter_index=gen_timestep_r_scatter_index,
            # -- video --
            cond_video_timestep_scatter_index=cond_video_timestep_scatter_index,
            gen_video_timestep_scatter_index=gen_video_timestep_scatter_index,
            # -- audio --
            gen_audio_timestep_scatter_index=gen_audio_timestep_scatter_index,
        )

    def get_cot_sections(self, cot_text, uncond_kwargs, cot_max_length=None, drop_think=False):
        if not cot_text:  # None or empty
            return []
        if self.think_token in cot_text and self.end_of_think_token in cot_text:
            before_think_sec = cot_text.split(self.think_token)[0]
            after_think_sec = cot_text.split(self.end_of_think_token)[1]
            think_sec = cot_text.split(self.think_token)[1].split(self.end_of_think_token)[0]
            return self.get_cot_sections(before_think_sec, uncond_kwargs, drop_think=drop_think) + \
                ([
                    dict(type="text", text=self.think_token),
                    dict(type="text", text=think_sec, max_length=cot_max_length, **uncond_kwargs),
                    dict(type="text", text=self.end_of_think_token)
                ] if not drop_think else []) + \
                self.get_cot_sections(after_think_sec, uncond_kwargs, drop_think=drop_think)

        if self.recaption_token in cot_text and self.end_of_recaption_token in cot_text:
            before_recaption_sec = cot_text.split(self.recaption_token)[0]
            after_recaption_sec = cot_text.split(self.end_of_recaption_token)[1]
            recaption_sec = cot_text.split(self.recaption_token)[1].split(self.end_of_recaption_token)[0]
            return self.get_cot_sections(before_recaption_sec, uncond_kwargs, drop_think=drop_think) + \
                [
                    dict(type="text", text=self.recaption_token),
                    dict(type="text", text=recaption_sec, max_length=cot_max_length, **uncond_kwargs),
                    dict(type="text", text=self.end_of_recaption_token)
                ] + \
                self.get_cot_sections(after_recaption_sec, uncond_kwargs, drop_think=drop_think)

        return [
            dict(type="text", text=cot_text, **uncond_kwargs),
        ]

    def apply_general_template(
            self,
            message_list,
            conv_template,
            max_length=None,
            add_assistant_prefix=False,
            answer="auto",
            bot_task="auto",
            sequence_template="instruct",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=False,
            image_base_size=None,
            drop_think=False,
            und_token_type=None,
            gen_token_type=None,
            audio_token_type=None,
            uncond_length="equal",
            use_text_mask=False,
    ):
        if bot_task == "img_ratio":
            assert image_base_size is not None, "image_base_size should be provided for img_ratio task."

        # If cfg_factor > 1, we need to repeat the unconditioned part
        if batchify:
            assert isinstance(message_list[0], list), \
                f"When batchify is True, message_list should be a list of list, but got [{type(message_list[0])}, ...]."
            return self.batch_gen_infer(
                infer_fn=self.apply_general_template,
                prompt_list=[[] for _ in range(len(message_list))],
                infer_fn_kwargs_list=[dict(
                    message_list=message_list_i,
                    conv_template=conv_template,
                    max_length=max_length,
                    add_assistant_prefix=add_assistant_prefix,
                    answer=answer,
                    bot_task=bot_task,
                    sequence_template=sequence_template,
                    image_base_size=image_base_size,
                    drop_think=drop_think,
                    und_token_type=und_token_type,
                    gen_token_type=gen_token_type,
                    audio_token_type=audio_token_type,
                    uncond_length=uncond_length,
                    use_text_mask=use_text_mask,
                ) for message_list_i in message_list],
                do_classifier_free_guidance=cfg_factor > 1,
                condition_repeat_times=1,
                uncondition_repeat_times=cfg_factor - 1,
                und_token_type=und_token_type,
                gen_token_type=gen_token_type,
                audio_token_type=audio_token_type,
            )

        uncond_kwargs = dict(
            uncond_enabled=uncond_p == 1.0,
            uncond_p=uncond_p,
            uncond_length=uncond_length,
        )

        def process_successive_message(_message_list, _cur_message_idx, role, prefix, suffix,
                                       answer_prefix="", answer_suffix=""):
            _sub_sections = []
            while _cur_message_idx < len(message_list) and _message_list[_cur_message_idx]['role'] == role:
                message = _message_list[_cur_message_idx]
                if message['type'] == 'text':
                    text = message['content']
                    if role == "system":
                        _sub_sections.append(dict(type="text", text=text))
                    elif role == "assistant":  # recaption or think is before answer
                        if (self.recaption_token in text and self.end_of_recaption_token in text) or (
                                self.think_token in text and self.end_of_think_token in text):
                            _sub_sections.extend(self.get_cot_sections(text, uncond_kwargs, drop_think=drop_think))
                        else:
                            _sub_sections.append(dict(
                            type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs))
                    else: # user
                        _sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                elif message['type'] == 'gen_image':
                    info = message['content']
                    assert isinstance(info, ImageInfo), f"Expected ImageInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message['type'] in ['cond_joint_image', 'cond_vae_image', 'cond_vit_image']:   # only in user message
                    info = message['content']
                    assert isinstance(info, (ImageInfo, JointImageInfo)), \
                        f"Expected ImageInfo or JointImageInfo, but got {type(info)}"
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                elif message['type'] == 'cond_vae_video':   # r2v source-video latent (user message)
                    info = message['content']
                    assert isinstance(info, VideoInfo), \
                        f"Expected VideoInfo for {message['type']}, but got {type(info)}"
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                elif message['type'] == 'cond_vit_video':   # native temporal video (user message)
                    info = message['content']
                    assert isinstance(info, VideoInfo), \
                        f"Expected VideoInfo for {message['type']}, but got {type(info)}"
                    if info.timestamps is None:
                        raise ValueError(
                            "cond_vit_video requires one timestamp per temporal tubelet."
                        )
                    if len(info.timestamps) != int(info.token_duration):
                        raise ValueError(
                            "Condition-video timestamp count must equal token_duration: "
                            f"{len(info.timestamps)} != {info.token_duration}."
                        )
                    frame_token_length = int(info.token_height) * int(info.token_width)
                    if int(info.video_token_length) != int(info.token_duration) * frame_token_length:
                        raise ValueError(
                            "Condition-video token length is inconsistent with its temporal grid: "
                            f"{info.video_token_length} != "
                            f"{info.token_duration} * {frame_token_length}."
                        )
                    _sub_sections.extend({
                        "type": message['type'],
                        "token_length": frame_token_length,
                        "token_height": int(info.token_height),
                        "token_width": int(info.token_width),
                        "token_duration": 1,
                        "timestamp": float(timestamp),
                        "temporal_index": temporal_index,
                        "temporal_length": int(info.token_duration),
                        "video_id": int(info.video_id or 0),
                    } for temporal_index, timestamp in enumerate(info.timestamps))
                elif message['type'] == 'gen_video':
                    info = message['content']
                    assert isinstance(info, VideoInfo), f"Expected VideoInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info, with_audio=message["with_audio"]))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message['type'] == 'gen_audio':
                    info = message['content']
                    assert isinstance(info, AudioInfo), f"Expected AudioInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info, with_video=message["with_video"]))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                else:
                    raise ValueError(f"Unknown message type: {message['type']}")
                _cur_message_idx += 1
            if len(_sub_sections) > 0:
                # Add role prefix and suffix
                _sub_sections.insert(0, dict(type='text', text=prefix))
                _sub_sections.append(dict(type='text', text=suffix))
            return _sub_sections, _cur_message_idx

        # Define assistant prefix and suffix
        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = self.answer_token, self.end_of_answer_token
        else:
            answer_prefix, answer_suffix = "", ""
        if sequence_template == "pretrain":
            system_suffix = conv_template.pretrain_sep_sp
            user_prefix = conv_template.get_role_prefix(conv_template.pretrain_roles[0])
            user_suffix = conv_template.pretrain_sep
            bot_prefix = conv_template.get_role_prefix(conv_template.pretrain_roles[1])
            bot_suffix = conv_template.pretrain_sep2
        else:
            system_suffix = conv_template.sep_sp
            user_prefix = conv_template.get_role_prefix(conv_template.roles[0])
            user_suffix = f"{conv_template.sep}"
            if "hy_" in user_suffix and "quickly_think" in bot_task:
                user_suffix = f"{self.quickly_think_token}{conv_template.sep}"
            elif "hy_" in user_suffix and "slowly_think" in bot_task:
                user_suffix = f"{self.slowly_think_token}{conv_template.sep}"
            bot_prefix = conv_template.get_role_prefix(conv_template.roles[1])
            bot_suffix = f"{conv_template.sep2}"

        # Process successive user and assistant messages
        sections = []
        cur_message_idx = 0
        final_role = None
        while cur_message_idx < len(message_list):
            # Process successive system messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="system", prefix="", suffix=system_suffix)
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "system"

            # Process successive user messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="user", prefix=user_prefix, suffix=user_suffix)
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "user"

            # Process successive assistant messages
            sub_sections, cur_message_idx = process_successive_message(
                message_list, cur_message_idx, role="assistant", prefix=bot_prefix, suffix=bot_suffix,
                answer_prefix=answer_prefix, answer_suffix=answer_suffix,
            )
            # Add to the template and sections
            sections.extend(sub_sections)
            if len(sub_sections) > 0:
                final_role = "assistant"

        if add_assistant_prefix:
            if final_role == "assistant":
                # Avoid adding prefix twice
                _bot_prefix = ""
                # Remove the final bot_suffix
                if len(sections) > 0 and sections[-1]['type'] == 'text' and sections[-1]['text'] == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            # We can add special tokens for the bot lastest message according to different tasks
            bot_response_prefix = dict(
                auto=lambda: f"{_bot_prefix}{answer_prefix}",
                quickly_think = lambda: f"{_bot_prefix}",
                slowly_think = lambda: f"{_bot_prefix}",
                image=lambda: "",
                think=lambda: f"{_bot_prefix}{self.think_token}",
                recaption=lambda: f"{_bot_prefix}{self.recaption_token}",
                img_ratio=lambda: f"{_bot_prefix}{answer_prefix}{self.boi_token}{self.size_token(image_base_size)}",
                img_tw_th=lambda: f"{_bot_prefix}{answer_prefix}{self.boi_token}",
            )[bot_task]()
            sections.append(dict(type='text', text=bot_response_prefix))

        if und_token_type is None:
            und_token_type = []
        if gen_token_type is None:
            gen_token_type = []
        if audio_token_type is None:
            audio_token_type = []

        output = self.encode_general(
            sections=sections,
            use_text_mask=use_text_mask,
            add_eos=conv_template.add_eos,
            add_pad=conv_template.add_pad,
            add_bos=conv_template.add_bos,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
            audio_token_type=audio_token_type,
        )

        if max_length is not None:
            if output.tokens.shape[-1] > max_length:
                raise ValueError(
                    f"Encoded token length {output.tokens.shape[-1]} exceeds max_length {max_length}.\n"
                    f"Please set a larger max_length or check the input messages:\n{message_list}"
                )

        return output, sections

    def apply_chat_template(
            self,
            batch_prompt: Optional[list[str]] = None,
            batch_message_list: Optional[list[list[dict[str, Any]]]] = None,
            mode: str = "gen_text",
            batch_gen_image_info: Optional[list[ImageInfo]] = None,
            batch_gen_video_info: Optional[list[VideoInfo]] = None,
            batch_cond_images: Optional[Union[list[CondImage], list[list[CondImage]]]] = None,
            batch_system_prompt: Optional[list[str]] = None,
            batch_cot_text: Optional[list[str]] = None,
            max_length: Optional[int] = None,
            bot_task: str = "auto",
            image_base_size: Optional[int] = None,
            cond_image_section_type: str = None,
            sequence_template: str = "pretrain",
            cfg_factor: int = 1,
            add_assistant_prefix: Optional[bool] = None,
            drop_think: bool = False,
            conv_template: Optional[Conversation] = None,
            und_token_type: list[str] = None,
            gen_token_type: list[str] = None,
            uncond_length: str = "equal",
            use_text_mask: bool = False,
            **kwargs,
    ) -> dict[str, Any]:
        allowed_tasks = ["image", "auto", "think", "recaption", "img_ratio", "img_tw_th", "quickly_think", "slowly_think", "video"]
        assert bot_task in allowed_tasks, f"bot_task should be one of {allowed_tasks}, but got {bot_task}."

        if batch_message_list is None:
            # Simple text-to-image or text-cot-to-image task
            batch_size = len(batch_prompt)

            # Batchify inputs
            if not isinstance(batch_system_prompt, list):
                batch_system_prompt = [batch_system_prompt] * batch_size
            if not isinstance(batch_gen_image_info, list):
                batch_gen_image_info = [batch_gen_image_info] * batch_size
            if batch_cot_text is not None:
                assert len(batch_cot_text) == batch_size, \
                    (f"batch_cot_text should have the same length as batch_size ({batch_size}), "
                     f"but got {len(batch_cot_text)}.")
            else:
                batch_cot_text = [None] * batch_size
            if batch_cond_images is not None:
                assert len(batch_cond_images) == batch_size, \
                    (f"batch_cond_image_info should have the same length as batch_size ({batch_size}), "
                     f"but got {len(batch_cond_images)}.")
                batch_cond_images = [
                    cond_images if isinstance(cond_images, list) else [cond_images]
                    for cond_images in batch_cond_images
                ]
            else:
                batch_cond_images = [[] for _ in range(batch_size)]

            # Convert single round materials into standard message list
            batch_message_list = []
            for prompt, system_prompt, cot_text, gen_image_info, cond_images in zip(
                    batch_prompt, batch_system_prompt, batch_cot_text, batch_gen_image_info,
                    batch_cond_images,
            ):
                message_list = []
                # 1. system prompt section
                if system_prompt:
                    message_list.append(dict(role="system", type="text", content=system_prompt))
                # 2. user inputs sections
                #   2.1 image inputs
                if len(cond_images) > 0:
                    message_list.extend([
                        dict(role="user", type=cond_image_section_type, content=cond_image.i)
                        for cond_image in cond_images
                    ])
                #   2.2 text inputs
                message_list.append(dict(role="user", type="text", content=prompt))
                # 3. assistant answer sections
                if cot_text is not None:
                    message_list.append(dict(role="assistant", type="text", content=cot_text))
                if mode == "gen_image":
                    message_list.append(dict(
                        role="assistant", type="gen_image", content=gen_image_info))
                # ---
                batch_message_list.append(message_list)

        output, sections = self.apply_general_template(
            message_list=batch_message_list,
            conv_template=conv_template,
            max_length=max_length,
            add_assistant_prefix=default(add_assistant_prefix, mode == "gen_text"),
            bot_task=bot_task,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
            image_base_size=image_base_size,
            drop_think=drop_think,
            und_token_type=und_token_type,
            gen_token_type=gen_token_type,
            uncond_length=uncond_length,
            use_text_mask=use_text_mask,
            **kwargs,
        )
        return dict(output=output, sections=sections)

    def pad(self, tensor_list, dim=0, pad_val=None, key=None):
        if pad_val is None:
            pad_val = self.pad_token_id
        max_len = max([t.shape[dim] for t in tensor_list])
        padded_tensor_list = []
        for t in tensor_list:
            if t.shape[dim] < max_len:
                assert pad_val is not False, f"Not allowed/implemented pad for key: {key}"
                t = F.pad(t, (0, max_len - t.shape[dim]), value=pad_val)
            padded_tensor_list.append(t)
        return padded_tensor_list

    def batch_gen_infer(
            self,
            infer_fn,
            prompt_list: list,
            negative_prompt_list: list = None,
            infer_fn_kwargs_list: list[dict[str, int]] = None,
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
        condition_repeat_times: int
        uncondition_repeat_times: int
            Support multi-condition and multi-uncondition. e.g, [pred_cond, pred_uncond_text, pred_uncond_text_uncond_src]
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
                # key: tokens(tensor), mask(tensor), slices(list of slice), scatter_index(tensor), indices(tensor), real_pos(tensor)
                # key needs to be padded: tokens(tensor), mask(tensor), indices(tensor)
                # key no need to be padded: slices(list of slice), scatter_index(tensor), real_pos(tensor)
                for key in list(first.keys()):
                    merged_list = [cond_item[key] for cond_item in batch_cond_item] * condition_repeat_times + \
                        [uncond_item[key] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    if isinstance(first[key], torch.Tensor):
                        if 'mask' in key:
                            pad_val = 0.0
                        elif key == 'tokens':
                            pad_val = self.pad_token_id
                        elif key in ['und_token_indices', 'gen_token_indices', 'audio_token_indices']:
                            continue
                        else:
                            # Should not pad for other tensors
                            # key that comes here: real_pos(tensor), scatter_index(tensor)
                            pad_val = False
                        if key not in ('und_token_indices', 'gen_token_indices', 'audio_token_indices'):
                            stacked_item[key] = torch.stack(self.pad(merged_list, pad_val=pad_val, key=key), dim=0)
                    elif isinstance(first[key], list):
                        stacked_item[key] = merged_list
                    elif first[key] is None:
                        pass
                    else:
                        raise ValueError(f"Unsupported type of {key}: {type(first[key])}.")

                stacked_item = TokenizerEncodeOutput(stacked_item)

                if 'und_token_indices' in first.keys() and first['und_token_indices'] is not None and 'gen_token_indices' in first.keys() and first['gen_token_indices'] is not None:
                    und_token_indices_merged_list = [cond_item['und_token_indices'] for cond_item in batch_cond_item] * condition_repeat_times + [uncond_item['und_token_indices'] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    gen_token_indices_merged_list = [cond_item['gen_token_indices'] for cond_item in batch_cond_item] * condition_repeat_times + [uncond_item['gen_token_indices'] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    audio_token_indices_merged_list = [cond_item['audio_token_indices'] for cond_item in batch_cond_item] * condition_repeat_times + [uncond_item['audio_token_indices'] for uncond_item in batch_uncond_item] * uncondition_repeat_times
                    sequence_length = stacked_item["tokens"].shape[1]

                    # original token sequence is padded with pad token from max_index+1 to sequence_length-1
                    # therefore, we pad und_token_indices with arange(max_index+1, sequence_length)
                    def get_max_token_index(*token_index_tensors):
                        non_empty_max = [
                            tensor.max().item()
                            for tensor in token_index_tensors
                            if tensor is not None and tensor.numel() > 0 # in gen_text mode, gen_token_indices is empty
                        ]
                        return max(non_empty_max) if len(non_empty_max) > 0 else -1
                    max_index = [
                        get_max_token_index(
                            und_token_indices_merged_list[i],
                            gen_token_indices_merged_list[i],
                            audio_token_indices_merged_list[i],
                        )
                        for i in range(len(und_token_indices_merged_list))
                    ]
                    for i, (und_token_indices_item, max_index_item) in enumerate(zip(und_token_indices_merged_list, max_index)):
                        if max_index_item == sequence_length - 1:
                            continue
                        und_token_indices_merged_list[i] = torch.cat([und_token_indices_item, torch.arange(max_index_item + 1, sequence_length)])

                    # 多分辨率支持
                    # -- 1. 不同 sample 的 gen token 数量可能不同（image size 不同），需要对齐到 max_gen_count 才能 torch.stack。做法是将 gen 较少的 
                    #       sample 末尾 und 位置（pad token）移入 gen，这些 pad token走 gen pathway 不影响结果，因为它们的输出不被有效 token attend 到。
                    # -- 2. 当 batch 内文本长度差异较大时，Step 1 产生的 pad 位置可能不够移入 gen（gen 缺口 > pad 数量，剩余会侵入真实 und token）。
                    #       此时额外扩展序列长度（tokens 补 pad_token_id，masks 补 0），为 und 增加足够的 pad 位置，保证移入 gen 的始终是 pad 而非真实文本 token。
                    max_gen_count = max(g.shape[0] for g in gen_token_indices_merged_list)

                    max_extra_needed = 0
                    for i in range(len(gen_token_indices_merged_list)):
                        pad_needed = max_gen_count - gen_token_indices_merged_list[i].shape[0]
                        pad_available = max(0, sequence_length - 1 - max_index[i])
                        max_extra_needed = max(max_extra_needed, pad_needed - pad_available)

                    if max_extra_needed > 0:
                        for key in list(stacked_item.keys()):
                            if key == 'tokens':
                                stacked_item[key] = F.pad(stacked_item[key], (0, max_extra_needed), value=self.pad_token_id)
                            elif 'mask' in key and isinstance(stacked_item[key], torch.Tensor):
                                stacked_item[key] = F.pad(stacked_item[key], (0, max_extra_needed), value=0.0)
                        new_positions = torch.arange(sequence_length, sequence_length + max_extra_needed)
                        for i in range(len(und_token_indices_merged_list)):
                            und_token_indices_merged_list[i] = torch.cat([und_token_indices_merged_list[i], new_positions])
                        sequence_length += max_extra_needed

                    for i in range(len(gen_token_indices_merged_list)):
                        pad_needed = max_gen_count - gen_token_indices_merged_list[i].shape[0]
                        if pad_needed > 0:
                            moved = und_token_indices_merged_list[i][-pad_needed:]
                            und_token_indices_merged_list[i] = und_token_indices_merged_list[i][:-pad_needed]
                            gen_token_indices_merged_list[i] = torch.cat([gen_token_indices_merged_list[i], moved])

                    stacked_item['und_token_indices'] = torch.stack(und_token_indices_merged_list, dim=0)
                    stacked_item['gen_token_indices'] = torch.stack(gen_token_indices_merged_list, dim=0)
                    stacked_item['audio_token_indices'] = torch.stack(audio_token_indices_merged_list, dim=0)

                elif ('und_token_indices' in first.keys() and first['und_token_indices'] is not None) or ('gen_token_indices' in first.keys() and first['gen_token_indices'] is not None):
                    raise ValueError(f"Only one of 'und_token_indices' and 'gen_token_indices' exists.")

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


class DecoratorSections(object):
    """ Define predefined sections in a multimodal template. """

    def __init__(
            self,
            tokenizer: HunyuanMultimodalTokenizerFast,
            conv: Conversation,
            sequence_template: str,
            ignore_start_tokens: set,
    ):
        self.tokenizer = tokenizer
        self.conv = conv
        self.sequence_template = sequence_template
        self.ignore_start_tokens = ignore_start_tokens
        self.roles = self.conv.roles

        # Define sections based on the sequence template
        if self.sequence_template == "pretrain":
            self.user = []
            self.bot = []
            self.user_sep = [dict(type="text", text=self.conv.pretrain_sep, ignore=True)] if self.conv.pretrain_sep else []  # not compatible with A13B
            self.bot_sep = [dict(type="text", text=self.conv.pretrain_sep2)] if self.conv.pretrain_sep2 else []
            self.user_sep_length = len(self.tokenizer.encode(self.conv.pretrain_sep))
            self.bot_sep_length = len(self.tokenizer.encode(self.conv.pretrain_sep2))
            self.answer_ = []
            self._answer = []

        elif self.sequence_template == "instruct":
            self.user = [dict(type="text", text=self.conv.get_role_prefix(self.roles[0]), ignore=True)]
            self.bot = [dict(type="text", text=self.conv.get_role_prefix(self.roles[1]), ignore=True)]
            self.user_sep = [dict(type="text", text=self.conv.sep, ignore=True)] if self.conv.sep else []  # not compatible with A13B
            self.bot_sep = [dict(type="text", text=self.conv.sep2)] if self.conv.sep2 else []
            self.user_sep_length = len(self.tokenizer.encode(self.conv.sep))
            self.bot_sep_length = len(self.tokenizer.encode(self.conv.sep2))
            self.answer_ = [dict(type="text", text=self.tokenizer.answer_token,
                                 ignore=(self.tokenizer.answer_token in self.ignore_start_tokens))] \
                if conv.use_answer else []
            self._answer = [dict(type="text", text=self.tokenizer.end_of_answer_token)] \
                if conv.use_answer else []
            self.tool_responses_ = [dict(type="text", text="<tool_responses>", ignore=True)]
            self._tool_responses = [dict(type="text", text="</tool_responses>", ignore=True)]

        else:
            raise NotImplementedError(f"Unsupported sequence_template: {self.sequence_template}")

        self.user_length = sum([len(self.tokenizer.encode(section["text"])) for section in self.user])
        self.bot_length = sum([len(self.tokenizer.encode(section["text"])) for section in self.bot])
        self.answer_length = sum([len(self.tokenizer.encode(section["text"])) for section in self.answer_ + self._answer])

        # Define eos token
        eos_token = self.tokenizer.eos_token
        if isinstance(eos_token, int):
            eos_token = self.tokenizer.convert_ids_to_tokens(eos_token)
        assert isinstance(eos_token, str), f"eos_token should be a string, got {type(eos_token)}."
        self.eos = [dict(type="text", text=eos_token)]

        # Define think sections
        self.think_ = lambda do_uncond: [
            dict(type="text", text=self.tokenizer.think_token,
                 ignore=(self.tokenizer.think_token in self.ignore_start_tokens) or do_uncond)
        ]
        self._think = lambda do_uncond: [
            dict(type="text", text=self.tokenizer.end_of_think_token, ignore=do_uncond)
        ]

        # Define recaption sections
        if hasattr(self.tokenizer, "recaption_token"):
            self.recaption_ = lambda do_uncond: [
                dict(type="text", text=self.tokenizer.recaption_token,
                     ignore=(self.tokenizer.recaption_token in self.ignore_start_tokens) or do_uncond)
            ]
            self._recaption = lambda do_uncond: [
                dict(type="text", text=self.tokenizer.end_of_recaption_token, ignore=do_uncond)
            ]

    def bot_sep_sections(self, ignore: bool = False) -> list[dict]:
        """Return ``bot_sep`` sections, optionally masked out from the loss.

        When ``ignore`` is True, every section is shallow-copied with
        ``ignore=True`` so that the bot_sep token does not contribute to the
        training loss.
        """
        if not ignore:
            return self.bot_sep
        return [{**section, "ignore": True} for section in self.bot_sep]

    def answer(self, section):
        if isinstance(section, dict):
            section = [section]
        return self.answer_ + section + self._answer

    def think(self, section, do_uncond=False):
        if isinstance(section, dict):
            section = [section]
        return self.think_(do_uncond) + section + self._think(do_uncond)

    def recaption(self, section, do_uncond=False):
        if not hasattr(self, "recaption_"):
            raise AttributeError("This tokenizer does not support recaption sections.")
        if isinstance(section, dict):
            section = [section]
        return self.recaption_(do_uncond) + section + self._recaption(do_uncond)

    def tool_responses(self, section):
        if isinstance(section, dict):
            section = [section]
        return self.tool_responses_ + section + self._tool_responses
