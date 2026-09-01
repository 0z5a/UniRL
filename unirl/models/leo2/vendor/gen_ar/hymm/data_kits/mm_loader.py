# A Generalized dataset for multi-modal tasks.
#
import importlib
import itertools
import math
import random
from itertools import chain, cycle
from typing import Any, TYPE_CHECKING, Union

import torch
from index_kits.sampler import DistributedSampler, IndexBatchSampler
from loguru import logger as all_logger
from torch.utils.data import DataLoader

from .instruction_template import text2image_instructions, image_captioning_instructions_dict
from .system_prompt import t2i_system_prompts, unified_system_prompts, vanilla_system_prompts, get_system_prompt
from ..utils.image_base import ImageInfo
from .utils import (
    ImageMixin,
    TextMixin,
    ImageCaptionMixin,
    MultimodalTasksState,
    IndexDataset,
    resample_for_errors,
)
from .utils.data_container import MultimodalDataContainer, maybe_stack
from ..models.tokenizers import load_tokenizer
from ..models.tokenizers.conversation import get_conversation_template
from ..models.tokenizers.tokenization_hunyuan_multimodal import DecoratorSections
from ..utils.helpers import default, as_tuple
from ..core.parallel_states import ParallelState
from hymm.constants import VISION_ENCODER_META_INFO
from hymm.core.global_vars import get_parallel_state


if TYPE_CHECKING:
    from ..models.tokenizers import TokenizerWrapper, HunyuanMultimodalTokenizerFast


REQUIRED_TASK_KWARGS = [
    "modality", "batch_size", "max_token_length", "attn_type", "shuffle_seed"
]
REQUIRED_INDEX_KWARGS = [
    "index_file"
]

class MultimodalIndexDataset(
    IndexDataset, ImageMixin, TextMixin, ImageCaptionMixin,
):
    def __init__(
            self,
            args,
            dataset_tag,
            task_kwargs=None,
            index_kwargs=None,
            post_kwargs=None,
            logger=None,
            use_tokenizer=True,
    ):
        # Initialize base class according to MRO
        super().__init__(logger=logger)

        self.args = args
        self.dataset_tag = dataset_tag
        self.use_tokenizer = use_tokenizer
        # Define logger prefix
        if torch.distributed.is_initialized():
            self.log_prefix = f"[rank{torch.distributed.get_rank()}] [Dataset-{self.dataset_tag}] "
        else:
            self.log_prefix = f"[Dataset-{self.dataset_tag}] "

        self.parse_task_and_index_kwargs(task_kwargs, index_kwargs, log_prefix=self.log_prefix)
        # Data modalities
        self.modality = as_tuple(self.task_kwargs.get("modality", args.modality))
        # Micro batch size
        self.batch_size = self.task_kwargs.get("batch_size", args.micro_batch_size)
        # Max sequence length
        self.max_token_length = self.task_kwargs["max_token_length"]

        # Shuffle dataset
        # TODO(kevinkhwu): When merging dev and dev_video, I found the arg conflicts: `first_epoch_no_shuffle` and `no_shuffle`
        self.disable_shuffle = self.task_kwargs.get("no_shuffle", args.no_shuffle)

        # Enable resample for error
        self.enable_resample = args.enable_resample
        self.drop_resampled_samples = args.drop_resampled_samples
        # Leave some room for dummy tokens when building the sequence
        self.dummy_number_dict = dict(
            ts_proj=1 + (1 if args.add_timestep_token else 0) if args.use_vae else 0,
            vit=VISION_ENCODER_META_INFO[args.vit_type]["dummy_number"] if args.use_vit else 0,
        )
        # Attention mask sequence length -1
        self.attn_mask_seq_m1 = True
        # Whether to drop last sections
        # - auto: drop last tokens exceeding the max_total_length. If the first dropped token is in the middle
        #         of the image tokens, an error will be raised.
        # - True: drop last tokens exceeding the max_total_length. If the first dropped token is in the middle
        #         of the image tokens, all the successive image tokens will be dropped.
        # - False: keep the last tokens exceeding the max_total_length, even if the total_length is reached.
        self.drop_last = self.task_kwargs.get("drop_last", "auto")

        self.interleave_only_last_round = self.task_kwargs.get("interleave_only_last_round", False)

        # If True, the bot_sep token (separator at the end of bot turns) will be excluded from
        # the training loss by setting `ignore=True` on its sections.
        self.ignore_bot_sep_token = self.args.ignore_bot_sep_token

        self.und_token_type = self.args.und_token_type
        self.gen_token_type = self.args.gen_token_type

        # Sequence pack related and affected
        self.sequence_pack = self.task_kwargs.get('sequence_pack', False)
        if self.sequence_pack:
            # if sequence pack is enabled, attention mask sequence length -1 is disabled in __getitem__,
            # and performed in seq_collate_fn instead.
            self.attn_mask_seq_m1 = False
            # if sequence pack is enabled, we will use the sequence-wise dummy_number to pad the packed sequence,
            # instead of dummy_number to pad the sample. `block_size` will be used as the maximum sequence length
            # for all the datasets.
            self.seq_length = self.task_kwargs.get('max_token_length', args.seq_length) \
                if args.independent_seq_length else args.seq_length

            use_mot = getattr(args, "use_mot", False)
            use_audio_branch = getattr(args, "audio_branch_model_name", None) is not None
            if (use_mot or use_audio_branch) and get_parallel_state().backend == 'megatron':
                # 预留 CP align 所需 pad（仅 MoT 需要 und/gen 对齐），使 bin 容量 = max_sequence_length - reserve
                # =============================== 公式推导 ===============================
                # 为保证 CP align 一定能成功，最保守需要预留的 pad_length 下界（不切 und 时的最坏情况）。
                # - 条件：pad >= und_min + gen_min，且 (pad - und_min - gen_min) % cp_size == 0；
                # 当 und_min==0 时还需 pad >= cp_size + gen_min。
                # - und_min, gen_min 均在 [0, cp_size-1]，最坏为 und_min=0, gen_min=cp_size-1，
                # 此时最少需要 pad = cp_size + gen_min = 2*cp_size - 1。
                # 若允许切 und 再分配，实际所需可能更小；此值用于规划/预留。
                p_state = get_parallel_state()
                cp_size = p_state.cp_size
                n_branches = 2 + int(use_audio_branch)
                # TODO(jarvizhang): Is `+1` necessary?
                self.reserved_pad_length = n_branches * (cp_size - 1) + 1 if cp_size > 1 else 0
            else:
                self.reserved_pad_length = 0

            self.max_sequence_length = self.seq_length + 1 - sum(self.dummy_number_dict.values()) - self.reserved_pad_length
            self.logger.info(f"{self.log_prefix}Sequence pack is enabled with "
                             f"{self.seq_length=} and {self.max_sequence_length=}.")

        # ===========================================================================
        # Common setup
        self.setup_data(
            enable_crypto=args.cos_file_is_encrypted,
            cos_base=args.cos_base,
        )
        self.setup_index_manager(batch_size=self.batch_size)
        self.setup_tokenizer(args)
        self.setup_system_prompt(args)

        # ===========================================================================
        # Modality specific setup

        # TEXT: setup pure text
        if "text" in self.modality:
            self.setup_text(args)

        # IMAGE: setup vae info and vit info
        if "vae_image" in self.modality or "vit_image" in self.modality:
            self.setup_image(args)

        # ===========================================================================
        # TASK specific setup

        # IMAGE CAPTION: setup caption manager. For T2I and TI2I
        if "t2i" in self.dataset_tag:
            self.setup_image_caption(args)
            if args.gen_image_template != "dit_it":
                assert self.image_text_token_length + self.vae_image_token_length <= self.max_token_length, (
                    f"Text length ({self.image_text_token_length}) + image length ({self.vae_image_token_length}) "
                    f"exceeds specified max_token_length ({self.max_token_length})"
                )

        # MMU
        if "mmu" in self.dataset_tag:
            if self.task_kwargs["data_format"] == "captioning":
                self.setup_image_caption(args)
            self.mmu_caption_short_keys = task_kwargs.get("short_keys", ["short_caption"])

        # ===========================================================================
        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)
        # After init, we set the logger to all_logger to print warnings and errors of all ranks
        self.logger = all_logger

    def __post_init__(self, **kwargs):
        pass

    # =========================
    #     Common Helpers
    # =========================

    def setup_tokenizer(self, args):
        if not self.use_tokenizer:
            return
        
        self.require_configs(args, ["sequence_template", "tokenizer_name", "tokenizer_class"], "text modality")
        # Tokenizer
        self.tokenizer: Union[TokenizerWrapper, HunyuanMultimodalTokenizerFast] = \
            load_tokenizer(args.tokenizer_name, args.tokenizer_class)

        # Sequence template
        self.sequence_template = args.sequence_template
        assert self.sequence_template in ["pretrain", "instruct"], f"Unsupported template: {self.sequence_template}"
        # -- for instruct
        conv_template = default(args.conv_template, args.model_name.split(".")[-1])
        self.default_conv = get_conversation_template(conv_template)
        self.roles = self.default_conv.roles

        # Define start tokens to be ignored during training when using predicted tokens as input
        # Pass empty list to disable this feature
        self.ignore_start_tokens = set(args.ignore_start_tokens)

        # Predefined sections
        self.decorator_sections = DecoratorSections(
            self.tokenizer, self.default_conv, self.sequence_template, self.ignore_start_tokens
        )

    def setup_system_prompt(self, args):
        _ = args
        # System prompt related
        self.system_prompt_token_length = self.task_kwargs.get('system_prompt_token_length', 0)

        system_prompt_type = self.task_kwargs.get("system_prompt_type", "none").lower()
        self.system_prompt_type, *extra = system_prompt_type.split('@')
        self.system_prompt_prob = float(extra[0]) if extra else 1.0
        # assert self.system_prompt_type in {"none", "conv_template", "fixed_set", "en_unified", "vanilla"}, \
        #     f"Unsupported system_prompt_type: {self.system_prompt_type}"

    def get_system_prompt(self, data: MultimodalDataContainer, return_section=False, ignore=True):
        if self.system_prompt_type == "none":
            return [] if return_section else ""

        assert self.system_prompt_token_length > 0, \
            (f"`system_prompt_token_length` should be greater than 0 when using "
             f"system prompt type: {self.system_prompt_type}")

        if self.system_prompt_type == "en_unified":
            candidates = unified_system_prompts["en_unified"]
        elif self.system_prompt_type == "vanilla":
            candidates = vanilla_system_prompts["en"]
        elif self.system_prompt_type.startswith("li-dit"):
            # Will be added in the prepare_xxx_model_inputs.
            return [] if return_section else ""
        else:   # fixed_set
            sp_type = (
                "en_think_recaption" if data.reasoning else "en_recaption"
            ) if data.recaption else "en_vanilla"
            candidates = t2i_system_prompts[sp_type]

        sp = (random.choice(candidates) if len(candidates) > 1 else candidates[0]).strip("\n ")
        if self.system_prompt_prob < 1.0 and random.random() > self.system_prompt_prob:
            sp = ""

        if return_section:
            if sp == "":
                return []
            else:
                assert self.system_prompt_token_length > 0, "system_prompt_token_length should be greater than 0"
                return [
                    dict(type="text", text=sp, ignore=ignore, max_length=self.system_prompt_token_length - 1),
                    dict(type="text", text=self.default_conv.sep_sp, ignore=ignore),  # "\n\n" 1 token
                ]

        return sp

    def parse_task_and_index_kwargs(self, task_kwargs, index_kwargs, log_prefix):
        self.task_kwargs = self.strip_leading_tag(default(task_kwargs, {}), required=False)
        missing_task_kwargs = []
        for key in REQUIRED_TASK_KWARGS:
            if key not in self.task_kwargs:
                missing_task_kwargs.append(key)
        if missing_task_kwargs:
            raise ValueError(f"{log_prefix}Missing required task_kwargs: {missing_task_kwargs}")

        self.index_kwargs = self.strip_leading_tag(default(index_kwargs, {}), required=False)
        missing_index_kwargs = []
        for key in REQUIRED_INDEX_KWARGS:
            if key not in self.index_kwargs:
                missing_index_kwargs.append(key)
        if missing_index_kwargs:
            raise ValueError(f"{log_prefix}Missing required index_kwargs: {missing_index_kwargs}")

        self.logger.info(f"{log_prefix}{self.task_kwargs=}")
        self.logger.info(f"{log_prefix}{self.index_kwargs=}")

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

        if "vae_image" in self.modality or "vit_image" in self.modality:
            num_image_token_prefix = ImageInfo.num_image_token_prefix()
            num_image_token_suffix = ImageInfo.num_image_token_suffix()
        else:
            num_image_token_prefix = num_image_token_suffix = 0

        # Handle special cases
        meta_dict: dict[str, dict | list[dict]]
        if hasattr(self, "vit_info") and self.vit_info.encoder_type == "anyres-vit-for-a3b":
            meta_dict = dict(
                gen_image={},
                cond_vae_image={},
                cond_vit_image={"start_offset": 1},
            )
        else:
            meta_dict = dict(
                gen_image={},
                cond_vae_image={},
                cond_vit_image={},
            )
        meta_dict["cond_joint_image"] = [meta_dict["cond_vae_image"], meta_dict["cond_vit_image"]]

        if self.args.rope_type_extended in ["2d", "xdrope", "interleaved_mrope"]:
            image_slices = output.all_image_slices
            image_idx = 0
            offset = 0
            num_overlapped_tokens = 0
            image_shapes = []
            shifted_image_slices = []
            metas = []
            last_is_gen = False
            for section in sections:
                if image_idx >= len(image_slices):
                    break
                if section['type'] == 'gen_image':
                    # Add gen_image
                    image_shapes.append([section['token_height'], section['token_width']])
                    metas.append(meta_dict['gen_image'])
                    sli = image_slices[image_idx]
                    shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                    image_idx += 1
                    # Add <eoi>
                    image_shapes.append([None, None])
                    metas.append({})
                    suffix_start = sli.stop
                    suffix_end = sli.stop + num_image_token_suffix
                    shifted_image_slices.append(slice(suffix_start - offset, suffix_end - offset))
                    # Add offset for next sections
                    offset += section['token_length'] + num_image_token_prefix + num_image_token_suffix
                    last_is_gen = True

                elif section['type'] in ['cond_joint_image', 'cond_vae_image', 'cond_vit_image']:
                    # We have an important assumption here that, if a gen_image is not the last section,
                    # it must be followed by a joint_image section immediately.
                    # That means [gen_image][other sections...][joint_image] is not allowed.
                    if last_is_gen:
                        # Append a text section to shift <boi><size><ratio><timestep> tokens
                        image_shapes.append([None, None])
                        metas.append({})
                        sli = image_slices[image_idx]
                        prefix_start = sli.start - num_image_token_prefix
                        prefix_end = sli.start
                        shifted_image_slices.append(slice(prefix_start - offset, prefix_end - offset))
                        num_overlapped_tokens = offset

                    if isinstance(section['token_height'], list):
                        assert len(section['token_height']) == len(section['token_width']), \
                            (f"token_height and token_width should have the same length, "
                             f"but got {len(section['token_height'])} and {len(section['token_width'])}")
                        assert len(section['token_height']) == 2, \
                            f"For cond_joint_image, two images are required, but got {len(section['token_height'])}"
                        # shift image slice
                        for i in range(len(section['token_height'])):
                            image_shapes.append([section['token_height'][i], section['token_width'][i]])
                            metas.append(meta_dict[section['type']][i])
                            sli = image_slices[image_idx + i]
                            shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                        image_idx += len(section['token_height'])
                    else:
                        image_shapes.append((section['token_height'], section['token_width']))
                        metas.append(meta_dict[section['type']])
                        # shift image slice
                        sli = image_slices[image_idx]
                        shifted_image_slices.append(slice(sli.start - offset, sli.stop - offset))
                        if last_is_gen:
                            # Append a text section to shift <eoi> token
                            image_shapes.append([None, None])
                            metas.append({})
                            suffix_start = sli.stop
                            suffix_end = sli.stop + num_image_token_suffix
                            shifted_image_slices.append(slice(suffix_start - offset, suffix_end - offset))
                        image_idx += 1
                    last_is_gen = False

                elif section['type'] == 'text':
                    pass

                else:
                    raise NotImplementedError(f"Unsupported section type: {section['type']} for rope image info.")

            # Remove tailing text sections (they will be automatically appended, remove for backward compatibility)
            while image_shapes and image_shapes[-1] == [None, None]:
                image_shapes.pop()
                shifted_image_slices.pop()
                metas.pop()
            assert len(shifted_image_slices) == len(image_shapes), (
                f"Size miss matched: {len(shifted_image_slices)=} != {len(image_shapes)=}"
            )
            assert len(shifted_image_slices) == len(metas), (
                f"Size miss matched: {len(shifted_image_slices)=} != {len(metas)=}"
            )
            return list(map(list, zip(shifted_image_slices, image_shapes, metas))), num_overlapped_tokens

        return None, 0

    # ==========================================
    #     Data getters and template builders
    # ==========================================

    @resample_for_errors
    def get_t2i_data(self, index) -> MultimodalDataContainer:
        self.require_configs(self.index_columns, "image_col",
                             f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")
        # 增加离线提取latent时的index_columns assert，确保index_columns中包含image_latent_col
        if getattr(self, "image_data_format", "pixels") == "latents":
            self.require_configs(self.index_columns, "image_latent_col",
                                 f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")
        # Only t2i task use random crop. For other editing tasks, make sure random_crop as False to use center crop.
        # T2i multireso bucket relies on the height/width property in data-pipeline which may not correctly handled
        # exif. So we disable exif here.
        tgt_image, img_success = self.get_image_with_size(
            index,
            random_crop=self.task_kwargs.get("random_crop", True),
            target_size_type=self.task_kwargs.get("target_size_type", "image"),  
            return_type="vae",
            apply_exif=self.task_kwargs.get("apply_exif", False),
            url_cos_col=self.index_columns.get("url_cos_col"),
            **self.index_columns["image_col"],
        )

        # Get caption
        cap_out, cap_success = self.get_image_caption(index, tgt_image)

        # -- user prompt and recaption (optional)
        if isinstance(cap_out, tuple):
            prompt = cap_out[0].caption
            recaption = cap_out[1].caption
            lang = cap_out[0].lang
        else:
            prompt = cap_out.caption
            recaption = None
            lang = cap_out.lang

        text_max_length = 8192 if self.single_text_max_length == 0 else self.single_text_max_length
        if cap_success and isinstance(prompt, str) and len(prompt) > text_max_length:
            cap_success = False
        if cap_success and isinstance(recaption, str) and len(recaption) > text_max_length:
            cap_success = False

        # -- for instruction (optional)
        if self.sequence_template == "instruct" and lang is not None:
            instruction = self.get_instruct(text2image_instructions[lang])
            prompt = f"{instruction}{prompt}"

        # -- for reasoning (optional)
        if recaption is not None and self.reasoning_cot_prob > 0 and random.random() < self.reasoning_cot_prob:
            # self.extra_think_col should already registered in self.register_extra_cols()
            reasoning = self.index_manager.get_attribute(index, **self.index_columns[f"think_{lang}_col"])   # reasoning text
            # Remove leading and trailing <think> tags if they exist. These tags will be added in the template for
            # detailed control.
            if reasoning is not None and reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>"):]
            if reasoning is not None and reasoning.endswith("</think>"):
                reasoning = reasoning[:-len("</think>")]
            if cap_success and isinstance(reasoning, str) and len(reasoning) > text_max_length:
                cap_success = False
        else:
            reasoning = None

        return MultimodalDataContainer(
            prompt=prompt, recaption=recaption, reasoning=reasoning,
            images=[tgt_image],
            success=img_success and cap_success, index=index,
            dataset_tag=self.dataset_tag,
        )

    def build_t2i_template(self, data: MultimodalDataContainer, num_predicted_image_token_offsets: tuple[int, int]) -> list[dict]:
        # Unconditional
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        uncond_kwargs = dict(
            uncond_enabled=do_uncond,
            uncond_p=(1.0 if do_uncond else 0.0),
            uncond_length=self.args.uncond_length,
        )

        # Sequence decorator sections
        deco = self.decorator_sections

        extra_num_tokens = self.task_kwargs.get("extra_num_tokens", -1)
        dit_it_section_mode = self.args.gen_image_template == "dit_it" 
        final_bot_sep = self.args.gen_image_template == "default" or self.task_kwargs.get("pretrain_add_bot_sep", False)
        if extra_num_tokens < 0:
            # 1 mean <bos> token
            extra_num_tokens = 1 + data.num_image_special_tokens + deco.user_length + deco.user_sep_length
            if not self.sequence_pack and "vit" in self.args.dummy_type:
                extra_num_tokens += self.dummy_number_dict["vit"]
            if not dit_it_section_mode:
                extra_num_tokens += deco.bot_length + deco.answer_length
            if final_bot_sep:
                extra_num_tokens += deco.bot_sep_length

        system_prompt_section = self.get_system_prompt(data, return_section=True)

        prompt_max_length = self.image_prompt_token_length - extra_num_tokens
        assert prompt_max_length > 0, f"prompt_max_length should be greater than 0, got {prompt_max_length}"
        prompt_section = [
            dict(type="text", text=data.prompt, max_length=prompt_max_length,
                 **uncond_kwargs, ignore=not self.predict_prompt or do_uncond),
        ]
        num_predicted_image_token_start_offset, num_predicted_image_token_end_offset = num_predicted_image_token_offsets
        gen_section = [
            dict(type="text", text='', start_offset=num_predicted_image_token_start_offset, 
                end_offset=num_predicted_image_token_end_offset, ignore=do_uncond),
            dict(type="gen_image", **data.images[0].i.meta_info),
        ]
        cot_sections = []
        if data.reasoning:  # if not None and not empty
            cot_sections += deco.think(
                dict(type="text", text=data.reasoning, ignore=do_uncond,
                     max_length=self.image_reasoning_token_length - 2, **uncond_kwargs),
                do_uncond=do_uncond,
            )
        if data.recaption:
            cot_sections += deco.recaption(
                dict(type="text", text=data.recaption, ignore=do_uncond,
                     max_length=self.image_recaption_token_length - 2, **uncond_kwargs),
                do_uncond=do_uncond,
            )

        # Compose all sections
        if dit_it_section_mode:
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep
            )
        else:
            sections = (
                    system_prompt_section +
                    deco.user + prompt_section + deco.user_sep +
                    deco.bot + cot_sections + deco.answer(gen_section)
            )
        # For non-default gen_image_template, final bot_sep is not needed
        if final_bot_sep:
            sections += deco.bot_sep_sections(ignore=self.ignore_bot_sep_token)

        return sections

    @resample_for_errors
    def get_lm_data(self, index) -> MultimodalDataContainer:
        # We support two types of text data:
        # 1. pure_text format
        # 2. conversation format
        self.require_configs(self.task_kwargs, "data_format", f"{self.dataset_tag} task")

        messages = []

        if self.task_kwargs["data_format"] == "conversation":
            columns = self.index_manager.get_columns(index)
            conversations = self.index_manager.get_attribute(index, **self.index_columns["conversation_col"])
            system_prompt = self.index_manager.get_attribute(index, "document") if "document" in columns else None
            success = len(conversations) > 0 and len(self.preprocess_text(conversations[0]["Assistant"])) > 0
            success_recaption = len(conversations) > 0 and conversations[0]["Assistant"] is None and len(self.preprocess_text(conversations[0]["recaption"])) > 0
            success = success or success_recaption
            if success:
                # if system_prompt is None or "", build_interleave_template will prepend a system prompt
                if system_prompt is not None and system_prompt != "":
                    messages.append(dict(type="system", text=self.preprocess_text(system_prompt)))
                for msg in conversations:
                    messages.append(dict(type="cond_text", text=self.preprocess_text(msg["User"])))
                    assistant = msg["Assistant"]
                    if assistant is not None:
                        assistant = self.preprocess_text(assistant)
                    bot = dict(type="gen_text", text=assistant)
                    if "reasoning" in msg and msg["reasoning"] is not None and msg["reasoning"].strip() != "":
                        reasoning = self.preprocess_text(msg["reasoning"], strip_think=True)
                        if reasoning:   # if not empty
                            bot["reasoning"] = [dict(type="gen_text", text=reasoning)]
                    if "recaption" in msg and msg["recaption"] is not None and msg["recaption"].strip() != "":
                        recaption = self.preprocess_text(msg["recaption"], strip_recaption=True)
                        if recaption:   # if not empty
                            bot["recaption"] = [dict(type="gen_text", text=recaption)]
                    messages.append(bot)

        elif self.task_kwargs["data_format"] == "pure_text":
            data = self.index_manager.get_attribute(index, **self.index_columns["text_col"])
            success = len(self.preprocess_text(data)) > 0
            if success:
                messages.extend([dict(type="gen_text", text=data)])

        else:
            raise ValueError(f"Unsupported data_format: {self.task_kwargs['data_format']} for lm task.")

        return MultimodalDataContainer(
            messages=messages,
            success=success, index=index,
            dataset_tag=self.dataset_tag,
        )

    @staticmethod
    def merge_consecutive_text_messages(raw_messages):
        """合并连续的文本消息，支持 text 类型及其语言字段"""
        if not raw_messages:
            return raw_messages
        
        merged = []
        for msg in raw_messages:
            msg_type = msg.get('type')
            # 支持的文本消息类型
            text_types = ['text', 'cond_text', 'gen_text']
            
            if msg_type in text_types:
                # 如果最后一个消息是相同类型的文本，合并
                if merged and merged[-1].get('type') == msg_type:
                    # 支持的语言字段列表
                    lang_fields = ['text_zh', 'text_en', 'text']
                    
                    # 合并所有相同的字段
                    for field in lang_fields:
                        if field in merged[-1] and field in msg:
                            prev_text = merged[-1].get(field)
                            curr_text = msg.get(field)
                            # 只有当两个文本都不为 None 时才合并
                            if prev_text is not None and curr_text is not None:
                                merged[-1][field] = prev_text + '\n\n' + curr_text if prev_text else curr_text
                            elif curr_text is not None:
                                merged[-1][field] = curr_text
                        elif field in msg:
                            # 如果当前消息有这个字段但前一个没有，直接添加
                            merged[-1][field] = msg.get(field)
                else:
                    merged.append(msg.copy())
            else:
                merged.append(msg)
        
        return merged

    @resample_for_errors
    def get_mmu_data(self, index) -> MultimodalDataContainer:
        # We assume the images are always at the start of the sequence
        # MMU supports two types of data:
        # - captioning format: the reversed t2i
        # - message format: visual question answering, ocr, chart, ...
        self.require_configs(self.task_kwargs, "data_format", f"{self.dataset_tag} task")

        if self.task_kwargs["data_format"] == "message":
            self.require_configs(self.index_columns, "message_col",
                                 f"{self.dataset_tag}[message].{self.dataset_tag}_index_kwargs.index_columns")
            self.require_configs(self.index_keys, "image_key",
                                 f"{self.dataset_tag}[message].{self.dataset_tag}_index_kwargs.index_keys")

            raw_messages = self.index_manager.get_attribute(index, **self.index_columns["message_col"])
            raw_messages = self.merge_consecutive_text_messages(raw_messages)
            lang = random.choice(["en", "zh"])
            messages, cond_images = [], []
            all_img_success, all_text_success = True, True

            for raw_msg in raw_messages:
                if raw_msg['type'] == 'cond_image':
                    cond_image, img_success = self.get_image_with_size(
                        src=raw_msg,
                        random_crop="resize",
                        target_size_type="image",
                        return_type=self.cond_image_type,
                        real_index=index,
                        apply_exif=self.task_kwargs.get("apply_exif", False),
                        url_cos_col=self.index_keys["url_cos_key"],
                        column=self.index_keys["image_key"]
                    )
                    cond_images.append(cond_image)
                    all_img_success = all_img_success and img_success
                    messages.append(dict(type=self.cond_image_section_type, metadata=cond_image.i.meta_info))

                elif raw_msg['type'] in ['cond_text', 'gen_text']:
                    text, text_success = self.preprocess_multilingual_text_with_ocr(raw_msg, lang, index)
                    messages.append(dict(type=raw_msg['type'], text=text))
                    all_text_success = all_text_success and text_success and (text != "")

                else:
                    raise ValueError(f"Unsupported message: {raw_msg['type']} in {index=}.")

        elif self.task_kwargs["data_format"] == "captioning":
            self.require_configs(self.index_columns, "image_col",
                                 f"{self.dataset_tag}[captioning].{self.dataset_tag}_index_kwargs.index_columns")

            cond_image, all_img_success = self.get_image_with_size(
                src=index,
                random_crop="resize",
                target_size_type="image",
                return_type=self.cond_image_type,
                real_index=index,
                apply_exif=self.task_kwargs.get("apply_exif", False),
                url_cos_col=self.index_columns.get("url_cos_col"),
                **self.index_columns["image_col"],
            )
            cond_images = [cond_image]

            cap_out, all_text_success = self.get_image_caption(index)
            if cap_out.key in self.mmu_caption_short_keys:
                user_input = random.choice(image_captioning_instructions_dict["short"][cap_out.lang])
            else:
                user_input = random.choice(image_captioning_instructions_dict["long"][cap_out.lang])

            messages = [
                dict(type=self.cond_image_section_type, metadata=cond_image.i.meta_info),
                dict(type="cond_text", text=user_input),
                dict(type="gen_text", text=cap_out.caption),
            ]

        else:
            raise ValueError(f"Unsupported data_format: {self.task_kwargs['data_format']} for mmu task.")

        container_cond_key = dict(
            vae_vit="cond_images",
            vae="cond_vae_images",
            vit="cond_vit_images",
        )
        return MultimodalDataContainer(
            **{container_cond_key[self.cond_image_type]: cond_images},
            messages=messages,
            success=all_img_success and all_text_success, index=index,
            dataset_tag=self.dataset_tag,
        )

    @resample_for_errors
    def get_interleave_data(self, index) -> MultimodalDataContainer:
        raw_messages = self.index_manager.get_attribute(index, **self.index_columns["message_col"])
        raw_messages = self.merge_consecutive_text_messages(raw_messages)
        lang = random.choice(["en", "zh"])
        messages, gen_images, cond_images = [], [], []
        all_img_success, all_text_success = True, True

        reasoning_messages_stash = []
        recaption_messages_stash = []

        last_round_gen_start_index = None
        for idx, msg in enumerate(raw_messages[::-1]):
            if "cond" in msg['type']:
                last_round_gen_start_index = len(raw_messages) - 1 - idx
                break
        assert last_round_gen_start_index is not None, "No cond message found in the messages"

        for idx, msg in enumerate(raw_messages):
            is_last_message = idx == len(raw_messages) - 1
            next_is_cond = not is_last_message and raw_messages[idx + 1]["type"].startswith("cond")
            ignore_gen = self.interleave_only_last_round and idx < last_round_gen_start_index and not is_last_message
            if msg['type'] == 'cond_image':
                cond_image, img_success = self.get_image_with_size(
                    src=msg,
                    random_crop=False,
                    target_size_type="image",
                    return_type=self.cond_image_type,
                    real_index=index,
                    apply_exif=self.task_kwargs.get("apply_exif", False),
                    url_cos_col=self.index_keys["url_cos_key"],
                    column=self.index_keys["image_key"]
                )
                cond_images.append(cond_image)
                all_img_success = all_img_success and img_success
                messages.append(
                    dict(
                        type=self.cond_image_section_type,
                        metadata=cond_image.i.meta_info,
                    )
                )

            elif msg['type'] == 'gen_image':
                if not ignore_gen:
                    gen_image, img_success = self.get_image_with_size(
                        src=msg,
                        random_crop=False,
                        target_size_type="image",
                        return_type="vae",
                        real_index=index,
                        apply_exif=self.task_kwargs.get("apply_exif", False),
                        url_cos_col=self.index_keys["url_cos_key"],
                        column=self.index_keys["image_key"]
                    )
                    gen_images.append(gen_image)
                    all_img_success = all_img_success and img_success

                # if not is_last_message:
                # get cond_image for next round including last message
                cond_image, img_success = self.get_image_with_size(
                    src=msg,
                    random_crop=False,
                    target_size_type="image",
                    return_type=self.cond_image_type,
                    real_index=index,
                    apply_exif=self.task_kwargs.get("apply_exif", False),
                    url_cos_col=self.index_keys["url_cos_key"],
                    column=self.index_keys["image_key"]
                )
                cond_images.append(cond_image)
                all_img_success = all_img_success and img_success

                bot = dict(
                    type="gen_image" if not ignore_gen else "gen_image_ignore",
                    metadata=gen_image.i.meta_info if not ignore_gen else None,
                )
                if reasoning_messages_stash:
                    bot['reasoning'] = reasoning_messages_stash
                if recaption_messages_stash:
                    bot['recaption'] = recaption_messages_stash
                messages.append(bot)
                cond_msg = dict(
                    type=self.cond_image_section_type,
                    metadata=cond_image.i.meta_info,
                )
                if next_is_cond:
                    # The next raw message starts a real condition turn, so keep bot_sep
                    # between the generated image and the reused condition image.
                    messages.append(cond_msg)
                else:
                    # Otherwise keep the reused condition image inside the bot turn,
                    # so bot_sep is emitted after it.
                    bot["post_messages"] = [
                        *bot.get("post_messages", []),
                        cond_msg,
                    ]
                reasoning_messages_stash = []
                recaption_messages_stash = []

            elif msg['type'] == 'gen_image_think':
                if not ignore_gen:
                    gen_image, img_success = self.get_image_with_size(
                        src=msg,
                        random_crop=False,
                        target_size_type="image",
                        return_type="vae",
                        real_index=index,
                        apply_exif=self.task_kwargs.get("apply_exif", False),
                        url_cos_col=self.index_keys["url_cos_key"],
                        column=self.index_keys["image_key"]
                    )
                    gen_images.append(gen_image)
                    all_img_success = all_img_success and img_success

                cond_image, img_success = self.get_image_with_size(
                    src=msg,
                    random_crop=False,
                    target_size_type="image",
                    return_type=self.cond_image_type,
                    real_index=index,
                    apply_exif=self.task_kwargs.get("apply_exif", False), 
                    url_cos_col=self.index_keys["url_cos_key"],
                    column=self.index_keys["image_key"]
                )
                cond_images.append(cond_image)
                all_img_success = all_img_success and img_success

                reasoning_messages_stash.append(
                    dict(
                        type="gen_image" if not ignore_gen else "gen_image_ignore",
                        metadata=gen_image.i.meta_info if not ignore_gen else None,
                        post_messages=[
                            dict(
                                type=self.cond_image_section_type,
                                metadata=cond_image.i.meta_info,
                            )
                        ]
                    )
                )

            elif msg['type'] == 'system':
                system_prompt = self.preprocess_multilingual_text(msg, lang)
                messages.append(dict(type="system", text=self.preprocess_text(system_prompt)))
            elif msg['type'] == 'cond_text':
                text, cur_text_success = self.preprocess_multilingual_text_with_ocr(msg, lang, index)
                messages.append(dict(type='cond_text', text=text))
                all_text_success = all_text_success and cur_text_success and (text != "")
            elif msg['type'] == 'gen_text':
                text, cur_text_success = self.preprocess_multilingual_text_with_ocr(msg, lang, index)
                bot = dict(type='gen_text' if not ignore_gen else 'gen_text_ignore', text=text)
                if reasoning_messages_stash:
                    bot['reasoning'] = reasoning_messages_stash
                if recaption_messages_stash:
                    bot['recaption'] = recaption_messages_stash
                messages.append(bot)
                reasoning_messages_stash = []
                recaption_messages_stash = []
                all_text_success = all_text_success and cur_text_success and (text != "")
            
            elif msg['type'] == 'gen_text_think':
                text, cur_text_success = self.preprocess_multilingual_text_with_ocr(msg, lang, index)
                reasoning_messages_stash.append(dict(type='gen_text' if not ignore_gen else 'gen_text_ignore', text=text))
                all_text_success = all_text_success and cur_text_success and (text != "")
            
            elif msg['type'] == 'gen_text_recaption':
                text, cur_text_success = self.preprocess_multilingual_text_with_ocr(msg, lang, index)
                recaption_messages_stash.append(dict(type='gen_text' if not ignore_gen else 'gen_text_ignore', text=text))
                all_text_success = all_text_success and cur_text_success and (text != "")

            else:
                raise ValueError(f"Unsupported message: {msg['type']} in {index=}.")

        return MultimodalDataContainer(
            images=gen_images,
            cond_images=cond_images,
            messages=messages,
            success=all_img_success and all_text_success, index=index,
            dataset_tag=self.dataset_tag,
        )

    @resample_for_errors
    def get_std_messages_data(self, index) -> MultimodalDataContainer:
        raw_data = self.index_manager.get_attribute(index, **self.index_columns["message_col"])
        messages, gen_images, cond_images = [], [], []
        all_img_success, all_text_success = True, True

        image_list = raw_data["image_list"]
        image_idx = 0
        loss_mask = raw_data["loss_mask"]
        raw_messages = raw_data["messages"]
        assert len(loss_mask) == len(raw_messages), \
            f"Length of loss_mask ({len(loss_mask)}) and messages ({len(raw_messages)}) should be the same."
        mask2ignore = [True, False]
        image_placeholder = "<image_0bf67t>"
        image_placeholder_len = len(image_placeholder)

        system_counter = 0
        tool_calls_buffer = None
        tool_responses_buffer = []
        for idx, msg in enumerate(raw_messages):

            if msg["role"] == "system":
                system_counter += 1
                if system_counter > 1:
                    raise ValueError(f"Only one system message is allowed, but got multiple in {index=}.")

                system_prompt = self.preprocess_text(msg["content"])
                if "tools" in raw_data:
                    system_prompt += self.format_tool_descriptions(raw_data["tools"])

                messages.append(dict(
                    type="system",
                    text=system_prompt,
                    ignore=mask2ignore[loss_mask[idx]],
                ))

            elif msg["role"] == "user":
                user_content = self.preprocess_text(msg["content"])
                # If has image placeholder, replace with cond_image
                if image_placeholder in user_content:
                    user_prompt = user_content.replace(image_placeholder, "")
                    num_images = (len(user_content) - len(user_prompt)) // image_placeholder_len
                    assert image_idx + num_images <= len(image_list), \
                        f"Not enough images for user message in {index=}: " \
                        f"need {num_images}, but only {len(image_list) - image_idx} left."
                    for _ in range(num_images):
                        cond_image, img_success = self.get_image_with_size(
                            src=dict(cache_image=image_list[image_idx]),
                            random_crop=False,
                            target_size_type="image",
                            return_type=self.cond_image_type,
                            real_index=index,
                            apply_exif=self.task_kwargs.get("apply_exif", False),
                            column="cache_image",
                        )
                        cond_images.append(cond_image)
                        all_img_success = all_img_success and img_success
                        image_idx += 1
                        messages.append(dict(
                            type=self.cond_image_section_type,
                            metadata=cond_image.i.meta_info,
                        ))
                else:
                    user_prompt = user_content

                # Finally add user message
                user_prompt = self.preprocess_text(user_prompt)
                messages.append(dict(
                    type="cond_text",
                    text=user_prompt,
                    ignore=mask2ignore[loss_mask[idx]],
                ))
                all_text_success = all_text_success and (user_prompt != "")

            elif msg["role"] == "assistant":
                # 1. reasoning
                reasoning = self.preprocess_text(msg.get("reasoning_content", ""))
                if reasoning != "":
                    reasoning = dict(
                        type="gen_text",
                        text=self.preprocess_text(reasoning),
                        ignore=mask2ignore[loss_mask[idx]],
                    )

                # 2. bot answer
                bot_content = self.preprocess_text(msg["content"])
                bot_message = dict(
                    type="gen_text",
                    text=bot_content,
                    ignore=mask2ignore[loss_mask[idx]],
                )
                if reasoning:
                    bot_message["reasoning"] = [reasoning]

                messages.append(bot_message)

                # 3. tool calls
                if (tool_calls := msg.get("tool_calls")) is not None:
                    # Add to buffer, until tool responses are available for processing at the same time
                    assert isinstance(tool_calls, list), f"tool_calls should be a list, got {type(tool_calls)}"
                    tool_calls_buffer = tool_calls

            elif msg["role"] == "tool":
                # The text of tool defaults to be ignored.
                tool_responses_buffer.append(msg)

                if len(tool_responses_buffer) == len(tool_calls_buffer):
                    call_message, response_messages, status = self.format_tool_calls(
                        tool_calls=tool_calls_buffer,
                        tool_responses=tool_responses_buffer,
                        index=index,
                        image_list=image_list,
                        image_idx=image_idx,
                        img_success=True,
                    )

                    # TODO: tool-specific args and status should have a better way to manager
                    # - generate function
                    image_idx = status.get("image_idx", image_idx)
                    all_img_success = all_img_success and status.get("img_success", True)
                    gen_images.extend(status.get("gen_images", []))

                    messages.extend([call_message] + response_messages)

                    # Clear buffers
                    tool_calls_buffer = None
                    tool_responses_buffer.clear()

                else:
                    # Wait for all tool responses to be collected
                    pass

            else:
                raise ValueError(f"Unsupported message role: {msg['role']} in {index=}.")

        return MultimodalDataContainer(
            images=gen_images,
            cond_images=cond_images,
            messages=messages,
            success=all_img_success and all_text_success, index=index,
            dataset_tag=self.dataset_tag,
        )

    def build_interleave_template(self, data: MultimodalDataContainer, num_predicted_image_token_offsets: tuple[int, int]) -> list[dict]:
        deco = self.decorator_sections
        sections = []

        # Unconditional
        if hasattr(self, "uncond_p"):
            do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        else:
            do_uncond = False
        uncond_kwargs = dict(uncond_enabled=do_uncond, uncond_p=(1.0 if do_uncond else 0.0))

        # If data does not contain system prompt, add system prompt to the beginning of the sections, otherwise skip.
        # We assume there can be and there can be only one system prompt message in the message list and it is the first message.
        if data.messages[0]["type"] != "system":
            sections += self.get_system_prompt(data, return_section=True)

        def message_to_sections(msg: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
            if len(msg) == 0:   # skip for empty messages
                return []
            if isinstance(msg, list):
                return sum([message_to_sections(m) for m in msg], [])

            assert isinstance(msg, dict), f"msg should be dict or list of dict, got {type(msg)}"
            if msg["type"] in ["cond_vae_image", "cond_vit_image", "cond_joint_image"]:
                return [dict(type=msg["type"], **msg["metadata"])]
            elif msg["type"] == "gen_image":
                num_predicted_image_token_start_offset, num_predicted_image_token_end_offset = num_predicted_image_token_offsets
                return [
                    dict(type="text", text='', start_offset=num_predicted_image_token_start_offset, 
                        end_offset=num_predicted_image_token_end_offset, ignore=do_uncond),
                    dict(type="gen_image", **msg["metadata"])
                ] + message_to_sections(msg.get("post_messages", []))
            elif msg["type"] == "gen_image_ignore":
                return message_to_sections(msg.get("post_messages", []))
            elif msg["type"] == "system":
                assert self.system_prompt_token_length > 0, \
                    (f"A system message is found in the data, but system_prompt_token_length "
                     f"is set to {self.system_prompt_token_length}.")
                return [
                    dict(type="text", text=msg["text"], ignore=msg.get("ignore", True), max_length=self.system_prompt_token_length - 1),
                ] + (
                    [dict(type="text", text=self.default_conv.sep_sp, ignore=msg.get("ignore", True))]  # "\n\n" 1 token
                    if self.default_conv.sep_sp else []
                )
            elif msg["type"] == "cond_text":
                return [dict(type='text', text=msg["text"], ignore=msg.get("ignore", True), **uncond_kwargs, max_length=self.input_max_token_length)]
            elif msg["type"] == "gen_text":
                return [dict(type='text', text=msg["text"], ignore=do_uncond or msg.get("ignore", False), **uncond_kwargs)]
            elif msg["type"] == "gen_text_ignore":
                return [dict(type='text', text=msg["text"], ignore=True, **uncond_kwargs)]
            elif msg["type"] == "tool_response_gen_image":
                return [dict(type="gen_image", **msg["metadata"])]
            elif msg["type"] == "tool_response_text":
                return [dict(type='text', text=msg["text"], ignore=True)]
            else:
                raise ValueError(f"Unsupported message type: {msg['type']} for {self.dataset_tag} task.")

        def collect_cot(msg: dict[str, Any], cot_type: str) -> list[dict[str, Any]]:
            if cot_type in msg:
                return message_to_sections(msg[cot_type])
            return []

        # Process general message list. Successive messages are allowed from the same role.
        msg_idx = 0
        while msg_idx < len(data.messages):
            cur_type = data.messages[msg_idx]["type"][0]    # 'c'(cond), 'g'(gen), 's'(system), 't'(tool_response)
            if cur_type == 's':   # system
                sections += message_to_sections(data.messages[msg_idx])
                msg_idx += 1
                continue
            # Collect successive messages from the same type
            # think sections always be placed at the start of gen sections
            sub_sections = []
            think_sections = []
            recaption_sections = []
            while msg_idx < len(data.messages) and data.messages[msg_idx]["type"][0] == cur_type:
                msg = data.messages[msg_idx]
                think_sections += collect_cot(msg, cot_type="reasoning")
                recaption_sections += collect_cot(msg, cot_type="recaption")
                sub_sections += message_to_sections(msg)
                msg_idx += 1

            # Add decorator sections
            if cur_type == 'c':   # cond
                sections += deco.user + sub_sections + deco.user_sep
            elif cur_type == 'g':   # gen
                cot_sections = []
                if think_sections:
                    cot_sections += deco.think(think_sections)
                if recaption_sections:
                    cot_sections += deco.recaption(recaption_sections)
                sections += deco.bot + cot_sections + deco.answer(sub_sections) + deco.bot_sep_sections(ignore=self.ignore_bot_sep_token)
            elif cur_type == 't':  # tool_response
                sections += deco.tool_responses(sub_sections)
            else:
                raise ValueError(f"Unsupported message type: {data.messages[msg_idx]['type']} for mmu task.")

        return sections

    def __getitem__(self, index):
        index = int(index)

        if "vae_image" in self.modality or "vit_image" in self.modality:
            num_predicted_image_token_offsets = (ImageInfo.num_predicted_image_token_start_offset(), ImageInfo.num_predicted_image_token_end_offset())
            num_image_token_prefix = ImageInfo.num_image_token_prefix()
            num_image_token_suffix = ImageInfo.num_image_token_suffix()
        else:
            num_predicted_image_token_offsets = (0, 0)
            num_image_token_prefix = num_image_token_suffix = 0

        # Get data and build template
        if self.dataset_tag.startswith("t2i"):
            data = self.get_t2i_data(index)
            sections = self.build_t2i_template(data, num_predicted_image_token_offsets)

        elif self.dataset_tag.startswith("lm"):
            data = self.get_lm_data(index)
            sections = self.build_interleave_template(data, num_predicted_image_token_offsets)

        elif self.dataset_tag.startswith("mmu"):
            data = self.get_mmu_data(index)
            sections = self.build_interleave_template(data, num_predicted_image_token_offsets)

        elif self.dataset_tag.startswith("interleave"):
            data = self.get_interleave_data(index)
            sections = self.build_interleave_template(data, num_predicted_image_token_offsets)

        elif self.dataset_tag.startswith("pair"):
            data = self.get_std_messages_data(index)
            sections = self.build_interleave_template(data, num_predicted_image_token_offsets)

        else:
            raise ValueError(f"Unsupported dataset tag: {self.dataset_tag}")

        if data.cond_vit_images and getattr(data.cond_vit_images[0], "vision_encoder_kwargs", None) is not None:
            # Check if it's qwen3vl by image_type
            image_type = getattr(data.cond_vit_images[0].i, "image_type", None)
            if image_type == "qwen3vl":
                # qwen3vl needs grid_thw parameter
                cond_vit_image_kwargs = {
                    "grid_thw": maybe_stack(data.cond_vit_images, key="vision_encoder_kwargs", subkey="grid_thw"),
                }
            else:
                # Other vision encoders (e.g., siglip2) need spatial_shapes and attention_mask
                cond_vit_image_kwargs = {
                    "spatial_shapes": maybe_stack(data.cond_vit_images, key="vision_encoder_kwargs", subkey="spatial_shapes"),
                    "attention_mask": maybe_stack(data.cond_vit_images, key="vision_encoder_kwargs", subkey="pixel_attention_mask"),
                }
        else:
            cond_vit_image_kwargs = None
        
        dummy_type_dict = {}
        has_vit = any([(section["type"] == "cond_vit_image") or (section["type"] == "cond_joint_image") for section in sections])
        has_ts_proj = any([(section["type"] in ["gen_image"]) for section in sections])
        if "vit" in self.args.dummy_type and not has_vit:
            dummy_type_dict["vit"] = self.dummy_number_dict["vit"]
        if "ts_proj" in self.args.dummy_type and not has_ts_proj:
            dummy_type_dict["ts_proj"] = self.dummy_number_dict["ts_proj"]
        dummy_number = sum(dummy_type_dict.values())
        if self.sequence_pack:
            max_token_length = self.seq_length + 1 - sum(self.dummy_number_dict.values()) - self.reserved_pad_length
        else:
            max_token_length = self.max_token_length + 1 - dummy_number

        try:
            # If last valid section is gen_image, we won't add eos token at the end.
            last_is_gen_media = False
            for section in sections[::-1]:
                # Skip empty text sections
                if section["type"] == "text" and not section.get("text"):
                    continue
                last_is_gen_media = section["type"] in ["gen_image"]
                break
            
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_eos=False,
                drop_last=self.drop_last,
                add_pad=False if self.sequence_pack else 'auto',
                add_bos=self.default_conv.add_bos if hasattr(self, 'default_conv') else True,
                und_token_type=self.und_token_type,
                gen_token_type=self.gen_token_type,
            )
        except TypeError as e:
            self.logger.error(
                f"TypeError in encoding sections (dataset_tag={self.dataset_tag}, index={data.index}): "
                f"{max_token_length=}, {self.sequence_pack=}, {sections=}. Original error: {e}"
            )
            return self[(index + 100000) % len(self)]
        except AssertionError as e:
            self.logger.error(
                f"Error in encoding sections (dataset_tag={self.dataset_tag}, index={data.index}): "
                f"{max_token_length=}, {self.sequence_pack=}, {sections=}"
            )
            raise e

        target_tokens = output.tokens.clone()
        # If the ending images are dropped, the dummy text previous of the image still move end_offset, which
        # causes that target tokens will contain the successive pad tokens. So we always set pad tokens to -100.
        pad_token_mask = target_tokens == self.tokenizer.pad_token_id
        output.text_mask[pad_token_mask] = 0
        target_tokens[output.text_mask == 0] = -100

        # Remove unused images according to output (drop_last may drop some image sections)
        if self.task_kwargs.get("remove_unused_images", True):
            data.remove_unused_images(output)

        # Prepare attention mask. If no cond_image_type attribute, it means no media modalities,
        # therefore attention_mask can be skipped.
        cond_image_full_attn_slices = self.prepare_full_attn_slices(output, with_gen=False)
        gen_image_full_attn_slices = self.prepare_gen_full_attn_slices(output)
        if self.task_kwargs.get('attn_type', 'auto') == 'auto' and hasattr(self, "cond_image_type"):
            n_tokens = output.tokens.shape[0] - int(self.attn_mask_seq_m1) + dummy_number
            attention_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            for full_attn_sli, sli, num_prefix, num_suffix in chain(
                    zip(
                        gen_image_full_attn_slices,
                        output.gen_image_slices,
                        cycle([num_image_token_prefix]),
                        cycle([num_image_token_suffix]),
                    ),
            ):
                attention_mask[full_attn_sli, full_attn_sli] = True
                # Gen images will not to be attended by following tokens, except the last one following by eos.
                if sli.stop + 1 < output.tokens.shape[0] and output.tokens[sli.stop + 1] != self.tokenizer.eos_token_id:
                    # The hole follows the full-attn slice start so timestep+VAE are masked together.
                    hole_slice = slice(min(full_attn_sli.start, sli.start - num_prefix), sli.stop + num_suffix)
                    attention_mask[hole_slice.stop:, hole_slice] = False  # Make a hole under noise-kv

            for full_attn_slice in cond_image_full_attn_slices:
                attention_mask[full_attn_slice, full_attn_slice] = True
            attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = None

        # 2d rope
        rope_image_info, num_overlapped = self.get_rope_image_info(sections, output)

        ret = {
            # === required ===
            "dataset_tag": self.dataset_tag,
            "n_samples": 1,                                 # ()
            "index": data.index,                            # ()

            "status": data.status,                          # ()

            "tokens": output.tokens,                        # (seqlen)
            "target_tokens": target_tokens,                 # (seqlen)
            "text_mask": output.text_mask,                  # (seqlen)
            # === optional ===
            "attention_mask": attention_mask,               # (1, seqlen - 1, seqlen - 1)
            # -- gen image
            "images": maybe_stack(data.images),
            "image_mask": output.gen_image_mask,
            "image_slices": output.gen_image_slices,
            "image_full_attn_slices": gen_image_full_attn_slices,
            # -- cond image
            "cond_full_attn_slices": cond_image_full_attn_slices,   # [n1]
            "cond_vae_images": maybe_stack(data.cond_vae_images),   # (n1, 3, H, W) or [n1, (3, H, W)]
            "cond_vae_image_mask": output.vae_image_mask,           # (seqlen)
            "cond_vae_image_slices": output.vae_image_slices,       # [n1]
            "cond_vit_images": maybe_stack(data.cond_vit_images),   # (n1, vit_seqlen, ndim)
            "cond_vit_image_mask": output.vit_image_mask,           # (seqlen)
            "cond_vit_image_slices": output.vit_image_slices,       # [n1]
            "cond_vit_image_kwargs": cond_vit_image_kwargs,         # spatial_shapes: (n1, 2), attention_mask: (n1, vit_seqlen)]
            # -- position related
            "timesteps_index": output.gen_timestep_scatter_index,   # (n0)
            "cond_timesteps_index": output.cond_timestep_scatter_index, # (n1)
            "rope_image_info": rope_image_info,                     # [n0 + n1 + n1]
            # -- others
            "num_overlapped": num_overlapped,
            # -- dummy
            "dummy_type_dict": dummy_type_dict,
        }
        if self.args.use_mot:
            ret["und_token_indices"] = output.und_token_indices
            ret["gen_token_indices"] = output.gen_token_indices

        ret = {k: v for k, v in ret.items() if v is not None}

        return ret

    def seq_collate_fn(self, items):
        """ sequence collate function. It is used to combine multiple (non-batched) samples into one sample. """
        assert len(items) > 0
        p_state = get_parallel_state()
        cp_size = p_state.cp_size
        # max_sequence_length is used for solving bin packing
        assert self.max_sequence_length is not None, "max_sequence_length should be set in sequence packing mode."

        # TODO(ckczzjzhang): support different dummy_type_dict in a packed sequence or a batch for sequence pack
        assert all(item["dummy_type_dict"] == items[0]["dummy_type_dict"] for item in items), f"Different dummy_type_dict {[item['dummy_type_dict'] for item in items]} in a packed sequence has not been supported yet in {items[0]['dataset_tag']}."

        # # for seq_pack, we take the intersection of dummy_type_dict in all items as the final dummy_type_dict
        # common_keys = set.intersection(*[set(item["dummy_type_dict"].keys()) for item in items])
        # seq_pack_dummy_type_dict = {key: self.dummy_number_dict[key] for key in common_keys}

        # final_seq_length is the final length of packed sequence
        # after adding dummy and NTP shifting, its length should exactly be self.seq_length
        final_seq_length = self.seq_length + 1 - sum(items[0]["dummy_type_dict"].values())
        # final_seq_length = self.seq_length + 1 - sum(seq_pack_dummy_type_dict.values())
        first = items[0]
        seq_pad_value = {
            "tokens": self.tokenizer.pad_token_id,
            "target_tokens": -100,
            "text_mask": 0.0,
            "cond_vae_image_mask": False,
            "image_mask": False,
            "cond_vit_image_mask": False,
            "cond_vae_video_mask": False,
            "cond_vit_video_mask": False,
            "video_mask": False,
            "audio_mask": False,
        }

        lengths = [item["tokens"].shape[0] for item in items]
        offsets = [0] + list(itertools.accumulate(lengths))
        assert offsets[-1] <= final_seq_length, \
            f"Total length {offsets[-1]} exceeds final_seq_length {final_seq_length}. The lengths are {lengths}."

        new_item = {"offsets": torch.tensor(offsets)[None]}
        # Let batch size = 1
        for key, _ in first.items():
            # ================ No sequence & no position keys ================
            # Simple constant values
            if key in {"dataset_tag"}:
                new_item[key] = [item[key] for item in items]     # noqa
            elif key in {"dummy_type_dict"}:
                new_item[key] = [item[key] for item in items]
                # new_item[key] = [seq_pack_dummy_type_dict for _ in items]
            # Accumulated values
            elif key in {"n_samples"}:
                new_item[key] = torch.tensor([sum(item[key] for item in items)])    # noqa
            # Image/Video/Audio tensors
            elif key in {"images", "cond_vae_images", "cond_vae_videos", "videos", "audios", "video_last_frames"}:
                media_list = []
                # Atomic media shape [c, ...] with a-dimensions.
                # 离线 latent 模式下 images 是 [C, 1, H, W]（4 维），与 pixel 模式 [3, H, W]（3 维）不同。
                images_atom_ndim = 4 if getattr(self, "image_data_format", "pixels") == "latents" else 3
                # 离线 last frame latent 是 [C, 1, H, W]（4 维），像素是 [3, H, W]（3 维）。
                lf_atom_ndim = 4 if getattr(self, "last_frame_data_format", "pixels") == "latents" else 3
                atom_ndim = dict(
                    images=images_atom_ndim, cond_vae_images=3, cond_vae_videos=4,
                    videos=4, audios=2, video_last_frames=lf_atom_ndim,
                )[key]
                for item in items:
                    if isinstance(item[key], list): # must be list of a-D tensors
                        for im in item[key]:
                            if im.ndim == atom_ndim:
                                media_list.append(im)
                            else:
                                raise ValueError(f"Unsupported image tensor shape: {im.shape} for key {key}")
                    elif isinstance(item[key], torch.Tensor): # must be (a+1)-D tensor
                        if item[key].ndim == atom_ndim + 1:
                            media_list.extend(item[key])  # will unbind the batch dimension
                        else:
                            raise ValueError(f"Unsupported image tensor shape: {item[key].shape} for key {key}")
                    else:
                        raise ValueError(f"Unsupported image type: {type(item[key])} for key {key}")
                new_item[key] = [media_list]  # [list of 3-D Tensor]     # noqa
            elif key in {"cond_vit_images", "cond_vit_videos"}:
                if all([isinstance(item[key], torch.Tensor) for item in items]) and all(item[key].shape == items[0][key].shape for item in items):
                    new_item[key] = torch.cat([item[key] for item in items])[None]  # 4-D Tensor  # noqa
                else:
                    # Flatten variable-length media to a single list per batch for downstream use.
                    media_list = []
                    for item in items:
                        value = item[key]
                        if value is None:
                            continue
                        # A tensor here is (n_media, media_seqlen, ndim); unbind the media dim.
                        media_list.extend(value)
                    new_item[key] = [media_list]  # list of list of 2-D Tensor  # noqa
            elif key in {"cond_vit_image_kwargs", "cond_vit_video_kwargs"}:
                valid_items = [item for item in items if key in item and item[key] is not None]
                if valid_items:
                    cond_vit_images_key = "cond_vit_images" if key == "cond_vit_image_kwargs" else "cond_vit_videos"
                    if cond_vit_images_key in new_item and isinstance(new_item[cond_vit_images_key], torch.Tensor):
                        # If cond_vit_images is tensor (shape: (1, Σn_j, vit_seqlen, ndim)), 
                        # cond_vit_image_kwargs should be dict with tensor values (shape: (1, Σn_j, ...))
                        if "spatial_shapes" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "spatial_shapes": torch.cat([item[key]["spatial_shapes"] for item in valid_items])[None],   # (1, Σn_j, 2)
                                "attention_mask": torch.cat([item[key]["attention_mask"] for item in valid_items])[None],     # (1, Σn_j, seq_len)
                            }
                        elif "grid_thw" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "grid_thw": torch.cat([item[key]["grid_thw"] for item in valid_items])[None],  # (1, Σn_j, ...)
                            }
                        elif "video_grid_thw" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "video_grid_thw": torch.cat([item[key]["video_grid_thw"] for item in valid_items])[None],
                            }
                        else:
                            raise ValueError(f"Unknown cond_vit_image_kwargs structure: {valid_items[0][key].keys()}")
                    else:
                        # If cond_vit_images is list (format: [[tensor1, tensor2, ...]]),
                        # cond_vit_image_kwargs should be dict with list values (format: {"spatial_shapes": [tensor1, tensor2, ...], ...})
                        # This matches the structure expected by instantiate_vit_image_tokens when images is a list
                        if "spatial_shapes" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "spatial_shapes": [item[key]["spatial_shapes"] for item in valid_items],   # list of (n_i, 2) tensors
                                "attention_mask": [item[key]["attention_mask"] for item in valid_items],   # list of (n_i, seq_len) tensors
                            }
                        elif "grid_thw" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "grid_thw": [item[key]["grid_thw"] for item in valid_items],  # list of (n_i, ...) tensors
                            }
                        elif "video_grid_thw" in valid_items[0][key]:
                            new_item[key] = {   # noqa
                                "video_grid_thw": [item[key]["video_grid_thw"] for item in valid_items],
                            }
                        else:
                            raise ValueError(f"Unknown cond_vit_image_kwargs structure: {valid_items[0][key].keys()}")
                else:
                    # All items have None cond_vit_image_kwargs, set to None
                    new_item[key] = None

            # ================ No sequence & positional keys ================
            # Slices list
            elif key in {
                "cond_vae_image_slices", "cond_vit_image_slices", "cond_full_attn_slices",
                "cond_vae_video_slices", "cond_vit_video_slices", "cond_vit_video_context_slices",
                "image_slices", "image_full_attn_slices", "text_slices", "video_slices", "audio_slices",
            }:
                shifted = []
                for item, offset in zip(items, offsets):
                    for sli in item[key]:
                        shifted.append(slice(sli.start + offset, sli.stop + offset))
                new_item[key] = [shifted]   # noqa
            # Rope image info
            elif key in {"rope_image_info", "rope_media_info"}:
                shifted = []
                num_overlapped_tokens = 0
                for item, offset in zip(items, offsets):
                    # Here we need minus the number of overlap tokens from offset for interleave sequence.
                    offset -= num_overlapped_tokens
                    for sli, shape, meta in item[key]:
                        shifted.append((slice(sli.start + offset, sli.stop + offset), shape, meta))
                    # accumulate the number of overlap tokens
                    num_overlapped_tokens += item["num_overlapped"]
                new_item[key] = [shifted]   # noqa
            # Scatter index (gen / cond 已分 key，逐项 offset 后拼接即可)
            elif key in {
                "timesteps_index",
                "cond_timesteps_index",
                "video_timesteps_index",
                "audio_timesteps_index",
                "und_token_indices",
                "gen_token_indices",
                "audio_token_indices",
            }:
                shifted_indices = []
                for item, offset in zip(items, offsets):
                    shifted_indices.append(item[key] + offset)
                new_item[key] = torch.cat(shifted_indices)[None]  # noqa
            elif key in {'index'}:
                new_item[key] = [[item[key] for item in items]]  # noqa
            # ================ Sequence & no positional keys ================
            elif key in seq_pad_value:
                cat_list = [item[key] for item in items]
                pad_length = final_seq_length - sum(len(t) for t in cat_list)
                if self.args.pack_seq_reduce_pad:
                    is_megatron_cp = p_state.backend == "megatron" and cp_size > 1
                    needs_flex_attn_alignment = self.task_kwargs.get("attn_type") == "flex"
                    min_pad_length = self.reserved_pad_length if is_megatron_cp else 0

                    if is_megatron_cp:
                        assert pad_length >= self.reserved_pad_length, (
                            f"Megatron packing must leave at least reserved_pad_length positions, got "
                            f"{pad_length=}, reserved_pad_length={self.reserved_pad_length}."
                        )

                    alignment = cp_size if is_megatron_cp else 1
                    if needs_flex_attn_alignment:
                        alignment = math.lcm(alignment, 128)

                    # Reduce pad_length by the largest multiple of `alignment` while:
                    # 1. assuming the original model sequence is already aligned, preserving its
                    #    divisibility by `alignment`;
                    # 2. keeping pad_length greater than or equal to min_pad_length.
                    pad_length -= (
                        (pad_length - min_pad_length) // alignment
                    ) * alignment
                new_item[key] = torch.cat(  # noqa
                    cat_list + [torch.full((pad_length,), seq_pad_value[key], dtype=cat_list[0].dtype)]
                )[None]
            # ================ Not implemented keys ================
            elif key in {"num_overlapped", "status"}:
                pass
            elif key in {
                "attention_mask",
            }:
                raise NotImplementedError()
            else:
                raise ValueError(f"Unsupported key: {key}")

        # MoT and Multi-stream-DiT have some different postprocess logic for packed indices.
        new_item = self.postprocess_packed_indices(new_item, new_item["tokens"].shape[1], items)

        return new_item

    def postprocess_packed_indices(self, new_item, max_length, items):
        # When cp_size > 1: align und/gen for context parallel (last token in und, both divisible by cp_size after slice+dummy).
        # Pad reserve is done at sequence pack time so we always have enough pad to align.
        if "und_token_indices" in new_item and "gen_token_indices" in new_item:
            # MoT indices must be integer positions; casting pad ranges to dtype_und/dtype_gen can
            # corrupt values if those dtypes are floating-point (then valid_local_token_mask[index] OOBs).
            new_item["und_token_indices"] = new_item["und_token_indices"].long()
            new_item["gen_token_indices"] = new_item["gen_token_indices"].long()
            und_count = new_item["und_token_indices"].shape[1]
            gen_count = new_item["gen_token_indices"].shape[1]
            token_indices_length = und_count + gen_count
            pad_length = max_length - token_indices_length
            p_state = get_parallel_state()
            cp_size = p_state.cp_size
            device = new_item["und_token_indices"].device

            if self.args.pack_seq_reduce_pad and (p_state.backend != "megatron" or cp_size <= 1):
                if self.task_kwargs.get("attn_type") == "flex":
                    assert 0 <= pad_length < 128
                else:
                    assert pad_length == 0

            if pad_length > 0:
                padded_indices = torch.arange(
                    token_indices_length,
                    token_indices_length + pad_length,
                    device=device,
                    dtype=torch.long,
                )[None]

                # Megatron requires each branch length to be divisible by cp_size.
                # Pure Torch supports uneven CP shards when reducing padding, while
                # the no-reduction path keeps its existing branch-wise padding behavior.
                if cp_size > 1 and (p_state.backend == "megatron" or not self.args.pack_seq_reduce_pad):
                    first = items[0]
                    d0 = first.get("dummy_type_dict") or {}
                    und_dummy = d0.get("vit", 0)
                    gen_dummy = d0.get("ts_proj", 0)
                    if self.args.und_pad_min:
                        # `-1` for next token prediction's shift.
                        und_pad = (cp_size - (und_count + und_dummy - 1) % cp_size) % cp_size
                    else:
                        gen_pad = (cp_size - (gen_count + gen_dummy) % cp_size) % cp_size
                        und_pad = pad_length - gen_pad
                    # und pad must >= 1
                    if und_pad <= 0:
                        und_pad += cp_size

                    und_from_pad = torch.cat(
                        [padded_indices[:, : und_pad - 1], padded_indices[:, -1:]],
                        dim=1
                    )
                    gen_from_pad = padded_indices[:, und_pad - 1 : pad_length - 1]
                    new_item["und_token_indices"] = torch.cat(
                        [new_item["und_token_indices"], und_from_pad], dim=1
                    )
                    new_item["gen_token_indices"] = torch.cat(
                        [new_item["gen_token_indices"], gen_from_pad], dim=1
                    )
                    und_final = new_item["und_token_indices"].shape[1]
                    gen_final = new_item["gen_token_indices"].shape[1]
                    assert (und_final - 1 + und_dummy) % cp_size == 0 and (gen_final + gen_dummy) % cp_size == 0, (
                        f"After data_provider slice and add_dummy: (und_final-1+und_dummy) and (gen_final+gen_dummy) "
                        f"must be divisible by cp_size={cp_size}, got und_final={und_final}, gen_final={gen_final}, "
                        f"und_dummy={und_dummy}, gen_dummy={gen_dummy}"
                    )
                else:
                    if "pad" in self.und_token_type:
                        new_item["und_token_indices"] = torch.cat(
                            [new_item["und_token_indices"], padded_indices], dim=1
                        )
                    elif "pad" in self.gen_token_type:
                        new_item["gen_token_indices"] = torch.cat(
                            [new_item["gen_token_indices"], padded_indices], dim=1
                        )

        # Get valid_local_token_mask for MoE (skip padding in router).
        if self.args.moe_router_correct_valid_tokens:
            tokens_tensor = new_item["tokens"]  # (1, total_len)
            # valid = non-padding: positions where token is not pad_token_id
            valid_local_token_mask = (tokens_tensor != self.tokenizer.pad_token_id).squeeze(0).to(device=tokens_tensor.device, dtype=torch.bool)
            valid_total_num_tokens = int(valid_local_token_mask.sum().item())
            new_item['valid_local_token_mask'] = valid_local_token_mask

            # For MoT: build per-branch packed params so MoE router gets correct valid_local_token_mask length.
            if "und_token_indices" in new_item and "gen_token_indices" in new_item:
                seq_len = valid_local_token_mask.shape[0]
                und_indices = new_item["und_token_indices"].squeeze(0).long()
                gen_indices = new_item["gen_token_indices"].squeeze(0).long()
                if und_indices.numel() > 0 and (
                    und_indices.max().item() >= seq_len or und_indices.min().item() < -seq_len
                ):
                    raise RuntimeError(
                        f"und_token_indices out of bounds for sequence length {seq_len}: "
                        f"min={und_indices.min().item()}, max={und_indices.max().item()}. "
                        f"Check MoT index dtype (expect int64) and sequence_pack offsets."
                    )
                if gen_indices.numel() > 0 and (
                    gen_indices.max().item() >= seq_len or gen_indices.min().item() < -seq_len
                ):
                    raise RuntimeError(
                        f"gen_token_indices out of bounds for sequence length {seq_len}: "
                        f"min={gen_indices.min().item()}, max={gen_indices.max().item()}. "
                        f"Check MoT index dtype (expect int64) and sequence_pack offsets."
                    )
                # Same non-padding criterion: valid_und/valid_gen = non-padding mask for und/gen segments (subset of valid_local_token_mask)
                valid_und = valid_local_token_mask[und_indices]
                valid_gen = valid_local_token_mask[gen_indices]
                new_item['und_valid_local_token_mask'] = valid_und
                new_item['gen_valid_local_token_mask'] = valid_gen
            else:
                new_item["und_packed_seq_params"] = None
                new_item["gen_packed_seq_params"] = None
        else:
            new_item["packed_seq_params"] = None
            new_item["und_packed_seq_params"] = None
            new_item["gen_packed_seq_params"] = None

        return new_item

    def collate_fn(self, batch):
        if self.sequence_pack:
            return batch

        if self.drop_resampled_samples:
            filtered_batch = [item for item in batch if item["status"] == "original"]
            if len(filtered_batch) == 0:
                # If all samples in the batch are resampled, we keep the first sample to avoid empty batch.
                filtered_batch = [batch[0]]
            batch = filtered_batch

        if "cond_vit_image_kwargs" in batch[0]:
            cond_vit_image_kwargs = {
                "spatial_shapes": maybe_stack(batch, key="cond_vit_image_kwargs", subkey="spatial_shapes"),
                "attention_mask": maybe_stack(batch, key="cond_vit_image_kwargs", subkey="attention_mask"),
            }
        else:
            cond_vit_image_kwargs = None
        
        # TODO(ckczzjzhang): support different dummy_type_dict in a batch
        assert all(item["dummy_type_dict"] == batch[0]["dummy_type_dict"] for item in batch), "Different dummy_type_dict in a batch has not been supported yet."

        # ==== optional fields ====
        # We denote () as tensor shape, [] as list, n0, n1 as number of images, cond images(vae/vit), respectively.
        # - mask will be: stacked tensor(bsz, seqlen)
        # - slices will be: list of lists of slices[bsz, n, slice]
        # - images will be: a 5-D tensor(bsz, n0, 3, H, W) or list of 4-D tensors[bsz, (n0, 3, H, W)]
        #   or list of lists of 3-D tensors[bsz, [n0, (3, H, W)]]
        # - vae images will be: a 5-D tensor(bsz, n1, 3, H, W) or list of 4-D tensors[bsz, (n1, 3, H, W)]
        #   or list of lists of 3-D tensors[bsz, [n1, (3, H, W)]]
        # - vit images will be: a 4-D tensor(bsz, n, vit_seqlen, ndim) or list of 3-D tensors[bsz, (n, vit_seqlen, ndim)]
        # - timesteps_index will be: a 2-D tensor(bsz, n0) or list of 1-D tensors [bsz, (n0)]
        # - cond_timesteps_index will be: a 2-D tensor(bsz, n1) or list of 1-D tensors [bsz, (n1)]
        ret = {
            # === required ===
            "dataset_tag": [item["dataset_tag"] for item in batch],
            "n_samples": torch.tensor([item["n_samples"] for item in batch]),         # (bsz),
            "index": [item["index"] for item in batch],                               # (bsz)
            "tokens": torch.stack([item["tokens"] for item in batch]),                # (bsz, seqlen)
            "target_tokens": torch.stack([item["target_tokens"] for item in batch]),  # (bsz, seqlen)
            "text_mask": torch.stack([item["text_mask"] for item in batch]),          # (bsz, seqlen)
            # === optional ===
            "attention_mask": maybe_stack(batch, key="attention_mask", strict=True),
            # -- gen image
            "images": maybe_stack(batch, key="images"),
            "image_mask": maybe_stack(batch, "image_mask"),
            "image_slices": maybe_stack(batch, key="image_slices"),
            "image_full_attn_slices": maybe_stack(batch, key="image_full_attn_slices"),
            # -- cond image
            "cond_full_attn_slices": maybe_stack(batch, key="cond_full_attn_slices"),
            "cond_vae_images": maybe_stack(batch, key="cond_vae_images"),
            "cond_vae_image_mask": maybe_stack(batch, "cond_vae_image_mask"),
            "cond_vae_image_slices": maybe_stack(batch, key="cond_vae_image_slices"),
            "cond_vit_images": maybe_stack(batch, key="cond_vit_images"),
            "cond_vit_image_mask": maybe_stack(batch, "cond_vit_image_mask"),
            "cond_vit_image_slices": maybe_stack(batch, "cond_vit_image_slices"),
            "cond_vit_image_kwargs": cond_vit_image_kwargs,
            # -- positional
            "timesteps_index": maybe_stack(batch, key="timesteps_index"),
            "cond_timesteps_index": maybe_stack(batch, key="cond_timesteps_index"),
            "rope_image_info": maybe_stack(batch, key="rope_image_info"),
            # -- dummy
            "dummy_type_dict": [item["dummy_type_dict"] for item in batch],
        }
        if self.args.use_mot:
            ret.update(dict(
                und_token_indices=torch.stack([item["und_token_indices"] for item in batch]),
                gen_token_indices=torch.stack([item["gen_token_indices"] for item in batch]),
            ))

        ret = {key: value for key, value in ret.items() if value is not None}

        return ret


def load_dataloader(args, mm_state: MultimodalTasksState, p_state: ParallelState, logger=None,
                    task_info_dict=None):
    import fnmatch

    from ..utils.import_utils import require_version, is_index_kits_version
    from ..utils.torch_utils import set_worker_seed_builder

    sampler_kwargs = dict(
        shuffle=False,      # shuffle will be handled by CombinedIterator
        seed=args.seed,     # not used
        drop_last=True,
    )
    if args.use_numpy_indices:
        require_version("index-kits", "0.5.11", "DistributedSampler with `use_numpy_indices` enabled")
        sampler_kwargs.update({"use_numpy_indices": True})
    if is_index_kits_version(">=", "0.5.14"):
        sampler_kwargs.update({"verbose": 1, "info_repr": f"{p_state}"})
    if args.resume_index_batch_sampler:
        require_version("index-kits", "1.1.0", "Resume IndexBatchSampler")
    if args.distributed_sampler_low_cpu_memory:
        require_version("index-kits", "1.2.2", "DistributedSampler with `low_cpu_memory` enabled")
        # Only worked when drop_last is True and batch_size is 1.
        sampler_kwargs.update({"low_cpu_memory": args.distributed_sampler_low_cpu_memory})

    def find_matched_item(_dataset_tag, _task_info_dict):
        item = None
        for key_pattern, item_ in _task_info_dict.items():
            if fnmatch.fnmatch(_dataset_tag, key_pattern):
                item = item_
                break
        return item

    # ds_state: Dataset State contains distributed dataset sampling strategy. For `fixed` sampling mode,
    # each rank will be assigned a fixed one of all the datasets (determined by ds_state.cur_key). Other
    # datasets will not be loaded.
    ds_state = mm_state.distributed_sampling_state
    datasets = {}
    samplers = {}
    dataloaders = {}

    # Task info dictionary with wildcard support. Notice that only the tasks with the same
    # `cur_task` (for determining dummy type) and the same class can share the same entry.
    if task_info_dict is None:
        task_info_dict = {
            "t2i*": dict(cur_task="t2i", cls=MultimodalIndexDataset),
            "lm*": dict(cur_task="lm", cls=MultimodalIndexDataset),
            "mmu*": dict(cur_task="mmu", cls=MultimodalIndexDataset),
            "interleave*": dict(cur_task="interleave", cls=MultimodalIndexDataset),
            "pair*": dict(cur_task="interleave", cls=MultimodalIndexDataset),
            "t2v*": dict(cur_task="t2v", cls="av_loader.MultimodalAVIndexDataset"),
            "i2v*": dict(cur_task="i2v", cls="av_loader.MultimodalAVIndexDataset"),
            "r2v*": dict(cur_task="r2v", cls="av_loader.MultimodalAVIndexDataset"),
            "fl2v*": dict(cur_task="fl2v", cls="av_loader.MultimodalAVIndexDataset"),
            "t2va*": dict(cur_task="t2va", cls="av_loader.MultimodalAVIndexDataset"),
            "t2a*": dict(cur_task="t2a", cls="av_loader.MultimodalAVIndexDataset"),
        }

    def get_dataset_cls(cls):
        if isinstance(cls, str):
            module_name, dataset_cls = cls.rsplit(".", 1)
            module = importlib.import_module(f"hymm.data_kits.{module_name}")
            return getattr(module, dataset_cls)
        else:
            return cls

    for dataset_tag in mm_state.all_dataset_keys:
        item = find_matched_item(dataset_tag, task_info_dict)
        if item is None:
            continue
        if ds_state.cur_key is not None and dataset_tag != ds_state.cur_key:
            continue
        if dataset_tag in datasets:
            continue

        task_kwargs = getattr(args, f'{dataset_tag}_task_kwargs')
        index_kwargs = getattr(args, f'{dataset_tag}_index_kwargs')

        # Define per-dataset num_workers and prefetch_factor
        num_workers = task_kwargs.get("num_workers", args.num_workers)
        prefetch_factor = task_kwargs.get("prefetch_factor", None if args.num_workers == 0 else args.prefetch_factor)
        prefetch_kwargs = dict(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        dataloader_kwargs = dict(
            **prefetch_kwargs,
            pin_memory=True,
            worker_init_fn=set_worker_seed_builder(p_state.dp_rank),
        )
        if dataloader_kwargs.get("num_workers", 0) > 0:
            dataloader_kwargs["persistent_workers"] = True

        micro_batch_size = task_kwargs["batch_size"]
        # dataset class. Config's cls has the highest priority, then task_info_dict.
        dataset_cls = task_kwargs.get("cls", item['cls'])

        datasets[dataset_tag] = get_dataset_cls(dataset_cls)(
            args=args,
            dataset_tag=dataset_tag,
            task_kwargs=task_kwargs,
            index_kwargs=index_kwargs,
            logger=logger,
        )
        if getattr(args, "deterministic_dataloader", False):
            # Per-sample RNG determinism wrapper
            from .seeded_dataset import DeterministicSeededDatasetWrapper
            import multiprocessing as mp
            datasets[dataset_tag] = DeterministicSeededDatasetWrapper(
                dataset=datasets[dataset_tag],
                base_seed=args.seed,
                dp_rank=p_state.dp_rank,
                epoch_value=mp.Value('i', 0),
                task_tag=dataset_tag,
            )
        # Build sampler and data loader
        samplers[dataset_tag] = DistributedSampler(
            datasets[dataset_tag],
            num_replicas=ds_state.dataset_num_replicas[dataset_tag],
            rank=ds_state.dataset_rank[dataset_tag],
            batch_size=micro_batch_size if index_kwargs.get("multireso") else 1,
            **sampler_kwargs,
        )
        # For offline multi-reso dataset (i.e., the dataset itself provides multi-reso samples in one batch),
        # using DistributedSampler with batch_size > 1 is fine.
        # For online multi-reso dataset (i.e., the dataset generates multi-reso samples on-the-fly), we have to
        # use batch_size = 1 in DistributedSampler, and let the IndexBatchSampler handle the online bucketing and
        # micro-batching.
        if index_kwargs.get("online_bucketing"):
            batch_sampler = IndexBatchSampler(
                index_manager=datasets[dataset_tag].index_manager,
                sampler=samplers[dataset_tag],
                batch_size=micro_batch_size,
                drop_last=True,
                multireso=index_kwargs.get("reso_bucket_kwargs") is not None,
                multidar=index_kwargs.get("dar_bucket_kwargs") is not None,
                **prefetch_kwargs,
                history_buffer_size=64,     # Save the history 64 batches of indices when checkpointing for debugging.
            )
            dataloaders[dataset_tag] = DataLoader(
                datasets[dataset_tag],
                batch_sampler=batch_sampler,
                collate_fn=getattr(datasets[dataset_tag], "collate_fn", None),
                **dataloader_kwargs,
            )
        else:
            dataloaders[dataset_tag] = DataLoader(
                datasets[dataset_tag],
                batch_size=micro_batch_size,
                sampler=samplers[dataset_tag],
                collate_fn=getattr(datasets[dataset_tag], "collate_fn", None),
                shuffle=False,
                drop_last=True,
                **dataloader_kwargs,
            )

    if len(datasets) == 0:
        raise ValueError(f"dataset {mm_state.all_dataset_keys} not implemented "
                         f"or not registered into task_info_dict yet.")

    return mm_state, datasets, samplers, dataloaders
