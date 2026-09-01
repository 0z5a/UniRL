import json
import random
import re
import time
import warnings
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
from urllib.parse import unquote

import math
import loguru
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from index_kits import ArrowIndexV2
from torchvision import transforms
from tqdm import tqdm
from transformers import TextStreamer

try:
    from torch.nn.attention.flex_attention import create_block_mask
except:
    pass

from hymm.config import parse_eval_initial_args
from hymm.data_kits.csv_dataset import MMUDataset, SimpleDataset
from hymm.data_kits.arrow_dataset import ArrowDataset
from hymm.data_kits.image_dataset import ImageDataset
from hymm.data_kits.instruction_template import (
    text2image_instructions,
    inpainting_instructions,
    editing_instructions,
    subject_driven_instructions,
    face_id_instructions,
)
from hymm.data_kits.system_prompt import t2i_system_prompts, unified_system_prompts
from hymm.models.autoregressive.flex_attn_layers import create_batch_text_image_mask_mod
from hymm.models.autoregressive.custom_cache import HunyuanGeminiStaticCache
from hymm.models.tokenizers.conversation import get_conversation_template
from hymm.parallelism.parallel_states import get_parallel_state
from hymm.samplers.base_sampler import setup_distributed_initialize, setup_ptm_initialize
from hymm.samplers.logits_processor import get_logits_processors
from hymm.samplers.multimodal_transfusion_sampler import \
    MultimodalTransfusionSampler as MultimodalTransfusionSamplerBase
from hymm.utils.eval_utils import batch_data_repr
from hymm.utils.file_utils import rank0_logger
from hymm.utils.helpers import default, count_zh_en_words
from hymm.utils.resolution import ResolutionGroup
from hymm.utils.torch_utils import except_collate_fn
from hymm.utils.torch_utils import set_manual_seed
from hymm.utils.image_base import ImageInfo, JointImageInfo
from processors.image_kits import read_image, image_to_base64, unpad_image

warnings.filterwarnings("ignore", category=FutureWarning, module="transformer_engine")

# Define a dummy placeholder for loading the PTM ckpt.
build_pretraining_data_loader = None


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    else:
        return data


class GeminiBetaSampler(MultimodalTransfusionSamplerBase):
    def __post_init__(self):
        args = self.args
        ImageInfo.args = args

        self.base_size = args.training_image_size

        if args.add_image_shape_token:
            self.reso_group = ResolutionGroup(base_size=args.training_image_size, extra_resolutions=args.get('extra_resolutions', None))

        # Restrict the predicted image ratio to a stable range.
        self.shape_token_offset = self.tkwrapper.special_token_map["<img_ratio_0>"]
        self.shape_token_start = self.tkwrapper.special_token_map["<img_ratio_0>"]
        self.shape_token_end = self.tkwrapper.special_token_map["<img_ratio_32>"] + 1

        self.other_slices = [
            (self.tkwrapper.special_token_map["<img_ratio_33>"], self.tkwrapper.special_token_map["<img_ratio_36>"] + 1),
        ]

        # image getter
        self.cos_base = self.args.get('cos_base', None)
        self.cos_base_sources, self.cos_base_targets = ImageDataset.parse_cos_base(self.cos_base)

    def apply_lm_template(self, prompt_list, max_length, sequence_template="pretrain", batch_system_prompt=None,
                          return_attention_mask=False, attn_type="auto"):
        bsz = len(prompt_list)

        def _build_and_encode(prompt, system_prompt=None):
            if sequence_template == "instruct":
                conv = get_conversation_template(self.conv_format)
                conv.system_message = "" if system_prompt is None else system_prompt
                conv.add_message(conv.roles[0], prompt)
                conv.add_message(conv.roles[1], "")
                prompt = conv.get_prompt(add_system=system_prompt is not None)

            template = 'text'
            sections = [
                dict(type="text", text=prompt),
            ]
            output = self.tkwrapper.encode_general(
                template=template,
                sections=sections,
                max_token_length=max_length,
                use_text_mask=False,
                add_eos=False,
                add_pad=False,
            )
            return output, sections

        output, sections = self.tkwrapper.batch_gen_infer(
            infer_fn=_build_and_encode,
            prompt_list=prompt_list,
            infer_fn_kwargs_list=[
                dict(system_prompt=system_prompt)
                for system_prompt in batch_system_prompt
            ],
        )
        out = dict(output=output, sections=sections)

        if return_attention_mask:
            attention_mask = self.batch_attention_mask(output, batch_size=bsz, attn_type=attn_type)
            out['attention_mask'] = attention_mask

        return out

    def apply_mmu_template(
            self,
            prompt_list,
            image_token_length=None,
            image_hw=None,
            max_length=None,
            batch_und_image_info_list=None,
            batch_joint_image_info_list=None,
            return_attention_mask=False,
            attn_type="auto",
    ):
        bsz = len(prompt_list)
        use_joint_image_feature = self.args.get("use_joint_image_feature", False)

        if not use_joint_image_feature:
            text_max_length = max_length - image_token_length - 5   # 5 for (bos, und_boi, und_eoi, iw, ih)
            h, w = image_hw
            base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(w, h)
            und_kwargs = dict(
                add_iw_ih_token=self.args.add_iw_ih_token, use_front_boi_token=self.args.use_front_boi_token,
                add_image_shape_token=self.args.add_image_shape_token, base_size=base_size, ratio_idx=ratio_idx,
            )

        if batch_joint_image_info_list is None:
            batch_joint_image_info_list = [None for _ in range(bsz)]
        if batch_und_image_info_list is None:
            batch_und_image_info_list = [None for _ in range(bsz)]

        def _build_and_encode(prompt, joint_image_info_list=None, und_image_info_list=None):
            if joint_image_info_list is not None:
                _image_token_length = sum([
                    joint_image_info.image_token_length + joint_image_info.num_special_tokens
                    for joint_image_info in joint_image_info_list
                ])
                text_max_length_i = max_length - _image_token_length - 1   # 1 for bos
            else:
                text_max_length_i = text_max_length
            question_section = [
                dict(type="text", text=prompt, max_length=text_max_length_i),
            ]
            if use_joint_image_feature:
                cond_section = [
                    dict(type="joint_image", **joint_image_info.meta_info)
                    for joint_image_info in joint_image_info_list
                ]
            else:
                cond_section = [
                    dict(type="und_image", token_length=image_token_length, **und_kwargs),
                ]
            if self.args.use_front_src_image:
                sections = cond_section + question_section
            else:
                sections = question_section + cond_section
            output = self.tkwrapper.encode_general(
                sections=sections,
                max_token_length=max_length,
                use_text_mask=False,
                add_eos=False,
                add_pad=False,
            )
            return output, sections

        output, sections = self.tkwrapper.batch_gen_infer(
            infer_fn=_build_and_encode,
            prompt_list=prompt_list,
            infer_fn_kwargs_list=[
                dict(
                    joint_image_info_list=joint,
                    und_image_info_list=und,
                )
                for joint, und in zip(batch_joint_image_info_list, batch_und_image_info_list)
            ]
        )
        out = dict(output=output, sections=sections)

        if return_attention_mask:
            attention_mask = self.batch_attention_mask(output, batch_size=bsz, attn_type=attn_type)
            out['attention_mask'] = attention_mask

        return out

    def get_cot_sections(
            self,
            cot_text,
            uncond_kwargs,
            cot_tokens=None,
            cot_max_length=None,
            drop_think=False,
            ignore_kwargs={},
    ):
        if not cot_text:  # None or empty
            return []
        if '<think>' in cot_text and '</think>' in cot_text:
            before_think_sec = cot_text.split('<think>')[0]
            after_think_sec = cot_text.split('</think>')[1]
            think_sec = cot_text.split('<think>')[1].split('</think>')[0]
            before_think_tokens, think_tokens, after_think_tokens = None, None, None

            if cot_tokens is not None:
                think_token_id = self.tkwrapper.think_token_id
                end_of_think_token_id = self.tkwrapper.end_of_think_token_id
                start_idx = cot_tokens.index(think_token_id)
                search_start = start_idx + 1
                end_idx_relative = cot_tokens[search_start:].index(end_of_think_token_id)
                end_idx = search_start + end_idx_relative
                before_think_tokens = cot_tokens[:start_idx]
                think_tokens = cot_tokens[start_idx + 1 : end_idx]
                after_think_tokens = cot_tokens[end_idx + 1:]

            return self.get_cot_sections(before_think_sec, uncond_kwargs, cot_tokens=before_think_tokens, drop_think=drop_think, ignore_kwargs=ignore_kwargs) + \
                ([
                    dict(type="text", text="<think>", ignore=ignore_kwargs.get("ignore_general_text", False)),
                    dict(type="text", tokens=think_tokens, max_length=cot_max_length, ignore=ignore_kwargs.get("ignore_think_except_think_start", False), **uncond_kwargs) if think_tokens is not None else dict(type="text", text=think_sec, max_length=cot_max_length, ignore=ignore_kwargs.get("ignore_think_except_think_start", False), **uncond_kwargs),
                    dict(type="text", text="</think>", ignore=ignore_kwargs.get("ignore_think_except_think_start", False))
                ] if not drop_think else []) + \
                self.get_cot_sections(after_think_sec, uncond_kwargs, cot_tokens=after_think_tokens, drop_think=drop_think, ignore_kwargs=ignore_kwargs)

        if '<recaption>' in cot_text and '</recaption>' in cot_text:
            before_recaption_sec = cot_text.split('<recaption>')[0]
            after_recaption_sec = cot_text.split('</recaption>')[1]
            recaption_sec = cot_text.split('<recaption>')[1].split('</recaption>')[0]
            before_recaption_tokens, recaption_tokens, after_recaption_tokens = None, None, None
            if cot_tokens is not None:
                recaption_token_id = self.tkwrapper.recaption_token_id
                end_of_recaption_token_id = self.tkwrapper.end_of_recaption_token_id
                start_idx = cot_tokens.index(recaption_token_id)
                search_start = start_idx + 1
                end_idx_relative = cot_tokens[search_start:].index(end_of_recaption_token_id)
                end_idx = search_start + end_idx_relative
                before_recaption_tokens = cot_tokens[:start_idx]
                recaption_tokens = cot_tokens[start_idx + 1 : end_idx]
                after_recaption_tokens = cot_tokens[end_idx + 1:]

            return self.get_cot_sections(before_recaption_sec, uncond_kwargs, cot_tokens=before_recaption_tokens, drop_think=drop_think, ignore_kwargs=ignore_kwargs) + \
                [
                    dict(type="text", text="<recaption>", ignore=ignore_kwargs.get("ignore_general_text", False)),
                    dict(type="text", tokens=recaption_tokens, max_length=cot_max_length, ignore=ignore_kwargs.get("ignore_recaption_except_recaption_start", False), **uncond_kwargs) if recaption_tokens is not None else dict(type="text", text=recaption_sec, max_length=cot_max_length,  ignore=ignore_kwargs.get("ignore_recaption_except_recaption_start", False), **uncond_kwargs),
                    dict(type="text", text="</recaption>", ignore=ignore_kwargs.get("ignore_recaption_except_recaption_start", False))
                ] + \
                self.get_cot_sections(after_recaption_sec, uncond_kwargs, cot_tokens=after_recaption_tokens, drop_think=drop_think, ignore_kwargs=ignore_kwargs)

        return [
            dict(type="text", text=cot_text, ignore=ignore_kwargs.get("ignore_general_text", False), **uncond_kwargs),
        ] if cot_tokens is None else [
            dict(type="text", tokens=cot_tokens, ignore=ignore_kwargs.get("ignore_general_text", False), **uncond_kwargs),
        ]

    def apply_x2image_template(
            self,
            batch_prompt_list, # list of list of prompts
            sequence_template,
            batch_target_image_info,
            batch_system_prompt=None,
            batch_cot_text=None,
            batch_cot_tokens=None,
            batch_und_image_info_list=None,
            batch_src_image_info_list=None,
            batch_joint_image_info_list=None,
            batch_face_image_info_list=None,
            batch_prefill_template=None,
            batch_negative_prompt_list=None,
            answer=True,
            cfg_factor=1,
            return_attention_mask=True,
            attn_type="auto",
            k_seq_len=None,
            grpo_gen_text_mask_type=None,
            sync_max_length_across_ranks=False,
    ) -> Dict[str, Any]:
        batch_size = len(batch_prompt_list)
        max_length = -1

        if batch_prefill_template is None:
            batch_prefill_template = [None for _ in range(batch_size)]

        for prompt_list, system_prompt, cot_text, target_image_info, und_image_info_list, src_image_info_list, joint_image_info_list, face_image_info_list in zip(batch_prompt_list, batch_system_prompt, batch_cot_text, batch_target_image_info, batch_und_image_info_list, batch_src_image_info_list, batch_joint_image_info_list, batch_face_image_info_list):
            # 5: <boi> <eoi> <timestep> (<iw> <ih> / <img_ratio>, <img_size>)
            if sequence_template == "instruct":
                # 11: <bos> 1, "User: " 3, "\n\n" 1, "Assistant: <answer>" 4, </answer> 1, <eos> 1
                # 处理mixed cot模式 (think + recaption) 下的max_length计算
                
                num_cot_sections = 0
                if cot_text:
                    if '<think>' in cot_text and '</think>' in cot_text:
                        num_cot_sections += 1
                    if '<recaption>' in cot_text and '</recaption>' in cot_text:
                        num_cot_sections += 1

                max_length = max(max_length, (
                    (self.args.system_prompt_token_length if system_prompt is not None else 0) +
                    self.args.text_token_length * len(prompt_list) +
                    (self.args.cot_max_length * num_cot_sections) +
                    sum([item.image_token_length for item in und_image_info_list]) +
                    len(und_image_info_list) * 4 +
                    sum([item.image_token_length for item in src_image_info_list]) +
                    len(src_image_info_list) * 5 +
                    sum([item.image_token_length for item in joint_image_info_list]) +
                    len(joint_image_info_list) * 6 + # <joint_img_sep>
                    sum([item.image_token_length for item in face_image_info_list]) +
                    len(face_image_info_list) * 2 + # <bof> <eof>
                    target_image_info.image_token_length +
                    1 * 5 +
                    11
                ))
                conv = get_conversation_template(self.conv_format)
            elif sequence_template == "pretrain":
                # 2: <bos> <eos>
                max_length = max(max_length, (
                    self.args.text_token_length * len(prompt_list) +
                    (self.args.cot_max_length if cot_text is not None else 0) +
                    sum([item.image_token_length for item in und_image_info_list]) +
                    len(und_image_info_list) * 4 + # <und_boi> <und_eoi> <img_size_*> <img_ratio_*>
                    sum([item.image_token_length for item in src_image_info_list]) +
                    len(src_image_info_list) * 5 +
                    sum([item.image_token_length for item in joint_image_info_list]) +
                    len(joint_image_info_list) * 6 + # <joint_img_sep>
                    sum([item.image_token_length for item in face_image_info_list]) +
                    len(face_image_info_list) * 2 + # <bof> <eof>
                    target_image_info.image_token_length +
                    1 * 5 +
                    2
                ))
            else:
                raise ValueError(f"Unknown sequence template: {sequence_template}")
        
        if sync_max_length_across_ranks:
            max_length_tensor = torch.tensor([max_length]).cuda()
            torch.distributed.all_reduce(max_length_tensor, 
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=torch.distributed.group.WORLD)
            max_length = max_length_tensor.item()

        def _build_and_encode(
                *prompt_list, system_prompt, cot_text, cot_tokens, target_image_info, und_image_info_list, src_image_info_list,
                joint_image_info_list, face_image_info_list, prefill_template=None, uncond_p=0.0,
        ):
            prompt_list = list(prompt_list)
            sections = []
            if prefill_template is not None:
                prompt_index = 0
                und_image_index = 0
                src_image_index = 0
                joint_image_index = 0
                # TODO add face_image_index and section append
                for template_part in prefill_template.split("-"):
                    if template_part == "text":
                        sections.append(
                            dict(type="text", text=prompt_list[prompt_index], max_length=self.args.text_token_length,
                                 uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
                        )
                        prompt_index += 1
                    elif template_part == "und_image":
                        sections.append(dict(type="und_image", **und_image_info_list[und_image_index].meta_info))
                        und_image_index += 1
                    elif template_part == "src_image":
                        sections.append(dict(type="src_image", **src_image_info_list[src_image_index].meta_info))
                        src_image_index += 1
                    elif template_part == "joint_image":
                        sections.append(dict(type="joint_image", **joint_image_info_list[joint_image_index].meta_info))
                        joint_image_index += 1
                    else:
                        raise ValueError(f"Unsupported template part '{template_part}' in prefill template")

                # post process
                prompt_list = prompt_list[prompt_index:]
                und_image_info_list = und_image_info_list[und_image_index:]
                src_image_info_list = src_image_info_list[src_image_index:]
                joint_image_info_list = joint_image_info_list[joint_image_index:]

            # ----- build template and sections -----
            # 1. pretrain and instruct use the same condition sections
            cond_section = [
                dict(type=image_info.image_type, **image_info.meta_info)
                for image_info in face_image_info_list + und_image_info_list + src_image_info_list + joint_image_info_list
            ]

            # 2. pretrain and instruct use different prompt and user/assistant prefix sections
            if sequence_template == "instruct":
                ignore_general_text = grpo_gen_text_mask_type is not None
                ignore_think_except_think_start = grpo_gen_text_mask_type == 'recaption'
                # In grpo training of cot mode, we always unmask tokens of the recaption content and </recaption> tokens
                ignore_recaption_except_recaption_start = False
                ignore_kwargs = dict(
                    ignore_general_text=ignore_general_text,
                    ignore_think_except_think_start=ignore_think_except_think_start,
                    ignore_recaption_except_recaption_start=ignore_recaption_except_recaption_start,
                )
                if system_prompt is not None:
                    system_prompt_section = [
                        dict(type="text", text=system_prompt, max_length=self.args.system_prompt_token_length - 1, ignore=ignore_general_text),
                        dict(type="text", text=f"{conv.sep}", ignore=ignore_general_text), # "\n\n" 1 token
                    ]
                else:
                    system_prompt_section = []
                
                user_prefix_section = [
                    dict(type="text", text=f"{conv.roles[0]}: ", ignore=ignore_general_text),
                ]

                prompt_and_bot_prefix_section = [
                    dict(type="text", text=prompt_list[0], max_length=self.args.text_token_length, ignore=ignore_general_text,
                         uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p) if len(prompt_list) > 0 else dict(type='text', text='', ignore=ignore_general_text),
                    dict(type="text", text=f"{conv.sep}{conv.roles[1]}: ", ignore=ignore_general_text),
                ]

                if cot_text is not None:
                    prompt_and_bot_prefix_section.extend(
                        self.get_cot_sections(
                            cot_text,
                            dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p),
                            cot_tokens=cot_tokens,
                            ignore_kwargs=ignore_kwargs
                        )
                    )

                if self.args.use_front_src_image:
                    sections += system_prompt_section + user_prefix_section + cond_section + prompt_and_bot_prefix_section
                else:
                    sections += system_prompt_section + user_prefix_section + prompt_and_bot_prefix_section + cond_section
                suffix = "<answer>" if answer else ""
            elif sequence_template == "pretrain":
                # TODO (yutaocui): Now only support `grpo_gen_text_mask_type` setting in `instrcut` mode,
                # since t2ti grpo training is performed after instruction tuning
                assert grpo_gen_text_mask_type is None, \
                    "Now only support `grpo_gen_text_mask_type` setting in `instrcut` mode, " \
                    "since t2ti grpo training is performed after instruction tuning"
                prompt_section = [
                    dict(type="text", text=prompt_list[0], max_length=self.args.text_token_length,
                         uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
                    if len(prompt_list) > 0 else dict(type='text', text='')
                ]
                if cot_text is not None:
                    if cot_tokens is not None:
                        prompt_section.append(
                            dict(type="text", tokens=cot_tokens, max_length=self.args.cot_max_length,
                                uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
                        )
                    else:
                        prompt_section.append(
                            dict(type="text", text=cot_text, max_length=self.args.cot_max_length,
                                uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)
                        )
                if self.args.use_front_src_image:
                    sections += cond_section + prompt_section
                else:
                    sections += prompt_section + cond_section
                suffix = ""
            else:
                raise ValueError(f"Unknown sequence template: {sequence_template}")

            # 3. pretrain and instruct use the same target sections
            if target_image_info.image_type == "gen_image":
                sections += [
                    dict(type="text", text=suffix),
                    dict(type="gen_image", **target_image_info.meta_info)
                ]
            elif target_image_info.image_type == "cot_think" or target_image_info.image_type == "mixed":
                sections += [
                    dict(type="text", text="<think>")
                ]
            elif target_image_info.image_type == "cot_recaption":
                sections += [
                    dict(type="text", text="<recaption>")
                ]
            elif target_image_info.image_type == "image_ratio":   # predict image shape tokens
                sections += [
                    dict(type="text", text=suffix),
                    dict(type="text", text=f'<boi><img_size_{self.args.training_image_size}>')
                ]
            else:
                raise ValueError(f"Unknown target image type: {target_image_info.image_type}")

            # ----- encode template and sections -----
            output = self.tkwrapper.encode_general(
                sections=sections,
                max_token_length=max_length,
                use_text_mask=grpo_gen_text_mask_type is not None,
                add_eos=False,
                add_pad=False,
            )
            return output, sections

        output, sections = self.tkwrapper.batch_gen_infer(
            infer_fn=_build_and_encode,
            infer_fn_kwargs_list=[
                dict(system_prompt=system_prompt,
                     cot_text=cot_text,
                     cot_tokens=cot_tokens,
                     target_image_info=target,
                     und_image_info_list=und,
                     src_image_info_list=src,
                     joint_image_info_list=joint,
                     face_image_info_list=face,
                     prefill_template=prefill_template)
                for system_prompt, cot_text, cot_tokens, target, und, src, joint, face, prefill_template in zip(
                    batch_system_prompt, batch_cot_text, batch_cot_tokens, batch_target_image_info, batch_und_image_info_list, batch_src_image_info_list,
                    batch_joint_image_info_list, batch_face_image_info_list, batch_prefill_template
                )
            ],
            prompt_list=batch_prompt_list,
            negative_prompt_list=batch_negative_prompt_list,
            do_classifier_free_guidance=cfg_factor > 1,
            condition_repeat_times=1,
            uncondition_repeat_times=cfg_factor - 1,
        )
        out = dict(output=output, sections=sections)

        if return_attention_mask:
            attention_mask = self.batch_attention_mask(
                output, batch_size=batch_size, attn_type=attn_type, cfg_factor=cfg_factor)
            out['attention_mask'] = attention_mask

        return out

    def apply_general_template(
            self,
            message_list,
            max_length=None,
            add_assistant_prefix=None,
            answer="auto",
            bot_task="auto",
            sequence_template="instruct",
            uncond_p=0.0,
            cfg_factor=1,
            batchify=False,
            image_base_size=1024,
            drop_think=False,
    ):
        """
        apply_general_template is only used for chatbot, and it doesn't support batch processing.
        """

        # If cfg_factor > 1, we need to repeat the unconditioned part
        if batchify:
            return self.tkwrapper.batch_gen_infer(
                infer_fn=self.apply_general_template,
                prompt_list=[[]],
                infer_fn_kwargs_list=[dict(
                    message_list=message_list_i,
                    max_length=max_length,
                    add_assistant_prefix=add_assistant_prefix,
                    answer=answer,
                    bot_task=bot_task,
                    sequence_template=sequence_template,
                    image_base_size=image_base_size,
                    drop_think=drop_think,
                ) for message_list_i in message_list],
                do_classifier_free_guidance=cfg_factor > 1,
                condition_repeat_times=1,
                uncondition_repeat_times=cfg_factor - 1,
            )

        conv = get_conversation_template(self.conv_format)
        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        def process_successive_message(_message_list, _cur_message_idx, role, prefix, suffix,
                                       answer_prefix="", answer_suffix="", final_role=None):
            _sub_sections = []
            while _cur_message_idx < len(message_list) and _message_list[_cur_message_idx]['role'] == role:
                message = _message_list[_cur_message_idx]
                if message['type'] == 'text':
                    text = message['content']
                    if role == "system":
                        _sub_sections.append(dict(type="text", text=text))
                    elif role == "assistant":
                        if ("<recaption>" in text and "</recaption>" in text) or ("<think>" in text and "</think>" in text):
                            _sub_sections.extend(self.get_cot_sections(text, uncond_kwargs, drop_think=drop_think))
                        else:
                            _sub_sections.append(dict(
                                type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs))
                    else:
                        _sub_sections.append(dict(type="text", text=text, uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p))
                elif message['type'] == "gen_image":
                    info = message['content']
                    assert isinstance(info, ImageInfo), f"Expected ImageInfo, but got {type(info)}"
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_prefix))
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                    if role == "assistant":
                        _sub_sections.append(dict(type="text", text=answer_suffix))
                elif message['type'] == 'joint_image':
                    info = message['content']
                    assert isinstance(info, JointImageInfo), f"Expected JointImageInfo, but got {type(info)}"
                    _sub_sections.append(dict(type=message['type'], **info.meta_info))
                elif message['type'] == 'face':
                    info: ImageInfo = message['content']
                    _sub_sections.append(dict(type=message['type'], token_length=info.image_token_length))
                else:
                    raise ValueError(f"Unknown message type: {message['type']}")
                _cur_message_idx += 1
            if len(_sub_sections) > 0:
                # Add role prefix and suffix if available
                if prefix:
                    _sub_sections.insert(0, dict(type='text', text=prefix))
                if suffix:
                    _sub_sections.append(dict(type='text', text=suffix))
                final_role = role
            return _sub_sections, _cur_message_idx, final_role

        # Define assistant prefix and suffix
        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""
        if sequence_template == "pretrain":
            system_suffix = ""
            user_prefix = ""
            user_suffix = ""
            bot_prefix = ""
            bot_suffix = ""
        else:
            system_suffix = f"{conv.sep}"
            user_prefix = f"{conv.roles[0]}: "
            user_suffix = f"{conv.sep}"
            bot_prefix = f"{conv.roles[1]}: "
            bot_suffix = f"{conv.sep}"

        # Process successive user and assistant messages
        sections = []
        cur_message_idx = 0
        final_role = None
        while cur_message_idx < len(message_list):
            # Process successive system messages
            sub_sections, cur_message_idx, final_role = process_successive_message(
                message_list, cur_message_idx, role="system", prefix="", suffix=system_suffix,
                final_role=final_role,
            )
            # Add to the template and sections
            sections.extend(sub_sections)

            # Process successive user messages
            sub_sections, cur_message_idx, final_role = process_successive_message(
                message_list, cur_message_idx, role="user", prefix=user_prefix, suffix=user_suffix,
                final_role=final_role,
            )
            # Add to the template and sections
            sections.extend(sub_sections)

            # Process successive assistant messages
            sub_sections, cur_message_idx, final_role = process_successive_message(
                message_list, cur_message_idx, role="assistant", prefix=bot_prefix, suffix=bot_suffix,
                answer_prefix=answer_prefix, answer_suffix=answer_suffix, final_role=final_role,
            )
            # Add to the template and sections
            sections.extend(sub_sections)

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
                auto=f"{_bot_prefix}{answer_prefix}",
                image="",
                think=f"{_bot_prefix}<think>",
                recaption=f"{_bot_prefix}<recaption>",
                img_ratio=f"{_bot_prefix}{answer_prefix}<boi><img_size_{image_base_size}>",
            )[bot_task]
            sections.append(dict(type='text', text=bot_response_prefix))

        output = self.tkwrapper.encode_general(
            sections=sections,
            max_token_length=max_length,
            use_text_mask=False,
            add_eos=False,
            add_pad=False,
        )

        return output, sections

    # batch_image_info_list can be batch_src_image_info_list or batch_joint_image_info_list
    def _encode_image(
            self,
            batch_image_info_list: List[List[Union[ImageInfo, JointImageInfo]]],
            cfg_factor=1
    ):
        input_src_x, input_src_t = None, None
        batch_input_src_x, batch_input_src_t = [], []
        batch_und_images = []
        if batch_image_info_list is not None:
            for image_info_list in batch_image_info_list:
                input_src_x_list, input_src_t_list, und_images_list = [], [], []
                for image_info in image_info_list:
                    # src_t_: 1  src_x_: 1 x c x h x w
                    if image_info.image_type == "joint_image":
                        src_t_, src_x_ = self.vae_encode(image_info.vae_image_info.image_tensor.to(self.device),
                                                         sample_type="sample_start")
                        und_images_list.append(image_info.vision_image_info.image_tensor)  # 1 x c x h x w or 1 x seq_len x dim
                    else:
                        src_t_, src_x_ = self.vae_encode(image_info.image_tensor.to(self.device),
                                                         sample_type="sample_start")
                    input_src_x_list.append(src_x_.squeeze(0))
                    input_src_t_list.append(src_t_)
                batch_input_src_x.append(input_src_x_list)
                batch_input_src_t.append(input_src_t_list)
                batch_und_images.append(torch.cat(und_images_list, dim=0) if und_images_list else None)

        if len(batch_input_src_x) > 0 and all([len(item) > 0 for item in batch_input_src_x]):
            if all([len(item) == 1 for item in batch_input_src_x]) and all(
                    item[0].shape == batch_input_src_x[0][0].shape for item in batch_input_src_x):
                # b x c x h x w
                input_src_x = torch.stack([item[0] for item in batch_input_src_x], dim=0)
                # b
                input_src_t = torch.cat([item[0] for item in batch_input_src_t], dim=0)
                if cfg_factor > 1:
                    input_src_t = input_src_t.repeat(cfg_factor)
                    input_src_x = input_src_x.repeat(cfg_factor, 1, 1, 1)
            else:
                input_src_t = [torch.cat(item, dim=0) for item in batch_input_src_t]
                input_src_x = []
                for item in batch_input_src_x:
                    try:
                        input_src_x.append(torch.stack(item, dim=0))
                    except:
                        input_src_x.append(item)
                if cfg_factor > 1:
                    input_src_t = input_src_t * cfg_factor
                    input_src_x = input_src_x * cfg_factor

        if cfg_factor > 1:
            batch_und_images = batch_und_images * cfg_factor
        if batch_und_images[0] is None:
            batch_und_images = None

        return input_src_x, input_src_t, batch_und_images

    @staticmethod
    def batch_face_cond_uncond(batch_face_image_info_list, cfg_factor=3, face_guidance_scale=6.0, guidance_scale=6.0):
        batch_face_image_tensor_list = []
        for face_image_info_list in batch_face_image_info_list:
            for face_image_info in face_image_info_list:
                batch_face_image_tensor_list.append(face_image_info["image_tensor"])

        if len(batch_face_image_tensor_list) > 0:
            # Concatenate tensor of shape (1, 512) to (N, 512)
            face_image_tensor = torch.cat(batch_face_image_tensor_list, dim=0)
            face_image_tensor_repeated = face_image_tensor.repeat(cfg_factor, 1)
            if face_guidance_scale > 1.0:
                face_uncond_index = 2 if guidance_scale > 1.0 else 1
                face_image_tensor_repeated[face_uncond_index] = torch.zeros_like(face_image_tensor)
        else:
            face_image_tensor_repeated = None
        return face_image_tensor_repeated

    @staticmethod
    def batch_image_cond_uncond(
            batch_image_info_list: List[List[Union[ImageInfo, JointImageInfo]]],
            cfg_factor=3, src_guidance_scale=6.0, guidance_scale=6.0
    ):
        batch_image_info_list = [
            [batch_image_info_list[0][i].copy() for i in range(len(batch_image_info_list[0]))]
            for _ in range(cfg_factor)
        ]
        if src_guidance_scale > 1.0:
            src_uncond_index = 2 if guidance_scale > 1.0 else 1
            for image_info in batch_image_info_list[src_uncond_index]:
                image_info.zeros_()
        return batch_image_info_list

    def batch_attention_mask(self, output, batch_size, attn_type="auto", cfg_factor=1, k_seq_len=None):
        bsz = batch_size * cfg_factor
        seq_len = output.tokens.shape[1]
        batch_image_slices = [
            output.und_image_slices[i] + output.src_image_slices[i] + output.joint_image_slices[i] + output.gen_image_slices[i] + output.face_image_slices[i]
            for i in range(bsz)
        ]
        if attn_type == "auto":
            k_seq_len = default(k_seq_len, seq_len)
            attention_mask = torch.ones(seq_len, k_seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
            for i in range(bsz):
                for j, image_slice in enumerate(batch_image_slices[i]):
                    attention_mask[i, image_slice, image_slice] = True
                    # align rope for interleave sampling
                    # if j%2==0 and j != len(batch_image_slices[i])-1:
                    #     attention_mask[i, image_slice.stop:, image_slice] = False
            attention_mask = attention_mask.unsqueeze(1)
        elif attn_type == "flex":
            mask_mod = create_batch_text_image_mask_mod(batch_image_slices, seq_len, self.device)
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        return attention_mask

    def cfg_factor(self, kwargs, allow_face=False, allow_src=False, cfg_distilled=False):
        # text classifier-free guidance
        if cfg_distilled:
            return 1
        guidance_scale = kwargs.get("guidance_scale", self.args.guidance_scale)
        cfg_factor = 1 + (guidance_scale > 1.0)
        # face classifier-free guidance
        use_face_id = kwargs.get('use_face_id', False)
        if allow_face or use_face_id:
            face_guidance_scale = kwargs.get("face_guidance_scale", self.args.face_guidance_scale)
            cfg_factor += (face_guidance_scale > 1.0)
        elif allow_src and (src_guidance_scale := kwargs.get("src_guidance_scale", self.args.src_guidance_scale)) is not None:
            # src_image classifier-free guidance
            cfg_factor += (src_guidance_scale > 1.0)

        return cfg_factor

    def x2image(
        self,
        batch_prompt_list: Optional[List[List[str]]],
        generator: Union[torch.Generator, List[torch.Generator]],
        sequence_template: str = "instruct",
        batch_system_prompt: Optional[List[str]] = None,
        batch_cot_text: Optional[List[List[str]]] = None,
        batch_cot_tokens: Optional[List[List[int]]] = None,
        message_list: Optional[List[Dict[str, Any]]] = None,
        batch_und_image_info_list: Optional[List[List[Union[Dict[str, Any], ImageInfo]]]] = None,
        batch_src_image_info_list: Optional[List[List[Union[Dict[str, Any], ImageInfo]]]] = None,
        batch_joint_image_info_list: Optional[List[List[Union[Dict[str, Any], ImageInfo]]]] = None,
        batch_face_image_info_list: Optional[List[List[Union[Dict[str, Any], ImageInfo]]]] = None,
        batch_target_image_info: Optional[List[Union[Dict[str, Any], ImageInfo]]] = None,
        batch_prefill_template: Optional[List[str]] = None,
        batch_negative_prompt_list: Optional[List[List[str]]] = None,
        model_input_custom_kwargs: Optional[Dict[str, Any]] = None,
        cfg_factor: Optional[int] = None,
        save_paths: Optional[List[str]] = None,
        extra_log_info: str = "",
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        pbar=None,
        return_only_samples=True,
        **kwargs,
    ):
        """
        Encode batch_src_image_info_list into input_src_x and input_src_t; Calculate src, tgt and und mask in apply_x2image_template; Run pipeline; Save samples.
        """
        args = self.args

        if not isinstance(generator, list):
            generator = [generator]

        meanflow = kwargs.get("meanflow", args.get("meanflow", False))
        guidance_scale = kwargs.get("guidance_scale", args.guidance_scale)
        cfg_distilled = kwargs.get("cfg_distilled", args.cfg_distilled)
        face_guidance_scale = kwargs.get("face_guidance_scale", args.face_guidance_scale)
        src_guidance_scale = kwargs.get("src_guidance_scale", args.src_guidance_scale)
        diff_infer_steps = kwargs.get("diff_infer_steps", args.diff_infer_steps)
        output_type = kwargs.get("output_type", "pil")
        verbose = kwargs.get("verbose", 1)
        drop_think = kwargs.get("drop_think", args.drop_think)
        grpo_gen_text_mask_type = kwargs.get('grpo_gen_text_mask_type', None)
        if grpo_gen_text_mask_type is not None:
            # -- generate text mask type: 'recaption', 'mixed'
            # ---- 'recaption': make only the tokens of {RECAPTION CONTENT} </recaption> to be unmasked, i.e., applying grpo logic
            # ---- 'mixed': make the tokens of both {THINK CONTENT} </think> and {RECAPTION CONTENT} </recaption> to be unmasked
            assert grpo_gen_text_mask_type in ['recaption', 'mixed'], \
                f"Invalid grpo_gen_text_mask_type: {grpo_gen_text_mask_type}"

        if batch_prompt_list is not None:
            batch_size = len(batch_prompt_list)

            if batch_system_prompt is None:
                batch_system_prompt = [None for _ in range(batch_size)]
            if batch_cot_text is None:
                batch_cot_text = [None for _ in range(batch_size)]

            if batch_und_image_info_list is None:
                batch_und_image_info_list = [[] for _ in range(batch_size)]
            if batch_src_image_info_list is None:
                batch_src_image_info_list = [[] for _ in range(batch_size)]
            if batch_joint_image_info_list is None:
                batch_joint_image_info_list = [[] for _ in range(batch_size)]
            if batch_face_image_info_list is None:
                batch_face_image_info_list = [[] for _ in range(batch_size)]

            cfg_factor = default(cfg_factor, self.cfg_factor(
                kwargs,
                allow_src=(len(batch_src_image_info_list[0]) > 0 or len(batch_joint_image_info_list[0]) > 0),
                allow_face=len(batch_face_image_info_list[0]) > 0,
                cfg_distilled=cfg_distilled,
            ))
            assert len(batch_prompt_list) == len(generator), \
                f"The number of prompts and generators should be the same, but got {len(batch_prompt_list)} and {len(generator)}."
            batch_size = len(batch_prompt_list)
            # Input one sample and output batched one
            out = self.apply_x2image_template(
                batch_prompt_list,
                sequence_template,
                batch_system_prompt=batch_system_prompt,
                batch_cot_text=batch_cot_text,
                batch_cot_tokens=batch_cot_tokens,
                batch_target_image_info=batch_target_image_info,
                batch_und_image_info_list=batch_und_image_info_list,
                batch_src_image_info_list=batch_src_image_info_list,
                batch_joint_image_info_list=batch_joint_image_info_list,
                batch_face_image_info_list=batch_face_image_info_list,
                batch_prefill_template=batch_prefill_template,
                batch_negative_prompt_list=batch_negative_prompt_list,
                answer=True,
                cfg_factor=cfg_factor,
                attn_type=args.infer_attn_type,
                grpo_gen_text_mask_type=grpo_gen_text_mask_type,
            )
            output, sections, attention_mask = out['output'], out['sections'], out['attention_mask']
        else:
            batch_size = 1
            batch_src_image_info_list = [
                [message['content'] for message in message_list if message['type'] == 'src_image']
            ]
            batch_joint_image_info_list = [
                [message['content'] for message in message_list if message['type'] == 'joint_image']
            ]
            batch_face_image_info_list = [
                [message['content'] for message in message_list if message['type'] == 'face']
            ]
            batch_target_image_info = [
                message['content'] for message in message_list if message['type'] == 'gen_image'
            ]
            cfg_factor = default(cfg_factor, self.cfg_factor(
                kwargs,
                allow_src=(len(batch_src_image_info_list[0]) > 0 or len(batch_joint_image_info_list[0]) > 0),
                allow_face=len(batch_face_image_info_list[0]) > 0,
                cfg_distilled=cfg_distilled,
            ))

            output, sections = self.apply_general_template(
                [message_list],
                max_length=kwargs.get('block_size', args.block_size),
                add_assistant_prefix=False,
                cfg_factor=cfg_factor,
                bot_task="image",
                sequence_template=sequence_template,
                batchify=True,
                image_base_size=self.base_size,
                drop_think=drop_think,
            )
            attention_mask = self.batch_attention_mask(
                output,
                batch_size=1,
                attn_type=args.infer_attn_type,
                cfg_factor=cfg_factor,
            )

        if args.infer_align_image_size_mode:
            batch_tgt_image_ratio_index = []
            batch_src_image_ratio_index_list = []
            batch_src_image_ori_width_list = []
            batch_src_image_ori_height_list = []
            src_image_info_list = batch_src_image_info_list if any(len(item) > 0 for item in batch_src_image_info_list) else batch_joint_image_info_list
            assert src_image_info_list is not None, "batch_src_image_info_list or batch_joint_image_info_list is required when infer_align_image_size is enabled."
            for batch_index in range(len(src_image_info_list)):
                batch_src_image_ratio_index_list.append([])
                batch_src_image_ori_width_list.append([])
                batch_src_image_ori_height_list.append([])
                for src_image_info in src_image_info_list[batch_index]:
                    if isinstance(src_image_info, ImageInfo):
                        batch_src_image_ratio_index_list[batch_index].append(src_image_info.ratio_index)
                        batch_src_image_ori_width_list[batch_index].append(src_image_info.ori_image_width)
                        batch_src_image_ori_height_list[batch_index].append(src_image_info.ori_image_height)
                    elif isinstance(src_image_info, JointImageInfo):
                        batch_src_image_ratio_index_list[batch_index].append(src_image_info.vae_image_info.ratio_index)
                        batch_src_image_ori_width_list[batch_index].append(src_image_info.vae_image_info.ori_image_width)
                        batch_src_image_ori_height_list[batch_index].append(src_image_info.vae_image_info.ori_image_height)
                    else:
                        raise ValueError(f"Unknown image info type: {type(src_image_info)}")
            for target_image_info in batch_target_image_info:
                batch_tgt_image_ratio_index.append(target_image_info.ratio_index if isinstance(target_image_info, ImageInfo) else target_image_info.vae_image_info.ratio_index)

        #  -- face cfg; update cfg_factor for face_guidance_scale
        has_face = batch_face_image_info_list is not None and len(batch_face_image_info_list[0]) > 0
        if has_face:
            if pipeline_kwargs is None:
                pipeline_kwargs = {}
            pipeline_kwargs["face_guidance_scale"] = face_guidance_scale
            # Build face batch tensor with conditional and unconditional branch according to face_guidance_scale
            src_face_embedding = self.batch_face_cond_uncond(
                batch_face_image_info_list, cfg_factor, face_guidance_scale, guidance_scale)
        else:
            src_face_embedding = None
        
        has_src = len(batch_src_image_info_list[0]) > 0
        has_joint = len(batch_joint_image_info_list[0]) > 0

        # -- vae cfg and encode if applicable
        assert not (has_src and has_joint), "has_src and has_joint are not supported at the same time"
        if not has_face and (has_src or has_joint) and src_guidance_scale > 1.0:
            if pipeline_kwargs is None:
                pipeline_kwargs = {}
            pipeline_kwargs["src_guidance_scale"] = src_guidance_scale
            # Build src batch tensor with conditional and unconditional branch according to src_guidance_scale
            batch_image_info_list = self.batch_image_cond_uncond(
                batch_src_image_info_list if has_src else batch_joint_image_info_list, cfg_factor, src_guidance_scale, guidance_scale)
            input_src_x, input_src_t, und_images = self._encode_image(batch_image_info_list)
        else:
            input_src_x, input_src_t, und_images = self._encode_image(batch_src_image_info_list if has_src else batch_joint_image_info_list, cfg_factor)

        # -- 2d rope
        if self.args.rope_type == "2d":
            rope_image_info = self._get_batch_rope_image_info(output, sections)
            rope_kwargs = dict(rope_image_info=rope_image_info)
        else:
            rope_kwargs = {}

        if args.get("use_hf"):
            cache_kwargs = dict(
                past_key_values=HunyuanGeminiStaticCache(
                    config=self.model_dict["model"].config,
                    batch_size=batch_size * cfg_factor,
                    # Image generation will not extend sequence length, using token length as max_cache_len is enough.
                    max_cache_len=output.tokens.shape[1],
                    dtype=torch.bfloat16,
                    layer_device_map=self.model_dict["model"].layer_device_map,
                )
            )
        else:
            cache_kwargs = dict()

        vision_encoder_kwargs = None
        if has_joint and self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            vision_encoder_kwargs = defaultdict(list)
            for joint_image_info in batch_joint_image_info_list:
                vision_encoder_kwargs["spatial_shapes"].append(torch.stack([item.vision_encoder_kwargs["spatial_shapes"] for item in joint_image_info]))
                vision_encoder_kwargs["attention_mask"].append(torch.stack([item.vision_encoder_kwargs["pixel_attention_mask"] for item in joint_image_info]))
            
            if cfg_factor > 1:
                vision_encoder_kwargs["spatial_shapes"] = vision_encoder_kwargs["spatial_shapes"] * cfg_factor
                vision_encoder_kwargs["attention_mask"] = vision_encoder_kwargs["attention_mask"] * cfg_factor

        model_input_extra_kwargs = dict(
            idx=output.tokens.to(self.device),
            src_x=to_device(input_src_x, self.device),
            src_t=to_device(input_src_t, self.device),
            src_image_mask=to_device(output.src_image_mask, self.device),
            image_mask=to_device(output.gen_image_mask, self.device),
            attention_mask=to_device(attention_mask, self.device),
            timestep_scatter_index=to_device(output.timestep_scatter_index, self.device),
            guidance_scatter_index=to_device(output.guidance_scatter_index, self.device),
            und_images=to_device(und_images, self.device),
            und_image_masks=to_device(output.und_image_mask if output.und_image_mask is not None else output.face_image_mask, self.device),
            vision_encoder_kwargs={
                k: to_device(v, self.device) for k, v in vision_encoder_kwargs.items()
            } if vision_encoder_kwargs is not None else None,
            src_face_embedding=to_device(src_face_embedding, self.device),
            gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, self.device),  # for split attention
            **rope_kwargs,
            **cache_kwargs,
            **default(model_input_custom_kwargs, {}),
        )

        if verbose >= 1:
            context = self.tkwrapper.tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
            if '<face>' in context:
                context = re.sub(r"(<face>)+", lambda m: f"[<face>]{{{len(m.group(0)) // 6}}}", context)

            def compact_binary_string(binary_string):
                if binary_string is not None:
                    binary_string = ''.join(["1" if v > 0.5 else "0" for v in binary_string[0].cpu().tolist()])
                    return re.sub(r'(0|1)\1*', lambda m: f'[{m.group(1)}]{{{len(m.group(0))}}}', binary_string)
                else:
                    return None

            info_str = f"""
         token shape: {output.tokens.shape}
          context[0]: {context}
          image_size: {[f"{target_image_info.image_height}x{target_image_info.image_width}" for target_image_info in batch_target_image_info]}
                seed: {[g.initial_seed() for g in generator]}
            meanflow: {meanflow}
         infer_steps: {diff_infer_steps}
          cfg_factor: {cfg_factor}
      guidance_scale: {guidance_scale}
          flow_shift: {self.args.sample_flow_shift}{extra_log_info}"""
            if has_face:
                info_str += f"""
 face_guidance_scale: {face_guidance_scale}"""
            if has_src or has_joint:
                info_str += f"""
  src_guidance_scale: {src_guidance_scale}"""
            if verbose >= 2:
                info_str += f"""
           text_mask: {compact_binary_string(output.text_mask)}
     face_image_mask: {compact_binary_string(output.face_image_mask)}
          image_mask: {compact_binary_string(output.gen_image_mask)}
      und_image_mask: {compact_binary_string(output.und_image_mask)}
      src_image_mask: {compact_binary_string(output.src_image_mask)}"""
            self.logger.info(info_str)

        results = self.pipeline(batch_size=batch_size,
                                image_size=(batch_target_image_info[0].image_height, batch_target_image_info[0].image_width),
                                num_inference_steps=diff_infer_steps,
                                guidance_scale=guidance_scale,
                                meanflow=meanflow,
                                generator=generator,
                                output_type=output_type,
                                model_input_extra_kwargs=model_input_extra_kwargs,
                                pbar=pbar,
                                cfg_distilled=cfg_distilled,
                                **default(pipeline_kwargs, {}),
                                )
        samples = results[0]

        if args.infer_align_image_size_mode:
            target_area = args.training_image_size ** 2
            batch_size = len(batch_src_image_ratio_index_list)
            for batch_index in range(batch_size):
                src_image_ratio_index_list = batch_src_image_ratio_index_list[batch_index]
                src_image_ori_width_list = batch_src_image_ori_width_list[batch_index]
                src_image_ori_height_list = batch_src_image_ori_height_list[batch_index]
                tgt_image_ratio_index = batch_tgt_image_ratio_index[batch_index]
                if len(src_image_ratio_index_list) == 1: # single src image
                    if src_image_ratio_index_list[0] == tgt_image_ratio_index:
                        if args.infer_align_image_size_mode == "resize":
                            if abs(src_image_ori_height_list[0] / src_image_ori_width_list[0] - self.reso_group[tgt_image_ratio_index].ratio) >= 0.01:
                                scale = math.sqrt(target_area / (src_image_ori_width_list[0] * src_image_ori_height_list[0]))
                                new_w = round(src_image_ori_width_list[0] * scale)
                                new_h = round(src_image_ori_height_list[0] * scale)
                                samples[batch_index] = samples[batch_index].resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                        elif args.infer_align_image_size_mode == "resize_and_pad":
                            samples[batch_index] = unpad_image(samples[batch_index], src_image_ori_width_list[0], src_image_ori_height_list[0])
                else: # multiple src images
                    for src_image_ratio_index, src_image_ori_width, src_image_ori_height in zip(src_image_ratio_index_list, src_image_ori_width_list, src_image_ori_height_list):
                        if src_image_ratio_index == tgt_image_ratio_index:
                            if args.infer_align_image_size_mode == "resize":
                                if abs(src_image_ori_height / src_image_ori_width - self.reso_group[tgt_image_ratio_index].ratio) >= 0.01:
                                    scale = math.sqrt(target_area / (src_image_ori_width * src_image_ori_height))
                                    new_w = round(src_image_ori_width * scale)
                                    new_h = round(src_image_ori_height * scale)
                                    samples[batch_index] = samples[batch_index].resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                                break
                            elif args.infer_align_image_size_mode == "resize_and_pad":
                                samples[batch_index] = unpad_image(samples[batch_index], src_image_ori_width, src_image_ori_height)
                                break

        if save_paths is not None:
            if self.args.launcher == 'pure_torch':
                pp_rank = get_parallel_state().pp_mesh.get_local_rank()
                if pp_rank == 0:
                    self.save_batch_image(samples, save_paths)
            else:
                self.save_batch_image(samples, save_paths)

        if return_only_samples:
            return samples
        
        if grpo_gen_text_mask_type is not None:
            results += ({
                "text_mask": output.text_mask
            },)
        return results

    def image_info_from_hw(self, input_height, input_width):
        if self.args.exact_size:
            image_width, image_height = input_width, input_height
        else:
            image_width, image_height = self.reso_group.get_target_size(input_width, input_height)
        token_height = image_height // (self.vae_downsample_factor[0] * self.args.patch_size)
        token_width = image_width // (self.vae_downsample_factor[1] * self.args.patch_size)
        base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
        info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, image_token_length=token_width * token_height,
            base_size=base_size, ratio_index=ratio_index,
        )
        return info

    def _image_info_from_shape_tokens(self, ratio_idx_token):
        raise NotImplementedError("This method is not implemented")
        ratio_index = ratio_idx_token - self.shape_token_offset
        reso = self.reso_group[ratio_index]
        image_height, image_width = reso.height, reso.width
        return self.image_info_from_hw(image_height, image_width)

    def get_model_input_kwargs(self, message_list, output, sections, model_input_custom_kwargs=None,
                               cfg_factor=None, **kwargs):
        # assert batch_size == 1
        if cfg_factor is None:
            cfg_factor = self.cfg_factor(kwargs)
        batch_src_image_info_list = [
            [message['content'] for message in message_list if message['type'] == 'src_image']
        ]
        batch_joint_image_info_list = [
            [message['content'] for message in message_list if message['type'] == 'joint_image']
        ]

        # -- vae encode if applicable
        if len(batch_src_image_info_list[0]) > 0 or len(batch_joint_image_info_list[0]) > 0:
            input_src_x, input_src_t, und_images = self._encode_image(batch_src_image_info_list if len(batch_src_image_info_list[0]) > 0 else batch_joint_image_info_list, cfg_factor)
        else:
            input_src_x, input_src_t, und_images = None, None, None

        # -- face
        src_face_embedding = [message['content'].image_tensor for message in message_list if message['type'] == 'face']
        if len(src_face_embedding) == 0:
            src_face_embedding = None
        else:
            src_face_embedding = torch.stack(src_face_embedding)

        # 计算 RoPE 2D
        rope_image_info = self._get_batch_rope_image_info(output, sections)
        rope_kwargs = dict(rope_image_info=rope_image_info)
        # 创建 StaticCache
        if self.args.get("use_hf"):
            cache_kwargs = dict(
                past_key_values=HunyuanGeminiStaticCache(
                    config=self.model_dict["model"].config,
                    batch_size=1,
                    # Image generation will not extend sequence length, using token length as max_cache_len is enough.
                    max_cache_len=self.args.block_size,
                    dtype=torch.bfloat16,
                    layer_device_map=self.model_dict["model"].layer_device_map,
                    dynamic=True,
                )
            )
        else:
            cache_kwargs = dict()

        # 创建 attention mask
        attention_mask = self.batch_attention_mask(output=output, batch_size=1)

        vision_encoder_kwargs = None
        has_joint = any(msg['type'] == 'joint_image' for msg in message_list)
        if has_joint and self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
            vision_encoder_kwargs = defaultdict(list)
            for joint_image_info in batch_joint_image_info_list:
                vision_encoder_kwargs["spatial_shapes"].append(
                    torch.stack([item.vision_encoder_kwargs["spatial_shapes"] for item in joint_image_info]))
                vision_encoder_kwargs["attention_mask"].append(
                    torch.stack([item.vision_encoder_kwargs["pixel_attention_mask"] for item in joint_image_info]))

        model_input_extra_kwargs = dict(
            src_x=to_device(input_src_x, self.device),
            src_t=to_device(input_src_t, self.device),
            src_image_mask=to_device(output.src_image_mask, self.device),
            image_mask=to_device(output.gen_image_mask, self.device),
            timestep_scatter_index=to_device(output.timestep_scatter_index, self.device),
            und_images=to_device(und_images, self.device),
            und_image_masks=to_device(output.und_image_mask if output.und_image_mask is not None else output.face_image_mask, self.device),
            attention_mask=to_device(attention_mask, self.device),
            vision_encoder_kwargs={
                k: to_device(v, self.device) for k, v in vision_encoder_kwargs.items()
            } if vision_encoder_kwargs is not None else None,
            src_face_embedding=to_device(src_face_embedding, self.device),
            **rope_kwargs,
            **cache_kwargs,
            **default(model_input_custom_kwargs, {}),
        )
        if self.args.use_joint_image_feature:
            image_slices = [output.joint_image_slices[0] + output.gen_image_slices[0]]
        else:
            image_slices = [output.src_image_slices[0] + output.gen_image_slices[0] + output.face_image_slices[0]]

        return model_input_extra_kwargs, image_slices

    @staticmethod
    def _random_choice(instructions, seed):
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            instruction = random.choice(instructions)
            random.setstate(state)
        else:
            instruction = random.choice(instructions)
        return instruction

    def _maybe_inject_template(self, message_list, seed=None, instruct_type=None):

        if instruct_type == 'face':
            if len(message_list) >= 2 and message_list[-2]['type'] == 'face' and message_list[-1]['type'] == 'text':
                instruction = self._random_choice(face_id_instructions, seed)
                message_list[-1]['content'] = instruction + message_list[-1]['content']

        return message_list

    def parse_cond_image(self, message):
        # Standardize various image source into ImageInfo instance.
        if message['content_type'] == 'image_info':
            pass
        elif message['content_type'] in ['local_path', 'base64', 'url']:
            pil_image = read_image(message['content_type'], message['content'], logger=self.logger)
            if pil_image is None:
                raise ValueError(f"Failed to read image from {message['content']}")
            new_message = dict(
                role=message['role'],
                type=message['type'],
                content=(
                    self.process_src_image(pil_image, use_joint_image_feature=message['type'] == 'joint_image')
                    if message['type'] in ['src_image', 'joint_image']
                    else self.process_src_image_face(pil_image)
                ),
                content_type='image_info',
            )
            message = new_message
        else:
            raise ValueError(f"Unknown content type: {message['content_type']}")
        return message

    @staticmethod
    def _determine_recaption(message_list, recaption):
        if not recaption:
            return recaption
        # recaption only can be enabled if and only if one src_image presents
        src_image_counts = 0
        for msg in message_list:
            if msg['type'] == 'src_image':
                src_image_counts += 1
        return src_image_counts == 1 and recaption

    def legacy_t2i(self, message_list, **kwargs):
        return_image_type = kwargs.get('return_image_type', 'pil')
        # seed
        seed = kwargs.get('seed')
        seed = self.prepare_seed(seed=seed, batch_size=1, num_sample_per_prompt=1)[0]
        generator = torch.Generator(self.device).manual_seed(seed)
        # Build image message if not exists
        if message_list[-1]['type'] != 'gen_image':
            if 'size' in kwargs:
                self.logger.warning("'size' is deprecated, please use 'image_size' instead.")
                if 'image_size' in kwargs:
                    raise ValueError("Both 'size' and 'image_size' are provided, please only use 'image_size'.")
                kwargs['image_size'] = kwargs.pop('size')
            size = kwargs.get('image_size', f'{self.args.training_image_size}x{self.args.training_image_size}')
            # image size
            try:
                height, width = self.parse_image_size(size, base_size=self.args.training_image_size)
            except Exception as e:
                self.logger.error(f"Failed to parse image size: {size}")
                height, width = self.args.training_image_size, self.args.training_image_size
            # messages
            gen_info = self.image_info_from_hw(input_height=height, input_width=width)
            message_list = message_list + [
                dict(role='assistant', type='gen_image', content=gen_info),
            ]

        image = self.x2image(
            # prompt-related
            batch_prompt_list=None,
            message_list=message_list,
            # others
            generator=[generator],
            **kwargs,
        )[0]
        if return_image_type == 'base64':
            image = image_to_base64(image)
        return {'role': 'Assistant', 'value': image, 'type': 'image'}

    @torch.no_grad()
    def _generate(
            self,
            message_list: List[Dict[str, Any]],
            verbose: int = 1,
            skip_special_tokens=True,
            save_image: bool = False,
            legacy_t2i: bool = False,
            **kwargs,
    ):
        """
        A uniform interface for all the t2i, general editing, lm, and mmu tasks.
        Only batch_size 1 is supported.

        Parameters
        ----------
        message_list : List[Dict[str, Any]]
            A list of dictionaries containing the history messages and new questions.
            [
                dict(role='system', type='text', content='xxxx', content_type='str')
                dict(role='user', type='text', content='xxxx', content_type='str'),
                dict(role='user', type='src_image', content='xxxx', content_type='pil'),
                dict(role='user', type='src_image', content='xxxx', content_type='base64'),
                dict(role='assistant', type='text', content='xxxx', content_type='str')
                dict(role='assistant', type='src_image', content='xxxx', content_type='pil')
                dict(role='assistant', type='src_image', content='xxxx', content_type='base64')
            ]
        think: bool
            Whether to add the <think> token to the prompt.
        verbose: int
            The verbosity level. 0 for silent, 1 for detailed info.
        skip_special_tokens: bool
            Whether to skip the special tokens.
        save_image: bool
            Whether to save the image.
        kwargs: dict
            block_size: int
                The maximum length of the input sequence. Default to args.block_size.
            chat_mode: str
                The chat mode. 'free' for free chat, 'text' for text generation, 'image_gen' for image generation.
                Default to 'free'.
            inherit_img_ratio: bool
                Whether to inherit the image ratio from the previous message. Default to False.
            recaption: bool
                Whether to recaption the prompt before generating the image. Default to False.
            use_face_id: bool
                Whether to extract face ID from the src image for image generation. Default to False.
            return_face: bool
                Whether to return the cropped face image. Default to False. Only used when use_face_id is True.
            return_image_type: str
                Can be 'pil' or 'base64'. Default to 'pil'.
            return_stop_message: bool
                Whether to return the stop message. Default to True.
        """
        args = self.args
        mode = kwargs.get('mode', 'gen_text')

        if legacy_t2i or mode == 'gen_image':
            return self.legacy_t2i(message_list, **kwargs)

        # Common arguments
        block_size = kwargs.get('block_size', args.block_size)
        seed = kwargs.get('seed')
        sequence_template = kwargs.get("sequence_template", "instruct")
        penalty = kwargs.get('penalty', 1.0)
        streamer: Optional[TextStreamer] = kwargs.get('streamer', None)
        bot_task = kwargs.get('bot_task', "auto")
        drop_think = kwargs.get('drop_think', False)
        image_size = kwargs.get('size', "auto")
        max_new_tokens = kwargs.get('max_new_tokens', None)

        tkw = self.tkwrapper
        if image_size == "auto":
            extra_auto_stops = [tkw.stmap[f"<img_ratio_{i}>"] for i in range(33)]
        else:
            extra_auto_stops = [tkw.stmap["<boi>"]]
        stop_token_ids_map = dict(
            auto=[tkw.eos_token] + extra_auto_stops,
            image=[tkw.eos_token],
            recaption=[tkw.stmap["</recaption>"], tkw.stmap["</answer>"], tkw.eos_token],
            think=[tkw.stmap["</think>"], tkw.stmap["</answer>"], tkw.eos_token],
            img_ratio=extra_auto_stops,
        )
        stop_token_ids = stop_token_ids_map[bot_task]

        default_logits_processor = self._get_logits_processor(kwargs)
        img_ratio_logits_processor = get_logits_processors([
            dict(SliceVocabLogitsWarper=dict(vocab_start=self.shape_token_start, vocab_end=self.shape_token_end, other_slices=self.other_slices)),
        ])
        #   -- seed
        seed = self.prepare_seed(
            seed=seed,
            batch_size=1,
            num_sample_per_prompt=1,
        )[0]
        generator = torch.Generator(self.device).manual_seed(seed)

        # recaption = self._determine_recaption(message_list, kwargs.get('recaption', False))
        output, sections = self.apply_general_template(
            [message_list],
            max_length=block_size,
            add_assistant_prefix=True,
            bot_task=bot_task,
            sequence_template=sequence_template,
            batchify=True,
            image_base_size=self.base_size,
            drop_think=drop_think,
        )
        model_input_kwargs, image_slices = self.get_model_input_kwargs(message_list, output, sections, cfg_factor=1)

        if streamer is not None:
            streamer.put(output.tokens[0])

        inputs = output.tokens.to(self.device)
        # `real_pos` is the length of valid input tokens (ignore pad tokens when batching).
        # It also indicates the position of the next token to be predicted in the input sequence.
        real_pos = output.real_pos.to(self.device)

        input_pos = torch.arange(0, inputs.shape[1], dtype=torch.long, device=self.device)
        if args.get("use_hf"):
            self.model_dict["model"].set_rope_cache(
                seq_len=args.block_size, rope_image_info=model_input_kwargs["rope_image_info"], device=self.device,
            )
        else:
            self.model_dict["model"].set_kv_cache(batch_size=1, device=self.device, image_slices=image_slices)
        infer_max_steps = default(max_new_tokens, block_size - real_pos.max())

        next_token_ids = torch.empty([1, 0], dtype=torch.long, device=self.device)
        next_pos = real_pos
        logits_processor = img_ratio_logits_processor if bot_task == "img_ratio" else default_logits_processor
        pbar = range(infer_max_steps)
        fixed_model_input_kwargs = {}

        if self.rank == 0:
            pbar = tqdm(pbar, desc="Text mode", leave=False, disable=streamer is not None)
        # assistant_text save the text generated by the assistant just before entering the image generation mode
        for i in pbar:
            if i == 0:
                if verbose >= 1:
                    context = self.tkwrapper.tokenizer.decode(inputs.cpu()[0], skip_special_tokens=False)
                    # Replace <img><img>...<img> with [<img>]{number}
                    context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
                    context = re.sub(r"(<face>)+", lambda m: f"[<face>]{{{len(m.group(0)) // 6}}}", context)
                    info_str = f"""
                 context: 
{context}
                    seed: {generator.initial_seed()}
        logits_processor: {default_logits_processor}
img_ratio_logits_processor: {img_ratio_logits_processor}"""
                    self.logger.info(info_str)

            outs = self.get_next_token(
                inputs,
                input_pos,
                generators=[generator],
                # `real_pos` is used in prefill stage. Here we use `real_pos - 1` instead of `next_pos - 1` to
                # allow to encode multiple tokens (using prefill) in the middle steps. In this way, one should
                # modify `real_pos` to the length of input multiple tokens.
                real_pos=real_pos - 1,
                logits_processor=logits_processor,
                next_token_ids=next_token_ids,
                penalty=penalty,
                do_sample=args.do_sample,
                **model_input_kwargs,
                **fixed_model_input_kwargs,
            )
            if isinstance(outs, dict):
                next_token = outs["next_token"]
                # For HF inferring
                if "past_key_values" in outs:
                    fixed_model_input_kwargs["past_key_values"] = outs["past_key_values"]
            else:
                next_token = outs
            next_token_ids = torch.cat([next_token_ids, next_token], dim=1)
            inputs = next_token
            input_pos = next_pos.clone()
            next_pos += 1
            model_input_kwargs = {}

            if streamer is not None:
                # print(f"{next_token.cpu()}, "
                #       f"{self.tkwrapper.tokenizer.decode(next_token.cpu()[0], skip_special_tokens=False)}")
                streamer.put(next_token.cpu())

            # TODO(jarvizhang): move route to a separated method
            # Route
            next_tokens = [next_token.item()]
            if next_tokens[0] in stop_token_ids:
                break

        if streamer is not None:
            streamer.end()

        # clear cache
        if args.get("use_hf"):
            self.model_dict["model"].clear_rope_cache()
        else:
            self.model_dict["model"].clear_kv_cache()

    def generate(self, *args, **kwargs):
        return self._generate(*args, **kwargs)

    # ================================================================================
    # Below are batch inference methods
    # ================================================================================
    @torch.no_grad()
    def batch_x2image(
        self,
        # prompt-related
        batch_prompt_list=None,
        sequence_template="pretrain",
        batch_system_prompt=None,
        batch_system_prompt_if_drop_think=None,
        # image-related
        batch_src_image_info_list=None,
        batch_und_image_info_list=None,
        batch_joint_image_info_list=None,
        batch_face_image_info_list=None,
        batch_prefill_template=None,
        batch_negative_prompt_list=None,
        predict_image_shape_token=False,
        sample_image_size=None,
        # others
        task=None,
        return_only_samples=True,
        # messages
        batch_message_list: Optional[List[List[Dict[str, Any]]]] = None,
        seeds=None,
        skip2txt=False,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Prepare prompts; Append task_instruction before user_prompt; Calculate target_image_shape_info; call x2image;
        """
        args = self.args
        out_dict = {}
        verbose = kwargs.get("verbose", 1)
        drop_think = kwargs.get("drop_think", False)
        only_recaption = kwargs.get("only_recaption", False)
        exact_size = kwargs.get("exact_size", args.exact_size)
        assert task, "Task should be provided."

        # -- Prompt and seeds
        if batch_message_list is None:
            batch_size = len(batch_prompt_list)
            batch_message_list = [None] * batch_size
        else:
            batch_size = len(batch_message_list)
        out_dict['prompt'] = batch_prompt_list

        if seeds is None:
            seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size, num_sample_per_prompt=1)
        out_dict["seeds"] = seeds
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        batch_cot_text=None
        replace_system_prompt = False
        if args.cot_max_length > 0 and not skip2txt:
            pred_text_mode = args.get("t2i_pred_text_mode")
            prefix_tokens = dict(
                cot_think="<think>",
                cot_recaption="<recaption>",
                mixed="<think>", # <think></think><recaption></recaption>
            )
            stop_tokens = dict(
                cot_think="</think>",
                cot_recaption="</recaption>",
                mixed="</think>",
            )
            sentinel_tokens = [self.tkwrapper.eos_token] + [
                self.tkwrapper.tokenizer.encode(token)[0] for token in ["<boi>", "</think>", "</recaption>"]
            ]

            def run_cot(pred_text_mode, previous_batch_cot_text=None, previous_batch_cot_tokens=None):
                stop_token_text = stop_tokens[pred_text_mode]
                desired_stop_token = self.tkwrapper.tokenizer.encode(stop_token_text)[0]
                cot_out = self.batch_x2text(
                    batch_prompt_list,
                    xtype="t2i",
                    seed=kwargs.get('text_gen_seed', kwargs.get('seed')),
                    verbose=max(2, verbose),
                    block_size=kwargs.get('block_size', args.block_size),
                    max_new_tokens=kwargs.get('max_new_tokens', args.cot_max_length),
                    skip_special_tokens=False,
                    sequence_template=sequence_template,
                    batch_system_prompt=batch_system_prompt,
                    batch_src_image_info_list=batch_src_image_info_list,
                    batch_und_image_info_list=batch_und_image_info_list,
                    batch_joint_image_info_list=batch_joint_image_info_list,
                    batch_face_image_info_list=batch_face_image_info_list,
                    batch_prefill_template=batch_prefill_template,
                    pred_text_mode=pred_text_mode,
                    batch_cot_text=previous_batch_cot_text,
                    batch_cot_tokens=previous_batch_cot_tokens,
                    stop_tokens=[
                        desired_stop_token,
                        self.tkwrapper.eos_token,
                        self.tkwrapper.tokenizer.encode("<boi>")[0],
                    ],
                    stop_by_all_rank=kwargs.get("stop_by_all_rank", False),
                )
                batch_next_tokens = cot_out['next_tokens'] # list of 1-d tensor with dtype=torch.long
                fixed_batch_next_tokens = []
                # 1. tensor to list
                # 2. 模型可能没有预测出 </recaption> 或 </think>，而是预测了 eos 或 boi 被强制结束返回了. 这里做个兜底

                for next_tokens in batch_next_tokens:
                    next_tokens = next_tokens.tolist()
                    if next_tokens[-1] not in sentinel_tokens:
                        next_tokens = next_tokens[:-1] + [desired_stop_token]
                    fixed_batch_next_tokens.append(next_tokens)
                batch_next_tokens = fixed_batch_next_tokens  # list of list of int
                batch_cot_text = []
                for next_tokens in batch_next_tokens:
                    text = prefix_tokens[pred_text_mode] + self.tkwrapper.tokenizer.decode(next_tokens, skip_special_tokens=False)
                    batch_cot_text.append(text)
                for i in range(len(batch_next_tokens)):
                    batch_next_tokens[i] = self.tkwrapper.tokenizer.encode(prefix_tokens[pred_text_mode]) + batch_next_tokens[i]
                return batch_cot_text, batch_next_tokens

            if pred_text_mode in ["cot_think", "cot_recaption"]:
                batch_cot_text, batch_cot_tokens = run_cot(pred_text_mode)
                batch_cot_text_saved = [text for text in batch_cot_text]
            elif pred_text_mode == "mixed":
                # 先思考再改写
                batch_cot_think_text, batch_cot_think_tokens = run_cot("cot_think")
                batch_cot_text, batch_cot_tokens = run_cot("cot_recaption", previous_batch_cot_text=batch_cot_think_text, previous_batch_cot_tokens=batch_cot_think_tokens)
                batch_cot_text_saved = [think + recap for think, recap in zip(batch_cot_think_text, batch_cot_text)]
                if drop_think:
                    if batch_system_prompt is not None and batch_system_prompt_if_drop_think is not None:
                        replace_system_prompt = True
                else:
                    # concat think and recaption
                    batch_cot_text = [think + recap for think, recap in zip(batch_cot_think_text, batch_cot_text)]
                    batch_cot_tokens = [think_tokens + recap_tokens for think_tokens, recap_tokens in zip(batch_cot_think_tokens, batch_cot_tokens)]
            else:
                raise ValueError(f"Unknown pred_text_mode: {pred_text_mode}")

            out_dict["cot_text"] = batch_cot_text
            out_dict["cot_text_saved"] = batch_cot_text_saved

            if only_recaption:
                out_dict["samples"] = None
                return out_dict

        # -- Image-related
        # if predict_image_shape_token, will predict the image shape token
        # if not predict_image_shape_token, will use sample_image_size or follow the last src image's size
        if predict_image_shape_token and not skip2txt:
            logits_processor = get_logits_processors([
                dict(SliceVocabLogitsWarper=dict(vocab_start=self.shape_token_start, vocab_end=self.shape_token_end, other_slices=self.other_slices)),
            ])

            # Predict image shape tokens
            do_sample = args.do_sample
            if args.predict_image_shape_no_do_sample:
                do_sample = False
            batch_lm_out_dict = self.batch_x2text(
                batch_prompt_list,
                xtype="t2i",   # also works for editing/subject_driven tasks
                seed=kwargs.get('seed'),
                verbose=2,
                block_size=kwargs.get('block_size', args.block_size),
                max_new_tokens=kwargs.get('max_new_tokens', 1),
                skip_special_tokens=False,
                logits_processor=logits_processor,
                do_sample=do_sample,
                sequence_template=sequence_template,
                batch_system_prompt=batch_system_prompt,
                batch_cot_text=batch_cot_text,
                batch_cot_tokens=batch_cot_tokens,
                batch_src_image_info_list=batch_src_image_info_list,
                batch_und_image_info_list=batch_und_image_info_list,
                batch_joint_image_info_list=batch_joint_image_info_list,
                batch_face_image_info_list=batch_face_image_info_list,
                batch_prefill_template=batch_prefill_template,
            )
            batch_next_tokens = batch_lm_out_dict['next_tokens']
            # Clear CUDA cache to avoid OOM

            # Prepare target image info
            batch_target_image_info = []
            for next_tokens in batch_next_tokens:
                text = self.tkwrapper.tokenizer.decode(next_tokens, skip_special_tokens=False)
                # base_size = int(re.search(r"<img_size_(\d+)>", text).group(1))
                ratio_index = int(re.search(r"<img_ratio_(\d+)>", text).group(1))
                image_width, image_height = self.reso_group.data[ratio_index].w, self.reso_group.data[ratio_index].h
                token_height = image_height // (self.vae_downsample_factor[0] * args.patch_size)
                token_width = image_width // (self.vae_downsample_factor[1] * args.patch_size)
                batch_target_image_info.append(ImageInfo(
                    image_type="gen_image",
                    image_width=image_width,
                    image_height=image_height,
                    token_width=token_width,
                    token_height=token_height,
                    base_size=self.reso_group.base_size,
                    ratio_index=ratio_index,
                ))
        else:
            # no prediction, use specified sample_image_size
            # sample_image_size maybe int or [height, width] or list of [height, width]
            # if sample_image_size is not None and not (isinstance(sample_image_size, int) and sample_image_size == -1):
            if sample_image_size is not None and sample_image_size != [-1]:
                # unify sample_image_size to list of [height, width]
                if isinstance(sample_image_size, int):
                    sample_image_size = [[sample_image_size, sample_image_size]] * batch_size
                elif isinstance(sample_image_size, list) and all(isinstance(item, int) for item in sample_image_size):
                    sample_image_size = [sample_image_size] * batch_size
                
                batch_target_image_info = []
                for batch_index in range(batch_size):
                    image_size = self.parse_image_size(sample_image_size[batch_index], align=[self.vae_downsample_factor[0] * args.patch_size, self.vae_downsample_factor[1] * args.patch_size])
                    if exact_size:
                        image_width, image_height = image_size[1], image_size[0]
                    else:
                        image_width, image_height = self.reso_group.get_target_size(image_size[1], image_size[0])
                    token_height = image_height // (self.vae_downsample_factor[0] * args.patch_size)
                    token_width = image_width // (self.vae_downsample_factor[1] * args.patch_size)
                    base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
                    batch_target_image_info.append(ImageInfo(
                        image_type="gen_image",
                        image_width=image_width,
                        image_height=image_height,
                        token_width=token_width,
                        token_height=token_height,
                        base_size=base_size,
                        ratio_index=ratio_index,
                    ))
                    # batch_tgt_image_ratio_index.append(ratio_index)
            else:
                # no prediction, follow the last src image or joint image's size
                if (batch_src_image_info_list is None or all(len(src_image_info_list) == 0 for src_image_info_list in batch_src_image_info_list)) and (batch_joint_image_info_list is None or all(len(joint_image_info_list) == 0 for joint_image_info_list in batch_joint_image_info_list)):
                    raise ValueError("Cannot determine sample image size. Please provide --sample-image-size or batch_src_image_info_list/batch_joint_image_info_list, or enable --predict-image-shape-token.")
                if batch_src_image_info_list is not None and all(len(src_image_info_list) > 0 for src_image_info_list in batch_src_image_info_list):
                    batch_target_image_info = [src_image_info_list[args.subject_driven_shape_index].copy(copy_image_tensor=False) for src_image_info_list in batch_src_image_info_list]
                else:
                    batch_target_image_info = [joint_image_info_list[args.subject_driven_shape_index].vae_image_info.copy(copy_image_tensor=False) for joint_image_info_list in batch_joint_image_info_list]
                for target_image_info in batch_target_image_info:
                    target_image_info.image_type = "gen_image"

        # -- Batch Run
        start_time = time.time()
        if batch_negative_prompt_list is not None and isinstance(batch_negative_prompt_list, str):
            batch_negative_prompt_list = [batch_negative_prompt_list] * batch_size
        
        if kwargs.get("explicit_set_manual_seed_for_image_gen", None) is not None:
            set_manual_seed(kwargs["explicit_set_manual_seed_for_image_gen"]["seed"])
            # Disable deterministic mode before image generation to allow randomness in diffusion sampling
            # (deterministic mode was only needed for text generation to ensure consistency across ranks)
            if torch.are_deterministic_algorithms_enabled():
                torch.use_deterministic_algorithms(False)
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                
        results = self.x2image(
            # prompt-related
            batch_prompt_list,
            sequence_template=sequence_template,
            batch_system_prompt=batch_system_prompt if not replace_system_prompt else batch_system_prompt_if_drop_think,
            batch_cot_text=batch_cot_text,
            batch_cot_tokens=batch_cot_tokens,
            # image-related
            batch_und_image_info_list=batch_und_image_info_list,
            batch_src_image_info_list=batch_src_image_info_list,
            batch_joint_image_info_list=batch_joint_image_info_list,
            batch_face_image_info_list=batch_face_image_info_list,
            batch_target_image_info=batch_target_image_info,
            batch_prefill_template=batch_prefill_template,
            batch_negative_prompt_list=batch_negative_prompt_list,
            # others
            generator=generators,
            return_only_samples=return_only_samples,
            # messages
            message_list=batch_message_list[0],
            pipeline_kwargs=pipeline_kwargs,
            **kwargs,
        )
        if return_only_samples:
            out_dict["samples"] = results
        else:
            out_dict["samples"] = results[0]
            out_dict["extra_outputs"] = results[1:]

        gen_time = time.time() - start_time
        if verbose > 0:
            self.logger.info(f"Predict time: {gen_time:.2f}s")
        return out_dict

    def _get_und_image_size(self, pixel_values):
        assert isinstance(pixel_values, torch.Tensor), "pixel_values should be a torch.Tensor."
        assert pixel_values.ndim == 4, "pixel_values should have 4 dimensions."
        assert pixel_values.shape[1] == 3, "pixel_values should have 3 channels."
        _, _, ph, pw = pixel_values.shape
        th = ph // self.vision_downsample_factor[0]
        tw = pw // self.vision_downsample_factor[1]
        return ph, pw, th * tw

    def _get_batch_rope_image_info(self, output, sections):
        rope_image_info = None
        # -- 2d rope
        if self.args.rope_type == "2d":
            rope_image_info = []
            for image_slices, sections_i in zip(output.all_image_slices, sections):
                image_shapes = []
                for section in sections_i:
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
                rope_image_info.append(list(zip(image_slices, image_shapes)))
        return rope_image_info

    def _task_preprocessor(
            self, xtype, prompt,
            image=None,
            sequence_template='pretrain',
            batch_system_prompt=None,
            batch_cot_text=None,
            batch_cot_tokens=None,
            batch_src_image_info_list=None,
            batch_und_image_info_list=None,
            batch_joint_image_info_list=None,
            batch_face_image_info_list=None,
            batch_prefill_template=None,
            model_input_custom_kwargs=None,
            **kwargs
    ):
        """
        Returns
        -------
        output: TokenizerEncodeOutput
        model_input_kwargs: dict
            The kwargs for the model input.
        image_slices: List[List[slice]]
            The slices for the image tokens.
        fixed_model_input_kwargs: dict
            The kwargs for the model input that are fixed and do not change with the input.
        """
        args = self.args
        block_size = kwargs.get('block_size', args.block_size)
        batch_size = len(prompt)
        fixed_model_input_kwargs = {}

        if xtype == "t2i":
            # This branch is used by t2i for predicting image ratio token.
            image_type = kwargs.get("pred_text_mode", "image_ratio")    # image_ratio/cot_think/cot_recaption

            if batch_system_prompt is None:
                batch_system_prompt = [None for _ in range(batch_size)]
            if batch_cot_text is None:
                batch_cot_text = [None for _ in range(batch_size)]
            if batch_cot_tokens is None:
                batch_cot_tokens = [None for _ in range(batch_size)]

            if batch_und_image_info_list is None:
                batch_und_image_info_list = [[] for _ in range(batch_size)]
            if batch_src_image_info_list is None:
                batch_src_image_info_list = [[] for _ in range(batch_size)]
            if batch_joint_image_info_list is None:
                batch_joint_image_info_list = [[] for _ in range(batch_size)]
            if batch_face_image_info_list is None:
                batch_face_image_info_list = [[] for _ in range(batch_size)]

            batch_target_image_info = [ImageInfo(image_type=image_type, image_token_length=0) for _ in range(batch_size)]
            # Input one sample and output batched one
            out = self.apply_x2image_template(
                batch_prompt_list=prompt,
                sequence_template=sequence_template,
                batch_system_prompt=batch_system_prompt,
                batch_cot_text=batch_cot_text,
                batch_cot_tokens=batch_cot_tokens,
                batch_target_image_info=batch_target_image_info,
                batch_und_image_info_list=batch_und_image_info_list,
                batch_src_image_info_list=batch_src_image_info_list,
                batch_joint_image_info_list=batch_joint_image_info_list,
                batch_face_image_info_list=batch_face_image_info_list,
                batch_prefill_template=batch_prefill_template,
                answer=True,
                cfg_factor=1,
                return_attention_mask=True,
                k_seq_len=args.block_size,
            )
            output, sections = out['output'], out['sections']
            if len(batch_src_image_info_list[0]) > 0 or len(batch_joint_image_info_list[0]) > 0:
                input_src_x, input_src_t, und_images = self._encode_image(batch_src_image_info_list if len(batch_src_image_info_list[0]) > 0 else batch_joint_image_info_list, cfg_factor=1)
            else:
                input_src_x, input_src_t, und_images = None, None, None

            # stack vision_image_kwargs
            vision_encoder_kwargs = dict(
                spatial_shapes=torch.stack([
                    torch.stack([
                        joint_image_info.vision_encoder_kwargs["spatial_shapes"]
                        for joint_image_info in joint_image_info_list
                    ])
                    for joint_image_info_list in batch_joint_image_info_list
                ]).to(self.device) if batch_joint_image_info_list[0] else None,  # batch_size x n x seq_len x dim
                attention_mask=torch.stack([
                    torch.stack([
                        joint_image_info.vision_encoder_kwargs["pixel_attention_mask"]
                        for joint_image_info in joint_image_info_list
                    ])
                    for joint_image_info_list in batch_joint_image_info_list
                ]).to(self.device) if batch_joint_image_info_list[0] else None,  # batch_size x n x seq_len
            )

            # 计算 RoPE 2D
            rope_image_info = self._get_batch_rope_image_info(output, sections)
            rope_kwargs = dict(rope_image_info=rope_image_info)
            # 创建 StaticCache
            if args.get("use_hf"):
                cache_kwargs = dict(
                    past_key_values=HunyuanGeminiStaticCache(
                        config=self.model_dict["model"].config,
                        batch_size=batch_size,
                        # Image generation will not extend sequence length, using token length as max_cache_len is enough.
                        max_cache_len=args.block_size,
                        dtype=torch.bfloat16,
                        layer_device_map=self.model_dict["model"].layer_device_map,
                        dynamic=True,   # dynamic cache: use real attention mask in prefill, and attention_mask=None in decode.
                    )
                )
            else:
                cache_kwargs = dict()

            model_input_kwargs = dict(
                src_x=to_device(input_src_x, self.device),
                src_t=to_device(input_src_t, self.device),
                src_image_mask=to_device(output.src_image_mask, self.device),
                und_images=to_device(und_images, self.device),
                und_image_masks=to_device(output.und_image_mask if output.und_image_mask is not None else output.face_image_mask, self.device),
                timestep_scatter_index=to_device(output.timestep_scatter_index, self.device),
                attention_mask=to_device(out.get("attention_mask"), self.device),
                vision_encoder_kwargs=vision_encoder_kwargs,
                **rope_kwargs,
                **cache_kwargs,
                **default(model_input_custom_kwargs, {}),
            )
            image_slices = output.joint_image_slices if output.joint_image_slices else output.src_image_slices

        elif xtype == "lm":
            out = self.apply_lm_template(prompt, max_length=block_size, sequence_template=sequence_template,
                                         batch_system_prompt=batch_system_prompt,
                                         return_attention_mask=True)
            output, sections = out['output'], out['sections']

            rope_image_info = self._get_batch_rope_image_info(output, sections)
            rope_kwargs = dict(rope_image_info=rope_image_info)
            # 创建 StaticCache
            if args.get("use_hf"):
                cache_kwargs = dict(
                    past_key_values=HunyuanGeminiStaticCache(
                        config=self.model_dict["model"].config,
                        batch_size=batch_size,
                        # Image generation will not extend sequence length, using token length as max_cache_len is enough.
                        max_cache_len=args.block_size,
                        dtype=torch.bfloat16,
                        layer_device_map=self.model_dict["model"].layer_device_map,
                        dynamic=True,   # dynamic cache: use real attention mask in prefill, and attention_mask=None in decode.
                    )
                )
            else:
                cache_kwargs = dict()

            model_input_kwargs = dict(
                attention_mask=to_device(out.get("attention_mask"), self.device),
                **rope_kwargs,
                **cache_kwargs,
            )
            image_slices = None

        elif xtype == "mmu":
            if args.get("use_joint_image_feature"):
                if not args.kv_cache or args.get("use_hf"):
                    return_attention_mask = True
                else:
                    return_attention_mask = False

                out = self.apply_mmu_template(
                    prompt, max_length=block_size,
                    batch_joint_image_info_list=batch_joint_image_info_list,
                    return_attention_mask=return_attention_mask,
                )
                output, sections = out['output'], out['sections']

                if len(batch_joint_image_info_list[0]) > 0:
                    input_src_x, input_src_t, und_images = self._encode_image(batch_joint_image_info_list, cfg_factor=1)
                else:
                    input_src_x, input_src_t, und_images = None, None, None

                # stack vision_image_kwargs
                vision_encoder_kwargs = dict(
                    spatial_shapes=torch.stack([
                        torch.stack([
                            joint_image_info.vision_encoder_kwargs["spatial_shapes"]
                            for joint_image_info in joint_image_info_list
                        ])
                        for joint_image_info_list in batch_joint_image_info_list
                    ]).to(self.device),  # batch_size x n x seq_len x dim
                    attention_mask=torch.stack([
                        torch.stack([
                            joint_image_info.vision_encoder_kwargs["pixel_attention_mask"]
                            for joint_image_info in joint_image_info_list
                        ])
                        for joint_image_info_list in batch_joint_image_info_list
                    ]).to(self.device)  # batch_size x n x seq_len
                )

                # 计算 RoPE 2D
                rope_image_info = self._get_batch_rope_image_info(output, sections)
                rope_kwargs = dict(rope_image_info=rope_image_info)
                # 创建 StaticCache
                if args.get("use_hf"):
                    cache_kwargs = dict(
                        past_key_values=HunyuanGeminiStaticCache(
                            config=self.model_dict["model"].config,
                            batch_size=batch_size,
                            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
                            max_cache_len=args.block_size,
                            dtype=torch.bfloat16,
                            layer_device_map=self.model_dict["model"].layer_device_map,
                            dynamic=True,   # dynamic cache: use real attention mask in prefill, and attention_mask=None in decode.
                        )
                    )
                else:
                    cache_kwargs = dict()

                model_input_kwargs = dict(
                    src_x=to_device(input_src_x, self.device),
                    src_t=to_device(input_src_t, self.device),
                    src_image_mask=to_device(output.src_image_mask, self.device),
                    und_images=to_device(und_images, self.device),
                    timestep_scatter_index=to_device(output.timestep_scatter_index, self.device),
                    und_image_masks=to_device(output.und_image_mask, self.device),
                    vision_encoder_kwargs=vision_encoder_kwargs,
                    attention_mask=to_device(out.get("attention_mask"), self.device),
                    **rope_kwargs,
                    **cache_kwargs,
                    **default(model_input_custom_kwargs, {}),
                )
                image_slices = output.joint_image_slices
            else:
                ph, pw, image_token_length = self._get_und_image_size(image)
                out = self.apply_mmu_template(
                    prompt, image_token_length, (ph, pw), max_length=block_size,
                )
                output, sections = out['output'], out['sections']
                model_input_kwargs = dict(
                    und_images=image.to(self.device),
                    und_image_masks=output.und_image_mask.to(self.device),
                )
                image_slices = output.und_image_slices

        else:
            raise ValueError(f"Unknown xtype: {xtype}")

        return output, model_input_kwargs, image_slices, fixed_model_input_kwargs

    @torch.no_grad()
    def batch_x2text(self, prompt, xtype='lm', image=None, **kwargs):
        args = self.args
        out_dict = {}
        verbose = kwargs.get("verbose", 1)
        block_size = kwargs.get('block_size', args.block_size)
        max_new_tokens = kwargs.get('max_new_tokens', 4096)
        logits_processor = self._get_logits_processor(kwargs)
        do_sample = kwargs.get('do_sample', args.do_sample)
        penalty = kwargs.get('penalty', 1.0)
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        # One can use custom stop token to perform early stopping.
        stop_tokens = kwargs.get('stop_tokens', [self.tkwrapper.eos_token])
        # for mmlu_bench evaluation
        return_first_pred_token_logits = kwargs.get('return_first_pred_token_logits', False)

        output, model_input_kwargs, image_slices, fixed_model_input_kwargs = \
            self._task_preprocessor(xtype, prompt, image, **kwargs)

        # -- Prompt and seeds
        if isinstance(prompt, str):
            prompt = [prompt]
        out_dict['prompt'] = prompt
        bsz = len(prompt)

        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=bsz, num_sample_per_prompt=1)
        out_dict["seeds"] = seeds
        generators = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        inputs = output.tokens.to(self.device)
        real_pos = output.real_pos.to(self.device)
        assert real_pos.ndim == 2, f"Invalid real_pos shape: {real_pos.shape}"

        if verbose >= 1:
            context = self.tkwrapper.tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}
            context = re.sub(r"(<img>)+", lambda m: f"[<img>]{{{len(m.group(0)) // 5}}}", context)
            context = re.sub(r"(<pad>)+", lambda m: f"[<pad>]{{{len(m.group(0)) // 5}}}", context)
            info_str = f"""
                   token shape: {inputs.shape}
                    context[0]: {context}
                          seed: {[g.initial_seed() for g in generators]}
                max_new_tokens: {max_new_tokens}
              logits_processor: {logits_processor}
return_first_pred_token_logits: {return_first_pred_token_logits}
                       verbose: {verbose}
                        """
            self.logger.info(info_str)

        is_dummy = kwargs.get("is_dummy")
        finished_by_dummy = False

        start_time = time.time()
        infer_max_steps = min(max_new_tokens, block_size - real_pos.max())
        pbar = range(infer_max_steps)
        # if self.rank == 0:
        #     pbar = tqdm(pbar, desc="Text mode", leave=False)
        next_token_ids = torch.empty([bsz, 0], dtype=torch.long, device=self.device)

        def _check_model_input_kwargs(kwargs):
            """Filter out None values and empty dicts from model input kwargs, used for pure-torch parallel."""
            if self.args.launcher != 'pure_torch':
                return kwargs
            else:
                filtered_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, dict):
                        # Filter out None values from nested dict
                        filtered_dict = {k_: v_ for k_, v_ in v.items() if v_ is not None}
                        # Only keep the dict if it's not empty
                        if filtered_dict:
                            filtered_kwargs[k] = filtered_dict
                    elif v is not None:
                        # Keep non-None values that aren't dicts
                        filtered_kwargs[k] = v
                return filtered_kwargs

        if args.kv_cache:
            batch_input_pos = torch.arange(
                0, inputs.shape[1], dtype=torch.long, device=self.device)[None].expand(
                bsz, -1)  # use expand to share indices to save memory
            # HF model will handle kv_cache automatically, so we don't need to set it here.

            if args.get("use_hf"):
                self.model_dict["model"].set_rope_cache(
                    seq_len=args.block_size, rope_image_info=model_input_kwargs["rope_image_info"], device=self.device,
                )
            else:
                self.model_dict["model"].set_rope_cache(
                    seq_len=args.block_size, rope_image_info=model_input_kwargs["rope_image_info"], device=self.device,
                )
                self.model_dict["model"].set_kv_cache(batch_size=bsz, device=self.device, image_slices=image_slices)

            stop_flag = torch.zeros([bsz], dtype=torch.bool, device=self.device)
            current_pos = real_pos
            # Sometimes a word is represented by multiple tokens, so we need to cache the tokens
            # until we get the full word.
            work_tokens_cache = []

            # Used for expert parallel: if stop_by_all_rank is True, the inference will stop when all ranks finished. 
            stop_by_all_rank = kwargs.get("stop_by_all_rank", False)
            if stop_by_all_rank:
                is_stop_cur_rank = torch.tensor([0], dtype=torch.int32, device=self.device)
            
            for i in pbar:
                # NOTE: Filter out None values and empty dicts from model input kwargs, which is demanded by pure-torch parallel.
                fixed_model_input_kwargs = _check_model_input_kwargs(fixed_model_input_kwargs)
                outs = self.get_next_token(
                    inputs,
                    batch_input_pos,
                    generators=generators,
                    real_pos=real_pos - 1,
                    logits_processor=logits_processor,
                    return_logits=return_first_pred_token_logits,
                    next_token_ids=next_token_ids,
                    penalty=penalty,
                    do_sample=do_sample,
                    **(_check_model_input_kwargs(model_input_kwargs) if i == 0 else {}),
                    **fixed_model_input_kwargs,
                )
                if return_first_pred_token_logits:
                    outs, logits = outs
                if isinstance(outs, dict):
                    next_token = outs["next_token"]
                    # For HF inferring
                    if "past_key_values" in outs:
                        fixed_model_input_kwargs["past_key_values"] = outs["past_key_values"]
                else:
                    next_token = outs

                # verbose 3: streaming print decoded text. Don't use it in dp mode.
                if verbose == 3:
                    work_tokens_cache.append(next_token.item())
                    text = self.tkwrapper.tokenizer.decode(work_tokens_cache, skip_special_tokens=False)
                    # '�' (\ufffd) is a unicode character that indicates an incomplete word
                    if '�' not in text:
                        print(text, end="", flush=True)
                        work_tokens_cache.clear()

                next_token_ids = torch.cat([next_token_ids, next_token], dim=1)
                inputs = next_token
                batch_input_pos = current_pos.clone()
                current_pos += 1
                # Update stop flag and determine whether to break
                for stop_token in stop_tokens:
                    stop_flag |= (next_token.squeeze(-1) == stop_token)

                if stop_by_all_rank:
                    if stop_flag.all() or i == infer_max_steps - 1:
                        is_stop_cur_rank = torch.tensor([1], dtype=torch.int32, device=self.device)
                    else:
                        is_stop_cur_rank = torch.tensor([0], dtype=torch.int32, device=self.device)
                    dist.all_reduce(is_stop_cur_rank, op=dist.ReduceOp.SUM)
                    if is_stop_cur_rank.item() == dist.get_world_size():
                        break
                elif stop_flag.all():
                    break
                if return_first_pred_token_logits:
                    break

            if verbose == 3:
                print()

            # Clear cache
            if args.get("use_hf"):
                self.model_dict["model"].clear_rope_cache()
            else:
                self.model_dict["model"].clear_kv_cache()
                self.model_dict["model"].clear_rope_cache()

        else:
            def step_model_input_kwargs(kwargs):
                if "src_image_mask" in kwargs and kwargs["src_image_mask"] is not None:
                    kwargs["src_image_mask"] = torch.cat([
                        kwargs["src_image_mask"],
                        torch.zeros([bsz, 1], dtype=torch.bool, device=self.device)
                    ], dim=1)
                if "und_image_masks" in kwargs and kwargs["und_image_masks"] is not None:
                    kwargs["und_image_masks"] = torch.cat([
                        kwargs["und_image_masks"],
                        torch.zeros([bsz, 1], dtype=torch.bool, device=self.device)
                    ], dim=1)
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    mask = kwargs["attention_mask"]
                    *prefix_shapes, qlen, klen = mask.shape
                    kwargs["attention_mask"] = torch.ones((*prefix_shapes, qlen + 1, klen + 1),
                                                          dtype=mask.dtype, device=mask.device).tril(diagonal=0)
                    kwargs["attention_mask"][:, :, :-1, :-1] = mask
                return kwargs

            stop_flag = torch.zeros([bsz], dtype=torch.bool, device=self.device)
            # Used for expert parallel: if stop_by_all_rank is True, the inference will stop only when all ranks finished. 
            stop_by_all_rank = kwargs.get("stop_by_all_rank", False)
            if stop_by_all_rank:
                is_stop_cur_rank = torch.tensor([0], dtype=torch.int32, device=self.device)
            
            for i in pbar:
                # NOTE: Filter out None values and empty dicts from model input kwargs, which is demanded by pure-torch parallel.
                model_input_kwargs = _check_model_input_kwargs(model_input_kwargs)
                next_token = self.get_next_token(
                    inputs,
                    input_pos=None,
                    generators=generators,
                    real_pos=real_pos - 1,
                    logits_processor=logits_processor,
                    do_sample=do_sample,
                    return_logits=return_first_pred_token_logits,
                    **model_input_kwargs,
                )
                if return_first_pred_token_logits:
                    next_token, logits = next_token
                next_token_ids = torch.cat([next_token_ids, next_token], dim=1)

                # Check if all workers finished.
                if self.args.eval_stop_by_dummy and is_dummy is not None:
                    is_all_dummy = torch.tensor(int(is_dummy.all()), device=self.device)
                    dist.all_reduce(is_all_dummy, op=dist.ReduceOp.SUM)
                    # self.logger.info(f"[rank{self.rank}] is_all_dummy: {is_all_dummy.item()}")
                    if is_all_dummy.item() == self.world_size:
                        finished_by_dummy = True
                        self.logger.info(f"All ranks have dummy batch, stop evaluation.")
                        break

                # Update stop flag and determine whether to break. 
                # When 'stop_by_all_rank' is True, we need to check if all ranks finished.
                for stop_token in stop_tokens:
                    stop_flag |= (next_token.squeeze(-1) == stop_token)
                
                if stop_by_all_rank:
                    if stop_flag.all() or i == infer_max_steps - 1:
                        is_stop_cur_rank = torch.tensor([1], dtype=torch.int32, device=self.device)
                    else:
                        is_stop_cur_rank = torch.tensor([0], dtype=torch.int32, device=self.device)
                    dist.all_reduce(is_stop_cur_rank, op=dist.ReduceOp.SUM)
                    if is_stop_cur_rank.item() == dist.get_world_size():
                        self.logger.info(f"Rank-{dist.get_rank()}, generated text length: {i}")
                        break
                elif stop_flag.all():
                    break
                    
                if return_first_pred_token_logits:
                    break
                # Update arguments for next iteration
                # Adding the pad token and then scatter is to ensure that each next_token is placed in the correct position, 
                # and to avoid potential issues caused by different ranks padding differently in the episode.
                pad_token = torch.tensor([self.tkwrapper.tokenizer.encode('<pad>')] * bsz, device=inputs.device, dtype=inputs.dtype)
                inputs = torch.cat([inputs, pad_token], dim=1)
                inputs.scatter_(dim=1, index=real_pos, src=next_token)
                real_pos += 1
                model_input_kwargs = step_model_input_kwargs(model_input_kwargs)

        # For MoE Expert Parallel stop criteria
        if self.args.eval_stop_by_dummy and is_dummy is not None and finished_by_dummy:
            out_dict["finished_by_dummy"] = True

        # For MMLU benchmark
        if return_first_pred_token_logits:
            out_dict["first_pred_token_logits"] = logits

        # Decode generated ids to text
        texts = []
        next_tokens = []
        for ids in next_token_ids:
            # stop_id_pos = torch.where(ids == self.tkwrapper.eos_token)[0]
            # Find first position where any stop token appears
            stop_mask = torch.zeros_like(ids, dtype=torch.bool)
            for stop_token in stop_tokens:
                stop_mask |= (ids == stop_token)
            stop_id_pos = torch.where(stop_mask)[0]
            if len(stop_id_pos) > 0:
                ids = ids[:stop_id_pos[0] + 1]
            next_tokens.append(ids)
            text = self.tkwrapper.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            texts.append(text)

        out_dict['samples'] = texts
        out_dict['next_tokens'] = next_tokens
        gen_time = time.time() - start_time
        if verbose == 1:
            loguru.logger.info(f"[dp_rank{self.rank}] Predict time: {gen_time:.2f}s")
        if verbose >= 2:
            loguru.logger.info(f"[dp_rank{self.rank}] Predict time: {gen_time:.2f}s. Predict texts: {texts}")
        return out_dict

    def batch_sample_general(
            self,
            dataloader,
            run_fn,
            input_batch_dict=None,
            run_fn_kwargs=None,
            save_dir=None,  # `save_dir` is used for saving text. image saving path is in `batch["save_path"]`
            rerank=1,
            **kwargs,
    ):
        infer_kwargs = self.get_infer_kwargs()
        for key, value in infer_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        # Rerank Prepare
        rerank_enabled = False
        if rerank > 1:
            rerank_enabled = True
            self.load_clip_score_model()
        else:
            self.rerank_clip_score_metric = None

        # Start sampling
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            batch: Dict[str, Any]
            self.logger.info(f"Batch {batch_index}/{total_batches}")
            # Adjust max_width according to your terminal width
            self.logger.info(f"\n{batch_data_repr(batch, max_width=150)}")

            # torch.cuda.empty_cache()
            inputs = {}
            if "type" in batch:
                inputs[batch["type"][0]] = batch["input"]
            if "seed" in batch:
                inputs["seed"] = batch["seed"]
            for k, v in default(input_batch_dict, {}).items():
                if v in batch:
                    inputs[k] = batch[v]

            run_fn_kwargs = default(run_fn_kwargs, {})

            # override run_fn_kwargs with inputs
            for k in inputs.keys():
                if k in run_fn_kwargs:
                    del run_fn_kwargs[k]

            if rerank_enabled:
                outputs = self.rerank_wrapper(
                    fn=run_fn, rerank=rerank, input_name="prompt", **inputs, output_type="pt", **run_fn_kwargs,
                )
            else:
                outputs = run_fn(**inputs, **run_fn_kwargs)
            final_samples = outputs["samples"]

            # Finish status determined by flag
            if "finished_by_dummy" in outputs and outputs["finished_by_dummy"]:
                break

            # Skip dummy batch processing
            if "is_dummy" in batch and batch["is_dummy"].all():
                assert len(batch["is_dummy"]) == 1, "Only support batch size 1 for dummy batch."
                self.logger.info(f"All dummy batch (batch_id={batch_index}), skip processing.")
                continue

            if isinstance(final_samples[0], Image.Image):
                save_names = [
                    batch["save_path"][i // self.args.num_sample_per_prompt].format(
                        i % self.args.num_sample_per_prompt
                    )
                    for i, sample in enumerate(final_samples)
                ]
                # Save prompts when prompts is not None
                self.save_batch_image(final_samples, save_names)

            # Save all types of results
            if isinstance(final_samples[0], str) or "cot_text_saved" in outputs:
                assert isinstance(save_dir, Path), "save_dir should be a Path object."
                text_data = final_samples if isinstance(final_samples[0], str) else outputs["cot_text_saved"]
                answers = [dict(
                    index=batch["id"][i].item() if isinstance(batch["id"][i], torch.Tensor) else batch["id"][i],
                    answer=ans,
                    **(dict(seed=inputs["seed"][i].item()) if "seed" in inputs else {}),
                    **(dict(question=inputs["prompt"][i]) if "prompt" in inputs else {}),
                ) for i, ans in enumerate(text_data)]
                self.save_batch_data(answers, save_dir / f"{self.rank}.csv")

    def process_src_image(self, src_pil_image, dataset_base_size=None, normalize=True, use_joint_image_feature=False):
        base_size = default(dataset_base_size, self.base_size)
        if isinstance(base_size, list):
            if len(base_size) == 1 or all(base_size[0] == size for size in base_size):
                base_size = base_size[0]
            else:
                raise ValueError(f"`base_size` for process_src_image should be a single value or all values are the same, but got {base_size}")
        reso_group = ResolutionGroup(base_size=base_size, extra_resolutions=self.args.get('extra_resolutions', None))
        ori_image_width = src_pil_image.width
        ori_image_height = src_pil_image.height
        image_width, image_height = reso_group.get_target_size(src_pil_image.width, src_pil_image.height)
        if self.args.infer_align_image_size_mode == "resize":
            # ArrowIndexV2.resize_and_crop uses LANCZOS to resize the image, so we use LANCZOS to resize the src image.
            resized_src_pil_image = src_pil_image.resize((image_width, image_height), resample=Image.Resampling.LANCZOS)
        elif self.args.infer_align_image_size_mode == "resize_and_pad":
            resized_src_pil_image, _ = ArrowIndexV2.resize_and_pad(src_pil_image, (image_width, image_height), pad_color=
            (0, 0, 0))
        else:
            resized_src_pil_image, _ = ArrowIndexV2.resize_and_crop(src_pil_image, (image_width, image_height), crop_type="center")
        resized_src_image_tensor = transforms.ToTensor()(resized_src_pil_image)
        if normalize:
            resized_src_image_tensor = self.vae_processor.normalize(resized_src_image_tensor)

        token_width = image_width // (self.vae_downsample_factor[1] * self.args.patch_size)
        token_height = image_height // (self.vae_downsample_factor[0] * self.args.patch_size)
        base_size, ratio_index = reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
        vae_image_info = ImageInfo(
            image_type="src_image",
            image_tensor=resized_src_image_tensor.unsqueeze(0),
            ori_image_width=ori_image_width, ori_image_height=ori_image_height,
            image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height,
            base_size=base_size, ratio_index=int(ratio_index),
        )

        if use_joint_image_feature:
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                inputs = self.vision_encoder_processor(src_pil_image)
                image = inputs["pixel_values"].squeeze(0)   # seq_len x dim
                pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)   # seq_len
                spatial_shapes = inputs["spatial_shapes"].squeeze(0)   # 2  (h, w)

                vision_encoder_kwargs = dict(
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                )

                vision_image_info = ImageInfo(
                    image_type="und_image",
                    image_tensor=image.unsqueeze(0),   # 1 x seq_len x dim
                    ori_image_width=ori_image_width, ori_image_height=ori_image_height,
                    image_width=spatial_shapes[1].item() * self.vision_encoder_w_factor,
                    image_height=spatial_shapes[0].item() * self.vision_encoder_h_factor,
                    token_width=spatial_shapes[1].item(),
                    token_height=spatial_shapes[0].item(),
                    image_token_length=self.vision_encoder_image_token_length,  # may not equal to token_width * token_height
                )
            else:
                vision_encoder_pil_image, _ = ArrowIndexV2.resize_and_pad(
                    src_pil_image, self.vision_encoder_image_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
                )
                vision_encoder_image_tensor = self.vision_encoder_transform(vision_encoder_pil_image)
                vision_encoder_image_tensor = vision_encoder_image_tensor.unsqueeze(0)

                vision_image_info = ImageInfo(
                    image_type="und_image",
                    image_tensor=vision_encoder_image_tensor,
                    ori_image_width=ori_image_width, ori_image_height=ori_image_height,
                    image_width=self.vision_encoder_image_size[1], image_height=self.vision_encoder_image_size[0],
                    token_width=self.vision_encoder_image_size[1] // self.vision_downsample_factor[1],
                    token_height=self.vision_encoder_image_size[0] // self.vision_downsample_factor[0],
                )

                vision_encoder_kwargs = {}
            return JointImageInfo(vae_image_info, vision_image_info, vision_encoder_kwargs)
        
        return vae_image_info

    def process_src_image_face(self, src_image, return_face=False):
        processed_image = self.model_dict["face_image_processor"].preprocess(src_image)
        src_face_embedding = self.model_dict["face_image_processor"].extract_face(
            self.model_dict["face_analysis"], processed_image, self.logger, return_face=return_face
        )
        face_image = None
        if return_face:
            src_face_embedding, face_image = src_face_embedding

        image_info = ImageInfo(
            image_type="face",
            image_tensor=src_face_embedding,
            image_token_length=self.args.face_id_embedding_qformer_token_length,
            face_image=face_image,
        )

        return image_info

    def parse_image_path(self, image_path):
        # It handles several cases:
        # 1. `image_path` is a cos path and contains special characters like '%20',
        #    which need to be unquoted.
        # 2. Replace the cos_base if necessary
        image_path = unquote(image_path)
        if self.cos_base is not None:
            for src_base, tgt_base in zip(self.cos_base_sources, self.cos_base_targets):
                if image_path.startswith(src_base):
                    image_path = image_path.replace(src_base, tgt_base)
                    break
        return image_path


def run_interactive(args, sampler: GeminiBetaSampler, logger):
    import datetime

    # ======== Check input format ========
    # 1. Must provide a valid JSON input file
    if not args.json_input or not Path(args.json_input).exists():
        raise ValueError(
            "Interactive mode requires a valid JSON input file. Please provide a valid path to --json-input."
        )
    with Path(args.json_input).open() as f:
        inputs = json.load(f)
    # 2. Must be a list of dict
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise ValueError(f"Invalid JSON input format: {type(inputs)}. It should be a list of dict.")
    # 3. Make saving directory
    save_dir = Path("__images/interactive")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ======== Prepare inputs for sampler ========
    kwargs = dict(
        sequence_template="pretrain",
        message_strategy="all",
        stream=args.stream,
        legacy_t2i=args.legacy_t2i,
        size=args.sample_image_size,
    )
    for out in sampler.generate(message_list=inputs, **kwargs):
        if isinstance(out["value"], str):
            print(out["value"], end="", flush=True)
        elif isinstance(out["value"], Image.Image):
            save_num = 0
            save_path = save_dir / f"{save_prefix}_{save_num}.png"
            while save_path.exists():
                save_num += 1
                save_path = save_dir / f"{save_prefix}_{save_num}.png"
            out["value"].save(save_path)
            print(f"Image saved to {save_path}")
        else:
            print(out)
    print()


def get_pil_image(image_path):
    from hymm.constants import ASSETS_BASE
    if ASSETS_BASE.startswith("/apdcephfs_wza"):
        image_path = image_path.replace("/apdcephfs_nj10/share_301739632", "/apdcephfs_wza")
    elif ASSETS_BASE.startswith("/apdcephfs_zwfy"):
        image_path = image_path.replace("/apdcephfs_nj10/share_301739632", "/apdcephfs_zwfy/share_303937731")
    pil_image = Image.open(image_path)
    return pil_image

def run_testset(args, sampler: GeminiBetaSampler, task, dataset, logger):
    use_joint_image_feature = args.get("use_joint_image_feature", False)
    logger.info(f"Running testset for {task=}, {dataset=}")

    # Batch sampling
    if task in ["t2i", "inpainting", "editing", "subject_driven", "face_id_embedding", "interleave", "pair"]:
        batch_negative_prompt_list = None

        segments = [f"{args.denoise_type}{args.diff_infer_steps}", f"cfg{args.guidance_scale}"]
        if task != "t2i" and args.src_guidance_scale > 1.0:
            segments.append(f"scfg{args.src_guidance_scale}")

        # column_dict: {batch key <- data column}                   used in dataset
        # input_batch_dict: {predict fn input key <- batch key}     used in batch_*() methods

        # only support add_image_shape_token=True
        assert args.add_image_shape_token, "add_image_shape_token should be True."

        if task == "t2i":
            def get_item_callback(item_dict, instruction_candidates=None, system_prompt_candidates=None, system_prompt_if_drop_think_candidates=None):
                if instruction_candidates is not None:
                    lang_cls = count_zh_en_words(item_dict["input"])
                    lang = "zh" if lang_cls['zh_ratio'] > lang_cls['en_ratio'] else "en"
                    instruction = instruction_candidates[lang][item_dict["seed"] % len(instruction_candidates[lang])]
                    item_dict["input"] = f'{instruction} {item_dict["input"]}'
                prompt_list = [item_dict["input"]]
                item_dict["prompt_list"] = prompt_list

                if system_prompt_candidates is not None:
                    system_prompt = system_prompt_candidates[item_dict["seed"] % len(system_prompt_candidates)]
                    item_dict["system_prompt"] = system_prompt.strip()
                
                if system_prompt_if_drop_think_candidates is not None:
                    system_prompt_if_drop_think = system_prompt_if_drop_think_candidates[item_dict["seed"] % len(system_prompt_if_drop_think_candidates)]
                    item_dict["system_prompt_if_drop_think"] = system_prompt_if_drop_think.strip()

                if item_dict["extra_width"] is not None and item_dict["extra_height"] is not None:
                    item_dict["sample_image_size"] = [item_dict["extra_height"], item_dict["extra_width"]]
                
                del item_dict["extra_width"]
                del item_dict["extra_height"]

                return item_dict

            if args.sequence_template == "instruct" and args.get('instruction_candidates_type', None) == 'fixed_set':
                get_item_callback = partial(get_item_callback, instruction_candidates=text2image_instructions)
            
            if args.sequence_template == "instruct" and args.get('system_prompt_candidates_type', 'off') != 'off':
                system_prompt_if_drop_think_candidates = None
                use_unified_system_prompt = args.get('use_unified_system_prompt', False)
                if use_unified_system_prompt:
                    system_prompt_candidates = unified_system_prompts["en_unified"]
                else:
                    pred_text_mode = args.get("t2i_pred_text_mode", "vanilla")
                    if pred_text_mode == "mixed":
                        system_prompt_candidates = t2i_system_prompts["en_think_recaption"]
                    elif pred_text_mode == "cot_recaption":
                        system_prompt_candidates = t2i_system_prompts["en_recaption"]
                    elif pred_text_mode == "vanilla":
                        system_prompt_candidates = t2i_system_prompts["en_vanilla"]
                    else:
                        raise NotImplementedError(f"Unknown pred_text_mode: {pred_text_mode}")

                    if pred_text_mode == "mixed" and args.drop_think:
                        system_prompt_if_drop_think_candidates = t2i_system_prompts["en_recaption"]
                
                get_item_callback = partial(get_item_callback, system_prompt_candidates=system_prompt_candidates, system_prompt_if_drop_think_candidates=system_prompt_if_drop_think_candidates)
            
            extra_cols = ["width", "height"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_system_prompt="system_prompt",
                batch_system_prompt_if_drop_think="system_prompt_if_drop_think",
                sample_image_size="sample_image_size",
            )
            custom_collate_fn = partial(except_collate_fn, except_keys=["prompt_list", "sample_image_size"])

            if args.neg_prompt is not None:
                batch_negative_prompt_list = args.neg_prompt

        elif task == "inpainting":
            def get_item_callback(item_dict, dataset_base_size=1024, instruction_candidates=None, system_prompt_candidates=None, use_joint_image_feature=False):
                if instruction_candidates is not None:
                    instruction = instruction_candidates[item_dict["seed"] % len(instruction_candidates)]
                    item_dict["input"] = f'{instruction} {item_dict["input"]}'
                prompt_list = [item_dict["input"]]
                item_dict["prompt_list"] = prompt_list

                if system_prompt_candidates is not None:
                    system_prompt = system_prompt_candidates[item_dict["seed"] % len(system_prompt_candidates)]
                    item_dict["system_prompt"] = system_prompt.strip()

                # 3 x h x w
                src_pil_image = get_pil_image(item_dict["extra_src_img_path"]).convert('RGB')
                # -1~1
                image_info = sampler.process_src_image(src_pil_image, dataset_base_size)
                # 1 x h x w
                mask_pil_image = get_pil_image(item_dict["extra_mask_img_path"])
                mask_pil_image = mask_pil_image.resize((image_info.image_width, image_info.image_height))
                # 0~1
                mask_image_tensor = transforms.ToTensor()(mask_pil_image).unsqueeze(0)
                masked_img_tensor = image_info.image_tensor * (1 - (mask_image_tensor > 0.5).float())
                image_info.image_tensor = masked_img_tensor

                if use_joint_image_feature:
                    masked_pil_image = transforms.Compose([
                        transforms.Normalize([-1], [2]),
                        transforms.ToPILImage(),
                    ])(masked_img_tensor.squeeze(0))
                    tmp_image_info = sampler.process_src_image(masked_pil_image, dataset_base_size,
                                                               use_joint_image_feature=True)
                    tmp_image_info.vae_image_info = image_info
                    image_info = tmp_image_info
                item_dict["image_info_list"] = [image_info]

                del item_dict["extra_src_img_path"]
                del item_dict["extra_mask_img_path"]

                return item_dict

            get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size,
                                        use_joint_image_feature=use_joint_image_feature)
            if args.sequence_template == "instruct" and args.get('instruction_candidates_type', 'fixed_set') == 'fixed_set':
                get_item_callback = partial(get_item_callback, instruction_candidates=inpainting_instructions)

            if args.sequence_template == "instruct" and args.get('system_prompt_candidates_type', 'off') != 'off':
                system_prompt_candidates = None
                use_unified_system_prompt = args.get('use_unified_system_prompt', False)
                if use_unified_system_prompt:
                    system_prompt_candidates = unified_system_prompts["en_unified"]
                get_item_callback = partial(get_item_callback, system_prompt_candidates=system_prompt_candidates)

            extra_cols = ["src_img_path", "mask_img_path"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_system_prompt="system_prompt",
            )
            if use_joint_image_feature:
                input_batch_dict["batch_joint_image_info_list"] = "image_info_list"
            else:
                input_batch_dict["batch_src_image_info_list"] = "image_info_list"
            custom_collate_fn = partial(except_collate_fn, except_keys=["image_info_list", "prompt_list"])

        elif task == "editing":
            def get_item_callback(item_dict, dataset_base_size=1024, instruction_candidates=None, 
                                  system_prompt_candidates=None, use_joint_image_feature=False):
                if instruction_candidates is not None:
                    instruction = instruction_candidates[item_dict["seed"] % len(instruction_candidates)]
                    item_dict["input"] = f'{instruction} {item_dict["input"]}'
                prompt_list = [item_dict["input"]]
                item_dict["prompt_list"] = prompt_list

                if system_prompt_candidates is not None:
                    system_prompt = system_prompt_candidates[item_dict["seed"] % len(system_prompt_candidates)]
                    item_dict["system_prompt"] = system_prompt.strip()

                # 3 x h x w
                pil_image = get_pil_image(item_dict["extra_src_img_path"]).convert('RGB')
                del item_dict["extra_src_img_path"]

                image_info = sampler.process_src_image(pil_image, dataset_base_size,
                                                       use_joint_image_feature=use_joint_image_feature)
                item_dict["image_info_list"] = [image_info]

                return item_dict

            get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size,
                                        use_joint_image_feature=use_joint_image_feature)

            if args.sequence_template == "instruct" and args.get('instruction_candidates_type', 'fixed_set') == 'fixed_set':
                instruction_candidates = editing_instructions
                if "style" in dataset or "controlnet" in dataset:  # already be instruct form
                    instruction_candidates = None
                get_item_callback = partial(get_item_callback, instruction_candidates=instruction_candidates)
            
            if args.sequence_template == "instruct" and args.get('system_prompt_candidates_type', 'off') != 'off':
                system_prompt_candidates = None
                use_unified_system_prompt = args.get('use_unified_system_prompt', False)
                if use_unified_system_prompt:
                    system_prompt_candidates = unified_system_prompts["en_unified"]
                get_item_callback = partial(get_item_callback, system_prompt_candidates=system_prompt_candidates)

            extra_cols = ["src_img_path"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_system_prompt="system_prompt",
            )
            if use_joint_image_feature:
                input_batch_dict["batch_joint_image_info_list"] = "image_info_list"
            else:
                input_batch_dict["batch_src_image_info_list"] = "image_info_list"
            custom_collate_fn = partial(except_collate_fn, except_keys=["image_info_list", "prompt_list"])

        elif task == "subject_driven":
            # ref_cols = ['count', 'src_img_path_1', 'src_img_path_2', 'src_img_path_3', 'src_img_path_4', 'src_img_path_5', 'src_img_path_6', 'src_img_path_7', 'src_img_path_8', 'src_img_path_9', 'src_img_path_10']
            # too many src images will cause out of memory error
            ref_cols = ['count', 'src_img_path', 'src_img_path_1', 'src_img_path_2', 'src_img_path_3']

            def get_item_callback(item_dict, dataset_base_size=1024, instruction_candidates=None, 
                                  system_prompt_candidates=None, use_joint_image_feature=False):
                if instruction_candidates is not None:
                    instruction = instruction_candidates[item_dict["seed"] % len(instruction_candidates)]
                    item_dict["input"] = f'{instruction} {item_dict["input"]}'
                prompt_list = [item_dict["input"]]
                item_dict["prompt_list"] = prompt_list

                if system_prompt_candidates is not None:
                    system_prompt = system_prompt_candidates[item_dict["seed"] % len(system_prompt_candidates)]
                    item_dict["system_prompt"] = system_prompt.strip()

                image_info_list = []
                count = item_dict['extra_count']
                is_editing_testset_flag = False
                if count is None:
                    count = 1
                    is_editing_testset_flag = True
                src_img_num = min(count, len(ref_cols) - 1)
                for i in range(1, src_img_num + 1):
                    if is_editing_testset_flag: # count and src_img_num must be 1
                        pil_image = get_pil_image(item_dict[f'extra_src_img_path']).convert('RGB')
                    else:
                        pil_image = get_pil_image(item_dict[f'extra_src_img_path_{i}']).convert('RGB')
                    image_info = sampler.process_src_image(pil_image, dataset_base_size,
                                                           use_joint_image_feature=use_joint_image_feature)
                    image_info_list.append(image_info)
                # delete to avoid collate batch
                for obj in ref_cols:
                    del item_dict[f'extra_{obj}']
                item_dict['image_info_list'] = image_info_list

                return item_dict

            get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size,
                                        use_joint_image_feature=use_joint_image_feature)
            if args.sequence_template == "instruct" and args.get('instruction_candidates_type', 'fixed_set') == 'fixed_set':
                instruction_candidates = subject_driven_instructions
                if "style" in dataset:  # already be instruction form
                    instruction_candidates = None
                get_item_callback = partial(get_item_callback, instruction_candidates=instruction_candidates)
            
            if args.sequence_template == "instruct" and args.get('system_prompt_candidates_type', 'off') != 'off':
                system_prompt_candidates = None
                use_unified_system_prompt = args.get('use_unified_system_prompt', False)
                if use_unified_system_prompt:
                    system_prompt_candidates = unified_system_prompts["en_unified"]
                get_item_callback = partial(get_item_callback, system_prompt_candidates=system_prompt_candidates)

            dataset_kwargs = {'extra_cols': ref_cols, 'get_item_callback': get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_system_prompt="system_prompt",
            )
            if use_joint_image_feature:
                input_batch_dict["batch_joint_image_info_list"] = "image_info_list"
            else:
                input_batch_dict["batch_src_image_info_list"] = "image_info_list"
            custom_collate_fn = partial(except_collate_fn, except_keys=["image_info_list", "prompt_list"])

        elif task == "interleave":
            extra_cols = ['prefill_template', 'interleave_data']

            def get_item_callback(item_dict, dataset_base_size=1024, align_rope=False):
                prompt_list = []
                src_image_info_list = []

                prefill_template = item_dict['extra_prefill_template']
                interleave_data = json.loads(item_dict['extra_interleave_data'])

                new_prefill_template = ""
                last_image_item_index = None
                for item_index, item_type in enumerate(prefill_template.split("-")):
                    if item_type == "text":
                        prompt_list.append(interleave_data[item_index]["data"])
                        new_prefill_template += "-text"
                    elif item_type == "image":
                        if last_image_item_index is None:
                            last_image_item_index = item_index
                            if align_rope:
                                pil_image = Image.open(interleave_data[item_index]["data"]).convert('RGB')
                                dummy_pil_image = Image.new("RGB", (pil_image.width, pil_image.height))
                                dummy_src_image_info = sampler.process_src_image(dummy_pil_image, dataset_base_size)
                                src_image_info_list.append(dummy_src_image_info)
                                new_prefill_template += "-src_image"
                        else:
                            pil_image = Image.open(interleave_data[last_image_item_index]["data"]).convert('RGB')
                            src_image_info = sampler.process_src_image(pil_image, dataset_base_size)
                            src_image_info_list.append(src_image_info)
                            new_prefill_template += "-src_image"

                            if align_rope:
                                pil_image = Image.open(interleave_data[item_index]["data"]).convert('RGB')
                                dummy_pil_image = Image.new("RGB", (pil_image.width, pil_image.height))
                                dummy_src_image_info = sampler.process_src_image(dummy_pil_image, dataset_base_size)
                                src_image_info_list.append(dummy_src_image_info)
                                new_prefill_template += "-src_image"

                            last_image_item_index = item_index

                if last_image_item_index is not None:
                    pil_image = Image.open(interleave_data[last_image_item_index]["data"]).convert('RGB')
                    src_image_info = sampler.process_src_image(pil_image, dataset_base_size)
                    src_image_info_list.append(src_image_info)
                    new_prefill_template += "-src_image"

                # delete to avoid collate batch
                for obj in extra_cols:
                    del item_dict[f'extra_{obj}']
                item_dict['prompt_list'] = prompt_list
                item_dict['src_image_info_list'] = src_image_info_list
                item_dict['prefill_template'] = new_prefill_template[1:]
                return item_dict

            get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size, align_rope=False)

            dataset_kwargs = {'extra_cols': extra_cols, 'get_item_callback': get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_src_image_info_list="src_image_info_list",
                batch_prefill_template="prefill_template",
            )
            custom_collate_fn = partial(except_collate_fn, except_keys=["src_image_info_list", "prompt_list"])

        elif task == "face_id_embedding":
            def get_item_callback(item_dict, dataset_base_size=1024, instruction_candidates=None):
                if instruction_candidates is not None:
                    instruction = instruction_candidates[item_dict["seed"] % len(instruction_candidates)]
                    item_dict["input"] = f'{instruction} {item_dict["input"]}'
                prompt_list = [item_dict["input"]]
                item_dict["prompt_list"] = prompt_list

                condition_type = args.get("face_id_embedding_condition_type", "face_embedding")
                if condition_type == "face_embedding":
                    condition_type_list = ["face"]
                elif condition_type == "face_embedding_src_image":
                    condition_type_list = ["face", "src_image"]
                else:
                    raise ValueError(f"Unknown condition type: {condition_type}")

                face_image_info_list = []
                src_image_info_list = []
                for condition_type_item in condition_type_list:
                    if condition_type_item == "face":
                        src_face_embedding = torch.load(item_dict["extra_src_face_embedding_path"])
                        if isinstance(src_face_embedding, np.ndarray):
                            src_face_embedding = torch.from_numpy(src_face_embedding)
                        if isinstance(src_face_embedding, torch.Tensor) and src_face_embedding.dim() == 1:
                            src_face_embedding = src_face_embedding.unsqueeze(0)
                        item_dict["src_face_embedding"] = src_face_embedding
                        # image_type name "face" should be same as tokenizer_wrapper.encode_general
                        face_image_info = ImageInfo(
                            image_type="face",
                            image_tensor=src_face_embedding,
                            image_token_length=args.face_id_embedding_qformer_token_length,
                        )
                        face_image_info_list.append(face_image_info)
                    elif condition_type_item == "src_image":
                        # 3 x h x w
                        pil_image = Image.open(item_dict["extra_src_img_path"])
                        # -1~1
                        src_image_info = sampler.process_src_image(pil_image, dataset_base_size)
                        src_image_info_list.append(src_image_info)

                item_dict["face_image_info_list"] = face_image_info_list
                item_dict["src_image_info_list"] = src_image_info_list
                del item_dict["extra_src_face_embedding_path"]
                return item_dict

            if args.sequence_template == "instruct" and args.face_id_embedding_task_kwargs.get(
                    'face_id_embedding_instruction_candidates_type', 'fixed_set') == 'fixed_set':
                instruction_candidates = face_id_instructions
                get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size,
                                            instruction_candidates=instruction_candidates)
            else:
                get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size)

            extra_cols = ["src_face_embedding_path", "src_img_path", ]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                batch_prompt_list="prompt_list",
                batch_face_image_info_list="face_image_info_list",
                batch_src_image_info_list="src_image_info_list",
            )
            custom_collate_fn = partial(except_collate_fn,
                                        except_keys=["face_image_info_list", "src_image_info_list", "prompt_list"])
            segments.append(f"face_guidance_scale{args.face_guidance_scale}")

        elif task == "pair":
            assert args.use_joint_image_feature

            def get_item_callback(item_dict, dataset_base_size=1024, system_prompt_candidates=None):
                if system_prompt_candidates is not None:
                    system_prompt = system_prompt_candidates[item_dict["seed"] % len(system_prompt_candidates)]
                    item_dict["system_prompt"] = system_prompt.strip()

                # Read from messages
                messages = json.loads(item_dict["extra_messages"])
                height = item_dict["extra_height"]
                width = item_dict["extra_width"]
                message_list = []
                for msg in messages:
                    if msg["type"] == "cond_image":
                        image_path = sampler.parse_image_path(msg["cache_image"])
                        pil_image = Image.open(image_path).convert('RGB')
                        image_info = sampler.process_src_image(pil_image, dataset_base_size, use_joint_image_feature=True)
                        message_list.append(dict(
                            role="user",
                            type="joint_image",
                            content=image_info,
                        ))
                    elif msg["type"] == "cond_text":
                        if msg["text"].startswith("{"):
                            text = json.loads(msg["text"])
                            if text["long_long_caption"].lower() in ["none", "无"]:
                                text = text["long_caption"]
                            else:
                                text = text["long_long_caption"]
                        else:
                            text = msg["text"]
                        message_list.append(dict(
                            role="user",
                            type="text",
                            content=text,
                        ))
                    elif msg["type"] == "gen_image":
                        message_list.append(dict(
                            role="assistant",
                            type="gen_image",
                            content=sampler.image_info_from_hw(input_height=height, input_width=width),
                        ))

                item_dict["message_list"] = message_list
                del item_dict["type"]
                del item_dict["input"]
                del item_dict["extra_messages"]
                del item_dict["extra_height"]
                del item_dict["extra_width"]
                return item_dict

            get_item_callback = partial(get_item_callback, dataset_base_size=args.training_image_size)

            if args.sequence_template == "instruct" and args.get('system_prompt_candidates_type', 'off') != 'off':
                system_prompt_candidates = None
                use_unified_system_prompt = args.get('use_unified_system_prompt', False)
                if use_unified_system_prompt:
                    system_prompt_candidates = unified_system_prompts["en_unified"]
                get_item_callback = partial(get_item_callback, system_prompt_candidates=system_prompt_candidates)

            extra_cols = ["messages", "height", "width"]
            dataset_kwargs = {"extra_cols": extra_cols, "callback": lambda x: x, "get_item_callback": get_item_callback}
            input_batch_dict = dict(
                batch_message_list="message_list",
                batch_system_prompt="system_prompt",
            )
            custom_collate_fn = partial(except_collate_fn, except_keys=["message_list"])

        else:
            raise NotImplementedError(f"Unknown task: {task}")

        save_base = sampler.get_sample_save_dir(
            task=task, testset=Path(dataset).stem, image_size=args.sample_image_size,
            segments=segments, rerank=args.rerank,
        )
        save_template = str(save_base / ("{}_{{}}" + f"{args.sample_save_file_suffix}.png"))
        logger.info(f"Save the generated images to: {save_base}")

        dataloader = sampler.build_sample_dataloader(
            data_source=dataset,
            save_template=save_template,
            batch_size=args.sample_batch_size,
            dataset_kwargs=dataset_kwargs,
            seed_type=args.seed_type,
            seed=args.seed,
            skip_exist=args.skip_exist,
            collate_fn=custom_collate_fn,
        )

        sampler.batch_sample_general(
            dataloader=dataloader,
            run_fn=partial(sampler.batch_x2image, task=task, verbose=args.verbose),
            input_batch_dict=input_batch_dict,
            save_dir=save_base,
            run_fn_kwargs=dict(
                sequence_template=args.sequence_template,
                predict_image_shape_token=args.predict_image_shape_token,
                sample_image_size=args.sample_image_size,
                batch_negative_prompt_list=batch_negative_prompt_list,
                drop_think=args.drop_think,
            ),
            rerank=args.rerank,
        )
        # Print again at final for easy reading
        logger.info(f"Save the generated images to: {save_base}")

    elif task == "mmu":
        infer_kwargs = sampler.get_infer_kwargs()
        top_p = infer_kwargs["top_p"]
        top_k = infer_kwargs["top_k"]
        temperature = infer_kwargs["temperature"]
        segments = [f"t{temperature}_tp{top_p}_tk{top_k}"]
        save_base = sampler.get_sample_save_dir(
            task=task, testset=Path(dataset).stem, segments=segments, subdir="mmu",
        )
        # detection tasks (with `_det_` in testset name) should keep the special tokens for specifying bounding boxes.
        skip_special_tokens = '_det_' not in Path(dataset).stem
        logger.info(f"Save the generated answers to: {save_base}")

        if args.get("use_joint_image_feature"):
            if args.mmu_prompt == "default_caption":
                uni_prompt = "Please describe the content of this image in detail."
            elif args.mmu_prompt == "default_ocr_det":
                uni_prompt = "Output the content in Latex format with two point boxes:"
            else:
                uni_prompt = None

            def get_item_callback(item_dict):
                pil_image = item_dict["image"]
                image_info = sampler.process_src_image(pil_image, args.sample_image_size, use_joint_image_feature=True)
                item_dict["image_info_list"] = [image_info]
                del item_dict["image"]
                return item_dict

            dataset = MMUDataset(dataset, prompt=uni_prompt, get_item_callback=get_item_callback,
                                 save_base=save_base, skip_exist=args.skip_exist)

            input_batch_dict = dict(
                prompt="prompt",
                batch_joint_image_info_list="image_info_list",
                is_dummy="is_dummy",
            )
            custom_collate_fn = partial(except_collate_fn, except_keys=["image_info_list"])
            kwargs = dict(verbose=3)

        else:
            arrow_dataset_kwargs = dict(
                arrow_file=dataset, seed_type=args.seed_type, seed=args.seed, save_template="{}",
                skip_exist=args.skip_exist, logger=logger,
            )
            column_dict = dict(image="image@bytes", prompt="prompt@default_const@Describe the image briefly.")
            image_processor = sampler.model_dict["image_processor"].preprocess
            dataset = ArrowDataset(column_dict=column_dict, image_processor=image_processor, **arrow_dataset_kwargs)
            input_batch_dict = dict(prompt="prompt", image="image")
            custom_collate_fn = None
            kwargs = dict()

        dataloader = sampler.build_sample_dataloader(
            data_source=dataset,
            batch_size=args.sample_batch_size,
            skip_exist=args.skip_exist,
            collate_fn=custom_collate_fn,
        )

        sampler.batch_sample_general(
            dataloader=dataloader,
            run_fn=sampler.batch_x2text,
            run_fn_kwargs=dict(
                xtype="mmu", verbose=args.verbose, skip_special_tokens=skip_special_tokens,
                max_new_tokens=args.max_new_tokens,
            ),
            input_batch_dict=input_batch_dict,
            save_dir=save_base,
            **kwargs,
        )
        # Print again at final for easy reading
        logger.info(f"Save the generated samples to: {save_base}")

    elif task == "lm":
        infer_kwargs = sampler.get_infer_kwargs()
        top_p = infer_kwargs["top_p"]
        top_k = infer_kwargs["top_k"]
        temperature = infer_kwargs["temperature"]
        segments = [f"t{temperature}_tp{top_p}_tk{top_k}"]
        save_base = sampler.get_sample_save_dir(
            task=task, testset=Path(dataset).stem, segments=segments, subdir="mmu",
        )
        logger.info(f"Save the generated answers to: {save_base}")

        dataset = SimpleDataset(dataset)

        input_batch_dict = dict(
            prompt="prompt",
            is_dummy="is_dummy",
        )
        kwargs = dict(verbose=3)

        dataloader = sampler.build_sample_dataloader(
            data_source=dataset,
            batch_size=args.sample_batch_size,
            skip_exist=args.skip_exist,
        )

        sampler.batch_sample_general(
            dataloader=dataloader,
            run_fn=sampler.batch_x2text,
            run_fn_kwargs=dict(
                xtype="lm", verbose=args.verbose, skip_special_tokens=True,
                max_new_tokens=args.max_new_tokens,
            ),
            input_batch_dict=input_batch_dict,
            save_dir=save_base,
            **kwargs,
        )
        # Print again at final for easy reading
        logger.info(f"Save the generated samples to: {save_base}")

    else:
        raise NotImplementedError(f"Unknown task: {task}")


class GeminiBetaSamplerLoaderHF:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._args = Namespace(
            ddp=False,
            num_nodes=1,
            node_index=0,
            deepspeed=False,
            tp_size=1,
            reproduce=False,
            global_seed=1,
            ckpt=self.model_path,
            task='sample',
            use_ptm=False,
            use_hf=True,
            ep_size=1,
            pp_size=1,
            pp_splits=None,
            launcher=1,
            puretorch_ckpt=None,
        )

    def load(self) -> Tuple[torch.nn.Module, GeminiBetaSampler]:
        world_size, rank, device = setup_distributed_initialize(self._args, mode=None)
        logger = rank0_logger(rank)

        sampler = GeminiBetaSampler.from_pretrained(
            ckpt_path=self._args.ckpt,
            rank=rank,
            world_size=world_size,
            device=device,
            logger=logger,
        )
        model = sampler.pipeline.model
        return model, sampler


def main():
    initial_args, mode = parse_eval_initial_args()
    finished = False
    if mode != 'ptm':
        world_size, rank, device = setup_distributed_initialize(initial_args, mode)
        logger = rank0_logger(rank)

        # huggingface deepspeed
        if mode == "deepspeed" and initial_args.use_hf:
            if dist.get_rank() % initial_args.pp_size != 0:
                loguru.logger.info(f"Skip rank {dist.get_rank()} for hf deepspeed mode.")
                finished = True

        if not finished:
            sampler = GeminiBetaSampler.from_pretrained(
                ckpt_path=initial_args.ckpt,
                rank=rank,
                world_size=world_size,
                device=device,
                logger=logger,
            )
            # Get updated args (include the yaml configs saved along with model checkpoint)
            args = sampler.args

            if mode == "pure_torch" and (initial_args.ep_size > 1 or initial_args.pp_size > 1):
                from hymm.parallelism.engines.gemini_parallel import GeminiParallelEngine
                model = sampler.pipeline.model
                engine_kwargs = dict(
                    model=model,
                    load_ckpt_path=initial_args.puretorch_ckpt if initial_args.puretorch_ckpt else None,
                    micro_batch_size=2,
                    pp_enable_autocast=True,
                    # pp_enable_autocast=False,
                    # weight_prec=self.args.precision,
                    weight_prec='bf16',
                    # cpu_offload=True,
                )
                if args.kv_cache:
                    if 'micro_batch_size' in engine_kwargs:
                        del engine_kwargs['micro_batch_size']
                    engine_kwargs['m_microbatch'] = 1
                model = GeminiParallelEngine(**engine_kwargs)
                model.eval()
                sampler.pipeline.model = model
                sampler.model_dict['model'] = model
                sampler.model = model
                model.reshard()
    # ptm inference
    else:
        # use extra_args to init megatron
        args = setup_ptm_initialize()
        logger = rank0_logger(torch.distributed.get_rank())
        sampler = GeminiBetaSampler.ptm_from_pretrained(args, logger)

    if not finished:
        if args.interactive:
            run_interactive(args, sampler, logger)
        elif args.testsets:
            for testset in args.testsets:
                dataset_name, task_name = testset.split("##")
                run_testset(args, sampler, task_name, dataset_name, logger)
        elif args.task != "sample":
            run_testset(args, sampler, args.task, args.csv, logger)
        else:
            raise ValueError(
                f"No tasks found. Please set valid --task ({args.task}) or --testsets ({args.testsets})."
            )

    loguru.logger.info(f"Rank {dist.get_rank()} finished.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
