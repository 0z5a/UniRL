import itertools
import json
import random
import numpy as np
from typing import Iterator, List, Tuple, Union
from functools import partial
from collections import defaultdict

import torch
from index_kits import ArrowIndexV2, MultiIndexV2, arrow_mapper
from loguru import logger as all_logger

from .caption_strategy import load_caption_processor
from .index_dataset import IndexDataset
from ..models.tokenizers import TokenizerWrapper
from ..models.tokenizers.conversation import get_conversation_template
from ..utils.helpers import default
from ..data_kits.system_prompt import t2i_system_prompts, vanilla_system_prompts


class TextArrowStream(IndexDataset):
    """
    Parameters
    ----------
    args
    index_file
    tokenizer_name
    t2t_text_token_length: int
        Deprecated, use `max_token_length` or `{dataset_tag}_token_length` instead.
    post_kwargs
    logger
    tokenizer
    index_kwargs
    dataset_type: str
        The dataset_tag.
    template: str
        `pretrain` or `instruct`.
    conv_format
    single_text_max_length: int
        If provided, it will be used to limit the maximum length of a single text to avoid encode timeout.
        If pre_extract_tokens is True, this value will be ignored.
    pre_extract_tokens: bool
        Whether to use pre-extracted tokens from the index. If True, the index-file should contain a `tokens` column.
    dummy_number
    task_kwargs
    token_length: int
        For backward compatibility.
    max_token_length: int
        Sequence max length, the value already +1 - dummy_number.
    """
    def __init__(
            self,
            args,
            index_file,
            tokenizer_name,
            task_kwargs=None,
            index_kwargs=None,
            t2t_text_token_length=None,
            post_kwargs=None,
            logger=None,
            tokenizer=None,
            dataset_type="lm",
            template="pretrain",
            conv_format="hunyuan-gemini-alpha",
            single_text_max_length=100000,
            pre_extract_tokens=False,
            dummy_number=0,
            token_length=None,
            max_token_length=None,
    ):
        self.dataset_tag = dataset_type

        self.task_kwargs = default(task_kwargs, {})
        self.index_kwargs = default(index_kwargs, {})

        self.task_kwargs = self.strip_leading_tag(self.task_kwargs, required=False)     # bc: required=False
        self.index_kwargs = self.strip_leading_tag(self.index_kwargs, required=False)   # bc: required=False

        index_file = self.index_kwargs.get("index_file", index_file)
        super().__init__(
            index_file,
            logger=logger,
        )
        self.logger.info(f"    (LM-{self.dataset_tag}) {task_kwargs=}")
        self.logger.info(f"    (LM-{self.dataset_tag}) {index_kwargs=}")

        if t2t_text_token_length is not None and max_token_length is not None:
            raise ValueError(
                "`t2t_text_token_length` and `max_token_length` cannot be set at the same time. `t2t_text_token_length` "
                "is deprecated, use `max_token_length` or `{dataset_tag}_token_length` instead."
            )

        self.args = args
        self.token_length = self.task_kwargs.get("token_length", token_length)
        self.dummy_number = dummy_number
        if t2t_text_token_length is not None:   # bc
            self.max_token_length = t2t_text_token_length
        else:
            self.max_token_length = default(max_token_length, self.token_length + 1 - self.dummy_number)
        self.logger.info(f"    (LM-{self.dataset_tag}) {self.dummy_number=}, {self.max_token_length=}")

        # Set a max length for single text to avoid tiktoken encode timeout.
        # For example, tiktoken encode a string like '-' * 500000 will timeout, and '-' * 1000000 will stackoverflow.
        self.single_text_max_length = self.task_kwargs.get("single_text_max_length", single_text_max_length)
        self.pre_extract_tokens = self.task_kwargs.get("pre_extract_tokens", pre_extract_tokens)
        # Whether to ignore the eos token when computing loss.
        self.open_recaption = self.task_kwargs.get("open_recaption", False)
        self.logger.info(f"    (LM-{self.dataset_tag}) {self.single_text_max_length=}, {self.pre_extract_tokens=}")

        self.system_prompt_candidates_type = self.task_kwargs.get("system_prompt_candidates_type", "off")
        assert self.system_prompt_candidates_type in {"fixed_set", "off"} or self.system_prompt_candidates_type.startswith("mix_up"), \
            f"Unsupported system_prompt_candidates_type: {self.system_prompt_candidates_type}"
        self.use_vanilla_system_prompt = self.task_kwargs.get("use_vanilla_system_prompt", False)

        if t2t_text_token_length is not None:   # bc, will be deprecated
            index_load_kwargs = dict(
                ceph_base=self.get_ceph_base(index_kwargs, key="lm_ceph_base"),
                ceph_base_inv=self.get_ceph_base_inv(index_kwargs, key="lm_ceph_base_inv"),
                verbose=index_kwargs.get("verbose", 0),
            )
            if (lm_shadow_cols := index_kwargs.get("lm_shadow_cols")) is not None:
                if isinstance(lm_shadow_cols, str):
                    lm_shadow_cols = [lm_shadow_cols]
                shadow_file_fn = {}
                for lm_shadow_col in lm_shadow_cols:
                    col, suffix = lm_shadow_col.split("@")
                    shadow_file_fn[col] = partial(arrow_mapper, suffix=suffix)
                index_load_kwargs["shadow_file_fn"] = shadow_file_fn

                self.logger.info(f"    (LM-{self.dataset_tag}) index load kwargs: {index_load_kwargs}")
                self.index_manager = self.load_index(**index_load_kwargs)
                self.logger.info(f"    (LM-{self.dataset_tag}) Using {self.index_manager}")
        else:
            # Prepare index manager
            self.setup_index_manager(f"LM-{self.dataset_tag}")

        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer: TokenizerWrapper = tokenizer
        self.eos_token = self.tokenizer.tokenizer.eos_token
        assert isinstance(self.eos_token, str), \
            f"eos_token should be a string, but got {self.eos_token}({type(self.eos_token)})"

        # Sequence pack related and affected
        self.sequence_pack = self.task_kwargs.get('sequence_pack', False)
        if self.sequence_pack:
            # if sequence pack is enabled, we will use the sequence-wise dummy_number to pad the packed sequence,
            # instead of dummy_number to pad the sample. `block_size` will be used as the maximum sequence length
            # for all the datasets.
            self.max_sequence_length = self.args.block_size + 1 - self.dummy_number

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

        # Image caption processor. Used for constructing recaption tasks.
        if lm_caption_processor := args.get('lm_caption_processor'):
            self.caption_processor = load_caption_processor(
                name=lm_caption_processor,
                caption_sample_ratio=json.loads(args.lm_caption_sample_ratio),
                logger=self.logger,
                kwargs=args.get('lm_caption_processor_kwargs'),
            )
        else:
            self.caption_processor = None

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        # After init, we set the logger to all_logger to print warnings and errors of all ranks
        self.logger = all_logger

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

    def get_pretrain_text(self, ind) -> str:
        columns = self.index_manager.get_columns(ind)
        if "caption_v2" in columns:
            try:
                caption = self.index_manager.get_attribute(ind, "caption_v2")
                text = ''.join(self.caption_processor.caption_aug(caption))
            except Exception as e:
                self.handle_exception_message(self.get_pretrain_text, e)
                text = ""
        elif "dataset_tag" in columns:
            # Instruct text data used for pretrain: concat User, Assistant, and System messages.
            messages, system_prompt = [], None
            for _ in range(5):
                try:
                    messages = self.index_manager.get_attribute(ind, "conversations")
                    system_prompt = self.index_manager.get_attribute(ind, "document") if "document" in columns else None
                    assert len(messages) > 0
                    break
                except Exception as e:
                    self.handle_exception_message(self.get_pretrain_text, e)
                    new_ind = self.index_manager.random_dindex(ind)
                    print(f"Error with index={ind}, trying new index={new_ind}")
                    ind = new_ind

            texts = []
            if system_prompt is not None:
                texts.append(system_prompt)
            for msg in messages:
                texts.append(msg["User"].strip())
                if "reasoning" in msg and msg["reasoning"] is not None and msg["reasoning"].strip():
                    texts.append(msg["reasoning"].strip())
                texts.append(msg["Assistant"].strip())
            text = '\n\n'.join(texts)
        else:
            # Pretrain text data
            text = self.index_manager.get_attribute(ind, "text")
            text = str(text).strip()

        if self.single_text_max_length > 0:
            text = text[:self.single_text_max_length]
        return text

    def format_message_list(self, message_list, return_type="list", system_prompt=None):
        conversation = get_conversation_template(self.conv_format)
        conversation.system_message = "" if system_prompt is None else system_prompt
        for msg in message_list:
            # User message
            conversation.add_message(conversation.roles[0], msg["User"].strip())
            # Assistant message, reasoning msg.
            if self.open_recaption:
                # User: prompt Assistant: Optional(<think>xx</think>)Optional(<recaption>yy</recaption>)<answer>xx<boi>[image]</boi></answer>
                # Since we are doing t2t, we will stop at <answer><boi>
                image_recaption = msg["Assistant"].strip() # User: prompt, Reasoing: think_cot, Assistant: caption
                if not image_recaption.startswith('<recaption>'):
                    image_recaption = '<recaption>' + image_recaption + '</recaption>'
                assistant_msg = image_recaption + '<answer>'
            else:
                assistant_msg = '<answer>' + msg["Assistant"].strip() + '</answer>'
            if 'reasoning' in msg and msg["reasoning"] is not None and msg["reasoning"].strip():
                assistant_msg = '<think>' + msg["reasoning"].strip() + '</think>' + assistant_msg
            conversation.add_message(conversation.roles[1], assistant_msg)
        text = conversation.get_prompt(return_type=return_type, add_system=system_prompt is not None)
        return text, conversation

    def get_instruct_text(self, ind) -> Tuple[List[str], List[Tuple[bool, int]]]:
        messages, system_prompt = None, None
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

        if self.use_vanilla_system_prompt and system_prompt is None:
            system_prompt = self.get_system_prompt(vanilla_system_prompts["en"])
        # recaption
        elif self.open_recaption and system_prompt is None:
            if 'reasoning' in messages[0] and messages[0]["reasoning"] is not None and messages[0]["reasoning"].strip():
                system_prompt = self.get_system_prompt(t2i_system_prompts["en_think_recaption"])
            else:
                system_prompt = self.get_system_prompt(t2i_system_prompts["en_recaption"])

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

    def get_tokens(self, ind):
        tokens = self.index_manager.get_attribute(ind, "tokens")
        return tokens

    def get_batch_tokens(self, indices):
        if self.template == "pretrain":
            tokens_list = [self.get_tokens(ind) for ind in indices]
            ignore_list = [False] * len(tokens_list)
            sections = []
            for tokens, ignore in zip(tokens_list, ignore_list):
                sections.extend([
                    dict(type='text', tokens=tokens, ignore=ignore),
                    dict(type='text', tokens=[self.tokenizer.eos_token], ignore=False),
                ])
        else:
            raise NotImplementedError(f"Unsupported template: {self.template}")
        return sections

    def preprocess_text(self, text):
        text = str(text).strip()
        if self.single_text_max_length > 0:
            text = text[:self.single_text_max_length]
        return text.strip()

    def get_pretrain_text_std(self, ind):
        columns = self.index_manager.get_columns(ind)
        sections = []
        if "dataset_tag" in columns:
            assert not self.pre_extract_tokens, "pre_extract_tokens is not supported for data with instruct template."
            # Instruct text data used for pretrain: concat User, Assistant, and System messages.
            messages, system_prompt = [], None
            for _ in range(5):
                try:
                    messages = self.index_manager.get_attribute(ind, "conversations")
                    system_prompt = self.index_manager.get_attribute(ind, "document") if "document" in columns else None
                    assert len(messages) > 0
                    break
                except Exception as e:
                    self.handle_exception_message(self.get_pretrain_text, e)
                    new_ind = self.index_manager.random_dindex(ind)
                    print(f"Error with index={ind}, trying new index={new_ind}")
                    ind = new_ind

            if system_prompt is not None:
                system_prompt = self.preprocess_text(system_prompt)
                sections.append(dict(type='text', text=system_prompt, ignore=True))

            for msg in messages:
                # user message
                user_msg = self.preprocess_text(msg["User"])
                sections.append(dict(type='text', text=user_msg, ignore=True))
                # reasoning
                if "reasoning" in msg and msg["reasoning"] is not None and msg["reasoning"].strip():
                    reasoning = self.preprocess_text(msg["reasoning"])
                    sections.append(dict(type='text', text=reasoning))
                # bot message
                bot_msg = self.preprocess_text(msg["Assistant"])
                sections.append(dict(type='text', text=bot_msg))
        else:
            # Pretrain text data
            if self.pre_extract_tokens:
                tokens = self.get_tokens(ind)
                sections.append(dict(type='text', tokens=tokens))
            else:
                text = self.index_manager.get_attribute(ind, "text")
                text = self.preprocess_text(text)
                sections.append(dict(type='text', text=text))
        return sections

    def get_instruct_text_std(self, ind):
        assert not self.pre_extract_tokens, "pre_extract_tokens is not supported for data with instruct template."
        messages, system_prompt = None, None
        sections = []
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

        if self.use_vanilla_system_prompt and system_prompt is None:
            system_prompt = self.get_system_prompt(vanilla_system_prompts["en"])
        # recaption
        if self.open_recaption and system_prompt is None:
            if 'reasoning' in messages[0] and messages[0]["reasoning"] is not None and messages[0]["reasoning"].strip():
                system_prompt = self.get_system_prompt(t2i_system_prompts["en_think_recaption"])
            else:
                system_prompt = self.get_system_prompt(t2i_system_prompts["en_recaption"])

        if system_prompt is not None:
            system_prompt = self.preprocess_text(system_prompt)
            sections.extend([
                dict(type='text', text=system_prompt.strip("\n"), ignore=True),
                dict(type='text', text=self.default_conv.sep, ignore=True),
            ])
        for msg in messages:
            # user message
            user_msg = self.preprocess_text(msg["User"])
            sections.extend([
                dict(type='text', text=f"{self.default_conv.roles[0]}: ", ignore=True),
                dict(type='text', text=user_msg, ignore=True),
                dict(type='text', text=self.default_conv.sep, ignore=True),
            ])
            # bot prefix
            sections.extend([
                dict(type='text', text=f"{self.default_conv.roles[1]}: ", ignore=True),
            ])
            # reasoning
            if 'reasoning' in msg and msg["reasoning"] is not None and msg["reasoning"].strip():
                reasoning = self.preprocess_text(msg["reasoning"])
                if reasoning.startswith('<think>'):
                    reasoning = reasoning[len('<think>'):]
                if reasoning.endswith('</think>'):
                    reasoning = reasoning[:-len('</think>')]
                reasoning = self.preprocess_text(reasoning)
                sections.extend([
                    dict(type='text', text="<think>", ignore=True),
                    dict(type='text', text=reasoning),
                    dict(type='text', text="</think>"),
                ])
            # recaption
            if self.open_recaption:
                recap = self.preprocess_text(msg["Assistant"].strip())
                if recap.startswith('<recaption>'):
                    recap = recap[len('<recaption>'):]
                if recap.endswith('</recaption>'):
                    recap = recap[:-len('<recaption>')]
                recap = self.preprocess_text(recap)
                sections.extend([
                    dict(type='text', text="<recaption>", ignore=True),
                    dict(type='text', text=recap),
                    dict(type='text', text="</recaption>"),
                ])
            # bot answer
            bot_msg = self.preprocess_text(msg["Assistant"])
            sections.append(dict(type='text', text=bot_msg))
        return sections

    def get_text_data(self, index):
        if self.template == "pretrain":
            sections = self.get_pretrain_text_std(index)
        elif self.template == "instruct":
            sections = self.get_instruct_text_std(index)
        else:
            raise NotImplementedError(f"Unsupported template: {self.template}")
        return sections

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            indices = [indices]

        if self.sequence_pack:
            assert len(indices) == 1, f"Only one index is supported for sequence packing"
            index = indices[0]
            sections = self.get_text_data(index)
            max_token_length = self.max_sequence_length if self.sequence_pack else self.max_token_length
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=max_token_length,
                add_pad=False if self.sequence_pack else 'auto',
            )
            tokens = output.tokens
            text_mask = output.text_mask
        elif self.pre_extract_tokens:
            sections = self.get_batch_tokens(indices)
            output = self.tokenizer.encode_general(
                sections=sections,
                max_token_length=self.max_token_length,
            )
            tokens = output.tokens
            text_mask = output.text_mask
        else:
            texts, text_mask_sections = self.get_text(indices)
            tokens, _, text_mask = self.tokenizer.encode_lm(
                *texts,
                max_token_length=self.max_token_length,
                return_text_mask=True,
                text_mask_sections=text_mask_sections,
            )

        target_tokens = tokens.clone()
        # If open_recaption is True, we don't need to compute loss on the eos token.
        if self.open_recaption:
            text_mask[target_tokens == self.tokenizer.eos_token] = 0.0
        target_tokens[text_mask == 0.0] = -100

        ret = {
            "data_type": "lm",
            "dtype": self.dataset_tag,
            "n_samples": len(indices),
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
        }
        return ret

    def collate_fn(self, batch):
        if self.sequence_pack:
            return batch

        data_type = [item["data_type"] for item in batch]
        dtype = [item["dtype"] for item in batch]
        n_samples = torch.tensor([item["n_samples"] for item in batch])

        tokens = torch.stack([item["tokens"] for item in batch])
        target_tokens = torch.stack([item["target_tokens"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])

        ret = {
            "data_type": data_type,
            "dtype": dtype,
            "n_samples": n_samples,
            "tokens": tokens,
            "target_tokens": target_tokens,
            "text_mask": text_mask,
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
                new_item[key] = [value]  # noqa
            # Accumulated values
            elif key in {"n_samples"}:
                new_item[key] = torch.tensor([sum(item[key] for item in items)])  # noqa
            # ================ Sequence & no positional keys ================
            elif key in seq_pad_value:
                cat_list = [item[key] for item in items]
                pad_length = max_length - sum(len(t) for t in cat_list)
                new_item[key] = torch.cat(  # noqa
                    cat_list + [torch.full((pad_length,), seq_pad_value[key], dtype=cat_list[0].dtype)]
                )[None]
            else:
                raise ValueError(f"Unsupported key: {key}")

        return new_item


class MaxLengthBatchSampler(object):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        index_manager (ArrowIndexV2 or List[ArrowIndexV2]): Index manager.
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
    """

    def __init__(self, index_manager: ArrowIndexV2, sampler, batch_size, max_length, length_getter) -> None:
        if not isinstance(index_manager, (ArrowIndexV2, MultiIndexV2)):
            raise ValueError(f"index_manager should be an instance of ArrowIndexV2 or MultiIndexV2, "
                             f"but got {type(index_manager)}.")

        self.index_manager = index_manager
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_length = max_length
        self.length_getter = length_getter

        # Unique iter
        self._unique_iter = False

    def __iter__(self) -> Iterator[List[List[int]]]:
        prefix = f"[dataset rank{self.sampler.rank}] " if hasattr(self.sampler, 'rank') else ""
        print(f"{prefix}Iterator for MaxLengthBatchSampler created.")
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        sampler_iter = iter(self.sampler)
        should_stop = False
        batch_indices = []
        while True:
            indices = []
            cum_length = -1
            while True:
                try:
                    ind = next(sampler_iter)

                    # Get token length of the sample with the index.
                    length = self.length_getter(self.index_manager, ind)
                    indices.append(ind)
                    cum_length += length + 1
                    if cum_length >= self.max_length:
                        break
                except StopIteration:
                    should_stop = True
                    break

            if len(indices) == 0 or should_stop:
                break

            batch_indices.append(indices)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                # Important: Don't use batch_indices.clear() here, it will cause the generator to yield empty list
                batch_indices = []

    def __len__(self) -> int:
        return len(self.index_manager)
