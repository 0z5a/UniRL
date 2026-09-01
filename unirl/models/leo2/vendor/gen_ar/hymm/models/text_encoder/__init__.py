from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.utils import ModelOutput

from ...core.global_vars import get_parallel_state
from ...constants import TEXT_ENCODER_PATH, TEXT_ENCODER_TOKENIZER_PATH
from ...utils.torch_utils import PRECISION_TO_TYPE


def use_default(value, default):
    return value if value is not None else default


def build_qwenvl_mm_token_type_ids(
    tokens: torch.Tensor,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> torch.Tensor:
    """Build Qwen modality IDs: text=0, image=1, video=2."""

    mm_token_type_ids = torch.zeros_like(tokens, dtype=torch.int32)
    if image_token_id is not None:
        mm_token_type_ids.masked_fill_(tokens == image_token_id, 1)
    if video_token_id is not None:
        mm_token_type_ids.masked_fill_(tokens == video_token_id, 2)
    return mm_token_type_ids


def _text_encoder_backbone_layers(backbone: nn.Module):
    """Return transformer layers for FSDP sharding (Qwen-VL uses language_model.layers; text-only Qwen uses .layers)."""
    if hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        return backbone.language_model.layers
    if hasattr(backbone, "layers"):
        return backbone.layers
    raise ValueError(
        f"Cannot find transformer layers on text encoder backbone {type(backbone).__name__} for FSDP."
    )


@dataclass
class _Qwen3OmniThinkerEncoderOutput(ModelOutput):
    """Aligns Qwen3-Omni Thinker outputs with TextEncoderWrapper (expects last_hidden_state / optional hidden_states)."""

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class _Qwen3OmniThinkerEncoderBridge(nn.Module):
    """
    Qwen3-Omni Thinker 的 forward 返回 CausalLM 风格输出，不含 ``last_hidden_state``。
    用 ``thinker.model`` 的 forward hook 取出末层隐状态；Thinker 文本骨干当前难以透出完整逐层
    ``hidden_states``，因此仅保证 ``hidden_state_skip_layer`` 为 ``None`` 或 ``0`` 时的行为正确。
    """

    def __init__(self, thinker: nn.Module):
        super().__init__()
        self.thinker = thinker
        self.layers = thinker.model.layers
        self.final_layer_norm = thinker.model.norm

    @property
    def dtype(self):
        return self.thinker.dtype

    @property
    def device(self):
        return self.thinker.device

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        captured: list = []

        def _hook(_mod, _inp, out):
            lh = getattr(out, "last_hidden_state", None)
            if lh is not None:
                captured.append(lh)

        hook_handle = self.thinker.model.register_forward_hook(_hook)
        try:
            self.thinker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                use_cache=False,
                **kwargs,
            )
        finally:
            hook_handle.remove()

        if not captured:
            raise RuntimeError(
                "Qwen3-Omni thinker.model did not return last_hidden_state; check transformers / checkpoint."
            )
        last_hs = captured[-1]
        if output_hidden_states:
            return _Qwen3OmniThinkerEncoderOutput(
                last_hidden_state=last_hs,
                hidden_states=(last_hs,),
            )
        return _Qwen3OmniThinkerEncoderOutput(last_hidden_state=last_hs)


class TextEncoderWrapper(nn.Module):
    """
    Wrap the text encoder to return a clean instance, avoiding conflicts with FSDP.
    """
    def __init__(self, text_encoder, apply_final_norm, output_key="last_hidden_state"):
        super().__init__()
        self.text_encoder = text_encoder
        self.apply_final_norm = apply_final_norm
        self.output_key = output_key

    def forward(self, *args, hidden_state_skip_layer=None, **kwargs):
        outputs = self.text_encoder(*args, **kwargs)
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.text_encoder.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]
        return last_hidden_state, outputs


def load_text_encoder(text_encoder_type,
                      text_encoder_precision=None,
                      text_encoder_path=None,
                      use_fsdp=False,
                      logger=None,
                      device=None,
                      attn_implementation=None,
                      ):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")
    if attn_implementation is None:
        attn_implementation = "flash_attention_2"
    if logger is not None:
        logger.info(f"Text encoder attn_implementation: {attn_implementation}")

    if text_encoder_type in ["t5", "t5_v11_xxl"]:
        from transformers import T5EncoderModel
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.encoder.final_layer_norm
    elif text_encoder_type == "clipL":
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llava-llama-3-8b":
        from transformers import LlamaModel
        text_encoder = LlamaModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "glm-4v-9b":
        text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True, trust_remote_code=True)
    elif text_encoder_type in ["qwen-2.5-vl-32b-instruct", "qwen-2.5-vl-72b-instruct"]:
        from transformers import Qwen2_5_VLModel
        text_encoder = Qwen2_5_VLModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
    elif text_encoder_type in ["qwen-2.5-vl-7b-instruct"]:
        text_encoder = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
        if hasattr(text_encoder, "language_model"):
            text_encoder = text_encoder.language_model
    elif text_encoder_type in ["qwen-3-vl-8b-instruct", "qwen-3vl-8b"]:
        from transformers import Qwen3VLForConditionalGeneration
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_path, attn_implementation=attn_implementation, low_cpu_mem_usage=True, dtype=torch.bfloat16,
        )
        text_encoder = text_encoder.model
    elif text_encoder_type in ["qwen-3.5-9b", "qwen-3.5-35-a3b"]:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as err:
            raise ImportError(
                "qwen-3.5-9b / qwen-3.5-35-a3b 需要 transformers 提供 AutoModelForImageTextToText（建议 >=4.57）。"
            ) from err
        _full = AutoModelForImageTextToText.from_pretrained(
            text_encoder_path,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        text_encoder = getattr(_full, "model", _full)

        # Apply packed sequence patch if enabled via args
        if text_encoder_type == "qwen-3.5-9b":
            from .patch_qwen3_5_packed import maybe_apply_patch
            maybe_apply_patch(logger=logger)
    elif text_encoder_type == "qwen-3-omni-30-a3b":
        try:
            from transformers import AutoModelForTextToWaveform
        except ImportError as err:
            raise ImportError(
                "qwen-3-omni-30-a3b 需要 transformers 提供 AutoModelForTextToWaveform（建议 >=4.57）。"
            ) from err
        _load_kw = dict(
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        try:
            _full = AutoModelForTextToWaveform.from_pretrained(
                text_encoder_path,
                attn_implementation=attn_implementation,
                **_load_kw,
            )
        except (ImportError, ValueError, OSError):
            _full = AutoModelForTextToWaveform.from_pretrained(text_encoder_path, **_load_kw)
        if not hasattr(_full, "thinker"):
            raise ValueError(
                f"AutoModelForTextToWaveform loaded from {text_encoder_path} has no `.thinker`; "
                "expected Qwen3-Omni style checkpoint."
            )
        text_encoder = _Qwen3OmniThinkerEncoderBridge(_full.thinker)
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(tokenizer_type,
                   tokenizer_path=None,
                   padding_side="right",
                   logger=None
                   ):
    if tokenizer_path is None:
        tokenizer_path = TEXT_ENCODER_TOKENIZER_PATH.get(tokenizer_type, TEXT_ENCODER_PATH[tokenizer_type])
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type in ["t5", "t5_v11_xxl"]:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    elif tokenizer_type == "clipL":
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llava-llama-3-8b":
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, padding_side=padding_side)
    elif tokenizer_type in ["qwen-3.5-9b", "qwen-3.5-35-a3b", "qwen-3-omni-30-a3b"]:
        from transformers import AutoProcessor
        _processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenizer = getattr(_processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                f"AutoProcessor loaded from {tokenizer_path} has no `.tokenizer`; cannot use as text tokenizer."
            )
        tokenizer.padding_side = padding_side
    elif tokenizer_type in [
        "glm-4v-9b",
        "qwen-2.5-vl-7b-instruct", "qwen-2.5-vl-32b-instruct", "qwen-2.5-vl-72b-instruct",
        "qwen-3-vl-8b-instruct", "qwen-3vl-8b",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(self,
                 text_encoder_type: str,
                 max_length: int,
                 text_encoder_precision: Optional[str] = None,
                 text_encoder_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 output_key: Optional[str] = None,
                 use_attention_mask: bool = True,
                 input_max_length: Optional[int] = None,
                 prompt_template: Optional[dict] = None,
                 hidden_state_skip_layer: Optional[int] = None,
                 apply_final_norm: bool = False,
                 reproduce: bool = False,
                 uncond_token: Optional[str] = None,
                 logger=None,
                 device=None,
                 no_encoder=False,
                 use_fsdp=False,
                 fsdp_overlap=False,
                 fsdp_mesh=None,
                 attn_implementation=None,
                 ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.attn_implementation = attn_implementation
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template = prompt_template
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger
        self.uncond_token = uncond_token

        self.use_template = self.prompt_template is not None
        if self.use_template and isinstance(self.prompt_template, dict):
            assert "template" in self.prompt_template, (
                f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            )
            assert '{}' in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.output_key = output_key
        if self.output_key is None:
            if text_encoder_type.startswith(("t5", "llama", "glm", "qwen")):
                self.output_key = "last_hidden_state"
            elif "clip" in text_encoder_type:
                self.output_key = "pooler_output"
            else:
                raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        if not no_encoder:
            text_encoder, self.model_path = load_text_encoder(
                text_encoder_type=self.text_encoder_type,
                text_encoder_precision=self.precision,
                text_encoder_path=self.model_path,
                use_fsdp=use_fsdp,
                logger=self.logger,
                device=device,
                attn_implementation=self.attn_implementation,
            )
            text_encoder = TextEncoderWrapper(
                text_encoder,
                self.apply_final_norm,
                output_key=self.output_key,
            )

            if use_fsdp:
                assert text_encoder_precision is not None, "text_encoder_precision must be specified when use_fsdp is True"
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=PRECISION_TO_TYPE[text_encoder_precision], reduce_dtype=torch.bfloat16,
                    cast_forward_inputs=False,
                )

                if fsdp_mesh is not None:
                    mesh = fsdp_mesh
                else:
                    p_state = get_parallel_state()
                    assert p_state.backend == "pure_torch", "FSDP is only supported in pure_torch backend."
                    mesh = p_state.backend_state.default_fsdp_mesh

                fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}

                layers = list(_text_encoder_backbone_layers(text_encoder.text_encoder))
                for layer in layers:
                    fully_shard(layer, **fsdp_config)

                if fsdp_overlap:
                    for cur, nxt in zip(layers[:-1], layers[1:]):
                        cur.set_modules_to_forward_prefetch([nxt])

                text_encoder = fully_shard(text_encoder, **fsdp_config)

            self.model = text_encoder
            self.dtype = self.model.text_encoder.dtype
            self.device = self.model.text_encoder.device

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @property
    def supports_packed_sequence(self):
        """Whether this text encoder supports cu_seqlens packed sequence encoding."""
        from .patch_qwen3_5_packed import is_packed_sequence_patched
        # Only dense Qwen3.5 is supported now
        return self.text_encoder_type == "qwen-3.5-9b" and is_packed_sequence_patched()

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llava-llama-3-8b.
            return template.format(text)
        elif isinstance(template, list):
            # Will send chat conversation to tokenizer. Used for glm-4v-9b.
            conversation = deepcopy(template)
            for message_id in range(len(conversation)):
                if '{}' in conversation[message_id]["content"]:
                    filled_text = conversation[message_id]["content"].format(text)
                    if prevent_empty_text and len(filled_text) == 0:
                        filled_text = ' '
                    conversation[message_id]["content"] = filled_text
                    # We assume there is only one placeholder in each conversation.
                    break
            return conversation
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input):
        """
        Automatically calculate the crop_start position based on identifying user tokens.

        Args:
            tokenized_input: The output from the tokenizer containing input_ids

        Returns:
            int: The position where the actual prompt content begins (after user markers)
        """
        input_ids = tokenized_input["input_ids"][0].tolist()  # Get the first example's tokens

        # Common user markers in different LLM tokenizers
        user_markers = {
            "llava": "<|start_header_id|>user<|end_header_id|>\n\n",  # LLaVA models
            "qwen": "<|im_start|>user\n",  # Qwen models
        }

        # Get relevant marker based on model type
        marker = None
        for model_type, token_marker in user_markers.items():
            if model_type in self.text_encoder_type.lower():
                marker = token_marker
                break

        if marker is None:
            # Default fallback: return 0 (no cropping) if we can't identify the model type
            return 0

        # Tokenize just the marker to get its token IDs
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]

        # Find the end position of the marker in the input sequence
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i:i + len(marker_tokens)] == marker_tokens:
                # Return the position after the marker
                # print(f"crop_start: {i + len(marker_tokens)}, {self.tokenizer.decode(tokenized_input["input_ids"][0][i:i+len(marker_tokens)+10])}") # check crop_start
                return i + len(marker_tokens)

        # If marker not found, try to find based on special tokens
        if hasattr(self.tokenizer, 'special_tokens_map'):
            # Check for user token or any other special token that might indicate user input start
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if 'user' in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1

        # Default fallback: return 0 (no cropping)
        return 0

    def safe_tokenizer(self, text, **kwargs):
        kwargs = {**dict(
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        ), **kwargs}
        try:
            return self.tokenizer(text, **kwargs)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            if isinstance(text, (list, tuple)):
                text = [t.encode('utf-8', errors='ignore').decode() for t in text]
            elif isinstance(text, str):
                text = text.encode('utf-8', errors='ignore').decode()
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
            print(f"fixed text: {text}")
            return self.tokenizer(text, **kwargs)

    def text2tokens(self, text, padding="max_length", max_length=None):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
            padding (str): Padding strategy for tokenization. Defaults to "max_length".
            max_length (int, optional): Maximum length for tokenization. If None, use self.max_length. Defaults to None.
        """
        assert self.use_template
        prompt_template = self.prompt_template["template"]
        crop_start = self.prompt_template.get("crop_start", -1)
        if crop_start == -1:
            # Use temporary max_length for the first pass (large enough)
            temp_kwargs = dict(
                truncation=True,
                max_length=256,  # Temporary large value
                padding="max_length",
                return_tensors="pt",
            )

            # First tokenization pass to calculate crop_start
            temp_tokenized = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **temp_kwargs,
            )

            # Calculate the crop_start from this first pass
            crop_start = self.calculate_crop_start(temp_tokenized)

            # Store the calculated crop_start for future use
            self.prompt_template["crop_start"] = crop_start

        kwargs = dict(
            truncation=True,
            max_length=(max_length or self.max_length) + (crop_start if crop_start > 0 else 0),
            padding=padding,
            return_tensors="pt",
        )
        if isinstance(text, (list, tuple)):
            if self.uncond_token == '<|im_end|>':
                text = [self.uncond_token if t == '' else t for t in text]
            text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
        elif isinstance(text, str):
            if self.uncond_token == '<|im_end|>':
                text = self.uncond_token if text == '' else text
            text = self.apply_text_to_template(text, prompt_template)
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        return self.safe_tokenizer(text, **kwargs)

    def encode(self, batch_encoding, use_attention_mask=None, output_hidden_states=False,
               hidden_state_skip_layer=None, crop_start=None, **extra_model_kwargs):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            crop_start (int): The position where the actual prompt content begins (after user markers).
            **extra_model_kwargs: Additional keyword arguments forwarded to the underlying model
                (e.g. ``pixel_values`` / ``image_grid_thw`` for Qwen3-VL multimodal inputs in r2v).
        """
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)

        attention_mask = batch_encoding["attention_mask"].to(self.device) if use_attention_mask else None
        last_hidden_state, outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(self.device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
            # Injected kwargs, will be stripped in TextEncoderWrapper.forward before calling the actual text encoder.
            hidden_state_skip_layer=hidden_state_skip_layer,
            **extra_model_kwargs,
        )
        # if hidden_state_skip_layer is not None:
        #     last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
        #     # Real last hidden state already has layer norm applied. So here we only apply it
        #     # for intermediate layers.
        #     if hidden_state_skip_layer > 0 and self.apply_final_norm:
        #         last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        # else:
        #     last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if self.use_template:
            if crop_start is None:
                crop_start = self.prompt_template.get("crop_start")
            assert crop_start is not None, (
                "crop_start must be provided or auto-calculated via text2tokens() before calling encode()"
            )
            last_hidden_state = last_hidden_state[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None

        if output_hidden_states:
            return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs.hidden_states)
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(self, text, use_attention_mask=None, output_hidden_states=False, hidden_state_skip_layer=None):
        batch_encoding = self.text2tokens(text)
        return self.encode(batch_encoding, use_attention_mask=use_attention_mask,
                           output_hidden_states=output_hidden_states,
                           hidden_state_skip_layer=hidden_state_skip_layer)

    @staticmethod
    def packed_sample_token_runs(batch, include_cond_vit_image: bool = False):
        """Split the packed (bsz=1) batch into the per-sample token runs fed to Qwen —
        text only, or text ∪ cond_vit blocks (r2v) when ``include_cond_vit_image``.

        r2v runs are reordered vision-first because Qwen3-VL wants vision before text while
        the packed DiT sequence may place cond_vit after the prompt; the returned
        ``restore_perms`` (None otherwise) undoes that reorder for the DiT text scatter,
        which consumes states in ascending sequence-position order.
        """
        tokens, text_mask, text_slices = batch["tokens"], batch["text_mask"], batch["text_slices"]
        assert tokens.size(0) == 1, "Only support batch size of 1 for sequence packing for now."

        if not include_cond_vit_image:
            packed_text_inputs = tokens[0, text_mask[0].bool()]
            text_token_lengths = [sli.stop - sli.start for sli in text_slices[0]]
            return list(packed_text_inputs.split(text_token_lengths)), None

        cond_vit_image_slices = (batch.get("cond_vit_image_slices") or [[]])[0]
        cond_vit_video_slices = (batch.get("cond_vit_video_slices") or [[]])[0]
        # 使用Qwen-VL Video 时必须要有cond_vit_video_context_slices
        # 用于明确加入的timestep等时间戳信息的位置
        cond_vit_video_context_slices = (
            batch.get("cond_vit_video_context_slices") or [[]]
        )[0]
        offsets = batch.get("offsets")
        assert offsets is not None, "r2v packed text encoding requires `offsets` from sequence packing."
        sample_offsets = offsets[0].tolist()
        n_samples = len(sample_offsets) - 1

        seqlen_full = text_mask.size(1)
        # cond_vit blocks (vision_start..vision_end) for both reference images and videos,
        # kept apart from plain text so the two can be told apart when reordering.
        # -1/+1 把图片的boi/eoi也纳入cond_vit_image_slices中
        vit_mask = torch.zeros(seqlen_full, dtype=torch.bool, device=tokens.device)
        for sli in list(cond_vit_image_slices) + list(cond_vit_video_slices):
            vit_mask[max(sli.start - 1, 0): min(sli.stop + 1, seqlen_full)] = True
        for sli in cond_vit_video_context_slices:
            vit_mask[max(sli.start, 0): min(sli.stop, seqlen_full)] = True
        te_mask = text_mask[0].clone().bool() | vit_mask

        runs = []
        restore_perms = []
        for i in range(n_samples):
            seg_start = sample_offsets[i]
            seg_end = sample_offsets[i + 1]
            seg_sel = te_mask[seg_start:seg_end]
            if not seg_sel.any():
                continue
            # Ascending local indices of the tokens fed to Qwen (text ∪ cond_vit).
            sel_local = torch.nonzero(seg_sel, as_tuple=False).flatten()
            is_vit_sel = vit_mask[seg_start:seg_end][sel_local]
            image_first_local = torch.cat([sel_local[is_vit_sel], sel_local[~is_vit_sel]])
            restore_perm = torch.argsort(image_first_local)
            runs.append(tokens[0, seg_start:seg_end][image_first_local])
            restore_perms.append(restore_perm)

        return runs, restore_perms

    @staticmethod
    def unpack_inputs(batch, include_cond_vit_image: bool = False):
        """
        Unpack the token sequence with the text_mask and text slices, make it a padded batch for encoding.
        """
        text_inputs, restore_perms = TextEncoder.packed_sample_token_runs(
            batch, include_cond_vit_image=include_cond_vit_image)
        device = batch["tokens"].device
        text_token_lengths = [t.size(0) for t in text_inputs]
        bsz = len(text_inputs)

        # Pad and make batch
        max_length = max(text_token_lengths)
        padded_text_inputs = []
        attention_mask = torch.ones(bsz, max_length, dtype=torch.long, device=device)
        for i, text_input in enumerate(text_inputs):
            padding_length = max_length - text_input.size(0)
            if padding_length > 0:
                padded_text_input = torch.cat([
                    text_input,
                    torch.zeros(padding_length, dtype=text_input.dtype, device=text_input.device)
                ])
                attention_mask[i, -padding_length:] = 0
            else:
                padded_text_input = text_input
            padded_text_inputs.append(padded_text_input)
        batch_encoding = {
            "input_ids": torch.stack(padded_text_inputs, dim=0),
            "attention_mask": attention_mask,
            "restore_perms": restore_perms,
        }
        return batch_encoding

    def _resolve_qwenvl_image_token_id(self):
        """Return the `<|image_pad|>` token id used by Qwen-VL family tokenizers.
        """
        # Cache to avoid repeated lookups.
        if hasattr(self, "_qwenvl_image_token_id_cache"):
            return self._qwenvl_image_token_id_cache

        tokenizer = self.tokenizer
        candidate_id = getattr(tokenizer, "img_token_id", None)
        if candidate_id is None:
            candidate_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if candidate_id is None or candidate_id == unk_id:
                raise ValueError(
                    "Qwen3-VL multimodal path expects `<|image_pad|>` to exist in the tokenizer "
                    "vocabulary, but it was not found. Got "
                    f"convert_tokens_to_ids('<|image_pad|>')={candidate_id}, unk_token_id={unk_id}."
                )

        self._qwenvl_image_token_id_cache = candidate_id
        return candidate_id

    def _resolve_qwenvl_video_token_id(self):
        """Return the `<|video_pad|>` token id used by Qwen-VL family tokenizers."""

        if hasattr(self, "_qwenvl_video_token_id_cache"):
            return self._qwenvl_video_token_id_cache

        tokenizer = self.tokenizer
        candidate_id = getattr(tokenizer, "video_token_id", None)
        if candidate_id is None:
            candidate_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if candidate_id is None or candidate_id == unk_id:
                raise ValueError(
                    "Qwen3-VL native video path expects `<|video_pad|>` in the tokenizer "
                    f"vocabulary, got id={candidate_id}, unk_token_id={unk_id}."
                )

        self._qwenvl_video_token_id_cache = candidate_id
        return candidate_id

    @staticmethod
    def _restore_image_first_output(hidden_state, restore_perms):
        """Permute per-sample rows of ``hidden_state`` ([bsz, L, dim]) from the image-first
        order used to feed Qwen back to the ascending sequence-position order expected by the
        DiT text scatter"""
        if not restore_perms:
            return hidden_state
        hs = hidden_state.clone()
        for i, perm in enumerate(restore_perms):
            n = perm.numel()
            if n > 0:
                hs[i, :n] = hs[i, :n].index_select(0, perm.to(hs.device))
        return hs

    def batch_encode_with_sp(
        self,
        batch,
        system_prompt,
        sequence_pack=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        def contains_slice(value):
            if isinstance(value, slice):
                return True
            if isinstance(value, (list, tuple)):
                return any(contains_slice(item) for item in value)
            return False

        has_video_pixels = pixel_values_videos is not None
        has_video_grid = video_grid_thw is not None
        if has_video_pixels != has_video_grid:
            raise ValueError(
                "Native condition videos require both pixel_values_videos and video_grid_thw."
            )
        has_video_placeholders = contains_slice(batch.get("cond_vit_video_slices"))
        if has_video_placeholders and not has_video_pixels:
            raise ValueError(
                "cond_vit_video placeholders cannot be encoded without native video inputs."
            )

        sp_tokens = torch.tensor(self.tokenizer.encode(system_prompt), dtype=torch.long).unsqueeze(0)
        crop_start = sp_tokens.size(1)

        extra_model_kwargs = {}
        if pixel_values is not None:
            extra_model_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            extra_model_kwargs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            extra_model_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            extra_model_kwargs["video_grid_thw"] = video_grid_thw
        include_cond_image = pixel_values is not None or image_grid_thw is not None
        include_cond_video = has_video_pixels
        include_cond_vit = include_cond_image or include_cond_video
        img_token_id = self._resolve_qwenvl_image_token_id() if include_cond_image else None
        video_token_id = self._resolve_qwenvl_video_token_id() if include_cond_video else None
        restore_perms = None
        if not sequence_pack:
            sp_tokens = sp_tokens.expand(batch["tokens"].size(0), -1).to(batch["tokens"].device)
            token_attention_mask = batch["text_mask"]
            if include_cond_vit:
                seqlen_full = batch["tokens"].size(1)
                # cond_vit block mask (vision_start .. vision_end) covering BOTH reference images
                # and reference videos, kept separate from plain text so the two groups can be told
                # apart when reordering to image-first.
                vit_mask = torch.zeros_like(batch["text_mask"], dtype=torch.bool)
                found_slices = False
                for slices_key in ("cond_vit_image_slices", "cond_vit_video_slices"):
                    cond_vit_slices = batch.get(slices_key)
                    if cond_vit_slices is None:
                        continue
                    for batch_idx, slices_i in enumerate(cond_vit_slices):
                        if isinstance(slices_i, slice):
                            slices_i = [slices_i]
                        for sli in slices_i:
                            found_slices = True
                            vit_mask[batch_idx, max(sli.start - 1, 0): min(sli.stop + 1, seqlen_full)] = True
                video_context_slices = batch.get("cond_vit_video_context_slices")
                if video_context_slices is not None:
                    for batch_idx, slices_i in enumerate(video_context_slices):
                        if isinstance(slices_i, slice):
                            slices_i = [slices_i]
                        for sli in slices_i:
                            found_slices = True
                            vit_mask[
                                batch_idx,
                                max(sli.start, 0): min(sli.stop, seqlen_full),
                            ] = True
                if not found_slices:
                    for mask_key in ("cond_vit_image_mask", "cond_vit_video_mask"):
                        if mask_key in batch:
                            vit_mask |= batch[mask_key].bool()
                token_attention_mask = batch["text_mask"].clone().bool() | vit_mask
                # Feed Qwen image-first (cond_vit before text) regardless of DiT layout.
                text_inputs = []
                restore_perms = []
                for b in range(token_attention_mask.size(0)):
                    sel_idx = torch.nonzero(token_attention_mask[b], as_tuple=False).flatten()
                    is_vit_sel = vit_mask[b][sel_idx]
                    image_first_idx = torch.cat([sel_idx[is_vit_sel], sel_idx[~is_vit_sel]])
                    restore_perm = torch.argsort(image_first_idx)
                    restore_perms.append(restore_perm)
                    text_inputs.append(batch["tokens"][b][image_first_idx])
                tokens = torch.nn.utils.rnn.pad_sequence(
                    text_inputs,
                    batch_first=True,
                    padding_value=getattr(self.tokenizer, "pad_token_id", 0) or 0,
                )
                text_token_lengths = token_attention_mask.sum(dim=1)
                compact_attention_mask = (
                    torch.arange(tokens.size(1), device=tokens.device)[None, :] < text_token_lengths[:, None]
                ).to(dtype=batch["text_mask"].dtype)
                tokens = torch.cat([sp_tokens, tokens], dim=1)
                attention_mask = torch.cat([torch.ones_like(sp_tokens), compact_attention_mask], dim=1)
                extra_model_kwargs["mm_token_type_ids"] = build_qwenvl_mm_token_type_ids(
                    tokens,
                    image_token_id=img_token_id,
                    video_token_id=video_token_id,
                ).to(self.device)
            else:
                tokens = torch.cat([sp_tokens, batch["tokens"]], dim=1)
                attention_mask = torch.cat(
                    [torch.ones_like(sp_tokens), token_attention_mask.to(dtype=batch["text_mask"].dtype)],
                    dim=1,
                )
            inputs = dict(input_ids=tokens, attention_mask=attention_mask)
            outputs = self.encode(inputs, crop_start=crop_start, **extra_model_kwargs)
            hidden_state = self._restore_image_first_output(outputs.hidden_state, restore_perms)
            outputs = TextEncoderModelOutput(hidden_state, outputs.attention_mask)

        else:   # packed
            batch_inputs = self.unpack_inputs(batch, include_cond_vit_image=include_cond_vit)
            sp_tokens = sp_tokens.expand(batch_inputs["input_ids"].size(0), -1).to(batch_inputs["input_ids"].device)
            tokens = torch.cat([sp_tokens, batch_inputs["input_ids"]], dim=1)
            attention_mask = torch.cat([torch.ones_like(sp_tokens), batch_inputs["attention_mask"]], dim=1)
            if include_cond_vit:
                # to distinguish text/image/video for ROPE index
                extra_model_kwargs["mm_token_type_ids"] = build_qwenvl_mm_token_type_ids(
                    tokens,
                    image_token_id=img_token_id,
                    video_token_id=video_token_id,
                ).to(self.device)
            inputs = dict(input_ids=tokens, attention_mask=attention_mask)
            outputs = self.encode(inputs, crop_start=crop_start, **extra_model_kwargs)
            restore_perms = batch_inputs.get("restore_perms")
            hidden_state = self._restore_image_first_output(outputs.hidden_state, restore_perms)
            outputs = TextEncoderModelOutput(hidden_state, batch_inputs["attention_mask"])

        return outputs

    def _vl_backbone(self):
        """The Qwen3-VL model behind the FSDP/wrapper layers (owns get_rope_index)."""
        model = getattr(self.model, "text_encoder", self.model)
        assert hasattr(model, "get_rope_index"), (
            f"packed cond_vit encoding needs a Qwen3-VL backbone exposing get_rope_index, "
            f"got {type(model).__name__}."
        )
        return model

    def _packed_vision_kwargs(self, segments, packed_input_ids, vision_inputs):
        """Vision kwargs for the packed (cu_seqlens) forward; M-RoPE is the one part that is
        not packing-aware, so position ids come from ``get_rope_index`` on a padded view."""
        def _cat(key):
            parts = [v[key] for v in vision_inputs if v is not None and v.get(key) is not None]
            return torch.cat(parts, dim=0) if parts else None

        image_grid_thw = _cat("image_grid_thw")
        video_grid_thw = _cat("video_grid_thw")
        kwargs = {}
        for key in ("pixel_values", "pixel_values_videos"):
            value = _cat(key)
            if value is not None:
                kwargs[key] = value
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            kwargs["video_grid_thw"] = video_grid_thw

        mm_token_type_ids = build_qwenvl_mm_token_type_ids(
            packed_input_ids,
            image_token_id=self._resolve_qwenvl_image_token_id() if image_grid_thw is not None else None,
            video_token_id=self._resolve_qwenvl_video_token_id() if video_grid_thw is not None else None,
        )
        kwargs["mm_token_type_ids"] = mm_token_type_ids

        lengths = [s.size(0) for s in segments]
        n_seg, max_len = len(segments), max(lengths)
        pad_ids = torch.zeros(n_seg, max_len, dtype=packed_input_ids.dtype, device=self.device)
        pad_mm = torch.zeros(n_seg, max_len, dtype=mm_token_type_ids.dtype, device=self.device)
        pad_mask = torch.zeros(n_seg, max_len, dtype=torch.bool, device=self.device)
        offset = 0
        for i, length in enumerate(lengths):
            pad_ids[i, :length] = segments[i]
            pad_mm[i, :length] = mm_token_type_ids[0, offset:offset + length]
            pad_mask[i, :length] = True
            offset += length
        position_ids, _ = self._vl_backbone().get_rope_index(
            pad_ids,
            pad_mm,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=pad_mask,
        )
        # [3, n_seg, max_len] -> [3, 1, total_len], dropping the padding columns.
        kwargs["position_ids"] = position_ids.permute(1, 2, 0)[pad_mask].t().unsqueeze(1).contiguous()
        return kwargs

    def sequence_pack_encode(self, batch, system_prompt, sequence_pack=True, vision_inputs=None):
        """Packed sequence encoding for ONE micro-batch. Thin wrapper over
        `sequence_pack_encode_batches` (a single batch is the 1-element case).

        Requires: text encoder model supports cu_seqlens parameter (modified Qwen3.5).
        """
        assert sequence_pack, "sequence_pack_encode only supports sequence_pack=True"
        return self.sequence_pack_encode_batches(
            [batch], [system_prompt], vision_inputs=[vision_inputs])[0]

    def sequence_pack_encode_batches(self, batches, system_prompts, vision_inputs=None):
        """Encode several packed micro-batches in a SINGLE cu_seqlens forward.

        Each batch is itself a packed sequence of samples (bsz=1). `system_prompts`
        is aligned with `batches` (one per batch), so batches of DIFFERENT tasks can
        be packed together — each sample is prefixed with its own batch's system
        prompt (sp length may differ per task). All samples are concatenated into
        ONE forward, then split and padded back per batch to a
        [n_samples, max_text_len, dim] hidden state + mask. Returns a list aligned
        with batches.

        `vision_inputs` is aligned with `batches`; a non-None entry makes that batch's
        cond_vit placeholder tokens part of the packed run (r2v).
        """
        assert self.supports_packed_sequence, (
            f"sequence_pack_encode requires packed sequence support, but text encoder "
            f"'{self.text_encoder_type}' does not support it. "
            f"Currently only qwen-3.5-9b with patch applied is supported."
        )
        assert len(batches) == len(system_prompts) and len(batches) >= 1, (
            f"sequence_pack_encode_batches needs len(batches)==len(system_prompts)>=1, "
            f"got {len(batches)} and {len(system_prompts)}")
        if vision_inputs is None:
            vision_inputs = [None] * len(batches)
        assert len(vision_inputs) == len(batches), (
            f"vision_inputs must align with batches, got {len(vision_inputs)} vs {len(batches)}")

        # Per-batch system prompt tokens; crop length is per-batch since sp differs by task.
        sp_tokens_per_batch = [
            torch.tensor(self.tokenizer.encode(sp), dtype=torch.long, device=self.device)
            for sp in system_prompts
        ]

        # Build one global packed input [sp + text, ...] across all batches; remember
        # each batch's per-sample lengths and its sp length so outputs split back exactly.
        packed_segments = []
        cu_seqlens_list = [0]
        offset = 0
        per_batch_lengths = []
        per_batch_crop = []
        per_batch_perms = []
        for batch, sp_tokens, vision in zip(batches, sp_tokens_per_batch, vision_inputs):
            runs, restore_perms = self.packed_sample_token_runs(
                batch, include_cond_vit_image=vision is not None)
            per_batch_lengths.append([run.size(0) for run in runs])
            per_batch_crop.append(sp_tokens.size(0))
            per_batch_perms.append(restore_perms)
            for run in runs:
                # prepend this batch's system prompt
                segment = torch.cat([sp_tokens, run.to(self.device)])
                packed_segments.append(segment)
                offset += segment.size(0)
                cu_seqlens_list.append(offset)

        packed_input_ids = torch.cat(packed_segments).unsqueeze(0)  # [1, total_len]
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=self.device)

        extra_model_kwargs = {}
        if any(vision is not None for vision in vision_inputs):
            extra_model_kwargs = self._packed_vision_kwargs(
                packed_segments, packed_input_ids, vision_inputs)

        # Forward with packed sequences — no padding, no attention_mask needed
        hidden_state_skip_layer = self.hidden_state_skip_layer
        with torch.no_grad():
            last_hidden_state, _ = self.model(
                input_ids=packed_input_ids,
                attention_mask=None,
                output_hidden_states=hidden_state_skip_layer is not None,
                hidden_state_skip_layer=hidden_state_skip_layer,
                cu_seqlens=cu_seqlens,
                **extra_model_kwargs,
            )
        hidden_state_flat = last_hidden_state.squeeze(0)  # [total_len, hidden_dim]
        hidden_dim = hidden_state_flat.size(-1)

        # Split by cu_seqlens, crop each segment's (per-batch) system prompt, pad per
        # batch to its own max text length for downstream compatibility.
        outputs = []
        seg = 0  # global segment index into cu_seqlens_list
        for text_token_lengths, crop_start, restore_perms in zip(
                per_batch_lengths, per_batch_crop, per_batch_perms):
            cropped, text_only = [], []
            for i in range(len(text_token_lengths)):
                start = cu_seqlens_list[seg] + crop_start  # skip system prompt tokens
                end = cu_seqlens_list[seg + 1]
                states = hidden_state_flat[start:end]
                if restore_perms is not None:
                    # back to ascending sequence-position order for the DiT text scatter
                    states = states.index_select(0, restore_perms[i].to(states.device))
                cropped.append(states)
                text_only.append(end - start)
                seg += 1
            max_text_len = max(text_only)
            n_samples = len(text_token_lengths)
            padded = torch.zeros(n_samples, max_text_len, hidden_dim,
                                 dtype=hidden_state_flat.dtype, device=self.device)
            amask = torch.zeros(n_samples, max_text_len, dtype=torch.long, device=self.device)
            for i, (hs, length) in enumerate(zip(cropped, text_only)):
                padded[i, :length] = hs
                amask[i, :length] = 1
            outputs.append(TextEncoderModelOutput(padded, amask))
        return outputs
