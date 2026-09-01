import json
import random
import math
import re
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable, Any, Union, TYPE_CHECKING

import torch
import torch.distributed as dist
from PIL import Image
from accelerate import dispatch_model
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import ALL_CACHE_NAMES, GenerationMixin, GenerateOutput, GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel, GenerationConfig
try:
    from transformers.modeling_utils import PretrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import ModelOutput

from hymm.ar.pipelines.pipeline_hunyuan_multimodal import HunyuanMultimodalPipeline
from hymm.diffusion.flow.transport import compute_empirical_mu
from hymm.diffusion.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hymm.core.global_vars import get_parallel_state
from hymm.data_kits.system_prompt import get_system_prompt
from hymm.data_kits.utils.image_utils import ImageProcessor
from hymm.models.autoregressive.custom_cache import HunyuanStaticCache
from hymm.models.tokenizers.conversation import get_conversation_template
from hymm.models.utils.generation_utils import MultimodalGenerationOutputs
from hymm.utils.helpers import default
from hymm.utils.rank_log import RankPrefixedTextStreamer, rank_print, rank_print_multiline
from hymm.utils.image_base import ImageInfo, ImageTensor, CondImage
from hymm.utils.import_utils import is_package_version
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from .hunyuan_multimodal import HunyuanMultimodalBase
from .hunyuan_multimodal_config import HunyuanMultimodalConfig

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

InputImage = Optional[Union[Image.Image, str, bytes]]
Messages = list[dict[str, Any]]


def _dp_any_active(local_active: bool, device) -> bool:
    """All-reduce(SUM)：任一 DP rank 还 active 就返回 True。
    用于让先逻辑结束的 rank 继续 no-op 陪跑，避免 FSDP collective 因
    各 rank forward 次数不一致而卡住。"""
    ps = get_parallel_state()
    dp_size = ps.dp_size if ps is not None else 1
    if dp_size <= 1:
        return local_active
    flag = torch.tensor(
        [1 if local_active else 0], device=device, dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM, group=ps.dp_group)
    return flag.item() > 0


def to_device(data, device):
    if device is None:
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data


def map_absolute_to_local_token_indices(
    input_pos: torch.Tensor,
    abs_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Map absolute sequence positions to local (current forward) indices for use with KV cache.

    When using KV cache, the model only forwards a subset of positions; input_pos[b, i] is the
    absolute position of the i-th token in the current forward. This function converts absolute
    indices (e.g. und_token_indices / gen_token_indices from tokenizer) to local indices in [0,
    current_len-1], so that gather/scatter in MOT layers index into the current hidden_states.

    Supports batch >= 1 and CFG: each batch row has its own input_pos and abs_indices.
    Padding: if different batch rows have different counts of "in-current" indices, we pad by
    repeating valid local indices so that output shape is (B, max_count).

    Args:
        input_pos: (B, current_len), absolute positions for this forward step.
        abs_indices: (B, N) or (N,) absolute positions (e.g. und or gen token positions in full seq).
        device: target device for output tensor.

    Returns:
        local_indices: (B, max_count) local indices in [0, current_len-1], padded by repetition.
    """
    if abs_indices.numel() == 0:
        return abs_indices.to(device)
    B = input_pos.size(0)

    if abs_indices.dim() == 1:
        abs_indices = abs_indices.unsqueeze(0).expand(B, -1).to(device)
    else:
        abs_indices = abs_indices.to(device)
    input_pos = input_pos.to(device)

    # For each b, find which abs_indices[b] are in input_pos[b] and their local index
    # (input_pos[b].unsqueeze(1) == abs_indices[b].unsqueeze(0)) -> (current_len, N)
    local_indices_list = []
    for b in range(B):
        in_current = (input_pos[b].unsqueeze(1) == abs_indices[b].unsqueeze(0))  # (cur_len, N)
        # local_idx[n] = i s.t. input_pos[b,i] == abs_indices[b,n]; argmax gives 0 when not found (filtered below)
        local_idx = in_current.float().argmax(0)  # (N,)
        found = in_current.any(0)  # (N,) False for padding (-1) or positions not in current forward
        valid_local = local_idx[found]
        local_indices_list.append(valid_local)
    max_count = max(len(x) for x in local_indices_list)
    if max_count == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=device)
    result_list = []
    for b in range(B):
        valid = local_indices_list[b]
        if len(valid) == 0:
            result_list.append(torch.zeros(max_count, dtype=torch.long, device=device))
        else:
            repeat_times = (max_count + len(valid) - 1) // len(valid)
            padded = (valid.repeat(repeat_times))[:max_count]
            result_list.append(padded)
    return torch.stack(result_list)


class HunyuanMultimodalGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.diff_infer_steps = kwargs.pop("diff_infer_steps", 50)
        self.diff_guidance_scale = kwargs.pop("diff_guidance_scale", 5.0)
        self.cfg_distilled = kwargs.pop("cfg_distilled", False)
        self.meanflow = kwargs.pop("meanflow", False)
        self.flow_reverse = kwargs.pop("flow_reverse", False)
        self.flow_solver = kwargs.pop("flow_solver", "euler")
        self.flow_shift = kwargs.get("flow_shift", 3.0)
        self.use_flux_shift = kwargs.pop("use_flux_shift", False) # if True, flow_shift will be ignored
        self.use_flux2_shift = kwargs.pop("use_flux2_shift", False)
        assert not (self.use_flux_shift and self.use_flux2_shift), (
            "use_flux_shift and use_flux2_shift cannot both be True"
        )
        self.flux2_empirical_num_steps = kwargs.pop("flux2_empirical_num_steps", None)
        self.flux_base_num_tokens = kwargs.pop("flux_base_num_tokens", 256)
        self.flux_base_log_shift = kwargs.pop("flux_base_log_shift", 0.5)
        self.flux_max_num_tokens = kwargs.pop("flux_max_num_tokens", 4096)
        self.flux_max_log_shift = kwargs.pop("flux_max_log_shift", 1.15)
        self.flow_start_sigma = kwargs.pop("flow_start_sigma", 1.0)
        self.flow_end_sigma = kwargs.pop("flow_end_sigma", 0.0)
        self.use_system_prompt = kwargs.get("use_system_prompt", None)
        self.drop_think = kwargs.pop("drop_think", False)
        self.drop_think_use_system_prompt = kwargs.get("drop_think_use_system_prompt", None)
        self.bot_task = kwargs.get("bot_task", "image")
        self.sequence_template = kwargs.pop("sequence_template", "pretrain")


class HunyuanMultimodalHFConfig(PretrainedConfig):
    def __init__(self, hf_config):
        super().__init__()
        for key, value in hf_config.items():
            setattr(self, key, value)


class HunyuanMultimodalPreTrainedModel(PreTrainedModel):
    config_class = HunyuanMultimodalConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunyuanMultimodalLayer", "HunyuanMultimodalLayerMoT"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True


class HunyuanMultimodalHF(HunyuanMultimodalBase, HunyuanMultimodalPreTrainedModel, GenerationMixin):
    def __init__(
            self,
            args: Namespace,
            config: HunyuanMultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            initialize_weights: bool = True,
            gen_config: Optional[HunyuanMultimodalConfig] = None,
    ):
        hf_config = HunyuanMultimodalHFConfig(config.to_hf_config())
        super().__init__(hf_config)
        self.args = args
        self._dtype = dtype
        self.config = hf_config
        self.__post_init__(config, dtype, device, args, initialize_weights, gen_config)

        # Make sure to tie the weights correctly
        self.tie_weights()

        # Initialize image processor
        self.image_processor = ImageProcessor(args)

        self._tokenizer = None
        self._diffusion_pipeline = None
        self.vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        # Use model_dict instead of directly assigning attributes to avoid unintentionally registering them
        # as submodules of the model, which may cause issues for applying FSDP for both self and these extra models.
        self.model_dict = dict(vae=None)

    @property
    def dtype(self):
        return self._dtype

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def diffusion_pipeline(self):
        return self._diffusion_pipeline

    def build_diffusion_pipeline(self, gen_config: HunyuanMultimodalGenerationConfig):
        scheduler = FlowMatchDiscreteScheduler(
            shift=gen_config.flow_shift,
            reverse=gen_config.flow_reverse,
            solver=gen_config.flow_solver,
            use_flux_shift=gen_config.use_flux_shift,
            use_flux2_shift=gen_config.use_flux2_shift,
            flux_base_num_tokens=gen_config.flux_base_num_tokens,
            flux_base_log_shift=gen_config.flux_base_log_shift,
            flux_max_num_tokens=gen_config.flux_max_num_tokens,
            flux_max_log_shift=gen_config.flux_max_log_shift,
            start_sigma=gen_config.flow_start_sigma,
            end_sigma=gen_config.flow_end_sigma,
        )
        assert self.model_dict["vae"] is not None, "VAE must be initialized before building diffusion pipeline."
        self._diffusion_pipeline = HunyuanMultimodalPipeline(
            model=self, scheduler=scheduler, vae=self.model_dict["vae"],
        )

    def load_pretrained_model(self, dtype, ckpt_path):
        """
        Huggingface style load pretrained model with device map support.
        It is used for model inference with automatic pipeline model parallel.
        """
        from transformers.modeling_utils import (
            PreTrainedModel,
            _get_device_map, _get_resolved_checkpoint_files,    # noqa
        )

        keep_in_fp32_regex = re.compile(r"\.gate\.wg")
        get_device_map_kwargs = dict(keep_in_fp32_regex=keep_in_fp32_regex)
        if is_package_version("transformers", ">=", "4.56"):
            get_device_map_kwargs["dtype"] = dtype
        else:
            get_device_map_kwargs["torch_dtype"] = dtype
        self.device_map = _get_device_map(
            self,  # noqa
            device_map="auto" if torch.cuda.device_count() > 1 else "sequential",
            max_memory=None,
            hf_quantizer=None,
            **get_device_map_kwargs,
        )
        print(f"Device map: \n{json.dumps(self.device_map, indent=4)}", flush=True)

        kwargs = {}
        if is_package_version("transformers", ">=", "4.53"):
            kwargs["is_remote_code"] = False

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=ckpt_path,
            subfolder='',
            variant=None,
            gguf_file=None,
            from_tf=False,
            from_flax=False,
            use_safetensors=None,   # noqa
            cache_dir=None,     # noqa
            force_download=False,
            proxies=None,
            local_files_only=False,
            token=False,
            user_agent={'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False},
            revision='main',
            commit_hash=None,
            **kwargs,
        )

        # Load model weights
        state_dict = None
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = PreTrainedModel._load_pretrained_model(     # noqa
            self,   # noqa
            state_dict,
            checkpoint_files,
            ckpt_path,
            sharded_metadata=sharded_metadata,
            device_map=self.device_map,
            dtype=dtype,
            keep_in_fp32_regex=keep_in_fp32_regex,
            key_mapping=self.get_key_mapping(),
            weights_only=True,
        )
        # Fix missing param's
        if len(missing_keys) > 0:
            # For FlashInferMoE: checkpoint stores fused expert weights
            # (experts.gate_proj_weights shape [N, out, in]) but model expects per-expert
            # weights (experts.{i}.gate_proj.weight).  Detect and unfuse here.
            moe_impl = getattr(self._config, "moe_impl", "hunyuan")
            per_expert_missing = [
                k for k in missing_keys
                if re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight", k)
            ] if moe_impl == "flashinfer" else []

            if per_expert_missing:
                raise NotImplementedError("Fused experts loading is not supported for HF model because of slow loading speed")
            else:
                model_state_dict = self.state_dict()
                for key in missing_keys:
                    param = model_state_dict[key]
                    module, param_type = get_module_from_name(self, key)
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
        # make sure token embedding weights are still tied if needed
        self.tie_weights()

        if self.args.ignore_unexpected_keys is not None:
            ignore_unexpected_keys = [re.compile(prefix) for prefix in self.args.ignore_unexpected_keys]
            unexpected_keys = [
                k for k in unexpected_keys
                if not any(prefix.match(k) for prefix in ignore_unexpected_keys)
            ]
        print(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}", flush=True)

        # Dispatch model to devices and bind align device hooks. With this line, we don't need
        # to call `to(device)` for model inputs anymore when using hf's auto device map.
        if self.device_map is not None:
            dispatch_model(self, device_map=self.device_map)

    def load_generation_config(self, generation_config_path: str | Path):
        # Load generation config
        generation_config_path = Path(generation_config_path)
        if not generation_config_path.exists():
            raise ValueError(f"Generation config path {generation_config_path} does not exist.")
        if generation_config_path.is_file():
            config_dir = generation_config_path.parent
            config_file_name = generation_config_path.name
        else:
            assert generation_config_path.is_dir() and (generation_config_path / "generation_config.json").exists(), \
                (f"No generation_config.json found in {generation_config_path}. "
                 f"Please check the weight format or provide by --generation-config argument.")
            config_dir = generation_config_path
            config_file_name = None

        # Load values from args to override default generation config
        config_keys = HunyuanMultimodalGenerationConfig().to_dict().keys()
        overrides = {}
        for key in config_keys:
            if getattr(self.args, key, None) is not None:
                overrides[key] = getattr(self.args, key)

        print(f"Loading generation config from {config_dir} with overrides: {overrides}", flush=True)
        self.generation_config = HunyuanMultimodalGenerationConfig.from_pretrained(
            config_dir, config_file_name=config_file_name, **overrides,
        )
        print(f"Generation config: {self.generation_config}", flush=True)

    @property
    def layer_device_map(self):
        # Used for HunyuanGeminiStaticCache
        if self.device_map is None:
            raise ValueError("device_map is not set, please call load_hf_model() first.")
        layer_device_map = {}
        for key, value in self.device_map.items():
            if key.startswith("model.layers"):
                layer_idx = int(key.split(".")[-1])
                layer_device_map[layer_idx] = f"cuda:{value}"
        return layer_device_map

    @staticmethod
    def check_inputs(prompt=None, image=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("`prompt` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."
        if image is not None:
            error_msg = \
                "`image` should be a PIL Image, a string path, a base64 string, bytes, or a list of them, but got {}."
            if isinstance(image, list):
                for im in image:
                    assert isinstance(im, (Image.Image, str, bytes)), error_msg.format(type(im))
            else:
                assert isinstance(image, (Image.Image, str, bytes)), error_msg.format(type(image))

    @staticmethod
    def _validate_and_batchify_text(text, name, check_batch_size=None):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), \
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        if isinstance(text, str):
            text = [text]
        assert len(text) > 0 and all(isinstance(p, str) and len(p) > 0 for p in text), \
            f"Input `{name}` should be a non-empty list of non-empty strings, got {text}."
        if check_batch_size is not None:
            assert len(text) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size}), got {len(text)}."
        return text

    @staticmethod
    def _validate_and_batchify_image(image, name, check_batch_size=None):
        if image is None:
            return image
        if not isinstance(image, list):
            raise ValueError(f"Input `{name}` should be a list of images, but got {type(image)}.")
        batch_image_list = [image] if not isinstance(image[0], list) else image
        for image_list in batch_image_list:
            assert all(isinstance(im, InputImage) for im in image_list), \
                (f"Each item in `{name}` should be a PIL Image, a string path, a base64 string, or bytes, "
                 f"got {[type(im) for im in image_list]}.")
        if check_batch_size is not None:
            assert len(batch_image_list) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size})"
        return batch_image_list

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    def build_batch_rope_image_info(self, output, sections):
        # Rope 1D. No need to build rope_image_info
        if self.config.rope_type == "default":
            return None

        # Handle special cases
        meta_dict: dict[str, dict | list[dict]]
        if self.config.vit_type == "anyres-vit-for-a3b":
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

        # RoPE
        assert self.config.rope_type in ["2d", "interleaved_mrope", "xdrope", "3d"], \
            f"Rope type {self.config.rope_type} not supported by method 'build_batch_rope_image_info'."
        rope_image_info = []
        for image_slices, sections_i in zip(output.all_image_slices, sections):
            rope_image_slices = []
            rope_image_shapes = []
            rope_image_metas = []
            image_idx = 0

            for section in sections_i:
                if section['type'] in ["gen_image", "cond_vae_image", "cond_vit_image"]:
                    assert image_idx < len(image_slices), \
                        f"Image index {image_idx} out of range for image slices with length {len(image_slices)}."
                    rope_image_slices.append(image_slices[image_idx])
                    if self.config.rope_type == "3d":
                        rope_image_shapes.append((1, section['token_height'], section['token_width']))
                    else:
                        rope_image_shapes.append((section['token_height'], section['token_width']))
                    rope_image_metas.append(meta_dict[section['type']])
                    image_idx += 1

                elif section['type'] == "cond_joint_image":
                    # We assume `joint` means two image features.
                    assert image_idx + 1 < len(image_slices), \
                        f"Image index {image_idx + 1} out of range for image slices with length {len(image_slices)}."
                    assert len(section['token_height']) == len(section['token_width']), \
                        (f"token_height and token_width should have the same length, "
                         f"but got {len(section['token_height'])} and {len(section['token_width'])}")

                    rope_image_slices.extend([image_slices[image_idx], image_slices[image_idx + 1]])
                    if self.config.rope_type == "3d":
                        rope_image_shapes.extend(
                            list(zip([1, 1], section['token_height'], section['token_width']))
                        )
                    else:
                        rope_image_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    rope_image_metas.extend([meta_dict[section['type']][i] for i in range(2)])
                    image_idx += 2

            rope_image_info.append(list(zip(rope_image_slices, rope_image_shapes, rope_image_metas)))

        return rope_image_info

    def vae_encode(self, image, cfg_factor=1):
        config = self.model_dict["vae"].config

        with torch.autocast(
                device_type="cuda", dtype=self.vae_autocast_dtype,  # noqa
                enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32
        ):
            vae_encode_result = self.model_dict["vae"].encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            else:
                latents = vae_encode_result.latent_dist.sample()
            if hasattr(config, 'shift_factor') and config.shift_factor:
                latents.sub_(config.shift_factor)
            if hasattr(config, 'scaling_factor') and config.scaling_factor:
                latents.mul_(config.scaling_factor)

        if hasattr(self.model_dict["vae"], "ffactor_temporal"):
            assert latents.shape[2] == 1, "latents should have shape [B, C, T, H, W] and T should be 1"
            latents = latents.squeeze(2)

        # Here we always use t=0 to declare it is a clean conditional image
        t = torch.zeros((latents.shape[0],))

        if cfg_factor > 1:
            t = t.repeat(cfg_factor)
            latents = latents.repeat(cfg_factor, 1, 1, 1)

        return t, latents

    def _encode_cond_image(
            self,
            batch_cond_images: list[list[Union[ImageTensor, CondImage]]],
            cfg_factor: int = 1,
    ):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None, None, None

        # 1. If vae_image presents
        if self.image_processor.cond_image_section_type in ["cond_vae_image", "cond_joint_image"]:
            # VAE encode one by one, as we assume cond images have different sizes
            batch_cond_vae_images, batch_cond_t = [], []
            for cond_images in batch_cond_images:
                cond_vae_image_list, cond_t_list = [], []
                for cond_image in cond_images:
                    vae_image = (
                        cond_image.vae_image
                        if self.image_processor.cond_image_section_type == "cond_joint_image"
                        else cond_image
                    )
                    cond_t_, cond_vae_image_ = self.vae_encode(
                        vae_image[None].to(self.device),
                    )
                    cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                    cond_t_list.append(cond_t_)
                batch_cond_vae_images.append(cond_vae_image_list)
                batch_cond_t.append(cond_t_list)

            # If only one cond image for each sample and all have the same size, we can batch them together
            # In this case, cond_vae_images is a 4-D tensor.
            if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
                    items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
                cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
                cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
                if cfg_factor > 1:
                    cond_t = cond_t.repeat(cfg_factor)
                    cond_vae_images = cond_vae_images.repeat(cfg_factor, 1, 1, 1)
            else:
                # In this case, cond_vae_images is a list of 4-D tensors or a list of lists of 3-D tensors.
                cond_t = [torch.cat(item, dim=0) for item in batch_cond_t]
                cond_vae_images = []
                for items in batch_cond_vae_images:
                    if all(items[0].shape == item.shape for item in items):
                        cond_vae_images.append(torch.stack(items, dim=0))
                    else:
                        cond_vae_images.append(items)
                if cfg_factor > 1:
                    cond_t = cond_t * cfg_factor
                    cond_vae_images = cond_vae_images * cfg_factor

        else:
            cond_vae_images = None
            cond_t = None

        # 2. If vit_image presents
        if self.image_processor.cond_image_section_type in ["cond_vit_image", "cond_joint_image"]:
            cond_vit_images = []
            for cond_images in batch_cond_images:
                cond_vit_image_list = []
                for cond_image in cond_images:
                    vit_image = (
                        cond_image.vit_image
                        if self.image_processor.cond_image_section_type == "cond_joint_image"
                        else cond_image
                    )
                    vit_image = vit_image.to(dtype=torch.float32)
                    cond_vit_image_list.append(vit_image)
                # Here we force convert the tensor to dtype
                # cond_vit_images.append(
                #     torch.stack(cond_vit_image_list, dim=0).to(dtype=torch.float32)
                # )
                cond_vit_images.append(cond_vit_image_list)


            if cfg_factor > 1:
                cond_vit_images = cond_vit_images * cfg_factor

        else:
            cond_vit_images = None

        return cond_vae_images, cond_t, cond_vit_images

    @staticmethod
    def _prepare_vit_image_kwargs(batch_cond_images, cfg_factor):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None
        first_image = batch_cond_images[0][0]
        if isinstance(first_image, CondImage):
            vit_image = first_image.vit_image
        else:
            vit_image = first_image
        if not hasattr(vit_image, "vision_encoder_kwargs") or len(vit_image.vision_encoder_kwargs) == 0:
            return None

        image_type = vit_image.i.image_type
        if image_type == "qwen3vl":
            cond_vit_image_kwargs = {"grid_thw": []}
            for cond_images in batch_cond_images:
                cond_vit_image_kwargs["grid_thw"].append(torch.stack([
                    cond_image.vision_encoder_kwargs["grid_thw"]
                    for cond_image in cond_images
                ]))
        else:
            # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
            cond_vit_image_kwargs = {"spatial_shapes": [], "attention_mask": []}
            for cond_images in batch_cond_images:
                cond_vit_image_kwargs["spatial_shapes"].append(
                    torch.stack([
                        cond_image.vit_image.vision_encoder_kwargs["spatial_shapes"]
                        for cond_image in cond_images
                    ]))
                cond_vit_image_kwargs["attention_mask"].append(
                    torch.stack([
                        cond_image.vit_image.vision_encoder_kwargs["pixel_attention_mask"]
                        for cond_image in cond_images
                    ]))
            if cfg_factor > 1:
                cond_vit_image_kwargs["spatial_shapes"] = cond_vit_image_kwargs["spatial_shapes"] * cfg_factor
                cond_vit_image_kwargs["attention_mask"] = cond_vit_image_kwargs["attention_mask"] * cfg_factor
        return cond_vit_image_kwargs

    def prepare_message_list(
            self,
            message_list,
            cond_images: list[CondImage] = None,
            gen_image_info: ImageInfo = None,
    ):
        """ Convert a batch message list of OpenAI style to the internal format. """
        inner_message_list = []
        image_idx = 0
        for message in message_list:
            content = message["content"]
            if isinstance(content, str):
                inner_message_list.append(dict(role=message["role"], type="text", content=content))
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        inner_message_list.append(dict(role=message["role"], type="text", content=item['text']))
                    elif item["type"] == "image":
                        if all(key not in item for key in ["image", "url", "path", "base64"]):
                            continue
                        assert cond_images is not None and image_idx < len(cond_images), \
                            f"Image index {image_idx} out of range for cond images with length {len(cond_images)}."
                        image = cond_images[image_idx]
                        inner_message_list.append(dict(role=message["role"], type=self.image_processor.cond_image_section_type, content=image.i))
                        image_idx += 1
                    else:
                        raise NotImplementedError(f"Message content type {item['type']} not supported.")
            else:
                raise ValueError(f"Message content should be str or list, but got {type(content)}.")

        if gen_image_info is not None:
            inner_message_list.append(dict(role="assistant", type="gen_image", content=gen_image_info))

        return inner_message_list
    
    def _build_batch_gen_image_info(self, image_size, batch_size):
        # Support variable resolution, i.e, image_size is a list of (height, width) pairs
        if isinstance(image_size, list):
            assert len(image_size) == batch_size, \
                f"image_size should have the same length as batch_size, got {len(image_size)} and {batch_size}"
            return [self.image_processor.build_gen_image_info(image_size[i]) for i in range(batch_size)]
        return [self.image_processor.build_gen_image_info(image_size) for _ in range(batch_size)]

    def prepare_model_inputs(
            self,
            prompt: str | list[str] = None,
            image: list[InputImage] = None,
            mode="gen_text",
            system_prompt: Optional[str] = None,
            cot_text: str | list[str] = None,
            image_size: str | list[tuple[int, int]] = "auto",
            message_list: Optional[Messages | list[Messages]] = None,
            device=None,
            max_new_tokens=None,
            bot_task="auto",
            **kwargs,
    ):
        # 1. Sanity check
        self.check_inputs(prompt, image, message_list)
        device = default(device, self.device)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt
        batch_cot_text = cot_text
        batch_system_prompt = system_prompt

        #   -- 2.1 message_list
        batch_cond_images = kwargs.get('batch_cond_images', None)

        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            # message_list may be modified later, so we deepcopy it here to avoid side effects
            batch_message_list = deepcopy(batch_message_list)

            # Prepend system prompt if available
            batch_system_prompt = self._validate_and_batchify_text(batch_system_prompt, 'system_prompt', batch_size)
            if batch_system_prompt is not None:
                batch_message_list = [
                    [dict(role="system", type="text", content=sp)] + message_list_
                    for sp, message_list_ in zip(batch_system_prompt, batch_message_list)
                ]

            # Multiple cond images are allowed.
            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(message_list=message_list_)
                    for message_list_ in batch_message_list
                ]
            if mode == "gen_image":
                batch_gen_image_info = self._build_batch_gen_image_info(image_size, batch_size)
            else:
                batch_gen_image_info = [None] * batch_size

            # Convert OpenAI message list into inner message list
            batch_message_list = [
                self.prepare_message_list(message_list_, cond_images, gen_image_info)
                for message_list_, cond_images, gen_image_info in zip(
                    batch_message_list, batch_cond_images, batch_gen_image_info
                )
            ]

        #   -- 2.2 Prompt, image, cot text, system prompt
        else:
            batch_prompt = self._validate_and_batchify_text(batch_prompt, 'prompt')
            batch_size = len(batch_prompt)

            batch_cot_text = self._validate_and_batchify_text(batch_cot_text, 'cot_text', batch_size)
            batch_system_prompt = self._validate_and_batchify_text(batch_system_prompt, 'system_prompt', batch_size)

            batch_image_list = self._validate_and_batchify_image(image, 'image', batch_size)
            if batch_cond_images is None:
                batch_cond_images = [
                    self.image_processor.build_cond_images(image_list=image_list)
                    for image_list in batch_image_list
                ] if batch_image_list is not None else None

            if mode == "gen_image":
                batch_gen_image_info = self._build_batch_gen_image_info(image_size, batch_size)
            else:
                batch_gen_image_info = [None] * batch_size

        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 3. apply chat template
        cfg_factor = {
            "gen_text": 1,
            "gen_image": 1 if self.args.cfg_distilled else 2
        }
        # If `drop_think` enabled, always drop <think> parts in the context.
        drop_think = kwargs.get('drop_think', self.generation_config.drop_think)
        # Get conversation template according to the model_name
        conv_template = get_conversation_template(self.args.model_name.split('.')[-1])
        # Apply batched prompt or batched message_list to build input sequence with associated info.
        out = self._tokenizer.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_cond_images=batch_cond_images,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=kwargs.get('max_length'),
            bot_task=bot_task,  # if mode is "gen_image", bot_task will not be used
            image_base_size=self.image_processor.vae_reso_group.base_size if bot_task == "img_ratio" else None,
            cond_image_section_type=self.image_processor.cond_image_section_type,
            sequence_template=self.generation_config.sequence_template,
            cfg_factor=cfg_factor[mode],
            drop_think=drop_think,
            conv_template=conv_template,
            und_token_type=self.args.und_token_type if self.args.use_mot else [],
            gen_token_type=self.args.gen_token_type if self.args.use_mot else [],
            uncond_length=self.args.uncond_length,
        )

        output, sections = out['output'], out['sections']
        # 4. Encode conditional images
        cond_vae_images, cond_timesteps, cond_vit_images = self._encode_cond_image(
            batch_cond_images, cfg_factor[mode]
        )
        cond_vit_image_kwargs = self._prepare_vit_image_kwargs(batch_cond_images, cfg_factor[mode])

        # 5. Build position embeddings
        rope_image_info = self.build_batch_rope_image_info(output, sections)

        # 6. Build kv cache
        max_new_tokens = default(
            default(max_new_tokens, self.generation_config.max_new_tokens),
            self.generation_config.max_length,
        )
        if mode == "gen_image":
            # Image generation will not extend sequence length, using token length as max_cache_len is enough.
            max_cache_len = output.tokens.shape[1]
        else:
            max_cache_len = output.tokens.shape[1] + max_new_tokens
        cache = HunyuanStaticCache(
            config=self.config,
            max_batch_size=batch_size * cfg_factor[mode],
            max_cache_len=max_cache_len,
            dtype=self.dtype,
            dynamic=mode == "gen_text",
        )

        # 7. Build position ids
        # max_batch_size x input_seq_len
        batch_input_pos = torch.arange(
            0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
            batch_size * cfg_factor[mode], -1)  # use expand to share indices to save memory

        # 8. Define stop tokens by tasks
        tkw = self._tokenizer
        if mode == "gen_image":
            eos_token_id = None  # don't need to define eos_token_id for image generation
        else:
            if bot_task == "auto":
                stop_token_id = dict(
                    auto=conv_template.stop_token_ids,
                )
            else:
                if image_size == "auto":
                    if bot_task == "img_tw_th":
                        # Anyres auto-size predicts <img_tw_th_h> and <img_tw_th_w>;
                        # max_new_tokens controls completion, so no EOS size token is needed.
                        extra_auto_stops = []
                    else:
                        if hasattr(tkw, "get_all_ratio_token_ids"):
                            extra_auto_stops = tkw.get_all_ratio_token_ids()
                        else:
                            extra_auto_stops = list(range(
                                tkw.ratio_token_id(0),
                                tkw.ratio_token_id(0) + len(self.image_processor.vae_reso_group)
                            ))
                else:
                    extra_auto_stops = [tkw.boi_token_id]
                stop_token_id = dict(
                    auto=conv_template.stop_token_ids + extra_auto_stops,
                    recaption=[tkw.end_of_recaption_token_id],
                    think=[tkw.end_of_think_token_id],
                    img_ratio=extra_auto_stops,
                    img_tw_th=conv_template.stop_token_ids,
                    quickly_think=conv_template.stop_token_ids + extra_auto_stops,
                    slowly_think=conv_template.stop_token_ids + extra_auto_stops,
                )
            eos_token_id = stop_token_id[bot_task]

        # Compute batch_image_sizes for variable resolution in 'gen_image' mode
        batch_image_sizes = None
        if mode == "gen_image" and all(batch_gen_image_info[i] is not None for i in range(len(batch_gen_image_info))):
            all_same_size = all(
                info.image_height == batch_gen_image_info[0].image_height and
                info.image_width == batch_gen_image_info[0].image_width
                for info in batch_gen_image_info
            )
            # if all_same_size, batch_image_sizes will be None
            if not all_same_size:
                vae_df = self.config.vae_downsample_factor
                batch_image_sizes = []
                for info in batch_gen_image_info:
                    batch_image_sizes.append((info.image_height // vae_df, info.image_width // vae_df))

        # 9. Build model input kwargs
        model_input_kwargs = dict(
            input_ids=output.tokens.to(device),
            input_pos=batch_input_pos,
            past_key_values=cache,
            mode=mode,
            rope_image_info=rope_image_info,
            image_mask=to_device(output.gen_image_mask, device),
            timesteps_index=to_device(output.gen_timestep_scatter_index, device),
            timestep_r_index=to_device(output.gen_timestep_r_scatter_index, device),
            guidance_index=to_device(output.guidance_scatter_index, device),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_vae_image_mask=to_device(output.vae_image_mask, device),
            cond_timesteps=to_device(cond_timesteps, device),
            cond_timesteps_index=to_device(output.cond_timestep_scatter_index, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.vit_image_mask, device),
            cond_vit_image_kwargs=to_device(cond_vit_image_kwargs, device),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            generator=generator,
            batch_cond_images=batch_cond_images,
            batch_image_sizes=batch_image_sizes,
            # generation config
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=kwargs.get("return_dict_in_generate", False),
            output_logits=kwargs.get("output_logits", None),
        )
        if mode == "gen_image":
            model_input_kwargs["cfg_factor"] = cfg_factor[mode]
        if self.args.use_mot:
            model_input_kwargs["und_token_indices"] = to_device(output.und_token_indices, device)
            model_input_kwargs["gen_token_indices"] = to_device(output.gen_token_indices, device)

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
        tokenizer_output = model_kwargs["tokenizer_output"]
        batch_full_attn_slices = [
            self.image_processor.prepare_full_attn_slices(tokenizer_output, i)
            for i in range(bsz)
        ]
        if len(batch_full_attn_slices[0]) == 0:
            return None

        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).tril(
            diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_full_attn_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    # 对于生文，prepare_inputs_for_generation 入参 input_ids 永远是最新的、完整的、结束后不会新增序列。
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_cond_images=None, batch_gen_image_info=None, generator=None,
            **kwargs
    ):
        input_pos = kwargs.get("input_pos", kwargs.get("position_ids"))
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            assert input_pos is not None, "input_pos or position_ids must be provided in kwargs."
            # In gen_image mode (loop in pipeline_hunyuan_multimodal.py):
            #   - (Case 1) 1st step: input_ids is required with full seqlen
            #   - (Case 2) >=2nd steps: input_ids is passed as None in pipeline
            # In gen_text mode (loop in transformers generation api):
            #   - (Case 3) prefill step: input_ids is required with full seqlen
            #   - (Case 4) decode step(kv-cache): input_ids is required with the single token generated in last step
            if input_ids is not None and input_ids.shape[1] != input_pos.shape[1]:
                # only Case 4 can come here
                assert input_pos.size(0) == 1, f"Only support batch size 1, got {input_pos.size(0)}"
                # For case 4, the input_pos has two situations (with synced_gpus enabled):
                #   - input_pos[0, -1] < input_ids's seqlen: not finished
                #   - input_pos[0, -1] >= input_ids's seqlen: already finished, but still running for waiting
                #   other gpus finished for fsdp model.
                # super().generate calls _update_model_kwargs_for_generation every step, but doesn't append new token to input_ids when synced_gpus enabled and finised generation.
                if input_pos[0, -1] >= input_ids.shape[1]:
                    input_ids = input_ids[:, -input_pos.shape[1]:]
                    # 如果不同 rank 上的 max_new_tokens 不同，比如 rank1 的 max_new_tokens 为 n1，rank2 的 max_new_tokens 为 n2， n2 > n1，当 rank1 生成结束而 rank2 还在继续生成时，rank1 上每轮生成依然会调用 _update_model_kwargs_for_generation 对 input_pos + 1，当 rank2 生成长度超过 n1 时，rank1 上的 input_pos 也会超过 n1，由于 kvcache 的长度是根据 max_new_tokens 设置的，此时 rank1 上 kvcache 就会越界报错
                    # 所以这里为 input_pos 赋值，让 rank1 在生成结束后一直重复跑最后一个 token 来 synced_gpus
                    input_pos[0, -1] = input_ids.shape[1] - 1
                else:
                    input_ids = torch.gather(input_ids, dim=1, index=input_pos)
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "input_pos": input_pos,
                "past_key_values": past_key_values,
                # "use_cache": kwargs.get("use_cache"),
                "mode": kwargs["mode"],
                "rope_image_info": kwargs["rope_image_info"],
                "images": kwargs.get("images"),
                "image_mask": kwargs.get("image_mask"),
                "timesteps": kwargs.get("timesteps"),
                "timesteps_index": kwargs.get("timesteps_index"),
                "timestep_r": kwargs.get("timestep_r"),
                "timestep_r_index": kwargs.get("timestep_r_index"),
                "guidance": kwargs.get("guidance"),
                "guidance_index": kwargs.get("guidance_index"),
                "cond_vae_images": kwargs.get("cond_vae_images"),
                "cond_vae_image_mask": kwargs.get("cond_vae_image_mask"),
                "cond_timesteps": kwargs.get("cond_timesteps"),
                "cond_timesteps_index": kwargs.get("cond_timesteps_index"),
                "cond_vit_images": kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": kwargs.get("cond_vit_image_mask"),
                "cond_vit_image_kwargs": kwargs.get("cond_vit_image_kwargs"),
                "batch_image_sizes": kwargs.get("batch_image_sizes"),
            }
        )
        if self.args.use_mot:
            und_abs = kwargs.get("und_token_indices")
            gen_abs = kwargs.get("gen_token_indices")
            # When using KV cache, the model only forwards a subset of positions; input_pos gives
            # absolute -> current local mapping. Convert absolute und/gen_token_indices to local.
            if past_key_values is not None and und_abs is not None and gen_abs is not None:
                device = input_pos.device
                model_inputs["und_token_indices"] = map_absolute_to_local_token_indices(
                    input_pos, und_abs, device
                )
                model_inputs["gen_token_indices"] = map_absolute_to_local_token_indices(
                    input_pos, gen_abs, device
                )
                # In case 4 (gen_text mode, decode step(kv-cache)), newly generated text tokens (index by input_pos) are not part of the original tokenizer indices (not in und_token_indices and gen_token_indices), so map_absolute_to_local_token_indices return two zero-length indices tensor, we create und_token_indices to index newly generated text tokens to stay on the understanding branch
                if kwargs.get("mode") == "gen_text" and \
                        model_inputs["und_token_indices"].shape[1] == 0 and \
                        model_inputs["gen_token_indices"].shape[1] == 0:
                    
                    batch_size, current_len = input_pos.shape
                    model_inputs["und_token_indices"] = torch.arange(
                        current_len, device=device, dtype=torch.long
                    ).unsqueeze(0).expand(batch_size, -1)
                    model_inputs["gen_token_indices"] = torch.zeros(
                        batch_size, 0, device=device, dtype=torch.long
                    )
            else:
                model_inputs["und_token_indices"] = und_abs
                model_inputs["gen_token_indices"] = gen_abs
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        """ This function is run after each step of model forward. It updates model kwargs for next forward step.
        """
        mode = model_kwargs["mode"]

        updated_model_kwargs = {
            "mode": mode,
            "rope_image_info": model_kwargs["rope_image_info"],
        }

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                updated_model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # After the first forward pass
        # prepare_model_inputs put "tokenizer_output" in model_kwargs, so it's in the model_kwargs passed to this function after the first forward pass
        if "tokenizer_output" in model_kwargs:
            if mode == "gen_text":
                # When enable batching, we use right padding, which requires a real_pos to index the valid
                # end position of the sequence. If tokenizer_output in model_kwargs, it means we are in the
                # prefill step of generation.
                real_pos = to_device(model_kwargs["tokenizer_output"].real_pos, self.device)
                updated_model_kwargs["input_pos"] = real_pos
            else:
                # inputs_pos
                image_mask = model_kwargs["image_mask"]
                bsz, seq_len = image_mask.shape
                index = torch.arange(seq_len, device=image_mask.device).unsqueeze(0).repeat(bsz, 1)

                batch_image_sizes = model_kwargs.get("batch_image_sizes")
                if batch_image_sizes is not None:
                    # 多分辨率batch推理：
                    # -- 1. 不能在latent维度上2D padding，需要在patch_embed之后进行1D padding，input_pos长度也需要等于batch中最大长度，保持tensor shape一致;
                    # -- 2. 与 batch_gen_infer 中的逻辑保持一致，那里将序列最后的 pad 移入 gen 的，所以这里将序列最后的 pos 移入 input_pos
                    img_token_counts = image_mask.sum(dim=1).long()  # (bsz,)
                    max_img_tokens = img_token_counts.max().item()
                    pad_counts = max_img_tokens - img_token_counts  # (bsz,)
                    input_pos_list = []
                    for i in range(bsz):
                        indices = index[i].masked_select(image_mask[i].bool())
                        if pad_counts[i] > 0:
                            indices = torch.cat([indices, index[i][-pad_counts[i]:]])
                        input_pos_list.append(indices)
                    input_pos = torch.stack(input_pos_list, dim=0)
                else:
                    input_pos = index.masked_select(image_mask.bool()).reshape(bsz, -1)

                timestep_position_ids = \
                    index[torch.arange(bsz), model_kwargs["timesteps_index"][:, -1]].unsqueeze(-1)
                updated_model_kwargs["input_pos"] = torch.cat([timestep_position_ids, input_pos], dim=1)

                # attention mask
                mask_list = []
                for attention_mask_i, position_ids_i in zip(
                        model_kwargs["attention_mask"], updated_model_kwargs["input_pos"]):
                    mask_list.append(torch.index_select(attention_mask_i, dim=1, index=position_ids_i.reshape(-1)))
                attention_mask = torch.stack(mask_list, dim=0)
                updated_model_kwargs["attention_mask"] = attention_mask
        # Following step
        else:
            # After decode steps
            if mode == "gen_text":
                # Now we are in the decode steps.
                updated_model_kwargs["input_pos"] = model_kwargs["input_pos"] + 1
                # Remove attention mask to use full attention of 1 x seqlen in decode steps
            else:
                updated_model_kwargs["input_pos"] = model_kwargs["input_pos"]
                updated_model_kwargs["attention_mask"] = model_kwargs["attention_mask"]

        # Keep absolute und_token_indices and gen_token_indices for MOT so that
        # prepare_inputs_for_generation can map them to local indices when using KV cache.
        if self.args.use_mot:
            if "und_token_indices" in model_kwargs:
                updated_model_kwargs["und_token_indices"] = model_kwargs["und_token_indices"]
            if "gen_token_indices" in model_kwargs:
                updated_model_kwargs["gen_token_indices"] = model_kwargs["gen_token_indices"]

        if "batch_image_sizes" in model_kwargs:
            updated_model_kwargs["batch_image_sizes"] = model_kwargs["batch_image_sizes"]
            # for token num check in ragged_final_layer, not actually used
            updated_model_kwargs["image_mask"] = model_kwargs["image_mask"]

        return updated_model_kwargs

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            use_model_defaults: Optional[bool] = None,
            generator: Optional[list[torch.Generator]] = None,
            decode_text: bool = False,
            verbose: int = 0,
            image_output_type: str = "pil",
            skip_special_tokens: bool = False,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        gen_config = default(generation_config, self.generation_config)
        mode = kwargs.get("mode", "gen_text")
        return_dict_in_generate = kwargs.get("return_dict_in_generate", gen_config.return_dict_in_generate)

        # Log info
        if verbose >= 1:
            output = kwargs["tokenizer_output"]
            context = self._tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}. The same as <pad>
            img_token = self._tokenizer.get_img_token()
            pad_token = self._tokenizer.pad_token
            if isinstance(pad_token, int):
                pad_token = self._tokenizer.decode(pad_token)
            # context = re.sub(f"({img_token})+", lambda m: f"[{img_token}]{{{len(m.group(0)) // len(img_token)}}}", context)
            # context = re.sub(f"({pad_token})+", lambda m: f"[{pad_token}]{{{len(m.group(0)) // len(pad_token)}}}", context)
            context = re.sub(f"({re.escape(img_token)})+", lambda m: f"[{img_token}]{{{len(m.group(0)) // len(img_token)}}}", context)
            context = re.sub(f"({re.escape(pad_token)})+", lambda m: f"[{pad_token}]{{{len(m.group(0)) // len(pad_token)}}}", context)

            info_list = [
                ("token shape", output.tokens.shape),
                ("context[0]", context),
            ]
            start_time = time.time()

        if mode == "gen_text":
            if verbose >= 1:
                eos_ids = kwargs.get("eos_token_id")
                info_list.extend([
                    ("do_sample", kwargs.get("do_sample", gen_config.do_sample)),
                    ("max_new_tokens", kwargs.get("max_new_tokens", gen_config.max_new_tokens)),
                    ("top_k", kwargs.get("top_k", gen_config.top_k)),
                    ("top_p", kwargs.get("top_p", gen_config.top_p)),
                    ("temperature", kwargs.get("temperature", gen_config.temperature)),
                    ("repetition_penalty", kwargs.get("repetition_penalty", gen_config.repetition_penalty)),
                    ("sequence_template", gen_config.sequence_template),
                    ("eos_token_id", eos_ids),
                    ("eos_token", [self._tokenizer.decode(id, skip_special_tokens=False) for id in eos_ids]),
                ])
                self.print_info(info_list)

            if verbose >= 2 and streamer is None:
                streamer = RankPrefixedTextStreamer(
                    self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=False,
                )
            
            dp_enabled = get_parallel_state().dp_size > 1

            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                results = super().generate(
                    inputs,
                    gen_config,
                    logits_processor,
                    stopping_criteria,
                    prefix_allowed_tokens_fn,
                    synced_gpus or dp_enabled,
                    assistant_model,
                    streamer,
                    negative_prompt_ids,
                    negative_prompt_attention_mask,
                    use_model_defaults,
                    **kwargs,
                )
                if isinstance(results, torch.Tensor):
                    samples = results
                else:
                    samples = results.sequences
                if decode_text:
                    samples = self.decode_text(samples, input_length=kwargs["input_ids"].shape[1], skip_special_tokens=skip_special_tokens)
                samples = MultimodalGenerationOutputs(texts=samples)

        elif mode == "gen_image":
            batch_gen_image_info: list[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")

            # support overriding generation config by kwargs, super().generate() does the same thing for text generation
            gen_config.update(**kwargs)
            self.build_diffusion_pipeline(gen_config)

            if verbose >= 1:
                if generator is not None:
                    info_list.extend([
                        ("seed", [g.initial_seed() for g in generator]),
                    ])
                info_list.extend([
                    ("image_size", [f"{info.image_height}x{info.image_width}" for info in batch_gen_image_info]),
                    ("infer_steps", gen_config.diff_infer_steps),
                    ("guidance_scale", gen_config.diff_guidance_scale),
                    ("cfg_distilled", gen_config.cfg_distilled),
                    ("meanflow", gen_config.meanflow),
                ])
                if gen_config.use_flux2_shift:
                    info_list.extend([
                        ("use_flux2_shift", [
                            "%.3f" % math.exp(compute_empirical_mu(
                                info.token_height * info.token_width,
                                gen_config.diff_infer_steps,
                            ))
                            for info in kwargs["batch_gen_image_info"]
                        ]),
                    ])
                elif gen_config.use_flux_shift:
                    lin_func = self.diffusion_pipeline.scheduler.get_lin_function(gen_config.flux_base_num_tokens, gen_config.flux_base_log_shift, gen_config.flux_max_num_tokens, gen_config.flux_max_log_shift)
                    info_list.extend([
                        ("use_flux_shift", [f"%.3f" % math.exp(lin_func(info.token_height * info.token_width)) for info in kwargs["batch_gen_image_info"]]),
                    ])
                else:
                    info_list.extend([
                        ("flow_shift", gen_config.flow_shift),
                    ])
                self.print_info(info_list)

            # Build per-sample image_size list for pipeline with variable resolution support
            all_same = all(
                info.image_height == batch_gen_image_info[0].image_height and
                info.image_width == batch_gen_image_info[0].image_width
                for info in batch_gen_image_info
            )
            if all_same:
                pipeline_image_size = (batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width)
            else:
                pipeline_image_size = [(info.image_height, info.image_width) for info in batch_gen_image_info]

            results = self.diffusion_pipeline(
                batch_size=len(batch_gen_image_info),
                image_size=pipeline_image_size,
                num_inference_steps=gen_config.diff_infer_steps,
                guidance_scale=gen_config.diff_guidance_scale,
                cfg_distilled=gen_config.cfg_distilled,
                meanflow=gen_config.meanflow,
                cfg_factor=kwargs['cfg_factor'],
                generator=generator,
                output_type=image_output_type,
                model_kwargs=kwargs,
            )
            samples = MultimodalGenerationOutputs(images=results[0])

        else:
            raise ValueError(f"Unknown mode {mode}, only `gen_text` and `gen_image` are supported.")

        if verbose >= 1:
            end_time = time.time()
            rank_print(f"Generation completed in {end_time - start_time:.2f} seconds.")

        # Force to delete cache to release memory. Don't know why not be released automatically here.
        kv_cache: HunyuanStaticCache = kwargs.get('past_key_values', None)
        if kv_cache is not None:
            for layer in kv_cache.layers:
                del layer.keys
                del layer.values
            torch.cuda.empty_cache()

        if return_dict_in_generate:
            return MultimodalGenerationOutputs(
                texts=GenerateDecoderOnlyOutput(sequences=samples.texts, logits=results.logits),
            )

        return samples

    def decode_text(self, output: torch.Tensor, input_length: int = None, skip_special_tokens: bool = False):
        if output.ndim == 2:
            assert output.size(0) == 1, "Batch decoding is not supported yet."
            return [self.decode_text(output_i, input_length, skip_special_tokens=skip_special_tokens) for output_i in output]
        elif output.ndim == 1:
            if input_length is not None:
                output = output[input_length:]
            text = self._tokenizer.decode(output, skip_special_tokens=skip_special_tokens)
            return text
        else:
            raise ValueError(f"output should be 1D or 2D tensor, but got {output.ndim}D tensor.")

    def generate_image(
            self,
            prompt=None,
            image=None,
            message_list=None,
            seed=None,
            image_size="auto",
            use_system_prompt=None,
            drop_think_use_system_prompt=None,
            system_prompt=None,
            drop_think_system_prompt=None,
            bot_task=None,
            max_new_tokens=None,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        drop_think_use_system_prompt = default(drop_think_use_system_prompt, self.generation_config.drop_think_use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)
        drop_think_system_prompt = get_system_prompt(drop_think_use_system_prompt, bot_task, drop_think_system_prompt)

        if self.generation_config.drop_think and system_prompt:
            assert drop_think_system_prompt is not None, "drop_think_system_prompt should be provided when drop_think is True and system_prompt is provided."

        if message_list is not None:
            # We will update message_list, so deepcopy it first to avoid changing it outside.
            message_list = deepcopy(message_list)

        batch_cond_images_cache = None
        if bot_task in ["think", "recaption", "think_recaption"]:
            # Cot step 1
            cur_bot_task, *remain_bot_tasks = bot_task.split("_")
            model_inputs = self.prepare_model_inputs(
                mode="gen_text", bot_task=cur_bot_task, max_new_tokens=max_new_tokens,
                prompt=prompt, image=image, system_prompt=system_prompt,
                message_list=message_list, batch_cond_images=batch_cond_images_cache,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            outputs = self.generate(**model_inputs, decode_text=True, **kwargs)

            def wrap_with_special_tokens(output, cur_bot_task):
                if cur_bot_task == "think":
                    if not output.endswith(self._tokenizer.end_of_think_token):
                        output += self._tokenizer.end_of_think_token
                    return self._tokenizer.think_token + output
                else:
                    if not output.endswith(self._tokenizer.end_of_recaption_token):
                        output += self._tokenizer.end_of_recaption_token
                    return self._tokenizer.recaption_token + output

            cot_text = [wrap_with_special_tokens(output, cur_bot_task) for output in outputs.texts]
            if message_list is not None:
                for index, cot_text_i in enumerate(cot_text):
                    message_list[index].append(dict(role="assistant", type="text", content=cot_text_i))

            # Cot step 2
            if len(remain_bot_tasks) > 0:
                cur_bot_task = remain_bot_tasks[0]
                assert cur_bot_task == "recaption", f"Unexpected remaining bot task {cur_bot_task}."
                model_inputs = self.prepare_model_inputs(
                    mode="gen_text", bot_task=cur_bot_task, max_new_tokens=max_new_tokens,
                    prompt=prompt, image=image, cot_text=cot_text, system_prompt=system_prompt,
                    message_list=message_list, batch_cond_images=batch_cond_images_cache,
                )
                batch_cond_images_cache = model_inputs['batch_cond_images']
                outputs = self.generate(**model_inputs, decode_text=True, **kwargs)

                if self.generation_config.drop_think:
                    cot_text = [wrap_with_special_tokens(output, cur_bot_task) for output in outputs.texts]
                    # Switch system_prompt to drop_think_system_prompt
                    if system_prompt:
                        system_prompt = drop_think_system_prompt
                        if message_list is not None:
                            for i in range(len(message_list)):
                                message_list[i][0] = dict(role="system", type="text", content=drop_think_system_prompt)
                    if message_list is not None:
                        for index, cot_text_i in enumerate(cot_text):
                            message_list[index] = message_list[index][:-1] + [
                                dict(role="assistant", type="text", content=cot_text_i)
                            ]
                else:
                    cot_text_2 = [wrap_with_special_tokens(output, cur_bot_task) for output in outputs.texts]
                    cot_text = [
                        cot_text_i + cot_text_2_i
                        for cot_text_i, cot_text_2_i in zip(cot_text, cot_text_2)
                    ]
                    if message_list is not None:
                        for index, cot_text_2_i in enumerate(cot_text_2):
                            message_list[index].append(dict(role="assistant", type="text", content=cot_text_2_i))
        else:
            cot_text = kwargs.pop("cot_text", None)

        # Image ratio, supports per-sample auto resolution
        if image_size == "auto":
            self.image_processor.build_img_ratio_slice_logits_processor(self.tokenizer)
            model_inputs = self.prepare_model_inputs(
                seed=seed, mode="gen_text", bot_task="img_ratio", max_new_tokens=1,
                prompt=prompt, image=image, cot_text=cot_text, system_prompt=system_prompt,
                message_list=message_list, batch_cond_images=batch_cond_images_cache,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            outputs = self.generate(
                **model_inputs,
                do_sample=False,
                logits_processor=self.image_processor.img_ratio_slice_logits_processor,
                **kwargs,
            )
            bsz = outputs.texts.size(0)
            if bsz == 1:
                ratio_index = outputs.texts[0, -1].item()
                reso = self.image_processor.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            else:
                image_size = []
                for b in range(bsz):
                    ratio_index = outputs.texts[b, -1].item()
                    reso = self.image_processor.vae_reso_group[ratio_index]
                    image_size.append((reso.height, reso.width))

        # Generate image
        # if mode is "gen_image", add_assistant_prefix is False in apply_general_template, bot_task will not be used, so here we don't need to pass bot_task, let it be default value "auto"
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, image=image, cot_text=cot_text, message_list=message_list, system_prompt=system_prompt,
            seed=seed, image_size=image_size, mode="gen_image", batch_cond_images=batch_cond_images_cache,
        )
        batch_cond_images_cache = model_inputs['batch_cond_images']
        outputs = self.generate(**model_inputs, **kwargs)

        outputs.texts = cot_text
        outputs.images = self.image_processor.postprocess_outputs(outputs.images, batch_cond_images_cache)
        return outputs

    def init_dummy_interleaved_message_list(
            self,
            prompt: str | list[str] = None,
            image: list = None,
            message_list: Messages | list[Messages] = None,
    ) -> list[Messages]:
        """(batch=1) Build a OpenAI-style message_list for dummy interleaved mode (prompt or message_list).
        """
        if message_list is not None:
            ml = deepcopy(message_list)
            if isinstance(ml[0], dict):
                ml = [ml]
            return ml
        assert prompt is not None
        if isinstance(prompt, list):
            assert len(prompt) == 1, "dummy interleaved path only supports batch_size=1."
            prompt = prompt[0]
        if image is None:
            return [[{"role": "user", "content": prompt}]]
        image_list = image if isinstance(image, list) else [image]
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in image_list]
        content.append({"type": "text", "text": prompt})
        return [[{"role": "user", "content": content}]]
    
    def resolve_dummy_segments(
            self,
            segments: Optional[list[str]] = None,
            num_images: Optional[int] = None,
    ) -> Optional[list[str]]:
        """Merge `dummy_text_segments` and `dummy_num_images` into one segment list per image round."""
        n = None if num_images is None else int(num_images)
        seg_list = None if segments is None else list(segments)
        if seg_list is None and n is None:
            return None
        if n is not None and n <= 0:
            raise ValueError("`dummy_num_images` must be a positive int when set.")
        seg_list = [] if seg_list is None else seg_list
        if n is not None:
            if len(seg_list) < n:
                seg_list = seg_list + [""] * (n - len(seg_list))
            else:
                seg_list = seg_list[:n]
        if len(seg_list) == 0:
            raise ValueError(
                "dummy interleaved requires non-empty segments after merge, "
                "or set `dummy_num_images` > 0 with/without partial `dummy_text_segments`."
            )
        return seg_list

    # =====================================================================
    #  Interleaved multi-image-text generation  (batch_size = 1)
    #  TODO: 目前只支持batch_size = 1，batch推理需要考虑轮次不同/生成文本长度不同/
    #        图片生成的context长度不同，每个样本的message_list长度不同等问题
    # =====================================================================

    def generate_interleaved(
            self,
            prompt: str | list[str] = None,
            image: list = None,
            message_list: Messages | list[Messages] = None,
            seed: int | list[int] = None,
            image_size: str | tuple[int, int] = "auto",
            system_prompt: str = None,
            max_images: int = 9,
            max_new_tokens_per_round: int = 2048,
            use_system_prompt=None,
            bot_task: str = None,
            verbose: int = 0,
            dummy_text_segments: Optional[list[str]] = None,
            dummy_num_images: Optional[int] = None,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)
        system_prompt = get_system_prompt(use_system_prompt, bot_task, system_prompt)
        self.check_inputs(prompt, image, message_list)
        if message_list is not None:
            message_list = deepcopy(message_list)

        tkw = self._tokenizer
        conv_template = get_conversation_template(self.args.model_name.split('.')[-1])

        # Interleaved generation requires boi-token prediction
        gen_image_template = getattr(self.args, "gen_image_template", "default")
        assert gen_image_template == "default", (
            f"generate_interleaved requires gen_image_template='default' (with <boi>), "
            f"but got '{gen_image_template}'. The model must be trained with <boi> so it "
            f"can predict when to generate images."
        )

        # 用 <｜/hy_Assistant｜>（assistant 整段结束符）作为 interleaved loop 的终止 token。
        eot_token_id = tkw.convert_tokens_to_ids("<｜/hy_Assistant｜>")
        text_stop_tokens = [eot_token_id, tkw.boi_token_id]

        seeds = self.prepare_seed(seed, batch_size=1)

        all_texts: list[str] = []
        all_images: list[Image.Image] = []
        batch_cond_images_cache = None

        # inactive rank 的占位 image size（其生成结果会被丢弃），dummy / 非 dummy 路径共用。
        def _get_fallback_image_size():
            if self.image_processor.vae_reso_group is not None:
                reso = self.image_processor.vae_reso_group[0]
                return (reso.height, reso.width)
            base_size = getattr(self.image_processor, "reso_base_size", None) or getattr(self.args, "reso_base_size", 1024)
            return (base_size, base_size)

        def _uses_tw_th_auto_resolution():
            return (
                getattr(self.image_processor, "reso_strategy", None) == "anyres"
                and getattr(self.args, "add_tw_th_token", False)
            )

        def _tw_th_token_ids_to_image_size(th_token_id, tw_token_id):
            first_tw_th_token_id = self.tokenizer.tw_th_token_id(1)
            token_height = int(th_token_id) - first_tw_th_token_id + 1
            token_width = int(tw_token_id) - first_tw_th_token_id + 1
            return (
                token_height * self.image_processor.vae_info.h_factor,
                token_width * self.image_processor.vae_info.w_factor,
            )

        def _predict_tw_th_image_size(
                *,
                prompt_for_size=None,
                image_for_size=None,
                message_list_for_size=None,
                batch_cond_images_for_size=None,
        ):
            self.image_processor.build_img_tw_th_slice_logits_processor(self.tokenizer)
            tw_th_inputs = self.prepare_model_inputs(
                prompt=prompt_for_size,
                image=image_for_size,
                message_list=message_list_for_size,
                system_prompt=system_prompt,
                mode="gen_text",
                bot_task="img_tw_th",
                max_new_tokens=2,
                seed=seed,
                batch_cond_images=batch_cond_images_for_size,
            )
            tw_th_out = self.generate(
                **tw_th_inputs,
                do_sample=False,
                logits_processor=self.image_processor.img_tw_th_slice_logits_processor,
                **kwargs,
            )
            return tw_th_inputs["batch_cond_images"], _tw_th_token_ids_to_image_size(
                tw_th_out.texts[0, -2].item(),
                tw_th_out.texts[0, -1].item(),
            )

        # Dummy 多图生成路径：因为前期 boi 没打开训练，模型没有预测 boi 的能力，所以需要手动设定推理路径
        resolved_dummy_segments = self.resolve_dummy_segments(dummy_text_segments, dummy_num_images)
        if resolved_dummy_segments is not None:
            work_ml = self.init_dummy_interleaved_message_list(prompt, image, message_list)
            num_images_plan = min(max_images, len(resolved_dummy_segments))

            round_idx = 0
            while True:
                active_dummy = round_idx < num_images_plan
                if not _dp_any_active(active_dummy, self.device):
                    break

                if verbose >= 1 and active_dummy:
                    rank_print(
                        f"\n{'='*60}\n[Interleaved dummy BOI] Round {round_idx} / {num_images_plan - 1}\n{'='*60}",
                    )

                if active_dummy:
                    seg = resolved_dummy_segments[round_idx]
                    seg = seg if isinstance(seg, str) else str(seg)
                    if seg:
                        work_ml[0].append(dict(role="assistant", type="text", content=seg))
                    if seg.strip():
                        all_texts.append(seg.strip())

                cur_image_size = image_size
                if cur_image_size == "auto" and _uses_tw_th_auto_resolution():
                    batch_cond_images_cache, cur_image_size = _predict_tw_th_image_size(
                        message_list_for_size=work_ml,
                        batch_cond_images_for_size=batch_cond_images_cache,
                    )
                    if not active_dummy:
                        cur_image_size = _get_fallback_image_size()
                elif cur_image_size == "auto":
                    self.image_processor.build_img_ratio_slice_logits_processor(self.tokenizer)
                    ratio_inputs = self.prepare_model_inputs(
                        prompt=None,
                        image=None,
                        message_list=work_ml,
                        system_prompt=system_prompt,
                        mode="gen_text",
                        bot_task="img_ratio",
                        max_new_tokens=1,
                        seed=seed,
                        batch_cond_images=batch_cond_images_cache,
                    )
                    batch_cond_images_cache = ratio_inputs['batch_cond_images']
                    ratio_out = self.generate(
                        **ratio_inputs, do_sample=False,
                        logits_processor=self.image_processor.img_ratio_slice_logits_processor,
                        **kwargs,
                    )
                    if active_dummy:
                        ratio_index = ratio_out.texts[0, -1].item()
                        reso = self.image_processor.vae_reso_group[ratio_index]
                        cur_image_size = (reso.height, reso.width)
                    else:
                        cur_image_size = _get_fallback_image_size()

                if verbose >= 1 and active_dummy:
                    rank_print(
                        f"[Interleaved dummy BOI] Generating image {len(all_images)} at size {cur_image_size}",
                    )

                img_inputs = self.prepare_model_inputs(
                    prompt=None,
                    image=None,
                    message_list=work_ml,
                    system_prompt=system_prompt,
                    mode="gen_image",
                    image_size=cur_image_size,
                    seed=seed,
                    batch_cond_images=batch_cond_images_cache,
                )
                batch_cond_images_cache = img_inputs['batch_cond_images']
                img_inputs["generator"] = [
                    torch.Generator(self.device).manual_seed(seeds[0] + len(all_images))
                ]
                img_outputs = self.generate(
                    **img_inputs,
                    image_output_type="pil",
                    verbose=verbose,
                    **kwargs,
                )

                if active_dummy:
                    pil_image = img_outputs.images[0]
                    all_images.append(pil_image)

                    bot_sep = conv_template.pretrain_sep2 if self.generation_config.sequence_template == "pretrain" else conv_template.sep2
                    if bot_sep:
                        work_ml[0].append(dict(role="assistant", type="text", content=bot_sep))
                    work_ml[0].append(dict(
                        role="assistant",
                        content=[dict(type="image", image=pil_image)],
                    ))
                    batch_cond_images_cache = None

                round_idx += 1

            return MultimodalGenerationOutputs(
                texts=all_texts if all_texts else None,
                images=all_images if all_images else None,
            )

        # DP 同步循环：先完成的 rank 不直接 break，改为 inactive 继续陪跑相同次数的 generate()，
        # 输出丢弃，避免 FSDP collective 因各 rank forward 次数不同而卡住。
        active = True
        for round_idx in range(max_images + 1):
            if not _dp_any_active(active, self.device):
                break

            if verbose >= 1 and active:
                rank_print(
                    f"\n{'='*60}\n[Interleaved] Round {round_idx} — "
                    f"texts={len(all_texts)}, images={len(all_images)}\n{'='*60}",
                )

            # 1. Text generation (stop at <boi> or eos token)
            model_inputs = self.prepare_model_inputs(
                prompt=prompt,
                image=image,
                message_list=message_list,
                system_prompt=system_prompt,
                mode="gen_text",
                bot_task="auto",
                max_new_tokens=max_new_tokens_per_round,
                seed=seed,
                batch_cond_images=batch_cond_images_cache,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            model_inputs['eos_token_id'] = text_stop_tokens

            text_outputs = self.generate(
                **model_inputs, decode_text=False, verbose=verbose, **kwargs,
            )

            if active:
                generated_ids = text_outputs.texts  # (1, total_len) including prompt
                last_token_id = generated_ids[0, -1].item()
                stopped_by_boi = (last_token_id == tkw.boi_token_id)

                input_len = model_inputs['input_ids'].shape[1]
                new_ids = generated_ids[0, input_len:]
                # 若本轮以 <boi> 结束，把这个 trailing token 切掉：下一轮 image 生成的
                # chat template 会自己再插入一个 <boi> 前缀，避免序列里出现两个相邻 <boi>。
                if stopped_by_boi and new_ids.numel() > 0:
                    new_ids = new_ids[:-1]
                text = tkw.decode(new_ids, skip_special_tokens=False)
                if text.strip():
                    all_texts.append(text.strip())

                # Append generated text to message_list for next round's context
                if message_list is not None and text.strip():
                    if isinstance(message_list[0], dict):
                        message_list = [message_list]
                    message_list[0].append(
                        dict(role="assistant", type="text", content=text)
                    )

                if not stopped_by_boi or len(all_images) >= max_images:
                    active = False  # logical finish; keep running forwards for sync

            # If no rank still wants image gen this round, break together.
            if not _dp_any_active(active, self.device):
                break

            # 2. Predict image size
            cur_image_size = image_size
            if cur_image_size == "auto" and _uses_tw_th_auto_resolution():
                batch_cond_images_cache, cur_image_size = _predict_tw_th_image_size(
                    prompt_for_size=prompt,
                    image_for_size=image,
                    message_list_for_size=message_list,
                    batch_cond_images_for_size=batch_cond_images_cache,
                )
                if not active:
                    cur_image_size = _get_fallback_image_size()
            elif cur_image_size == "auto":
                self.image_processor.build_img_ratio_slice_logits_processor(self.tokenizer)
                ratio_inputs = self.prepare_model_inputs(
                    prompt=prompt, image=image,
                    message_list=message_list,
                    system_prompt=system_prompt,
                    mode="gen_text",
                    bot_task="img_ratio",
                    max_new_tokens=1,
                    seed=seed,
                    batch_cond_images=batch_cond_images_cache,
                )
                batch_cond_images_cache = ratio_inputs['batch_cond_images']
                ratio_out = self.generate(
                    **ratio_inputs, do_sample=False,
                    logits_processor=self.image_processor.img_ratio_slice_logits_processor,
                )
                if active:
                    ratio_index = ratio_out.texts[0, -1].item()
                    reso = self.image_processor.vae_reso_group[ratio_index]
                    cur_image_size = (reso.height, reso.width)
                else:
                    cur_image_size = _get_fallback_image_size()

            if verbose >= 1 and active:
                rank_print(f"[Interleaved] Generating image {len(all_images)} at size {cur_image_size}")

            # 3. image generation
            img_inputs = self.prepare_model_inputs(
                prompt=prompt, image=image,
                message_list=message_list,
                system_prompt=system_prompt,
                mode="gen_image",
                image_size=cur_image_size,
                seed=seed,
                batch_cond_images=batch_cond_images_cache,
            )
            batch_cond_images_cache = img_inputs['batch_cond_images']
            img_inputs["generator"] = [
                torch.Generator(self.device).manual_seed(seeds[0] + len(all_images))
            ]
            img_outputs = self.generate(
                **img_inputs,
                image_output_type="pil",
                verbose=verbose,
            )

            if active:
                pil_image = img_outputs.images[0]
                all_images.append(pil_image)

                # 4. Re-inject image as condition for next round
                # Mimic training sequence structure:
                #   ... [gen_text] <ts> <img>×N [bot_sep] [cond_joint_image] [next_text] ...
                if message_list is not None:
                    # bot_sep 应该能自动推理出来
                    bot_sep = conv_template.pretrain_sep2 if self.generation_config.sequence_template == "pretrain" else conv_template.sep2
                    if bot_sep:
                        message_list[0].append(
                            dict(role="assistant", type="text", content=bot_sep)
                        )
                    message_list[0].append(dict(
                        role="assistant",
                        content=[dict(type="image", image=pil_image)],
                    ))
                elif prompt is not None:
                    if image is None:
                        image = []
                    elif not isinstance(image, list):
                        image = list(image)
                    image.append(pil_image)

                # Force re-build cond images on next round (new image added).
                # Inactive ranks keep the cached value since their message_list is frozen.
                batch_cond_images_cache = None

        return MultimodalGenerationOutputs(
            texts=all_texts if all_texts else None,
            images=all_images if all_images else None,
        )

    def print_info(self, info_list):
        max_key_len = max(len(k) for k, _ in info_list)
        info_str = "=" * 50 + \
                    f"\nModel input info:\n" + \
                    "\n".join([f"    {k.rjust(max_key_len)}: {v}" for k, v in info_list]) + \
                    "\n--------------------------------------------------"
        rank_print_multiline(info_str)
