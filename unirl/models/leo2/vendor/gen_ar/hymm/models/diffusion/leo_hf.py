import json
import random
import re
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, Union, TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import dispatch_model
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel, GenerationConfig
try:
    from transformers.modeling_utils import PretrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import ModelOutput

# fix new transformers and diffusers compatibility issue
import transformers.utils as _transformers_utils
if not hasattr(_transformers_utils, "FLAX_WEIGHTS_NAME"):
    _transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

from hymm.ar.pipelines.pipeline_leo import Leo2Pipeline
from hymm.core.global_vars import get_logger
from hymm.diffusion.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hymm.data_kits.system_prompt import get_system_prompt
from hymm.data_kits.utils.image_utils import ImageProcessor
from hymm.data_kits.utils.video_utils import (
    VideoProcessor,
    merge_qwen3vl_video_grid_thw,
    normalize_cond_video_vae_paths,
    open_video,
    sample_video_frames,
)
from hymm.models.utils.generation_utils import MultimodalGenerationOutputs
from hymm.models.tokenizers.conversation import get_conversation_template
from hymm.models.autoencoders import normalize_vae_latents
from hymm.utils.helpers import default
from hymm.utils.audio_base import AudioInfo
from hymm.utils.image_base import ImageInfo, ImageTensor, CondImage
from hymm.utils.video_base import VideoInfo
from hymm.utils.import_utils import is_package_version
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.utils.validation_loss_utils import (parse_latent_shape, latent_shape_to_media_size, load_video_latent,
                                              load_audio_latent, build_av_denoiser)
from .leo import LeoModelBase
from .leo_config import LeoConfig
from ..multimodal.hunyuan_multimodal_hf import to_device
from ...data_kits.utils.audio_utils import AudioProcessor
from hymm.models.autoencoders import add_noise_and_extend_channel, _add_noise

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

InputImage = Optional[Union[Image.Image, str, bytes]]
Messages = list[dict[str, Any]]


class Leo2GenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.diff_infer_steps = kwargs.pop("diff_infer_steps", 50)
        self.diff_guidance_scale = kwargs.pop("diff_guidance_scale", 5.0)
        self.diff_guidance_scale_audio = kwargs.pop("diff_guidance_scale_audio", None)
        self.flow_shift = kwargs.get("flow_shift", 3.0)
        self.flow_shift_video = kwargs.get("flow_shift_video", self.flow_shift)
        self.flow_shift_audio = kwargs.get("flow_shift_audio", self.flow_shift)
        self.use_system_prompt = kwargs.get("use_system_prompt", None)
        self.bot_task = kwargs.get("bot_task", "image")
        self.sequence_template = kwargs.get("sequence_template", "instruct")
        self.ref_mode = kwargs.get("ref_mode", "sequence")  # or "channel"
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 48000)


class Leo2HFConfig(PretrainedConfig):
    def __init__(self, hf_config):
        super().__init__()
        for key, value in hf_config.items():
            setattr(self, key, value)


class Leo2PreTrainedModel(PreTrainedModel):
    config_class = LeoConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["LeoDualLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True


class LeoModelHF(LeoModelBase, Leo2PreTrainedModel, GenerationMixin):
    def __init__(
            self,
            args: Namespace,
            config: LeoConfig,
            txt_config: Optional[LeoConfig] = None,
            audio_config: Optional[LeoConfig] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            initialize_weights: bool = True,
    ):
        hf_config = Leo2HFConfig(config.to_hf_config())
        super().__init__(hf_config)
        self.args = args
        self._dtype = dtype
        self.config = hf_config
        self.__post_init__(config, txt_config, audio_config, dtype, device, args, initialize_weights)

        # Initialize image processor
        self.image_processor = ImageProcessor(args) if "vae_image" in args.modality else None
        self.video_processor = VideoProcessor(args) if "vae_video" in args.modality else None
        self.audio_processor = AudioProcessor(args) if "vae_audio" in args.modality else None

        self._tokenizer = None
        self._diffusion_pipeline = None
        self.vae_autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]
        # Use model_dict instead of directly assigning attributes to avoid unintentionally registering them
        # as submodules of the model, which may cause issues for applying FSDP for both self and these extra models.
        self.model_dict = dict(vae=None, text_encoder=None, audio_vae=None)
        # Get Rank0 logger
        self.logger = get_logger()

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

    def build_diffusion_pipeline(self):
        if self._diffusion_pipeline is None:
            scheduler_config = self.args
            kwargs = dict(
                reverse=scheduler_config.flow_reverse,
                solver=scheduler_config.flow_solver,
                use_flux_shift=scheduler_config.use_flux_shift,
                flux_base_num_tokens=scheduler_config.flux_base_num_tokens,
                flux_base_log_shift=default(scheduler_config.flux_base_log_shift, scheduler_config.flux_base_shift),
                flux_max_num_tokens=scheduler_config.flux_max_num_tokens,
                flux_max_log_shift=default(scheduler_config.flux_max_log_shift, scheduler_config.flux_max_shift),
                start_sigma=scheduler_config.flow_start_sigma,
                end_sigma=scheduler_config.flow_end_sigma,
            )
            scheduler = FlowMatchDiscreteScheduler(shift=self.generation_config.flow_shift, **kwargs)
            video_scheduler = FlowMatchDiscreteScheduler(shift=self.generation_config.flow_shift_video, **kwargs)
            audio_scheduler = FlowMatchDiscreteScheduler(shift=self.generation_config.flow_shift_audio, **kwargs)
            assert self.model_dict["vae"] is not None, "VAE must be initialized before building diffusion pipeline."
            assert self.model_dict["text_encoder"] is not None, \
                "text_encoder must be initialized before building diffusion pipeline."
            self._diffusion_pipeline = Leo2Pipeline(
                model=self,
                scheduler=scheduler,
                vae=self.model_dict["vae"],
                text_encoder=self.model_dict["text_encoder"],
                args=self.args,
                video_scheduler=video_scheduler,
                audio_scheduler=audio_scheduler,
                audio_processor=self.audio_processor,
                audio_vae=self.model_dict["audio_vae"],
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
        config_keys = Leo2GenerationConfig().to_dict().keys()
        overrides = {}
        for key in config_keys:
            if getattr(self.args, key, None) is not None:
                overrides[key] = getattr(self.args, key)

        self.logger.info(f"Loading generation config from {config_dir} with overrides: {overrides}")
        self.generation_config = Leo2GenerationConfig.from_pretrained(
            config_dir, config_file_name=config_file_name, **overrides,
        )
        self.logger.info(f"Generation config: {self.generation_config}")

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
    def _validate_and_batchify_text(text, name, check_batch_size=None, allow_expand=False):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), \
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        if isinstance(text, str):
            text = [text]
        assert len(text) > 0 and all(isinstance(p, str) and len(p) > 0 for p in text), \
            f"Input `{name}` should be a non-empty list of non-empty strings, got {text}."
        if check_batch_size is not None:
            if len(text) != check_batch_size:
                if allow_expand:
                    assert len(text) == 1, \
                        f"Input `{name}` should have only one element to allow expansion, got {len(text)}."
                    text = text * check_batch_size
                else:
                    raise ValueError(f"Input `{name}` should have the same batch size as other "
                                     f"inputs({check_batch_size}), got {len(text)}.")
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

    def build_batch_rope_media_info(self, output, sections, mode, num_channel_cond_images=0):
        assert self.config.rope_type in ["2d", "3d", "leo2_3d"], \
            f"Rope type {self.config.rope_type} not supported by method 'build_batch_rope_image_info'."
        rope_media_info = []
        for media_slices, sections_i in zip(output.all_media_slices, sections):
            rope_media_slices = []
            rope_media_shapes = []
            rope_media_metas = []
            media_idx = 0

            for section in sections_i:
                if section['type'] in [
                    "gen_image", "cond_vae_image", "cond_vit_image",
                    "gen_video", "cond_vae_video", "cond_vit_video",
                    "gen_audio",
                ]:
                    assert media_idx < len(media_slices), \
                        f"Image index {media_idx} out of range for image slices with length {len(media_slices)}."
                    rope_media_slices.append(media_slices[media_idx])

                    if section['type'] in ["gen_image", "cond_vae_image", "cond_vit_image"]:
                        if self.config.rope_type in ["3d", "leo2_3d"]:
                            rope_media_shapes.append((1, section['token_height'], section['token_width']))
                        else:
                            rope_media_shapes.append((section['token_height'], section['token_width']))
                    elif section['type'] in ["gen_video", "cond_vae_video", "cond_vit_video"]:
                        # Source-video conditions use 3D RoPE: cond_vae_video uses its VAE latent grid
                        # (token_d, h, w); one tubelet per section.
                        rope_media_shapes.append((section['token_duration'], section['token_height'], section['token_width']))
                    elif section['type'] == "gen_audio":
                        rope_media_shapes.append((section['token_length'], 1, 1))
                    else:
                        raise ValueError(f"Unknown media type {section['type']} for RoPE media info.")

                    meta_dict = {"type": section['type']}
                    # Tubelets of a native condition video share a RoPE base and advance along time.
                    for key in ("temporal_index", "temporal_length", "video_id", "timestamp"):
                        if key in section:
                            meta_dict[key] = section[key]
                    if mode == "gen_av":
                        if section['type'] == "gen_video":
                            meta_dict['with_audio'] = True
                            meta_dict["dataset_tag"] = {0: "t2va", 1: "i2va", 2: "fl2va"}[num_channel_cond_images]
                        elif section['type'] == "gen_audio":
                            meta_dict['with_video'] = True
                    if mode == "gen_audio" and section['type'] == "gen_audio":
                        meta_dict["rope_audio_rescale_factor"] = getattr(self.args, "rope_audio_rescale_factor", 1.0)

                    rope_media_metas.append(meta_dict)
                    media_idx += 1

                elif section['type'] == "cond_joint_image":
                    # We assume `joint` means two image features.
                    assert media_idx + 1 < len(media_slices), \
                        f"Image index {media_idx + 1} out of range for image slices with length {len(media_slices)}."
                    assert len(section['token_height']) == len(section['token_width']), \
                        (f"token_height and token_width should have the same length, "
                         f"but got {len(section['token_height'])} and {len(section['token_width'])}")

                    rope_media_slices.extend([media_slices[media_idx], media_slices[media_idx + 1]])
                    if self.config.rope_type in ["3d", "leo2_3d"]:
                        rope_media_shapes.extend(
                            list(zip([1, 1], section['token_height'], section['token_width']))
                        )
                    else:
                        rope_media_shapes.extend(list(zip(section['token_height'], section['token_width'])))
                    rope_media_metas.extend([{"type": section['type']} for i in range(2)])
                    media_idx += 2

            rope_media_info.append(list(zip(rope_media_slices, rope_media_shapes, rope_media_metas)))

        return rope_media_info

    def vae_encode(self, image, cfg_factor=1, keep_depth=True, vae_encode_type="sample"):

        with torch.autocast(
                device_type="cuda", dtype=self.vae_autocast_dtype,  # noqa
                enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32
        ):
            vae_encode_result = self.model_dict["vae"].encode(image)
            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            elif vae_encode_type == "sample":
                latents = vae_encode_result.latent_dist.sample()
            elif vae_encode_type == "mode":
                latents = vae_encode_result.latent_dist.mode()
            else:
                raise ValueError(f"Unknown vae_encode_type: {vae_encode_type}")

            latents = normalize_vae_latents(self.model_dict["vae"], latents)

        if hasattr(self.model_dict["vae"], "ffactor_temporal") and not keep_depth:
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
            vae_encode_type: Literal["sample", "mode"] = "sample",
    ):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None, None, None

        first_image = batch_cond_images[0][0]

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
                        vae_image[None].to(self.device), vae_encode_type=vae_encode_type,
                    )
                    cond_vae_image_list.append(cond_vae_image_.squeeze(0))
                    cond_t_list.append(cond_t_)
                batch_cond_vae_images.append(cond_vae_image_list)
                batch_cond_t.append(cond_t_list)

            # If only one cond image for each sample and all have the same size, we can batch them together
            # In this case, cond_vae_images is a 5-D tensor.
            if all([len(items) == 1 for items in batch_cond_vae_images]) and all(
                    items[0].shape == batch_cond_vae_images[0][0].shape for items in batch_cond_vae_images):
                cond_vae_images = torch.stack([items[0] for items in batch_cond_vae_images], dim=0)
                cond_t = torch.cat([items[0] for items in batch_cond_t], dim=0)
                if cfg_factor > 1:
                    cond_t = cond_t.repeat(cfg_factor)
                    remain_dims = (1,) * (cond_vae_images.ndim - 1)
                    cond_vae_images = cond_vae_images.repeat(cfg_factor, *remain_dims)
            else:
                # In this case, cond_vae_images is a list of 5-D tensors or a list of lists of 4-D tensors.
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
    def _get_vit_image(cond_image):
        if isinstance(cond_image, CondImage):
            return cond_image.vit_image
        if isinstance(cond_image, ImageTensor):
            return cond_image
        return None

    @staticmethod
    def _prepare_vit_image_kwargs(batch_cond_images, cfg_factor):
        if batch_cond_images is None or len(batch_cond_images[0]) == 0:
            return None
        vit_image = LeoModelHF._get_vit_image(batch_cond_images[0][0])
        if vit_image is None:
            return None
        if not hasattr(vit_image, "vision_encoder_kwargs") or len(vit_image.vision_encoder_kwargs) == 0:
            return None

        image_type = vit_image.i.image_type
        if image_type == "qwen3vl":
            cond_vit_image_kwargs = {"grid_thw": []}
            for cond_images in batch_cond_images:
                cond_vit_image_kwargs["grid_thw"].append(torch.stack([
                    LeoModelHF._get_vit_image(cond_image).vision_encoder_kwargs["grid_thw"]
                    for cond_image in cond_images
                ]))
            if cfg_factor > 1:
                cond_vit_image_kwargs["grid_thw"] = cond_vit_image_kwargs["grid_thw"] * cfg_factor
        else:
            # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
            cond_vit_image_kwargs = {"spatial_shapes": [], "attention_mask": []}
            for cond_images in batch_cond_images:
                cond_vit_image_kwargs["spatial_shapes"].append(
                    torch.stack([
                        LeoModelHF._get_vit_image(cond_image).vision_encoder_kwargs["spatial_shapes"]
                        for cond_image in cond_images
                    ]))
                cond_vit_image_kwargs["attention_mask"].append(
                    torch.stack([
                        LeoModelHF._get_vit_image(cond_image).vision_encoder_kwargs["pixel_attention_mask"]
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
            ref_mode: Literal["sequence", "channel"] = "sequence",
            gen_image_info: ImageInfo = None,
            gen_video_info: VideoInfo = None,
            gen_audio_info: AudioInfo = None,
            cond_videos: list[dict] = None,
            cond_after_gen: bool = False,
    ):
        assert gen_image_info is None or gen_video_info is None, \
            (f"gen_image_info and gen_video_info cannot be provided at the same time, "
             f"but got {gen_image_info} and {gen_video_info}.")
        # Notice: gen_video_info and gen_audio_info are allowed presented at the same time.

        inner_message_list = []
        image_idx = 0
        video_idx = 0
        for message in message_list:
            content = message["content"]
            if isinstance(content, str):
                inner_message_list.append(dict(role=message["role"], type="text", content=content))
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        inner_message_list.append(dict(role=message["role"], type="text", content=item['text']))
                    elif item["type"] == "video":
                        if all(key not in item for key in ["video", "url", "path"]):
                            continue
                        if ref_mode == "channel":
                            continue
                        if cond_videos is not None and video_idx < len(cond_videos):
                            cv = cond_videos[video_idx]
                            if cv.get("vit_info") is not None:
                                inner_message_list.append(dict(
                                    role=message["role"], type="cond_vit_video", content=cv["vit_info"]))
                            if cv.get("vae_info") is not None:
                                inner_message_list.append(dict(
                                    role=message["role"], type="cond_vae_video", content=cv["vae_info"]))
                        video_idx += 1
                    elif item["type"] == "image":
                        if all(key not in item for key in ["image", "url", "path", "base64"]):
                            continue
                        if ref_mode == "channel":
                            # Cond images in channel-concat mode will be processed in pipeline.
                            continue
                        # Here we only accept sequence-concat conditions.
                        assert cond_images is not None and image_idx < len(cond_images), \
                            f"Image index {image_idx} out of range for cond images with length {len(cond_images)}."
                        image = cond_images[image_idx]
                        # r2v pack mode expands `cond_joint_image` into two adjacent sections
                        # (cond_vit_image + cond_vae_image) so the layout matches Qwen3-VL's native
                        # `<vision_start><image_pad>*N<vision_end>` block.
                        if self.image_processor.cond_image_section_type == "cond_joint_image":
                            inner_message_list.append(dict(
                                role=message["role"], type="cond_vit_image", content=image.vit_image.i,
                            ))
                            inner_message_list.append(dict(
                                role=message["role"], type="cond_vae_image", content=image.vae_image.i,
                            ))
                        else:
                            inner_message_list.append(dict(
                                role=message["role"], type=self.image_processor.cond_image_section_type, content=image.i,
                            ))
                        image_idx += 1
                    else:
                        raise NotImplementedError(f"Message content type {item['type']} not supported.")
            else:
                raise ValueError(f"Message content should be str or list, but got {type(content)}.")

        if gen_image_info is not None:
            inner_message_list.append(dict(role="assistant", type="gen_image", content=gen_image_info))

        if gen_video_info is not None:
            inner_message_list.append(
                dict(role="assistant", type="gen_video", content=gen_video_info, with_audio=gen_audio_info is not None)
            )

        # Make sure `gen_audio` is after the `gen_video` section if both presented.
        if gen_audio_info is not None:
            inner_message_list.append(
                dict(role="assistant", type="gen_audio", content=gen_audio_info, with_video=gen_video_info is not None)
            )
        if cond_after_gen:
            cond_ref_types = ("cond_vit_image", "cond_vae_image", "cond_vit_video", "cond_vae_video")
            cond_ref_msgs = [m for m in inner_message_list if m["type"] in cond_ref_types]
            rest_msgs = [m for m in inner_message_list if m["type"] not in cond_ref_types]
            inner_message_list = rest_msgs + cond_ref_msgs

        return inner_message_list

    def _build_batch_gen_video_info(self, video_size, num_frames, batch_size, video_token_grid=None):
        # Support variable resolution, i.e, video_size is a list of (height, width) pairs
        if isinstance(video_size, list):
            assert len(video_size) == batch_size, \
                f"video_size should have the same length as batch_size, got {len(video_size)} and {batch_size}"
            return [self.video_processor.build_gen_video_info(video_size[i], num_frames, token_grid=video_token_grid)
                    for i in range(batch_size)]
        return [self.video_processor.build_gen_video_info(video_size, num_frames, token_grid=video_token_grid)
                for _ in range(batch_size)]

    # =========================================================================
    # r2v source-video conditioning (inference)
    # =========================================================================
    @staticmethod
    def _extract_video_paths_from_messages(message_list):
        """Extract source-video file paths from an OpenAI-style message list (in order)."""
        paths = []
        for message in message_list:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    for key in ["video", "url", "path"]:
                        if key in item:
                            paths.append(item[key])
                            break
        return paths

    @staticmethod
    def _normalize_batch_vae_paths(cond_video_vae_path, batch_size):
        """Normalize optional offline VAE paths into one ordered list per sample."""
        return normalize_cond_video_vae_paths(cond_video_vae_path, batch_size)

    def _sample_video_vit_frames(self, video_path, num_frames, video_id=0):
        """Encode one condition clip with the native Qwen video processor."""
        if self.video_processor is None or not hasattr(self.video_processor, "vit_video_info"):
            raise ValueError(
                "ViT condition videos require a VideoProcessor built with 'vit_video' in modality."
            )
        video_tensor, _ = self.video_processor.vit_process_video_frames(
            video_path, num_frames=num_frames, video_id=video_id,
        )
        return video_tensor

    @torch.inference_mode()
    def _encode_cond_video_vae_online(self, video_path):
        """Online VAE-encode a source video to a normalized (C, T, H, W) latent (model latent space).

        Each clip keeps its own shape, the same way reference images do via
        `get_image_with_size(target_size_type="image")`: its native resolution and frame count pick
        the VAE ratio / duration buckets, independently of the video being generated. Training
        behaves the same -- condition latents are read offline at whatever shape they were
        extracted with, and every latent is scattered against its own slice.
        """
        import torch.nn.functional as F
        vae = self.model_dict["vae"]
        device = next(vae.parameters()).device
        if self.video_processor is None or not hasattr(self.video_processor, "vae_dar_group"):
            raise ValueError(
                "Online condition-video VAE encoding requires a VideoProcessor built with "
                "'vae_video' in modality."
            )
        d_factor = self.config.vae_temporal_downsample_factor
        # The source resolution picks the duration bucket, so the clip is probed before its
        # frame count is known; the reader is reused so the file is only opened once.
        reader, meta = open_video(video_path)
        width, height, duration = self.video_processor.vae_dar_group.get_target_size(
            meta.width, meta.height, meta.total_frames
        )
        frames = sample_video_frames(reader, min(duration, meta.total_frames)).frames
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
        n = x.shape[0]
        r = (n - 1) % d_factor
        if r:
            x = x[:n - r]
        if x.shape[0] < duration:
            x = torch.cat([x, x[-1:].expand(duration - x.shape[0], -1, -1, -1)], dim=0)
        x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)
        video = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, 3, T, H, W)
        with torch.autocast(device_type="cuda", dtype=self.vae_autocast_dtype,
                            enabled=self.vae_autocast_dtype is not None and self.vae_autocast_dtype != torch.float32):
            enc = vae.encode(video)
        latents = enc if isinstance(enc, torch.Tensor) else enc.latent_dist.mode()
        # Freshly-encoded latents are always raw; bring them into the model's normalized latent
        # space (channel-wise latent_norm_stats or config scaling), same as reference images.
        latents = normalize_vae_latents(vae, latents)
        return latents.squeeze(0).float()  # (C, T, H, W)

    def _build_cond_videos(self, batch_message_list, cond_video_vae_path):
        batch_size = len(batch_message_list)
        cond_video_type = getattr(self.args, "cond_video_type", "none")
        if cond_video_type == "none":
            return [None] * batch_size
        vae = self.model_dict["vae"]
        num_frames_vit = getattr(self.args, "cond_video_vit_num_frames", None)
        batch_vae_paths = self._normalize_batch_vae_paths(cond_video_vae_path, batch_size)
        batch_cond_videos = []
        for b in range(batch_size):
            video_paths = self._extract_video_paths_from_messages(batch_message_list[b])
            sample_vae_paths = batch_vae_paths[b]
            if sample_vae_paths and not video_paths:
                raise ValueError(
                    f"cond_video_vae_path was provided for sample {b}, but its message_list has no video."
                )
            if sample_vae_paths and len(sample_vae_paths) != len(video_paths):
                raise ValueError(
                    "Offline cond_video_vae_path count must match the sample's condition videos: "
                    f"{len(sample_vae_paths)} latents for {len(video_paths)} videos in sample {b}."
                )
            cv_list = []
            for video_id, vp in enumerate(video_paths):
                entry = {}
                if "vit" in cond_video_type:
                    video_tensor = self._sample_video_vit_frames(vp, num_frames_vit, video_id=video_id)
                    entry["vit_video"] = video_tensor
                    entry["vit_info"] = video_tensor.i
                if "vae" in cond_video_type:
                    vae_path = sample_vae_paths[video_id] if sample_vae_paths else None
                    if vae_path:
                        lat = np.load(vae_path)
                        if lat.ndim == 5:
                            if lat.shape[0] != 1:
                                raise ValueError(
                                    f"Expected cond VAE batch dimension 1, got {lat.shape} from {vae_path}."
                                )
                            lat = lat.squeeze(0)
                        if lat.ndim != 4:
                            raise ValueError(
                                f"Expected cond VAE shape (C,T,H,W), got {lat.shape} from {vae_path}."
                            )
                        expected_channels = int(getattr(self.args, "vae_latent_dim", 48))
                        if lat.shape[0] != expected_channels:
                            raise ValueError(
                                f"Expected {expected_channels} cond VAE channels, got "
                                f"{lat.shape[0]} from {vae_path}."
                            )
                        # Like reference images, a condition video keeps the shape it was extracted
                        # with: clips may differ from each other and from the generated video, and
                        # each is scattered against its own slice.
                        if any(int(dim) <= 0 for dim in lat.shape[1:]):
                            raise ValueError(
                                f"Condition-video latent has an empty token grid {tuple(lat.shape)} "
                                f"in {vae_path}."
                            )
                        lat = torch.from_numpy(lat).float()
                        lat = normalize_vae_latents(vae, lat.unsqueeze(0)).squeeze(0)
                    else:
                        lat = self._encode_cond_video_vae_online(vp)
                    c, t, h, w = lat.shape
                    entry["vae_latent"] = lat
                    entry["vae_info"] = VideoInfo(
                        video_type="vae", token_duration=int(t), token_height=int(h), token_width=int(w),
                    )
                if entry:
                    cv_list.append(entry)
            batch_cond_videos.append(cv_list)
        return batch_cond_videos

    def _merge_cond_videos(self, batch_cond_videos, cond_vae_images, cond_vit_images,
                           cond_vit_image_kwargs, cfg_factor):
        if not batch_cond_videos or all(not cv for cv in batch_cond_videos):
            return cond_vae_images, cond_vit_images, cond_vit_image_kwargs, None, None
        bsz = len(batch_cond_videos)
        n_total = bsz * cfg_factor

        # ---- VAE latents: normalize to list-of-list per (cfg*bsz), then append source-video latents ----
        has_src_vae = any(cv and any("vae_latent" in e for e in cv) for cv in batch_cond_videos)
        if has_src_vae:
            if cond_vae_images is None:
                vae_lists = [[] for _ in range(n_total)]
            elif isinstance(cond_vae_images, torch.Tensor):
                vae_lists = [[cond_vae_images[i]] for i in range(cond_vae_images.size(0))]
            else:  # list per (cfg*bsz): each item is a (n,C,H,W) tensor or a list of tensors
                vae_lists = [
                    [item[j] for j in range(item.size(0))] if isinstance(item, torch.Tensor) else list(item)
                    for item in cond_vae_images
                ]
            assert len(vae_lists) == n_total, \
                f"cond_vae_images length {len(vae_lists)} != cfg*bsz {n_total}"
            for cfg_i in range(cfg_factor):
                for b in range(bsz):
                    idx = cfg_i * bsz + b
                    for e in (batch_cond_videos[b] or []):
                        if "vae_latent" in e:
                            vae_lists[idx].append(e["vae_latent"].to(self.device))
            cond_vae_images = vae_lists

        # ---- Native Qwen videos keep their own pixels and grid, separate from images ----
        has_src_vit = any(cv and any("vit_video" in e for e in cv) for cv in batch_cond_videos)
        cond_vit_videos = None
        cond_vit_video_kwargs = None
        if has_src_vit:
            cond_vit_videos = [[] for _ in range(n_total)]
            video_grid_list = [None] * n_total
            for cfg_i in range(cfg_factor):
                for b in range(bsz):
                    idx = cfg_i * bsz + b
                    grids = []
                    for e in (batch_cond_videos[b] or []):
                        video_tensor = e.get("vit_video")
                        if video_tensor is None:
                            continue
                        cond_vit_videos[idx].append(video_tensor)
                        grids.append(merge_qwen3vl_video_grid_thw(video_tensor))
                    if grids:
                        video_grid_list[idx] = torch.cat(grids, dim=0)
            cond_vit_video_kwargs = {"video_grid_thw": video_grid_list}

        return (
            cond_vae_images,
            cond_vit_images,
            cond_vit_image_kwargs,
            cond_vit_videos,
            cond_vit_video_kwargs,
        )

    @staticmethod
    def _merge_cond_vae_masks(image_mask, video_mask):
        """Union of reference-image and source-video VAE placeholder masks (either may be None)."""
        masks = [m for m in (image_mask, video_mask) if m is not None]
        if not masks:
            return None
        merged = masks[0].bool()
        for m in masks[1:]:
            merged = merged | m.bool()
        return merged.to(masks[0].dtype)

    def prepare_model_inputs(
            self,
            prompt: str | list[str] = None,
            image: list[InputImage] = None,
            mode="gen_text",
            system_prompt: Optional[str] = None,        # for ar
            use_system_prompt: Optional[str] = None,    # for dit
            cot_text: str | list[str] = None,
            media_size: Optional[str | tuple] = None,
            num_frames: Optional[int] = None,
            video_fps: Optional[int] = None,
            audio_duration: Optional[float] = None,
            ref_mode: Literal["sequence", "channel"] = "sequence",
            message_list: Optional[Messages | list[Messages]] = None,
            device=None,
            bot_task="auto",
            conv_template=None,
            audio_token_length: Optional[int] = None,
            video_token_grid: Optional[tuple] = None,
            cond_after_gen: bool = False,
            **kwargs,
    ):
        args = self.args

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
        batch_channel_cond_images = kwargs.get('batch_channel_cond_images', None)

        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_size = len(batch_message_list)

            # message_list may be modified later, so we deepcopy it here to avoid side effects
            batch_message_list = deepcopy(batch_message_list)

            # Prepend system prompt if available
            batch_system_prompt = self._validate_and_batchify_text(
                batch_system_prompt, 'system_prompt', batch_size, allow_expand=True,
            )
            if batch_system_prompt is not None:
                batch_message_list = [
                    [dict(role="system", type="text", content=sp)] + message_list_
                    for sp, message_list_ in zip(batch_system_prompt, batch_message_list)
                ]

            # Multiple cond images are allowed.
            if batch_cond_images is None and batch_channel_cond_images is None:
                _batch_cond_images = [
                    self.image_processor.build_cond_images(message_list=message_list_)
                    for message_list_ in batch_message_list
                ]
                if ref_mode == "channel":
                    batch_channel_cond_images = _batch_cond_images
                    batch_cond_images = [[] for _ in range(batch_size)]
                    # For fl2v, we must extend extra frames to the end of gen video.
                    if len(batch_channel_cond_images[0]) == 2:
                        # tk_duration += 1
                        num_frames = num_frames + self.config.vae_temporal_downsample_factor
                else:   # sequence
                    batch_channel_cond_images = [[] for _ in range(batch_size)]
                    batch_cond_images = _batch_cond_images

            # For generated medias.
            batch_gen_image_info: list[Optional[ImageInfo]] = [None] * batch_size
            batch_gen_video_info: list[Optional[VideoInfo]] = [None] * batch_size
            batch_gen_audio_info: list[Optional[AudioInfo]] = [None] * batch_size

            if mode == "gen_image":
                batch_gen_image_info = [
                    self.image_processor.build_gen_image_info(media_size) for _ in range(batch_size)
                ]
            elif mode in ["gen_video", "gen_av"]:
                # Check media_size when using channel-concat references.
                if ref_mode == "channel" and len(batch_channel_cond_images[0]) > 0:
                    media_size = [
                        f"{cond_images[0].i.h}x{cond_images[0].i.w}"
                        for cond_images in batch_channel_cond_images
                    ]
                batch_gen_video_info = self._build_batch_gen_video_info(
                    media_size, num_frames, batch_size, video_token_grid=video_token_grid)

            if mode in ["gen_audio", "gen_av"]:
                audio_duration_by_seconds = num_frames / video_fps if mode == "gen_av" else audio_duration
                batch_gen_audio_info = [
                    self.audio_processor.build_gen_audio_info(
                        audio_duration_by_seconds, token_duration=audio_token_length
                    )
                    for _ in range(batch_size)
                ]

            batch_cond_videos = self._build_cond_videos(
                batch_message_list, kwargs.get("cond_video_vae_path"),
            )

            # Convert OpenAI message list into inner message list.
            # Channel-concat cond images will not be added into message_list, and will be processed in pipeline.
            batch_message_list = [
                self.prepare_message_list(
                    message_list_, cond_images,
                    ref_mode=ref_mode,
                    # If gen_image_template is "dit", we won't add gen_image section, because dit uses text encoder,
                    # which only takes conditional sections as input.
                    gen_image_info=None if args.gen_template == "dit_it" else gen_image_info,
                    gen_video_info=None if args.gen_template == "dit_it" else gen_video_info,
                    gen_audio_info=None if args.gen_template == "dit_it" else gen_audio_info,
                    cond_videos=cond_videos,
                    cond_after_gen=cond_after_gen,
                )
                for message_list_, cond_images, gen_image_info, gen_video_info, gen_audio_info, cond_videos in zip(
                    batch_message_list, batch_cond_images, batch_gen_image_info, batch_gen_video_info,
                    batch_gen_audio_info, batch_cond_videos,
                )
            ]

        #   -- 2.2 Prompt, image, cot text, system prompt
        else:
            raise NotImplementedError("Only `message_list` input is supported currently.")

        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [
            torch.Generator("cpu" if args.generator_device == "cpu" else self.device).manual_seed(seed)
            for seed in seeds
        ]

        # 3. apply chat template
        cfg_factor = {
            "gen_text": 1,
            "gen_image": 2 if self.generation_config.diff_guidance_scale > 1 else 1,
            "gen_video": 2 if self.generation_config.diff_guidance_scale > 1 else 1,
            "gen_audio": 2 if self.generation_config.diff_guidance_scale > 1 else 1,
            "gen_av": 2 if self.generation_config.diff_guidance_scale > 1 else 1,
        }
        # Get conversation template according to the model_name
        if conv_template is None:
            conv_template = get_conversation_template(
                default(args.conv_template, args.model_name.split('.')[-1])
            )
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
            bot_task=bot_task,  # if mode is "gen_<media_type>", bot_task will not be used
            image_base_size=self.image_processor.vae_reso_group.base_size,
            cond_image_section_type=self.image_processor.cond_image_section_type,
            sequence_template=self.generation_config.sequence_template,
            cfg_factor=cfg_factor[mode],
            conv_template=conv_template,
            und_token_type=args.und_token_type if args.use_mot or args.gen_template == "multi_stream_dit" else [],
            gen_token_type=args.gen_token_type if args.use_mot or args.gen_template == "multi_stream_dit" else [],
            audio_token_type=args.audio_token_type if args.use_mot or args.gen_template == "multi_stream_dit" else [],
            uncond_length=args.uncond_length,
            use_text_mask=True,
            answer=False if args.gen_template in ["dit_it", "multi_stream_dit"] else "auto",
        )

        output, sections = out['output'], out['sections']
        # Convert to long, align with training in av_loader.
        output.text_mask = output.text_mask.to(torch.long)
        # 4. Encode conditional images
        cond_vae_images, cond_timesteps, cond_vit_images = self._encode_cond_image(
            batch_cond_images, cfg_factor[mode]
        )
        cond_vit_image_kwargs = self._prepare_vit_image_kwargs(batch_cond_images, cfg_factor[mode])
        (
            cond_vae_images,
            cond_vit_images,
            cond_vit_image_kwargs,
            cond_vit_videos,
            cond_vit_video_kwargs,
        ) = self._merge_cond_videos(
            batch_cond_videos, cond_vae_images, cond_vit_images, cond_vit_image_kwargs, cfg_factor[mode],
        )
        if isinstance(cond_vae_images, list) and len(cond_vae_images) > 0 and isinstance(cond_vae_images[0], list):
            _ts_dtype = cond_timesteps.dtype if isinstance(cond_timesteps, torch.Tensor) else (
                cond_timesteps[0].dtype if cond_timesteps else torch.float32)
            cond_timesteps = [torch.zeros(len(m), device=device, dtype=_ts_dtype) for m in cond_vae_images]
        # Channel-concat will be performed before cfg expand, so cfg_factor is not applied here.
        channel_cond_vae_images, _, _ = self._encode_cond_image(batch_channel_cond_images, vae_encode_type="mode")

        # 5. Build position embeddings
        if args.gen_template == "dit_it":
            if mode == "gen_image":
                rope_media_info = [
                    [(None, (1, gen_info.tk_h, gen_info.tk_w), {"type": "gen_image"})]
                    for gen_info in batch_gen_image_info
                ] * cfg_factor[mode]
            elif mode == "gen_video":
                rope_media_info = [
                    [(None, (gen_info.tk_d, gen_info.tk_h, gen_info.tk_w), {"type": "gen_video"})]
                    for gen_info in batch_gen_video_info
                ] * cfg_factor[mode]
            elif mode == "gen_av":
                rope_media_info = [
                    [
                        (None, (gen_video_info.tk_d, gen_video_info.tk_h, gen_video_info.tk_w), {"type": "gen_video", "with_audio": True}),
                        (None, (gen_audio_info.tk_d, 1, 1), {"type": "gen_audio", "with_video": True}),
                    ]
                    for gen_video_info, gen_audio_info in zip(batch_gen_video_info, batch_gen_audio_info)
                ] * cfg_factor[mode]
            else:
                rope_media_info = None

        elif args.gen_template == "multi_stream_dit":
            num_channel_cond_images = len(batch_channel_cond_images[0]) if batch_channel_cond_images is not None else 0
            rope_media_info = self.build_batch_rope_media_info(output, sections, mode, num_channel_cond_images)
        else:
            raise ValueError(f"Unsupported generation template {args.gen_template} for RoPE media info preparation.")

        # 8. Build model input kwargs
        visual_mask = dict(
            gen_image=output.gen_image_mask,
            gen_video=output.gen_video_mask,
            gen_av=output.gen_video_mask,
            gen_audio=output.gen_audio_mask,
        )
        model_input_kwargs = dict(
            # text encoder
            input_ids=output.tokens.to(device),
            text_mask=to_device(output.text_mask, device),
            system_prompt=get_system_prompt(use_system_prompt, None),
            # --
            mode=mode,
            rope_media_info=rope_media_info,
            # image/video
            visual_mask=to_device(visual_mask.get(mode), device),
            timesteps_index=to_device(output.gen_timestep_scatter_index, device),
            # audio
            audio_mask=to_device(output.gen_audio_mask, device),
            cond_vae_images=to_device(cond_vae_images, device),
            cond_vae_mask=to_device(
                self._merge_cond_vae_masks(output.vae_image_mask, output.vae_video_mask), device),
            cond_timesteps=to_device(cond_timesteps, device),
            cond_timesteps_index=to_device(output.cond_timestep_scatter_index, device),
            cond_vit_images=to_device(cond_vit_images, device),
            cond_vit_image_mask=to_device(output.vit_image_mask, device),
            cond_vit_image_kwargs=to_device(cond_vit_image_kwargs, device),
            cond_vit_videos=to_device(cond_vit_videos, device),
            cond_vit_video_mask=to_device(output.vit_video_mask, device),
            cond_vit_video_slices=output.vit_video_slices,
            cond_vit_video_context_slices=output.vit_video_context_slices,
            cond_vit_video_kwargs=to_device(cond_vit_video_kwargs, device),
            # channel conditions
            channel_cond_vae_images=to_device(channel_cond_vae_images, device),
            # for inner usage
            tokenizer_output=output,
            batch_gen_image_info=batch_gen_image_info,
            batch_gen_video_info=batch_gen_video_info,
            batch_gen_audio_info=batch_gen_audio_info,
            generator=generator,
            batch_cond_images=batch_cond_images,
            batch_channel_cond_images=batch_channel_cond_images,
            video_fps=video_fps,
        )
        if self.args.use_mot or self.args.gen_template == "multi_stream_dit":
            model_input_kwargs["und_token_indices"] = to_device(output.und_token_indices, device)
            model_input_kwargs["gen_token_indices"] = to_device(output.gen_token_indices, device)
            model_input_kwargs["audio_token_indices"] = to_device(output.audio_token_indices, device)

        return model_input_kwargs

    def _prepare_attention_mask_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            generation_config: GenerationConfig,
            model_kwargs: dict[str, Any],
    ) -> Optional[torch.Tensor]:
        # Leo use full attention. Here we just need build an attention_mask_in_length instead of calling
        # prepare_full_attn_slices.
        bsz, seq_len = inputs_tensor.shape
        media_seqlen = 0

        # Vision token length
        if model_kwargs["mode"] == "gen_image":
            batch_gen_image_info = model_kwargs["batch_gen_image_info"]
            tk_height = batch_gen_image_info[0].tk_h
            tk_width = batch_gen_image_info[0].tk_w
            media_seqlen += tk_height * tk_width
        elif model_kwargs["mode"] in ["gen_video", "gen_av"]:
            batch_gen_video_info = model_kwargs["batch_gen_video_info"]
            tk_duration = batch_gen_video_info[0].tk_d
            tk_height = batch_gen_video_info[0].tk_h
            tk_width = batch_gen_video_info[0].tk_w
            media_seqlen += tk_duration * tk_height * tk_width

        # Audio token length
        if model_kwargs["mode"] in ["gen_audio", "gen_av"]:
            batch_gen_audio_info = model_kwargs["batch_gen_audio_info"]
            audio_seqlen = batch_gen_audio_info[0].tk_d
            media_seqlen += audio_seqlen

        if media_seqlen == 0:
            raise ValueError(f"Unsupported generation mode {model_kwargs['mode']} for attention mask preparation.")

        cond_text_mask = model_kwargs["cond_text_mask"]

        # attention mask for flash/flash3 (binary valid-token mask)
        if self.args.gen_template == "dit_it":
            if self.args.add_timestep_token:
                cond_text_mask_2 = F.pad(cond_text_mask, (1, 0), value=True)
            else:
                cond_text_mask_2 = cond_text_mask
            attention_mask = F.pad(cond_text_mask_2, (media_seqlen, 0), value=True)

        elif (
            self.args.gen_template == "multi_stream_dit"
            and self.args.attn_impl in ["flash", "flash3"]
        ):
            # MoT scatter layout: [und..., gen..., pad...].
            attention_mask = (inputs_tensor != self.tokenizer.pad_token_id).to(torch.long)

        # attention mask for flash_packed/flash3_packed (segment-length mask)
        elif self.args.gen_template == "multi_stream_dit":
            assert self.args.attn_impl in ["flash_packed", "flash3_packed", "sageattn"], \
                f"Unsupported attention implementation {self.args.attn_impl} for multi-stream dit."
            # Keep the length mask for pipeline bookkeeping. Dense SageAttention
            # accepts the tensor here but does not consume it in the kernel.
            if self.args.use_input_ids:
                sample_seqlen = (inputs_tensor != self.tokenizer.pad_token_id).sum(dim=1)
            else:
                text_seqlen = cond_text_mask.sum(dim=1)
                sample_seqlen = text_seqlen + media_seqlen  # tensor + int
            attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long, device=inputs_tensor.device)
            attention_mask[:, 0] = sample_seqlen

        else:
            raise ValueError(f"Unsupported generation template {self.args.gen_template} for attention mask preparation.")

        return attention_mask

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            tokenizer_output=None, batch_cond_images=None, batch_gen_image_info=None, generator=None,
            batch_gen_video_info=None, batch_channel_cond_images=None, batch_gen_audio_info=None,
            **kwargs
    ):
        model_inputs = {
            "attention_mask": attention_mask,
            "rope_media_info": kwargs["rope_media_info"],
            # for gen image/video
            "latents": kwargs["latents"],
            "timesteps": kwargs["timesteps"],
            "cond_vae_latents": kwargs.get("cond_vae_images", None),
            # for gen audio
            "audio_latents": kwargs.get("audio_latents", None),
            "audio_timesteps": kwargs.get("audio_timesteps", None),
            # for cond text
            "cond_text_states": kwargs["cond_text_states"],
            "cond_text_mask": kwargs["cond_text_mask"],
        }
        if self.args.use_input_ids:
            model_inputs["input_ids"] = input_ids
            model_inputs["visual_mask"] = kwargs["visual_mask"]
            model_inputs["text_mask"] = kwargs["text_mask"]
            model_inputs["timesteps_index"] = kwargs["timesteps_index"]
            model_inputs["audio_mask"] = kwargs["audio_mask"]
            model_inputs["cond_vae_mask"] = kwargs.get("cond_vae_mask")
            model_inputs["cond_vae_timesteps"] = kwargs.get("cond_timesteps")
            model_inputs["cond_text_scatter_mask"] = kwargs.get("cond_text_scatter_mask")
        if self.args.use_mot or self.args.gen_template == "multi_stream_dit":
            model_inputs["und_token_indices"] = kwargs["und_token_indices"]
            model_inputs["gen_token_indices"] = kwargs["gen_token_indices"]
            model_inputs["audio_token_indices"] = kwargs["audio_token_indices"]
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
        return model_kwargs

    @staticmethod
    def parse_modality_kwargs(mode, kwargs):
        modality_kwargs = {"image_size": None}

        # Visual modalities.
        if mode == "gen_image":
            batch_gen_image_info: list[ImageInfo] = kwargs.get("batch_gen_image_info")
            if batch_gen_image_info is None:
                raise ValueError("`batch_gen_image_info` should be provided when `mode` is `gen_image`.")
            modality_kwargs.update(dict(
                batch_size=len(batch_gen_image_info),
                image_size=(batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width),
                video_duration=1,
            ))
        elif mode in ["gen_video", "gen_av"]:
            batch_gen_video_info: list[VideoInfo] = kwargs.get("batch_gen_video_info")
            assert batch_gen_video_info is not None, \
                "`batch_gen_video_info` should be provided when `mode` is `gen_video`."
            modality_kwargs.update(dict(
                batch_size=len(batch_gen_video_info),
                image_size=(batch_gen_video_info[0].video_height, batch_gen_video_info[0].video_width),
                video_duration=batch_gen_video_info[0].video_duration,
            ))

        # Audio modality.
        if mode in ["gen_audio", "gen_av"]:
            batch_gen_audio_info: list[AudioInfo] = kwargs.get("batch_gen_audio_info")
            assert batch_gen_audio_info is not None, \
                "`batch_gen_audio_info` should be provided when `mode` is `gen_audio` or `gen_av`."
            if "batch_size" not in modality_kwargs:
                modality_kwargs["batch_size"] = len(batch_gen_audio_info)
            modality_kwargs["audio_duration"] = batch_gen_audio_info[0].audio_duration
            # Pass the exact audio token count so the freshly-created audio latents match the sequence's audio
            # token count. For normal generation this equals calc_token_duration(audio_duration); for validation
            # loss (where token_duration is pinned to the loaded latent), it keeps generation consistent too.
            modality_kwargs["audio_token_length"] = batch_gen_audio_info[0].token_duration

        return modality_kwargs

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
            verbose: int = 0,
            output_type: str = "pil",
            skip_special_tokens: bool = False,
            video_fps: Optional[int] = None,
            return_latents: bool = False,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        gen_config = default(generation_config, self.generation_config)
        mode = kwargs.get("mode", "gen_text")

        # Log info
        if verbose >= 1:
            output = kwargs["tokenizer_output"]
            context = self._tokenizer.decode(output.tokens[0], skip_special_tokens=False)
            # Replace <img><img>...<img> with [<img>]{number}. The same as <pad>
            img_token = re.escape(self._tokenizer.get_img_token())
            video_token = re.escape(self._tokenizer.video_token)
            audio_token = re.escape(self._tokenizer.audio_token)
            pad_token = self._tokenizer.pad_token
            if isinstance(pad_token, int):
                pad_token = self._tokenizer.decode(pad_token)
            pad_token = re.escape(pad_token)
            context = re.sub(f"({img_token})+", lambda m: f"[{img_token}]{{{len(re.escape(m.group(0))) // len(img_token)}}}", context)
            context = re.sub(f"({video_token})+", lambda m: f"[{video_token}]{{{len(re.escape(m.group(0))) // len(video_token)}}}", context)
            context = re.sub(f"({audio_token})+", lambda m: f"[{audio_token}]{{{len(re.escape(m.group(0))) // len(audio_token)}}}", context)
            context = re.sub(f"({pad_token})+", lambda m: f"[{pad_token}]{{{len(re.escape(m.group(0))) // len(pad_token)}}}", context)
            info_list = [
                ("system_prompt", kwargs["system_prompt"]),
                ("token shape", output.tokens.shape),
                ("context[0]", context),
            ]
            if generator is not None:
                info_list.extend([
                    ("seed", [g.initial_seed() for g in generator]),
                ])
            if kwargs["batch_gen_image_info"] is not None and kwargs["batch_gen_image_info"][0] is not None:
                info_list.extend([
                    ("image_size", [
                        f"{info.image_height}x{info.image_width}" for info in kwargs["batch_gen_image_info"]
                    ]),
                ])
            if kwargs["batch_gen_video_info"] is not None and kwargs["batch_gen_video_info"][0] is not None:
                info_list.extend([
                    ("video_size", [
                        f"{info.video_height}x{info.video_width}" for info in kwargs["batch_gen_video_info"]
                    ]),
                    ("video_length", [
                        info.video_duration for info in kwargs["batch_gen_video_info"]
                    ]),
                    ("video_fps", video_fps),
                ])
            if kwargs["batch_channel_cond_images"] is not None and kwargs["batch_channel_cond_images"][0] is not None:
                info_list.extend([
                    ("channel_cond_image_size", [
                        [f"{image.i.h}x{image.i.w}" for image in channel_cond_images]
                        for channel_cond_images in kwargs["batch_channel_cond_images"]
                    ]),
                ])
            if kwargs["batch_gen_audio_info"] is not None and kwargs["batch_gen_audio_info"][0] is not None:
                info_list.extend([
                    ("audio_duration", [
                        info.audio_duration_by_seconds for info in kwargs["batch_gen_audio_info"]
                    ]),
                ])
            info_list.extend([
                ("infer_steps", gen_config.diff_infer_steps),
                ("guidance_scale", gen_config.diff_guidance_scale),
                ("guidance_scale_audio", gen_config.diff_guidance_scale_audio),
                ("flow_shift", gen_config.flow_shift),
                ("flow_shift_video", gen_config.flow_shift_video),
                ("flow_shift_audio", gen_config.flow_shift_audio),
            ])
            max_key_len = max(len(k) for k, _ in info_list)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            info_str = "=" * 50 + \
                       f"\n[Rank {rank}] Model input info:\n" + \
                       "\n".join([f"    {k.rjust(max_key_len)}: {v}" for k, v in info_list]) + \
                       "\n--------------------------------------------------"
            print(info_str, flush=True)
            start_time = time.time()

        modality_kwargs = self.parse_modality_kwargs(mode, kwargs)

        self.build_diffusion_pipeline()
        results = self.diffusion_pipeline(
            **modality_kwargs,
            num_inference_steps=gen_config.diff_infer_steps,
            guidance_scale=gen_config.diff_guidance_scale,
            guidance_scale_audio=gen_config.diff_guidance_scale_audio,
            generator=generator,
            output_type=output_type,
            model_kwargs=kwargs,
            return_latents=return_latents,
        )
        outputs = MultimodalGenerationOutputs(
            images=results.visuals if mode == "gen_image" else None,
            videos=results.visuals if mode in ["gen_video", "gen_av"] else None,
            audios=results.audios,
        )

        if verbose >= 1:
            end_time = time.time()
            print(f"Generation completed in {end_time - start_time:.2f} seconds.", flush=True)

        if return_latents:
            latent_outputs = MultimodalGenerationOutputs(
                images=results.visual_latents if mode == "gen_image" else None,
                videos=results.visual_latents if mode in ["gen_video", "gen_av"] else None,
                audios=results.audio_latents,
            )
            return outputs, latent_outputs
        else:
            return outputs

    def generate_image(
            self,
            prompt=None,
            image=None,
            message_list=None,
            seed=None,
            image_size=None,
            num_frames=None,
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            image_output_type: str = "pil",
            **kwargs,
    ):
        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)

        if message_list is not None:
            # We will update message_list, so deepcopy it first to avoid changing it outside.
            message_list = deepcopy(message_list)

        # Generate image
        # if mode is "gen_image", add_assistant_prefix is False in apply_general_template, bot_task will not be used,
        # so here we don't need to pass bot_task, let it be default value "auto"
        # TODO: Support batch variable resolution for leo; 
        # Now Unpack per-sample image_size list (e.g. [(H, W)] -> (H, W)) for leo which does not support batch variable resolution yet.
        if isinstance(image_size, list) and len(image_size) > 0 and isinstance(image_size[0], (list, tuple)):
            if len(set(image_size)) > 1:
                print(f"Leo does not support variable resolution in a batch, using the first sample's size.")
            image_size = image_size[0]

        model_inputs = self.prepare_model_inputs(
            prompt=prompt, image=image, message_list=message_list, use_system_prompt=use_system_prompt,
            seed=seed, media_size=image_size, num_frames=num_frames, mode="gen_image",
        )
        batch_cond_images_cache = model_inputs['batch_cond_images']
        outputs = self.generate(**model_inputs, output_type=image_output_type, **kwargs)

        outputs.images = self.image_processor.postprocess_outputs(outputs.images, batch_cond_images_cache)
        return outputs

    def generate_video(
            self,
            prompt=None,
            message_list=None,
            seed=None,
            video_size: Optional[str | tuple] = None,
            num_frames: Optional[int] = None,
            video_fps: int = 24,
            ref_mode: Literal["sequence", "channel"] = "sequence",
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            audio_duration: Optional[float] = None,     # only for pure audio generation
            cond_after_gen: bool = False,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)

        if message_list is not None:
            # We will update message_list, so deepcopy it first to avoid changing it outside.
            message_list = deepcopy(message_list)

        mode = dict(
            video="gen_video",
            av="gen_av",
            audio="gen_audio",
        )[bot_task]

        # Generate video
        model_inputs = self.prepare_model_inputs(
            prompt=prompt, message_list=message_list, use_system_prompt=use_system_prompt,
            seed=seed, media_size=video_size, num_frames=num_frames, video_fps=video_fps, ref_mode=ref_mode, mode=mode,
            audio_duration=audio_duration,
            cond_video_vae_path=kwargs.pop("cond_video_vae_path", None),  # r2v: offline source-video VAE latent
            cond_after_gen=cond_after_gen,
        )
        outputs = self.generate(**model_inputs, **kwargs)

        return outputs

    def generate_validation_loss(
            self,
            prompt=None,
            message_list=None,
            seed=None,
            video_fps: int = 24,
            ref_mode: Literal["sequence", "channel"] = "channel",
            use_system_prompt=None,
            system_prompt=None,
            bot_task=None,
            audio_duration: Optional[float] = None,     # only for pure audio generation
            vae_info=None,
            audio_vae_info=None,
            video_latent_path=None,
            audio_latent_path=None,
            timestep_points=None,
            args=None,
            **kwargs,
    ) -> MultimodalGenerationOutputs:
        use_system_prompt = default(use_system_prompt, self.generation_config.use_system_prompt)
        bot_task = default(bot_task, self.generation_config.bot_task)

        if message_list is not None:
            # We will update message_list, so deepcopy it first to avoid changing it outside.
            message_list = deepcopy(message_list)

        mode = dict(
            video="gen_video",
            av="gen_av",
            audio="gen_audio",
        )[bot_task]

        video_latent = to_device(load_video_latent(video_latent_path, vae_info.latent_dim, args.video_cos_base), self.device)
        audio_latent = to_device(load_audio_latent(audio_latent_path, audio_vae_info.latent_dim, args.audio_cos_base), self.device)

        video_latent = normalize_vae_latents(self.model_dict["vae"], video_latent)

        latent_shape = tuple(video_latent.shape)  # [1, C, tk_d, tk_h, tk_w]
        num_frames, height, width = latent_shape_to_media_size(
            latent_shape, vae_info.down_d_factor, vae_info.down_h_factor, vae_info.down_w_factor,
        )

        # Pin the video token grid to the loaded latent (tk_d, tk_h, tk_w). Otherwise build_gen_video_info would
        # re-snap num_frames/size to VAE DAR buckets, shifting the token count and breaking the scatter alignment.
        video_token_grid = tuple(int(x) for x in video_latent.shape[2:5]) if mode in ["gen_video", "gen_av"] else None

        # AudioProjection maps (B, C, L) -> (B, L, hidden), so the audio media-sequence length equals the loaded audio
        # latent length. Pin the sequence audio token count to it to avoid an off-by-one scatter mismatch.
        audio_token_length = audio_latent.size(-1) if mode in ["gen_audio", "gen_av"] else None

        # Debug-gen mode runs the normal CFG denoising (self.generate) to sanity-check model_inputs, so it keeps
        # the configured guidance. The default validation-loss mode only needs a conditional forward to compare the
        # model prediction against the ground-truth velocity, so CFG (which duplicates the batch into cond/uncond and
        # would make input_ids batch=2 vs the single loaded latent batch=1) is disabled by pinning guidance to 1.
        debug_gen = args is not None and getattr(args, "validation_loss_debug_gen", False)

        _prev_diff_guidance_scale = self.generation_config.diff_guidance_scale
        _prev_diff_guidance_scale_audio = getattr(self.generation_config, "diff_guidance_scale_audio", None)
        if not debug_gen:
            self.generation_config.diff_guidance_scale = 1.0
            if _prev_diff_guidance_scale_audio is not None:
                self.generation_config.diff_guidance_scale_audio = 1.0
        try:
            model_inputs = self.prepare_model_inputs(
                prompt=prompt, message_list=message_list, use_system_prompt=use_system_prompt,
                seed=seed, media_size=None, num_frames=num_frames, video_fps=video_fps, ref_mode=ref_mode,
                mode=mode, audio_duration=audio_duration,
                audio_token_length=audio_token_length,
                video_token_grid=video_token_grid,
            )
        finally:
            self.generation_config.diff_guidance_scale = _prev_diff_guidance_scale
            if _prev_diff_guidance_scale_audio is not None:
                self.generation_config.diff_guidance_scale_audio = _prev_diff_guidance_scale_audio

        if debug_gen:
            # Verify the model_inputs by running the full (normal) denoising loop and returning the decoded
            # media. Does NOT use the loaded latents / add-noise path — it generates from fresh noise.
            gen_kwargs = {"output_type": {"visual": "np", "audio": "np"}, "verbose": 2}
            return self.generate(**model_inputs, **gen_kwargs)

        # `build_av_denoiser` registers process-global denoisers that can only be initialized once,
        # so build them a single time and cache on the model across samples/timesteps.
        if getattr(self, "_validation_denoisers", None) is None:
            self._validation_denoisers = build_av_denoiser(args)
        video_denoiser, audio_denoiser = self._validation_denoisers
        self.build_diffusion_pipeline()

        if timestep_points is None:
            n_ts = getattr(args, "validation_loss_timesteps", 8)
            timestep_points = np.linspace(0.0, 1.0, n_ts + 2).tolist()[1:-1]

        generator = model_inputs["generator"][0]
        video_losses, audio_losses = [], []
        for t_val in timestep_points:
            t_tensor = torch.tensor([float(t_val)], device=self.device)
            video_vae_out = add_noise_and_extend_channel(
                video_latent, None, True, video_denoiser, generator,
                "sample", return_dict=True, latent_channel_extend_type="t2v", timesteps=t_tensor)
            audio_vae_out = _add_noise(
                audio_latent, audio_denoiser, generator, sample_type="sample",
                return_dict=True, timesteps=t_tensor)

            # A fresh shallow copy per timestep: the pipeline pops `input_ids` and injects `cond_text_states`/
            # `attention_mask` into the mapping, so reusing the same dict across timesteps would raise on the 2nd call.
            results = self.diffusion_pipeline.generate_validation_loss(
                video_vae_out=video_vae_out,
                audio_vae_out=audio_vae_out,
                model_kwargs=dict(model_inputs),
                video_denoiser=video_denoiser,
                audio_denoiser=audio_denoiser,
            )

            vloss = results.get("video_loss")
            aloss = results.get("audio_loss")
            video_losses.append(None if vloss is None else float(vloss))
            audio_losses.append(None if aloss is None else float(aloss))

        return {
            "timesteps": [float(t) for t in timestep_points],
            "video_losses": video_losses,
            "audio_losses": audio_losses,
        }
