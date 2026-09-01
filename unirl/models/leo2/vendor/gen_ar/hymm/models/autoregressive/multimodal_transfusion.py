import re
from argparse import Namespace
from contextlib import nullcontext
from typing import Optional, List, Union, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from diffusers.models import ModelMixin
try:
    from megatron import get_args, print_rank_0, mpu # type: ignore
    from deepspeed.runtime.utils import see_memory_usage
except (ModuleNotFoundError, ImportError):
    print("not found ptm")

from hymm.utils.import_utils import is_package_version
from .config import Config
from .mlp_layers import load_projector
from .resampler import load_qformer
from .transfusion import Transfusion
from .hunyuan import HunYuanPreTrainedModel
from .configuration_hunyuan import HunYuanConfig
from ..visual_encoders import load_vision_model


def strip_leading_tag(src_dict, tag, required=True):
    res_dict = {}
    leading_str = f"{tag}_"
    for key, value in src_dict.items():
        if key.startswith(leading_str):
            key = key[len(leading_str):]
        elif required:
            raise ValueError(f"Key {key} does not start with {leading_str}")
        res_dict[key] = value
    return res_dict

def set_no_grad(module):
    if isinstance(module, nn.Parameter):
        module.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = False


# ModelMixin 和 HunyuanPreTrainedModel 的 __init__ 类参数不一样, 无法同时继承.
# 因此抽一个 Base 类分别自定义 __init__ 以兼容二者.
class MultiModalTransfusionBase(nn.Module):
    def __post_init__(
            self,
            args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device
        self.args = args
        self.pl_config = config

        self.use_ptm = args.get('use_ptm', False)
        if hasattr(args, "launcher") and args.launcher == "pure_torch":
            from hymm.parallelism.parallel_states import get_parallel_state
            from torch import distributed as dist
            self.enable_pp = get_parallel_state().pp_enabled
            config.launcher = args.launcher
            if self.enable_pp:
                # dirty hack
                self.pre_process = dist.get_rank() == get_parallel_state().pp_mesh.mesh[0]
                self.post_process = dist.get_rank() == get_parallel_state().pp_mesh.mesh[-1]
            else:
                self.pre_process = True
                self.post_process = True
        elif self.use_ptm:
            factory_kwargs['device'] = torch.device("cuda", args.local_rank)
            self.pre_process = mpu.is_pipeline_first_stage()
            self.post_process = mpu.is_pipeline_last_stage()
        else:
            self.pre_process = True
            self.post_process = True

        # Fix backward compatibility for `self.vision_model` instead of `self.vision_model_so`
        self.fix_bc = args.get("fix_bc", [])

        if self.pre_process:
            vision_model_config = config.vision_model
            if hasattr(vision_model_config, "vision_model_type"):

                vision_model = load_vision_model(
                    vision_model_type=vision_model_config["vision_model_type"],
                    vision_model_precision=vision_model_config.get('vision_model_precision'),
                    device=device,
                    require_grad=not vision_model_config["vision_model_freeze"],
                    eval_mode=vision_model_config["vision_model_freeze"],
                    no_load_pretrained=args.get("no_load_pretrained_vision_model", False),
                )
                if "vision_model" in self.fix_bc:
                    # For backward compatibility, we use `self.vision_model` instead of `self.vision_model_so`
                    self.vision_model = vision_model
                else:
                    self.vision_model_so = vision_model
                self.vision_model_context = torch.no_grad \
                    if vision_model_config["vision_model_freeze"] else nullcontext

                vision_aligner_config = config.vision_aligner
                vision_aligner = load_projector(
                    projector_type=vision_aligner_config["vision_aligner_type"],
                    projector_precision=vision_aligner_config.get('vision_aligner_precision'),
                    projector_params=strip_leading_tag(vision_aligner_config.get("vision_aligner_params", {}), "vision_aligner"),
                    device=device,
                    require_grad=not vision_aligner_config["vision_aligner_freeze"],
                    eval_mode=vision_aligner_config["vision_aligner_freeze"],
                )
                if "vision_aligner" in self.fix_bc:
                    # For backward compatibility, we use `self.vision_aligner` instead of `self.vision_aligner_so`
                    self.vision_aligner = vision_aligner
                else:
                    self.vision_aligner_so = vision_aligner
                self.vision_aligner_context = torch.no_grad \
                    if vision_aligner_config["vision_aligner_freeze"] else nullcontext

            face_aligner_config = config.face_aligner
            if hasattr(face_aligner_config, "face_aligner_type"):
                self.face_aligner = load_qformer(
                    qformer_type=face_aligner_config["face_aligner_type"],
                    qformer_precision=face_aligner_config.get('face_aligner_precision'),
                    qformer_params=strip_leading_tag(face_aligner_config.get("face_aligner_params", {}), "face_aligner"),
                    device=device,
                    require_grad=not face_aligner_config["face_aligner_freeze"],
                    eval_mode=face_aligner_config["face_aligner_freeze"],
                )
                self.face_aligner_context = torch.no_grad \
                    if face_aligner_config["face_aligner_freeze"] else nullcontext

        language_config = config
        self.language_model = Transfusion(args, language_config, hf_config=self.config, **factory_kwargs)

        if args.get('convert_tp_friendly_qkv', False):
            from hymm.trainers.transfusion_parallel import convert_tp_friendly_qkv
            convert_tp_friendly_qkv(self.language_model)

        if args.get('freeze_language_model', False):
            set_no_grad(self.language_model)

    def _forward_vision_encoder(self, images, **vision_encoder_kwargs):
        with self.vision_model_context():
            if "vision_model" in self.fix_bc:
                image_embeds = self.vision_model(images, **vision_encoder_kwargs).last_hidden_state
            else:
                image_embeds = self.vision_model_so(images, **vision_encoder_kwargs).last_hidden_state
        with self.vision_aligner_context():
            if "vision_aligner" in self.fix_bc:
                image_embeds = self.vision_aligner(image_embeds)
            else:
                image_embeds = self.vision_aligner_so(image_embeds)
        return image_embeds

    def forward_vision_encoder(self, images, **vision_encoder_kwargs):
        if isinstance(images, torch.Tensor):
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                if images.ndim == 3: # batch_size x seq_len x dim
                    image_embeds = self._forward_vision_encoder(images, **vision_encoder_kwargs)
                elif images.ndim == 4: # batch_size x n x seq_len x dim
                    bsz, n, seq_len, dim = images.shape
                    images = images.view(bsz * n, seq_len, dim)
                    for k, v in vision_encoder_kwargs.items():
                        vision_encoder_kwargs[k] = v.reshape(bsz * n, *v.shape[2:])
                    image_embeds = self._forward_vision_encoder(images, **vision_encoder_kwargs)
                    new_dim = image_embeds.shape[-1]
                    image_embeds = image_embeds.reshape(bsz, n * seq_len, new_dim)
                else:
                    raise ValueError(
                        f"und_images should be 3D or 4D tensor for siglip2-so400m-patch16-naflex, but got {images.ndim}D tensor"
                    )
            else:
                if images.ndim == 4:
                    image_embeds = self._forward_vision_encoder(images, **vision_encoder_kwargs)
                elif images.ndim == 5:
                    bsz, n, c, h, w = images.shape
                    images = images.view(bsz * n, c, h, w)
                    image_embeds = self._forward_vision_encoder(images, **vision_encoder_kwargs)
                    _, seq_len, dim = image_embeds.shape
                    image_embeds = image_embeds.reshape(bsz, n * seq_len, dim)
                else:
                    raise ValueError(
                        f"und_images should be 4D or 5D tensor, but got {images.ndim}D tensor"
                    )
        elif isinstance(images, list):
            if self.args.vision_model_type == "siglip2-so400m-patch16-naflex":
                image_embeds = []
                for batch_idx, image in enumerate(images):
                    cur_kwargs = {k: v[batch_idx] for k, v in vision_encoder_kwargs.items()}
                    image_embed = self._forward_vision_encoder(image, **cur_kwargs)
                    n, seq_len, dim = image_embed.shape
                    image_embed = image_embed.reshape(n * seq_len, dim)
                    image_embeds.append(image_embed)
            else:
                image_embeds = []
                for image in images:
                    image_embed = self._forward_vision_encoder(image, **vision_encoder_kwargs)
                    n, seq_len, dim = image_embed.shape
                    image_embed = image_embed.reshape(n * seq_len, dim)
                    image_embeds.append(image_embed)
        else:
            raise ValueError(
                f"und_images should be Tensor or List, but got {type(images)}"
            )
        return image_embeds

    def forward(
            self,
            idx: torch.Tensor,  # batch_size x seq_len-1
            x_t: Optional[torch.Tensor] = None,  # batch_size x c x h x w
            t: Optional[torch.Tensor] = None,  # batch_size
            target: Optional[torch.Tensor] = None,  # batch_size x seq_len-1, for calulating dicrete loss
            diffusion_loss_fn: Optional[nn.Module] = None,  # for calculating diffusion loss, can be None when sampling
            src_x: Optional[torch.Tensor] = None,  # batch_size x c x h x w, only used for instruction tuning
            src_t: Optional[torch.Tensor] = None,  # batch_size, only used for instruction tuning
            src_image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1, only used for instruction tuning
            input_pos: Optional[torch.Tensor] = None,  # batch_size x seq_len-1, used for KVCache
            iw_ih_scatter_index: Optional[torch.Tensor] = None,  # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2k  (w, h)
            timestep_scatter_index: Optional[torch.Tensor] = None,  # batch_size x k
            timestep_r_scatter_index: Optional[torch.Tensor] = None,  # bsz x k, or bsz x (k_i)
            guidance_scatter_index: Optional[torch.Tensor] = None,  # batch_size x k
            timestep_scatter_src: Optional[torch.Tensor] = None,  # batch_size x k
            text_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            image_loss_weight: float = 1.0,
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x 1 x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            data_type: Optional[str] = "image",
            und_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
            und_image_masks: Optional[torch.Tensor] = None,
            src_face_embedding: Optional[torch.Tensor] = None, # batch_size x c x 1 x 1, only used for face id generation;
            rope_image_info: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
            vision_encoder_kwargs: Optional[dict] = None,
            sample_offsets: Optional[List[torch.Tensor]] = None,  # only used for training with sequence pack enabled
            n_samples: Optional[torch.Tensor] = None,   # If batch of x_t is larger than batch of idx, this is used to indicate the number of samples in each batch.
            return_loss: Optional[bool] = None,  # used for rl training
            past_key_values=None,
            guidance: Optional[torch.Tensor] = None,
            r: Optional[torch.Tensor] = None,
            cache_dic=None,
            **kwargs,
    ):
        """
        data_type: str
            Selected from ["t2i"/"image", "lm"/"text", "mmu"]
        """
        if self.pre_process:
            if und_images is not None:
                vision_encoder_kwargs = vision_encoder_kwargs or {}
                und_image_embeds = self.forward_vision_encoder(und_images, **vision_encoder_kwargs)
            else:
                und_image_embeds = None

            if src_face_embedding is not None:
                # src_face_embedding_tokens will reuse src_image_mask
                with self.face_aligner_context():
                    face_image_embeds = self.face_aligner(src_face_embedding)
                    if und_image_masks is None:
                        assert src_image_mask is not None
                        und_image_masks = src_image_mask
                        und_image_embeds = face_image_embeds
                    else:
                        if und_image_embeds is not None:
                            und_image_embeds = torch.cat([und_image_embeds, face_image_embeds], dim=1)
                        else:
                            und_image_embeds = face_image_embeds
        else:
            und_image_embeds = None

        out = self.language_model(
            idx=idx,
            x_t=x_t,
            t=t,
            target=target,
            diffusion_loss_fn=diffusion_loss_fn,
            src_x=src_x,
            src_t=src_t,
            src_image_mask=src_image_mask,
            input_pos=input_pos,
            iw_ih_scatter_index=iw_ih_scatter_index,
            iw_ih_scatter_src=iw_ih_scatter_src,
            timestep_scatter_index=timestep_scatter_index,
            timestep_r_scatter_index=timestep_r_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            timestep_scatter_src=timestep_scatter_src,
            text_mask=text_mask,
            image_mask=image_mask,
            image_loss_weight=image_loss_weight,
            attention_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            data_type=data_type,
            und_image_embeds=und_image_embeds,
            und_image_masks=und_image_masks,
            rope_image_info=rope_image_info,
            sample_offsets=sample_offsets,
            n_samples=n_samples,
            return_loss=return_loss,
            past_key_values=past_key_values,
            guidance=guidance,
            r=r,
            cache_dic=cache_dic
        )

        return out

    def infer_forward(
            self,
            idx: torch.Tensor,  # batch_size x seq_len-1
            x_t: torch.Tensor, # batch_size x c x h x w
            t: torch.Tensor,  # batch_size
            src_x: Optional[torch.Tensor] = None,  # batch_size x c x h x w, only used for instruction tuning
            src_t: Optional[torch.Tensor] = None,  # batch_size, only used for instruction tuning
            src_image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1, only used for instruction tuning
            first_step=True,
            input_pos: Optional[torch.Tensor] = None,
            iw_ih_scatter_index: Optional[torch.Tensor] = None,   # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2k  (w, h)
            timestep_scatter_index: Optional[torch.Tensor] = None,  # batch_size x 1
            timestep_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 1
            image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x 1 x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            und_images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
            und_image_masks: Optional[torch.Tensor] = None,
            src_face_embedding: Optional[torch.Tensor] = None, # batch_size x c x 1 x 1, only used for face id generation;
            rope_image_info: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
            vision_encoder_kwargs: Optional[dict] = None,
            n_samples: Optional[torch.Tensor] = None,
            past_key_values=None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,
            cache_dic=None,
            ext_model_forward: Optional[Any] = None,
            **kwargs,
    ):
        if und_images is not None:
            vision_encoder_kwargs = vision_encoder_kwargs or {}
            und_image_embeds = self.forward_vision_encoder(und_images, **vision_encoder_kwargs)
        else:
            und_image_embeds = None
        
        if src_face_embedding is not None:
            # src_face_embedding_tokens will reuse src_image_mask 
            with self.face_aligner_context():
                face_image_embeds = self.face_aligner(src_face_embedding)
                if und_image_masks is None:
                    assert src_image_mask is not None
                    und_image_masks = src_image_mask
                    und_image_embeds = face_image_embeds
                else:
                    if und_image_embeds is not None:
                        und_image_embeds = torch.cat([und_image_embeds, face_image_embeds], dim=1)
                    else:
                        und_image_embeds = face_image_embeds

        return self.language_model.infer_forward(
            idx=idx,
            x_t=x_t,
            t=t,
            src_x=src_x,
            src_t=src_t,
            src_image_mask=src_image_mask,
            first_step=first_step,
            input_pos=input_pos,
            iw_ih_scatter_index=iw_ih_scatter_index,   # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src=iw_ih_scatter_src,  # batch_size x 2k  (w, h)
            timestep_scatter_index=timestep_scatter_index,  # batch_size x 1
            timestep_scatter_src=timestep_scatter_src,  # batch_size x 1
            image_mask=image_mask,  # batch_size x seq_len-1
            attention_mask=attention_mask,  # batch_size x 1 x seq_len-1 x seq_len-1
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            und_image_embeds=und_image_embeds,
            und_image_masks=und_image_masks,
            rope_image_info=rope_image_info,
            n_samples=n_samples,
            past_key_values=past_key_values,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            cache_dic=cache_dic,
            ext_model_forward=ext_model_forward,
        )

    def set_kv_cache(self, *args, **kwargs):
        self.language_model.set_kv_cache(*args, **kwargs)

    def update_mask_cache(self, *args, **kwargs):
        self.language_model.update_mask_cache(*args, **kwargs)

    def clear_kv_cache(self):
        self.language_model.clear_kv_cache()

    def set_rope_cache(self, seq_len, rope_image_info, device):
        self.language_model.set_rope_cache(seq_len, rope_image_info, device)

    def clear_rope_cache(self):
        self.language_model.clear_rope_cache()

    def enable_deterministic(self) -> None:
        pass

    def disable_deterministic(self) -> None:
        pass

    def set_input_tensor(self, input_tensor) -> None:
        self.language_model.set_input_tensor(input_tensor)


class MultiModalTransfusion(MultiModalTransfusionBase, ModelMixin):
    def __init__(
            self,
            args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        # hf_config is None when using ptm
        self.config = config
        self.__post_init__(args, config, dtype, device)


class MultiModalTransfusionHF(MultiModalTransfusionBase, HunYuanPreTrainedModel):
    def __init__(
            self,
            hf_config: Optional[HunYuanConfig],
            args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            pp_size: int = 1,
            start_device_id: int = None,
    ):
        super().__init__(hf_config)
        self.config = hf_config
        # When using huggingface transformers, we use meta device to accelerate model building
        # and saving memory.
        self.__post_init__(args, config, dtype)
        # Define device_map for modules
        self.device_map = self.get_device_map(pp_size=pp_size, start_device_id=start_device_id)
        self._tp_plan = {}

    @property
    def layer_device_map(self):
        if self.device_map is None:
            raise ValueError("device_map is not set, please call load_hf_model() first.")
        layer_device_map = {}
        for key, value in self.device_map.items():
            if key.startswith("language_model.transformer.h."):
                layer_idx = int(key.split(".")[-1])
                value = f"cuda:{value}"
                layer_device_map[layer_idx] = value
        return layer_device_map

    def get_device_map(self, pp_size, start_device_id):
        # pp_size = 1 will return huggingface default device map
        from transformers.modeling_utils import _get_device_map

        if pp_size == 1:
            device_map = "auto"
        else:
            print(f"=========== {torch.distributed.get_rank()} {pp_size=}, {start_device_id=} =========== ")
            device_map = {
                "vision_model_so": start_device_id,
                "vision_aligner_so": start_device_id,
                "language_model.timestep_emb": start_device_id,
                "language_model.patch_embed": start_device_id,
                "language_model.time_embed": start_device_id,
                "language_model.final_layer": start_device_id,
                "language_model.time_embed_2": start_device_id,
                "language_model.lm_head": start_device_id,
                "language_model.transformer.wte": start_device_id,
                "language_model.transformer.ln_f": start_device_id + pp_size - 1,
            }
            num_layers_per_stage = np.array_split(np.arange(self.config.num_hidden_layers), pp_size)
            for stage_id, layer_ids in enumerate(num_layers_per_stage):
                for layer_id in layer_ids:
                    device_map[f"language_model.transformer.h.{int(layer_id)}"] = start_device_id + stage_id

        keep_in_fp32_regex = re.compile(r"\.gate\.wg")
        get_device_map_kwargs = dict(keep_in_fp32_regex=keep_in_fp32_regex)
        if is_package_version("transformers", ">=", "4.56"):
            get_device_map_kwargs["dtype"] = torch.bfloat16
        else:
            get_device_map_kwargs["torch_dtype"] = torch.bfloat16
        return _get_device_map(self, device_map=device_map, max_memory=None, hf_quantizer=None,
                               **get_device_map_kwargs)
