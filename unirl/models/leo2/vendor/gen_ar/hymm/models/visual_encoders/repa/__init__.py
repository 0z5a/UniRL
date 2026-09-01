from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import bradford
from transformers.utils import ModelOutput
from transformers import AutoProcessor

from hymm.constants import VISION_ENCODER_META_INFO
from hymm.utils.torch_utils import PRECISION_TO_TYPE

def load_image_processor(
    processor_type,
    processor_path=None,
    logger=None
):
    if processor_path is None:
        processor_path = VISION_ENCODER_META_INFO[processor_type]['path']
    if logger is not None:
        logger.info(f"Loading image processor ({processor_type}) from: {processor_path}")

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    return processor, processor_path

def _load_qwen_visual_only(repa_encoder_path, logger=None):

    import json
    from pathlib import Path
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    full_config = AutoConfig.from_pretrained(repa_encoder_path, trust_remote_code=True)
    vision_config = full_config.vision_config

    common_kwargs = dict(torch_dtype=torch.bfloat16)

    visual = Qwen3_5VisionModel._from_config(
        vision_config,
        attn_implementation="flash_attention_2",
        **common_kwargs,
    )
    visual.eval()

    # Stream only `model.visual.*` weights from the safetensors shards.
    ckpt_dir = Path(repa_encoder_path)
    visual_prefix = "model.visual."
    vision_sd: dict = {}

    index_file = ckpt_dir / "model.safetensors.index.json"
    if index_file.is_file():
        import safetensors.torch as st_io
        with index_file.open() as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shards_needed: dict = {}
        for full_key, shard_name in weight_map.items():
            if full_key.startswith(visual_prefix):
                shards_needed.setdefault(shard_name, []).append(full_key)
        if not shards_needed:
            raise RuntimeError(
                f"Can't find {visual_prefix}* from {index_file}."
            )
        for shard_name, keys in shards_needed.items():
            shard_path = ckpt_dir / shard_name
            shard_sd = st_io.load_file(str(shard_path), device="cpu")
            for full_key in keys:
                vision_sd[full_key[len(visual_prefix):]] = shard_sd[full_key]
            del shard_sd
    else:
        # Single-file safetensors fallback.
        import safetensors.torch as st_io
        single_files = sorted(ckpt_dir.glob("*.safetensors"))
        if not single_files:
            raise RuntimeError(f"Can't find *.safetensors from {ckpt_dir}.")
        for st_path in single_files:
            shard_sd = st_io.load_file(str(st_path), device="cpu")
            for full_key, val in shard_sd.items():
                if full_key.startswith(visual_prefix):
                    vision_sd[full_key[len(visual_prefix):]] = val
            del shard_sd

    if not vision_sd:
        raise RuntimeError(
            f"Can't find `{visual_prefix}*` weights from {repa_encoder_path}."
        )

    # Match the dtype the encoder will run in.
    vision_sd = {k: v.to(torch.bfloat16) for k, v in vision_sd.items()}

    missing, unexpected = visual.load_state_dict(vision_sd, strict=False)
    if logger is not None:
        if missing:
            logger.warning(
                f"Visual encoder missing {len(missing)} key(s); first few: {missing[:5]}"
            )
        if unexpected:
            logger.warning(
                f"Visual encoder got {len(unexpected)} unexpected key(s); "
                f"first few: {unexpected[:5]}"
            )
        logger.info(
            f"Loaded {len(vision_sd)} visual tensor(s) from {repa_encoder_path}"
        )

    return visual

def load_repa_encoder(
    repa_encoder_type,
    repa_encoder_precision=None,
    repa_encoder_path=None,
    logger=None,
    device=None,
):

    if repa_encoder_type is not None:
        repa_encoder_path = VISION_ENCODER_META_INFO[repa_encoder_type]['path']

    if "DINOv2" in repa_encoder_type:
        from transformers import Dinov2Model
        repa_encoder = Dinov2Model.from_pretrained(repa_encoder_path)

    elif "DINOv3" in repa_encoder_type:
        from transformers import DINOv3ViTModel
        repa_encoder = DINOv3ViTModel.from_pretrained(repa_encoder_path, local_files_only=True,)

    elif "qwen-3.5-9b" in repa_encoder_type:
        repa_encoder = _load_qwen_visual_only(repa_encoder_path, logger=logger)
    else:
        raise ValueError(f"Unsupported repa encoder type: {repa_encoder_type}")

    # from_pretrained will ensure that the model is in eval mode.
    # if repa_encoder_precision is not None:
    #     repa_encoder = repa_encoder.to(dtype=PRECISION_TO_TYPE[repa_encoder_precision])
    if "DINOv3" in repa_encoder_type:
        # repa_encoder = repa_encoder.to(dtype=torch.bfloat16)    # TODO(leo2): fix precision problem
        repa_encoder = repa_encoder.to(dtype=torch.float32)    # TODO(leo2): fix precision problem
    repa_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Repa encoder to dtype: {repa_encoder.dtype}")

    if device is not None:
        repa_encoder = repa_encoder.to(device)

    return repa_encoder, repa_encoder_path


@dataclass
class RepaEncoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class RepaEncoder(nn.Module):
    def __init__(
        self,
        repa_encoder_type: str,
        repa_encoder_precision: Optional[str] = None,
        repa_encoder_path: Optional[str] = None,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.repa_encoder_type = repa_encoder_type
        self.precision = repa_encoder_precision
        self.model_path = repa_encoder_path
        self.logger = logger

        self.model, _ = load_repa_encoder(
            repa_encoder_type=self.repa_encoder_type,
            repa_encoder_precision=self.precision,
            repa_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.processor = None
        self.processor_path = None
        if "dino" not in self.repa_encoder_type:
            self.processor, self.processor_path = load_image_processor(
                processor_type=repa_encoder_type,
                processor_path=repa_encoder_path,
                logger=self.logger,
            )

        if "DINO" in self.repa_encoder_type:
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _as_pil_images(self, imgs):
        if isinstance(imgs, list) and all(isinstance(item, dict) and "image" in item for item in imgs):
            extracted = [item["image"] for item in imgs]
            normalized = []
            for img in extracted:
                if isinstance(img, Image.Image):
                    normalized.append(img)
                elif isinstance(img, np.ndarray):
                    normalized.extend(self._as_pil_images(img))
                else:
                    raise ValueError(f"Unsupported nested image entry: {type(img)}")
            return normalized

        if isinstance(imgs, list):
            pil_images = []
            for item in imgs:
                if isinstance(item, Image.Image):
                    pil_images.append(item)
                elif isinstance(item, dict) and isinstance(item.get("image"), Image.Image):
                    pil_images.append(item["image"])
                elif isinstance(item, np.ndarray):
                    pil_images.extend(self._as_pil_images(item))
                else:
                    raise ValueError(f"Unsupported image entry for Qwen encoder: {type(item)}")
            return pil_images

        if isinstance(imgs, np.ndarray):
            arr = imgs
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)
            if arr.ndim != 4:
                raise ValueError(f"Expected image tensor with 3 or 4 dims, got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            return [Image.fromarray(arr_i) for arr_i in arr]

        raise ValueError(f"Unsupported image format for Qwen encoder: {type(imgs)}")

    def _process_features(self, last_hidden_state, b, t):
        is_video = (t is not None and t > 1) 
        if last_hidden_state.dim() == 2:
            last_hidden_state = last_hidden_state.unsqueeze(1)
        b_total, n_tokens, d_dim = last_hidden_state.shape

        if is_video:
            last_hidden_state = last_hidden_state.view(b, t, n_tokens, d_dim)
            feat_first = last_hidden_state[:, 0:1, :, :]
            padding = torch.zeros(
                b, 1, n_tokens, 3 * d_dim, 
                device=last_hidden_state.device, 
                dtype=last_hidden_state.dtype
            )
            feat_first_out = torch.cat([feat_first, padding], dim=-1)

            # Time unshuffle
            if t > 1:
                feat_rest = last_hidden_state[:, 1:, :, :]  # [B, T-1, N, D]
                t_rest = t - 1
                if t_rest % 4 == 0:
                    # Reshape: [B, T_rest/4, 4, N, D]
                    feat_rest = feat_rest.view(b, t_rest // 4, 4, n_tokens, d_dim)
                    # Permute: [B, T_rest/4, N, 4, D] (Time folded into D)
                    feat_rest = feat_rest.permute(0, 1, 3, 2, 4)
                    # Flatten: [B, T_rest/4, N, 4D]
                    feat_rest_out = feat_rest.reshape(b, t_rest // 4, n_tokens, 4 * d_dim)
                    
                    # Concat Time
                    last_hidden_state = torch.cat([feat_first_out, feat_rest_out], dim=1)
                else:
                    if self.logger:
                        self.logger.warning(f"Video frames {t} is not 4n+1, skipping time unshuffle logic.")
                    last_hidden_state = feat_first_out
            else:
                last_hidden_state = feat_first_out
            
            # Merge Time and Space: [B, T_new * N, 4D]
            last_hidden_state = last_hidden_state.flatten(1, 2)

        else:
            # Image Logic, used when image input during video training
            # Pad to match 4*D dimension
            padding = torch.zeros(
                b_total, n_tokens, 3 * d_dim, 
                device=last_hidden_state.device, 
                dtype=last_hidden_state.dtype
            )
            last_hidden_state = torch.cat([last_hidden_state, padding], dim=-1)
        
        return last_hidden_state

    def encode_images(self, images, texts=None, data_type="image"):
        # if use video data, encode video frames, otherwise encode images
        pooler_output = None
        hidden_states = None

        is_video = False
        b, c, t, h_in, w_in = None, None, None, None, None

        images_is_list = isinstance(images, list)

        if images_is_list:
            assert "qwen-3.5-9b" in self.repa_encoder_type, (
                "List input is only supported for Qwen-VL REPA encoders; "
                f"got repa_encoder_type={self.repa_encoder_type}."
            )
            for it in images:
                assert isinstance(it, torch.Tensor) and it.ndim == 4, (
                    f"List items must be 4D tensors, got {type(it).__name__} with shape "
                    f"{getattr(it, 'shape', None)}"
                )
        elif isinstance(images, torch.Tensor) and images.ndim == 5:
            is_video = True
            b, c, t, h_in, w_in = images.shape
            images = images.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h_in, w_in)
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            b = images.shape[0]  # Batch size for images

        assert self.repa_encoder_type == "qwen-3.5-9b", "only supports Qwen-3.5 VL REPA encoders"
        if "DINO" in self.repa_encoder_type:
            # DINOv2 Preprocessing
            h, w = images.shape[-2], images.shape[-1]
            # DINOv2 likes 14x multiples
            if "DINOv2" in self.repa_encoder_type:
                new_h = round(h / 14) * 14
                new_w = round(w / 14) * 14

                if new_h != h or new_w != w:
                    images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

            # use registered buffer
            images = images.to(device=self.model.device, dtype=self.model.dtype)
            if self.imagenet_mean.device != images.device or self.imagenet_mean.dtype != images.dtype:
                self.imagenet_mean = self.imagenet_mean.to(device=images.device, dtype=images.dtype)
                self.imagenet_std = self.imagenet_std.to(device=images.device, dtype=images.dtype)
            images = (images - self.imagenet_mean) / self.imagenet_std

            outputs = self.model(pixel_values=images)

            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, "vision_model_output"):
                last_hidden_state = outputs.vision_model_output.last_hidden_state
            else:
                last_hidden_state = outputs[0]

            if last_hidden_state is not None:
                # Remove CLS token
                if "DINOv2" in self.repa_encoder_type:
                    last_hidden_state = last_hidden_state[:, 1:, :]
                elif "DINOv3" in self.repa_encoder_type:
                    last_hidden_state = last_hidden_state[:, 5:, :]
        elif "qwen-3.5-9b" in self.repa_encoder_type:

            # Normalize input to a list of 4D tensors so we can run Qwen processor + visual
            # encoder ONCE for the whole batch, even when individual samples differ in (H, W).
            tensors_list = list(images) if images_is_list else [images]

            # Qwen3.5 likes 32x multiples; resize per-input then convert to PIL list.
            patch_size = 32
            pil_images = []
            for tensor_in in tensors_list:
                _, _, h, w = tensor_in.shape
                new_h = round(h / patch_size) * patch_size
                new_w = round(w / patch_size) * patch_size
                if new_h != h or new_w != w:
                    tensor_in = F.interpolate(
                        tensor_in, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                tensor_pil = tensor_in.permute(0, 2, 3, 1).contiguous()
                imgs_np = tensor_pil.cpu().numpy()
                new_pil = self._as_pil_images(imgs_np)
                pil_images.extend(new_pil)

            if len(pil_images) == 0:
                raise ValueError("Batch size is 0")

            # Text Prompt Handling — preserve previous behavior; aligned to total #pil_images.
            text_input_qwen = [""] * len(pil_images)
            if texts is not None:
                text_list = list(texts) if isinstance(texts, (list, tuple)) else [texts]
                if is_video and b is not None and len(text_list) == b:
                    expanded_texts = []
                    for txt in text_list:
                        expanded_texts.extend([str(txt) if txt is not None else ""] * t)
                    text_input_qwen = expanded_texts
                elif len(text_list) == len(pil_images):
                    text_input_qwen = [str(_t) if _t is not None else "" for _t in text_list]
                elif len(text_list) == 1:
                    text_input_qwen = [
                                          str(text_list[0]) if text_list[0] is not None else ""
                                      ] * len(pil_images)

            # Single batched processor call (handles variable image sizes via image_grid_thw).
            processor_inputs = self.processor(
                text=text_input_qwen,
                images=pil_images,
                videos=None,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            processor_inputs = processor_inputs.to(device=self.model.device)
            pixel_values = processor_inputs.pixel_values.to(dtype=self.model.dtype)
            image_grid_thw = getattr(processor_inputs, "image_grid_thw", None)

            # Single batched visual-encoder call.
            model_outputs = self.model(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw,
            )

            # Extract Hidden States
            if isinstance(model_outputs, torch.Tensor):
                visual_hidden = model_outputs
            elif isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 0:
                visual_hidden = model_outputs[0]
            else:
                visual_hidden = getattr(model_outputs, "last_hidden_state", None)
                if visual_hidden is None and hasattr(model_outputs, "image_embeds"):
                    visual_hidden = getattr(model_outputs, "image_embeds")

            if visual_hidden is None:
                raise ValueError("Qwen-VL vision encoder did not return hidden states.")

            total_tokens, d_dim = visual_hidden.shape

            # Per-image token counts. Use image_grid_thw when available — robust to
            # variable image sizes. Merge factor (= spatial_merge_size**2 for Qwen3.5)
            # is inferred empirically from total_patches / total_tokens to avoid hard-coding.
            if image_grid_thw is not None and image_grid_thw.numel() > 0:
                thw_list = image_grid_thw.detach().cpu().tolist()
                patches_per_image = [
                    int(t_i) * int(h_i) * int(w_i) for (t_i, h_i, w_i) in thw_list
                ]
                total_patches = sum(patches_per_image)
                if total_tokens == 0 or total_patches % total_tokens != 0:
                    raise ValueError(
                        f"Cannot infer merge factor: total_patches={total_patches}, "
                        f"total_tokens={total_tokens}."
                    )
                merge_factor = total_patches // total_tokens
                if merge_factor < 1 or any(p % merge_factor != 0 for p in patches_per_image):
                    raise ValueError(
                        f"Merge factor {merge_factor} does not evenly divide all "
                        f"per-image patch counts {patches_per_image}."
                    )
                tokens_per_image = [p // merge_factor for p in patches_per_image]
            else:
                if total_tokens % len(pil_images) != 0:
                    raise ValueError(
                        f"Total tokens {total_tokens} not divisible by batch size "
                        f"{len(pil_images)}."
                    )
                n = total_tokens // len(pil_images)
                tokens_per_image = [n] * len(pil_images)

            if sum(tokens_per_image) != total_tokens:
                raise ValueError(
                    f"Token split mismatch: sum {sum(tokens_per_image)} vs "
                    f"total {total_tokens}."
                )

            # Split visual_hidden into per-image features [N_i, D] -> [1, N_i, D]
            last_hidden_state = [
                chunk.unsqueeze(0)
                for chunk in visual_hidden.split(tokens_per_image, dim=0)
            ]

        else:
            raise ValueError(f"Unsupported repa encoder type: {self.repa_encoder_type}")

        if (
                last_hidden_state is not None
                and data_type == 'image_video'
                and not isinstance(last_hidden_state, list)
        ):
            current_t = t if is_video else None
            current_b = b if is_video else last_hidden_state.shape[0]

            last_hidden_state = self._process_features(
                last_hidden_state,
                b=current_b,
                t=current_t,
            )

        return RepaEncoderModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states
        )

    def encode_latents(self, latents, vae):
        """
        Encode latents by first converting to images, then encoding.
        This is the main function that replaces siglip_vision_encode.
        
        Args:
            latents: Input latent tensors
            vae: VAE model for decoding latents to images
            
        Returns:
            Encoded image features
        """
        # Convert latents to images
        images = self.encode_latents_to_images(latents, vae)
        
        # Encode images
        outputs = self.encode_images(images)
        
        return outputs.last_hidden_state

    def forward(self, images):
        """
        Forward pass for direct image encoding.
        
        Args:
            images: Input images
            
        Returns:
            RepaEncoderModelOutput with encoded features
        """
        return self.encode_images(images) 

    @staticmethod
    def encode_latents_to_images(latents, vae):
        """
        Convert latents to images using VAE decoder.
        
        Args:
            latents: Input latents tensor
            vae: VAE model for decoding
        Returns:
            images: Decoded images as numpy array
        """
        # Handle both 4D and 5D latents (for video, take first frame)
        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
        first_image = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0].cpu()
        first_image = first_image[:, :, 0, :, :]
        first_image = (first_image / 2 + 0.5).clamp(0, 1)
        first_image = (first_image * 255.0).clamp(0, 255.0)
        first_image = first_image.to(torch.uint8).numpy()
        first_image = first_image.transpose(0, 2, 3, 1)

        assert isinstance(first_image, np.ndarray)
        assert first_image.ndim == 4 and first_image.shape[3] == 3
        assert first_image.dtype == np.uint8

        return first_image
