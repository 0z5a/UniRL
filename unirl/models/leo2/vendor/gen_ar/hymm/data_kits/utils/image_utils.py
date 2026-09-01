import math
import os
import random
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Any, List, Tuple
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from index_kits import ArrowIndexV2, MultiIndexV2, ResolutionGroup
from loguru import logger
from transformers import BaseImageProcessor
from transformers.image_utils import load_image
from processors.image_kits import read_local_image, read_binary_image, read_url_image
from .data_utils import DataMixin
from .index_utils import IndexColumn
from ...constants import VAE_META_INFO, VISION_ENCODER_META_INFO
from ...models.visual_encoders import load_vit_processor
from ...samplers.logits_processor import get_logits_processors
from ...utils.image_base import ImageTensor, ImageInfo, CondImage
from ...utils.states import DataClassMixin

InputImage = Union[Image.Image, str]

ResampleType = dict(
    bilinear=Image.Resampling.BILINEAR,
    bicubic=Image.Resampling.BICUBIC,
    lanczos=Image.Resampling.LANCZOS,
)


def compute_target_size_with_area_cap(
    w_ori: int,
    h_ori: int,
    max_area: Optional[int] = None,
    max_size: Optional[int] = None,
    align: int = 16,
    random_crop: Optional[bool | str] = None,
) -> Tuple[int, int]:
    """
    Compute target (width, height) with area and optional max_side constraint.
    - If current area <= max_area and (max_side is None or longest side <= max_side): no resize.
    - Else: scale down proportionally so area <= max_area and longest side <= max_side (if set).
    Returns (width, height) in PIL order; sizes are aligned to `align` for VAE.
    """
    if w_ori <= 0 or h_ori <= 0:
        return (align, align), random_crop
    current_area = w_ori * h_ori
    long_side_ori = max(w_ori, h_ori)
    need_resize = (max_area is not None and current_area > max_area) \
                or (max_size is not None and long_side_ori > max_size)
    if not need_resize:
        # No resize: align to align for VAE compatibility
        w_new = (w_ori // align) * align
        h_new = (h_ori // align) * align
        w_new = max(align, w_new)
        h_new = max(align, h_new)
        return (w_new, h_new), 'center_and_no_resize'
    # Scale down: (1) area <= max_area, (2) max(w,h) <= max_size if set
    scale = 1.0
    if max_area is not None and current_area > max_area:
        scale_by_area = (max_area / current_area) ** 0.5
        scale = min(scale, scale_by_area)
    if max_size is not None and long_side_ori > max_size:
        scale_by_side = max_size / long_side_ori
        scale = min(scale, scale_by_side)
    w_new = int(w_ori * scale) // align * align
    h_new = int(h_ori * scale) // align * align
    w_new = max(align, w_new)
    h_new = max(align, h_new)
    return (w_new, h_new), random_crop


@dataclass
class ResolutionGroupConfig(DataClassMixin):
    base_size: int = None
    step: Optional[int] = None
    align: int = 16
    mode: Optional[str] = None
    preset: Optional[str] = None
    aspect_ratios: Optional[list[str]] = None
    num_buckets: Optional[int] = None
    added_method: Optional[str] = None

    @classmethod
    def from_args(cls, args, **kwargs):
        config = dict(
            base_size=kwargs.get("base_size", args.reso_base_size),
            step=kwargs.get("step", args.reso_step),
            align=kwargs.get("align", args.reso_align),
            mode=kwargs.get("mode", args.reso_mode),
            preset=kwargs.get("preset", args.reso_preset),
            aspect_ratios=kwargs.get("aspect_ratios", args.reso_aspect_ratios),
            num_buckets=kwargs.get("num_buckets", args.reso_num_buckets),
            added_method=kwargs.get("added_method", args.reso_added_method),
        )
        return cls(**config)


@dataclass
class VAEInfo:
    encoder_type: str
    down_h_factor: int = -1
    down_w_factor: int = -1
    patch_size: int = 1
    h_factor: int = -1
    w_factor: int = -1
    image_type: str = None

    def __post_init__(self):
        self.h_factor = self.down_h_factor * self.patch_size
        self.w_factor = self.down_w_factor * self.patch_size
        if self.image_type is None:
            self.image_type = "vae"


@dataclass
class ViTInfo:
    encoder_type: str
    h_factor: int = -1
    w_factor: int = -1
    max_token_length: int = 0   # pad to max_token_length
    processor: Callable = field(default_factory=BaseImageProcessor)
    image_type: str = None

    def __post_init__(self):
        if self.image_type is None:
            self.image_type = self.encoder_type.split("-")[0]


class ImageMixin(DataMixin):
    task_kwargs: dict
    index_kwargs: dict
    modality: list[str]
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"
    reso_strategy: str
    vae_info: VAEInfo
    vit_info: ViTInfo

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prepare image transformations
        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

    def setup_image(self, args):
        # image_data_format 是增量功能，按 dataset 从 task_kwargs 取（与 audio_data_format 一致，无全局回退）
        self.image_data_format = self.task_kwargs.get("image_data_format", "pixels")
        ImageInfo.args = dict(
            add_timestep_token=args.add_timestep_token,
            add_timestep_r_token=args.add_timestep_r_token,
            add_guidance_token=args.add_guidance_token,
            use_front_boi_token=args.use_front_boi_token,
            add_image_shape_token=args.add_image_shape_token,
            add_tw_th_token=args.add_tw_th_token,
            cond_image_type=args.cond_image_type,
            gen_image_template=args.gen_image_template,
            ignore_boi_token=args.ignore_boi_token,
        )
        self.pixel_resample_type = getattr(args, "pixel_resample_type", "bicubic")

        # -- conditional image --
        self.cond_image_type = args.cond_image_type
        self.cond_token_attn_type = args.cond_token_attn_type
        self.timestep_vae_full_attn = args.timestep_vae_full_attn
        # cond-token-attn-type and cond-image-type should be matched
        # vae: causal, full
        # vit: causal, full
        # vae_vit: causal, full, joint_full, causal_full, full_causal
        if self.cond_image_type in ["vae", "vit"]:
            assert self.cond_token_attn_type in ["causal", "full"], \
                f"When cond-image-type is 'vae', cond-token-attn-type should be 'causal' or 'full', " \
                f"but got {self.cond_token_attn_type}."
        if args.cond_token_attn_type in ["joint_full", "full_causal"]:
            assert args.cond_image_type == "vae_vit", \
                f"When cond-token-attn-type is {args.cond_token_attn_type}, " \
                f"cond-image-type should be set to 'vae_vit', but got {args.cond_image_type}."
        self.cond_image_section_type = dict(
            vae="cond_vae_image",
            vit="cond_vit_image",
            vae_vit="cond_joint_image",
        )[self.cond_image_type]

        # -- vae image --
        if "vae_image" in self.modality:
            self.require_configs(args, ["vae_type", "patch_size", "vae_image_token_length"], "vae_image modality")

            self.vae_image_token_length = self.task_kwargs.get("vae_image_token_length", args.vae_image_token_length)

            self.reso_strategy = args.reso_strategy
            if self.reso_strategy == "reso_group":
                self.reso_base_size = args.reso_base_size
                self.reso_group_config = ResolutionGroupConfig.from_args(
                    args, **self.index_kwargs.get("reso_bucket_kwargs", {})
                )
                video_driven = (
                    "vae_video" in self.modality
                    and self.index_kwargs.get("dar_bucket_kwargs") is not None
                )
                if (
                    hasattr(self, "index_manager")
                    and not video_driven
                    and (self.index_kwargs.get("online_bucketing") or not self.index_kwargs.get("multireso", False))
                ):
                    self.index_manager.set_resolution_buckets(**self.reso_group_config.to_dict())
                self.vae_reso_group = ResolutionGroup(**self.reso_group_config.to_dict())
            elif self.reso_strategy == "anyres":
                self.vae_reso_group = None
                self.reso_base_size = self.index_kwargs.get("reso_base_size", args.reso_base_size)
                self.reso_max_area = self.index_kwargs.get("reso_max_area", args.reso_max_area)
                self.reso_max_size = self.index_kwargs.get("reso_max_size", args.reso_max_size)
                self.reso_align = self.index_kwargs.get("reso_align", args.reso_align)
            else:
                raise ValueError(f"Unknown resolution strategy: {self.reso_strategy}")
            # vae info
            vae_meta_info = VAE_META_INFO[args.vae_type]
            downsample_factor = vae_meta_info["downsample_factor"]
            self.vae_info = VAEInfo(
                encoder_type=args.vae_type,
                down_h_factor=downsample_factor[0], down_w_factor=downsample_factor[1],
                patch_size=args.patch_size,
            )
            # 兼容vae读取latent
            if self.image_data_format == "latents":
                # latent_dim 仅离线 latent 管道需要，且只有部分 VAE 条目提供该字段
                latent_dim = vae_meta_info.get("latent_dim")
                assert latent_dim is not None, (
                    f"VAE_META_INFO[{args.vae_type!r}] 缺少 latent_dim，"
                    "离线 image latent 模式需要该字段来校验 latent shape。"
                )
                self.vae_latent_dim = latent_dim
                image_cos_base = getattr(args, "image_cos_base", None)
                assert image_cos_base, (
                    "image_data_format='latents' 需要配置 DATA_ARGS.image-cos-base"
                    "（离线 image latent .npy 的 COS 根路径），否则相对路径会静默拼到当前目录。"
                )
                self.image_cos_base = Path(image_cos_base)

        # -- vit image --
        if "vit_image" in self.modality:
            self.require_configs(args, ["vit_type", "vit_image_token_length"], "vit_image modality")

            self.vit_image_token_length = self.task_kwargs.get("vit_image_token_length", args.vit_image_token_length)
            if args.vit_type.startswith("qwen3vl-vit"):
                # for qwen3vl, we set max_pixels self.vit_image_token_length*32*32
                # set min_pixels to None, so the min token length default is set to 256
                self.min_vit_image_token_length = self.task_kwargs.get("min_vit_image_token_length", args.min_vit_image_token_length)
                if self.min_vit_image_token_length is None: self.min_vit_image_token_length = 256
                vit_max_pixels = self.vit_image_token_length*32*32
                vit_min_pixels = self.min_vit_image_token_length*32*32
                processor = load_vit_processor(args.vit_type, min_pixels=vit_min_pixels, max_pixels=vit_max_pixels)
            else:
                # vit info
                processor = load_vit_processor(args.vit_type, max_num_patches=self.vit_image_token_length)
            
            self.vit_info = ViTInfo(
                encoder_type=args.vit_type,
                h_factor=processor.patch_size,
                w_factor=processor.patch_size,
                max_token_length=self.vit_image_token_length,
                processor=processor,
            )

        # -- unconditions --
        self.uncond_p = self.task_kwargs.get('uncond_p', 0.0)

    def read_local_image_latent(self, latent_key: str) -> np.ndarray:
        """读取离线提取的 image latent（.npy）。相对路径拼 image_cos_base，绝对路径走 parse_cos_path。"""
        if not latent_key.endswith(".npy"):
            latent_key += ".npy"
        if latent_key.startswith("/"):
            latent_path = self.parse_cos_path(latent_key)
        else:
            latent_path = self.image_cos_base / latent_key.lstrip("/")
        latent = np.load(latent_path)
        assert latent.ndim == 5 and latent.shape[:3] == (1, self.vae_latent_dim, 1), \
            f"Image latent should be (1, {self.vae_latent_dim}, 1, H, W), but got {latent.shape}."
        return latent

    def get_image_latents(
            self,
            src: int,
            real_index: int = None,
            **latent_col,
    ):
        """
        参考video_utils.get_video_latents()
        """
        latent_path = None
        try:
            latent_path = self.index_manager.get_attribute(src, **latent_col)
            latent = self.read_local_image_latent(latent_path)
            latent = latent.squeeze(0)       # [1, C, 1, H, W] -> [C, 1, H, W]
            # 保留 T=1 维度！shape = [C, 1, H, W] = [C, T, H, W]
            # 与 scale_and_extend_frame / add_noise_to_latents 输入格式一致
            # 异常值检查（与 video_utils.py get_video_latents() 对齐）
            if np.isinf(latent).any():
                raise ValueError(f"Inf detected in image latents: {src=}, {latent_path=}")
            max_val = np.abs(latent).max()
            if max_val > 10:
                raise ValueError(f"Abnormal max value {max_val} in image latents: {src=}, {latent_path=}")
            image_flag = "normal"
        except Exception as e:
            index = src if isinstance(src, int) else real_index
            logger.error(f"({latent_col=}, {index=}, {latent_path=}) {type(e)}: {e}.")
            image_flag = "error"
            latent = np.zeros(
                (
                    self.vae_latent_dim,
                    1,      # T=1
                    self.reso_base_size // self.vae_info.h_factor,
                    self.reso_base_size // self.vae_info.w_factor,
                ),
                dtype=np.float16,
            )
        return latent, image_flag

    def read_local_image(self, file_path, is_encrypted='auto', convert_mode=None, apply_exif=False):
        if is_encrypted == 'auto' and str(file_path).endswith('.enc') or is_encrypted is True:
            file_path = self.decrypt_file(file_path)
        return read_local_image(file_path, convert_mode=convert_mode, apply_exif=apply_exif)

    def get_image_from_arrow(self, index, column="image", is_list=False, return_first=None, shadow=None,
                             apply_exif=False):
        if is_list:
            if return_first:
                image = read_binary_image(
                    self.index_manager.get_attribute(index, column=column, shadow=shadow)[0],
                    apply_exif=apply_exif,
                )
            else:
                image = [
                    read_binary_image(binary, apply_exif=apply_exif)
                    for binary in self.index_manager.get_attribute(index, column=column, shadow=shadow)
                ]
        else:
            image = self.index_manager.get_image(index, column=column, shadow=shadow, apply_exif=apply_exif)
        return image

    def get_image_from_url_cos(self, index, column="url_cos", max_retry=3, shadow=None, apply_exif=False):
        if isinstance(index, str):
            url_cos = index
        else:
            url_cos = self.index_manager.get_attribute(index, column=column, shadow=shadow)

        if 'prc-videoframe' in url_cos:
            url_cos = url_cos.split('?')[0]

        # Replace external link with internal link. Remove the unexpected single quotes.
        if "'prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn'" in url_cos:
            url_cos = url_cos.replace(
                "'prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn'",
                "prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn",
            )
        url_cos = url_cos.replace("cos.ap-guangzhou.myqcloud.com", "cos-internal.ap-guangzhou.tencentcos.cn")
        url_cos = unquote(url_cos)

        image = None
        for _ in range(max_retry):
            try:
                image = read_url_image(url_cos, apply_exif=apply_exif)
                break
            except Exception as e:
                logger.warning(f"[PID={os.getpid()}] {e}. Failed to get image ({index=}) from {url_cos}, retrying...")
        if image is None:
            # If failed here, the exception will be caught by get_raw_image and return a gray image
            raise ValueError(f"Failed to get image from {url_cos}")
        return image

    def get_dummy_image(self):
        image = Image.new("RGB", (self.reso_base_size, self.reso_base_size), (128, 128, 128))
        image_flag = "gray"
        return image, image_flag

    def get_raw_image(
            self,
            src: int | np.integer | dict | str,
            real_index: int | None = None,
            convert_mode: str = "RGB",
            apply_exif: bool = False,
            url_cos_col: IndexColumn | str | None = None,
            **image_col,
    ):
        image_path = None
        index = src if isinstance(src, (int, np.integer)) else real_index
        try:
            inline_dict_path = (
                isinstance(src, dict)
                and image_col['column'] in src
                and "url" not in image_col['column']
            )
            if inline_dict_path or 'cache_image' in image_col['column'] or 'cache_material' in image_col['column']:
                if isinstance(src, (int, np.integer)):
                    image_path = self.index_manager.get_attribute(src, **image_col)
                elif isinstance(src, dict):
                    image_path = src[image_col['column']]
                else:
                    image_path = src

                kwargs = {}
                if 'url_cos' in self._cos_required_keys:
                    try:
                        if isinstance(src, (int, np.integer)):
                            kwargs["url_cos"] = self.index_manager.get_attribute(src, **url_cos_col)
                        elif isinstance(src, dict):
                            kwargs["url_cos"] = src[url_cos_col]
                    except Exception as e:
                        pass
                image_path = self.parse_cos_path(image_path, **kwargs)
                image = self.read_local_image(image_path, convert_mode=convert_mode, apply_exif=apply_exif)

            elif image_col['column'] == "url_bucket_key":
                url_cos = src[url_cos_col.key]
                parsed_url = urlparse(url_cos)
                key = parsed_url.path.lstrip('/')
                image_path = f"/cos_nj1/share_302243067/{key}"
                image_path = self.parse_cos_path(image_path, url_cos=url_cos)
                image = self.read_local_image(image_path, convert_mode=convert_mode, apply_exif=apply_exif)

            elif "url" in image_col['column']:
                image = self.get_image_from_url_cos(src, apply_exif=apply_exif, **image_col)

            else:
                image = self.get_image_from_arrow(src, apply_exif=apply_exif, **image_col)

            image_flag = "normal"

            if np.all(np.array(image) == (0, 0, 0)):
                raise ValueError(f"Image is all black, index: {index}")

        except OSError as e:
            # Don't log 'OSError` keyword to avoid triggering platform's restart mechanism
            logger.error(f"({image_col=}, {index=}, {image_path=}): {e}.")
            image, image_flag = self.get_dummy_image()

        except Exception as e:
            logger.error(f"({image_col=}, {index=}, {image_path=}) {type(e)}: {e}.")
            image, image_flag = self.get_dummy_image()

        return image, image_flag

    def as_image_tensor(self, image, image_type, **kwargs) -> ImageTensor:
        if isinstance(image, Image.Image):
            tensor = self.pil_image_to_tensor(image)
        elif isinstance(image, np.ndarray):
            # 离线 latent（ndarray）：裸转 tensor，不做归一化（latent 非像素）
            tensor = torch.from_numpy(image)
        else:
            tensor = image

        origin_size = kwargs["origin_size"]
        ori_image_width = origin_size[0]
        ori_image_height = origin_size[1]

        if image_type == "vae":
            assert tensor.ndim == 3 or tensor.ndim == 4
            h, w = tensor.shape[-2], tensor.shape[-1]

            # ---- 新增：latent 模式，从 latent 维度反算像素尺寸 ----
            # 用显式 is_latent 标记而非全局 image_data_format，避免 latent 模式下
            # REPA 的像素图走进 latent 分支被误算尺寸。
            if kwargs.get("is_latent", False):
                tk_height, tk_width = h, w
                w = int(tk_width * self.vae_info.w_factor)
                h = int(tk_height * self.vae_info.h_factor)
            else:
                assert (h % self.vae_info.h_factor == 0 and w % self.vae_info.w_factor == 0), \
                    (f"Image size should be divisible by ({self.vae_info.h_factor}, {self.vae_info.w_factor}), "
                    f"but got ({h} x {w}).")
                tk_height = h // self.vae_info.h_factor
                tk_width = w // self.vae_info.w_factor

            if self.reso_strategy == "reso_group":
                base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(w, h)
            else:
                base_size, ratio_idx = None, None
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=w, image_height=h, token_width=tk_width, token_height=tk_height,
                base_size=base_size, ratio_index=ratio_idx,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
        elif image_type == "siglip2":
            spatial_shapes = kwargs["spatial_shapes"]  # 2  (h, w)
            pixel_attention_mask = kwargs["pixel_attention_mask"]  # seq_len
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=spatial_shapes[1].item() * self.vit_info.w_factor,
                image_height=spatial_shapes[0].item() * self.vit_info.h_factor,
                token_width=spatial_shapes[1].item(),
                token_height=spatial_shapes[0].item(),
                image_token_length=self.vit_info.max_token_length,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.vision_encoder_kwargs = {
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            }
        elif image_type == "anyres":
            token_width = kwargs["resized_image_width"] // self.vit_info.w_factor
            token_height = kwargs["resized_image_height"] // self.vit_info.h_factor
            # Get cat_extra_token from VISION_ENCODER_META_INFO
            encoder_meta = VISION_ENCODER_META_INFO.get(self.vit_info.encoder_type, {})
            cat_extra_token = encoder_meta.get("cat_extra_token", True)  # Default to 2 if not found
            extra_token_length = 2 if cat_extra_token else 0
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=kwargs["resized_image_width"],
                image_height=kwargs["resized_image_height"],
                token_width=token_width + 1,
                token_height=token_height,
                image_token_length=token_height * (token_width + 1) + extra_token_length,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
        elif image_type  == "qwen3vl":
            # Get spatial_merge_size from VISION_ENCODER_META_INFO
            encoder_meta = VISION_ENCODER_META_INFO.get(self.vit_info.encoder_type, {})
            spatial_merge_size = encoder_meta.get("spatial_merge_size", 2)  # Default to 2 if not found

            # qwen3vl image_grid_thw is (t, h, w)
            grid_height, grid_width = kwargs["image_grid_thw"][1].item(), kwargs["image_grid_thw"][2].item()
            token_height, token_width = grid_height // spatial_merge_size, grid_width // spatial_merge_size
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=grid_width * self.vit_info.w_factor,
                image_height=grid_height * self.vit_info.h_factor,
                token_width=token_width,
                token_height=token_height,
                image_token_length=token_width * token_height, 
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.vision_encoder_kwargs = {
                "grid_thw": kwargs["image_grid_thw"],
            }
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        return tensor

    def crop(self, image, target_size,  crop_type='center', crop_coords=None):
        """
        Crop the center/random part.

        Parameters
        ----------
        image: PIL.Image.Image
            The image to be cropped.
        target_size: tuple
            The target size of the image. A tuple of (width, height).
        crop_type: str
            Supported values include ('center', 'random', 'fixed'). Default to 'resize'.
            - If 'center', crop the center part of the image.
            - If 'random', crop a random part of the image.
            - If 'fixed', crop the part specified by crop_coords.
        crop_coords: tuple
            The left top coordinates of the crop. (crop_left, crop_top)
        Returns
        -------
        image: PIL.Image.Image
            The cropped image.
        crop_pos: tuple
            The position of the cropped part. (crop_left, crop_top)
        """
        tw, th = target_size
        w, h = image.size

        if crop_type == 'center':
            crop_top = int(round((h - th) / 2.0))
            crop_left = int(round((w - tw) / 2.0))
        elif crop_type == 'random':
            crop_top = random.randint(0, h - th)
            crop_left = random.randint(0, w - tw)
        elif crop_type == 'fixed':
            assert crop_coords is not None, 'crop_coords should be provided when crop_type is fixed.'
            crop_left, crop_top = crop_coords
        else:
            raise ValueError(f'crop_type must be center, random or fixed, but got {crop_type}')

        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

        return image, (crop_left, crop_top)

    def vae_process_image(self, image, target_size=None, random_crop: bool | str = False) -> ImageTensor:
        if self.image_data_format == "latents":
            assert isinstance(image, np.ndarray) and image.ndim == 4, \
                f"Image latent should be ndarray with 4 dims [C, 1, H, W], got {type(image)} ndim={getattr(image, 'ndim', '?')}"
            origin_size = (
                image.shape[3] * self.vae_info.w_factor,    # width
                image.shape[2] * self.vae_info.h_factor,    # height
            )
            processed = image # 转 tensor 交给 as_image_tensor 统一处理（ndarray 分支）
            is_latent = True
        else:
            # hunyuan-vae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
            # TODO: maybe we can try LANCZOS
            origin_size = image.size
            crop_type = random_crop if isinstance(random_crop, str) else ("random" if random_crop else "center")
            if crop_type == "center_and_no_resize":
                processed, _ = self.crop(image, target_size, crop_type="center")
            else:
                processed, _ = ArrowIndexV2.resize_and_crop(
                    image, target_size, crop_type=crop_type, resample=ResampleType[self.pixel_resample_type]
                )
            is_latent = False

        return self.as_image_tensor(processed, image_type=self.vae_info.image_type,
                                    origin_size=origin_size, is_latent=is_latent)

    def vit_process_image(self, image) -> ImageTensor:
        if not hasattr(self, "vit_info"):
            raise ValueError("'vit_info' is not defined. Please check if 'vit_image' is in 'modality'.")

        origin_size = image.size
        # ViT side also uses `reso_max_area` / `reso_max_size` as the VAE side.
        if getattr(self, "reso_max_area", None) is not None or getattr(self, "reso_max_size", None) is not None:
            target_size, ret_random_crop = compute_target_size_with_area_cap(
                origin_size[0], origin_size[1],
                max_area=self.reso_max_area,
                max_size=self.reso_max_size,
                align=self.reso_align,
                random_crop=False,
            )
            if ret_random_crop != "center_and_no_resize":
                image = image.resize(
                    target_size,
                    resample=ResampleType[self.pixel_resample_type],
                )
        inputs = self.vit_info.processor(image)
        image = inputs["pixel_values"].squeeze(0)   # (C, H, W)

        remain_keys = set(inputs.keys()) - {"pixel_values"}
        remain_kwargs = {}
        for key in remain_keys:
            if isinstance(inputs[key], torch.Tensor):
                remain_kwargs[key] = inputs[key].squeeze(0)
            else:
                remain_kwargs[key] = inputs[key]

        return self.as_image_tensor(image, image_type=self.vit_info.image_type, origin_size=origin_size, **remain_kwargs)

    def get_image_with_size(
            self,
            src: int | np.integer | dict[str, str | None] | InputImage,
            random_crop: bool | str = False,
            target_size_type: str = "index",
            return_type: str = "vae",
            real_index: int = None,
            apply_exif: bool = True,
            url_cos_col: IndexColumn | str | None = None,
            **image_col,
    ) -> tuple[ImageTensor | CondImage, bool]:
        """ For various image generation tasks, dynamic image sizes """

        is_latent_mode = self.image_data_format == "latents"

        # ======================== VAE 部分 ========================
        if is_latent_mode and "vae" in return_type:
            # ---- latent 模式：从 arrow 的 latent 字段加载离线 latent ----
            latent_col = self.index_columns["image_latent_col"]
            latent_np, flag = self.get_image_latents(src, real_index=real_index, **latent_col)
            img_success = flag != "error"
            vae_image_tensor = self.vae_process_image(latent_np)
        else:
            # ---- 原有像素模式 ----
            if isinstance(src, InputImage):
                assert target_size_type == "image", \
                    f"When `src` is an InputImage, `target_size_type` must be 'image', got {target_size_type}."
                image = load_image(src)
                image_flag = "normal"
            else:
                assert "column" in image_col, "`column` must be specified when calling get_image_with_size()."
                image, image_flag = self.get_raw_image(
                    src, real_index=real_index, apply_exif=apply_exif, url_cos_col=url_cos_col, **image_col
                )
            img_success = image_flag != "gray"
            origin_size = image.size  # (w_ori, h_ori)

            if "vae" in return_type:
                if self.reso_strategy == "anyres":
                    target_size, random_crop = compute_target_size_with_area_cap(
                        origin_size[0], origin_size[1],
                        max_area=self.reso_max_area,
                        max_size=self.reso_max_size,
                        align=self.reso_align,
                        random_crop=random_crop,
                    )
                elif target_size_type == "index":
                    target_size = self.index_manager.get_target_size(src)  # (w_tgt, h_tgt)
                elif target_size_type == "image":
                    target_size = self.vae_reso_group.get_target_size(*origin_size)
                else:
                    target_size = (self.reso_base_size, self.reso_base_size)
                vae_image_tensor = self.vae_process_image(image, target_size, random_crop=random_crop)
            else:
                vae_image_tensor = None

        # ======================== VIT 部分（无论哪种模式，逻辑一致）========================
        if "vit" in return_type:
            if is_latent_mode:
                image, image_flag = self.get_raw_image(
                    src, real_index=real_index, apply_exif=apply_exif, url_cos_col=url_cos_col, **image_col
                )
                img_success = img_success and image_flag != "gray"
            vit_image_tensor = self.vit_process_image(image)
        else:
            vit_image_tensor = None

        if return_type == "vae":
            image_tensor = vae_image_tensor
        elif return_type == "vit":
            image_tensor = vit_image_tensor
        elif return_type == "vae_vit":
            image_tensor = CondImage(image_type=return_type, vae_image=vae_image_tensor, vit_image=vit_image_tensor)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        return image_tensor, img_success

    def _prepend_timestep_to_vae_slices(self, slices, vae_slices, timestep_indices, batch_idx=None):
        if not self.timestep_vae_full_attn or timestep_indices is None:
            return slices

        vae_slices = vae_slices[batch_idx] if batch_idx is not None else vae_slices
        if batch_idx is not None and isinstance(timestep_indices, torch.Tensor) and timestep_indices.ndim > 1:
            timestep_indices = timestep_indices[batch_idx]
        if isinstance(timestep_indices, torch.Tensor):
            timestep_indices = timestep_indices.tolist()
        if isinstance(timestep_indices, int):
            timestep_indices = [timestep_indices]

        assert len(timestep_indices) == len(vae_slices), (
            f"Number of timestep tokens ({len(timestep_indices)}) should match "
            f"number of VAE image slices ({len(vae_slices)})."
        )

        timestep_by_vae_start = {
            vae_slice.start: int(timestep_idx)
            for timestep_idx, vae_slice in zip(timestep_indices, vae_slices)
        }
        return [
            slice(timestep_by_vae_start[sli.start], sli.stop)
            if sli.start in timestep_by_vae_start else sli
            for sli in slices
        ]

    def prepare_gen_full_attn_slices(self, output, batch_idx=None):
        gen_image_slices = output.gen_image_slices[batch_idx] if batch_idx is not None else output.gen_image_slices
        return self._prepend_timestep_to_vae_slices(
            gen_image_slices,
            output.gen_image_slices,
            output.gen_timestep_scatter_index,
            batch_idx=batch_idx,
        )

    def prepare_full_attn_slices(self, output, batch_idx=None, with_gen=True):
        """ Determine full attention image slices according to strategies. """
        if not hasattr(self, "cond_image_type"):
            return []

        if self.cond_image_type == "vae":
            cond_choices = dict(
                causal=[],
                full=output.vae_image_slices[batch_idx] if batch_idx is not None else output.vae_image_slices
            )

        elif self.cond_image_type == "vit":
            cond_choices = dict(
                causal=[],
                full=output.vit_image_slices[batch_idx] if batch_idx is not None else output.vit_image_slices
            )

        elif self.cond_image_type == "vae_vit":
            cond_choices = {
                "causal": [],
                "full": (
                    output.vae_image_slices[batch_idx] + output.vit_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.vae_image_slices + output.vit_image_slices
                ),
                "joint_full": (
                    output.joint_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.joint_image_slices
                ),
                "full_causal": (
                    output.vae_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.vae_image_slices
                ),
            }

        else:
            raise ValueError(f"Unknown cond_image_type: {self.cond_image_type}")
        slices = cond_choices[self.cond_token_attn_type]
        if self.cond_image_type in ["vae", "vae_vit"] and self.cond_token_attn_type != "causal":
            slices = self._prepend_timestep_to_vae_slices(
                slices,
                output.vae_image_slices,
                output.cond_timestep_scatter_index,
                batch_idx=batch_idx,
            )

        if with_gen:
            slices = slices + self.prepare_gen_full_attn_slices(output, batch_idx=batch_idx)
        return slices


class ImageProcessor(ImageMixin):
    def __init__(self, args: Namespace):
        super().__init__()
        self.modality = args.modality
        self.infer_align_image_size = args.infer_align_image_size
        self.img_ratio_slice_logits_processor = None
        self.img_tw_th_slice_logits_processor = None
        self.task_kwargs = {}
        self.index_kwargs = {}
        self.setup_image(args)

    def build_gen_image_info(self, image_size) -> ImageInfo:
        # parse image size (HxW, H:W, or <img_ratio_i>)
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif 'x' in image_size:
                image_size = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                image_size = [int(s) for s in image_size.split(':')]
                assert len(image_size) == 2, f"`image_size` should be in the format of 'W:H', got {image_size}."
                # Note that ratio is width:height
                image_size = [image_size[1], image_size[0]]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'W:H' or <img_ratio_i>, got {image_size}.")
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {image_size}.")
        if self.reso_strategy == "reso_group":
            image_width, image_height = self.vae_reso_group.get_target_size(image_size[1], image_size[0])
        elif self.reso_strategy == "anyres":
            ori_width, ori_height = image_size[1], image_size[0]
            target_size, _ = compute_target_size_with_area_cap(
                    ori_width, ori_height,
                    max_area=self.reso_max_area,
                    max_size=self.reso_max_size,
                    align=self.reso_align,
                    random_crop=False,
                )
            image_width, image_height = target_size[0], target_size[1]
            if image_width != ori_width or image_height != ori_height:
                logger.warning(f"Image size is resized from {ori_width}x{ori_height} to {image_width}x{image_height}.")
            assert image_width > 0 and image_height > 0, \
                f"Image size should be greater than 0, but got ({image_width} x {image_height})."
            assert image_width % self.reso_align == 0 and image_height % self.reso_align == 0, \
                f"Image size should be divisible by {self.reso_align}, but got ({image_width} x {image_height})."
        else:
            raise ValueError(f"Unknown resolution strategy: {self.reso_strategy}")
        token_height = image_height // self.vae_info.h_factor
        token_width = image_width // self.vae_info.w_factor
        if self.reso_strategy == "reso_group":
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        else:
            base_size, ratio_idx = None, None
        image_info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, base_size=base_size, ratio_index=ratio_idx,
        )
        return image_info

    def build_cond_images(
            self,
            image_list: Optional[list[InputImage]] = None,
            message_list: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[list[CondImage | ImageTensor]]:
        if image_list is not None and message_list is not None:
            raise ValueError("`image_list` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            image_list = []
            for message in message_list:
                visuals = [
                    content
                    for content in message["content"]
                    if isinstance(content, dict) and content["type"] in ["image"]
                ]
                image_list.extend([
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["image", "url", "path", "base64"]
                    if key in vision_info and vision_info["type"] == "image"
                ])

        if self.infer_align_image_size:
            random_crop = "resize"
        else:
            random_crop = "center"
        return [
            self.get_image_with_size(
                src, target_size_type="image", random_crop=random_crop, return_type=self.cond_image_type,
            )[0]
            for src in image_list
        ]
    
    def build_img_ratio_slice_logits_processor(self, tokenizer):
        if self.img_ratio_slice_logits_processor is None:
            if hasattr(tokenizer, "start_ratio_token_id") and hasattr(tokenizer, "end_ratio_token_id"):
                vocab_kwargs = dict(
                    vocab_start=tokenizer.start_ratio_token_id,
                    vocab_end=tokenizer.end_ratio_token_id + 1,
                )
            else:
                vocab_kwargs = dict(
                    vocab_start=tokenizer.ratio_token_id(0),
                    vocab_end=tokenizer.ratio_token_id(0) + len(self.vae_reso_group),
                )
            self.img_ratio_slice_logits_processor = get_logits_processors([
                dict(
                    SliceVocabLogitsWarper=dict(
                        **vocab_kwargs,
                        other_slices=getattr(tokenizer, "ratio_token_other_slices", []),
                    ),
                ),
            ])

    def build_img_tw_th_slice_logits_processor(self, tokenizer):
        if self.img_tw_th_slice_logits_processor is None:
            self.img_tw_th_slice_logits_processor = get_logits_processors([
                dict(
                    MaskVocabLogitsWarper=dict(
                        vocab_start=tokenizer.tw_th_token_id(1),
                        vocab_end=tokenizer.tw_th_token_id(512) + 1,
                    ),
                ),
            ])

    def postprocess_outputs(self, outputs: list[Image.Image], batch_cond_images):
        if self.infer_align_image_size:
            target_area = self.vae_reso_group.base_size ** 2

            for batch_index, (output_image, cond_images) in enumerate(zip(outputs, batch_cond_images)):
                output_image_ratio_index = self.vae_reso_group.get_base_size_and_ratio_index(width=output_image.width, height=output_image.height)[1]
                cond_images_ratio_index_list = []
                cond_images_ori_width_list = []
                cond_images_ori_height_list = []
                for cond_image in cond_images:
                    if isinstance(cond_image, ImageTensor):
                        cond_images_ratio_index_list.append(cond_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.i.ori_image_height)
                    else: # CondImage
                        cond_images_ratio_index_list.append(cond_image.vae_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.vae_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.vae_image.i.ori_image_height)

                if len(cond_images) == 0:
                    continue
                elif len(cond_images) == 1:
                    if output_image_ratio_index == cond_images_ratio_index_list[0]:
                        if abs(cond_images_ori_height_list[0] / cond_images_ori_width_list[0] - self.vae_reso_group[output_image_ratio_index].ratio) >= 0.01:
                            scale = math.sqrt(target_area / (cond_images_ori_width_list[0] * cond_images_ori_height_list[0]))
                            new_w = round(cond_images_ori_width_list[0] * scale)
                            new_h = round(cond_images_ori_height_list[0] * scale)
                            outputs[batch_index] = output_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                else:
                    for cond_image_ratio_index, cond_image_ori_width, cond_image_ori_height in zip(cond_images_ratio_index_list, cond_images_ori_width_list, cond_images_ori_height_list):
                        if output_image_ratio_index == cond_image_ratio_index:
                            if abs(cond_image_ori_height / cond_image_ori_width - self.vae_reso_group[output_image_ratio_index].ratio) >= 0.01:
                                scale = math.sqrt(target_area / (cond_image_ori_width * cond_image_ori_height))
                                new_w = round(cond_image_ori_width * scale)
                                new_h = round(cond_image_ori_height * scale)
                                outputs[batch_index] = output_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                            break

        return outputs
