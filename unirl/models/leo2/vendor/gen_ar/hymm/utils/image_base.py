from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple, List

import torch
from PIL import Image
from easydict import EasyDict
from torchvision import transforms
from index_kits import ArrowIndexV2

from .helpers import default
from .resolution import ResolutionGroup


class ImageInfo:
    """ Class to store image information for processing and generation. """
    args: EasyDict | dict

    def __init__(
            self,
            image_type: str = None,
            image_tensor: torch.Tensor = None,
            image_width: int = None,
            image_height: int = None,
            token_width: int = None,
            token_height: int = None,
            image_token_length: int = None,
            base_size: int = None,
            ratio_index: int = None,
            face_image: Optional[Image.Image] = None,
            ori_image_width: int = None,
            ori_image_height: int = None,
    ):
        if self.args is None:
            raise ValueError("ImageInfo requires `args` attribute to be set.")

        self.image_type = image_type
        self.image_tensor = image_tensor
        self.ori_image_width = ori_image_width
        self.image_width = image_width
        self.w = image_width
        self.ori_image_height = ori_image_height
        self.image_height = image_height
        self.h = image_height
        self.token_width = token_width
        self.tk_w = token_width
        self.token_height = token_height
        self.tk_h = token_height
        self.image_token_length = default(
            image_token_length,
            token_width * token_height if (token_width is not None and token_height is not None) else None
        )
        self.base_size = base_size
        self.ratio_index = ratio_index
        self.face_image = face_image

        # args
        self.add_iw_ih_token = self.args.get("add_iw_ih_token", False)
        self.add_timestep_token = self.args.get("add_timestep_token", False)
        self.add_timestep_r_token = self.args.get("add_timestep_r_token", False)
        self.add_guidance_token = self.args.get("add_guidance_token", False)
        self.use_front_boi_token = self.args.get("use_front_boi_token", False)
        self.add_image_shape_token = self.args.get("add_image_shape_token", False)
        self.add_tw_th_token = self.args.get("add_tw_th_token", False)
        self.gen_image_template = self.args.get("gen_image_template", "default")
        self.ignore_boi_token = self.args.get("ignore_boi_token", False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to attributes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the ImageInfo object."""
        return hasattr(self, key)

    def __repr__(self):
        return (f"ImageInfo(image_type={self.image_type}, image_tensor={self.image_tensor}, "
                f"ori_image_width={self.ori_image_width}, ori_image_height={self.ori_image_height}, "
                f"image_width={self.image_width}, image_height={self.image_height}, "
                f"token_width={self.token_width}, token_height={self.token_height}, "
                f"image_token_length={self.image_token_length}, "
                f"base_size={self.base_size}, ratio_index={self.ratio_index}, face_image={self.face_image})")

    @property
    def meta_info(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        # Used for image sections of tkwrapper.encode_general()
        if self.image_type in ["vae", "src_image", "gen_image"]:
            return dict(
                token_length=self.image_token_length,
                add_iw_ih_token=self.add_iw_ih_token,
                add_timestep_token=self.add_timestep_token,
                add_guidance_token=self.add_guidance_token,
                add_timestep_r_token=self.add_timestep_r_token,
                use_front_boi_token=self.use_front_boi_token,
                add_image_shape_token=self.add_image_shape_token,
                add_tw_th_token=self.add_tw_th_token,
                gen_image_template=self.gen_image_template,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                # for rope 2d and tw/th token (token_height/token_width used as h/w when add_tw_th_token)
                token_height=self.token_height,
                token_width=self.token_width,
                # for iw_ih_scatter_src
                image_height=self.image_height,
                image_width=self.image_width,
                ori_image_width=self.ori_image_width,
                ori_image_height=self.ori_image_height,
            )
        elif self.image_type in ["vision_encoder", "und_image", "siglip2", "anyres", "qwen3vl"]:
            return dict(
                token_length=self.image_token_length,
                add_iw_ih_token=self.add_iw_ih_token,
                use_front_boi_token=self.use_front_boi_token,
                add_image_shape_token=self.add_image_shape_token,
                add_tw_th_token=self.add_tw_th_token,
                # for rope 2d
                token_height=self.token_height,
                token_width=self.token_width,
                # for iw_ih_scatter_src
                image_height=self.image_height,
                image_width=self.image_width,
                ori_image_width=self.ori_image_width,
                ori_image_height=self.ori_image_height,
            )
        elif self.image_type == "face":
            return dict(
                token_length=self.image_token_length,
            )
        else:
            raise ValueError(f"Unknown image type '{self.image_type}'")

    @property
    def tool_call_meta_info(self):
        return dict(
            token_length=self.image_token_length,
            add_timestep_token=self.add_timestep_token,
            add_timestep_r_token=self.add_timestep_r_token,
            add_guidance_token=self.add_guidance_token,
            gen_image_template=self.gen_image_template,
            # for rope 2d
            token_height=self.token_height,
            token_width=self.token_width,
        )

    @property
    def num_special_tokens(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        if self.image_type in ["vae", "src_image", "gen_image"]:
            count = (
                    (2 if self.gen_image_template == "default" else 0) +
                    (2 if self.add_iw_ih_token else 0) +
                    (1 if self.add_timestep_token else 0) +
                    (1 if self.add_timestep_r_token else 0) +
                    (1 if self.add_guidance_token else 0) +
                    (2 if self.add_image_shape_token else 0) +
                    (2 if self.add_tw_th_token else 0)
            )
        elif self.image_type in ["vision_encoder", "und_image", "siglip2", "anyres", "qwen3vl"]:
            count = (
                    2  # <und_boi> + <und_eoi>
                    # TODO: Add special tokens for vision_encoder, und_image, siglip2, anyres, qwen3vl
                    # (2 if self.add_iw_ih_token else 0) +
                    # (2 if self.add_image_shape_token else 0) +
                    # (2 if self.add_tw_th_token else 0)
            )
        elif self.image_type == "face":
            count = 2   # <bof> + <eof>
        else:
            raise ValueError(f"Unknown image_type: {self.image_type}")
        return count
    
    @staticmethod
    def num_predicted_image_token_start_offset():
        return (
            (1 if ImageInfo.args["gen_image_template"] == "default" and ImageInfo.args["ignore_boi_token"] else 0)
        )

    @staticmethod
    def num_predicted_image_token_end_offset():
        return (
            (1 if ImageInfo.args["gen_image_template"] == "default" else 0) +
            (2 if ImageInfo.args["add_image_shape_token"] else 0) +
            (2 if ImageInfo.args["add_tw_th_token"] else 0)
        )

    @staticmethod
    def num_image_token_prefix():
        return (
            (1 if ImageInfo.args["gen_image_template"] == "default" else 0) +
            (2 if ImageInfo.args["add_image_shape_token"] else 0) +
            (2 if ImageInfo.args["add_tw_th_token"] else 0) +
            (1 if ImageInfo.args["add_timestep_token"] else 0)
        )

    @staticmethod
    def num_image_token_suffix():
        return (
            (1 if ImageInfo.args["gen_image_template"] == "default" else 0)
        )

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor and self.image_tensor is None:
            raise ValueError("image_tensor is None, cannot copy")
        return ImageInfo(
            image_type=self.image_type,
            image_tensor=self.image_tensor.clone() if copy_image_tensor else None,
            image_width=self.image_width,
            image_height=self.image_height,
            ori_image_width=self.ori_image_width,
            ori_image_height=self.ori_image_height,
            token_width=self.token_width,
            token_height=self.token_height,
            image_token_length=self.image_token_length,
            base_size=self.base_size,
            ratio_index=self.ratio_index,
            face_image=self.face_image,     # shared
        )

    def zeros_(self):
        self.image_tensor = torch.zeros_like(self.image_tensor)


class ImageTensor(torch.Tensor):
    # This class is just for type hinting purposes. Attribute `i` should be defined
    # as an instance attribute of the torch.Tensor instance, like: tensor.i = ImageInfo(...)
    i: ImageInfo
    vision_encoder_kwargs: dict[str, torch.Tensor]


class JointImageInfo(object):
    def __init__(self, vae_image_info: ImageInfo, vision_image_info: ImageInfo, vision_encoder_kwargs: dict = None):
        self.vae_image_info = vae_image_info
        self.vision_image_info = vision_image_info
        self.vision_encoder_kwargs = vision_encoder_kwargs

        # Define key attributes to align with ImageInfo for uniformity
        self.image_type = "joint_image"
        self.image_token_length = vae_image_info.image_token_length + vision_image_info.image_token_length

        self.add_iw_ih_token = vae_image_info.add_iw_ih_token
        self.add_timestep_token = vae_image_info.add_timestep_token
        self.use_front_boi_token = vae_image_info.use_front_boi_token
        self.add_image_shape_token = vae_image_info.add_image_shape_token
        self.add_tw_th_token = vae_image_info.add_tw_th_token

    def __repr__(self):
        return f"JointImageInfo(vae_image={self.vae_image_info}, vision_image={self.vision_image_info})"

    @property
    def meta_info(self):
        # Used for image sections of tkwrapper.encode_general()
        return dict(
            token_length=[self.vae_image_info.image_token_length, self.vision_image_info.image_token_length],
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            add_image_shape_token=self.add_image_shape_token,
            add_tw_th_token=self.add_tw_th_token,
            base_size=self.vae_image_info.base_size,
            ratio_idx=self.vae_image_info.ratio_index,
            # for rope 2d and tw/th token (vae token_height/token_width used as h/w when add_tw_th_token)
            token_height=[self.vae_image_info.token_height, self.vision_image_info.token_height],
            token_width=[self.vae_image_info.token_width, self.vision_image_info.token_width],
            # for iw_ih_scatter_src
            image_height=[self.vae_image_info.image_height, self.vision_image_info.image_height],
            image_width=[self.vae_image_info.image_width, self.vision_image_info.image_width],
        )

    @property
    def num_special_tokens(self):
        return (
                2 +  # <boi> + <eoi>
                (2 if self.add_iw_ih_token else 0) +
                (1 if self.add_timestep_token else 0) +
                (2 if self.add_image_shape_token else 0) +
                (2 if self.add_tw_th_token else 0) +
                1   # <joint_image_sep>
        )

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor and (self.vae_image_info.image_tensor is None or self.vision_image_info.image_tensor is None):
            raise ValueError("image_tensor is None, cannot copy")
        return JointImageInfo(self.vae_image_info.copy(copy_image_tensor), self.vision_image_info.copy(copy_image_tensor), self.vision_encoder_kwargs)

    def zeros_(self):
        self.vae_image_info.zeros_()
        self.vision_image_info.zeros_()


class JointImage(object):
    def __init__(self, vae_image: ImageTensor, vision_image: ImageTensor):
        self.vae_image = vae_image
        self.vision_image = vision_image
        self.vit_image = vision_image
        self.i = JointImageInfo(vae_image.i, vision_image.i)


class CondImage(object):
    def __init__(self, image_type: str, vae_image: ImageTensor, vit_image: ImageTensor):
        self.image_type = image_type
        self.vae_image = vae_image
        self.vit_image = vit_image

        if image_type == "vae":
            self.i = vae_image.i

        elif image_type == "vit":
            self.i = vit_image.i

        elif image_type == "vae_vit":
            self.i = JointImageInfo(vae_image.i, vit_image.i)

        else:
            raise ValueError(f"Unknown image_type: {image_type}")


@dataclass
class ImageProcessorConfig:
    image_base_size: int = 1024
    vit_processor: Dict = field(default_factory=dict)
    patch_size: int = 1
    vae_downsample_factor: Tuple = field(default_factory=lambda: (16, 16))  # height, width
    extra_resolutions: List[Tuple[int, int]] = field(default_factory=list)


class GeminiMoeImageProcessor(object):
    def __init__(self, config):
        from transformers import Siglip2ImageProcessorFast
        self.config = config

        self.reso_group = ResolutionGroup(base_size=config.image_base_size, extra_resolutions=config.extra_resolutions)
        self.vae_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
        ])
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_dict(config.vit_processor)

    def build_image_info(self, image_size):
        # parse image size (HxW, H:W, or <img_ratio_i>)
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif 'x' in image_size:
                image_size = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                image_size = [int(s) for s in image_size.split(':')]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'H:W' or <img_ratio_i>, got {image_size}.")
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {image_size}.")
        image_width, image_height = self.reso_group.get_target_size(image_size[1], image_size[0])
        token_height = image_height // (self.config.vae_downsample_factor[0] * self.config.patch_size)
        token_width = image_width // (self.config.vae_downsample_factor[1] * self.config.patch_size)
        base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        image_info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, base_size=base_size, ratio_index=ratio_idx,
        )
        return image_info

    def preprocess(self, image: Image.Image):
        # ==== VAE processor ====
        image_width, image_height = self.reso_group.get_target_size(image.width, image.height)
        resized_image, _ = ArrowIndexV2.resize_and_crop(image, (image_width, image_height), crop_type='center')
        image_tensor = self.vae_processor(resized_image)
        token_height = image_height // (self.config.vae_downsample_factor[0] * self.config.patch_size)
        token_width = image_width // (self.config.vae_downsample_factor[1] * self.config.patch_size)
        base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
        vae_image_info = ImageInfo(
            image_type="vae",
            image_tensor=image_tensor.unsqueeze(0),     # include batch dim
            image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height,
            base_size=base_size, ratio_index=ratio_index,
        )

        # ==== ViT processor ====
        inputs = self.vision_encoder_processor(image)
        image = inputs["pixel_values"].squeeze(0)  # seq_len x dim
        pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)  # seq_len
        spatial_shapes = inputs["spatial_shapes"].squeeze(0)  # 2  (h, w)
        vision_encoder_kwargs = dict(
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        vision_image_info = ImageInfo(
            image_type="vit",
            image_tensor=image.unsqueeze(0),  # 1 x seq_len x dim
            image_width=spatial_shapes[1].item() * self.config.vit_processor["patch_size"],
            image_height=spatial_shapes[0].item() * self.config.vit_processor["patch_size"],
            token_width=spatial_shapes[1].item(),
            token_height=spatial_shapes[0].item(),
            image_token_length=self.config.vit_processor["max_num_patches"],
            # may not equal to token_width * token_height
        )
        return JointImageInfo(vae_image_info, vision_image_info, vision_encoder_kwargs)
