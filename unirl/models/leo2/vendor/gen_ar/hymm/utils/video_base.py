from typing import Any

import torch
from easydict import EasyDict

from .helpers import default


class VideoInfo:
    """ Class to store video information for processing and generation. """
    args: EasyDict | dict

    def __init__(
            self,
            video_type: str = None,
            video_tensor: torch.Tensor = None,
            video_width: int = None,
            video_height: int = None,
            video_duration: int = None,
            token_width: int = None,
            token_height: int = None,
            token_duration: int = None,
            video_token_length: int = None,
            base_size: int = None,
            ratio_index: int = None,
            duration_index: int = None,
            ori_video_width: int = None,
            ori_video_height: int = None,
            ori_video_duration: int = None,
            timestamps: list[float] = None,
            video_id: int = None,
    ):
        if self.args is None:
            raise ValueError("VideoInfo requires `args` attribute to be set.")

        self.video_type = video_type
        self.video_tensor = video_tensor
        # Original video size
        self.ori_video_width = ori_video_width
        self.ori_video_height = ori_video_height
        self.ori_video_duration = ori_video_duration
        # Processed video size (after resizing/cropping)
        self.video_width = video_width
        self.video_height = video_height
        self.video_duration = video_duration
        self.w = video_width
        self.h = video_height
        self.d = video_duration
        # Video token size (after VAE encoding)
        self.token_width = token_width
        self.token_height = token_height
        self.token_duration = token_duration
        self.tk_w = token_width
        self.tk_h = token_height
        self.tk_d = token_duration
        self._video_token_length = video_token_length
        self.base_size = base_size
        self.ratio_index = ratio_index
        self.duration_index = duration_index
        # one timestamp per temporal tubelet, and an id that
        # distinguishes tubelets belonging to different source videos of the same sample.
        self.timestamps = timestamps
        self.video_id = video_id

        # args
        self.add_timestep_token = self.args.get("add_timestep_token", False)
        self.add_guidance_token = self.args.get("add_guidance_token", False)
        self.add_video_shape_token = self.args.get("add_video_shape_token", False)
        self.gen_template = self.args.get("gen_template", "default")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in VideoInfo")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to attributes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in VideoInfo")

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the VideoInfo object."""
        return hasattr(self, key)

    def __repr__(self):
        return (f"VideoInfo(video_type={self.video_type}, video_tensor={self.video_tensor}, "
                f"ori_video_width={self.ori_video_width}, ori_video_height={self.ori_video_height}, "
                f"ori_video_duration={self.ori_video_duration}, "
                f"video_width={self.video_width}, video_height={self.video_height}, "
                f"video_duration={self.video_duration}, "
                f"token_width={self.token_width}, token_height={self.token_height}, "
                f"token_duration={self.token_duration}, "
                f"video_token_length={self.video_token_length}, "
                f"base_size={self.base_size}, ratio_index={self.ratio_index}, duration_index={self.duration_index})")

    def inc_token_duration(self):
        if self.token_duration is not None:
            self.token_duration += 1
            self.tk_d += 1
            if self._video_token_length is not None:
                raise ValueError(
                    "video_token_length is already set, cannot increment token_duration without updating "
                    "video_token_length"
                )
        else:
            raise ValueError("token_duration is None, cannot increment")

    @property
    def video_token_length(self):
        return default(
            self._video_token_length,
            self.token_width * self.token_height * self.token_duration
            if self.token_width is not None and self.token_height is not None and self.token_duration is not None
            else None
        )

    @property
    def meta_info(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        # Used for video sections of tokenizer.encode_general()
        if self.video_type in ["vae", "gen_video"]:
            return dict(
                token_length=self.video_token_length,
                gen_template=self.gen_template,
                add_timestep_token=self.add_timestep_token,
                add_guidance_token=self.add_guidance_token,
                add_video_shape_token=self.add_video_shape_token,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                duration_idx=self.duration_index,
                # for rope 2d
                token_height=self.token_height,
                token_width=self.token_width,
                token_duration=self.token_duration,
            )
        elif self.video_type in ["vision_encoder", "und_video", "siglip2", "anyres", "qwen3vl"]:
            return dict(
                token_length=self.video_token_length,
                token_height=self.token_height,
                token_width=self.token_width,
                token_duration=self.token_duration,
            )
        else:
            raise ValueError(f"Unknown video type '{self.video_type}'")

    @property
    def num_special_tokens(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        if self.video_type in ["vae", "gen_video"]:
            count = (
                    (2 if self.gen_template == "default" else 0) +  # <bov> + <eov>
                    (1 if self.add_timestep_token else 0) +
                    (1 if self.add_guidance_token else 0) +
                    (3 if self.add_video_shape_token else 0)
            )
        elif self.video_type in ["vision_encoder", "und_video", "siglip2", "anyres", "qwen3vl"]:
            # Qwen video processing layout <vision_start>/<vision_end> pair per temporal tubelet.
            count = 2 * (self.token_duration or 1)
        else:
            raise ValueError(f"Unknown video type: {self.video_type}")
        return count

    def copy(self, copy_video_tensor=True):
        if copy_video_tensor and self.video_tensor is None:
            raise ValueError("video_tensor is None, cannot copy")
        return VideoInfo(
            video_type=self.video_type,
            video_tensor=self.video_tensor.clone() if copy_video_tensor else None,
            video_width=self.video_width,
            video_height=self.video_height,
            video_duration=self.video_duration,
            ori_video_width=self.ori_video_width,
            ori_video_height=self.ori_video_height,
            ori_video_duration=self.ori_video_duration,
            token_width=self.token_width,
            token_height=self.token_height,
            token_duration=self.token_duration,
            video_token_length=self.video_token_length,
            base_size=self.base_size,
            ratio_index=self.ratio_index,
            duration_index=self.duration_index,
            timestamps=list(self.timestamps) if self.timestamps is not None else None,
            video_id=self.video_id,
        )

    def zeros_(self):
        self.video_tensor = torch.zeros_like(self.video_tensor)


class VideoTensor(torch.Tensor):
    # This class is just for type hinting purposes. Attribute `i` should be defined
    # as an instance attribute of the torch.Tensor instance, like: tensor.i = VideoInfo(...)
    i: VideoInfo


class JointVideoInfo(object):
    def __init__(self, vae_video_info: VideoInfo, vision_video_info: VideoInfo, vision_encoder_kwargs: dict = None):
        self.vae_video_info = vae_video_info
        self.vision_video_info = vision_video_info
        self.vision_encoder_kwargs = vision_encoder_kwargs

        # Define key attributes to align with ImageInfo for uniformity
        self.video_type = "joint_video"
        self.video_token_length = vae_video_info.video_token_length + vision_video_info.video_token_length

        self.add_timestep_token = vae_video_info.add_timestep_token
        self.add_guidance_token = vae_video_info.add_guidance_token
        self.add_video_shape_token = vae_video_info.add_video_shape_token
        self.gen_template = vae_video_info.gen_template

    def __repr__(self):
        return f"JointVideoInfo(vae_video={self.vae_video_info}, vision_video={self.vision_video_info})"

    @property
    def meta_info(self):
        # Used for image sections of tkwrapper.encode_general()
        return dict(
            token_length=[self.vae_video_info.video_token_length, self.vision_video_info.video_token_length],
            gen_template=self.gen_template,
            add_timestep_token=self.add_timestep_token,
            add_video_shape_token=self.add_video_shape_token,
            base_size=self.vae_video_info.base_size,
            ratio_idx=self.vae_video_info.ratio_index,
            duration_idx=self.vae_video_info.duration_index,
            # for rope 2d and tw/th token (vae token_height/token_width used as h/w when add_tw_th_token)
            token_height=[self.vae_video_info.token_height, self.vision_video_info.token_height],
            token_width=[self.vae_video_info.token_width, self.vision_video_info.token_width],
            token_duration=[self.vae_video_info.token_duration, self.vision_video_info.token_duration],
        )

    @property
    def num_special_tokens(self):
        return (
                2 +  # <boi> + <eoi>
                (1 if self.add_timestep_token else 0) +
                (2 if self.add_video_shape_token else 0) +
                1   # <joint_image_sep>
        )

    def copy(self, copy_image_tensor=True):
        if copy_image_tensor \
                and (self.vae_video_info.video_tensor is None or self.vision_video_info.video_tensor is None):
            raise ValueError("image_tensor is None, cannot copy")
        return JointVideoInfo(
            self.vae_video_info.copy(copy_image_tensor),
            self.vision_video_info.copy(copy_image_tensor),
            self.vision_encoder_kwargs,
        )

    def zeros_(self):
        self.vae_video_info.zeros_()
        self.vision_video_info.zeros_()


class CondVideo(object):
    def __init__(self, video_type: str, vae_video: VideoTensor, vit_video: VideoTensor):
        self.video_type = video_type
        self.vae_video = vae_video
        self.vit_video = vit_video

        if video_type == "vae":
            self.i = vae_video.i

        elif video_type == "vit":
            self.i = vit_video.i

        elif video_type == "vae_vit":
            self.i = JointVideoInfo(
                vae_video.i, vit_video.i, getattr(vit_video, "vision_encoder_kwargs", None)
            )

        else:
            raise ValueError(f"Unknown video_type: {video_type}")
