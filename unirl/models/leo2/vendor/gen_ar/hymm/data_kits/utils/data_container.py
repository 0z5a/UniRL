from dataclasses import dataclass, field

import torch
from loguru import logger

from hymm.models.tokenizers.tokenization_hunyuan_multimodal import TokenizerEncodeOutput
from ...utils.helpers import default
from ...utils.audio_base import AudioTensor
from ...utils.image_base import ImageTensor, CondImage
from ...utils.video_base import VideoTensor, CondVideo


@dataclass
class BaseContainer:
    # Define common attributes for all data types
    success: bool     # set to False if any error occurs during data fetching
    index: int


def maybe_stack(values, key=None, subkey=None, strict=False):
    """
    Maybe stack a list of tensors into a single tensor.
    `value` is allowed to be following types:
    - None: return None
    - list of tensors with the same shape: return torch.stack(value, dim=0)
    - list of tensors with different shapes: return value as is
    - list of dicts with key specified:
      - key not in dict: return None
      - key in dict and all tensors have the same shape: return torch.stack([d[key] for d in value], dim=0)
      - key in dict and tensors have different shapes: return [d[key] for d in value]
    - others: raise error

    Args:
        values: The input value to be maybe stacked.
        key: The key to access the tensor in case value is a list of dicts.
        subkey: The subkey to access the tensor in case value is a list of dicts.
        strict: If True, raise error when shapes are different.
    """
    if values is None:
        return None
    if isinstance(values, list):
        if len(values) == 0:
            return None
        if key is not None:
            if isinstance(values[0], dict) and all(key not in d for d in values):
                return None
            # key must in all dicts
            values = [d[key] if isinstance(d, dict) else getattr(d, key) for d in values]
            if subkey is not None:
                values = [v[subkey] if isinstance(v, dict) else getattr(v, subkey) for v in values]
        # maybe stack tensors
        if not isinstance(values[0], torch.Tensor):
            return values
        shape0 = values[0].size()
        if all(isinstance(tensor, torch.Tensor) and tensor.size() == shape0 for tensor in values):
            return torch.stack(values, dim=0)
        elif strict:
            raise ValueError(
                f"maybe_stack strict mode: all tensors must have the same shape, "
                f"got {[tensor.size() if isinstance(tensor, torch.Tensor) else type(tensor) for tensor in values]}."
            )
        else:
            return values
    else:
        raise ValueError(f"maybe_stack only supports list type, got {type(values)}.")


@dataclass
class MultimodalDataContainer(BaseContainer):
    prompt: str | None = None
    recaption: str | None = None
    reasoning: str | None = None
    images: list[ImageTensor] | None = None
    cond_images: list[CondImage] | None = None
    messages: list[dict] | None = None
    cond_vae_images: list[ImageTensor] | None = None
    cond_vit_images: list[ImageTensor] | None = None
    videos: list[VideoTensor] | None = None
    video_last_frames: list[torch.Tensor] | None = None
    cond_videos: list[CondVideo] | None = None
    cond_vae_videos: list[VideoTensor] | None = None
    cond_vit_videos: list[VideoTensor] | None = None
    audios: list[AudioTensor] | None = None
    cond_audios: list[AudioTensor] | None = None
    status: str = "original"
    dataset_tag: str | None = None
    error: str | None = None

    def __post_init__(self):
        if self.cond_images:
            if not self.cond_vae_images:
                self.cond_vae_images = [cond_image.vae_image for cond_image in self.cond_images]
            if not self.cond_vit_images:
                self.cond_vit_images = [cond_image.vit_image for cond_image in self.cond_images]

        if self.cond_videos:
            if not self.cond_vae_videos:
                self.cond_vae_videos = [cond_video.vae_video for cond_video in self.cond_videos]
            if not self.cond_vit_videos:
                self.cond_vit_videos = [cond_video.vit_video for cond_video in self.cond_videos]

        # Sanity check
        # 1. messages should contain at least one gen field (text / image / video / audio)
        if self.messages is not None:
            gen_field_count = sum(
                1 for msg in self.messages
                if msg['type'] in ['gen_text', 'gen_image', 'gen_video', 'gen_audio']
            )
            if gen_field_count == 0:
                logger.error(f"(MultimodalDataContainer, index={self.index}) "
                             f"messages should contain at least one gen_text/gen_image/gen_video/gen_audio field.")
                self.success = False
        # 2. images/videos/videos should contain at least one media if messages is None
        if self.messages is None:
            if (
                    (self.images is None or len(self.images) == 0)
                    and (self.videos is None or len(self.videos) == 0)
                    and (self.audios is None or len(self.audios) == 0)
            ):
                if self.error is not None:
                    logger.error(f"(MultimodalDataContainer, index={self.index}): "
                                 f"error={self.error}.")
                else:
                    logger.error(f"(MultimodalDataContainer, index={self.index}): "
                                 f"images/videos/audios should contain at least one media if messages is None.")
                self.success = False

    def remove_unused_images(self, output: TokenizerEncodeOutput):
        num_gen_images = len(output.gen_image_slices)
        num_cond_images = len(output.joint_image_slices)

        if self.images and len(self.images) != num_gen_images:
            self.images = self.images[:num_gen_images]
        if self.cond_images and len(self.cond_images) != num_cond_images:
            self.cond_images = self.cond_images[:num_cond_images]
            self.cond_vae_images = self.cond_vae_images[:num_cond_images]
            self.cond_vit_images = self.cond_vit_images[:num_cond_images]

    @property
    def num_image_special_tokens(self):
        total = (
            sum(image.i.num_special_tokens for image in default(self.images, []))
            + sum(cond_image.i.num_special_tokens for cond_image in default(self.cond_images, []))
        )
        return total

    def remove_unused_videos(self, output: TokenizerEncodeOutput):
        if not getattr(output, "gen_video_slices", []):     # bc
            return
        num_gen_videos = len(output.gen_video_slices)

        if self.videos and len(self.videos) != num_gen_videos:
            self.videos = self.videos[:num_gen_videos]

    @property
    def num_video_special_tokens(self):
        total = (
            sum(video.i.num_special_tokens for video in default(self.videos, []))
        )
        return total

    def remove_unused_audios(self, output: TokenizerEncodeOutput):
        if not getattr(output, "gen_audio_slices", []):     # bc
            return
        num_gen_audios = len(output.gen_audio_slices)

        if self.audios and len(self.audios) != num_gen_audios:
            self.audios = self.audios[:num_gen_audios]

    @property
    def num_audio_special_tokens(self):
        total = (
            sum(audio.num_special_tokens for audio in default(self.audios, []))
        )
        return total

    def __repr__(self):
        return (
            f"MultimodalDataContainer(success={self.success}, status={self.status}, index={self.index}, "
            f"prompt={self.prompt}, "
            f"recaption={self.recaption}, "
            f"reasoning={self.reasoning}, "
            f"#images={len(self.images) if self.images is not None else 0}, "
            f"#cond_images={len(self.cond_images) if self.cond_images is not None else 0}, "
            f"#cond_vae_images={len(self.cond_vae_images) if self.cond_vae_images is not None else 0}, "
            f"#cond_vit_images={len(self.cond_vit_images) if self.cond_vit_images is not None else 0}, "
            f"#videos={len(self.videos) if self.videos is not None else 0}, "
            f"#video_last_frames={len(self.video_last_frames) if self.video_last_frames is not None else 0}, "
            f"#cond_videos={len(self.cond_videos) if self.cond_videos is not None else 0},"
            f"#cond_vae_videos={len(self.cond_vae_videos) if self.cond_vae_videos is not None else 0}, "
            f"#cond_vit_videos={len(self.cond_vit_videos) if self.cond_vit_videos is not None else 0}, "
            f"#audios={len(self.audios) if self.audios is not None else 0}, "
            f"messages={self.messages})"
        )
