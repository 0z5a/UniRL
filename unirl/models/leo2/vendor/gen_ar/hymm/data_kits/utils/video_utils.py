import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Union, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from index_kits import ArrowIndexV2, MultiIndexV2, DurationAndResolutionGroup
from loguru import logger
from transformers import BaseImageProcessor
from transformers.image_utils import SizeDict

from processors.image_kits import read_local_image
from .data_utils import DataMixin
from ...constants import VAE_META_INFO
from ...models.visual_encoders import load_vit_video_processor
from ...utils.states import DataClassMixin
from ...utils.video_base import VideoInfo, VideoTensor, CondVideo


@dataclass
class SampledVideoFrames:
    """Uniformly sampled RGB frames and their source-video timing metadata."""

    frames: np.ndarray
    frame_indices: list[int]
    total_frames: int
    fps: float | None

    @property
    def timestamps(self) -> list[float] | None:
        if self.fps is None or self.fps <= 0:
            return None
        return [index / self.fps for index in self.frame_indices]


@dataclass
class VideoSourceInfo:
    """Source-clip properties, for callers that size their sampling grid before decoding."""

    total_frames: int
    height: int
    width: int
    fps: float | None


def calculate_tubelet_timestamps(
    frame_indices: Sequence[int],
    fps: float | None,
    temporal_patch_size: int,
) -> list[float]:
    """Match Qwen3-VL's timestamp for each temporal patch."""

    if len(frame_indices) == 0:
        raise ValueError("frame_indices must contain at least one frame.")
    if temporal_patch_size <= 0:
        raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}.")
    if fps is None or fps <= 0:
        raise ValueError(
            f"Tubelet timestamps need the clip's own frame rate, got {fps!r}. Guessing one (Qwen "
            "defaults to 24) would put the timestamps on a clock the source never had."
        )

    video_fps = float(fps)
    padded_indices = [int(index) for index in frame_indices]
    pad = -len(padded_indices) % temporal_patch_size
    if pad:
        padded_indices.extend([padded_indices[-1]] * pad)

    return [
        (
            padded_indices[start] / video_fps
            + padded_indices[start + temporal_patch_size - 1] / video_fps
        )
        / 2
        for start in range(0, len(padded_indices), temporal_patch_size)
    ]


def _uniform_frame_indices(total_frames: int, num_frames: int | None) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}.")
    if num_frames is None:
        num_frames = total_frames
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")

    sample_count = min(total_frames, num_frames)
    return np.linspace(0, total_frames - 1, sample_count).round().astype(np.int64).tolist()


def frame_indices_for_fps(
    total_frames: int,
    source_fps: float | None,
    target_fps: float,
    max_frames: int | None = None,
) -> list[int]:
    """Frame indices on a grid anchored at t=0 and spaced at `target_fps`.

    Anchoring is what makes 2 FPS mean timestamps 0.0, 0.5, 1.0 ... instead of whatever an
    endpoint-to-endpoint split of the clip happens to produce, so the same instant of two clips
    carries the same timestamp. The stride only widens when the clip cannot serve the rate: never
    finer than one source frame, and wide enough that `max_frames` still spans the whole clip.
    The tail beyond the last grid point is not sampled, the same way a real 2 FPS capture ends.
    """

    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}.")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}.")
    if source_fps is None or source_fps <= 0:
        raise ValueError(
            f"Sampling at {target_fps} FPS needs the clip's own frame rate, got {source_fps!r}. "
            "Pin a frame count with --cond-video-vit-num-frames if the source cannot report one."
        )

    stride = float(source_fps) / target_fps
    if max_frames is not None:
        stride = max(stride, total_frames / max_frames)
    stride = max(stride, 1.0)

    count = int((total_frames - 1) / stride) + 1
    if max_frames is not None:
        count = min(count, max_frames)
    return [min(total_frames - 1, int(index * stride + 0.5)) for index in range(count)]


def _reader_fps(reader: Any) -> float | None:
    fps = float(reader.get_avg_fps())
    return fps if np.isfinite(fps) and fps > 0 else None


def _open_video_reader(video_path: str | Path) -> Any:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) <= 0:
        raise ValueError(f"Empty video: {video_path}")
    return reader


def open_video(video_path: str | Path) -> tuple[Any, VideoSourceInfo]:
    """Open a clip with decord and probe it, for callers that pick the frame count themselves.

    The reader comes back with the metadata so the probe and the following `sample_video_frames`
    share one decode session instead of opening the file twice.
    """

    reader = _open_video_reader(video_path)
    height, width = reader.get_batch([0]).asnumpy().shape[1:3]
    return reader, VideoSourceInfo(
        total_frames=len(reader),
        height=int(height),
        width=int(width),
        fps=_reader_fps(reader),
    )


def sample_video_frames(
    video: str | Path | Any,
    frames: int | Sequence[int] | None,
) -> SampledVideoFrames:
    """Read frames with decord, from a path or a reader from `open_video`.

    `frames` is either a count to spread evenly across the clip, or the exact indices to read.
    """

    reader = _open_video_reader(video) if isinstance(video, (str, Path)) else video
    total_frames = len(reader)
    if frames is None or isinstance(frames, int):
        indices = _uniform_frame_indices(total_frames, frames)
    else:
        indices = [int(index) for index in frames]
    frames = reader.get_batch(indices).asnumpy()
    return SampledVideoFrames(
        frames=frames,
        frame_indices=indices,
        total_frames=total_frames,
        fps=_reader_fps(reader),
    )


def process_qwen3vl_video_frames(
    video_processor: Any,
    frames: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one sampled frame sequence through Qwen3-VL's native video processor."""

    if frames.ndim != 4:
        raise ValueError(
            "Qwen3-VL video processor expects (num_frames, height, width, channels) "
            f"or (num_frames, channels, height, width), got {tuple(frames.shape)}."
        )

    outputs = video_processor(videos=frames, do_sample_frames=False, return_tensors="pt")
    pixel_values = outputs["pixel_values_videos"]
    grid_thw = outputs["video_grid_thw"]

    if pixel_values.ndim != 2:
        raise ValueError(
            f"Expected flattened Qwen3-VL video patches to be 2-D, got {pixel_values.shape}."
        )
    if grid_thw.shape != (1, 3):
        raise ValueError(f"Expected one video_grid_thw row, got {grid_thw.shape}.")

    expected_patches = int(grid_thw.prod().item())
    if pixel_values.shape[0] != expected_patches:
        raise ValueError(
            "Qwen3-VL video patch/grid mismatch: "
            f"{pixel_values.shape[0]} patches != {expected_patches} from grid {grid_thw.tolist()}."
        )
    return pixel_values, grid_thw.squeeze(0)


def qwen3vl_video_token_shape(grid_thw, spatial_merge_size: int) -> tuple[int, int, int, int]:
    """Convert a patch grid into the post-merge token grid (tubelets, height, width, total)."""

    grid = torch.as_tensor(grid_thw).reshape(-1, 3)
    if grid.shape[0] != 1:
        raise ValueError(f"Expected one video_grid_thw row per video, got {grid.tolist()}.")
    num_tubelets, grid_height, grid_width = (int(value) for value in grid[0])
    token_height = grid_height // spatial_merge_size
    token_width = grid_width // spatial_merge_size
    return num_tubelets, token_height, token_width, num_tubelets * token_height * token_width


def merge_qwen3vl_video_grid_thw(video_tensor: torch.Tensor) -> torch.Tensor:
    """Return the patch grid for one native Qwen video tensor."""

    if getattr(getattr(video_tensor, "i", None), "video_type", None) != "qwen3vl":
        raise ValueError(
            "Condition videos require the native Qwen video route; "
            "per-frame image tensors are not supported."
        )
    vision_kwargs = getattr(video_tensor, "vision_encoder_kwargs", None)
    if not vision_kwargs or "grid_thw" not in vision_kwargs:
        raise ValueError("Native Qwen condition video is missing grid_thw metadata.")
    grid = vision_kwargs["grid_thw"].reshape(-1, 3)
    timestamps = vision_kwargs.get("timestamps")
    if timestamps is None or len(timestamps) != int(grid[0, 0]):
        raise ValueError(
            "Native Qwen condition video requires one timestamp per tubelet: "
            f"{timestamps=} for grid {grid.tolist()}."
        )
    return grid


def modality_with_cond_video_vit(modality: Sequence[str], cond_video_type: str | None) -> tuple[str, ...]:
    """Condition videos go through Qwen's native video processor, which the `vit_video` modality sets up.

    Only for callers that cannot declare the modality themselves, such as the inference-side
    `VideoProcessor` reading the global `args.modality`; datasets declare it in their task kwargs.
    """

    modality = tuple(modality)
    if "vit" in (cond_video_type or "") and "vit_video" not in modality:
        return modality + ("vit_video",)
    return modality


def normalize_cond_video_vae_paths(cond_video_vae_path, batch_size: int) -> list[list[str | None]]:
    """Normalize offline VAE inputs to one ordered path list per sample.

    A flat list keeps its per-sample meaning when ``batch_size > 1``. For a single sample it may
    contain one path per condition video. Nested lists express multiple paths for every batch item;
    ``None`` selects online VAE encoding for that video.
    """

    if cond_video_vae_path is None:
        sample_values = [None] * batch_size
    elif isinstance(cond_video_vae_path, str):
        sample_values = [cond_video_vae_path] * batch_size
    elif isinstance(cond_video_vae_path, (list, tuple)):
        values = list(cond_video_vae_path)
        if batch_size == 1 and all(value is None or isinstance(value, str) for value in values):
            sample_values = [values]
        elif len(values) == batch_size:
            sample_values = values
        else:
            raise ValueError(
                "cond_video_vae_path batch size mismatch: expected either "
                f"{batch_size} per-sample entries or one flat per-video list "
                f"for a single sample, got {len(values)} entries."
            )
    else:
        raise TypeError(
            "cond_video_vae_path must be None, a path string, a per-sample path list, "
            f"or a nested per-sample/per-video path list, got {type(cond_video_vae_path)}."
        )

    result = []
    for sample_value in sample_values:
        if isinstance(sample_value, (list, tuple)):
            paths = list(sample_value)
        elif sample_value is None:
            paths = []
        else:
            paths = [sample_value]

        sample_paths = []
        for path in paths:
            if path is None or (
                isinstance(path, str) and (not path.strip() or path.strip().lower() == "nan")
            ):
                sample_paths.append(None)
                continue
            if not isinstance(path, str):
                raise TypeError(
                    f"Each cond_video_vae_path must be a string or None, got {type(path)}."
                )
            path = path.strip()
            if not Path(path).is_file():
                raise FileNotFoundError(f"Condition-video VAE latent does not exist: {path}")
            sample_paths.append(path)
        result.append(sample_paths if any(path is not None for path in sample_paths) else [])
    return result


@dataclass
class DurationAndResolutionGroupConfig(DataClassMixin):
    duration_range: tuple[int, int] = None
    duration_step: int = None
    additional_durations: list[int] = None
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
            duration_range=tuple(kwargs.get("duration_range", args.duration_range)),
            duration_step=kwargs.get("duration_step", args.duration_step),
            additional_durations=kwargs.get("additional_durations", args.additional_durations),
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
class VideoVAEInfo:
    encoder_type: str
    down_h_factor: int = -1
    down_w_factor: int = -1
    down_d_factor: int = -1
    patch_size: int = 1
    h_factor: int = -1
    w_factor: int = -1
    d_factor: int = -1
    latent_dim: int = -1
    video_type: str = None

    def __post_init__(self):
        self.h_factor = self.down_h_factor * self.patch_size
        self.w_factor = self.down_w_factor * self.patch_size
        self.d_factor = self.down_d_factor
        if self.video_type is None:
            self.video_type = "vae"


@dataclass
class VideoViTInfo:
    encoder_type: str
    h_factor: int = -1
    w_factor: int = -1
    max_token_length: int = 0   # pad to max_token_length
    processor: Callable = field(default_factory=BaseImageProcessor)
    video_type: str = None
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2

    def __post_init__(self):
        if self.video_type is None:
            self.video_type = self.encoder_type.split("-")[0]


class VideoMixin(DataMixin):
    """
    A mixin class for video processing and generation.

    Notice: Make sure all the method and attribute names contain 'video' to avoid name conflicts with other mixins,
            unless they are compatible with both image and video modalities.
    """
    dataset_tag: str
    task_kwargs: dict
    index_kwargs: dict
    modality: list[str]
    index_manager: "Union[ArrowIndexV2, MultiIndexV2]"
    index_columns: dict

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

    def setup_video(self, args):
        VideoInfo.args = dict(
            gen_template=args.gen_template,
            add_timestep_token=args.add_timestep_token,
            add_video_shape_token=args.add_video_shape_token,
        )

        self.cond_video_type = args.cond_video_type
        self.cond_video_token_attn_type = args.cond_video_token_attn_type
        self.cond_video_vit_fps = args.cond_video_vit_fps
        self.cond_video_vit_max_frames = args.cond_video_vit_max_frames
        self.cond_video_vit_num_frames = args.cond_video_vit_num_frames
        self.cond_video_frame_cache_interval_sec = self.task_kwargs.get(
            "cond_video_frame_cache_interval_sec", args.cond_video_frame_cache_interval_sec
        )
        if (self.cond_video_frame_cache_interval_sec is not None
                and self.cond_video_frame_cache_interval_sec <= 0):
            raise ValueError(
                "cond_video_frame_cache_interval_sec must be positive, got "
                f"{self.cond_video_frame_cache_interval_sec}."
            )
        if "vae" in self.cond_video_type:
            assert self.cond_video_token_attn_type in ["causal", "full"], \
                f"When cond-video-type includes 'vae', cond-video-token-attn-type should be 'causal' or 'full', " \
                f"but got {self.cond_video_token_attn_type}."

        # -- vae video --
        if "vae_video" in self.modality:
            self.require_configs(args, [
                "vae_type", "patch_size", "vae_video_token_length", "video_cos_base", "video_data_format"
            ], "vae_video modality")

            self.video_data_format = args.video_data_format
            self.video_cos_base = Path(self.task_kwargs.get("video_cos_base", args.video_cos_base))
            # last_frame_data_format 是增量功能，按 dataset 从 task_kwargs 取（与 audio_data_format 一致，无全局回退）
            self.last_frame_data_format = self.task_kwargs.get("last_frame_data_format", "pixels")
            # last frame latent 使用独立 cos_base；未配置时回退到 video_cos_base。
            self.last_frame_cos_base = Path(getattr(args, "last_frame_cos_base", None) or args.video_cos_base)
            if self.last_frame_data_format == "latents":
                assert self.last_frame_cos_base and str(self.last_frame_cos_base) != "", (
                    "last_frame_data_format='latents' 需要配置 DATA_ARGS.last-frame-cos-base "
                    "或 DATA_ARGS.video-cos-base（离线 last frame latent .npy 的 COS 根路径）。"
                )
            self.vae_video_token_length = self.task_kwargs.get("vae_video_token_length", args.vae_video_token_length)
            self.dar_group_config = DurationAndResolutionGroupConfig.from_args(
                args, **self.index_kwargs.get("dar_bucket_kwargs", {})
            )
            if self.index_kwargs.get("online_bucketing") and hasattr(self, "index_manager"):
                self.index_manager.set_duration_and_resolution_buckets(**self.dar_group_config.to_dict())
                self.index_manager.register_get_size_fn(self.get_video_size_fn())
            self.vae_dar_group = DurationAndResolutionGroup(**self.dar_group_config.to_dict())
            # vae info
            vae_meta_info = VAE_META_INFO[args.vae_type]
            downsample_factor = vae_meta_info["downsample_factor"]
            self.video_vae_info = VideoVAEInfo(
                encoder_type=args.vae_type,
                down_h_factor=downsample_factor[0],
                down_w_factor=downsample_factor[1],
                down_d_factor=vae_meta_info["duration_downsample_factor"],
                patch_size=args.patch_size,
                latent_dim=vae_meta_info["latent_dim"],
            )

        # -- vit video --
        if "vit_video" in self.modality:
            self.require_configs(args, ["vit_type", "vit_video_token_length"], "vit_video modality")
            if not args.vit_type.startswith("qwen3vl-vit"):
                raise NotImplementedError(
                    f"Condition-video ViT requires a native Qwen video processor, got {args.vit_type!r}."
                )

            self.vit_video_token_length = self.task_kwargs.get("vit_video_token_length",
                                                               args.vit_video_token_length)
            self.min_vit_video_token_length = self.task_kwargs.get("min_vit_video_token_length",
                                                                   args.min_vit_video_token_length)
            processor = load_vit_video_processor(args.vit_type)
            # smart_resize budgets raw pixels (frames * h * w) while the configured lengths count
            # post-merge tokens, and one video token spans a (patch*merge)^2 block across
            # temporal_patch_size frames -- unlike an image token, which covers a single frame.
            # 对QwenVL-3.5 序列来说，最终的分辨率下采样32*32*2=2048，一个视频token span 2048个像素
            pixels_per_token = (processor.patch_size * processor.merge_size) ** 2 \
                * processor.temporal_patch_size
            processor.size = SizeDict(
                shortest_edge=self.min_vit_video_token_length * pixels_per_token,
                longest_edge=self.vit_video_token_length * pixels_per_token,
            )

            self.vit_video_info = VideoViTInfo(
                encoder_type=args.vit_type,
                h_factor=processor.patch_size,
                w_factor=processor.patch_size,
                max_token_length=self.vit_video_token_length,
                processor=processor,
                temporal_patch_size=processor.temporal_patch_size,
                spatial_merge_size=processor.merge_size,
            )

        # -- unconditions --
        self.video_uncond_p = self.task_kwargs.get('uncond_p', 0.0)

    def read_local_video_latent(self, latent_key: str) -> np.ndarray:
        if not latent_key.endswith(".npy"):
            latent_key += ".npy"
        latent_path = Path(latent_key) if latent_key.startswith("/") else self.video_cos_base / latent_key.lstrip("/")
        latent = np.load(latent_path)
        # Sanity check
        assert latent.ndim == 5 and latent.shape[:2] == (1, self.video_vae_info.latent_dim), \
            f"Video latent should have 5 dimensions (1, C, D, H, W), but got {latent.shape}."
        return latent

    def read_local_last_frame_latent(self, latent_key: str) -> np.ndarray:
        """读取离线提取的 last frame latent（单帧，使用独立的 last_frame_cos_base）。"""
        if not latent_key.endswith(".npy"):
            latent_key += ".npy"
        latent_path = Path(latent_key) \
            if latent_key.startswith("/") \
            else self.last_frame_cos_base / latent_key.lstrip("/")
        latent = np.load(latent_path)
        latent_dim = self.video_vae_info.latent_dim
        assert latent.ndim == 5 and latent.shape[:3] == (1, latent_dim, 1), \
            f"Last frame latent should be (1, {latent_dim}, 1, H, W), but got {latent.shape}."
        return latent

    def get_video_size_fn(self):
        # Return the video original size when only reading latents.
        self.require_configs(self.index_columns, ["video_size_col"],
                             f"{self.dataset_tag}.{self.dataset_tag}_index_kwargs.index_columns")
        column = self.index_columns["video_size_col"].key

        def get_video_size(index_manager: "Union[ArrowIndexV2, MultiIndexV2]", index: int):
            latent_shape = index_manager.get_attribute(index, column)
            if isinstance(latent_shape, str):
                latent_shape = json.loads(latent_shape)
            if isinstance(latent_shape[0], list):
                latent_shape = latent_shape[0]
            assert isinstance(latent_shape, list) and len(latent_shape) == 5, \
                f"{column} should contain a list of 5 integers representing the latent shape, but got {latent_shape}."
            width = latent_shape[-1] * self.video_vae_info.w_factor
            height = latent_shape[-2] * self.video_vae_info.h_factor
            duration = (latent_shape[-3] - 1) * self.video_vae_info.d_factor + 1
            return width, height, duration

        return get_video_size

    def get_video_latents(
            self,
            src: int | dict,
            real_index: int = None,
            **video_col,
    ):
        video_latent_path = None
        try:
            if 'latent_cos_path' in video_col['column']:
                if isinstance(src, dict):
                    # Interleave `message` item: the latent path lives inline in the message dict.
                    video_latent_path = src[video_col['column']]
                else:
                    video_latent_path = self.index_manager.get_attribute(src, **video_col)
            else:
                video_latent_path = self.index_manager.get_attribute(src, **video_col)
            if not video_latent_path:
                raise ValueError(f"empty video latent path (column={video_col.get('column')!r})")
            latents = self.read_local_video_latent(video_latent_path).squeeze(0)
            # Check inf for video latent(ndarray)
            if np.isinf(latents).any():
                raise ValueError(f"Inf detected in video latents: {src=}, {video_latent_path=}")
            max_val = np.abs(latents).max()
            if max_val > 10:
                raise ValueError(f"Abnormal max value {max_val} detected in video latents: {src=}, {video_latent_path=}")
            video_flag = "normal"
        except Exception as e:
            index = src if isinstance(src, int) else real_index
            logger.error(f"({video_col=}, {index=}, {video_latent_path=}) {type(e)}: {e}.")
            video_flag = "error"
            latents = np.zeros(
                (
                    self.video_vae_info.latent_dim,
                    (self.vae_dar_group.duration_start - 1) // self.video_vae_info.d_factor + 1,
                    self.vae_dar_group.reso_base_size // self.video_vae_info.h_factor,
                    self.vae_dar_group.reso_base_size // self.video_vae_info.w_factor,
                ),
                dtype=np.float16
            )
        return latents, video_flag

    def as_video_tensor(self, video, video_type, **kwargs) -> VideoTensor:
        origin_size = kwargs["origin_size"]

        if video_type == "vae":
            if self.video_data_format == "latents":
                video_tensor = torch.from_numpy(video)
                _, tk_duration, tk_height, tk_width = video_tensor.shape
                width = tk_width * self.video_vae_info.w_factor
                height = tk_height * self.video_vae_info.h_factor
                duration = (tk_duration - 1) * self.video_vae_info.d_factor + 1
            else:
                raise NotImplementedError(f"Video data format {self.video_data_format} not implemented.")

            base_size, ratio_idx, duration_idx = \
                self.vae_dar_group.get_base_size_and_ratio_index(width, height, duration)
            video_tensor.i = VideoInfo(
                video_type=video_type,
                video_width=width, video_height=height, video_duration=duration,
                token_width=tk_width, token_height=tk_height, token_duration=tk_duration,
                base_size=base_size, ratio_index=ratio_idx, duration_index=duration_idx,
                ori_video_width=origin_size[0],
                ori_video_height=origin_size[1],
                ori_video_duration=origin_size[2],
            )

        elif video_type == "qwen3vl":
            # Native Qwen video grid is a single (t, h, w) row where t counts temporal tubelets,
            # each folding `temporal_patch_size` consecutive frames together.
            grid_thw = kwargs["video_grid_thw"].reshape(-1, 3)
            num_tubelets, token_height, token_width, token_length = qwen3vl_video_token_shape(
                grid_thw, self.vit_video_info.spatial_merge_size
            )
            grid_height, grid_width = int(grid_thw[0][1]), int(grid_thw[0][2])

            timestamps = kwargs["timestamps"]
            if len(timestamps) != num_tubelets:
                raise ValueError(
                    f"Expected {num_tubelets} tubelet timestamps, got {len(timestamps)}."
                )

            video_tensor = video
            video_tensor.i = VideoInfo(
                video_type=video_type,
                video_width=grid_width * self.vit_video_info.w_factor,
                video_height=grid_height * self.vit_video_info.h_factor,
                video_duration=origin_size[2] if len(origin_size) > 2 else None,
                token_width=token_width,
                token_height=token_height,
                token_duration=num_tubelets,
                video_token_length=token_length,
                ori_video_width=origin_size[0],
                ori_video_height=origin_size[1],
                timestamps=list(timestamps),
                video_id=kwargs.get("video_id"),
            )
            video_tensor.vision_encoder_kwargs = {
                "grid_thw": grid_thw,
                "timestamps": list(timestamps),
            }

        else:
            raise ValueError(f"Unknown video type: {video_type}")

        return video_tensor     # noqa

    def vae_process_video(self, video: np.ndarray):
        # Resize and crop to target size
        assert video.ndim == 4, f"Video data for vae should have 4 dimensions, but got {video.ndim}."

        if self.video_data_format == "latents":
            origin_size = (
                video.shape[3] * self.video_vae_info.w_factor,
                video.shape[2] * self.video_vae_info.h_factor,
                (video.shape[1] - 1) * self.video_vae_info.d_factor + 1,
            )
            # Crop to target size
            width, height, duration = self.vae_dar_group.get_target_size(*origin_size)
            tk_duration = (duration - 1) // self.video_vae_info.d_factor + 1
            video = video[:, :tk_duration]
            assert origin_size[:2] == (width, height), \
                f"Original size {origin_size[:2]} does not match target size {(width, height)}."
        else:
            raise NotImplementedError(f"Video data format {self.video_data_format} not implemented.")

        return self.as_video_tensor(video, video_type=self.video_vae_info.video_type, origin_size=origin_size)

    def cond_video_frame_indices(
        self,
        total_frames: int,
        source_fps: float | None,
        num_frames: int | None = None,
    ) -> list[int]:
        """Frames one condition clip contributes: a pinned count spread evenly, else a `cond_video_vit_fps` grid."""

        pinned = num_frames if num_frames is not None else self.cond_video_vit_num_frames
        if pinned is not None:
            return _uniform_frame_indices(total_frames, pinned)
        return frame_indices_for_fps(
            total_frames, source_fps, self.cond_video_vit_fps, self.cond_video_vit_max_frames
        )

    def sample_cached_frames(self, frame_paths: Sequence[str], num_frames: int | None) -> SampledVideoFrames:
        """Load pre-extracted frame images as if they were a freshly decoded clip."""

        if not frame_paths:
            raise ValueError("Condition-video ViT got an empty frame cache list.")
        if self.cond_video_frame_cache_interval_sec is None:
            raise ValueError(
                "Cached frames carry no timing of their own, so tubelet timestamps cannot be "
                "derived until --cond-video-frame-cache-interval-sec declares how far apart they "
                "sit. Point the dataset's `video_material_key` at the raw video to skip this."
            )
        paths = [self.parse_cos_path(p) for p in frame_paths]
        # The dump stride plays the role of the frame rate a decoded clip would report, which is
        # what keeps timestamps on the source video's clock.
        fps = 1.0 / self.cond_video_frame_cache_interval_sec
        indices = self.cond_video_frame_indices(len(paths), fps, num_frames)
        frames = np.stack([
            np.asarray(self.read_local_image(paths[i], convert_mode="RGB")) for i in indices
        ])
        return SampledVideoFrames(
            frames=frames,
            frame_indices=indices,
            total_frames=len(paths),
            fps=fps,
        )

    def vit_process_video_frames(self, video_src: str | Sequence[str], num_frames: int = None,
                                 video_id: int = None):
        """Encode one condition clip through the native Qwen video processor.

        Takes either a raw video path or a list of pre-extracted frame images. Either way the
        frames are folded into temporal tubelets, so the ViT sees inter-frame motion rather than
        a bag of independent stills.
        """
        if not hasattr(self, "vit_video_info"):
            raise ValueError("'vit_video_info' is not defined. Please check if 'vit_video' is in 'modality'.")

        if isinstance(video_src, str):
            # The sampling grid follows from the clip's own rate, so it is probed before sampling.
            reader, source = open_video(self.parse_cos_path(video_src))
            sampled = sample_video_frames(
                reader, self.cond_video_frame_indices(source.total_frames, source.fps, num_frames)
            )
        elif isinstance(video_src, (list, tuple)):
            sampled = self.sample_cached_frames(video_src, num_frames)
        else:
            raise TypeError(
                "Condition-video ViT expects a raw video path or a frame-image list, "
                f"got {type(video_src)}."
            )
        origin_size = (sampled.frames.shape[2], sampled.frames.shape[1])  # (w, h)

        pixel_values, grid_thw = process_qwen3vl_video_frames(self.vit_video_info.processor, sampled.frames)
        timestamps = calculate_tubelet_timestamps(
            sampled.frame_indices, sampled.fps, self.vit_video_info.temporal_patch_size
        )

        video_tensor = self.as_video_tensor(
            pixel_values,
            video_type=self.vit_video_info.video_type,
            origin_size=origin_size,
            video_grid_thw=grid_thw,
            timestamps=timestamps,
            video_id=video_id,
        )
        # A native clip yields a single consistent grid, so there is no cross-frame size mismatch.
        return video_tensor, True

    def get_video_with_size(
            self,
            src: int | dict,
            return_type: str = "vae",
            real_index: int = None,
            video_frame_col: str = None,
            video_id: int = None,
            **video_col,
    ):
        if self.video_data_format == "latents":
            video, video_flag = self.get_video_latents(
                src, real_index, **video_col
            )
        else:
            raise NotImplementedError(f"Video data format {self.video_data_format} not implemented.")

        video_success = video_flag != "error"

        if "vae" in return_type:
            vae_video_tensor = self.vae_process_video(video)
        else:
            vae_video_tensor = None

        if "vit" in return_type:
            if video_frame_col is None:
                raise ValueError("`video_frame_col` is required when `return_type` includes 'vit'.")
            vit_video_tensor, size_success = self.vit_process_video_frames(
                src[video_frame_col], video_id=video_id,
            )
            video_success = video_success and size_success
        else:
            vit_video_tensor = None

        if return_type == "vae":
            video_tensor = vae_video_tensor
        elif return_type == "vit":
            video_tensor = vit_video_tensor
        elif return_type == "vae_vit":
            video_tensor = CondVideo(video_type=return_type, vae_video=vae_video_tensor, vit_video=vit_video_tensor)
        else:
            raise ValueError(f"Unknown return type: {return_type}")

        return video_tensor, video_success

    def read_local_image(self, file_path, is_encrypted='auto', convert_mode=None, apply_exif=False):
        if is_encrypted == 'auto' and str(file_path).endswith('.enc') or is_encrypted is True:
            file_path = self.decrypt_file(file_path)
        return read_local_image(file_path, convert_mode=convert_mode, apply_exif=apply_exif)

    def get_last_frame_with_size(
            self,
            src: int,
            latent_size: tuple[int, int, int],
            **last_frame_col,
    ):
        try:
            if self.last_frame_data_format == "latents":
                latent_key = self.index_manager.get_attribute(src, **last_frame_col)
                latent = self.read_local_last_frame_latent(latent_key).squeeze(0)  # [C, 1, H, W]
                if np.isinf(latent).any():
                    raise ValueError(f"Inf detected in last-frame latents: {src=}, {latent_key=}")
                max_val = np.abs(latent).max()
                if max_val > 10:
                    raise ValueError(f"Abnormal max value {max_val} in last-frame latents: {src=}, {latent_key=}")
                if tuple(latent.shape[-2:]) != tuple(latent_size[-2:]):
                    raise ValueError(f"Last-frame latent spatial size {tuple(latent.shape[-2:])} does not "
                                     f"match video latent size {tuple(latent_size[-2:])}.")
                tensor = torch.from_numpy(latent)
            else:
                image_path = self.index_manager.get_attribute(src, **last_frame_col)
                image_path = self.parse_cos_path(image_path)
                image = self.read_local_image(image_path, convert_mode="RGB")
                tensor = self.pil_image_to_tensor(image)
                # Make sure the frame size is matched with video latent size
                _, height, width = tensor.shape
                latent_height = latent_size[-2] * self.video_vae_info.h_factor
                latent_width = latent_size[-1] * self.video_vae_info.w_factor
                if (width, height) != (latent_width, latent_height):
                    raise ValueError(f"Frame size {(width, height)} does not match "
                                     f"video latent size {(latent_width, latent_height)}.")
            success = True
        except Exception as e:
            logger.error(f"Error in get_last_frame_with_size: {e}")
            tensor = None
            success = False
        return tensor, success


class VideoProcessor(VideoMixin):
    def __init__(self, args: Namespace):
        super().__init__()
        self.modality = modality_with_cond_video_vit(args.modality, args.cond_video_type)
        self.task_kwargs = {}
        self.index_kwargs = {}
        # Condition clips are read from disk at inference time, so the COS rewrites must be live here too.
        self.setup_data(enable_crypto=args.cos_file_is_encrypted, cos_base=args.cos_base, verbose=0)
        self.setup_video(args)

    def build_gen_video_info(self, video_size, num_frames, token_grid=None) -> VideoInfo:
        # When an explicit latent token grid is provided (e.g. validation loss, where the ground-truth video
        # latent is loaded from disk), build the info directly from it — exactly like the training latent path
        # (`as_video_tensor`) — instead of snapping `num_frames`/`video_size` to VAE DAR buckets via
        # `get_target_size` (which can shift the temporal/spatial token count and break the scatter alignment).
        if token_grid is not None:
            tk_duration, tk_height, tk_width = (int(x) for x in token_grid)
            video_width = tk_width * self.video_vae_info.w_factor
            video_height = tk_height * self.video_vae_info.h_factor
            video_duration = (tk_duration - 1) * self.video_vae_info.d_factor + 1
            base_size, ratio_idx, duration_idx = \
                self.vae_dar_group.get_base_size_and_ratio_index(video_width, video_height, video_duration)
            return VideoInfo(
                video_type="gen_video",
                video_width=video_width, video_height=video_height, video_duration=video_duration,
                token_width=tk_width, token_height=tk_height, token_duration=tk_duration,
                base_size=base_size, ratio_index=ratio_idx, duration_index=duration_idx,
            )

        # parse video size (HxW, H:W, or <img_ratio_i>)
        size = video_size
        if isinstance(size, str):
            if size.startswith("<img_ratio_"):
                ratio_index = int(size.split("_")[-1].rstrip(">"))
                reso = self.vae_dar_group[ratio_index]
                size = reso.height, reso.width
            elif 'x' in size:
                size = [int(s) for s in size.split('x')]
            elif ':' in size:
                size = [int(s) for s in size.split(':')]
                assert len(size) == 2, f"`image_size` should be in the format of 'W:H', got {size}."
                # Note that ratio is width:height
                size = [size[1], size[0]]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'W:H' or <img_ratio_i>, got {size}.")
            assert len(size) == 2, f"`image_size` should be in the format of 'HxW', got {size}."
        elif isinstance(size, (list, tuple)):
            assert len(size) == 2 and all(isinstance(s, int) for s in size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {size}.")

        assert isinstance(num_frames, int), f"`num_frames` should be an integer, got {type(num_frames)}."
        video_width, video_height, video_duration = self.vae_dar_group.get_target_size(size[1], size[0], num_frames)
        token_duration = (video_duration - 1) // self.video_vae_info.d_factor + 1
        token_height = video_height // self.video_vae_info.h_factor
        token_width = video_width // self.video_vae_info.w_factor
        base_size, ratio_idx, duration_idx = self.vae_dar_group.get_base_size_and_ratio_index(size[1], size[0], num_frames)
        image_info = VideoInfo(
            video_type="gen_video", video_width=video_width, video_height=video_height, video_duration=video_duration,
            token_width=token_width, token_height=token_height, token_duration=token_duration,
            base_size=base_size, ratio_index=ratio_idx, duration_index=duration_idx,
        )
        return image_info
