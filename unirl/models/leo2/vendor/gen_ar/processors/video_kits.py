import os
import tempfile
import uuid
from pathlib import Path

import torch
import torchaudio
import imageio
import numpy as np
try:
    from moviepy import ImageSequenceClip, AudioArrayClip
except (ModuleNotFoundError, ImportError):
    ImageSequenceClip = None
    AudioArrayClip = None


def assert_saver_available(task):
    error_msg = "{package} is required for saving {task} files. Please install first."
    if task == "av":
        try:
            import moviepy
        except (ModuleNotFoundError, ImportError):
            raise RuntimeError(error_msg.format(package="moviepy", task=task))
    elif task == "video":
        try:
            import imageio
        except (ModuleNotFoundError, ImportError):
            raise RuntimeError(error_msg.format(package="imageio", task=task))
    elif task == "audio":
        try:
            import torchaudio
        except (ModuleNotFoundError, ImportError):
            raise RuntimeError(error_msg.format(package="torchaudio", task=task))


def save_video(video: np.ndarray | list[np.ndarray], save_path: str | Path, fps: int = 24):
    """ Save a video to file. """
    if isinstance(video, list):
        assert all(isinstance(video, np.ndarray) and video.ndim == 3 for video in video), \
            "When video is a list, all elements should be 3-D numpy arrays (h, w, c)."
        assert all(video.dtype == np.uint8 for video in video), \
            "When video is a list, all elements should be of dtype uint8."
        image_list = video
    elif isinstance(video, np.ndarray):
        assert video.ndim == 4, f"Video should be 4-D array (t, h, w, c), got {video.ndim}-D array."
        assert video.dtype == np.uint8, f"Video should be of dtype uint8, got {video.dtype}."
        image_list = [video[i] for i in range(video.shape[0])]
    else:
        raise ValueError(f"Video should be either a list of 3-D numpy arrays or a 4-D numpy array," 
                         f"got {type(video)}.")

    save_path = Path(save_path).absolute()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        imageio.mimsave(save_path, image_list, fps=fps)
    except ValueError as e:
        if "Could not find a backend" in str(e):
            raise RuntimeError("Failed to save video. Please ensure that imageio-ffmpeg is installed.") from e
        else:
            raise e

    return save_path


def save_audio(audio: torch.Tensor, save_path: str | Path, sample_rate: int = 48000, save_format: str = "wav"):
    """ Save an audio to file. """
    assert audio.ndim == 2, f"Audio should be 2-D array (channels, samples), got {audio.ndim}-D array."
    assert audio.dtype in [torch.float32], f"Audio should be of dtype float32 or int16, got {audio.dtype}."

    save_path = Path(save_path).absolute()
    save_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        torchaudio.save(save_path, audio, sample_rate, format=save_format)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio: {e}") from e

    return save_path


def save_video_audio(
        video: np.ndarray | list[np.ndarray],
        audio: np.ndarray,
        save_path: str | Path,
        fps: int = 24,
        sample_rate: int = 48000,
):
    """
    Save a video with audio to file. This function uses MoviePy to combine video frames and audio into a single file.

    Args:
        video: A 4-D numpy array (t, h, w, c) or a list of 3-D numpy arrays (h, w, c) representing video frames.
        audio: A 2-D numpy array (channels, samples) representing audio data.
        save_path: The path to save the output video file.
        fps: Frames per second for the video.
        sample_rate: Sample rate for the audio.
    """
    if audio is None:
        return save_video(video, save_path, fps)

    # 1. 确保视频数据类型正确 (uint8)
    if video.dtype != np.uint8:
        video = (video * 255).astype(np.uint8)

    # 2. 创建视频轨道
    # ImageSequenceClip 接受 numpy 数组列表
    clip = ImageSequenceClip(list(video), fps=fps)

    # 3. 创建音频轨道
    # 注意：MoviePy 期望 (L, 2)，所以必须转置 .T
    audio_track = AudioArrayClip(audio.T, fps=sample_rate)

    # 4. 合成
    final_clip = clip.with_audio(audio_track)  # 新版建议用 with_audio

    # 5. 导出
    save_path = Path(save_path).absolute()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    # MoviePy 默认会用输出文件的 basename 生成临时音频文件（相对路径，落在共享的当前工作目录里）。
    # 多卡/重复样本会写出同名临时文件而互相踩踏，导致 BrokenPipe / os.remove 时 FileNotFoundError。
    # 这里显式指定一个进程唯一、且落在本地磁盘的临时音频路径来规避。
    temp_audiofile = os.path.join(
        tempfile.gettempdir(),
        f"{save_path.stem}_{os.getpid()}_{uuid.uuid4().hex}_TEMP.m4a",
    )
    final_clip.write_videofile(
        str(save_path),
        codec="libx264",
        audio_codec="aac",
        audio_fps=sample_rate,
        temp_audiofile=temp_audiofile,
    )

    return save_path
