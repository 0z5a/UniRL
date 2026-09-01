from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple

import torch
import torchaudio
from index_kits import ArrowIndexV2, MultiIndexV2
from torchaudio.transforms import Resample
import soundfile as sf
import numpy as np

from .data_utils import DataMixin
from ...constants import AUDIO_ENCODER_META_INFO
from ...utils.audio_base import AudioInfo, AudioTensor


@dataclass
class AudioVAEInfo:
    encoder_type: str
    latent_dim: int = None
    audio_type: str = None
    downsampling_ratio: int = None

    def __post_init__(self):
        self.encoder_type = self.encoder_type
        self.downsampling_ratio = self.downsampling_ratio
        if self.audio_type is None:
            self.audio_type = "vae"

    def calc_token_duration(self, duration):
        # For WaveFlow VAE. Simulate the conv ops.
        if self.encoder_type == "waveflow-v1_0":
            strides = [2, 2, 2, 3, 4, 5]
            kernels = [s * 2 for s in strides]
            paddings = [(k - 1) // 2 for k in kernels]
            for s, k, p in zip(strides, kernels, paddings):
                duration = (duration + 2 * p - k) // s + 1
            return duration
        elif self.encoder_type.startswith("dual_channel_48k"):
            return (duration + self.downsampling_ratio - 1) // self.downsampling_ratio
        else:
            raise ValueError(f"Not support encoder type {self.encoder_type}")


class AudioMixin(DataMixin):
    """
    A mixin class for audio processing and generation.

    Notice: Make sure all the method and attribute names contain 'audio' to avoid name conflicts with other mixins,
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

    def setup_audio(self, args):
        AudioInfo.args = dict(
            gen_template=args.gen_template,
            add_timestep_token=args.add_timestep_token,
        )

        # -- vae audio --
        if "vae_audio" in self.modality:
            self.require_configs(args, [
                "vae_type", "vae_audio_token_length", "audio_cos_base",
            ], "vae_audio modality")

            # -- conditional audio --
            self.cond_audio_type = getattr(args, "cond_audio_type", "none")
            self.cond_audio_token_attn_type = getattr(args, "cond_audio_token_attn_type", "none")
            # cond-audio-token-attn-type and cond-audio-type should be matched
            # vae: causal, full
            if self.cond_audio_type in ["vae"]:
                assert self.cond_audio_token_attn_type in ["causal", "full"], \
                    f"When cond-audio-type is 'vae', cond-audio-token-attn-type should be 'causal' or 'full', " \
                    f"but got {self.cond_audio_token_attn_type}."
            self.cond_audio_section_type = dict(
                none="none",
                vae="cond_vae_audio",
            )[self.cond_audio_type]

            self.audio_vae_type = args.audio_vae_type
            self.audio_cos_base = Path(args.audio_cos_base)
            self.vae_audio_token_length = self.task_kwargs.get("vae_audio_token_length", args.vae_audio_token_length)
            # vae info
            vae_meta_info = AUDIO_ENCODER_META_INFO[args.audio_vae_type]
            self.audio_vae_info = AudioVAEInfo(
                encoder_type=args.audio_vae_type,
                latent_dim=args.audio_vae_latent_dim,
                audio_type="vae",
                downsampling_ratio=vae_meta_info.get("downsampling_ratio", None),
            )
            self.audio_channels = vae_meta_info["channels"]
            self.audio_sample_rate = vae_meta_info["sample_rate"]
            self.audio_downsampling_ratio = vae_meta_info.get("downsampling_ratio", None)
            assert self.audio_channels in [1, 2], f"audio_channels should be 1 or 2, but got {self.audio_channels}."

        # -- unconditions --
        self.audio_uncond_p = self.task_kwargs.get('uncond_p', 0.0)

    def read_local_audio(self, latent_key: str, data_format: str = "single_slice_file",
                         start: float = None, end: float = None) -> Tuple[torch.Tensor, int]:
        if latent_key.startswith("/"): # absolute cos path
            latent_path = self.parse_cos_path(latent_key)
        else: # relative cos path
            latent_path = self.audio_cos_base / latent_key.lstrip("/")

        if data_format == "single_slice_file":
            waveform, sample_rate = torchaudio.load(latent_path)
        elif data_format == "multi_slice_file":
            assert start is not None and end is not None, f"start: {start} and end: {end} should not be None."
            with sf.SoundFile(latent_path, 'r') as sf_file:
                sample_rate_full = sf_file.samplerate

            start_frame = int(start * sample_rate_full)
            stop_frame = int(end * sample_rate_full)

            waveform, sample_rate = sf.read(latent_path, start=start_frame, stop=stop_frame)
            waveform = torch.from_numpy(waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.transpose(0, 1)
            assert sample_rate == sample_rate_full, (f"sample_rate: {sample_rate} != "
                                                     f"sample_rate_full: {sample_rate_full}")
        else:
            raise ValueError(f"Not support data_format {data_format}")

        return waveform, sample_rate

    def get_raw_audio(self, src: int, real_index: int = None, data_format: str = "single_slice_file",
                      start: float = None, end: float = None, **audio_col):
        _ = real_index
        audio_path = self.index_manager.get_attribute(src, **audio_col)
        waveform, sample_rate = self.read_local_audio(audio_path, data_format=data_format, start=start, end=end)
        audio_flag = "normal"
        return waveform, sample_rate, audio_flag

    def read_local_audio_latent(self, latent_key: str) -> torch.Tensor:
        if not latent_key.endswith(".npy"):
            latent_key += ".npy"
        if latent_key.startswith("/"): # absolute cos path
            latent_path = self.parse_cos_path(latent_key)
        else: # relative cos path
            latent_path = self.audio_cos_base / latent_key.lstrip("/")
        # latent_path = latent_key
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).squeeze(0)

        # Sanity check
        assert (
            latent.dim() == 2
            and latent.size(0) == self.audio_vae_info.latent_dim
        ), f"Audio latent should have 2 dimensions (C, D), but got {tuple(latent.shape)}."
        return latent

    def get_raw_latent(self, src: int, real_index: int = None, **audio_col):
        _ = real_index
        if isinstance(src, dict):
            # Interleave `message` item: the latent path lives inline in the message dict.
            audio_latent_path = src[audio_col['column']]
        else:
            audio_latent_path = self.index_manager.get_attribute(src, **audio_col)
        audio_latent = self.read_local_audio_latent(audio_latent_path)
        audio_flag = "normal"
        return audio_latent, audio_flag

    def as_audio_tensor(self, audio_tensor, audio_type, audio_origin_length=None,
                        data_format="single_slice_file") -> AudioTensor:
        if audio_type == "vae":
            if data_format.endswith("file"):
                duration = audio_tensor.shape[1]
                tk_duration = self.audio_vae_info.calc_token_duration(duration)
                audio_tensor.i = AudioInfo(
                    audio_type=audio_type,
                    audio_sample_rate=self.audio_sample_rate,
                    audio_duration=duration,
                    token_duration=tk_duration,
                    audio_origin_length=audio_origin_length,
                )
            elif data_format.endswith("latent"):
                downsampling_ratio = self.audio_vae_info.downsampling_ratio
                assert downsampling_ratio is not None, (
                    f"{data_format} mode requires audio_vae_info.downsampling_ratio to be set, "
                    f"got None for encoder_type={self.audio_vae_info.encoder_type}."
                )
                tk_duration = audio_tensor.shape[1]
                duration = tk_duration * downsampling_ratio
                audio_tensor.i = AudioInfo(
                    audio_type=audio_type,
                    audio_sample_rate=self.audio_sample_rate,
                    audio_duration=duration,
                    token_duration=tk_duration,
                    audio_origin_length=None,
                )
        else:
            raise ValueError(f"Unknown audio type: {audio_type}")

        return audio_tensor     # noqa

    def vae_process_audio(self, waveform: torch.Tensor, sample_rate: int, data_format: str) -> AudioTensor:

        audio_origin_length = None
        if data_format.endswith("file"):
            if self.audio_vae_type == "waveflow-v1_0":
                if self.audio_channels == 1:
                    # Stereo to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                else:
                    # Mono to stereo
                    if waveform.shape[0] == 1:
                        waveform = waveform.repeat(2, 1)

                if sample_rate != self.audio_sample_rate:
                    resampler = Resample(orig_freq=sample_rate, new_freq=self.audio_sample_rate)
                    waveform = resampler(waveform)
            elif self.audio_vae_type.startswith("dual_channel_48k"):
                if sample_rate != self.audio_sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, self.audio_sample_rate)

                audio_origin_length = waveform.shape[-1]

                if waveform.shape[-1] % self.audio_downsampling_ratio != 0:
                    pad_len = self.audio_downsampling_ratio - (waveform.shape[-1] % self.audio_downsampling_ratio)
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))

                if waveform.shape[0] == 1:
                    # Stereo to mono
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    # Mono to stereo
                    waveform = waveform[:2, :]

        return self.as_audio_tensor(waveform, audio_type=self.audio_vae_info.audio_type,
                                    audio_origin_length=audio_origin_length, data_format=data_format)

    def get_audio_with_size(
            self,
            src: int | dict,
            return_type: str = "vae",
            real_index: int = None,
            data_format: str = "single_slice_file",
            start: float = None,
            end: float = None,
            **audio_col,
    ) -> Tuple[AudioTensor, bool]:
        try:
            if data_format.endswith("file"):
                waveform, sr, audio_flag = self.get_raw_audio(src, real_index,
                                                              data_format=data_format,
                                                              start=start,
                                                              end=end,
                                                              **audio_col)
            elif data_format.endswith("latent"):
                sr = None
                waveform, audio_flag = self.get_raw_latent(src, real_index, **audio_col)
                # Check inf for audio latents(tensor)
                if torch.isinf(waveform).any():
                    raise ValueError(f"Inf detected in audio latents: {src=}, {real_index=}")
            else:
                raise ValueError(f"Not support data_format {data_format}")
        except Exception as e:
            audio_path = src[audio_col['column']] if isinstance(src, dict) else self.index_manager.get_attribute(src, **audio_col)
            if data_format.endswith("file"):
                print(f"Error in get_raw_audio: {e}, audio_path={audio_path}")
            elif data_format.endswith("latent"):
                print(f"Error in get_raw_latent: {e}, audio_path={audio_path}")
            else:
                raise ValueError(f"Not support data_format {data_format}")
            audio_flag = "error"
            waveform = torch.zeros(1, 16000)
            sr = 16000

        audio_success = audio_flag != "error"
        vae_audio_tensor = self.vae_process_audio(waveform, sr, data_format=data_format)

        if return_type == "vae":
            audio_tensor = vae_audio_tensor
        else:
            raise ValueError(f"Unknown return type: {return_type}")

        return audio_tensor, audio_success

    def prepare_audio_full_attn_slices(self, output, batch_idx=None, with_gen=True):
        """ Determine full attention audio slices according to strategies. """
        if not hasattr(self, "cond_audio_type"):
            return []

        if self.cond_audio_type == "none":
            cond_choices = dict(
                causal=[],
                full=[]
            )

        elif self.cond_audio_type == "vae":
            cond_choices = dict(
                causal=[],
                full=output.vae_audio_slices[batch_idx] if batch_idx is not None else output.vae_audio_slices
            )

        else:
            raise ValueError(f"Unknown cond_audio_type: {self.cond_audio_type}")

        if self.cond_audio_token_attn_type == "none":
            slices = []
        else:
            slices = cond_choices[self.cond_audio_token_attn_type]

        if with_gen:
            gen_audio_slices = (
                output.gen_audio_slices[batch_idx]
                if batch_idx is not None
                else output.gen_audio_slices
            )
            slices = slices + gen_audio_slices
        return slices


class AudioProcessor(AudioMixin):
    def __init__(self, args: Namespace):
        super().__init__()
        self.modality = args.modality
        self.task_kwargs = {}
        self.index_kwargs = {}
        self.setup_audio(args)

    def build_gen_audio_info(self, audio_duration_by_seconds, token_duration=None) -> AudioInfo:
        assert isinstance(audio_duration_by_seconds, (int, float)), \
            f"`audio_duration_by_seconds` should be an integer or float, got {type(audio_duration_by_seconds)}."
        audio_duration = round(audio_duration_by_seconds * self.audio_sample_rate)
        # When `token_duration` is explicitly provided (e.g. validation loss, where the ground-truth audio latent is
        # loaded from disk), use it directly so the number of audio tokens in the sequence exactly matches the loaded
        # latent length. Otherwise derive it from the audio duration as usual.
        if token_duration is None:
            token_duration = self.audio_vae_info.calc_token_duration(audio_duration)

        audio_info = AudioInfo(
            audio_type="gen_audio",
            audio_sample_rate=self.audio_sample_rate,
            audio_duration_by_seconds=audio_duration_by_seconds,
            audio_duration=audio_duration,
            token_duration=token_duration,
        )
        return audio_info

    def postprocess_audio(self, audio_tensor: torch.Tensor, output_type="np") -> torch.Tensor:
        assert audio_tensor.ndim == 3, \
            f"Expected audio tensor to have 3 dimensions (B, C, L), but got {audio_tensor.ndim}."
        # Ensure the audio tensor has the correct number of channels
        if audio_tensor.size(1) == 1:
            audio_tensor = audio_tensor.repeat(1, 2, 1)
        assert audio_tensor.size(1) == 2, \
            f"Expected audio tensor to have 2 channels, but got {audio_tensor.size(1)}."
        audio = audio_tensor.cpu()

        if output_type == "np":
            audio = audio.numpy()

        return audio
