import math
from typing import Any

import torch
from easydict import EasyDict

from .helpers import default


class AudioInfo:
    """ Class to store audio information for processing and generation. """
    args: EasyDict | dict

    def __init__(
            self,
            audio_type: str = None,
            audio_tensor: torch.Tensor = None,
            audio_sample_rate: int = None,
            audio_duration_by_seconds: float = None,
            audio_duration: int = None,
            token_duration: int = None,
            audio_token_length: int = None,
            audio_origin_length: int = None,
    ):
        if self.args is None:
            raise ValueError("AudioInfo requires `args` attribute to be set.")

        self.audio_type = audio_type
        self.audio_tensor = audio_tensor

        # Processed audio size
        assert audio_sample_rate is not None, "audio_sample_rate must be provided."
        if audio_duration_by_seconds is None and audio_duration is None:
            raise ValueError("Either `audio_duration_by_seconds` or `audio_duration` must be provided.")
        if audio_duration_by_seconds is None:
            # Make sure round up in seconds, so that the following int(audio_duration_by_seconds * audio_sample_rate)
            # will not be smaller than the provided audio_duration
            audio_duration_by_seconds = audio_duration / audio_sample_rate
        if audio_duration is None:
            audio_duration = round(audio_duration_by_seconds * audio_sample_rate)
        assert audio_duration == round(audio_duration_by_seconds * audio_sample_rate), \
            (f"audio_duration ({audio_duration}) does not match "
             f"audio_duration_by_seconds ({audio_duration_by_seconds}) * audio_sample_rate ({audio_sample_rate}).")
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration_by_seconds = audio_duration_by_seconds
        self.audio_duration = audio_duration
        self.d = audio_duration

        # Audio token size (after VAE encoding)
        self.token_duration = token_duration
        self.tk_d = token_duration
        self.audio_token_length = default(
            audio_token_length, token_duration if token_duration is not None else None
        )
        self.audio_origin_length = audio_origin_length

        # args
        self.add_timestep_token = self.args.get("add_timestep_token", False)
        self.gen_template = self.args.get("gen_template", "default")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in AudioInfo")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment to attributes."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Key '{key}' not found in AudioInfo")

    def __contains__(self, key: str) -> bool:
        """Check if the key exists in the AudioInfo object."""
        return hasattr(self, key)

    def __repr__(self):
        return (f"AudioInfo(audio_type={self.audio_type}, audio_tensor={self.audio_tensor}, "
                f"audio_duration={self.audio_duration}, "
                f"token_duration={self.token_duration}, "
                f"audio_token_length={self.audio_token_length})")

    @property
    def meta_info(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        # Used for audio sections of tokenizer.encode_general()
        if self.audio_type in ["vae", "gen_audio"]:
            return dict(
                token_length=self.audio_token_length,
                gen_template=self.gen_template,
                add_timestep_token=self.add_timestep_token,
            )
        else:
            raise ValueError(f"Unknown audio type '{self.audio_type}'")

    @property
    def num_special_tokens(self):
        if self.args is None:
            raise ValueError("meta_info requires `args` attribute to be set.")
        if self.audio_type in ["vae", "gen_audio"]:
            count = (
                    (2 if self.gen_template == "default" else 0) +  # <boa> + <eoa>
                    (1 if self.add_timestep_token else 0)
            )
        else:
            raise ValueError(f"Unknown audio type: {self.audio_type}")
        return count

    def copy(self, copy_audio_tensor=True):
        if copy_audio_tensor and self.audio_tensor is None:
            raise ValueError("audio_tensor is None, cannot copy")
        return AudioInfo(
            audio_type=self.audio_type,
            audio_tensor=self.audio_tensor.clone() if copy_audio_tensor else None,
            audio_duration=self.audio_duration,
            token_duration=self.token_duration,
            audio_token_length=self.audio_token_length,
        )

    def zeros_(self):
        self.audio_tensor = torch.zeros_like(self.audio_tensor)


class AudioTensor(torch.Tensor):
    # This class is just for type hinting purposes. Attribute `i` should be defined
    # as an instance attribute of the torch.Tensor instance, like: tensor.i = AudioInfo(...)
    i: AudioInfo
