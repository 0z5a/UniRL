import os
from argparse import Namespace
from datetime import timedelta
from typing import Optional

# args
_GLOBAL_ARGS: Optional[Namespace] = None

# parallel state
_GLOBAL_PARALLEL_STATE = None

# logger
_GLOBAL_LOGGER = None

# vae
_GLOBAL_VAE = None

# audio vae
_GLOBAL_AUDIO_VAE = None

# Diffusion/Flow matching denoiser
_GLOBAL_DENOISER = None
_GLOBAL_VIDEO_DENOISER = None
_GLOBAL_AUDIO_DENOISER = None

# Tokenizer wrapper
_GLOBAL_TKWRAPPER = None

# Multimodal scalar states
_GLOBAL_SCALAR_STATE = None

# Combined batch iterator
_GLOBAL_COMBINED_ITERATOR = None

# multimodal state
_GLOBAL_MM_STATE = None

# prerun sizes
_GLOBAL_VAE_PRERUN_SIZES = None
_GLOBAL_CONV_PRERUN_SIZES = None

# text encoder
_GLOBAL_TEXT_ENCODER = None

# repa encoder
_GLOBAL_REPA_ENCODER = None

# Global Switch
_ensure_initialized: bool = True
_ensure_not_initialized: bool = True


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    if not _ensure_initialized:
        return
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    if not _ensure_not_initialized:
        return
    assert var is None, '{} is already initialized.'.format(name)


def set_ensure_initialized_flag(flag: bool):
    global _ensure_initialized
    _ensure_initialized = flag


def set_ensure_not_initialized_flag(flag: bool):
    global _ensure_not_initialized
    _ensure_not_initialized = flag


def set_args(args):
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = args


def get_args():
    global _GLOBAL_ARGS
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def reset_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None
    set_args(args)


def set_parallel_state(parallel_state):
    global _GLOBAL_PARALLEL_STATE
    _ensure_var_is_not_initialized(_GLOBAL_PARALLEL_STATE, "parallel_state")
    _GLOBAL_PARALLEL_STATE = parallel_state


def get_parallel_state():
    global _GLOBAL_PARALLEL_STATE
    _ensure_var_is_initialized(_GLOBAL_PARALLEL_STATE, "parallel_state")
    return _GLOBAL_PARALLEL_STATE

def set_logger(logger):
    global _GLOBAL_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_LOGGER, 'logger')
    _GLOBAL_LOGGER = logger


def get_logger():
    global _GLOBAL_LOGGER
    _ensure_var_is_initialized(_GLOBAL_LOGGER, 'logger')
    return _GLOBAL_LOGGER


def set_vae(vae):
    global _GLOBAL_VAE
    _ensure_var_is_not_initialized(_GLOBAL_VAE, 'vae')
    _GLOBAL_VAE = vae


def get_vae():
    global _GLOBAL_VAE
    _ensure_var_is_initialized(_GLOBAL_VAE, 'vae')
    return _GLOBAL_VAE


def set_audio_vae(audio_vae):
    global _GLOBAL_AUDIO_VAE
    _ensure_var_is_not_initialized(_GLOBAL_AUDIO_VAE, 'audio_vae')
    _GLOBAL_AUDIO_VAE = audio_vae


def get_audio_vae():
    global _GLOBAL_AUDIO_VAE
    _ensure_var_is_initialized(_GLOBAL_AUDIO_VAE, 'audio_vae')
    return _GLOBAL_AUDIO_VAE


def set_denoiser(denoiser):
    global _GLOBAL_DENOISER
    _ensure_var_is_not_initialized(_GLOBAL_DENOISER, 'denoiser')
    _GLOBAL_DENOISER = denoiser


def get_denoiser():
    global _GLOBAL_DENOISER
    _ensure_var_is_initialized(_GLOBAL_DENOISER, 'denoiser')
    return _GLOBAL_DENOISER


def set_video_denoiser(denoiser):
    global _GLOBAL_VIDEO_DENOISER
    _ensure_var_is_not_initialized(_GLOBAL_VIDEO_DENOISER, 'video_denoiser')
    _GLOBAL_VIDEO_DENOISER = denoiser


def get_video_denoiser():
    global _GLOBAL_VIDEO_DENOISER
    _ensure_var_is_initialized(_GLOBAL_VIDEO_DENOISER, 'video_denoiser')
    return _GLOBAL_VIDEO_DENOISER


def set_audio_denoiser(denoiser):
    global _GLOBAL_AUDIO_DENOISER
    _ensure_var_is_not_initialized(_GLOBAL_AUDIO_DENOISER, 'audio_denoiser')
    _GLOBAL_AUDIO_DENOISER = denoiser


def get_audio_denoiser():
    global _GLOBAL_AUDIO_DENOISER
    _ensure_var_is_initialized(_GLOBAL_AUDIO_DENOISER, 'audio_denoiser')
    return _GLOBAL_AUDIO_DENOISER


def set_tkwrapper(tkwrapper):
    global _GLOBAL_TKWRAPPER
    _ensure_var_is_not_initialized(_GLOBAL_TKWRAPPER, 'tkwrapper')
    _GLOBAL_TKWRAPPER = tkwrapper


def get_tkwrapper():
    global _GLOBAL_TKWRAPPER
    _ensure_var_is_initialized(_GLOBAL_TKWRAPPER, 'tkwrapper')
    return _GLOBAL_TKWRAPPER


def set_scalar_state(scalar_state):
    global _GLOBAL_SCALAR_STATE
    _ensure_var_is_not_initialized(_GLOBAL_SCALAR_STATE, 'scalar_state')
    _GLOBAL_SCALAR_STATE = scalar_state


def get_scalar_state():
    global _GLOBAL_SCALAR_STATE
    _ensure_var_is_initialized(_GLOBAL_SCALAR_STATE, 'scalar_state')
    return _GLOBAL_SCALAR_STATE


def set_combined_iterator(combined_iterator):
    global _GLOBAL_COMBINED_ITERATOR
    _ensure_var_is_not_initialized(_GLOBAL_COMBINED_ITERATOR, "combined_iterator")
    _GLOBAL_COMBINED_ITERATOR = combined_iterator


def get_combined_iterator():
    global _GLOBAL_COMBINED_ITERATOR
    _ensure_var_is_initialized(_GLOBAL_COMBINED_ITERATOR, "combined_iterator")
    return _GLOBAL_COMBINED_ITERATOR


def set_mm_state(mm_state):
    global _GLOBAL_MM_STATE
    _ensure_var_is_not_initialized(_GLOBAL_MM_STATE, "MM_STATE")
    _GLOBAL_MM_STATE = mm_state


def get_mm_state():
    global _GLOBAL_MM_STATE
    _ensure_var_is_initialized(_GLOBAL_MM_STATE, "MM_STATE")
    return _GLOBAL_MM_STATE


def set_prerun_sizes(vae_prerun_sizes, conv_prerun_sizes):
    global _GLOBAL_VAE_PRERUN_SIZES, _GLOBAL_CONV_PRERUN_SIZES
    _ensure_var_is_not_initialized(_GLOBAL_VAE_PRERUN_SIZES, "prerun_sizes")
    _ensure_var_is_not_initialized(_GLOBAL_CONV_PRERUN_SIZES, "prerun_sizes")
    _GLOBAL_VAE_PRERUN_SIZES = vae_prerun_sizes
    _GLOBAL_CONV_PRERUN_SIZES = conv_prerun_sizes


def get_prerun_sizes():
    global _GLOBAL_VAE_PRERUN_SIZES, _GLOBAL_CONV_PRERUN_SIZES
    _ensure_var_is_initialized(_GLOBAL_VAE_PRERUN_SIZES, "prerun_sizes")
    _ensure_var_is_initialized(_GLOBAL_CONV_PRERUN_SIZES, "prerun_sizes")
    return _GLOBAL_VAE_PRERUN_SIZES, _GLOBAL_CONV_PRERUN_SIZES


def set_text_encoder(text_encoder):
    global _GLOBAL_TEXT_ENCODER
    _ensure_var_is_not_initialized(_GLOBAL_TEXT_ENCODER, 'text_encoder')
    _GLOBAL_TEXT_ENCODER = text_encoder


def get_text_encoder():
    global _GLOBAL_TEXT_ENCODER
    _ensure_var_is_initialized(_GLOBAL_TEXT_ENCODER, 'text_encoder')
    return _GLOBAL_TEXT_ENCODER


def set_repa_encoder(repa_encoder):
    global _GLOBAL_REPA_ENCODER
    _ensure_var_is_not_initialized(_GLOBAL_REPA_ENCODER, 'repa_encoder')
    _GLOBAL_REPA_ENCODER = repa_encoder


def get_repa_encoder():
    global _GLOBAL_REPA_ENCODER
    _ensure_var_is_initialized(_GLOBAL_REPA_ENCODER, 'repa_encoder')
    return _GLOBAL_REPA_ENCODER


def get_nccl_timeout() -> timedelta:
    return timedelta(minutes=int(os.environ.get("HYMM_NCCL_TIMEOUT_MINUTES", "10")))
