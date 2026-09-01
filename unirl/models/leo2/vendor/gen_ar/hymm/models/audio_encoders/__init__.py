import torch

from ..autoencoders import VAEEncodeOutput, _add_noise  # noqa
from hymm.constants import AUDIO_ENCODER_META_INFO
from hymm.utils.torch_utils import is_torch_tensor
import json


def load_audio_encoder(
    audio_vae_type: str,
    audio_vae_latent_dim: int,
    device=None,
    only_encoder=False,
):
    if audio_vae_type == "waveflow-v1_0":
        from .waveflow.bigvgan_flow_vae_export import init_vae_stat

        vae_path = AUDIO_ENCODER_META_INFO[audio_vae_type]['path']
        vae_stats = AUDIO_ENCODER_META_INFO[audio_vae_type]['stats']

        vae_model = init_vae_stat(vae_path, vae_stats)

        vae_model.autocast_dtype = torch.float32
        assert audio_vae_latent_dim == 64, (f"audio_vae_latent_dim should be 64 but got {audio_vae_latent_dim} "
                                            f"in {audio_vae_type}")

    elif audio_vae_type.startswith("dual_channel_48k"):
        from .dual_channel_48k.models.autoencoders import create_autoencoder_from_config

        vae_path = AUDIO_ENCODER_META_INFO[audio_vae_type]['path']
        vae_config = AUDIO_ENCODER_META_INFO[audio_vae_type]['config']
        vae_mean_std = AUDIO_ENCODER_META_INFO[audio_vae_type]['mean_std']

        with open(vae_config) as f:
            model_config = json.load(f)
        vae_model = create_autoencoder_from_config(model_config)

        state_dict = torch.load(vae_path, map_location="cpu", weights_only=True)["state_dict"]
        vae_model.load_state_dict(state_dict)
        vae_model.load_latent_norm(vae_mean_std)

        # Training only encodes, so drop the decoder to save GPU memory.
        if only_encoder and getattr(vae_model, "decoder", None) is not None:
            vae_model.decoder = None

        vae_model.float()

        vae_model.autocast_dtype = torch.float32
        assert audio_vae_latent_dim == 96, (f"audio_vae_latent_dim should be 96 but got {audio_vae_latent_dim} "
                                            f"in {audio_vae_type}")

    else:
        raise ValueError(f"Unsupported audio VAE type: {audio_vae_type}")

    if device is not None:
        vae_model.to(device)

    vae_model.requires_grad_(False)
    vae_model.eval()

    return vae_model


def _vae_encode_only_tensor(vae, audio, vae_encode_type="sample", audio_data_format="single_slice_file"):
    """Encode-only part of `_vae_encode_tensor`; returns latents (no noise add).

    `audio_data_format` ending with 'latent' means `audio` is already latents.
    """
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
        has_bsz_dim = False
    else:
        assert audio.ndim == 3, "audio should have shape [B, C, L]"
        has_bsz_dim = True

    if audio_data_format.endswith('file'):
        with torch.autocast(device_type="cuda", dtype=vae.autocast_dtype, enabled=vae.autocast_dtype != torch.float32):
            # [bsz, C, L] -> [bsz, c, l]
            vae_encode_result = vae.encode(audio, generator=vae.generator)

            if isinstance(vae_encode_result, torch.Tensor):
                latents = vae_encode_result
            elif vae_encode_type == "sample":
                latents = vae_encode_result.latent_dist.sample(vae.generator)
            elif vae_encode_type == "mode":
                latents = vae_encode_result.latent_dist.mode()
            else:
                raise ValueError(f"Unknown vae_encode_type: {vae_encode_type}")

            if hasattr(vae, "config"):
                if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
                    latents.sub_(vae.config.shift_factor)
                if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor:
                    latents.mul_(vae.config.scaling_factor)
    elif audio_data_format.endswith('latent'):
        latents = audio
    else:
        raise ValueError(f"Unsupported audio_data_format: {audio_data_format}")

    if not has_bsz_dim:
        latents = latents.squeeze(0)
    return latents


def _vae_add_noise_only_tensor(vae, latents, denoiser, sample_type, noise_dtype=None, timesteps=None):
    """Noise-add part of `_vae_encode_tensor`, using `vae.noise_generator`."""
    has_bsz_dim = latents.ndim == 3
    if not has_bsz_dim:
        latents = latents.unsqueeze(0)
        if timesteps is not None:
            assert timesteps.ndim == 1 and timesteps.size(0) == 1, \
                "timesteps should have shape [1] when latents has shape [c, l]"
    elif timesteps is not None:
        assert timesteps.ndim == 1 and timesteps.size(0) == latents.size(0), \
            (f"timesteps should have shape [B] when latents has shape [B, c, l], "
             f"got {timesteps.shape=} and {latents.shape=}")

    if noise_dtype == "fp32":
        latents = latents.float()
    t, model_t, x_0, x_t, u_t = _add_noise(latents, denoiser, vae.noise_generator, sample_type, timesteps=timesteps)

    if not has_bsz_dim:
        t = t.squeeze(0)
        model_t = model_t.squeeze(0)
        x_0 = x_0.squeeze(0)
        x_t = x_t.squeeze(0)
        u_t = u_t.squeeze(0)
    return VAEEncodeOutput(t=t, model_t=model_t, x_0=x_0, x_t=x_t, u_t=u_t, latents=latents)


def _vae_encode_tensor(vae, audio, denoiser=None, sample_type=None, vae_encode_type="sample", noise_dtype=None,
                       timesteps=None, audio_data_format="single_slice_file"):
    latents = _vae_encode_only_tensor(vae, audio, vae_encode_type=vae_encode_type,
                                      audio_data_format=audio_data_format)
    if sample_type is not None:
        return _vae_add_noise_only_tensor(
            vae, latents, denoiser, sample_type, noise_dtype=noise_dtype, timesteps=timesteps,
        )
    return VAEEncodeOutput(latents=latents)


def _iter_nested_audios(items, leaf_fn, device=None, stack=None, build=None, timesteps=None):
    """Walk a nested audio/latents structure and apply `leaf_fn(tensor, ts)` per leaf."""
    aggregate = stack is not None
    call = (lambda t, ts: leaf_fn(t.to(device), ts)) if device is not None else leaf_fn

    if isinstance(items, list):
        ts_list = timesteps if timesteps is not None else [None] * len(items)
        outer = []
        for item, ts_i in zip(items, ts_list):
            if is_torch_tensor(item) and item.ndim == 2:
                item = item.unsqueeze(0)
            if isinstance(item, list):
                if ts_i is None:
                    ts_i = [None] * len(item)
                inner = [
                    call(x, ts_i[j:j+1] if is_torch_tensor(ts_i) else ts_i[j])
                    for j, x in enumerate(item)
                ]
                outer.append(stack(inner) if aggregate else inner)
            else:
                outer.append(call(item, ts_i))
        return build(outer) if aggregate else outer

    if is_torch_tensor(items):
        if device is not None:
            items = items.to(device)
        if items.ndim == 4 and items.size(1) == 1:
            items = items.squeeze(1)
        if items.ndim == 3:
            return leaf_fn(items, timesteps)
        if items.ndim == 4:
            outs = [
                leaf_fn(x, timesteps[ai] if timesteps is not None else None)
                for ai, x in enumerate(items)
            ]
            return build(outs) if aggregate else outs
        raise ValueError(f"tensor should have shape [B,C,L] or [B,n,C,L], got {items.shape}")
    raise ValueError(f"Unknown items type, expected [list, torch.Tensor], got {type(items)}")


# three cases for audios:
# 1. tensor [B, 1, C, L], return tensor [B, c, l]
# 2. tensors [B, N, C, L] (N > 1), return list of tensor B x [N, c, l]
# 3. list of lists of tensors B x (N_i x [C, L_ij]), return list of lists of tensors B x (N_i x [c, l_ij])
def audio_vae_encode_and_add_noise(vae, audios, device, denoiser=None, sample_type=None, vae_encode_type="sample",
                                   noise_dtype=None, timesteps=None, audio_data_format="single_slice_file"):
    kw = dict(denoiser=denoiser, sample_type=sample_type, vae_encode_type=vae_encode_type,
              noise_dtype=noise_dtype, audio_data_format=audio_data_format)
    return _iter_nested_audios(
        audios, lambda x, ts: _vae_encode_tensor(vae, x, **kw, timesteps=ts),
        device=device, stack=VAEEncodeOutput.stack, build=VAEEncodeOutput.build,
        timesteps=timesteps,
    )


def audio_vae_encode_pure(vae, audios, device, vae_encode_type="sample", audio_data_format="single_slice_file"):
    """Encode-only counterpart of `audio_vae_encode_and_add_noise`."""
    return _iter_nested_audios(
        audios,
        lambda x, _ts: _vae_encode_only_tensor(vae, x, vae_encode_type, audio_data_format=audio_data_format),
        device=device,
    )


def add_audio_noise(vae, nested_latents, denoiser, sample_type, noise_dtype=None, timesteps=None, device=None):
    """Noise-add counterpart of `audio_vae_encode_and_add_noise`, taking encoded latents.

    `device` moves the latents onto the compute device before noise add.
    """
    kw = dict(denoiser=denoiser, sample_type=sample_type, noise_dtype=noise_dtype)
    return _iter_nested_audios(
        nested_latents, lambda x, ts: _vae_add_noise_only_tensor(vae, x, **kw, timesteps=ts),
        device=device, stack=VAEEncodeOutput.stack, build=VAEEncodeOutput.build,
        timesteps=timesteps,
    )
