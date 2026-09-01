# =============================================================================
# This file is part of HunyuanMultimodal, a collection of models for AIGC.
#
# This file contains the code to load the various VQ-VAE, VQ-GAN, and other
# autoencoder models used in the project.
#
# Registered models are required to have the following attributes and methods:
#   Attributes:
#     - self.codebook_size
#     - self.downsample_factor
#   Methods:
#     - self.from_pretrained() / self.from_config()
#     - self.vq_encode()
#     - self.vq_decode()
#
# =============================================================================

import importlib
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL
from diffusers.utils import BaseOutput
from transformers import AutoConfig

from ...constants import VAE_META_INFO, SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES
from ...utils.torch_utils import PRECISION_TO_TYPE, is_torch_tensor


class AutoEncoderRegistry(object):
    def __init__(self):
        self._registry = {
            "AutoencoderKL":            ("diffusers", "AutoencoderKL"),      # VAE in SDXL =

            "EVACLIP":                  ("emu2.modeling_emu", "EmuForCausalLM"),      # EVA-CLIP as image encoder, finetune SDXL as decoder in EMU2
            "EVACLIP_256x16x16_SDXL":   ("emu2.vit_enc_dif_dec", "EVACLIP_256x16x16_SDXL"),      # EVA-CLIP as image encoder, finetune SDXL as decoder in EMU2
            "MoVQGAN":                  ("emu3.modeling_emu3visionvq", "Emu3VisionVQModel"),   # Emu3 MoVQGAN
            "AutoencoderKL2DVQ":        ("hy.autoencoder_kl_2d_vq", "AutoencoderKL2DVQ"), # HY VQ VAE before Nov. 2024
            "HYVAE3D":                  ("hy.autoencoder_kl_3d", "AutoencoderKLConv3D"),  # 16x16x4 32c VAE
            "HYVAE3D_RMSNorm":          ("hy.autoencoder_kl_causal_rmsnorm_3d", "AutoencoderKLConv3D"),  # 16x16x4 32c VAE
            "HYVAE3D_RMSNorm_v3":       ("hy.autoencoder_kl_causal_rmsnorm_3d_v3", "WanVAE_"),  # 16x16x4 48c VAE
            "HYVAE3D_RMSNorm_v3_3":     ("hy.autoencoder_kl_causal_rmsnorm_3d_v3_3", "AutoencoderKLConv3D"),  # 16x16x4 48c VAE
            "HYVAE2D":                  ("hy.hy_vae_2d", "HYVAE2D"),   # 32x32 64c VAE
            "HunyuanVQVAE2D":           ("hy.vqvae_2d", "HunyuanVQVAE"), # HY VQ VAE after Dec. 2024
            "JanusVQ":                  ("janus.janus_vq", "VQ_16"),  # 16x16, LlamaGen vq-tokenizer, used in Janus
            "MOVQ":                     ("movqgan.vqgan", "MOVQ"),                   # Vanilla MoVQGAN
            "VQModel":                  ("sd_vq.sd_vq", "VQModel"),
            "MaskGITVQModel":           ("sd_vq.sd_vq", "VQModel"),
            "MAGViTv2":                 ("showo.modeling_magvitv2", "MAGVITv2"),
            "AutoencoderKLFlux2":       ("flux.flux2", "AutoEncoder"),
            "HYVAE3_5":                 ("hy3_5.autoencoder_kl", "AutoencoderKL"),  # 16x16-64c VAE
        }

    def __getitem__(self, key):
        if key not in self._registry:
            raise NotImplementedError(f"Model {key} not found in registry.")

        # Get submodules in hymm.models.autoencoders
        local_submodules = set([x.stem for x in Path(__file__).parent.glob("*") if x.stem != "__init__"])

        module, cls_name = self._registry[key]
        if module.split(".")[0] in local_submodules:
            module_spec = importlib.import_module(f"hymm.models.autoencoders.{module}")
        else:
            module_spec = importlib.import_module(module)
        cls = getattr(module_spec, cls_name)
        return cls


_registry = AutoEncoderRegistry()

def _as_latent_stat_tensor(value: Any, *, device=None, dtype=None):
    if value is None:
        return None
    if is_torch_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)

def _reshape_latent_stat(stat, latents):
    # latents should be [B, C, ...]
    stat = _as_latent_stat_tensor(stat, device=latents.device, dtype=latents.dtype)
    if stat is None or stat.numel() == 1:
        return stat
    if stat.ndim == 1:
        assert stat.numel() == latents.shape[1], (
            f"Latent norm stat has {stat.numel()} channels, expected {latents.shape[1]}"
        )
        shape = [1] * latents.ndim
        shape[1] = stat.numel()
        return stat.reshape(shape)
    return stat.to(device=latents.device, dtype=latents.dtype)

def _load_latent_norm_stats(stats_path):
    if stats_path in (None, "", "null", "None"):
        return None
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return None
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    if not isinstance(stats, dict):
        raise ValueError(f"VAE latent norm stats must be a dict, got {type(stats)} from {stats_path}")

    def first_present(*keys):
        for key in keys:
            if key in stats and stats[key] is not None:
                return stats[key]
        return None

    scale = first_present("latents_scale", "latent_scale", "scale")
    bias = first_present("latents_bias", "latent_bias", "bias", "mean", "latent_mean", "latents_mean")
    std = first_present("std", "latent_std", "latents_std")
    if scale is None and std is not None:
        std = _as_latent_stat_tensor(std).float()
        if torch.any(std == 0):
            raise ValueError(f"VAE latent norm std at {stats_path} must not contain zero values")
        scale = 1.0 / std
    if scale is None:
        raise ValueError(f"VAE latent norm stats at {stats_path} must contain latents_scale/scale or std")
    return {"scale": scale, "bias": 0.0 if bias is None else bias}

def configure_vae_latent_normalization(vae, args=None, logger=None, vae_path=None):
    stats_path = None if vae_path is None else Path(vae_path) / "latent_norm_stats.pt"
    stats = _load_latent_norm_stats(stats_path)

    vae.latent_norm_enabled = stats is not None
    vae.latent_norm_scale = None if stats is None else stats["scale"]
    vae.latent_norm_bias = None if stats is None else stats["bias"]
    if vae.latent_norm_enabled and logger is not None:
        scale = vae.latent_norm_scale
        bias = vae.latent_norm_bias
        logger.info(
            f"VAE latent normalization enabled: stats_path={stats_path}, "
            f"scale={scale if not is_torch_tensor(scale) else tuple(scale.shape)}, "
            f"bias={bias if not is_torch_tensor(bias) else tuple(bias.shape)}"
        )

def normalize_vae_latents(vae, latents):
    # if has latent stats, skip scaling factor in config file.
    if getattr(vae, "latent_norm_enabled", False):
        bias = _reshape_latent_stat(getattr(vae, "latent_norm_bias", 0.0), latents)
        scale = _reshape_latent_stat(getattr(vae, "latent_norm_scale", 1.0), latents)
        return (latents - bias) * scale

    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
        latents.sub_(vae.config.shift_factor)
    if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor:
        latents.mul_(vae.config.scaling_factor)
    return latents

def denormalize_vae_latents(vae, latents):
    # if has latent stats, skip scaling factor in config file.
    if getattr(vae, "latent_norm_enabled", False):
        bias = _reshape_latent_stat(getattr(vae, "latent_norm_bias", 0.0), latents)
        scale = _reshape_latent_stat(getattr(vae, "latent_norm_scale", 1.0), latents)
        return latents / scale + bias

    if hasattr(vae.config, 'scaling_factor') and vae.config.scaling_factor:
        latents = latents / vae.config.scaling_factor
    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
        latents = latents + vae.config.shift_factor
    return latents

def load_vae(
    vae_type,
    vae_precision=None,
    device=None,
    logger=None,
    args=None,
    weights_only=False,
    only_encoder=False,
    only_decoder=False,
    sample_size=None,
):
    if logger is None:
        from loguru import logger

    vae_meta_info = VAE_META_INFO[vae_type]
    vae_path = Path(vae_meta_info["path"])

    # Read the config file
    config_file = vae_path / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # Build the encoder structure
    if "_class_name" in config:
        classname = config.pop("_class_name")  
    else:
        raise ValueError(f"Cannot find the _class_name or _name_or_path in {config_file}")
    logger.info(f"Load VAE with class {classname} from {config_file}")

    logger.info(f"Load vae_type: {vae_type} from path: {vae_path}")
    if "show-o" in vae_type or "emu3" in vae_type:
        vae = _registry[classname].from_pretrained(vae_path)

    elif "evaclip-emu2" in vae_type:
        logger.info(f"Load evaclip config from {vae_path}")
        config = AutoConfig.from_pretrained(
            f"{vae_path}",
            trust_remote_code=True
        )
        config.pooling_stride = args.model_kwargs.get("pooling_stride", None)
        logger.info(f"Start Init evaclip model from config, set pooling_stride to {config.pooling_stride}")
        vae = _registry[classname](config)
        ckpt_path = vae_path / "evaclip.pt"
        logger.info(f"Load evaclip checkpoint from {ckpt_path}")
        vae.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        logger.info(f"Load evaclip checkpoint from {ckpt_path} done")
    elif "evaclip-sdxl" in vae_type:
        vae = _registry[classname].from_config(
            config_file, 
            args=args,
            dtype=PRECISION_TO_TYPE[vae_precision],
            device=device,
            logger=logger,
            _trans_type=vae_meta_info["trans_type"],
            return_dict=vae_meta_info["return_dict"],
        )
    elif "emu2" in vae_type or "vae-sdxl" in vae_type:
        vae = _registry[classname].from_pretrained(
            vae_path,
            trust_remote_code=True,
            variant='bf16' if "sdxl" in vae_type else None,
            use_safetensors=True,
        )
    elif "flux" in vae_type:
        vae = _registry[classname].from_pretrained(vae_path)
    elif "16x16x4-48c-hy-v3" == vae_type:
        AutoencoderKLConv3D = _registry[vae_meta_info["class_type"]]
        config = AutoencoderKLConv3D.load_config(vae_path)
        vae = AutoencoderKLConv3D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.pt", map_location="cpu", weights_only=weights_only)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('vae.'):
                vae_ckpt[k.replace("vae.", "")] = v
        vae.load_state_dict(vae_ckpt)
        if only_encoder:
            # encode() only uses encoder + conv1; drop decode-only modules to save GPU memory.
            vae.decoder = None
            vae.conv2 = None
        logger.info(f"Load checkpoint from {Path(vae_path)}")
    elif "16x16x4-48c-hy-v3_3" in vae_type:
        AutoencoderKLConv3D = _registry[vae_meta_info["class_type"]]
        config = AutoencoderKLConv3D.load_config(vae_path)
        vae = AutoencoderKLConv3D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.pt", map_location="cpu", weights_only=weights_only)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('vae.'):
                vae_ckpt[k.replace("vae.", "")] = v
        vae.load_state_dict(vae_ckpt)
        logger.info(f"Load checkpoint from {Path(vae_path)}")
    elif "16x16x4-32c-hy" in vae_type:
        AutoencoderKLConv3D = _registry[vae_meta_info["class_type"]]
        config = AutoencoderKLConv3D.load_config(vae_path)
        config.update({
            "only_encoder": only_encoder,
            "only_decoder": only_decoder,
            **({'sample_size': sample_size} if sample_size is not None else {}),
        })
        vae = AutoencoderKLConv3D.from_config(config)
        if not only_encoder and not only_decoder:
            ckpt = torch.load(Path(vae_path) / "pytorch_model.ckpt", map_location="cpu", weights_only=weights_only)
        elif only_encoder:
            ckpt = torch.load(Path(vae_path) / "encoder.pt", map_location="cpu", weights_only=weights_only)
        elif only_decoder:
            ckpt = torch.load(Path(vae_path) / "decoder.pt", map_location="cpu", weights_only=weights_only)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('vae.'):
                vae_ckpt[k.replace("vae.", "")] = v
        vae.load_state_dict(vae_ckpt)
        if only_encoder:
            # encode() only uses encoder + conv1; drop decode-only modules to save GPU memory.
            vae.decoder = None
            vae.conv2 = None
        logger.info(f"Load checkpoint from {Path(vae_path)}")
    elif "32x32-64c-hy" in vae_type:
        HYVAE2D = _registry["HYVAE2D"]
        config = HYVAE2D.load_config(vae_path)
        vae = HYVAE2D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.ckpt", map_location="cpu", weights_only=weights_only)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('vae.'):
                vae_ckpt[k.replace("vae.", "")] = v
        vae.load_state_dict(vae_ckpt)
        logger.info(f"Load checkpoint from {Path(vae_path)}")
    elif "16x16-64c-hy3.5" in vae_type:
        cls = _registry[classname]
        vae = cls()
        vae_ckpt = torch.load(Path(vae_path) / "pytorch_model.ckpt", map_location="cpu", weights_only=weights_only)
        vae.load_state_dict(vae_ckpt)
        logger.info(f"Load checkpoint from {Path(vae_path)}")
    elif "1616-vq-janus" in vae_type:
        vae = _registry[classname]()
        ckpt_path = vae_path / "pytorch_model.pt"
        logger.info(f"Load vq-janus checkpoint from {ckpt_path}")
        vae.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        logger.info(f"Load vq-janus checkpoint from {ckpt_path} done")
    else:
        cls = _registry[classname]
        if hasattr(cls, "init_from_ckpt"):
            vae = cls.from_config(config_file)
            # Load the checkpoint
            ckpt_path = vae_path / "pytorch_model.pt"
            vae.init_from_ckpt(ckpt_path)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path)
            ckpt_path = vae_path

        logger.info(f"Load checkpoint from {ckpt_path}")

    if "codebook_size" in vae_meta_info.items():
        vae._codebook_size = vae_meta_info["codebook_size"]
    vae._downsample_factor = vae_meta_info["downsample_factor"]
    if not hasattr(vae, 'downsample_factor'):
        vae.downsample_factor = vae_meta_info["downsample_factor"][0]
    vae._trans_type = vae_meta_info["trans_type"]

    if args is not None:
        vae.autocast_dtype = PRECISION_TO_TYPE[args.vae_autocast_dtype]

    if vae_precision is not None:
        logger.warning(f"You are transforming VAE to {vae_precision} precision! Please make sure this is what you want.")
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    # Set to eval mode
    if device is not None:
        vae = vae.to(device=device)
    vae.requires_grad_(False)
    vae.eval()
    configure_vae_latent_normalization(vae, args=args, logger=logger, vae_path=vae_path)

    return vae


class VAEEncodeOutput(BaseOutput):
    t: torch.FloatTensor = None
    model_t: torch.FloatTensor = None
    x_0: torch.FloatTensor = None
    x_t: torch.FloatTensor = None
    u_t: torch.FloatTensor = None

    # Temporary latents only for reuse.
    latents: torch.FloatTensor = None

    @staticmethod
    def cat(output_list):
        assert isinstance(output_list, list) and len(output_list) > 0

        def try_cat(lst):
            if lst[0] is None:
                assert all(x is None for x in lst), "Mixed None/non-None fields are not supported in cat."
                return None
            if not is_torch_tensor(lst[0]):
                return lst
            if all(is_torch_tensor(x) and x.shape[1:] == lst[0].shape[1:] for x in lst):
                return torch.cat(lst)
            return lst

        return VAEEncodeOutput(
            t=try_cat([o.t for o in output_list]),
            model_t=try_cat([o.model_t for o in output_list]),
            x_0=try_cat([o.x_0 for o in output_list]),
            x_t=try_cat([o.x_t for o in output_list]),
            u_t=try_cat([o.u_t for o in output_list]),
            latents=try_cat([o.latents for o in output_list]),
        )

    @staticmethod
    def stack(output_list):
        assert isinstance(output_list, list) and len(output_list) > 0

        def try_stack(lst):
            if lst[0] is None:
                assert all(x is None for x in lst), "Mixed None/non-None fields are not supported in stack."
                return None
            if not is_torch_tensor(lst[0]):
                return lst
            if all(is_torch_tensor(x) and x.shape[1:] == lst[0].shape[1:] for x in lst):
                return torch.stack(lst)
            return lst

        return VAEEncodeOutput(
            t=try_stack([o.t for o in output_list]),
            model_t=try_stack([o.model_t for o in output_list]),
            x_0=try_stack([o.x_0 for o in output_list]),
            x_t=try_stack([o.x_t for o in output_list]),
            u_t=try_stack([o.u_t for o in output_list]),
            latents=try_stack([o.latents for o in output_list]),
        )

    @staticmethod
    def build(output_list):
        assert isinstance(output_list, list) and len(output_list) > 0

        return VAEEncodeOutput(
            t=[o.t for o in output_list],
            model_t=[o.model_t for o in output_list],
            x_0=[o.x_0 for o in output_list],
            x_t=[o.x_t for o in output_list],
            u_t=[o.u_t for o in output_list],
            latents=[o.latents for o in output_list],
        )


def _add_noise(latents, denoiser, generator, sample_type=None, return_dict=False, timesteps=None):
    if sample_type == "sample":
        n_tokens = latents.shape[-2] * latents.shape[-1]
        t, x_0, x_1 = denoiser.sample(latents, n_tokens, generator=generator)
    elif sample_type == "sample_start":
        t, x_0, x_1 = denoiser.sample_start(latents, generator=generator)
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    # If timesteps is given, override the sampled t with the given timesteps.
    # This is used for blended video and audio noise timestep sampling.
    if timesteps is not None:
        assert t.shape == timesteps.shape, f"timesteps shape {timesteps.shape} does not match sampled t shape {t.shape}"
        t = timesteps.to(t.device)
    t, x_t, u_t = denoiser.path_sampler.plan(t, x_0, x_1)

    model_t = denoiser.get_model_t(t)  # t*1000
    if return_dict:
        return VAEEncodeOutput(t=t, model_t=model_t, x_0=x_0, x_t=x_t, u_t=u_t, latents=latents)
    return t, model_t, x_0, x_t, u_t


def add_noise_and_extend_channel(
        latents,
        last_frame_latents,
        has_bsz_dim,
        *args,
        noise_dtype=None,
        latent_channel_extend_type=None,
        **kwargs
):
    if noise_dtype is not None:
        latents = latents.float()    # Align with HunyuanVideo_pureTorch
    out = _add_noise(latents, *args, **kwargs)

    assert out.x_t.ndim == 5, f"latents should have shape [B, c, t, h, w], got {latents.shape}"

    # Channel extension for channel-mode i2v and fl2v tasks. The type must be explicitly declared the
    # `latent_channel_extend_type` field in the task's YAML kwargs.
    if latent_channel_extend_type is not None:
        assert latent_channel_extend_type in SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES, (
            f"Unsupported latent_channel_extend_type: {latent_channel_extend_type!r}, "
            f"expected one of {list(SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES)}."
        )
        extended_latents = torch.zeros_like(latents, dtype=latents.dtype, device=latents.device)
        mask = torch.zeros(latents.size(2), dtype=latents.dtype, device=latents.device)
        if latent_channel_extend_type == "i2v":
            extended_latents[:, :, 0:1] = latents[:, :, 0:1]
            mask[0] = 1.0
        elif latent_channel_extend_type == "fl2v":
            assert last_frame_latents is not None, (
                "`last_frame_latents` is required when `latent_channel_extend_type='fl2v'`, but got None."
            )
            extended_latents[:, :, 0:1] = latents[:, :, 0:1]
            extended_latents[:, :, -1:] = last_frame_latents
            mask[0] = 1.0
            mask[-1] = 1.0
        elif latent_channel_extend_type in {"t2i", "t2v", "t2va", "r2v", "r2va"}:
            # These tasks use the 2C+1 projector shape but have no channel-concat
            # frame condition on the generated latent.
            pass
        else:
            raise ValueError(f"Unsupported latent_channel_extend_type: {latent_channel_extend_type}")

        bsz, c, _, h, w = out.x_t.shape
        extended_mask = mask.view(1, 1, -1, 1, 1).expand(bsz, -1, -1, h, w)     # (t,) -> (b, 1, t, h, w)
        out.x_t = torch.cat([out.x_t, extended_latents, extended_mask], dim=1)  # (b, 2c+1, t, h, w)

    # Drop the batch dimension if the most outer inputs don't have batch dimension, for nested list input case.
    if not has_bsz_dim:
        out = VAEEncodeOutput(
            t=out.t.squeeze(0),
            model_t=out.model_t.squeeze(0),
            x_0=out.x_0.squeeze(0),
            x_t=out.x_t.squeeze(0),
            u_t=out.u_t.squeeze(0),
            latents=latents.squeeze(0),
        )

    return out


def _vae_encode_only_tensor(vae, image, vae_encode_type="sample", keep_depth=False):
    """Encode-only part of `_vae_encode_tensor`; returns ALWAYS-BATCHED latents.

    Unbatched 3D input is unsqueezed to 4D and the bsz dim is kept; the caller
    squeezes it back. This keeps `keep_depth` 4D latents `[c, T, h, w]` from
    being mistaken for batched `[B, c, h, w]`.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    else:
        assert image.ndim == 4, "image should have shape [B, C, H, W]"
    with torch.autocast(device_type="cuda", dtype=vae.autocast_dtype, enabled=vae.autocast_dtype != torch.float32):
        vae_encode_result = vae.encode(image)
        if isinstance(vae_encode_result, torch.Tensor):
            latents = vae_encode_result
        elif vae_encode_type == "sample":
            latents = vae_encode_result.latent_dist.sample(vae.generator)
        elif vae_encode_type == "mode":
            latents = vae_encode_result.latent_dist.mode()
        else:
            raise ValueError(f"Unknown vae_encode_type: {vae_encode_type}")
        
        latents = normalize_vae_latents(vae, latents)

    # b c t h w
    if hasattr(vae, "ffactor_temporal") and not keep_depth:
        assert latents.shape[2] == 1, "latents should have shape [B, C, D, H, W] and D should be 1"
        latents = latents.squeeze(2)

    return latents   # always batched [B, c, ...]


def _vae_add_noise_only_tensor(vae, latents, denoiser, sample_type, noise_dtype=None,
                               latent_channel_extend_type=None, has_bsz_dim=True):
    """Noise-add part of `_vae_encode_tensor`, taking BATCHED latents.

    When `has_bsz_dim` is False the t/model_t/x_0/x_t/u_t fields are squeezed
    along dim 0 (the `latents` field always stays batched).
    """
    assert latents.ndim >= 4, f"latents should be batched [B, c, ...], got ndim={latents.ndim}"

    if noise_dtype == "fp32":
        latents = latents.float()
    if latent_channel_extend_type is None:
        t, model_t, x_0, x_t, u_t = _add_noise(latents, denoiser, vae.noise_generator, sample_type)
    else:
        # For t2i task in video dit.
        out = add_noise_and_extend_channel(
            latents, None, True, denoiser, vae.noise_generator, sample_type,
            return_dict=True, latent_channel_extend_type=latent_channel_extend_type,
        )
        t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t

    if not has_bsz_dim:
        t = t.squeeze(0)
        model_t = model_t.squeeze(0)
        x_0 = x_0.squeeze(0)
        x_t = x_t.squeeze(0)
        u_t = u_t.squeeze(0)
    return VAEEncodeOutput(t=t, model_t=model_t, x_0=x_0, x_t=x_t, u_t=u_t, latents=latents)


def _vae_encode_tensor(vae, image, denoiser=None, sample_type=None, keep_depth=False,
                       vae_encode_type="sample", noise_dtype=None, latent_channel_extend_type=None):
    has_bsz_dim = image.ndim == 4
    latents = _vae_encode_only_tensor(vae, image, vae_encode_type=vae_encode_type, keep_depth=keep_depth)
    if sample_type is None:
        if not has_bsz_dim:
            latents = latents.squeeze(0)
        return VAEEncodeOutput(latents=latents)
    return _vae_add_noise_only_tensor(
        vae, latents, denoiser, sample_type,
        noise_dtype=noise_dtype, latent_channel_extend_type=latent_channel_extend_type,
        has_bsz_dim=has_bsz_dim,
    )


def _iter_nested_images(items, leaf_fn, device=None, stack=None, build=None):
    """Walk a nested image/latents structure and apply `leaf_fn(tensor, has_bsz)` per leaf."""
    aggregate = stack is not None
    call = (lambda t, hb: leaf_fn(t.to(device), hb)) if device is not None else leaf_fn

    if isinstance(items, list):
        outer = []
        for item in items:
            if is_torch_tensor(item) and item.ndim == 3:
                item = item.unsqueeze(0)
            if isinstance(item, list):
                inner = [call(x, False) for x in item]
                outer.append(stack(inner) if aggregate else inner)
            else:
                outer.append(call(item, True))
        return build(outer) if aggregate else outer

    if is_torch_tensor(items):
        if device is not None:
            items = items.to(device)
        if items.ndim == 5 and items.size(1) == 1:
            items = items.squeeze(1)
        if items.ndim == 4:
            return leaf_fn(items, True)
        if items.ndim == 5:
            outs = [leaf_fn(x, True) for x in items]
            return build(outs) if aggregate else outs
        raise ValueError(f"tensor should have shape [B,C,H,W] or [B,n,C,H,W], got {items.shape}")
    raise ValueError(f"Unknown items type, expected [list, torch.Tensor], got {type(items)}")


# three cases for images:
# 1. tensor [B, 1, C, H, W], return tensor [B, c, h, w]
# 2. tensors [B, N, C, H, W] (N > 1), return list of tensor B x [N, c, h, w]
# 3. list of lists of tensors B x (N_i x [C, H_ij, W_ij]), return list of lists of tensors B x (N_i x [c, h_ij, w_ij])
def vae_encode_and_add_noise(vae, images, device, denoiser=None, sample_type=None, keep_depth=False,
                             vae_encode_type="sample", noise_dtype=None, latent_channel_extend_type=None):
    kw = dict(denoiser=denoiser, sample_type=sample_type, keep_depth=keep_depth,
              vae_encode_type=vae_encode_type, noise_dtype=noise_dtype,
              latent_channel_extend_type=latent_channel_extend_type)
    return _iter_nested_images(
        images, lambda x, hb: _vae_encode_tensor(vae, x, **kw),
        device=device, stack=VAEEncodeOutput.stack, build=VAEEncodeOutput.build,
    )


def vae_encode_pure(vae, images, device, vae_encode_type="sample", keep_depth=False):
    """Encode-only counterpart of `vae_encode_and_add_noise`. No noise add."""
    return _iter_nested_images(
        images,
        lambda x, hb: _vae_encode_only_tensor(vae, x, vae_encode_type=vae_encode_type, keep_depth=keep_depth),
        device=device,
    )


def add_image_noise(vae, nested_latents, denoiser, sample_type, noise_dtype=None,
                    latent_channel_extend_type=None):
    """Noise-add counterpart of `vae_encode_and_add_noise`, taking encoded latents."""
    kw = dict(denoiser=denoiser, sample_type=sample_type, noise_dtype=noise_dtype,
              latent_channel_extend_type=latent_channel_extend_type)
    return _iter_nested_images(
        nested_latents, lambda x, hb: _vae_add_noise_only_tensor(vae, x, **kw, has_bsz_dim=hb),
        stack=VAEEncodeOutput.stack, build=VAEEncodeOutput.build,
    )


def _encode_last_frame(vae, last_frame, vae_encode_type="sample"):
    """VAE-encode one fl2v conditioning frame -> latents (bsz, c, 1, h, w)."""
    if last_frame.ndim == 3:
        last_frame = last_frame.unsqueeze(0)
    return _vae_encode_tensor(
        vae, last_frame.squeeze(1), vae_encode_type=vae_encode_type, keep_depth=True).latents


def vae_encode_last_frames(vae, last_frames, device, vae_encode_type="sample"):
    """Pre-encode fl2v last frames (preserving nesting) into `last_frame_latents`."""
    if last_frames is None:
        return None
    if isinstance(last_frames, (list, tuple)):
        return [vae_encode_last_frames(vae, lf, device, vae_encode_type) for lf in last_frames]
    return _encode_last_frame(vae, last_frames.to(device), vae_encode_type)


def scale_and_extend_frame(vae, latents, last_frame=None, vae_encode_type="sample",
                           last_frame_latents=None, last_frame_is_offline_latent=False, latent_pre_scaled=False):
    """
    Args:
        last_frame_latents: encode-balance 产出的 latent，已经 normalize 过。
        last_frame_is_offline_latent: 若为 True，表示 last_frame 是离线提取的原始 latent，
            不带 normalize，需要在此函数内调用 normalize_vae_latents。
    """
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)   # (c, t, h, w) -> (1, c, t, h, w)
        has_bsz_dim = False
    else:
        has_bsz_dim = True
    
    # Normalize after ensuring a batch dim so channel-wise latent-norm stats align
    # with latents.shape[1] (dim 1 must be channels, not the temporal axis).
    if not latent_pre_scaled:
        latents = normalize_vae_latents(vae, latents)

    # Use the pre-encoded last frame (encode-balance) when provided, else encode here.
    if last_frame_latents is None and last_frame is not None:
        if last_frame_is_offline_latent:
            # 离线编码的last_frame latent，只需要normalize
            lf = last_frame if last_frame.ndim == 5 else last_frame.unsqueeze(0)  # -> [B, c, 1, h, w]
            last_frame_latents = normalize_vae_latents(vae, lf)
        else:
            last_frame_latents = _encode_last_frame(vae, last_frame, vae_encode_type)  # (bsz, c, 1, h, w)
    if last_frame_latents is not None:
        assert last_frame_latents.size(2) == 1, \
            f"last_frame_latents should have shape [B, c, 1, h, w], got {last_frame_latents.shape}"
        latents = torch.cat([latents, last_frame_latents], dim=2)

    return latents, last_frame_latents, has_bsz_dim


# For video latents.
# Three cases for latents
# 1. tensor [B, 1, c, ...], return tensor [B, c, ...]
# 2. tensors [B, N, c, ...] (N > 1), return list of tensor B x [N, c, ...]
# 3. list of lists of tensors B x (N_i x [c, ...]), return list of lists of tensors B x (N_i x [c, ...])
def add_noise_to_latents(vae, latents, device, denoiser, sample_type=None, last_frame=None,
                         vae_encode_type="sample", noise_dtype=None, latent_channel_extend_type=None,
                         timesteps=None, last_frame_latents=None, last_frame_is_offline_latent=False, latent_pre_scaled=False):
    # last_frame_is_offline_latent: 是否离线提取latent，离线提取的latent不带normalization，需要做normalize
    kwargs = dict(denoiser=denoiser, generator=vae.noise_generator, sample_type=sample_type,
                  return_dict=True, noise_dtype=noise_dtype, latent_channel_extend_type=latent_channel_extend_type, timesteps=timesteps)

    if isinstance(latents, list):
        batch_output = []
        for bsz_i, latent_item in enumerate(latents):
            last_frame_i = last_frame[bsz_i] if last_frame is not None else None
            lfl_i = last_frame_latents[bsz_i] if last_frame_latents is not None else None
            if isinstance(latent_item, list):
                outputs = [
                    add_noise_and_extend_channel(
                        *scale_and_extend_frame(
                            vae,
                            latents=item.to(device),
                            last_frame=last_frame_i[j] if last_frame_i is not None else None,
                            vae_encode_type=vae_encode_type,
                            last_frame_latents=lfl_i[j] if lfl_i is not None else None,
                            last_frame_is_offline_latent=last_frame_is_offline_latent,
                            latent_pre_scaled=latent_pre_scaled
                        ),
                        **kwargs,
                    )
                    for j, item in enumerate(latent_item)
                ]
                outputs = VAEEncodeOutput.stack(outputs)
            else:
                outputs = add_noise_and_extend_channel(
                    *scale_and_extend_frame(
                        vae, latent_item.to(device), last_frame_i, vae_encode_type=vae_encode_type,
                        last_frame_latents=lfl_i,
                        last_frame_is_offline_latent=last_frame_is_offline_latent, latent_pre_scaled=latent_pre_scaled
                    ),
                    **kwargs,
                )
            batch_output.append(outputs)
        batch_output = VAEEncodeOutput.build(batch_output)

    elif is_torch_tensor(latents):
        latents = latents.to(device)
        if latents.ndim == 6 and latents.size(1) == 1:
            # Squeeze for simplicity: (B, 1, C, ...) -> (B, C, ...)
            latents = latents.squeeze(1)
            if last_frame is not None:
                last_frame = last_frame.squeeze(1)

        assert latents.ndim == 5, \
            f"latents should have shape [B, c, t, h, w] or [B, 1, c, t, h, w], got {latents.shape}"
        if last_frame is not None:
            if last_frame_is_offline_latent:
                assert last_frame.ndim == 5 and last_frame.size(2) == 1, \
                    f"Offline last-frame latent should be [B, c, 1, h, w], got {last_frame.shape}"
            else:
                assert last_frame.ndim == 4, \
                    f"last_frame should have shape [B, c, h, w], got {last_frame.shape}"

        # return tensor [B, c, ...]
        batch_output = add_noise_and_extend_channel(
            *scale_and_extend_frame(vae, latents, last_frame, vae_encode_type=vae_encode_type,
                                    last_frame_latents=last_frame_latents,
                                    last_frame_is_offline_latent=last_frame_is_offline_latent, latent_pre_scaled=latent_pre_scaled),
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown latents type, expected [list, torch.Tensor], got {type(latents)}")

    return batch_output
