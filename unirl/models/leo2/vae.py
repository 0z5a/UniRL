"""Decode Leo2 video latents into the UniRL Videos primitive."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Video, Videos

if TYPE_CHECKING:
    from .bundle import Leo2Bundle


_DUMP_COUNT = 0


@contextmanager
def video_vae_ctx(bundle: "Leo2Bundle"):
    """Host the frozen VAE only during codec work when vae_on_gpu is false."""
    vae = bundle.model.model_dict.get("vae")
    if vae is None:
        raise RuntimeError("Leo2 VAE is not loaded; enable load_video_vae for video encoding or decoded rollouts.")
    transient = not bundle.config.vae_on_gpu
    try:
        vae.to(bundle.device)
        yield vae
    finally:
        if transient:
            if callable(getattr(vae, "clear_cache", None)):
                vae.clear_cache()
            vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@torch.no_grad()
def encode_target_video(bundle: "Leo2Bundle", uri: str, blob: dict, *, max_decode_frames: int) -> torch.Tensor:
    """Encode a deterministically sampled and resized target video into normalized model-space x0."""
    from hymm.models.autoencoders import normalize_vae_latents

    from unirl.utils.video import load_video

    frames_count = blob["video_duration"]
    if max_decode_frames < frames_count:
        raise ValueError(f"max_decode_frames={max_decode_frames} must cover effective num_frames={frames_count}.")
    pixels = load_video(uri, max_frames=max_decode_frames)
    indices = torch.linspace(0, pixels.shape[0] - 1, frames_count).round().long()
    pixels = torch.nn.functional.interpolate(
        pixels[indices],
        size=tuple(blob["image_size"]),
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    pixels = (pixels.permute(1, 0, 2, 3).unsqueeze(0) * 2 - 1).to(bundle.device)
    with video_vae_ctx(bundle) as vae:
        # Match the native pipeline's default codec precision (autocast disabled).
        latent = vae.encode(pixels).latent_dist.mode()
        latent = normalize_vae_latents(vae, latent.float())
        return latent.detach().cpu()


def _maybe_dump_frames(visuals: torch.Tensor, tag: str = "decode", max_dumps: int = 6) -> None:
    """Write first/middle/last-frame contact sheets when ``LEO2_DUMP_VIDEOS`` is set."""
    global _DUMP_COUNT
    out_dir = os.environ.get("LEO2_DUMP_VIDEOS")
    if not out_dir or _DUMP_COUNT >= max_dumps:
        return
    _DUMP_COUNT += 1
    try:
        from PIL import Image

        v = visuals[0].detach().float().cpu()  # [C, T, H, W]
        t = v.shape[1]
        sheet = torch.cat([v[:, i] for i in (0, t // 2, t - 1)], dim=-1)  # [C, H, 3W]
        arr = (sheet.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        os.makedirs(out_dir, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        path = os.path.join(out_dir, f"{tag}{_DUMP_COUNT}_rank{rank}_{os.getpid()}.png")
        Image.fromarray(arr).save(path)
        print(f"[leo2 dump] wrote {path} T={t} range=({v.min():.3f},{v.max():.3f})", flush=True)
    except Exception as exc:  # never let a debug dump kill a rollout
        print(f"[leo2 dump] failed: {exc}", flush=True)


class Leo2VideoDecodeStage:
    def __init__(self, bundle: "Leo2Bundle") -> None:
        self.bundle = bundle

    @torch.no_grad()
    def decode_to_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        """``latents [B, C, T, H, W]`` (model space) -> pixels ``[B, C, T, H, W]`` in [0, 1]."""
        from hymm.models.autoencoders import denormalize_vae_latents

        pipeline = self.bundle.model.diffusion_pipeline
        vae_dtype = getattr(pipeline, "vae_autocast_dtype", None)
        with video_vae_ctx(self.bundle) as vae:
            latents = latents.to(device=self.bundle.device, dtype=torch.float32)
            latents = denormalize_vae_latents(vae, latents)
            with torch.autocast(
                device_type="cuda",
                dtype=vae_dtype,
                enabled=vae_dtype is not None and vae_dtype != torch.float32,
            ):
                visuals = vae.decode(latents, return_dict=False)[0]
        return (visuals.float() / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> Videos:
        """``latents [B, C, T, H, W]`` (fp32/traj dtype) -> Videos ([T,C,H,W] each)."""
        visuals = self.decode_to_tensor(latents)
        _maybe_dump_frames(visuals, tag="rollout")
        return Videos.from_list(
            [
                Video(frames=visuals[i].permute(1, 0, 2, 3).contiguous().cpu())  # [T, C, H, W]
                for i in range(visuals.shape[0])
            ]
        )


__all__ = ["Leo2VideoDecodeStage"]
