"""Decode Leo2 video latents into the UniRL Videos primitive."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Video, Videos

if TYPE_CHECKING:
    from .bundle import Leo2Bundle


_DUMP_COUNT = 0


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
        vae = pipeline.vae
        latents = latents.to(device=self.bundle.device, dtype=torch.float32)
        latents = denormalize_vae_latents(vae, latents)

        vae_dtype = pipeline.vae_autocast_dtype
        with torch.autocast(
            device_type="cuda",
            dtype=vae_dtype,
            enabled=vae_dtype is not None and vae_dtype != torch.float32,
        ):
            visuals = vae.decode(latents, return_dict=False)[0]  # [B, C, T, H, W] in [-1, 1]
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
