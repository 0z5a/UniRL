"""Build Leo2 supervised Parts from frozen preprocessing caches."""

from __future__ import annotations

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.train.sft.track_builder import SupervisedTrackBuilder
from unirl.types.media import MediaRef
from unirl.types.primitives import Texts
from unirl.types.sample import Part
from unirl.types.segments.latent import make_video_segment


def target_video_uri(record: dict) -> str:
    """Require a text-only prompt with exactly one target video."""
    refs = record.get("media_refs") or []
    if (
        "messages" in record
        or any(not isinstance(ref, MediaRef) or ref.role != "target" or ref.modality != "video" for ref in refs)
        or len(refs) != 1
    ):
        raise ValueError(
            "Leo2 cached SFT requires prompt:str and exactly one role='target', modality='video' media ref."
        )
    return refs[0].uri


def validate_target(latents: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """Reject stale or malformed targets before constructing a supervised segment."""
    if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != (1, *expected_shape):
        raise ValueError(
            f"Leo2 cached target must have shape {(1, *expected_shape)}, got {getattr(latents, 'shape', None)}."
        )
    if not latents.is_floating_point() or not torch.isfinite(latents).all():
        raise ValueError("Leo2 cached target must contain finite floating-point model-space latents.")


class Leo2CachedSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Read prompt conditions and clean video targets without loading frozen encoders."""

    def __init__(self, *, pipeline, height: int, width: int, num_frames: int, max_decode_frames: int = 256) -> None:
        super().__init__()
        if pipeline.cond_stage.disk_cache is None:
            raise ValueError("Leo2 cached SFT requires preprocessing_cache_mode='readonly'.")
        if min(height, width, num_frames) < 1 or max_decode_frames < num_frames:
            raise ValueError("Leo2 SFT geometry must be positive and max_decode_frames must cover num_frames.")
        self.pipeline = pipeline
        self.geometry = dict(height=height, width=width, num_frames=num_frames)
        self.max_decode_frames = max_decode_frames

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    @torch.no_grad()
    def build(self, records: list[dict]) -> Part:
        """Read a shard of cached x0 and conditions while preserving lineage and evaluation padding."""
        if not records:
            raise ValueError("Leo2 cached SFT received an empty record shard.")
        uris = [target_video_uri(record) for record in records]
        conditions = self.pipeline.cond_stage.build(
            Texts(texts=[record["prompt"] for record in records]),
            seeds=[0] * len(records),
            **self.geometry,
        )
        expected = self.pipeline._latent_shape_from_conditions(conditions)
        cache = self.pipeline.cond_stage.disk_cache
        targets = []
        for uri, blob in zip(uris, conditions.hymm):
            key = cache.target_key(uri, blob, max_decode_frames=self.max_decode_frames)
            latent = cache.read(key)
            validate_target(latent, expected)
            targets.append(latent)
        latents = torch.cat(targets).to(self.pipeline.bundle.device)
        loss_mask = torch.tensor(
            [not record.get("_eval_pad", False) for record in records],
            device=latents.device,
            dtype=torch.float32,
        )
        return Part(
            sample_ids=[str(record["sample_id"]) for record in records],
            conditions=conditions.to_dict(),
            segment=make_video_segment(latents=latents.unsqueeze(1), loss_mask=loss_mask),
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )
