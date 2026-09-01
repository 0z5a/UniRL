"""Leo2 pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

LEO2_VAE_LATENT_CHANNELS = 48
LEO2_VAE_SPATIAL = 16
LEO2_VAE_TEMPORAL = 4
LEO2_TIMESTEP_SCALE = 1000.0

_VENDORED_GEN_AR_ROOT = Path(__file__).resolve().parent / "vendor" / "gen_ar"
_DEFAULT_LEO2_CONFIG = _VENDORED_GEN_AR_ROOT / "hymm/configs/leo2/leo2_moe_v1_1_a12b_muon_wzd_256p_stage2_part2.yaml"


@dataclass
class Leo2PipelineConfig:
    # --- code + weights ---
    hymm_repo_path: str = str(_VENDORED_GEN_AR_ROOT)
    config_yaml: str = str(_DEFAULT_LEO2_CONFIG)
    ckpt_path: str = (
        "/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0/ckpts/leo2_moe_a12b_480p/"
        "iter_0063300_torch/weights"
    )
    generation_config_path: str = (
        "/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0/configs/leo2_genconfig_rl.json"
    )
    assets_base: str = (
        "/apdcephfs_zwfy8/share_305110755/hunyuan/zuhaoding/HYV2.0/assets/hymm_ar_assets"
    )
    # extra hymm cmd args appended after the yaml (mirrors t2v_smoke.sh minus
    # sampling params, which UniRL owns). EP must stay 1: UniRL FSDP hosts the
    # experts locally (non-fused path).
    extra_hymm_args: List[str] = field(default_factory=lambda: [
        "--bot-task", "video",
        "--use-system-prompt", "li-dit-encode-visual-qwen-3.5",
        "--gate-impl", "deepseek",
        "--vae-type", "16x16x4-48c-hy-v3_3-release2",
    ])

    # --- precisions (H3-shaped knobs) ---
    model_precision: str = "bf16"
    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # --- schedule ---
    video_shift: float = 3.0

    # --- placement ---
    # Qwen3.5-9B (18 GB bf16) parked on CPU; moved to GPU transiently for the
    # once-per-rollout encode when this is true, then moved back.
    text_encoder_gpu_transient: bool = True
    vae_on_gpu: bool = True

    # loading
    skip_load_ckpt: bool = False  # debug only: random weights
    # FSDP2 needs one dtype per shard group; hymm keeps the MoE router fp32
    # inside blocks. True casts the whole DiT to bf16 (smoke trade-off).
    uniform_bf16: bool = True

    device: Optional[str] = None


__all__ = [
    "Leo2PipelineConfig",
    "LEO2_VAE_LATENT_CHANNELS",
    "LEO2_VAE_SPATIAL",
    "LEO2_VAE_TEMPORAL",
    "LEO2_TIMESTEP_SCALE",
]
