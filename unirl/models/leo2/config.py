"""Leo2 pipeline configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

LEO2_VAE_LATENT_CHANNELS = 48
LEO2_VAE_SPATIAL = 16
LEO2_VAE_TEMPORAL = 4
LEO2_TIMESTEP_SCALE = 1000.0

_VENDORED_GEN_AR_ROOT = Path(__file__).resolve().parent / "vendor" / "gen_ar"
_DEFAULT_LEO2_CONFIG = _VENDORED_GEN_AR_ROOT / "hymm/configs/leo2/leo2_moe_v1_1_a12b_muon_wzd_480p_stage3.yaml"
_DEFAULT_GENERATION_CONFIG = Path(__file__).resolve().parent / "resources" / "generation_config_rl_video.json"
_QWEN_ASSET_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model.safetensors-00001-of-00004.safetensors",
    "model.safetensors-00002-of-00004.safetensors",
    "model.safetensors-00003-of-00004.safetensors",
    "model.safetensors-00004-of-00004.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_VAE_ASSET_FILES = ("config.json", "latent_norm_stats.pt", "pytorch_model.pt")


def _required_external_path(value: str, env_name: str) -> str:
    """Resolve an explicit artifact path and reject an unset value."""
    path = value or os.environ.get(env_name, "")
    if not path.strip():
        raise ValueError(f"Leo2 requires {env_name}; point it at the external artifact described in artifacts.yaml.")
    return str(Path(path).expanduser().resolve())


def _require_path(path: Path, label: str, *, directory: bool) -> None:
    """Reject a missing file or directory with its Leo2 artifact label."""
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"Leo2 {label} {kind} does not exist: {path}")


@dataclass
class Leo2PipelineConfig:
    # --- code + weights ---
    hymm_repo_path: str = str(_VENDORED_GEN_AR_ROOT)
    config_yaml: str = str(_DEFAULT_LEO2_CONFIG)
    ckpt_path: str = ""
    generation_config_path: str = str(_DEFAULT_GENERATION_CONFIG)
    assets_base: str = ""
    # Extra hymm cmd args appended after the yaml (mirrors t2v_smoke.sh minus
    # sampling params, which UniRL owns).
    extra_hymm_args: List[str] = field(
        default_factory=lambda: [
            "--bot-task",
            "video",
            "--use-system-prompt",
            "li-dit-encode-visual-qwen-3.5",
            "--gate-impl",
            "deepseek",
            "--vae-type",
            "16x16x4-48c-hy-v3_3-release2",
        ]
    )

    # --- precisions (H3-shaped knobs) ---
    model_precision: str = "bf16"  # pinned native DCP tensor profile
    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # --- schedule ---
    video_shift: float = 3.0

    # --- native inference acceleration ---
    # This is intentionally inactive in UniRL rollout/replay: only hymm's
    # request-scoped diffusion pipeline enters the model cache context.
    inference_cache_method: str = "none"
    inference_cache_threshold: float = 0.05
    inference_cache_taylor_max_extrapolation: float = 1.0
    inference_cache_magcache_max_skip_steps: int = 6
    inference_cache_magcache_retention_ratio: float = 0.2
    inference_cache_magcache_ratios: List[float] = field(default_factory=list)
    inference_cache_magcache_expected_timesteps: List[float] = field(default_factory=list)
    inference_cache_magcache_calibrate: bool = False
    inference_cache_fastercache_start_step: int = 4
    inference_cache_fastercache_end_step: int = 46
    inference_cache_fastercache_interval: int = 2
    inference_cache_fastercache_layers: Optional[List[int]] = None

    # --- placement ---
    # Qwen3.5-9B (18 GB bf16) parked on CPU; moved to GPU transiently for the
    # once-per-rollout encode when this is true, then moved back.
    text_encoder_gpu_transient: bool = True
    vae_on_gpu: bool = True
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    enable_deepep: bool = False

    # loading
    skip_load_ckpt: bool = False  # debug only: random weights
    # FSDP2 needs one dtype per shard group; hymm keeps the MoE router fp32
    # inside blocks. True casts the whole DiT to bf16 (smoke trade-off).
    uniform_bf16: bool = True

    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve external artifacts and validate the portable Leo2 layout."""
        if type(self.context_parallel_size) is not int:
            raise TypeError(
                "Leo2 context_parallel_size must be int, "
                f"got {type(self.context_parallel_size).__name__}: {self.context_parallel_size!r}."
            )
        if self.context_parallel_size < 1:
            raise ValueError(
                f"Leo2 context_parallel_size must be >= 1, got {self.context_parallel_size}."
            )
        if type(self.expert_parallel_size) is not int:
            raise TypeError(
                "Leo2 expert_parallel_size must be int, "
                f"got {type(self.expert_parallel_size).__name__}: {self.expert_parallel_size!r}."
            )
        if self.expert_parallel_size < 1:
            raise ValueError(
                f"Leo2 expert_parallel_size must be >= 1, got {self.expert_parallel_size}."
            )
        if type(self.enable_deepep) is not bool:
            raise TypeError(
                "Leo2 enable_deepep must be bool, "
                f"got {type(self.enable_deepep).__name__}: {self.enable_deepep!r}."
            )
        if self.enable_deepep and self.expert_parallel_size == 1:
            raise ValueError(
                "Leo2 enable_deepep=true requires expert_parallel_size > 1, "
                f"got expert_parallel_size={self.expert_parallel_size}."
            )
        self.ckpt_path = _required_external_path(self.ckpt_path, "LEO2_CKPT_DIR")
        self.assets_base = _required_external_path(self.assets_base, "LEO2_ASSETS_BASE")
        self.hymm_repo_path = str(Path(self.hymm_repo_path).expanduser().resolve())
        self.config_yaml = str(Path(self.config_yaml).expanduser().resolve())
        self.generation_config_path = str(Path(self.generation_config_path).expanduser().resolve())

        repo = Path(self.hymm_repo_path)
        checkpoint = Path(self.ckpt_path)
        assets = Path(self.assets_base)
        _require_path(repo / "hymm", "vendored hymm runtime", directory=True)
        _require_path(repo / "processors", "vendored processors runtime", directory=True)
        _require_path(repo / "deps/hy_parallelism/hy_parallelism", "vendored hy_parallelism", directory=True)
        _require_path(repo / "deps/IndexKits/index_kits", "vendored IndexKits", directory=True)
        _require_path(Path(self.config_yaml), "model config", directory=False)
        _require_path(Path(self.generation_config_path), "generation config", directory=False)
        _require_path(checkpoint, "checkpoint", directory=True)
        _require_path(checkpoint / ".metadata", "checkpoint metadata", directory=False)
        if not any(checkpoint.glob("*.distcp")):
            raise FileNotFoundError(f"Leo2 checkpoint has no *.distcp shard: {checkpoint}")
        qwen_assets = assets / "text_encoder/Qwen3.5-9B"
        vae_assets = assets / "image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2"
        _require_path(qwen_assets, "Qwen3.5-9B assets", directory=True)
        _require_path(vae_assets, "video VAE assets", directory=True)
        for name in _QWEN_ASSET_FILES:
            _require_path(qwen_assets / name, f"Qwen3.5-9B asset {name}", directory=False)
        for name in _VAE_ASSET_FILES:
            _require_path(vae_assets / name, f"video VAE asset {name}", directory=False)


__all__ = [
    "Leo2PipelineConfig",
    "LEO2_VAE_LATENT_CHANNELS",
    "LEO2_VAE_SPATIAL",
    "LEO2_VAE_TEMPORAL",
    "LEO2_TIMESTEP_SCALE",
]
