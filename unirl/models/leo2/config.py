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
LEO2_AUDIO_LATENT_CHANNELS = 96
LEO2_AUDIO_SAMPLE_RATE = 48000

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
_AUDIO_VAE_ASSET_FILES = (
    "vae_audio_192d96l_3087k.ckpt",
    "stable_audio_1920_vae_htae_32gpu.json",
    "vae_audio_192d96l_3087k_mean_std/latent_mean.npy",
    "vae_audio_192d96l_3087k_mean_std/latent_std.npy",
)


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
            "av",
            "--use-system-prompt",
            "li-dit-encode-visual-qwen-3.5",
            "--gate-impl",
            "deepseek",
            "--vae-type",
            "16x16x4-48c-hy-v3_3-release2",
            "--use-audio-vae",
            "--audio-vae-type",
            "dual_channel_48k",
            "--audio-vae-latent-dim",
            "96",
        ]
    )

    # --- precisions (H3-shaped knobs) ---
    model_precision: str = "bf16"  # pinned native DCP tensor profile
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp32"
    logprob_precision: str = "fp32"

    # --- schedule ---
    video_shift: float = 3.0
    audio_shift: float = 3.0
    enable_audio: bool = True
    audio_stochastic_rollout: bool = True
    audio_joint_sde: bool = False
    native_rng_compat: bool = True
    reproduce: bool = False
    attention_impl: Optional[str] = None

    # --- native inference acceleration ---
    # The UniRL rollout stage enters the request-scoped model cache context;
    # replay/training deliberately remains exact.
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
    inference_cache_cfg_start_step: int = 1
    inference_cache_cfg_end_step: int = 50
    inference_cache_cfg_interval: int = 5
    inference_cache_cfg_low_frequency_weight: float = 1.1
    inference_cache_cfg_high_frequency_weight: float = 1.1
    inference_cache_cfg_low_frequency_start_step: int = 1
    inference_cache_cfg_low_frequency_end_step: int = 50
    inference_cache_cfg_high_frequency_start_step: int = 1
    inference_cache_cfg_high_frequency_end_step: int = 50

    # --- placement ---
    # Qwen3.5-9B (18 GB bf16) parked on CPU; moved to GPU transiently for the
    # once-per-rollout encode when this is true, then moved back.
    text_encoder_gpu_transient: bool = True
    # Reuse frozen text/model conditions for adjacent sibling samples. A size
    # of one is sufficient for prompt-major group rollouts.
    condition_cache_size: int = 1
    preprocessing_cache_dir: Optional[str] = None
    preprocessing_cache_mode: str = "off"
    load_video_vae: bool = True
    # Forward timing forces CUDA synchronization and is benchmark-only.
    profile_forward: bool = False
    vae_on_gpu: bool = True
    audio_vae_on_gpu: bool = True
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    enable_deepep: bool = False

    # loading
    skip_load_ckpt: bool = False  # debug only: random weights
    # FSDP2 needs one dtype per shard group; hymm keeps the MoE router fp32
    # inside blocks. True casts the whole DiT to bf16 (smoke trade-off).
    uniform_bf16: bool = True
    full_model_training: bool = False

    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve external artifacts and validate the portable Leo2 layout."""
        if self.preprocessing_cache_mode not in ("off", "readonly"):
            raise ValueError("Leo2 preprocessing_cache_mode must be 'off' or 'readonly'.")
        if self.preprocessing_cache_mode == "readonly" and not self.preprocessing_cache_dir:
            raise ValueError("Leo2 readonly preprocessing requires preprocessing_cache_dir.")
        for name in (
            "load_video_vae",
            "vae_on_gpu",
            "enable_audio",
            "audio_stochastic_rollout",
            "audio_joint_sde",
            "audio_vae_on_gpu",
            "native_rng_compat",
            "reproduce",
            "full_model_training",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(
                    f"Leo2 {name} must be bool, got {type(value).__name__}: {value!r}"
                )
        if not self.load_video_vae and self.preprocessing_cache_mode != "readonly":
            raise ValueError("Leo2 load_video_vae=false requires readonly preprocessing for cached SFT.")
        if self.enable_audio and "--use-audio-vae" not in self.extra_hymm_args:
            raise ValueError(
                "Leo2 enable_audio=true requires '--use-audio-vae' in extra_hymm_args"
            )
        if not isinstance(self.audio_shift, (int, float)):
            raise TypeError(
                f"Leo2 audio_shift must be numeric, got {type(self.audio_shift).__name__}: "
                f"{self.audio_shift!r}"
            )
        if float(self.audio_shift) <= 0:
            raise ValueError(f"Leo2 audio_shift must be positive, got {self.audio_shift}")
        if type(self.context_parallel_size) is not int:
            raise TypeError(
                "Leo2 context_parallel_size must be int, "
                f"got {type(self.context_parallel_size).__name__}: {self.context_parallel_size!r}."
            )
        if self.context_parallel_size < 1:
            raise ValueError(f"Leo2 context_parallel_size must be >= 1, got {self.context_parallel_size}.")
        if type(self.expert_parallel_size) is not int:
            raise TypeError(
                "Leo2 expert_parallel_size must be int, "
                f"got {type(self.expert_parallel_size).__name__}: {self.expert_parallel_size!r}."
            )
        if self.expert_parallel_size < 1:
            raise ValueError(f"Leo2 expert_parallel_size must be >= 1, got {self.expert_parallel_size}.")
        if type(self.enable_deepep) is not bool:
            raise TypeError(
                f"Leo2 enable_deepep must be bool, got {type(self.enable_deepep).__name__}: {self.enable_deepep!r}."
            )
        if self.attention_impl is not None and self.attention_impl not in (
            "flash",
            "flash_packed",
            "flash3",
            "flash3_packed",
            "flex",
            "sdpa",
            "sageattn",
        ):
            raise ValueError(f"Leo2 attention_impl is unsupported: {self.attention_impl!r}.")
        if self.enable_deepep and self.expert_parallel_size == 1:
            raise ValueError(
                "Leo2 enable_deepep=true requires expert_parallel_size > 1, "
                f"got expert_parallel_size={self.expert_parallel_size}."
            )
        if type(self.condition_cache_size) is not int:
            raise TypeError(
                "Leo2 condition_cache_size must be int, "
                f"got {type(self.condition_cache_size).__name__}: {self.condition_cache_size!r}."
            )
        if self.condition_cache_size < 0:
            raise ValueError(f"Leo2 condition_cache_size must be >= 0, got {self.condition_cache_size}.")
        if type(self.profile_forward) is not bool:
            raise TypeError(
                "Leo2 profile_forward must be bool, "
                f"got {type(self.profile_forward).__name__}: {self.profile_forward!r}."
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
        if self.enable_audio:
            audio_vae_assets = assets / "audio_encoder/dual_channel_48k"
            _require_path(audio_vae_assets, "dual-channel audio VAE assets", directory=True)
            for name in _AUDIO_VAE_ASSET_FILES:
                _require_path(
                    audio_vae_assets / name,
                    f"audio VAE asset {name}",
                    directory=False,
                )


__all__ = [
    "Leo2PipelineConfig",
    "LEO2_AUDIO_LATENT_CHANNELS",
    "LEO2_AUDIO_SAMPLE_RATE",
    "LEO2_VAE_LATENT_CHANNELS",
    "LEO2_VAE_SPATIAL",
    "LEO2_VAE_TEMPORAL",
    "LEO2_TIMESTEP_SCALE",
]
