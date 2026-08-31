"""Leo2 (HYVideo 2.0, MoE-A12B) model package for UniRL trainside FlowGRPO.

Local-only integration: the model implementation stays in the external
hunyuan_multimodal_gen_ar repo (see config.hymm_repo_path); this package holds
just the UniRL bundle / pipeline / stage adapters. t2v only.
"""

from .bundle import Leo2Bundle
from .conditions import Leo2Conditions
from .config import Leo2PipelineConfig
from .diffusion import Leo2DiffusionStage
from .pipeline import Leo2Pipeline
from .text_embed import Leo2CondStage
from .vae import Leo2VideoDecodeStage

__all__ = [
    "Leo2Bundle",
    "Leo2Conditions",
    "Leo2PipelineConfig",
    "Leo2DiffusionStage",
    "Leo2Pipeline",
    "Leo2CondStage",
    "Leo2VideoDecodeStage",
]
