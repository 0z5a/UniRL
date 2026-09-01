from dataclasses import dataclass, field
from typing import List
from hymm.utils.states import DataClassMixin

MODEL_ZOO: dict[str, "QwenViTConfig"] = {}


def register_model_config(name, base=None, **kwargs):
    """ Register model config to MODEL_ZOO."""
    if base is not None:
        if base not in MODEL_ZOO:
            raise ValueError(f"Base model {base} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        base_config = MODEL_ZOO[base].to_dict()
        base_config.update({**kwargs, "name": name})
        MODEL_ZOO[name] = QwenViTConfig(**base_config)
    else:
        MODEL_ZOO[name] = QwenViTConfig(name=name, **kwargs)


@dataclass
class QwenViTConfig(DataClassMixin):
    """ Configuration class for AnyResViT model. """
    name: str = ""
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    initializer_range: float = 0.02

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "QwenViTConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = MODEL_ZOO[model_name].to_dict()
        model_config.update(kwargs)
        return cls(**model_config)


# =========================================
#     Predefined Model Configurations
# =========================================
register_model_config(
    name="qwen3vl-vit-for-30b-a3b",
    depth=27,
    hidden_size=1152,
    hidden_act="gelu_pytorch_tanh",
    intermediate_size=4304,
    num_heads=16,
    in_channels=3,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    out_hidden_size=2048,
    num_position_embeddings=2304,
    deepstack_visual_indexes=[8, 16, 24],
)
