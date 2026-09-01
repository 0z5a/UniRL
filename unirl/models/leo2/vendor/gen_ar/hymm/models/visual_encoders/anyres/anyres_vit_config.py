from dataclasses import dataclass

from hymm.utils.states import DataClassMixin

MODEL_ZOO: dict[str, "AnyResViTConfig"] = {}


def register_model_config(name, base=None, **kwargs):
    """ Register model config to MODEL_ZOO."""
    if base is not None:
        if base not in MODEL_ZOO:
            raise ValueError(f"Base model {base} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        base_config = MODEL_ZOO[base].to_dict()
        base_config.update({**kwargs, "name": name})
        MODEL_ZOO[name] = AnyResViTConfig(**base_config)
    else:
        MODEL_ZOO[name] = AnyResViTConfig(name=name, **kwargs)


@dataclass
class AnyResViTConfig(DataClassMixin):
    """ Configuration class for AnyResViT model. """
    name: str = ""
    num_hidden_layers: int = 40
    hidden_size: int = 1536
    num_attention_heads: int = 16
    split_qkv: bool = True
    intermediate_size: int = 6144
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    # Embedding
    num_channels: int = 3
    patch_size: int = 16
    max_image_size: int = 2048
    max_vit_seq_len: int = 16384
    add_patch_emb_bias: bool = True
    anyres_vit_max_image_size: int = 2048
    interpolate_mode: str = "bicubic"
    remove_prenorm: bool = True
    # Adapter
    adaptor_patch_size: int = 2
    # Acceleration
    use_flash_attention: bool = False
    # Additional options
    use_after_rms: bool = True
    cat_extra_token: bool = True
    use_imagenet_norm: bool = True

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "AnyResViTConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = MODEL_ZOO[model_name].to_dict()
        model_config.update(kwargs)
        return cls(**model_config)


# =========================================
#     Predefined Model Configurations
# =========================================

register_model_config(
    name="anyres-vit-for-a3b",
    num_hidden_layers=27,
    hidden_size=1152,
    num_attention_heads=16,
    split_qkv=True,
    intermediate_size=4304,
    hidden_act="gelu",
    num_channels=3,
    patch_size=16,
    max_image_size=2048,
    max_vit_seq_len=16384,
    add_patch_emb_bias=True,
    anyres_vit_max_image_size=2048,
    interpolate_mode="bilinear",
    remove_prenorm=True,
    adaptor_patch_size=2,
    use_after_rms=True,
    cat_extra_token=True,
    use_imagenet_norm=True,
)

register_model_config(
    name="anyres-vit-for-a30b",
    num_hidden_layers=40,
    hidden_size=1536,
    num_attention_heads=16,
    split_qkv=True,
    intermediate_size=6144,
    hidden_act="gelu",
    num_channels=3,
    patch_size=16,
    max_image_size=2048,
    max_vit_seq_len=16384,
    add_patch_emb_bias=True,
    anyres_vit_max_image_size=2048,
    interpolate_mode="bilinear",
    remove_prenorm=True,
    adaptor_patch_size=2,
    use_after_rms=True,
    cat_extra_token=True,
    use_imagenet_norm=True,
)

register_model_config(
    name="anyres-vit-for-hy3-a3b",
    num_hidden_layers=27,
    hidden_size=1152,
    num_attention_heads=16,
    split_qkv=True,
    intermediate_size=4304,
    hidden_act="gelu",
    num_channels=3,
    patch_size=16,
    max_image_size=2048,
    max_vit_seq_len=16384,
    add_patch_emb_bias=True,
    anyres_vit_max_image_size=2048,
    interpolate_mode="bilinear",
    remove_prenorm=True,
    adaptor_patch_size=2,
    use_after_rms=False,
    cat_extra_token=False,
    use_imagenet_norm=False,
)

register_model_config(
    name="anyres-vit-ci-test",
    num_hidden_layers=2,
    hidden_size=32,
    num_attention_heads=4,
    split_qkv=True,
    intermediate_size=4 * 32, # 4 * hidden_size
    hidden_act="gelu",
    num_channels=3,
    patch_size=16,
    max_image_size=256,
    max_vit_seq_len=512,
    add_patch_emb_bias=True,
    anyres_vit_max_image_size=256,
    interpolate_mode="bilinear",
    remove_prenorm=True,
    adaptor_patch_size=1,
)