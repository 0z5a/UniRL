# coding=utf-8
# Copyright (C) 2026 THL A29 Limited, a Tencent company.  All rights reserved.
"""HunYuan model configuration"""

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from hymm.models.basic.model_config import TransformerConfig

MODEL_ZOO: dict[str, dict[str, Any]] = {}


def register_model_config(name, base=None, **kwargs):
    """ Register model config to MODEL_ZOO."""
    if base is not None:
        if base not in MODEL_ZOO:
            raise ValueError(f"Base model {base} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        base_config = deepcopy(MODEL_ZOO[base])
        base_config.update(deepcopy(kwargs))
        base_config["name"] = name
        # NOTE:
        # Do NOT instantiate config objects during "registration".
        # Because `from_name` could provide different arguments from the registration.
        # We should lazily instantiate the config object in `from_name` to ensure proper post_init.
        MODEL_ZOO[name] = base_config
    else:
        MODEL_ZOO[name] = {"name": name, **deepcopy(kwargs)}


@dataclass
class LeoConfig(TransformerConfig):
    # common
    name: str = ""

    # ===============================
    #     Transformer Config
    # ===============================
    # ---- model architecture ----
    # basic
    num_layers: int = 24
    hidden_size: int = 3072
    max_position_embeddings: int = 327_600
    # attention module
    attn_impl: str = "flash"
    num_attention_heads: int = 24
    num_kv_heads: int | None = None  # for multi-query attention, default to num_attention_heads
    attention_head_size: int | None = None  # default to hidden_size // num_attention_heads
    attention_bias: bool = False
    attention_dropout: float = 0.0
    split_qkv: bool = False
    use_qk_norm: bool = False
    pre_qk_norm: bool = False # whether to apply QK norm before applying RoPE
    qk_norm_type: str | None = None     # if not specified, default to norm_type.
    # norm module
    norm_type: str = "layer_f32"
    norm_elementwise_affine: bool = False
    # mlp module
    ffn_hidden_size: int | None = None  # default to 4 * hidden_size
    hidden_act: Literal["silu", "gelu"] = "silu"
    mlp_bias: bool = False
    split_gate_and_up: bool = False
    # moe module
    moe_impl: str = "deepseek"
    gate_impl: str = None
    num_experts: int | list = 0
    moe_ffn_hidden_size: int | list = 0
    moe_mixed_mlp: int | list = 0      # 0 for disabled, >1 for enabled shared experts
    moe_score_func: str = "softmax"
    moe_topk: int | list = 1
    norm_topk_prob: bool = True
    capacity_factor: float = 1.0
    moe_drop_tokens: bool = False
    moe_random_routing_dropped_tokens: bool = False
    routed_scaling_factor: float = 1.0
    moe_layer_num_skipped: int = 0
    moe_aux_loss: bool = True
    moe_seq_aux_loss: str = False  # compute aux loss batch-wise(False) or sequence-wise(True)
    use_mot: bool = False
    use_dense_mot_gen: bool = False
    # ---- numerical related ----
    # initialization
    init_std: float = 0.02
    norm_eps: float = 1e-6
    # rope
    rope_type: str = "leo2_3d"
    rope_interleave: bool = True
    rope_theta: float = 2000.0
    rope_scaling: float = 1.0
    rope_float_position: bool = False  # whether to use float position for rope calculation. If False, use integer position.
    rope_no_space: bool = False
    rope_fixed_space: Optional[int] = None
    rope_cond_image_fixed_space: Optional[int] = None
    rope_cond_video_fixed_space: Optional[int] = None
    mrope_section: list[int] | None = None
    use_scale_rope: bool = False
    apply_rope_in_fp32: bool = True

    # ===============================
    #           Text Config
    # ===============================
    use_modulation: bool = True
    modulate_hidden_size: int | None = None # default to hidden_size
    text_proj_type: Literal["linear", "single_refiner"] = "single_refiner"
    text_states_hidden_dim: int | None = None
    token_refiner_num_attention_heads: int | None = None    # default to modulate_hidden_size // attention_head_size
    # auto: In non-packing mode, image-text order; in packing mode, text-image order.
    # text_first: text-image order, only influences the non-packing mode.
    text_order: Literal["auto", "text_first"] = "auto"
    skip_txt_after_last_attn: bool = False  # The last attn o_proj and mlp_txt have no gradients and can be ignored.

    # ===============================
    #     Image/Video VAE Config
    # ===============================
    # ---- model architecture ----
    # image/video vae
    use_vae: bool = False
    vae_type: str = ""
    vae_latent_dim: int = 32
    vae_spatial_downsample_factor: int = 16
    vae_temporal_downsample_factor: int = 4
    vae_precision: str = "fp32"
    vae_autocast_dtype: str = "fp16"
    # vae image projector
    patch_size: int = 1
    img_proj_type: Literal["linear", "conv"] = "linear"
    img_proj_ndim: int = 3
    patch_embed_hidden_dim: int = 1024
    use_timestep_token: bool = False
    img_latent_in_channels: int | None = None # default to vae_latent_dim

    # ===============================
    #        Audio VAE Config
    # ===============================
    # ---- model architecture ----
    # audio vae
    use_audio_vae: bool = False
    audio_vae_type: str = ""
    audio_vae_latent_dim: int = 64
    # vae audio projector
    audio_proj_type: str = "audio_linear"

    # ===============================
    #           Other Config
    # ===============================
    cond_vae_zero_timestep: bool = True

    # REPA. Only for training.
    use_repa: bool = False
    repa_encoder_type: str = "DINOv3"
    repa_proj_hidden_size: int | None = None
    repa_proj_out_size: int | None = None

    # FP32 modules
    fp32_modules: list[str] = None

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

        if self.modulate_hidden_size is None:
            self.modulate_hidden_size = self.hidden_size

        if self.token_refiner_num_attention_heads is None:
            self.token_refiner_num_attention_heads = self.hidden_size // self.attention_head_size

        if self.img_latent_in_channels is None:
            self.img_latent_in_channels = self.vae_latent_dim

        # Sanity check
        if self.rope_type == "leo2_3d":
            assert self.rope_interleave, "leo2_3d rope requires interleaving. Please set rope_interleave to True."
        else:
            assert not self.rope_interleave, \
                f"{self.rope_type} rope does not support interleaving. Please set rope_interleave to False."

        # Compatibility with potential parent classes
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def to_hf_config(self) -> dict[str, Any]:
        # Convert to HuggingFace PretrainedConfig by extending HunYuanMoEV2Config
        hf_configs = dict(
            hidden_size=self.hidden_size,
            intermediate_size=self.ffn_hidden_size,
            moe_intermediate_size=self.moe_ffn_hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_kv_heads,
            attention_head_dim=self.attention_head_size,
            # used by Cache, and especially when head_dim != hidden_size // num_attention_heads
            head_dim=self.attention_head_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.init_std,
            rms_norm_eps=self.norm_eps,
            # RoPE configs
            rope_type=self.rope_type,
            rope_interleave=self.rope_interleave,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            rope_float_position=self.rope_float_position,
            rope_no_space=self.rope_no_space,
            rope_fixed_space=self.rope_fixed_space,
            mrope_section=self.mrope_section,
            xdrope_section=self.xdrope_section,
            use_scale_rope=self.use_scale_rope,
            apply_rope_in_fp32=self.apply_rope_in_fp32,
            # misc configs
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            attention_dropout=self.attention_dropout,
            use_qk_norm=self.use_qk_norm,
            use_rotary_pos_emb=True,
            norm_type=self.norm_type,
            # moe configs
            moe_impl=self.moe_impl,
            gate_impl=self.gate_impl,
            num_experts=self.num_experts,
            use_mixed_mlp_moe=self.moe_mixed_mlp > 0,
            num_shared_expert=self.moe_mixed_mlp,
            moe_score_func=self.moe_score_func,
            moe_topk=self.moe_topk,
            capacity_factor=self.capacity_factor,
            moe_drop_tokens=self.moe_drop_tokens,
            moe_random_routing_dropped_token=self.moe_random_routing_dropped_tokens,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            moe_layer_num_skipped=self.moe_layer_num_skipped,
            moe_aux_loss=self.moe_aux_loss,
            moe_seq_aux_loss=self.moe_seq_aux_loss,
            # mot
            use_mot=self.use_mot,
            use_dense_mot_gen=self.use_dense_mot_gen,
            # text configs
            use_modulation=self.use_modulation,
            modulate_hidden_size=self.modulate_hidden_size,
            text_proj_type=self.text_proj_type,
            text_states_hidden_dim=self.text_states_hidden_dim,
            token_refiner_num_attention_heads=self.token_refiner_num_attention_heads,
            text_order=self.text_order,
            skip_txt_after_last_attn=self.skip_txt_after_last_attn,
            # image/video configs
            use_vae=self.use_vae,
            vae_type=self.vae_type,
            vae_latent_dim=self.vae_latent_dim,
            vae_spatial_downsample_factor=self.vae_spatial_downsample_factor,
            vae_temporal_downsample_factor=self.vae_temporal_downsample_factor,
            vae_precision=self.vae_precision,
            vae_autocast_dtype=self.vae_autocast_dtype,
            patch_size=self.patch_size,
            img_proj_type=self.img_proj_type,
            img_proj_ndim=self.img_proj_ndim,
            patch_embed_hidden_dim=self.patch_embed_hidden_dim,
            use_timestep_token=self.use_timestep_token,
            img_latent_in_channels=self.img_latent_in_channels,
            # audio configs
            use_audio_vae=self.use_audio_vae,
            audio_vae_type=self.audio_vae_type,
            audio_vae_latent_dim=self.audio_vae_latent_dim,
            audio_proj_type=self.audio_proj_type,
        )
        return hf_configs

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "LeoConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = deepcopy(MODEL_ZOO[model_name])
        model_config.update(deepcopy(kwargs))
        return cls(**model_config)


def core_model_config_from_args(args: Namespace, prefix=None) -> dict[str, Any]:
    """ Convert training args to model config dict. """
    model_config = dict()
    # This mapping is for backward compatibility and avoid conflicts with other training backends.
    # E.g., mcore uses rope_type and restrict the choices, therefore we use rope_type_extended in args
    # to allow more choices, and then map it back to rope_type in model config.
    model_to_config_key_mapping = dict(
        rope_type="rope_type_extended",
        mrope_section="rope_dim_list",
        use_timestep_token="add_timestep_token",
    )
    # Inspect all fields in HunyuanMultimodalConfig and add them to model_config if they exist in args
    for model_key in LeoConfig().to_dict():
        if model_key in model_to_config_key_mapping:
            config_key = model_to_config_key_mapping[model_key]
        else:
            config_key = model_key
        if prefix is not None:
            config_key = prefix + config_key
        if hasattr(args, config_key) and getattr(args, config_key) is not None:
            model_config[model_key] = getattr(args, config_key)

    return model_config


# =========================================
#     Predefined Model Configurations
# =========================================

register_model_config(
    name="leo-2-dense-8b",
    num_layers=24,
    hidden_size=3072,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=24,
    num_kv_heads=24,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=8192,
    mlp_bias=False,
    split_gate_and_up=True,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-dense-8b-no-modulation",
    base="leo-2-dense-8b",
    use_modulation=False,
)

register_model_config(
    name="leo-2-dense-3b",
    num_layers=24,
    hidden_size=2304,
    max_position_embeddings=327_600,
    num_attention_heads=18,
    num_kv_heads=18,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=5952,
    mlp_bias=False,
    split_gate_and_up=True,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-dense-3b-txt-branch",
    base="leo-2-dense-3b",  # 与主维完全一致，YAML 仅用于单独绑定 text-branch-model-name
)

register_model_config(
    name="leo-2-dense-8b-repa",
    base="leo-2-dense-8b",
    use_repa=True,
    repa_encoder_type="DINOv3",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1024,
)

register_model_config(
    name="leo-2-dense-37b-skew-v3",
    num_layers=48,
    hidden_size=6144,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=48,
    num_kv_heads=48,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=16384,
    mlp_bias=False,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    modulate_hidden_size=3072,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-dense-37b-skew-v3-txt-branch",
    base="leo-2-dense-37b-skew-v3",
    hidden_size=3072,
    ffn_hidden_size=8192,
)

register_model_config(
    name="leo-2-dense-37b-skew-v3-repa",
    base="leo-2-dense-37b-skew-v3",
    use_repa=True,
    repa_encoder_type="DINOv3",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1024,
)

register_model_config(
    name="leo-2-dense-37b-skew-v3-repa-txt-branch",
    base="leo-2-dense-37b-skew-v3-repa",
    hidden_size=3072,
    ffn_hidden_size=8192,
)


# ---------------------------------- MoE V1-1-Pack-NoTR ----------------------------------

register_model_config(
    name="leo-2-moe-v1-1",
    num_layers=48,
    hidden_size=4096,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=32,
    num_kv_heads=32,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=13824,
    num_experts=64,
    moe_ffn_hidden_size=1536,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=1,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    modulate_hidden_size=1024,
    rope_theta=2000.0,
    mrope_section=[24, 20, 20],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-moe-v1-1-txt-branch",
    base="leo-2-moe-v1-1",
    hidden_size=3072,
    ffn_hidden_size=8192,
    num_experts=0,
)

register_model_config(
    name="leo-2-moe-v1-1-audio-branch",
    base="leo-2-moe-v1-1",
    hidden_size=3072,
    ffn_hidden_size=8192,
    num_experts=0,
    audio_vae_latent_dim=64,
)

register_model_config(
    name="leo-2-moe-v1-1-pack",
    base="leo-2-moe-v1-1",
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
    use_repa=True,
    repa_encoder_type="qwen-3.5-9b",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1152,
)

register_model_config(
    name="leo-2-moe-v1-1-pack-noTR",
    base="leo-2-moe-v1-1",
    text_proj_type="linear",
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
    use_repa=True,
    repa_encoder_type="qwen-3.5-9b",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1152,
)

register_model_config(
    name="leo-2-moe-v1-1-pack-txt-branch",
    base="leo-2-moe-v1-1-pack",
    hidden_size=3072,
    ffn_hidden_size=8192,
    num_experts=0,
    use_modulation=False,
    skip_txt_after_last_attn=True,
)

register_model_config(
    name="leo-2-moe-v1-1-pack-audio-branch",
    base="leo-2-moe-v1-1-pack",
    hidden_size=3072,
    ffn_hidden_size=8192,
    num_experts=0,
    audio_vae_latent_dim=64,
)

# ----------------------------------------------------------------------------------------

register_model_config(
    name="leo-2-moe-v1-2",
    num_layers=32,
    hidden_size=4096,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=32,
    num_kv_heads=32,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=13824,
    num_experts=64,
    moe_ffn_hidden_size=3072,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=8,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    modulate_hidden_size=1024,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-moe-v1-2-txt-branch",
    base="leo-2-moe-v1-2",
    num_experts=0,
)

register_model_config(
    name="leo-2-moe-v1-2-audio-branch",
    base="leo-2-moe-v1-2",
    num_experts=0,
    audio_vae_latent_dim=64,
)

register_model_config(
    name="leo-2-moe-v1-2-pack",
    base="leo-2-moe-v1-2",
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
)

register_model_config(
    name="leo-2-moe-v1-2-pack-txt-branch",
    base="leo-2-moe-v1-2-pack",
    num_experts=0,
)

register_model_config(
    name="leo-2-moe-v1-2-pack-audio-branch",
    base="leo-2-moe-v1-2-pack",
    num_experts=0,
    audio_vae_latent_dim=64,
)

# ------------------------------ MoE V2-1-Pack-NoTR-SplitGateAndUp ----------------------------------

register_model_config(
    name="leo-2-moe-v2-1-base",
    num_layers=56,
    hidden_size=4096,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=64,
    num_kv_heads=64,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    split_gate_and_up=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=13824,
    num_experts=128,
    moe_ffn_hidden_size=2048,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=1,
    use_modulation=True,
    text_proj_type="linear",
    text_states_hidden_dim=4096,
    modulate_hidden_size=1536,
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
    rope_theta=2000.0,
    mrope_section=[24, 20, 20],
    use_scale_rope=False,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-moe-v2-1",
    base="leo-2-moe-v2-1-base",
    # Only used when use_repa=True
    repa_encoder_type="qwen-3.5-9b",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1152,
)

register_model_config(
    name="leo-2-moe-v2-1-txt-branch",
    base="leo-2-moe-v2-1-base",
    num_experts=0,
    use_modulation=False,
    skip_txt_after_last_attn=True,
)

register_model_config(
    name="leo-2-moe-v2-1-audio-branch",
    base="leo-2-moe-v2-1-base",
    hidden_size=3072,
    ffn_hidden_size=8192,
    num_experts=0,
    audio_vae_latent_dim=64,
)

# ----------------------------------------------------------------------------------------

register_model_config(
    name="leo-2-dense-8b-v2-repa",
    num_layers=24,
    hidden_size=3072,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=24,
    num_kv_heads=24,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=8192,
    mlp_bias=False,
    split_gate_and_up=True,
    use_modulation=True,
    text_proj_type="linear",
    text_states_hidden_dim=4096,
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=False,
    vae_latent_dim=48,

    use_repa=True,
    repa_encoder_type="qwen-3.5-9b",
    repa_proj_hidden_size=2048,
    repa_proj_out_size=1152,
)

# For packed training with 3d rope and AV RoPE alignment.
register_model_config(
    name="leo-2-dense-8b-v2",
    num_layers=24,
    hidden_size=3072,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=24,
    num_kv_heads=24,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=8192,
    mlp_bias=False,
    split_gate_and_up=True,
    use_modulation=True,
    text_proj_type="linear",
    text_states_hidden_dim=4096,
    rope_type="3d",
    rope_interleave=False,
    rope_float_position=True,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=False,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-dense-8b-v2-audio-branch",
    base="leo-2-dense-8b-v2",
    audio_vae_latent_dim=64,
)

register_model_config(
    name="leo-2-ci-test",
    base="leo-2-dense-8b-v2",
    num_layers=2,
    hidden_size=32,
    max_position_embeddings=32768,
    num_attention_heads=2,
    num_kv_heads=2,
    ffn_hidden_size=32 * 4,
    text_states_hidden_dim=48,
    rope_theta=2000.0,
    mrope_section=[4, 6, 6],
    vae_latent_dim=8,
)

register_model_config(
    name="leo-2-moe-v1-1-small",
    num_layers=48,
    hidden_size=3072,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=32,
    num_kv_heads=32,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=8192,
    num_experts=64,
    moe_ffn_hidden_size=1024,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=1,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    modulate_hidden_size=1024,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-moe-v1-1-small-txt-branch",
    base="leo-2-moe-v1-1",
    hidden_size=2048,
    ffn_hidden_size=5376,
    num_experts=0,
)

register_model_config(
    name="leo-2-moe-v1-1-small-audio-branch",
    base="leo-2-moe-v1-1",
    hidden_size=2048,
    ffn_hidden_size=5376,
    num_experts=0,
    audio_vae_latent_dim=64,
)

register_model_config(
    name="leo-2-moe-v1-2-small",
    num_layers=32,
    hidden_size=3072,
    max_position_embeddings=327_600,    # (720 * 120 / 16 / 16) * ((361 - 1) / 4 + 1), --> 720p, 15s, 24fps
    num_attention_heads=32,
    num_kv_heads=32,
    attention_head_size=128,
    attention_bias=True,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    qk_norm_type="rms",
    norm_type="layer_f32",
    ffn_hidden_size=8192,
    num_experts=64,
    moe_ffn_hidden_size=2048,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=8,
    use_modulation=True,
    text_proj_type="single_refiner",
    text_states_hidden_dim=4096,
    modulate_hidden_size=1024,
    rope_theta=2000.0,
    mrope_section=[32, 48, 48],
    use_scale_rope=True,
    vae_latent_dim=48,
)

register_model_config(
    name="leo-2-moe-v1-2-small-txt-branch",
    base="leo-2-moe-v1-2",
    num_experts=0,
)

register_model_config(
    name="leo-2-moe-v1-2-small-audio-branch",
    base="leo-2-moe-v1-2",
    num_experts=0,
    audio_vae_latent_dim=64,
)
