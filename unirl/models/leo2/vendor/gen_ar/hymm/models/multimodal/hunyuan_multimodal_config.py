# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company.  All rights reserved.
"""HunYuan model configuration"""

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Literal
from copy import deepcopy

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
class HunyuanMultimodalConfig(TransformerConfig):
    # common
    name: str = ""

    # ===============================
    #     Language Model Config
    # ===============================
    # ---- model architecture ----
    # basic
    vocab_size: int = 0
    num_layers: int = 0
    hidden_size: int = 0
    max_position_embeddings: int = 0
    tie_word_embeddings: bool = False
    # attention module
    num_attention_heads: int = 0
    num_kv_heads: int | None = None  # for multi-query attention, default to num_attention_heads
    attention_head_size: int | None = None  # default to hidden_size // num_attention_heads
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_qk_norm: bool = False
    pre_qk_norm: bool = False # whether to apply QK norm before applying RoPE, QWenVL 3.0 use qknorm before applying rope
    split_qkv: bool = False
    # norm module
    norm_type: str = "rms"
    # mlp module
    ffn_hidden_size: int | None = None  # default to 4 * hidden_size
    hidden_act: Literal["silu", "gelu"] = "silu"
    mlp_bias: bool = False
    split_gate_and_up: bool = False
    # moe module
    moe_impl: str = "hunyuan"
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
    # expert bias for load balancing
    moe_enable_router_expert_bias: bool = False  # Enable expert bias for router load balancing
    moe_expert_bias_update_rate: float = 0.0  # Update rate for expert bias
    moe_enable_expert_bias_zero_mean_update: bool = False  # Enable zero-mean update for expert bias
    # Only effective when moe_impl == "ep_moe". "ep_moe": separate gate/up/down; "flashinfer": fused gate+up.
    ep_moe_weight_format: str = "ep_moe"
    # ---- numerical related ----
    # initialization
    init_std: float = 0.02
    norm_eps: float = 1e-5
    # rope
    rope_type: str = "default"
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    xdrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    mrope_section: list[int] | None = None
    apply_rope_in_fp32: bool = True

    # ===============================
    #     Multimodal Model Config
    # ===============================
    # ---- model architecture ----
    # vae
    use_vae: bool = False
    vae_type: str = "16x16x4-32c-hy-image"
    vae_latent_dim: int = 32
    vae_downsample_factor: int = 16
    vae_precision: str = "fp32"
    vae_autocast_dtype: str = "fp16"
    # vae image projector
    patch_size: int = 1
    img_proj_type: str = "conv"
    img_proj_ndim: int = 2
    patch_embed_hidden_dim: int = 1024
    use_timestep_token: bool = True
    use_timestep_r_token: bool = False
    use_guidance_token: bool = False
    # vit
    use_vit: bool = False
    vit_type: str = "siglip2-so400m-patch16-naflex"
    vit_config: dict[str, Any] = field(default_factory=dict)
    # vit image projector
    use_vit_aligner: bool = False
    vit_aligner_type: str = "mlp_gelu"
    vit_aligner_config: dict[str, Any] = field(default_factory=dict)

    # MoT generative config
    # num_attention_head, attention_head_dim must be kept same as the understanding model
    # num_kv_heads can be different from the understanding model, but it has not been implemented yet
    use_mot: bool = False
    hidden_size_mot_gen: int = None
    num_kv_heads_mot_gen: int = None
    ffn_hidden_size_mot_gen: int = None
    num_experts_mot_gen: int = None
    moe_topk_mot_gen: int = None
    moe_impl_mot_gen: str = None

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

        # Compatibility with potential parent classes
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def to_hf_config(self) -> dict[str, Any]:
        # Convert to HuggingFace PretrainedConfig by extending HunYuanMoEV2Config
        hf_configs = dict(
            vocab_size=self.vocab_size,
            org_vocab_size=self.vocab_size,
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
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.init_std,
            rms_norm_eps=self.norm_eps,
            rope_type=self.rope_type,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            xdrope_section=self.xdrope_section,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            attention_dropout=self.attention_dropout,
            use_qk_norm=self.use_qk_norm,
            use_rotary_pos_emb=True,
            norm_type=self.norm_type,
            # moe configs
            num_experts=self.num_experts,
            use_mixed_mlp_moe=self.moe_mixed_mlp > 0,
            num_shared_expert=self.moe_mixed_mlp,
            moe_topk=self.moe_topk,
            moe_drop_tokens=self.moe_drop_tokens,
            moe_random_routing_dropped_token=self.moe_random_routing_dropped_tokens,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            moe_layer_num_skipped=self.moe_layer_num_skipped,
            use_mot=self.use_mot,
            # multimodal configs
            use_vae=self.use_vae,
            vae_type=self.vae_type,
            vae_latent_dim=self.vae_latent_dim,
            vae_precision=self.vae_precision,
            vae_autocast_dtype=self.vae_autocast_dtype,
            vae_downsample_factor=self.vae_downsample_factor,
            patch_size=self.patch_size,
            img_proj_type=self.img_proj_type,
            img_proj_ndim=self.img_proj_ndim,
            patch_embed_hidden_dim=self.patch_embed_hidden_dim,
            use_timestep_token=self.use_timestep_token,
            use_timestep_r_token=self.use_timestep_r_token,
            use_guidance_token=self.use_guidance_token,
            use_vit=self.use_vit,
            vit_type=self.vit_type,
            vit_config=self.vit_config,
            use_vit_aligner=self.use_vit_aligner,
            vit_aligner_type=self.vit_aligner_type,
            vit_aligner_config=self.vit_aligner_config,
        )
        return hf_configs

    def from_hf_config(self, hf_config: dict[str, Any], **kwargs) -> "HunyuanMultimodalConfig":
        config = {**hf_config, **kwargs}

        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.ffn_hidden_size = config["intermediate_size"]
        self.moe_ffn_hidden_size = config["moe_intermediate_size"]
        self.num_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.attention_head_size = config["attention_head_dim"]
        self.hidden_act = config["hidden_act"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.init_std = config["initializer_range"]
        self.norm_eps = config["rms_norm_eps"]
        self.rope_type = config.get("rope_type", "2d")
        self.rope_theta = config["rope_theta"]
        self.rope_scaling = config["rope_scaling"]
        self.xdrope_section = config["xdrope_section"]
        self.attention_bias = config["attention_bias"]
        self.mlp_bias = config["mlp_bias"]
        self.attention_dropout = config["attention_dropout"]
        self.use_qk_norm = config["use_qk_norm"]
        self.use_rotary_pos_emb = config.get("use_rotary_pos_emb", True)
        self.norm_type = config["norm_type"]
        # moe configs
        self.num_experts = config["num_experts"]
        self.moe_mixed_mlp = config["num_shared_expert"] if config["use_mixed_mlp_moe"] else 0
        self.moe_topk = config["moe_topk"]
        self.moe_drop_tokens = config["moe_drop_tokens"]
        self.moe_random_routing_dropped_token = config.get("moe_random_routing_dropped_tokens", False)
        self.norm_topk_prob = config["norm_topk_prob"]
        self.routed_scaling_factor = config["routed_scaling_factor"]
        self.moe_layer_num_skipped = config["moe_layer_num_skipped"]
        self.moe_score_func = config.get("moe_score_func", "softmax")
        self.moe_seq_aux_loss = config.get("moe_seq_aux_loss", False)
        self.moe_enable_router_expert_bias = config.get("moe_enable_router_expert_bias", False)
        self.moe_expert_bias_update_rate = config.get("moe_expert_bias_update_rate", 0.0)
        self.moe_enable_expert_bias_zero_mean_update = config.get("moe_enable_expert_bias_zero_mean_update", False)
        self.use_mot = config["use_mot"]
        # multimodal configs
        multimodal_keys = [
            "use_vae", "vae_type", "vae_latent_dim", "vae_precision", "vae_autocast_dtype",
            "patch_size", "img_proj_type", "img_proj_ndim", "patch_embed_hidden_dim", "use_timestep_token",
            "use_timestep_r_token", "use_guidance_token",
            "use_vit", "vit_type", "vit_config", "use_vit_aligner", "vit_aligner_type", "vit_aligner_config",
            "use_mot", "use_dense_mot_gen",
        ]
        for key in multimodal_keys:
            if key in config:
                setattr(self, key, config[key])
        if "vae" in config:
            self.use_vae = True
        if "vit" in config:
            self.use_vit = True
            if self.vit_type is None:
                self.vit_type = "siglip2-so400m-patch16-naflex"

        return self

    @classmethod
    def from_name(cls, model_name: str, **kwargs) -> "HunyuanMultimodalConfig":
        if model_name not in MODEL_ZOO:
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO. Valid models: {list(MODEL_ZOO.keys())}")
        model_config = deepcopy(MODEL_ZOO[model_name])
        model_config.update(deepcopy(kwargs))
        return cls(**model_config)

    @property
    def norm_class(self):
        if self.norm_type == "rms":
            from ..basic.norm_layers import HunyuanRMSNorm
            return HunyuanRMSNorm
        else:
            raise NotImplementedError(f"Unsupported norm_type: {self.norm_type}")

    @property
    def act_class(self):
        if self.hidden_act == "silu":
            from torch.nn import SiLU
            return SiLU
        elif self.hidden_act == "gelu":
            from torch.nn import GELU
            return GELU
        else:
            raise NotImplementedError(f"Unsupported hidden_act: {self.hidden_act}")

    def to_mot_gen_config(self) -> "HunyuanMultimodalConfig":
        assert self.use_mot, "use_mot must be True when using mot_gen config"

        config_mot_gen = deepcopy(self)
        if self.hidden_size_mot_gen:
            config_mot_gen.hidden_size = self.hidden_size_mot_gen
        if self.num_kv_heads_mot_gen:
            config_mot_gen.num_kv_heads = self.num_kv_heads_mot_gen
        if self.ffn_hidden_size_mot_gen:
            config_mot_gen.ffn_hidden_size = self.ffn_hidden_size_mot_gen
        if self.num_experts_mot_gen:
            config_mot_gen.num_experts = self.num_experts_mot_gen
        if self.moe_impl_mot_gen:
            config_mot_gen.moe_impl = self.moe_impl_mot_gen
        if self.moe_topk_mot_gen:
            config_mot_gen.moe_topk = self.moe_topk_mot_gen
        
        # mot constraint check
        assert self.attention_head_size == config_mot_gen.attention_head_size, "attention_head_size must be the same as the understanding model"
        assert self.num_attention_heads == config_mot_gen.num_attention_heads, "num_attention_heads must be the same as the understanding model"
        assert self.num_kv_heads == config_mot_gen.num_kv_heads, "num_kv_heads must be the same as the understanding model"
        
        return config_mot_gen

def core_model_config_from_args(args: Namespace) -> dict[str, Any]:
    """ Convert training args to model config dict. """
    model_config = dict(
        use_timestep_token=args.add_timestep_token,
        use_timestep_r_token=args.add_timestep_r_token,
        use_guidance_token=args.add_guidance_token,
    )
    model_keys = [
        "vocab_size", "num_layers", "hidden_size", "max_position_embeddings",
        "num_attention_heads", "split_qkv", "ffn_hidden_size",
        "split_gate_and_up",
        "moe_impl", "moe_drop_tokens", "gate_impl", "num_experts",
        "ep_moe_weight_format",
        "use_vae", "vae_type", "vae_latent_dim", "vae_precision", "vae_autocast_dtype",
        "patch_size", "img_proj_type", "img_proj_ndim", "patch_embed_hidden_dim",
        "use_vit", "vit_type", "vit_config",
        "moe_aux_loss",
        dict(model_key="rope_type", config_key="rope_type_extended"),
        "rope_theta", "rope_scaling", "xdrope_section",
        dict(model_key="mrope_section", config_key="rope_dim_list"),
        "use_mot", "use_dense_mot_gen",
        "moe_enable_router_expert_bias", "moe_expert_bias_update_rate", "moe_score_func",
        "moe_seq_aux_loss", "moe_enable_expert_bias_zero_mean_update", 
    ]
    for key_mapping in model_keys:
        if isinstance(key_mapping, dict):
            model_key = key_mapping["model_key"]
            config_key = key_mapping["config_key"]
        else:
            model_key = config_key = key_mapping
        if hasattr(args, config_key) and getattr(args, config_key) is not None:
            value = getattr(args, config_key)
            if isinstance(value, dict):
                model_config[model_key] = dict(value)
            else:
                model_config[model_key] = value
    return model_config


# =========================================
#     Predefined Model Configurations
# =========================================

register_model_config(
    name="hunyuan-dense-3b",
    vocab_size=133120,
    num_layers=32,
    hidden_size=2560,
    max_position_embeddings=23040,
    num_attention_heads=20,
    num_kv_heads=10,
    attention_bias=True,
    use_qk_norm=True,
    ffn_hidden_size=9216,
    mlp_bias=True,
)

register_model_config(
    name="hunyuan-dense-3b-gemini",
    base="hunyuan-dense-3b",
    vocab_size=133120,
    use_vae=True,
    use_timestep_token=True,
    use_vit=True,
    vit_config=dict(vision_use_head=False),
    use_vit_aligner=True,
    vit_aligner_config=dict(depth=2),
)

register_model_config(
    name="hunyuan-dense-7b",
    vocab_size=128167,
    num_layers=32,
    hidden_size=4096,
    max_position_embeddings=23040,
    tie_word_embeddings=True,
    num_attention_heads=32,
    num_kv_heads=8,
    use_qk_norm=True,
    ffn_hidden_size=14336,
)

register_model_config(
    name="hunyuan-dense-7b-gemini",
    base="hunyuan-dense-7b",
    vocab_size=133120,
    tie_word_embeddings=False,
    use_vae=True,
    use_timestep_token=True,
    use_vit=True,
    vit_config=dict(vision_use_head=False),
    use_vit_aligner=True,
    vit_aligner_config=dict(depth=2),
)

register_model_config(
    name="hunyuan-moe-a13b",
    vocab_size=128167,
    num_layers=32,
    hidden_size=4096,
    max_position_embeddings=23040,
    tie_word_embeddings=True,
    num_attention_heads=32,
    num_kv_heads=8,
    use_qk_norm=True,
    ffn_hidden_size=3072,
    num_experts=64,
    moe_ffn_hidden_size=3072,
    moe_mixed_mlp=1,
    moe_topk=8,
)

register_model_config(
    name="hunyuan-moe-a13b-gemini",
    base="hunyuan-moe-a13b",
    vocab_size=133120,
    tie_word_embeddings=False,
    use_vae=True,
    use_timestep_token=True,
    use_vit=True,
    vit_config=dict(vision_use_head=False),
    use_vit_aligner=True,
    vit_aligner_config=dict(depth=2),
)

register_model_config(
    name="hunyuan-moe-a3b",
    vocab_size=120818,
    num_layers=48,
    hidden_size=2048,
    tie_word_embeddings=True,
    max_position_embeddings=23040,
    num_attention_heads=32,
    num_kv_heads=4,
    attention_head_size=128,
    use_qk_norm=True,
    split_qkv=True,
    ffn_hidden_size=6912,
    split_gate_and_up=True,
    num_experts=128,
    moe_ffn_hidden_size=768,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=1,
)

register_model_config(
    name="hunyuan-moe-a3b-vlm",
    base="hunyuan-moe-a3b",
    rope_type="xdrope",
    rope_scaling=1000.0,
    xdrope_section=[16, 16, 16, 16],
    use_vit=True,
    vit_type="anyres-vit-for-a3b",
    vit_config=dict(output_channels=2048),
)


register_model_config(
    name="hunyuan-moe-a3b-gemini",
    base="hunyuan-moe-a3b",
    vocab_size=126309,
    tie_word_embeddings=False,
    use_vae=True,
    use_timestep_token=True,
    use_vit=True,
    vit_config=dict(vision_use_head=False),
    use_vit_aligner=True,
    vit_aligner_config=dict(depth=2),
)

register_model_config(
    name="hunyuan-moe-a3b-vlm-gemini",
    base="hunyuan-moe-a3b-vlm",
    vocab_size=126309,
    tie_word_embeddings=False,
    use_vae=True,
    use_timestep_token=True,
)

register_model_config(
    name="hunyuan-moe-a3b-vlm-gemini-mot-gen-a3b",
    base="hunyuan-moe-a3b-vlm-gemini",
    use_mot=True,
)

register_model_config(
    name="hunyuan-moe-a3b-vlm-gemini-mot-gen-a5b",
    base="hunyuan-moe-a3b-vlm-gemini",
    hidden_size_mot_gen=3200,
    use_mot=True,
)

register_model_config(
    name="hunyuan-moe-a3b-vlm-gemini-video",
    base="hunyuan-moe-a3b-vlm-gemini",
    vocab_size=126569,
    img_proj_ndim=3,  # image and video share the same conv projector
)

register_model_config(
    name="qwen-vl-30b-a3b-instruct",
    vocab_size=151936,
    num_layers=48,
    hidden_size=2048,
    max_position_embeddings=262144,
    tie_word_embeddings=False,
    num_attention_heads=32,
    num_kv_heads=4,
    attention_head_size=128,
    attention_bias=False,
    attention_dropout=0.0,
    split_qkv=True,
    use_qk_norm=True,
    pre_qk_norm=True,
    ffn_hidden_size=6144,
    hidden_act="silu",
    num_experts=128,
    moe_ffn_hidden_size=768,
    moe_impl="qwen3",
    norm_topk_prob=True,
    moe_mixed_mlp=0,
    moe_topk=8,
    init_std=0.02,
    norm_eps=1e-6,
    rope_type="interleaved_mrope",
    rope_theta=5000000,
    apply_rope_in_fp32=False,
    mrope_interleaved=True,
    mrope_section=[24, 20, 20],
    use_vit=True,
    vit_type="qwen3vl-vit-for-30b-a3b",
    vit_config=dict(spatial_merge_size=2),
)

register_model_config(
    name="hunyuan-dense-3b-mot-gen",
    base="hunyuan-moe-a3b-vlm-gemini",
    num_experts=0,
)

register_model_config(
    name="hunyuan3-moe-a3b",
    vocab_size=120818,
    num_layers=48,
    hidden_size=2048,
    tie_word_embeddings=False,
    max_position_embeddings=32768,
    num_attention_heads=32,
    num_kv_heads=4,
    attention_head_size=128,
    use_qk_norm=True,
    pre_qk_norm=True,
    split_qkv=True,
    ffn_hidden_size=6912,
    split_gate_and_up=True,
    num_experts=128,
    moe_ffn_hidden_size=768,
    moe_mixed_mlp=1,
    moe_topk=8,
    moe_layer_num_skipped=1,
    rope_theta=11158840,
    rope_scaling=1.0,
    # apply_rope_in_fp32=False,
)

register_model_config(
    name="hunyuan3-moe-a3b-vlm",
    base="hunyuan3-moe-a3b",
    rope_type="interleaved_mrope",
    mrope_section=[24, 20, 20],
    use_vit=True,
    vit_type="anyres-vit-for-hy3-a3b",
    vit_config=dict(output_channels=2048),
    moe_enable_router_expert_bias=True,
    moe_expert_bias_update_rate=0.001,
    moe_enable_expert_bias_zero_mean_update=True,
    moe_score_func="sigmoid",
    moe_seq_aux_loss=True,  # compute aux loss batch-wise(False) or sequence-wise(True)
    routed_scaling_factor=2.826
)


register_model_config(
    name="hunyuan3-moe-a3b-gemini",
    base="hunyuan3-moe-a3b-vlm",
    vocab_size=126309,
    rope_type="interleaved_mrope",
    moe_enable_router_expert_bias=True,
    moe_expert_bias_update_rate=0.001,
    moe_enable_expert_bias_zero_mean_update=True,
    moe_score_func="sigmoid",
    moe_seq_aux_loss=True,  # compute aux loss batch-wise(False) or sequence-wise(True)
    routed_scaling_factor=2.826,
    tie_word_embeddings=False,
    use_vae=True,
    use_timestep_token=True,
)

register_model_config(
    name="hunyuan3-moe-a3b-gemini-mot",
    base="hunyuan3-moe-a3b-gemini",
    use_mot=True,
)

register_model_config(
    name="hunyuan3-moe-a3b-gemini-mot-gen-a5b",
    base="hunyuan3-moe-a3b-gemini",
    hidden_size_mot_gen=3200,
    use_mot=True,
)

register_model_config(
    name="hunyuan3-moe-a3b-gemini-mot-gen-a10b",
    base="hunyuan3-moe-a3b-gemini",
    hidden_size_mot_gen=3200,
    moe_topk_mot_gen=16,
    use_mot=True,
)

register_model_config(
    name="multimodal-ci-test",
    vocab_size = 128,
    tie_word_embeddings = False,
    max_position_embeddings = 2048,

    num_layers = 2,
    hidden_size = 32,
    num_attention_heads = 4,
    num_kv_heads = 4, # num_kv_heads == num_attention_heads, no gqa
    attention_head_size = 32 // 4, # hidden_size // num_attention_heads
    use_qk_norm = True,
    ffn_hidden_size = 4 * 32, # 4 * hidden_size

    use_vit = True,
    use_vit_aligner = True,
    vit_type = "anyres-vit-ci-test",
    vit_config=dict(output_channels=32),  # hidden_size

    use_vae = True,
    img_proj_type="linear",
    vae_latent_dim=8,
    use_timestep_token = True,
)