# coding=utf-8
# Copyright (C) 2026 THL A29 Limited, a Tencent company.  All rights reserved.
"""HunYuan model configuration"""

from dataclasses import dataclass
from typing import Any, Literal

from hymm.utils.states import DataClassMixin


@dataclass
class TransformerConfig(DataClassMixin):
    # common
    name: str = ""

    # ===============================
    #     Transformer Config
    # ===============================
    # ---- model architecture ----
    # basic
    num_layers: int = 0
    hidden_size: int = 0
    max_position_embeddings: int = 0
    # attention module
    attn_impl: str = "sdpa"
    inference_attn_impl: str | None = None
    num_attention_heads: int = 0
    num_kv_heads: int | None = None  # for multi-query attention, default to num_attention_heads
    attention_head_size: int | None = None  # default to hidden_size // num_attention_heads
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_qk_norm: bool = False
    pre_qk_norm: bool = False # whether to apply QK norm before applying RoPE
    qk_norm_type: str | None = None     # if not specified, default to norm_type.
    split_qkv: bool = False
    # norm module
    norm_type: str = "rms"
    norm_elementwise_affine: bool = True
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
    moe_fused_expert: bool = True
    # Only effective when moe_impl == "ep_moe". "ep_moe": separate gate/up/down; "flashinfer": fused gate+up.
    ep_moe_weight_format: str = "ep_moe"
    moe_enable_deepep: bool = False
    moe_drop_token_enabled: bool = False
    # MoT generative config
    use_mot: bool = False
    use_dense_mot_gen: bool = False
    # ---- numerical related ----
    # initialization
    init_std: float = 0.02
    norm_eps: float = 1e-6
    # rope
    rope_type: str = "default"
    rope_interleave: bool = False
    rope_theta: float = 2000.0
    rope_scaling: float = 1.0
    xdrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    mrope_section: list[int] | None = None
    apply_rope_in_fp32: bool = True

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads

        if self.attention_head_size is None:
            self.attention_head_size = self.hidden_size // self.num_attention_heads

        if self.qk_norm_type is None:
            self.qk_norm_type = self.norm_type

        # Compatibility with potential parent classes
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def to_hf_config(self) -> dict[str, Any]:
        # Convert to HuggingFace PretrainedConfig by extending HunYuanMoEV2Config
        return self.to_dict()

    @staticmethod
    def get_norm_class(norm_type):
        if norm_type == "rms":
            from .norm_layers import HunyuanRMSNorm
            return HunyuanRMSNorm
        elif norm_type == "layer_f32":
            from .norm_layers import LayerNormF32
            return LayerNormF32
        else:
            raise NotImplementedError(f"Unsupported norm_type: {norm_type}")

    @property
    def norm_class(self):
        return self.get_norm_class(self.norm_type)

    @property
    def qk_norm_class(self):
        return self.get_norm_class(self.qk_norm_type)

    def get_norm_kwargs(self, norm_type):
        if norm_type == "rms":
            norm_kwargs = dict(eps=self.norm_eps)
        elif norm_type == "layer_f32":
            norm_kwargs = dict(eps=self.norm_eps, elementwise_affine=self.norm_elementwise_affine)
        else:
            raise NotImplementedError(f"Unsupported norm_type: {norm_type}")
        return norm_kwargs

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
