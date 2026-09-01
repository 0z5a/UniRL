from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .attention import flash_attn_no_pad
from .embed_layers import TimestepEmbedder, TextProjection
from .initializers import zero_reset_parameters
from ..diffusion.leo_config import LeoConfig
from ..diffusion.mlp_layers import MLP
from ..diffusion.modulate_layers import apply_gate_fp32


def attention(q, k, v, drop_rate=0, attn_mask=None, causal=False, deterministic=False):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None,
                          deterministic=deterministic)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
            config: LeoConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = config.token_refiner_num_attention_heads
        attention_hidden_size = self.heads_num * config.attention_head_size
        self.deterministic = False

        if config.norm_type == "layer_f32":
            self.norm1 = config.norm_class(config.hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        else:
            self.norm1 = config.norm_class(config.hidden_size, **config.get_norm_kwargs(config.norm_type), **factory_kwargs)
        self.self_attn_qkv = nn.Linear(config.hidden_size, attention_hidden_size * 3, bias=True, **factory_kwargs)
        self.self_attn_proj = nn.Linear(attention_hidden_size, config.hidden_size, bias=True, **factory_kwargs)

        if config.norm_type == "layer_f32":
            self.norm2 = config.norm_class(config.hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        else:
            self.norm2 = config.norm_class(config.hidden_size, **config.get_norm_kwargs(config.norm_type), **factory_kwargs)
        self.mlp = MLP(
            in_channels=config.hidden_size,
            hidden_channels=config.hidden_size * 4,
            act_layer=config.act_class,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            config.act_class(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True, **factory_kwargs),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def prepare_reset_parameters(self):
        self.adaLN_modulation[1].reset_parameters = zero_reset_parameters.__get__(self.adaLN_modulation[1])

    def enable_deterministic(self) -> None:
        self.deterministic = True

    def disable_deterministic(self) -> None:
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Self-Attention
        attn = attention(q, k, v, attn_mask=attn_mask, deterministic=self.deterministic)
        x = x + apply_gate_fp32(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate_fp32(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(nn.Module):
    def __init__(
            self,
            depth: int,
            config: LeoConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(config=config, **factory_kwargs)
                for _ in range(depth)
            ]
        )

    def enable_deterministic(self) -> None:
        self.deterministic = True
        for layer in self.blocks:
            layer.enable_deterministic()

    def disable_deterministic(self) -> None:
        self.deterministic = False
        for layer in self.blocks:
            layer.disable_deterministic()

    def forward(
        self, x: torch.Tensor, c: torch.LongTensor, mask: Optional[torch.Tensor] = None,
    ):
        mask = mask.clone().bool()
        # avoid attention weight become NaN
        mask[:, 0] = True
        for block in self.blocks:
            x = block(x, c, mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """

    def __init__(
        self,
        depth,
        txt_config: LeoConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.deterministic = False
        self._txt_config = txt_config

        self.input_embedder = nn.Linear(
            txt_config.text_states_hidden_dim, txt_config.hidden_size, bias=True, **factory_kwargs
        )

        if self._txt_config.use_modulation:
            # Build timestep embedding layer
            self.t_embedder = TimestepEmbedder(txt_config.hidden_size, act_layer=txt_config.act_class, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            txt_config.text_states_hidden_dim, txt_config.hidden_size, act_layer=txt_config.act_class, **factory_kwargs
        )

        self.individual_token_refiner = IndividualTokenRefiner(depth=depth, config=txt_config, **factory_kwargs)

    def enable_deterministic(self) -> None:
        self.deterministic = True
        self.individual_token_refiner.enable_deterministic()

    def disable_deterministic(self) -> None:
        self.deterministic = False
        self.individual_token_refiner.disable_deterministic()

    def forward(
            self,
            x: torch.Tensor,
            t: torch.LongTensor | list[torch.LongTensor],
            mask: Optional[torch.LongTensor] = None,
    ):

        if self._txt_config.use_modulation:
            timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)

        context_aware_representations = self.c_embedder(context_aware_representations)
        if self._txt_config.use_modulation:
            if isinstance(timestep_aware_representations, list):
                assert len(timestep_aware_representations) == 1, \
                    (f"In packing mode, len(timestep_aware_representations) should be 1, "
                     f"but got {len(timestep_aware_representations)}")
                c = timestep_aware_representations[0] + context_aware_representations
            else:
                c = timestep_aware_representations + context_aware_representations
        else:
            c = context_aware_representations

        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c, mask)

        return x
