import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from typing import Optional
from hymm.models.multimodal.hunyuan_multimodal_config import HunyuanMultimodalConfig


def swap(weight):
    is_dtensor = isinstance(weight, DTensor)
    local_weight = weight.to_local() if is_dtensor else weight

    *leading_dims, out_ch, in_ch = local_weight.size()
    flipped_weight = (
        local_weight
        .view(*leading_dims, 2, out_ch // 2, in_ch)
        .flip(dims=(len(leading_dims),))
        .view(*leading_dims, out_ch, in_ch)
    )

    if is_dtensor:
        weight.to_local().copy_(flipped_weight)
    else:
        weight.copy_(flipped_weight)

class HunyuanMLP(nn.Module):
    def __init__(
            self,
            config: HunyuanMultimodalConfig,
            layer_idx: int,
            is_shared_mlp: bool = False,
            is_moe: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.ffn_hidden_size = config.ffn_hidden_size

        # For expert
        if is_shared_mlp or is_moe:
            self.ffn_hidden_size = (
                config.moe_ffn_hidden_size
                if isinstance(config.moe_ffn_hidden_size, int)
                else config.moe_ffn_hidden_size[layer_idx % config.num_layers]
            )
            if is_shared_mlp:
                num_shared_expert = (
                    config.moe_mixed_mlp
                    if isinstance(config.moe_mixed_mlp, int)
                    else config.moe_mixed_mlp[layer_idx % config.num_layers]
                )
                self.ffn_hidden_size *= num_shared_expert

        if config.split_gate_and_up:
            self.gate_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias, **factory_kwargs)
            self.up_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=config.mlp_bias, **factory_kwargs)
        else:
            self.gate_and_up_proj = nn.Linear(
                config.hidden_size, self.ffn_hidden_size * 2, bias=config.mlp_bias, **factory_kwargs
            )
        self.down_proj = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=config.mlp_bias, **factory_kwargs)
        self.act_fn = config.act_class()

    def forward(self, x):
        if self._config.split_gate_and_up:
            up = self.up_proj(x)
            gate = self.gate_proj(x)
        else:
            gate_and_up_proj = self.gate_and_up_proj(x)
            up, gate = gate_and_up_proj.chunk(2, dim=-1)
        out = self.down_proj(up * self.act_fn(gate))
        return out

    @torch.no_grad()
    def swap_gate_and_up_weights(self):
        """
        For gate_and_up layers, the linear weights have two parts: gate and up.
        Torch/FlashInfer use [up, gate] layout, while PTMv2/TE use [gate, up] layout.
        This function swaps the two parts in-place.
        """
        assert not self._config.split_gate_and_up, \
            "swap_gate_and_up_weights only works for combined gate_and_up_proj."
        swap(self.gate_and_up_proj.weight)
