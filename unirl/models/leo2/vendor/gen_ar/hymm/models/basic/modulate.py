from typing import Callable

import torch
import torch.nn as nn

from .initializers import zero_reset_parameters


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
            self,
            hidden_size: int,
            factor: int,
            act_layer: Callable,
            output_hidden_size: int = None,
            dtype=None,
            device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        if output_hidden_size is None:
            output_hidden_size = hidden_size
        self.linear = nn.Linear(
            hidden_size, factor * output_hidden_size, bias=True, **factory_kwargs
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def prepare_reset_parameters(self):
        self.linear.reset_parameters = zero_reset_parameters.__get__(self.linear)

    def forward(self, x: torch.Tensor, is_token_replace=False, token_replace_vec=None) -> torch.Tensor:

        x_out = self.linear(self.act(x))

        if is_token_replace:
            x_token_replace_out = self.linear(self.act(token_replace_vec))
            return x_out, x_token_replace_out
        else:
            return x_out


@torch.autocast("cuda", dtype=torch.float32)
def modulate(x, shift=None, scale=None, unsqueeze_dim=1):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.
        unsqueeze_dim: dim to broadcast shift and scale (1 for [B, S, H], 0 for [S, B, H]).

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    input_dtype = x.dtype
    if scale is None and shift is None:
        output = x
    elif shift is None:
        output = x.float() * (1 + scale.unsqueeze(unsqueeze_dim))
    elif scale is None:
        output = x.float() + shift.unsqueeze(unsqueeze_dim)
    elif shift.ndim == x.ndim:
        assert scale.ndim == x.ndim, "shift and scale should have the same shape when shift is not broadcastable"
        output = x.float() * (1 + scale) + shift
    else:
        output = x.float() * (1 + scale.unsqueeze(unsqueeze_dim)) + shift.unsqueeze(unsqueeze_dim)
    return output.to(input_dtype)


@torch.autocast("cuda", dtype=torch.float32)
def apply_gate(x, gate=None, tanh=False, unsqueeze_dim=1):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.
        unsqueeze_dim: dim to broadcast gate (1 for [B, S, H], 0 for [S, B, H]).

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        if gate.ndim == x.ndim:
            return x.float() * gate.tanh()
        return x.float() * gate.unsqueeze(unsqueeze_dim).tanh()
    else:
        if gate.ndim == x.ndim:
            return x.float() * gate
        return x.float() * gate.unsqueeze(unsqueeze_dim)
