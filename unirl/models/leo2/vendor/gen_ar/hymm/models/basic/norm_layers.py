import numbers

import torch
import torch.nn as nn


class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        """
        HunyuanRMSNorm is equivalent to T5LayerNorm
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LayerNormF32(nn.Module):
    def __init__(
            self,
            normalized_shape,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            bias: bool = True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.enable_bias = bias
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        orig_type = inputs.dtype

        with torch.autocast(device_type='cuda', enabled=False):
            normed = torch.nn.functional.layer_norm(
                inputs.float(),
                self.normalized_shape,
                self.weight.float() if self.elementwise_affine else None,
                self.bias.float() if self.elementwise_affine else None,
                self.eps,
            ).to(orig_type)

        return normed

    def extra_repr(self) -> str:
        return (f"{self.normalized_shape}, elementwise_affine={self.elementwise_affine}, "
                f"bias={self.enable_bias if self.elementwise_affine else False}")
