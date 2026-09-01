import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Tuple
from diffusers.utils.torch_utils import randn_tensor

class DiagonalGaussianDistributionCompression(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2 # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1 # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            reduce_dim = list(range(1, self.mean.ndim))
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dim,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dim,
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class CompressionNet(nn.Module):
    config = {
            "64":{
                "in_channels": 1152,
                "out_channels": 64,
                "path": "/apdcephfs_nj10/share_301739632/noaltian/20250405_compression_ckpt/compression_64/compression_64.pt",
            },
            "128":{
                "in_channels": 1152,
                "out_channels": 128,
                "path": "/apdcephfs_nj10/share_301739632/noaltian/20250405_compression_ckpt/compression_128/compression_128.pt",
            },
            "64-gauss":{
                "in_channels": 1152,
                "out_channels": 64,
                "path": "/apdcephfs_nj10/share_301739632/noaltian/20250405_compression_ckpt/compression_gaussian_64/compression_gaussian_64.pt",
            },
        }
    def __init__(self, type="64", device=None, dtype=None):
        self.type = type
        factory_kwargs = {'device': device, 'dtype': dtype}

        assert type in CompressionNet.config, f"CompressionNet type {type} not found in config"
        in_channels = CompressionNet.config[type]["in_channels"]
        out_channels = CompressionNet.config[type]["out_channels"]
        path = CompressionNet.config[type]["path"]
        super().__init__()
        self.linear_down_A = nn.Linear(in_features=in_channels, out_features=512, bias=True, **factory_kwargs)
        self.act_down_A = nn.SiLU()
        self.linear_down_B = nn.Linear(in_features=512, out_features=out_channels*2 if "gauss" in type else out_channels, bias=True, **factory_kwargs)
        self.act_down_B = nn.SiLU()
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = state_dict["module"]
        self.load_state_dict(state_dict)

    def forward(self, caption):
        hidden_states = self.linear_down_A(caption)
        hidden_states = self.act_down_A(hidden_states)
        hidden_states = self.linear_down_B(hidden_states)

        if "gauss" in self.type:
            posterior = DiagonalGaussianDistributionCompression(hidden_states)
            hidden_states = posterior.sample()
        else:
            hidden_states = self.act_down_B(hidden_states)

        return hidden_states


if __name__ == "__main__":
    model = CompressionNet(type="64-gauss")
    hidden_states = torch.randn(1, 256, 1152)
    hidden_states = model(hidden_states)
    print(hidden_states.shape)
