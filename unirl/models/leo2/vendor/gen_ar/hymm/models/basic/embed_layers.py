import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import flash_attn_no_pad
from .initializers import normal_weight_reset_parameters
from .model_config import TransformerConfig


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self,
                 hidden_size,
                 act_layer=nn.GELU,
                 frequency_embedding_size=256,
                 max_period=10000,
                 out_size=None,
                 dtype=None,
                 device=None,
                 config=None,
                 ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.init_std = config.init_std if config is not None and hasattr(config, 'init_std') else 0.02
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=self.init_std)
        nn.init.normal_(self.mlp[2].weight, std=self.init_std)

    def prepare_reset_parameters(self):
        self.mlp[0].reset_parameters = normal_weight_reset_parameters(std=self.init_std).__get__(self.mlp[0])
        self.mlp[2].reset_parameters = normal_weight_reset_parameters(std=self.init_std).__get__(self.mlp[2])

    def forward(self, t):
        is_list_input = isinstance(t, list)
        if is_list_input: # For sequence pack situation.
            assert len(t) == 1, f"Only single timestep input is supported, but got {t}."
            t = t[0]

        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)

        if is_list_input:
            t_emb = [t_emb]

        return t_emb


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class AudioProjection(nn.Module):
    """
    Projects audio embeddings.
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, audio_latents, t):
        _, _, *token_sizes = audio_latents.shape
        audio_latents = audio_latents.transpose(1, 2)  # (B, C, L) -> (B, L, C)
        hidden_states = self.linear_1(audio_latents)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states, *token_sizes


if __name__ == "__main__":
    hidden_dim = 128
    t_e = TimestepEmbedder(hidden_dim)
    inp = torch.tensor([[1., 2.], [6., 7.], [3., 4.]]).long()
    print(inp.shape)
    output = t_e(inp.reshape(-1)).reshape(inp.shape[0], -1 , hidden_dim)
    print(output, output.shape)