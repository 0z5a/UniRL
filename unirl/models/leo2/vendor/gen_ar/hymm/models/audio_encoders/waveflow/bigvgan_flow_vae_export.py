# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import json
import math
from pathlib import Path
from typing import Optional, Union, Dict

import torch
from torch import sin, pow
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .alias_free_activation.torch.act import Activation1d as TorchActivation1d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x



class Conv1d_S(nn.Module):
    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        norm_type="weight_norm",
        init_type=None,
    ):

        super(Conv1d_S, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        pad = dilation * (kernel_size - 1) // 2

        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
        )
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)

        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = spectral_norm(self.layer)

    def forward(self, inputs):
        return self.layer(inputs)


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=base**i,
                            padding=base**i,
                        )
                    ),
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            channel,
                            channel,
                            kernel_size=kernel_size,
                            dilation=1,
                            padding=1,
                        )
                    ),
                )
                for i in range(nums)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=[12, 24, 48, 96, 192, 384, 768],
        down_sample_factors=[2, 2, 2, 2, 4, 4],
    ):
        super(Encoder, self).__init__()

        act_slope = 0.2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(
                in_channels, base_channels, kernel_size=proj_kernel_size, stride=1
            ),
            nn.LeakyReLU(act_slope, True),
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        for (in_c, out_c), down_f in zip(
            zip(channels[:-1], channels[1:]), down_sample_factors
        ):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(act_slope, True),
            ]

        # post layers
        layers += [
            Conv1d_S(channels[-1], out_channels, proj_kernel_size, stride=1),
            # nn.Tanh() TODO
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, z_inputs=None):
        return self.generator(conditions)




def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        bn: bool = False,
        activation = None,
        w_init_gain = None,
        input_transpose: bool = False,
        **kwargs
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super(Conv1d, self).forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(Conv1d, self).extra_repr(), self.causal, self.transpose)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = 'zeros',
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, \
                    "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x.transpose(1, 2) if self.transpose else x

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(ConvTranspose1d, self).extra_repr(), self.causal, self.transpose)



class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        causal: bool = True,
        act_causal: bool = False,
    ):
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        causal=causal,
                    )
                )
                for d in dilation
            ]
        )
        # self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        causal=causal,
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        # self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=h.snake_logscale
                        ),
                        causal=act_causal,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        ),
                        causal=act_causal,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
        causal: bool = True,
        act_causal: bool = False,
    ):
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        causal=causal,
                    )
                )
                for d in dilation
            ]
        )
        # self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=h.snake_logscale
                        ),
                        causal=act_causal,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        ),
                        causal=act_causal,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGANFlowVAE(nn.Module):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h, stat_path=None, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel
        causal = h.causal
        act_causal = h.get("act_causal", False)

        self.normalize_latent = False
        if stat_path:
            self.normalize_latent = True
            data = torch.load(stat_path)
            self.latent_mean = nn.Buffer(data['mean'].float().view(1, -1, 1)) # b,c,t
            self.latent_std = nn.Buffer(data['var'].float().sqrt().view(1, -1, 1))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim * 2,
            channels=h.downsample_channels,
            down_sample_factors=h.downsample_rates,
        )

        # self.flow = ResidualCouplingBlock(
        #     h.latent_dim, h.flow_hidden_channels, 5, 1, 4, gin_channels=0, causal=causal
        # )

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                f"Incorrect resblock class specified in hyperparameters. Got {h.resblock}"
            )

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(
                        h,
                        ch,
                        k,
                        d,
                        activation=h.activation,
                        causal=causal,
                        act_causal=act_causal,
                    )
                )

        # Post-conv
        activation_post = (
            Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(
            activation=activation_post, causal=act_causal
        )

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, causal=causal, bias=self.use_bias_at_final)
        )

        # # Weight initialization
        # for i in range(len(self.ups)):
        #     self.ups[i].apply(init_weights)
        # self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        x = self.audio_encoder(x)

        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        # def _log_stat(name, tensor):
        #     print(name, tensor.mean().item(), tensor.std().item(), tensor.min().item(), tensor.max().item())
        # _log_stat("Mean Stat:", m_q)
        # _log_stat("Std Stat:", torch.exp(logs_q))
        # _log_stat("Latent Stat:", z)

        # # Flow
        # mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        # z_p = self.flow(z, mask)
        # # _log_stat("Flow Stat:", z_p)

        # Pre-conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x, z_p, logs_q

    def _log_stat(self, name, tensor):
        print(name, tensor.mean().item(), tensor.std().item(), tensor.min().item(), tensor.max().item())

    @torch.no_grad()
    def encode(self, x, generator=None):
        x = self.audio_encoder(x)

        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.empty_like(m_q).normal_(generator=generator) * torch.exp(logs_q)
        # self._log_stat("Latent Stat:", z)
        if self.normalize_latent:
            z = (z - self.latent_mean) / self.latent_std
            # self._log_stat("Latent NormStat:", z)
        return z

    @torch.no_grad()
    def decode(self, z):
        if self.normalize_latent:
            z = z * self.latent_std + self.latent_mean

        # Pre-conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass


def init_vae(checkpoint_path, device=0):
    config = """{
        "resblock": "1",

        "upsample_rates": [5,4,3,2,2,2],
        "upsample_kernel_sizes": [10,8,6,4,4,4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        
        "downsample_rates": [2,2,2,3,4,5],
        "downsample_channels": [12, 24, 48, 96, 192, 384, 768],

        "use_tanh_at_final": false,
        "use_bias_at_final": false,

        "activation": "snakebeta",
        "snake_logscale": true,

        "causal": true,
        "act_causal": true,
        
        "latent_dim": 64,
        "sampling_rate": 24000
    }"""

    config = json.loads(config)
    h = AttrDict(config)

    torch.backends.cudnn.benchmark = False

    with torch.device(device):
        generator = BigVGANFlowVAE(h)
        state_dict_g = torch.load(checkpoint_path, map_location='cpu')
        missing, unexpected = generator.load_state_dict(state_dict_g['generator'] if 'generator' in state_dict_g else state_dict_g, strict=False)
        assert len(missing) == 0
        for name in unexpected:
            assert name.startswith('flow')
        # print(f'BigVGAN VAE Missing parameters: {missing}')
        # print(f'BigVGAN VAE Unexpected parameters: {unexpected}')
        generator.remove_weight_norm()
        generator.requires_grad_(False).eval()
        del state_dict_g
    return generator


def init_vae_stat(checkpoint_path, stat_path=None):
    config = """{
        "resblock": "1",

        "upsample_rates": [5,4,3,2,2,2],
        "upsample_kernel_sizes": [10,8,6,4,4,4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        
        "downsample_rates": [2,2,2,3,4,5],
        "downsample_channels": [12, 24, 48, 96, 192, 384, 768],

        "use_tanh_at_final": false,
        "use_bias_at_final": false,

        "activation": "snakebeta",
        "snake_logscale": true,

        "causal": true,
        "act_causal": true,
        
        "latent_dim": 64,
        "sampling_rate": 24000
    }"""

    config = json.loads(config)
    h = AttrDict(config)

    torch.backends.cudnn.benchmark = False

    generator = BigVGANFlowVAE(h, stat_path)
    state_dict_g = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = generator.load_state_dict(state_dict_g['generator'] if 'generator' in state_dict_g else state_dict_g, strict=False)
    # assert len(missing) == 0
    for name in missing:
        assert name.startswith('latent')
    for name in unexpected:
        assert name.startswith('flow')
    print(f'BigVGAN VAE Missing parameters: {missing}')
    # print(f'BigVGAN VAE Unexpected parameters: {unexpected}')
    generator.remove_weight_norm()
    del state_dict_g
    return generator


if __name__ == '__main__':
    import librosa
    from scipy.io.wavfile import write

    ckpt = '/apdcephfs_gy2/share_302533218/nickkhuang/models/flow-vae-fromscratch/g_00360000'
    stat = '/root/exp/flow-vae-fromscratch/global_mean_var_00002000.stat'
    vae = init_vae_stat(ckpt, stat)

    file = '/root/dataset/audio435/all/music_instrument_3e8da48dfa2bcc79fdb980f77dff541d_shot_v3_1815000-1830000_audio.wav'
    wav, sr = librosa.load(file, sr=vae.h.sampling_rate, mono=True)
    wav = torch.FloatTensor(wav.reshape(1, 1, -1)).cuda()

    latent = vae.encode(wav) # b,c,t
    y_hat = vae.decode(latent)
    print('Input:', wav.shape)
    print('Latent:', latent.shape)
    print('Output:', y_hat.shape)

    audio = y_hat.squeeze() * 32767
    audio = audio.cpu().numpy().astype('int16')
    write('/root/tmp/reconstruct.wav', vae.h.sampling_rate, audio)
