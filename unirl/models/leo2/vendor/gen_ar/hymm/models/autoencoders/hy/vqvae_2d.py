from typing import *

import einops
import loguru
import torch
from torch import nn

from torch.nn import functional as F
from diffusers.utils.deprecation_utils import deprecate

from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.configuration_utils import (ConfigMixin, register_to_config)

from dataclasses import dataclass

from collections import namedtuple
LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs', 'extra_weighted_loss'])



from typing import Callable, AnyStr
def replace_module(model, is_target_module: Callable, get_alternative: Callable):
    def __replace_module(model, full_name):
        for name, child in model.named_children():
            if is_target_module(full_name, child):
                new_module = get_alternative(full_name, child)
                setattr(model, name, new_module)
            __replace_module(child, full_name + '.' + name)
    __replace_module(model, '')


@dataclass
class VQEncoderOutput:
    latents: torch.Tensor

@dataclass
class DecoderOutput:
    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None
    quant: torch.Tensor = None
    loss_break: LossBreakdown = None


def bcfhw2bchw(x):
    assert x.ndim == 5 and x.shape[2] == 1, f"invalid shape: {x.shape}"
    x = x[:, :, 0]
    return x

def bchw2bcfhw(x):
    assert x.ndim == 4, f"invalid shape: {x.shape}"
    x = x[:, :, None]
    return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        self.add_temporal_downsample = add_temporal_downsample
        stride = (2, 2, 2) if add_temporal_downsample else (1, 2, 2)  # THW
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        spatial_pad = (0, 1, 0, 1, 0, 0)  # WHT
        x = nn.functional.pad(x, spatial_pad, mode="constant", value=0)

        temporal_pad = (0, 0, 0, 0, 0, 1) if self.add_temporal_downsample else (0, 0, 0, 0, 1, 1)
        x = nn.functional.pad(x, temporal_pad, mode="replicate")

        x = self.conv(x)
        return x


def depth_to_space2(x, *args, **kwargs):
    return einops.rearrange(x, 'b (nh nw c) h w -> b c (h nh) (w nw)', nh=2, nw=2)

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, h, w = x.shape[-3:]
    s = block_size**2
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )
    outer_dims = x.shape[:-3]
    x = x.view(-1, block_size, block_size, c // s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    return x

class Depth2SpaceUpsampler(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None
    ):
        super().__init__()
        self.channels = dim
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        # return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        out = self.conv1(x)
        # assert torch.allclose(self.depth2space(out, 2) , depth_to_space2(out))
        out = self.depth2space(out, block_size=2)
        return out

class Depth2SpaceUpsampler3D(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None
    ):
        super().__init__()
        dim_out = dim * 8
        self.conv1 = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C T H W]
        """
        # return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-3)
        out = self.conv1(x)
        out = einops.rearrange(out, 'b (nt nh nw c) t h w -> b c (t nt) (h nh) (w nw)', nt=2, nh=2, nw=2)

        return out

class SpatialNorm3D(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
            self,
            f_channels: int,
            zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-3:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

class MyAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if input_ndim == 5:
            batch_size, channel, n_frame, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, n_frame * height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if input_ndim == 5:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, n_frame, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class HunyuanVQVAE(VQModel):


    def set_quant_conv(self, enable):
        self.use_quant_conv = enable
        if not self.use_quant_conv:
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            self.quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, (1, 1))
            self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, (1, 1))

    def init_from_ckpt(self, path: str):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        contains_disc = False
        for k in sd.keys():
            if k.startswith("loss_with_disc."):
                contains_disc = True
                break
        if contains_disc:
            sd = {k.replace("vae.", ""): v for k, v in sd.items() if k.startswith("vae.")}
        m, u = self.load_state_dict(sd, strict=True)
        assert len(m) == 0
        assert len(u) == 0
        print(f"Restored from {path}")

    def switch_to(self, mode_2d_3d='2d'):
        if mode_2d_3d != self.mode_2d_3d:

            def conv_2d_param_to_3d(conv2d, conv3d):
                if isinstance(conv2d, nn.Conv3d):
                    conv3d.load_state_dict(conv2d.state_dict())
                    return
                conv3d.weight.data.fill_(0)
                if conv2d.kernel_size[0] == 3:
                    conv3d.weight.data[:, :, 1] = conv2d.weight.data
                else:
                    conv3d.weight.data[:, :, 0] = conv2d.weight.data
                if hasattr(conv3d, 'bias') and conv3d.bias is not None:
                    conv3d.bias.data[:] = conv2d.bias.data

            def conv_2d_to_3d(full_name, conv):
                # 3,3 -> 3,3,3
                # 1,1 -> 1,1,1
                # TODO: 参数复制
                assert len(set(conv.kernel_size)) == 1, f'{conv.kernel_size}'
                ret = nn.Conv3d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0], bias=conv.bias is not None)
                conv_2d_param_to_3d(conv, ret)
                return ret

            def conv_3d_to_2d(full_name, conv):
                raise NotImplementedError('参数还没复制')
                return nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size[0], stride=conv.stride[0], padding=conv.padding[0], bias=conv.bias is None)

            def downsample_2d_to_3d(name, downsampler):
                ret = Downsample3D(downsampler.channels)
                conv_2d_param_to_3d(downsampler.conv, ret.conv)
                return ret

            def upsample_2d_to_3d(name, upsampler):
                ret = Depth2SpaceUpsampler3D(upsampler.channels)

                ret.conv1.weight.data.fill_(0)
                # (nt nh nw)  (nh nw)
                assert ret.conv1.kernel_size[0] == 3
                ret.conv1.weight.data[:4 * upsampler.conv1.in_channels, :] = upsampler.conv1.weight.data

                if hasattr(ret.conv1, 'bias') and ret.conv1.bias is not None:
                    ret.conv1.bias.data[:4 * upsampler.conv1.in_channels] = upsampler.conv1.bias.data
                return ret

            def spatial_norm_2d_to_3d(name, spatial_norm):
                from diffusers.models.attention_processor import Attention, SpatialNorm
                spatial_norm: SpatialNorm
                ret = SpatialNorm3D(spatial_norm.norm_layer.num_channels, spatial_norm.conv_y.in_channels)
                conv_2d_param_to_3d(spatial_norm.conv_y, ret.conv_y)
                conv_2d_param_to_3d(spatial_norm.conv_b, ret.conv_b)
                return ret


            if self.mode_2d_3d == '2d':
                replace_module(self, lambda _, x: isinstance(x, nn.Conv2d), conv_2d_to_3d)

                from diffusers.models.downsampling import Downsample2D
                replace_module(self, lambda name, x: isinstance(x, Downsample2D), downsample_2d_to_3d)
                from diffusers.models.upsampling import Upsample2D
                replace_module(self, lambda name, x: isinstance(x, Upsample2D) or isinstance(x, Depth2SpaceUpsampler), upsample_2d_to_3d)

                from diffusers.models.attention_processor import Attention, SpatialNorm
                replace_module(self, lambda name, x: isinstance(x, SpatialNorm), spatial_norm_2d_to_3d)



                for module in self.modules():
                    if isinstance(module, Attention):
                        module.set_processor(MyAttnProcessor2_0())  # 原版的 attention processor 只能对空间上做self attention
            else:
                replace_module(self, lambda _, x: isinstance(x, nn.Conv3d), conv_3d_to_2d)

        self.to(self.device)
        self.mode_2d_3d = mode_2d_3d

    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 3,
            sample_size: int = 32,
            num_vq_embeddings: int = 256,
            norm_num_groups: int = 32,
            vq_embed_dim: Optional[int] = None,
            scaling_factor: float = 0.18215,
            norm_type: str = "group",  # group, spatial
            mid_block_add_attention=True,
            lookup_from_codebook=False,
            force_upcast=False,
            vq_type='vqgan',
            up_sample_type='default',
            use_quant_conv=False,
            simVQ=False,
            vqNorm=True,
    ):
        vq_embed_dim = latent_channels
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
            mid_block_add_attention=mid_block_add_attention,
            lookup_from_codebook=lookup_from_codebook,
            vq_embed_dim=vq_embed_dim,
            num_vq_embeddings=num_vq_embeddings,
        )
        self.mode_2d_3d = '2d'
        self.set_quant_conv(use_quant_conv)
        self.vq_type = vq_type
        if vq_type == 'lfq':
            loguru.logger.info('Use LFQ')
            from magvit.taming.models.lfqgan import LFQ
            # embed_dim = 8
            # embed_dim = 18
            embed_dim = latent_channels
            n_embed = 2**embed_dim
            learning_rate = 1e-4
            sample_minimization_weight = 1.0
            batch_maximization_weight = 1.0
            scheduler_type = None
            use_ema = True
            token_factorization = False
            self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed,
                                sample_minimization_weight=sample_minimization_weight,
                                batch_maximization_weight=batch_maximization_weight,
                                token_factorization=token_factorization)
        elif vq_type == 'bsq':
            loguru.logger.info('Use BSQ')
            from .bsq import BSQWrapper
            self.quantize = BSQWrapper(
                latent_channels,
                # 0.25,
                0,
                1.0, 2.0, 1.0,
                group_size=1,
                persample_entropy_compute='analytical',
                # persample_entropy_compute='group',
                cb_entropy_compute='group',
                input_format='bcthw',
                l2_norm=vqNorm,
                inv_temperature=1.0,
            )
        elif vq_type == 'vqgan':
            loguru.logger.info(f'Use VQGAN Codebook, codebook size {num_vq_embeddings}')
            from . import vqgan_quantize
            self.quantize = vqgan_quantize.VQGANCodebookWrapper2(
                num_codebook_vectors=num_vq_embeddings, latent_dim=latent_channels,
                entropy_loss_ratio=0.,
                beta=1.,     # 移动特征
                alpha=0.25,  # 移动 codebook
                # beta=0.25,     # 移动特征
                # alpha=1., # 移动 codebook
                l2_norm=vqNorm,
                simVQ=simVQ,
            )
        elif vq_type == 'dummy':
            from models.autoencoders.dummy_quantize import DummyQuantize
            self.quantize = DummyQuantize()
        else:
            raise ValueError(f'Unsupported vq type {vq_type}')


        if up_sample_type == 'd2s':
            from diffusers.models.upsampling import Upsample2D

            replace_module(self, lambda name, x: isinstance(x, Upsample2D), lambda name, child: Depth2SpaceUpsampler(child.channels))

        self.set_default_attn_processor = lambda: None  # placeholder

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_sample_only: bool = True):

        if len(x.shape) == 5 and self.mode_2d_3d == '2d':
            x = bcfhw2bchw(x)
        h = self.encoder(x)
        h = self.quant_conv(h)

        if self.mode_2d_3d == '2d':
            h = bchw2bcfhw(h)

        if return_sample_only:
            return h
        else:
            return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
            self, quant: torch.Tensor, shape=None, return_sample_only=True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        # also go through quantization layer

        if len(quant.shape) == 5 and self.mode_2d_3d == '2d':
            quant = bcfhw2bchw(quant)

        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant if self.config.norm_type == "spatial" else None)

        if self.mode_2d_3d == '2d':
            dec = bchw2bcfhw(dec)

        if return_sample_only:
            return dec
        return DecoderOutput(sample=dec)


    def forward(
            self, sample: torch.Tensor, return_dict: bool = True
    ):
        h = self.encode(sample, return_sample_only=False).latents
        (quant, diff, _,), loss_break = self.quantize(h.contiguous(), return_loss_breakdown=True)
        dec = self.decode(quant, return_sample_only=False)

        return dec.sample, diff, loss_break

    @torch.no_grad()
    def vq_encode(self, x):
        # x: b c h w, [-1, 1], cuda
        h = self.encode(x.cuda())  # b c 1 h w
        quant, diff, indices, = self.quantize(h.contiguous(), return_loss_breakdown=False)
        return indices.reshape(quant.shape[0], quant.shape[-2], quant.shape[-1])
        # return einops.rearrange(indices, 'b 1 h w -> b h w')

    @torch.no_grad()
    def vq_decode(self, code):
        b, h, w = code.shape
        if self.vq_type == 'bsq':
            codes = self.quantize.indexes_to_codes(code.cuda())  # b h w c
        elif self.vq_type == 'lfq':
            codes = self.quantize.indices_to_bits(code.cuda())
            codes = codes.to(self.dtype)
            codes = codes * 2 - 1
        elif self.vq_type == 'vqgan':
            codes = self.quantize.get_codebook_entry(code.cuda())
            codes = codes.to(self.dtype)
        else:
            raise ValueError(f'Unsupported quantization {self.vq_type}')

        codes = einops.rearrange(codes, 'b h w c -> b c 1 h w', h=h, w=w)  # todo
        return self.decode(codes)[:, :, 0]
