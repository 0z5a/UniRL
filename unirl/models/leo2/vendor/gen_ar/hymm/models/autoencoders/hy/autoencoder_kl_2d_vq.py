from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import einops

from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import Encoder, DiagonalGaussianDistribution
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

import loguru


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        scale = einops.rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        bias = einops.rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


def update_decoder(decoder):
    def forward(
            self,
            sample: torch.Tensor,
            latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        style = sample.clone()
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            from diffusers.utils import BaseOutput, is_torch_version
            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_idx, up_block in enumerate(self.up_blocks):
                    sample = self.adaptive[up_idx](sample, style)
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_idx, up_block in enumerate(self.up_blocks):
                    sample = self.adaptive[up_idx](sample, style)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_idx, up_block in enumerate(self.up_blocks):
                sample = self.adaptive[up_idx](sample, style)
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    decoder.forward = forward.__get__(decoder, decoder.__class__)
    decoder.adaptive = nn.ModuleList([
        AdaptiveGroupNorm(decoder.conv_in.in_channels, block.resnets[0].conv1.in_channels) for block in decoder.up_blocks
    ])


def bcfhw2bchw(x):
    assert x.ndim == 5 and x.shape[2] == 1, f"invalid shape: {x.shape}"
    x = x[:, :, 0]
    return x


def bchw2bcfhw(x):
    assert x.ndim == 4, f"invalid shape: {x.shape}"
    x = x[:, :, None]
    return x


class AutoencoderKL2DVQ(AutoencoderKL):

    def set_quant_conv(self, enable):
        self.use_quant_conv = enable
        if not self.use_quant_conv:
            self.quant_conv = None
            self.post_quant_conv = None

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        up_sample_type = 'default',
        act_fn: str = "silu",
        enable_agn = False,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        vq_type="bsq",
        codebook_size=None,
    ):
        # missing latents_mean and latents_std for AutoencoderKL.__init__, which are both default None
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
        )
        self.codebook_size = codebook_size
        self.downsample_factor = 2 ** (len(down_block_types) - 1)

        # missing mid_block_add_attention for Encoder.__init__, which is default True
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=False,  # no vq
        )
        # 兼容88-magvitv2-hy_240930
        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        # 88-magvitv2-hy_241017新增代码
        self.set_quant_conv(use_quant_conv)

        if up_sample_type == 'd2s':
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

            class Upsampler(nn.Module):
                def __init__(
                        self,
                        dim,
                        dim_out = None
                ):
                    super().__init__()
                    dim_out = dim * 4
                    self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
                    self.depth2space = depth_to_space

                def forward(self, x):
                    """
                    input_image: [B C H W]
                    """
                    out = self.conv1(x)
                    out = self.depth2space(out, block_size=2)
                    return out
            def replace_upsamplers(model):
                for name, child in model.named_children():

                    from diffusers.models.upsampling import Upsample2D
                    if isinstance(child, Upsample2D):
                        my_upsampler = Upsampler(child.channels)
                        setattr(model, name, my_upsampler)
                        # model._modules[child] = my_upsampler
                    replace_upsamplers(child)
            replace_upsamplers(self)
        if enable_agn:
            update_decoder(self.decoder)

        self.vq_type = vq_type
        if vq_type == "lfq":
            loguru.logger.info("Use LFQ Quantizer")
            from .lfqgan import LFQ

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
            self.quantize = LFQ(
                dim=embed_dim,
                codebook_size=n_embed,
                sample_minimization_weight=sample_minimization_weight,
                batch_maximization_weight=batch_maximization_weight,
                token_factorization=token_factorization,
            )
        elif vq_type == "bsq":
            loguru.logger.info("Use BSQ Quantizer")
            from .bsq import BSQWrapper

            self.quantize = BSQWrapper(
                embed_dim=latent_channels,
                # 0.25,
                beta=0,
                gamma0=1.0,
                gamma=2.0,
                zeta=1.0,
                input_format="bcthw",
                soft_entropy=True,
                group_size=1,
                persample_entropy_compute="analytical",
                # persample_entropy_compute='group',
                cb_entropy_compute="group",
                l2_norm=True,
                inv_temperature=1.0,
            )
        elif vq_type == "vqgan":
            loguru.logger.info("Use VQGAN Quantizer")
            from . import vqgan_quantize

            self.quantize = vqgan_quantize.VQGANCodebookWrapper2(
                num_codebook_vectors=codebook_size,
                latent_dim=latent_channels,
                # beta=0.25,
                beta=1,     # 移动特征
                alpha=0.25, # 移动 codebook
                l2_norm=True
            )
        elif vq_type == "dummy":
            loguru.logger.info("Use Dummy Quantizer")
            from .dummy_quantize import DummyQuantize

            self.quantize = DummyQuantize()
        else:
            raise ValueError(f"Unsupported vq type {vq_type}")

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
        
    def old_encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        if self.use_quant_conv:
            moments = self.quant_conv(h)
        else:
            moments = h
        return moments

    def encode(self, x, return_dict: bool = True):
        if len(x.shape) == 5:
            x = bcfhw2bchw(x)
        moments = self.old_encode(x)  # after quant_conv
        moments = bchw2bcfhw(moments)
        return moments

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
        codes = einops.rearrange(codes, 'b h w c -> b c 1 h w', h=h, w=w)  # todo
        return self.decode(codes)[:, :, 0]

    # def encode_to_1d(self, x):
    #     h = self.encode(x)
    #     quant, diff, indices, = self.quantize(h.contiguous(), return_loss_breakdown=False)
    #     return einops.rearrange(indices, 'b ... -> b (...)')
    #
    # def decode_from_1d(self, z, h, w):
    #     codes = self.quantize.indexes_to_codes(z.cuda()[None])  # mgvq 的实现没有batch维度，这里也懒得改其他，先和他统一
    #     codes = einops.rearrange(codes, 'b (h w) c -> b c 1 h w', h=h, w=w)  # todo
    #     return self.decode(codes)


    def encode_to_1d(self, x):
        h = self.encode(x)
        quant, diff, indices, = self.quantize(h.contiguous(), return_loss_breakdown=False)
        assert x.shape[0] == 1  # mgvq 的实现没有batch维度，这里也懒得改其他，先和他统一
        return einops.rearrange(indices, 'b ... -> b (...)')[0]

    def decode_from_1d(self, z):
        codes = self.quantize.indexes_to_codes(z.cuda()[None])  # mgvq 的实现没有batch维度，这里也懒得改其他，先和他统一
        max_token = 256
        max_token = 1024
        if codes.shape[1] > max_token:
            codes = codes[:, :max_token]
            loguru.logger.debug('token 太长了，裁剪')
        if codes.shape[1] < max_token:
            codes_zeros = torch.zeros(1, max_token, codes.shape[-1], dtype=codes.dtype, device=codes.device)
            codes_zeros[:, :codes.shape[1]] = codes
            codes = codes_zeros
        h = int(codes.shape[1] ** 0.5)
        w = h
        codes = einops.rearrange(codes, 'b (h w) c -> b c 1 h w', h=h, w=w)  # todo
        return self.decode(codes)

    def decode(self, z, return_dict: bool = True, generator=None):
        if len(z.shape) != 4:
            z = bcfhw2bchw(z)
        if self.use_quant_conv:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = bchw2bcfhw(dec)
        return dec

    def forward(
        self,
        sample,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator=None,
    ):
        h = self.encode(sample)
        (quant, diff, _,), loss_break = self.quantize(h.contiguous(), return_loss_breakdown=True)
        dec = self.decode(quant)
        return dec, diff, loss_break
