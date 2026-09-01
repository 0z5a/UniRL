import torch
import torch.nn as nn
import einops
import numpy as np
from typing import Tuple, Union, Optional
from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
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

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
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

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor

@dataclass
class DecoderOutput2(BaseOutput):
    sample: torch.FloatTensor
    code: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
         
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return torch.nn.functional.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = einops.rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = einops.rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = einops.rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return einops.rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, ch, out_ch):
        super().__init__()
        self.norm1 = RMS_norm(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = RMS_norm(ch)
        self.conv2 = nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.skip = nn.Identity() if out_ch == ch else nn.Conv2d(ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        return self.skip(x) + h

def patchfy(x, patch_size = 2):
    return einops.rearrange(x, 'n c (h p) (w q) -> n (c p q) h w', p = patch_size, q = patch_size)

def unpatchfy(x, patch_size = 2):
    return einops.rearrange(x, 'n (c p q) h w -> n c (h p) (w q)', p = patch_size, q = patch_size)

def repeat_upsample(x, out_x):
    if out_x.shape == x.shape or out_x.numel() < x.numel():
        return 0
    n,c,h,w = x.shape
    out_ch = out_x.shape[1]
    #pdb.set_trace()
    assert out_x.shape[-1] == x.shape[-1] * 2
    x = einops.rearrange(x.reshape(n, c // 4, 2, 2, h, w), 'n c p q h w -> n c (h p) (w q)')
    x = x[:,:,None].repeat_interleave(out_ch // x.shape[1], dim = 2).flatten(1,2)
    return x

def avg_downsample(x, out_x):
    if out_x.shape == x.shape:
        return 0
    assert out_x.shape[-1] == x.shape[-1] // 2
    n,c,h,w = x.shape
    out_ch = out_x.shape[1]
    x = x.reshape(n, c, h//2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.reshape(n, out_ch, -1, h // 2, w // 2).mean(dim = 2)
    return x

class Encoder(nn.Module):
    def __init__(
        self,
        init_patch_size = 2,
        in_channels = 3,
        blk_ch = 128,
        dim_mult = [1, 2, 4, 4],
        num_res_blocks = 2,
        z_channels = 64,
    ):
        super().__init__()
        self.init_patch_size = init_patch_size
        self.conv_in = nn.Conv2d(in_channels * self.init_patch_size ** 2, blk_ch * dim_mult[0], kernel_size=3, stride=1, padding = 1)

        dims = [blk_ch * mul for mul in [1] + dim_mult]

        self.enc = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            blk = []
            for _ in range(num_res_blocks):
                blk.append(ResnetBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                blk.append(nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                           nn.Conv2d(out_dim, out_dim, 3, stride=(2, 2))))
            self.enc.append(nn.Sequential(*blk))
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(out_dim, out_dim)
        self.mid.attn_1 = AttnBlock(out_dim)
        self.mid.block_2 = ResnetBlock(out_dim, out_dim)

        # end
        self.norm_out = RMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, 2 * z_channels, kernel_size=3, stride=1, padding=1)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            assert x.shape[2] == 1
            x = x[:,:,0]
        x = patchfy(x, self.init_patch_size)
        # downsampling
        x = self.conv_in(x)

        for idx, blk in enumerate(self.enc):
            out = blk(x)
            x = out + avg_downsample(x, out)
        # middle
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        # end
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        if len(x_shape) == 5:
            x = x[:,:,None]
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        init_patch_size = 2,
        in_channels = 3,
        blk_ch = 256,
        dim_mult = [1, 2, 4, 4],
        num_res_blocks = 2,
        z_channels = 64,
    ):
        super().__init__()
        self.init_patch_size = init_patch_size
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        # z to block_in
        dims = [blk_ch * mul for mul in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = nn.Conv2d(z_channels, dims[0], kernel_size=3, stride=1, padding=1)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dims[0], dims[0])
        self.mid.attn_1 = AttnBlock(dims[0])
        self.mid.block_2 = ResnetBlock(dims[0], dims[0])

        # upsampling
        self.dec = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            blk = []
            for _ in range(num_res_blocks + 1):
                blk.append(ResnetBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                blk.append(nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                                         nn.Conv2d(out_dim, out_dim, 3, padding=1)))
            self.dec.append(nn.Sequential(*blk))
        # end
        self.norm_out = RMS_norm(out_dim)
        self.conv_out = nn.Conv2d(out_dim, init_patch_size ** 2 * in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x_shape = z.shape
        if len(x_shape) == 5:
            assert z.shape[2] == 1
            z = z[:,:,0]
        #z = self.decoder.post_quant_conv(z)
        # z to block_in
        x = self.conv_in(z)
        #repa_x = self.repa_mlp(x)
        #self.repa_out = repa_x
        # middle
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        #z = self.repa_mlp(x)
        for idx, blk in enumerate(self.dec):
            out = blk(x)
            x = out + repeat_upsample(x, out) 
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = unpatchfy(x, self.init_patch_size)
        if len(x_shape) == 5:
            x = x[:,:,None]
        return x



class AutoencoderKL(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images/videos into latents and decoding latent representations into images/videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    @register_to_config
    def __init__(
        self,
        init_patch_size = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_ch = 128, 
        dec_ch = 256,
        enc_dim_mult = [1,2,4,4],
        dec_dim_mult = [1,2,4,4],
        spatial_compression_ratio: int = 16,
        latent_channels: int = 64,
        layers_per_block: int = 2,
        block_out_channels: Tuple[int] = (64,),
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        sample_size: int = 256,
        sample_tsize: int = 64,
        force_upcast: float = True,
        time_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
        with_t = 0.0,
    ):
        super().__init__()
        self.time_compression_ratio = time_compression_ratio
        downsample_lvl = int(np.log2(spatial_compression_ratio))
        self.config.block_out_channels = [block_out_channels[0] for i in range(downsample_lvl + 1)]
        self.register_buffer('latent_std', torch.ones(1, latent_channels, 1, 1, 1))
        self.register_buffer('latent_mean', torch.zeros(1, latent_channels, 1, 1, 1))
        self.encoder = Encoder(
            init_patch_size = init_patch_size,
            in_channels = in_channels,
            blk_ch = enc_ch,
            dim_mult = enc_dim_mult,
            num_res_blocks = layers_per_block,
            z_channels = latent_channels,
        )

        self.decoder = Decoder(
            init_patch_size = init_patch_size,
            in_channels = in_channels,
            blk_ch = dec_ch,
            dim_mult = dec_dim_mult,
            num_res_blocks = layers_per_block,
            z_channels = latent_channels,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // time_compression_ratio

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / spatial_compression_ratio)
        self.tile_overlap_factor = 0.25

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger videos.
        """
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def run_quant_conv(self, x):
        out = self.encoder.quant_conv(x[:,:,0])[:,:,None]
        out = (out - torch.cat([self.latent_mean, torch.zeros_like(self.latent_mean)], dim = 1)) \
                            / torch.cat([self.latent_std, torch.ones_like(self.latent_std)], dim = 1)
        return out

    def run_post_quant_conv(self, x):
        x = x * self.latent_std + self.latent_mean
        out = self.decoder.post_quant_conv(x[:,:,0])[:,:,None]
        return out

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images/videos into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images/videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if len(x.shape) != 5:
            assert len(x.shape) == 4, "The input tensor should have be B x C x H x W or B x C x T x H x W."
            x = x[:, :, None]

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            return self.temporal_tiled_encode(x, return_dict=return_dict)
        
        if self.use_spatial_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
        ):
            return self.spatial_tiled_encode(x, return_dict=return_dict)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)
        moments = self.run_quant_conv(h)

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)

        return posterior.mode()[:, :, 0, :, :]
        # return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_spatial_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.spatial_tiled_decode(z, return_dict=return_dict)

        z = self.run_post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images/videos.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if z.ndim == 4:
            z = z[:, :, None]
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        
        if len(decoded.shape) == 5:
            assert decoded.shape[2] == 1, "The decoded tensor should have 5 dimensions."
            decoded = decoded[:, :, 0]

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_t(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (
                1 - x / blend_extent
            ) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(
        self,
        x: torch.FloatTensor,
        return_dict: bool = True,
        return_moments: bool = False,
    ) -> AutoencoderKLOutput:
        r"""Encode a batch of images/videos using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image/videos size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.run_quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        if return_moments:
            return moments

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def spatial_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.run_post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def temporal_tiled_encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:

        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size
                or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.run_quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        moments = torch.cat(result_row, dim=2)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def temporal_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # Split z into overlapping tiles and decode them separately.

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size
                or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.run_post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        dec = torch.cat(result_row, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput2, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        z = self.encode(x)
        dec = self.decode(z)
        return dec


        # x = sample
        # posterior = self.encode(x).latent_dist

        # if sample_posterior:
        #     z = posterior.sample(generator=generator)
        # else:
        #     z = posterior.mode()
        # dec = self.decode(z).sample
        # if not return_dict:
        #     if return_posterior:
        #         return (dec, posterior)
        #     else:
        #         return (dec,)
        # if return_posterior:
        #     return DecoderOutput2(sample=dec, posterior=posterior)
        # else:
        #     return DecoderOutput2(sample=dec, code = z)
