import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .modulate import modulate
from .norm_layers import LayerNormF32
from .initializers import (
    proj_reset_parameters,
    zero_reset_parameters
)


class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size,
            in_chans,
            embed_dim,
            act_layer=None,
            bias=True,
            dims=2,
            use_modulation=True,
            norm_type="layer",
            dtype=None,
            device=None
    ):
        """ A patch embedding layer for dit-like models. Support 2D and 3D inputs. """
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.dims = dims
        assert dims in [2, 3], f"Unsupported dims: {dims}"

        self.use_modulation = use_modulation
        self.norm_type = norm_type
        assert norm_type == "layer", "Only layer norm is supported for PatchEmbed."

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size) if dims == 2 else (1, patch_size, patch_size)
        self.proj = conv_nd(dims, in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias,
                            **factory_kwargs)
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        if use_modulation:
            self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
            self.adaLN_modulation = nn.Sequential(
                act_layer(),
                nn.Linear(embed_dim, 2 * embed_dim, bias=True, **factory_kwargs)
            )
            # Zero-initialize the modulation
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def prepare_reset_parameters(self):
        self.proj.reset_parameters = proj_reset_parameters.__get__(self.proj)
        if self.use_modulation:
            self.adaLN_modulation[1].reset_parameters = zero_reset_parameters.__get__(self.adaLN_modulation[1])

    def forward(self, x, t):
        if self.dims == 3 and x.ndim == 4:
            x = x.unsqueeze(2)
        x = self.proj(x)
        _, _, *token_sizes = x.shape
        x = x.flatten(2).transpose(1, 2)  # BCHW/BCDHW -> BLC

        if self.use_modulation:
            shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift=shift, scale=scale)

        return x, *token_sizes


class FinalLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            patch_size,
            out_channels,
            act_layer,
            dims=None,
            norm_type="layer",
            modulate_hidden_size=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.dims = dims
        assert dims in [2, 3], f"Unsupported dims: {dims}"
        if modulate_hidden_size is None:
            modulate_hidden_size = hidden_size

        # Just use LayerNorm for the final layer
        if norm_type == "layer":
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        elif norm_type == "layer_f32":
            self.norm_final = LayerNormF32(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(modulate_hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def prepare_reset_parameters(self):
        self.linear.reset_parameters = zero_reset_parameters.__get__(self.linear)
        self.adaLN_modulation[1].reset_parameters = zero_reset_parameters.__get__(self.adaLN_modulation[1])

    def forward(self, x, t, *token_sizes):
        assert len(token_sizes) in [2, 3]
        if len(token_sizes) == 2:
            token_h, token_w = token_sizes
            token_d = 1  # for processing image with 3d layer
        elif len(token_sizes) == 3:
            token_d, token_h, token_w = token_sizes
        else:
            raise NotImplementedError()

        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        bsz = x.shape[0]
        ch = self.out_channels
        ps = self.patch_size
        if self.dims == 3:
            x = x.reshape(shape=(bsz, token_d, token_h, token_w, ch, ps, ps, ps))
            x = torch.einsum('ndhwcopq->ncdohpwq', x)
            x = x.reshape(shape=(bsz, ch, token_d * ps, token_h * ps, token_w * ps))
        else:
            x = x.reshape(shape=(bsz, token_h, token_w, ch, ps, ps))
            x = torch.einsum('nhwcpq->nchpwq', x)
            x = x.reshape(shape=(bsz, ch, token_h * ps, token_w * ps))
        return x


class LinearPatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans,
            embed_dim,
            act_layer,
            bias=True,
            dtype=None,
            device=None
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        # patch_size = to_2tuple(patch_size)

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias, **factory_kwargs)
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias,
        #                       **factory_kwargs)
        # nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        # if bias:
        #     nn.init.zeros_(self.proj.bias)

        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True, **factory_kwargs)
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def prepare_reset_parameters(self):
        self.adaLN_modulation[1].reset_parameters = zero_reset_parameters.__get__(self.adaLN_modulation[1])

    def forward(self, x, t):
        x = self.proj(x)
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        dtype = x.dtype
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        return x.to(dtype), 0, 0


class LinearFinalLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            out_channels,
            act_layer,
            norm_type="layer",
            modulate_hidden_size=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.out_channels = out_channels
        if modulate_hidden_size is None:
            modulate_hidden_size = hidden_size

        # Just use LayerNorm for the final layer
        if norm_type == "layer":
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        elif norm_type == "layer_f32":
            self.norm_final = LayerNormF32(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.linear = nn.Linear(hidden_size, out_channels, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(modulate_hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def prepare_reset_parameters(self):
        self.linear.reset_parameters = zero_reset_parameters.__get__(self.linear)
        self.adaLN_modulation[1].reset_parameters = zero_reset_parameters.__get__(self.adaLN_modulation[1])

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class AudioFinalLayer(LinearFinalLayer):
    def forward(self, x, t, *token_sizes):
        x = super().forward(x, t)
        x = x.transpose(1, 2)  # BLC -> BCL
        assert x.size(-1) == math.prod(token_sizes), \
            f"Expected the last dimension of output to be {math.prod(token_sizes)}, but got {x.size(-1)}"
        return x


# ---------------------------- UNet ----------------------------
def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, **kwargs)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param in_channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
        kernel_size=3,
        padding=1,
    ):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(dims, self.in_channels, self.out_channels, kernel_size, padding=padding, **factory_kwargs),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels, **factory_kwargs)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, **factory_kwargs)
            ),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, self.in_channels, self.out_channels, kernel_size, padding=padding, **factory_kwargs
            )
        else:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1, **factory_kwargs)

    def reset_parameters(self):
        self.out_layers[3].reset_parameters = zero_reset_parameters.__get__(self.out_layers[3])

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        # Adaptive Group Normalization
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1. + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: vae latent dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: transformer model dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0,
                 device=None, dtype=None, dims=2, kernel_size=3, padding=1, use_modulation=True, norm_type="group"):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]
        self.dims = dims
        assert dims in [2, 3]

        self.use_modulation = use_modulation
        self.norm_type = norm_type
        assert use_modulation, "use_modulation is required for UNetDown."
        assert norm_type == "group", "Only group norm is supported for UNetDown."

        self.model = nn.ModuleList(
            [conv_nd(
                dims, in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding,
                **factory_kwargs
            )]
        )

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=hidden_channels, emb_channels=emb_channels, out_channels=out_channels, dropout=dropout,
                dims=dims, kernel_size=kernel_size, padding=padding, **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=hidden_channels, emb_channels=emb_channels,
                    out_channels=hidden_channels if (i + 1) * 2 != self.patch_size else out_channels,
                    dropout=dropout, down=True, dims=dims, kernel_size=kernel_size, padding=padding, **factory_kwargs
                ))

    def forward(self, x, t):
        assert all([
            x.shape[dim] % self.patch_size == 0
            for dim in range(2, x.ndim)
        ])
        if self.dims == 3 and x.ndim == 4:
            x = x.unsqueeze(2)

        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, *token_sizes = x.shape

        if len(token_sizes) == 2:
            x = rearrange(x, 'b c h w -> b (h w) c')
        elif len(token_sizes) == 3:
            x = rearrange(x, 'b c d h w -> b (d h w) c')
        else:
            raise NotImplementedError()

        return x, *token_sizes


class UNetUp(nn.Module):
    """
    patch_size: one of [1, 2 ,4 ,8]
    in_channels: transformer model dim
    hidden_channels: hidden dim for reducing parameters
    out_channels: vae latent dim
    """
    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels, out_channels, dropout=0.0, device=None,
                 dtype=None, out_norm=False, dims=2, kernel_size=3, padding=1, use_modulation=True, norm_type="group"):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]
        self.dims = dims
        assert dims in [2, 3]

        self.use_modulation = use_modulation
        self.norm_type = norm_type
        assert use_modulation, "use_modulation is required for UNetUp."
        assert norm_type == "group", "Only group norm is supported for UNetUp."

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(ResBlock(
                in_channels=in_channels, emb_channels=emb_channels, out_channels=hidden_channels, dropout=dropout,
                dims=dims, kernel_size=kernel_size, padding=padding, **factory_kwargs
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(ResBlock(
                    in_channels=in_channels if i == 0 else hidden_channels, emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout, up=True, dims=dims, kernel_size=kernel_size, padding=padding, **factory_kwargs
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                conv_nd(
                    dims, in_channels=hidden_channels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, **factory_kwargs
                ),
            ))
        else:
            self.model.append(conv_nd(
                dims, in_channels=hidden_channels, out_channels=out_channels,
                kernel_size=kernel_size, padding=padding, **factory_kwargs
            ))

    # batch_size, seq_len, model_dim
    def forward(self, x, t, *token_sizes):
        assert len(token_sizes) in [2, 3]
        if len(token_sizes) == 2:
            token_h, token_w = token_sizes
            token_d = 1     # for processing image with 3d layer
        elif len(token_sizes) == 3:
            token_d, token_h, token_w = token_sizes
        else:
            raise NotImplementedError()

        if self.dims == 3:
            x = rearrange(x, 'b (d h w) c -> b c d h w', d=token_d, h=token_h, w=token_w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=token_h, w=token_w)

        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


def project_in_layer(proj_type, config, dims=2, **kwargs):
    if proj_type == "conv":
        return UNetDown(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.vae_latent_dim,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.hidden_size,
            dims=dims,
            **kwargs
        )
    elif proj_type == "linear":
        return PatchEmbed(
            patch_size=config.patch_size,
            in_chans=getattr(config, "img_latent_in_channels", config.vae_latent_dim),
            embed_dim=config.hidden_size,
            act_layer=nn.SiLU,
            dims=dims,
            **kwargs
        )
    else:
        raise ValueError(f"img_proj_type `{config.img_proj_type}` not supported")


def project_out_layer(proj_type, config, dims=2, **kwargs):
    if proj_type == "conv":
        return UNetUp(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.vae_latent_dim,
            out_norm=True,
            dims=dims,
            **kwargs
        )
    elif proj_type == "linear":
        return FinalLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=config.vae_latent_dim,
            act_layer=nn.SiLU,
            dims=dims,
            **kwargs
        )
    elif proj_type == "audio_linear":
        return AudioFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=config.audio_vae_latent_dim,
            act_layer=nn.SiLU,
            modulate_hidden_size=config.modulate_hidden_size,
            **kwargs
        )
    else:
        raise ValueError(f"img_proj_type `{config.img_proj_type}` not supported")


if __name__ == "__main__":
    batch_size = 3
    patch_size = 2
    model_dim = 2560
    vae_latent_dim = 16
    hidden_channels = 1024

    vae_latent_h = 32
    vae_latent_w = 32

    unet_down = UNetDown(patch_size=patch_size, in_channels=vae_latent_dim, hidden_channels=hidden_channels, out_channels=model_dim, dropout=0.1)
    print(sum([p.numel() for p in unet_down.parameters()]))  # 1,2: 85,360,128; 4: 104,240,640
    vae_latent = torch.randn(batch_size, vae_latent_dim, vae_latent_h, vae_latent_w)
    model_input, h, w = unet_down(vae_latent)
    print(model_input.shape)

    model_output = model_input

    unet_up = UNetUp(patch_size=patch_size, in_channels=model_dim, hidden_channels=hidden_channels, out_channels=vae_latent_dim, dropout=0.1)
    print(sum([p.numel() for p in unet_up.parameters()])) # 1,2: 35,809,296; 4: 54,689,808
    output = unet_up(model_output, h, w)
    print(output.shape)
