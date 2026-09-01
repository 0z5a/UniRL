import json

import torch
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from .vqvae_blocks import Encoder
from .movq_modules import MOVQDecoder
from .quantize import VectorQuantizer2 as VectorQuantizer


class MOVQ(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.codebook_size = n_embed
        self.downsample_factor = 2 ** (len(ddconfig["ch_mult"]) - 1)
        self.encoder = Encoder(**ddconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    @classmethod
    def from_config(cls, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        return cls(config["ddconfig"], config["n_embed"], config["embed_dim"], sane_index_shape=config["sane_index_shape"])

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def decode_code(self, code_b):
        batch_size, h, w = code_b.shape
        quant = self.quantize.embedding(code_b.flatten())
        quant = quant.view((batch_size, h, w, 4))
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def vq_encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        _, _, (_, _, indices) = self.quantize(h)
        return indices

    def vq_decode(self, indices):
        return self.decode_code(indices)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
