from argparse import Namespace
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from easydict import EasyDict

from .activation_layers import get_activation_layer
from .attn_layers import SelfAttentionLayer
from .mlp_layers import MLP
from .norm_layers import get_norm_layer


def xavier_initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normal_initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def zero_initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Block(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4,
                 attn_drop_rate=0,
                 mlp_drop_rate=0,
                 qk_norm=True,
                 act_type="silu",
                 norm_type="layer",
                 attn_mode="flash",
                 deterministic=False,
                 dtype=None,
                 device=None,
                 layer_idx=None,
                 elementwise_affine=True,
                 eps=1e-6,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        norm_kwargs = {'elementwise_affine': elementwise_affine, 'eps': eps, **factory_kwargs}
        super().__init__()
        self.layer_index = layer_idx

        norm_layer = get_norm_layer(norm_type)

        self.attn = SelfAttentionLayer(dim=hidden_size,
                                       num_heads=num_heads,
                                       qkv_bias=True,
                                       qk_norm=qk_norm,
                                       attn_drop=attn_drop_rate,
                                       **factory_kwargs,
                                       norm_type=norm_type,
                                       attn_mode="self_flash" if attn_mode == "flash" else attn_mode,
                                       deterministic=deterministic,
                                       )
        self.norm_1 = norm_layer(hidden_size, **norm_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(in_channels=hidden_size,
                       hidden_channels=int(hidden_size * mlp_ratio),
                       act_layer=act_layer,
                       drop=mlp_drop_rate,
                       **factory_kwargs,
                       )
        self.norm_2 = norm_layer(hidden_size, **norm_kwargs)

        # Initialize the weights
        # self.attn.apply(xavier_initialize)
        # self.mlp.apply(xavier_initialize)
        self.attn.apply(normal_initialize)
        self.mlp.apply(normal_initialize)

    def set_attn_mode(self, new_mode):
        if new_mode == "flash":
            self.attn.set_attn_mode("self_flash")
        else:
            self.attn.set_attn_mode(new_mode)

    def enable_deterministic(self):
        self.attn.enable_deterministic()

    def disable_deterministic(self):
        self.attn.disable_deterministic()

    def forward(self,
                x: torch.Tensor,
                freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        x = x + self.attn(self.norm_1(x), freqs_cis, mask)
        x = x + self.mlp(self.norm_2(x))

        return x


class MLMHead(nn.Module):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 share_embeddings=False,
                 dtype=None,
                 device=None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.share_embeddings = share_embeddings

        if share_embeddings:
            self.bias = nn.Parameter(torch.zeros(vocab_size, **factory_kwargs))
        else:
            self.head = nn.Linear(hidden_size, vocab_size, **factory_kwargs)

    def forward(self, x, embeddings=None):
        if self.use_embeddings and embeddings is None:
            raise ValueError("Embeddings must be provided when use_embeddings is True.")

        if self.use_embeddings:
            logits = F.linear(x, embeddings, bias=self.bias)
        else:
            logits = self.head(x)

        return logits


# ========================================
# Masked Language Modeling
# ========================================
class MLM(ModelMixin, ConfigMixin):
    """
    Masked language model with a Transformer backbone.
    Reference:
    [1] MaskGIT: http://arxiv.org/abs/2202.04200 https://github.com/google-research/maskgit

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Parameters
    ----------
    args: Namespace
        The arguments parsed from the command line.
    config: EasyDict
        The configuration dictionary. It should contain the following keys:
        - depth: int
        - hidden_size: int
        - num_heads: int
        - mlp_ratio: int
        - padded_vocab_size: int
        - media_vocab_size: int
        - embedding_norm: optional, bool, (default=False)
        - share_embedding: optional, bool, (default=True). Sec.3.4 in https://arxiv.org/pdf/1706.03762
        - max_pos_embed: optional, int, (default=-1)
        - vocab_image_first: optional, bool, (default=False)
        - uncond_p: optional, float, (default=0)
        - act_type: optional, str, (default='gelu')
        - norm_type: optional, str, (default='layer')
        - norm_affine: optional, bool, (default=True)
        - norm_eps: optional, float, (default=1e-6)
        - attn_drop_rate: optional, float, (default=0)
        - proj_drop_rate: optional, float, (default=0)
        - mlp_drop_rate: optional, float, (default=0)
        - qk_norm: optional, bool, (default=False)
        - attn_mode: optional, str, (default='torch')
    dtype: Optional[torch.dtype]
        The data type of the model parameters.
    device: Optional[torch.device]
        The device where the model is stored.
    """
    @register_to_config
    def __init__(self,
                 args: Namespace,
                 config: EasyDict,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.args = args
        # `self.config` is kept by ModelMixin
        self.model_config = config
        self.cfg_token = int(args.uncond_p > 0)

        # The first 1 is added for the mask token
        # The second 1 is added for the classifier-free guidance token
        self.total_vocab_size = config.padded_vocab_size + config.media_vocab_size + 1 + int(self.cfg_token)

        hidden_size = config.hidden_size
        attn_drop = config.get('attn_drop_rate', 0)
        mlp_drop = config.get('proj_drop_rate', 0)
        attn_mode = config.get('attn_mode', 'torch')
        qk_norm = config.get('qk_norm', False)
        act_type = config.get('act_type', 'gelu')
        act_layer = get_activation_layer(act_type)
        norm_type = config.get('norm_type', 'layer')
        norm_kwargs = {
            'elementwise_affine': config.get('norm_affine', True),
            'eps': config.get('norm_eps', 1e-6),
        }
        norm_layer = partial(get_norm_layer(norm_type), **norm_kwargs)

        self.x_embedder = nn.Embedding(self.total_vocab_size, hidden_size, **factory_kwargs)
        self.x_embedder.apply(normal_initialize)
        if config.get('embedding_norm', False):
            self.embedding_norm = norm_layer(hidden_size)
        if config.get('max_pos_embed', -1) > 0:
            self.pos_embed_layer = nn.Embedding(config.max_pos_embed, hidden_size, **factory_kwargs)
            self.pos_embed_layer.apply(normal_initialize)

        if config.get('first_layer', False):
            self.first_layer = nn.Sequential(
                norm_layer(hidden_size, **factory_kwargs),
                nn.Dropout(p=mlp_drop),
                nn.Linear(hidden_size, hidden_size, **factory_kwargs),
                act_layer(),
                norm_layer(hidden_size, **factory_kwargs),
                nn.Dropout(p=mlp_drop),
                nn.Linear(hidden_size, hidden_size, **factory_kwargs),
            )
            self.first_layer.apply(normal_initialize)

        self.blocks = nn.ModuleList([
            Block(hidden_size=hidden_size,
                  num_heads=config.num_heads,
                  mlp_ratio=config.mlp_ratio,
                  attn_drop_rate=attn_drop,
                  mlp_drop_rate=mlp_drop,
                  qk_norm=qk_norm,
                  act_type=act_type,
                  attn_mode=attn_mode,
                  norm_type=norm_type,
                  **norm_kwargs,
                  layer_idx=layer_idx,
                  **factory_kwargs
                  )
            for layer_idx in range(config.depth)
        ])

        self.final_layer = nn.Sequential(
            norm_layer(hidden_size, **factory_kwargs),
            nn.Dropout(p=mlp_drop),
            nn.Linear(hidden_size, hidden_size, **factory_kwargs),
            act_layer(),
            norm_layer(hidden_size, **factory_kwargs),
        )
        self.final_layer.apply(normal_initialize)

        if config.get('share_embedding', True):
            self.final_bias = nn.Parameter(torch.zeros((config.max_pos_embed, self.total_vocab_size), **factory_kwargs))
        else:
            self.head = nn.Linear(hidden_size, self.total_vocab_size, **factory_kwargs)

        # Gradient checkpoint
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= config.depth, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={config.depth}."

    def set_attn_mode(self, new_mode):
        for block in self.blocks:
            block.set_attn_mode(new_mode)

    def enable_deterministic(self):
        for block in self.blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.blocks:
            block.disable_deterministic()

    def forward(self,
                x: torch.Tensor,
                freqs_cos: Optional[torch.Tensor] = None,
                freqs_sin: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None,
                ):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor or token ids.
            freqs_cos (torch.Tensor, optional): Real part of the image RoPE.
            freqs_sin (torch.Tensor, optional): Imaginary part of the image RoPE.
            target (torch.Tensor, optional): Target token ids.

        .. note: We use separated :attr:`cos_cis` and :attr:`sin_cis` for RoPE computation in real space,
        because TensorRT does not support complex number computation.
        """
        output = {}

        # -------------------- Get image/video embedding ---------------------
        x = self.x_embedder(x)
        if hasattr(self, 'first_layer'):
            x = self.first_layer(x)
        B, T, C = x.shape

        # For MaskGiT
        if self.model_config.get('max_pos_embed', -1) > 0:
            position_ids = torch.arange(T, device=x.device)[None, :]
            position_embeddings = self.pos_embed_layer(position_ids)
            if self.model_config.get('embedding_norm', False):
                x = self.embedding_norm(x + position_embeddings)
            else:
                x = x + position_embeddings

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for layer_num, block in enumerate(self.blocks):
            block_args = [x, freqs_cis]
            if self.training and self.gradient_checkpoint and \
                    (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_args, use_reentrant=False)
            else:
                x = block(*block_args)

        x = self.final_layer(x)
        if self.model_config.get('share_embedding', True):
            logits = F.linear(x, self.x_embedder.weight) + self.final_bias
        else:
            logits = self.head(x)
        output['logits'] = logits

        # Compute the loss
        if target is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), target,
                                   ignore_index=self.args.ignore_index, reduction='none',
                                   label_smoothing=self.args.label_smoothing)
            output['loss'] = loss

        return output

    def params_count(self):
        counts = {
            "attn+mlp": sum([
                sum(p.numel() for p in block.attn.parameters()) +
                sum(p.numel() for p in block.mlp.parameters())
                for block in self.blocks
            ]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        return counts


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


#################################################################################
#                               MaskGIT Configs                                 #
#################################################################################

MASKGIT_CONFIG = {
    "maskgit-256-reimpl": dict(depth=24, hidden_size=768, num_heads=16, mlp_ratio=4,
                               embedding_norm=False, share_embeddings=True,
                               padded_vocab_size=1000, vocab_image_first=True),
    "maskgit-512": dict(depth=24, hidden_size=768, num_heads=16, mlp_ratio=4,
                        embedding_norm=True, share_embeddings=True,
                        padded_vocab_size=1000, max_pos_embed=1 + 32 ** 2, vocab_image_first=True),
    "maskgit-256-400m": dict(depth=24, hidden_size=1152, num_heads=16, mlp_ratio=4,
                             embedding_norm=False, share_embeddings=True, vocab_image_first=True),
}
