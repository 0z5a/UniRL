from argparse import Namespace
from typing import Dict, Optional

import torch
import torch.nn as nn
from diffusers.models import ModelMixin

from .config import Config
from .model import ckpt_wrapper, Block
from ..basic.diffloss import DiffLoss
from ..basic.embed_layers import TextProjection


# Mar with Text Encoder GPT
class MarTEGPT(ModelMixin):
    def __init__(
            self, args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.args = args
        self.config = config
        self.total_vocab_size = config.padded_vocab_size + args.media_vocab_size

        self.lm_head = nn.Linear(config.n_embd, self.total_vocab_size, bias=config.lm_head_bias, **factory_kwargs)
        self.transformer = nn.ModuleDict(
            dict(

                h=nn.ModuleList(Block(config, block_idx, attn_type="full", **factory_kwargs) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs),
            )
        )
        self.txt_in = TextProjection(
            4096,
            config.n_embd,
            nn.SiLU,
            **factory_kwargs,
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        # Gradient checkpoint
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= config.n_layer, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={config.n_layer}."

        self.patch_size = config.get('patch_size', 2)
        self.vae_embed_dim = config.get('vae_embed_dim', 4)
        self.token_embed_dim = self.vae_embed_dim * self.patch_size ** 2

        self.z_proj = nn.Linear(self.token_embed_dim, config.n_embd, **factory_kwargs)
        self.mask_token = nn.Parameter(torch.zeros(config.n_embd, **factory_kwargs), requires_grad=True)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=config.n_embd,
            depth=config.get('diffloss_d', 3),
            width=config.get('diffloss_w', config.n_embd),
            grad_checkpointing=args.gradient_checkpoint,
            diffusion_config=args,
            **factory_kwargs,
        )
        self.diff_batch_factor = config.get('diff_batch_factor', 1)

    def get_training_parts(self, mode):
        if mode == 'lrx10':
            diffloss_params = []
            other_params = []
            for name, param in self.named_parameters():
                if 'diffloss' in name:
                    diffloss_params.append(param)
                else:
                    other_params.append(param)
            params = [
                {'params': other_params},
                {'params': diffloss_params, 'lr': self.args.diffloss_lr},
            ]
        else:
            params = [{'params': self.parameters()}]
        return params

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            text_states,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            imgs: Optional[torch.Tensor] = None,   # imgs is continuous feature with shape (bs, c, h, w)
            imgs_mask: Optional[torch.Tensor] = None,  # mask for the image tokens
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = {"loss": None}
        cos = freqs_cos
        sin = freqs_sin

        # Patchify
        if imgs.ndim == 4:
            img_seqs, th, tw = self.patchify(imgs)
            img_len = th * tw
        elif imgs.ndim == 3:
            img_seqs = imgs
            img_len = img_seqs.size(1)
        else:
            raise ValueError(f"imgs should be 3D or 4D tensor, but got {imgs.ndim}D tensor.")

        proj_img_seqs = self.z_proj(img_seqs)  # token_embed_dim -> n_embd
        # Replace with mask token
        masked_img_seqs = proj_img_seqs.clone()
        masked_img_seqs[imgs_mask] = self.mask_token

        text_seqs = self.txt_in(text_states)
        text_len = text_seqs.size(1)

        x = torch.cat([text_seqs, masked_img_seqs], dim=1)

        for block_idx, block in enumerate(self.transformer.h):
            block_inputs = [x, cos, sin]
            if self.training and self.gradient_checkpoint and (
                    self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
            else:
                x = block(*block_inputs)
        x = self.transformer.ln_f(x)

        # Split out the image tokens
        z = x[:, text_len:]
        out["logits"] = z

        if not self.training:
            return out

        # image loss
        bs, seqlen, _ = img_seqs.shape
        diff_target = img_seqs.detach().reshape(bs * seqlen, -1).repeat(self.diff_batch_factor, 1)
        diff_z = z.reshape(bs * seqlen, -1).repeat(self.diff_batch_factor, 1)
        diff_mask = imgs_mask.reshape(bs * seqlen).repeat(self.diff_batch_factor)
        image_loss = self.diffloss(z=diff_z, target=diff_target, mask=diff_mask)
        out["image_loss"] = image_loss.detach()

        # total loss
        loss = image_loss
        out["loss"] = loss
        return out

    def patchify(self, x):
        bs, c, h, w = x.shape
        p = self.patch_size
        th, tw = h // p, w // p

        x = x.reshape(bs, c, th, p, tw, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bs, th * tw, c * p ** 2)
        return x, th, tw    # [bs, img_seq, token_dim]

    def unpatchify(self, x, th, tw):
        bs = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim

        x = x.reshape(bs, th, tw, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bs, c, th * p, tw * p)
        return x

    def enable_deterministic(self) -> None:
        raise NotImplementedError()

    def params_count(self):
        counts = {
            "attn+mlp": sum(
                [
                    sum(p.numel() for p in block.attn.parameters()) + sum(p.numel() for p in block.mlp.parameters())
                    for block in self.transformer.h
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        return counts
