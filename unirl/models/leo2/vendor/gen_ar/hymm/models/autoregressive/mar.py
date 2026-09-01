from argparse import Namespace
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import Config
from .model import GPT, ckpt_wrapper
from ..basic.diffloss import DiffLoss


class MarGPT(GPT):
    def __init__(
            self, args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(args, config, dtype=dtype, device=device)

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

    def forward(
            self,
            idx: torch.Tensor,  # batch_size x seq_len-1
            target: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            input_pos: Optional[torch.Tensor] = None,
            iw_ih_scatter_index: Optional[torch.Tensor] = None,   # batch_size x 2  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2  (w, h)
            no_iw_ih_scatter: bool = False,  # only used in kv cache inference
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            imgs: Optional[torch.Tensor] = None,   # imgs is continuous feature with shape (bs, c, h, w)
            imgs_pos: Optional[torch.Tensor] = None,  # start position of each image in each sequence. (bs)
            imgs_mask: Optional[torch.Tensor] = None,  # mask for the image tokens
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = {"loss": None}
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if freqs_cos is not None and freqs_sin is not None:   # use the provided frequencies
            cos = freqs_cos
            sin = freqs_sin
            mask = attention_mask
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = attention_mask

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if self.args.add_iw_ih_token and not no_iw_ih_scatter:
            assert iw_ih_scatter_index is not None and iw_ih_scatter_src is not None, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"

            n_embd = x.shape[-1]
            # batch_size x 2 x n_embd
            iw_ih_scatter_src = torch.cat([self.w_emb(iw_ih_scatter_src[:, 0]).unsqueeze(1), self.h_emb(iw_ih_scatter_src[:, 1]).unsqueeze(1)], dim=1)
            x.scatter_(
                dim=1,
                index=iw_ih_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=iw_ih_scatter_src.to(dtype=x.dtype)
            )

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

        # Assign img_seqs to the proper positions of x
        img_aligned = (imgs_pos == imgs_pos[0]).all()
        if img_aligned:
            x[:, imgs_pos[0]:imgs_pos[0] + img_len] = masked_img_seqs
        else:
            for i in range(x.size(0)):
                x[i, imgs_pos[i]:imgs_pos[i] + img_len] = masked_img_seqs[i]

        for block_idx, block in enumerate(self.transformer.h):
            block_inputs = [x, cos, sin, mask, input_pos]
            if self.training and self.gradient_checkpoint and (
                    self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
            else:
                x = block(*block_inputs)
        x = self.transformer.ln_f(x)

        # Split out the image tokens
        if img_aligned:
            z = x[:, imgs_pos[0]:imgs_pos[0] + img_len]
        else:
            img_outs = []
            for i in range(x.size(0)):
                img_outs.append(x[i, imgs_pos[i]:imgs_pos[i] + img_len])
            z = torch.stack(img_outs, dim=0)
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

        #  text loss
        x = self.lm_head(x)  # (b, t, vocab_size)
        text_loss = torch.nn.functional.cross_entropy(
            x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
        )
        out["text_loss"] = text_loss.detach()

        # total loss
        loss = text_loss + image_loss * self.args.get('diffloss_weight', 1.0)
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
