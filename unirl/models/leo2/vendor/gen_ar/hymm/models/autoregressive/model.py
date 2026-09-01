# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.

Modified from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
"""

from argparse import Namespace
from typing import Any, Optional, Tuple, Dict, List
from typing_extensions import Self

import torch
import torch.nn as nn
from diffusers.models import ModelMixin
from einops import rearrange

from .utils import build_mask_cache, batched_index_select
from .attn_layers import CausalSelfAttention, SelfAttention
from .config import Config
from .mlp_layers import NaiveMLP
from ..basic.embed_layers import TimestepEmbedder
from ..basic.pos_emb_layers import build_rope_cache
from hymm.models.visual_encoders import load_vision_model


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_type=None,
        use_compile=False,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs)
        if attn_type == "full":
            self.attn = SelfAttention(config, block_idx, **factory_kwargs)
        else:
            self.attn = CausalSelfAttention(config, block_idx, **factory_kwargs)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs)

        self.return_moe_loss = config.get('return_moe_loss', False)
        if config.mlp_class_name in ["HunYuanMoE"]:
            self.mlp = config.moe_class(config, block_idx=block_idx, **factory_kwargs)
        else:
            self.mlp = config.mlp_class(config, **factory_kwargs)
            assert not self.return_moe_loss, "`return_moe_loss` should be False when `mlp_class_name` is not `HunYuanMoE`"
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs) if config.post_mlp_norm else nn.Identity()
        )

        self.config = config
        self.use_compile = use_compile

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      |
           |  ↓                     |  ↓                   |
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """
        x_normed = self.norm_1(x)

        if self.use_compile:
            @torch.compile
            def compile_block(x_normed, cos, sin, mask, input_pos):
                return self.attn(x_normed, cos, sin, mask, input_pos)

            attention_output = compile_block(x_normed, cos, sin, mask, input_pos)
        else:
            attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            mlp_output = self.mlp(x_normed)
            if self.return_moe_loss and isinstance(mlp_output, tuple):
                x = mlp_output[0] + attention_output + x
                moe_loss = mlp_output[1]
            else:
                x = mlp_output + attention_output + x
                moe_loss = None
        else:
            x = attention_output + x
            mlp_output = self.mlp(self.norm_2(x))
            if self.return_moe_loss and isinstance(mlp_output, tuple):
                x = self.post_mlp_norm(mlp_output[0]) + x
                moe_loss = mlp_output[1]
            else:
                x = self.post_mlp_norm(mlp_output) + x
                moe_loss = None
        
        if self.training and self.return_moe_loss and moe_loss is not None:
            return x, moe_loss
        else:
            return x


class GPT(ModelMixin):
    def __init__(
            self, args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.args = args
        self.config = config
        self.enable_sep_modal_adaptors = args.get("enable_sep_modal_adaptors", False)
        if not self.enable_sep_modal_adaptors:
            self.total_vocab_size = config.padded_vocab_size + args.media_vocab_size
        else:
            self.total_vocab_size = config.padded_vocab_size

        if self.args.add_iw_ih_token:
            self.w_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
            self.h_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

        self.lm_head = nn.Linear(config.n_embd, self.total_vocab_size, bias=config.lm_head_bias, **factory_kwargs)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.total_vocab_size, config.n_embd, **factory_kwargs),
                h=nn.ModuleList(Block(config, block_idx, **factory_kwargs) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs),
            )
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
    
    def get_training_parts(self, training_parts):
        """Automatically retrieves trainable parameters for optimizer initialization.
    
        This method should be called during optimizer/engine setup to collect parameters
        that require gradient updates. The current implementation enforces automatic
        selection based solely on requires_grad status. 

        Args:
            training_parts (str): Must be set to "auto" to enable automatic filtering
                of parameters based on their gradient requirements. Other values are
                reserved for future compatibility. Defined in `args.training_parts`

        Returns:
            generator: An iterable generator yielding model parameters with 
            requires_grad=True, suitable for passing to optimizer constructors
            (e.g., DeepSpeed, Adam, SGD).
        """
        assert (
            training_parts == "auto"
        ), f"Invalid 'training_parts' of {training_parts}, please set to 'auto'."

        return (param for param in self.parameters() if param.requires_grad)

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."
                " This is likely because the input text exceeds the supported context length of this model."
            )
        self._max_seq_length = value
        if self.config.rope_type == "default":
            if not hasattr(self, "cos"):
                # first call
                cos, sin = self.rope_cache()
                self.register_buffer("cos", cos, persistent=False)
                self.register_buffer("sin", sin, persistent=False)
            # override
            elif value != self.cos.size(0):
                self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

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
            idx: torch.Tensor,  # batch_size x seq_len-1
            target: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            input_pos: Optional[torch.Tensor] = None,
            iw_ih_scatter_index: Optional[torch.Tensor] = None,   # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2k  (w, h)
            no_iw_ih_scatter: bool = False,  # only used in kv cache inference
            text_loss_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            image_loss_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            only_image_loss: bool = False,
            image_loss_weight: float = 1.0,
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, dim=0, idx=input_pos)
            sin = batched_index_select(self.sin, dim=0, idx=input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, dim=2, idx=input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        elif freqs_cos is not None and freqs_sin is not None:   # use the provided frequencies
            cos = freqs_cos[:T]
            sin = freqs_sin[:T]
            mask = attention_mask
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = attention_mask

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if self.args.add_iw_ih_token and not no_iw_ih_scatter:
            assert iw_ih_scatter_index is not None and iw_ih_scatter_src is not None, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"

            batch_size = x.shape[0]
            n_embd = x.shape[-1]

            # batch_size x 2k x n_embd
            iw_ih_scatter_src = torch.cat([self.w_emb(iw_ih_scatter_src[:, 0::2].reshape(-1)).reshape(batch_size, -1, n_embd), self.h_emb(iw_ih_scatter_src[:, 1::2].reshape(-1)).reshape(batch_size, -1, n_embd)], dim=1)
            # batch_size x 2k 
            iw_ih_scatter_index = torch.cat([iw_ih_scatter_index[:, 0::2], iw_ih_scatter_index[:, 1::2]], dim=1)

            x = x.float()
            x.scatter_(
                dim=1,
                index=iw_ih_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=iw_ih_scatter_src.float(),
            )

        # For Gemma-series
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block_idx, block in enumerate(self.transformer.h):
            block_inputs = [x, cos, sin, mask, input_pos]
            if self.training and self.gradient_checkpoint and (
                    self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
            else:
                x = block(*block_inputs)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        x = x.float()   # follow show-o implementation to avoid numerical issues
        # For Gemma-series
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping

        # output
        out = {"logits": x, "loss": None}

        if not self.training:
            return out

        # only calculate the image loss for instruction tuning
        if only_image_loss:
            loss = torch.nn.functional.cross_entropy(
                x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
            )
            loss = image_loss_weight * (loss * image_loss_mask.view(-1)).sum() / image_loss_mask.sum()
        else:
            if image_loss_weight==1.0: # vanilla
                loss = torch.nn.functional.cross_entropy(
                    x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
                )
            else: # for instruction tuning
                loss = torch.nn.functional.cross_entropy(
                    x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
                )
                loss = image_loss_weight * loss * image_loss_mask.view(-1) + loss * (1 - image_loss_mask.view(-1))

        out["loss"] = loss
        if text_loss_mask is not None or image_loss_mask is not None:
            detach_loss = torch.nn.functional.cross_entropy(
                x.detach().view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
            )
            if text_loss_mask is not None:
                if (n_text := text_loss_mask.sum()) > 0:
                    out["text_loss"] = (detach_loss * text_loss_mask.view(-1)).sum() / n_text
                else:
                    out["text_loss"] = (detach_loss * text_loss_mask.view(-1)).sum()
            if image_loss_mask is not None:
                out["image_loss"] = (detach_loss * image_loss_mask.view(-1)).sum() / image_loss_mask.sum()

        return out

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # return cos and sin both with shape of (max_seq_length, rope_n_elem)
        # where rope_n_elem = rotary_percentage * head_size
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            base_rescale_factor=self.config.rope_base_rescale_factor,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation.
            # example: when max_seq_length = 10, mask_cache =
            # [[[[ True, False, False, False, False, False, False, False, False, False],
            #    [ True,  True, False, False, False, False, False, False, False, False],
            #    [ True,  True,  True, False, False, False, False, False, False, False],
            #    [ True,  True,  True,  True, False, False, False, False, False, False],
            #    [ True,  True,  True,  True,  True, False, False, False, False, False],
            #    [ True,  True,  True,  True,  True,  True, False, False, False, False],
            #    [ True,  True,  True,  True,  True,  True,  True, False, False, False],
            #    [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
            #    [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
            #    [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]]],
            # with shape of 1 x 1 x 10 x 10
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None

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


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


class DiscreteMultiModalGPT(GPT):
    def __init__(self, args: Namespace, config: Config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(args, config, **factory_kwargs)
        
        self.image_vocab_size = args.get("media_vocab_size", None)
        self.image_token_offset = args.get("image_token_offset", None)
        self.enable_image_und = args.get("enable_image_und", False)

        # whether to add image understanding modules
        if self.enable_image_und:
            self.vision_model_type = args.get("vision_model_type", None)
            assert self.vision_model_type is not None
            self.enable_train_visual_encoder = args.get("enable_train_visual_encoder", False)
            if self.enable_train_visual_encoder:
                self.visual_und_encoder = load_vision_model(
                    args.vision_model_type,
                    require_grad=True,
                    eval_mode=False,
                    device=device
                ).to(dtype=dtype)

            # TODO (yutaocui): support mlp choice for visual_und_aligner
            self.visual_und_aligner = NaiveMLP(
                mlp_depth=config.vision_mlp_depth,
                input_dim=config.vision_und_dim,
                inter_dim=config.n_embd,
                out_dim=config.n_embd,
                **factory_kwargs
            )

        if self.enable_sep_modal_adaptors:
            self.image_gen_wte = nn.Embedding(self.image_vocab_size, config.image_gen_intermediate_dim, **factory_kwargs)
            self.image_gen_aligner = NaiveMLP(
                mlp_depth=config.vision_mlp_depth,
                input_dim=config.image_gen_intermediate_dim,
                inter_dim=config.n_embd,
                out_dim=config.n_embd,
                **factory_kwargs
            )
            self.image_gen_head = NaiveMLP(
                mlp_depth=config.vision_mlp_depth,
                input_dim=config.n_embd,
                inter_dim=config.n_embd,
                out_dim=self.image_vocab_size,  # self.image_token_offset + self.image_vocab_size,
                **factory_kwargs
            )
            
            pretrained_image_gen_wte_pt = args.get("pretrained_image_gen_wte_pt", None)
            if pretrained_image_gen_wte_pt is not None:
                pretrained_image_gen_wte = torch.load(pretrained_image_gen_wte_pt)["image_gen_wte"]
                pretrained_image_gen_wte = pretrained_image_gen_wte.to(device=device, dtype=dtype)
                self.image_gen_wte.weight.data.copy_(pretrained_image_gen_wte)
    
        # Whether freezing the specified layers parameters
        self.frozen_layers = args.get("frozen_layers", None)
        if self.training and self.frozen_layers is not None:
            self.freeze_specified_layers()
    
    def freeze_specified_layers(self) -> None:
        """
        Freeze specified layers of the model based on rules defined in `self.args.frozen_layers`.
        
        Supported layer specification formats:
            - "wte": Freezes the word token embedding layer
            - "visual_und_encoder": Freezes the visual understanding encoder
            - "transformer_layers": Freezes all Transformer layers
            - "transformer_layersS-E": Freezes Transformer layers from index S to E (exclusive)
        Example: "transformer_layers0-6" freezes layers 0 to 5
        """
        for layer_rule in self.frozen_layers:
            # freeze the 'wte' layer
            if layer_rule == "wte":
                for param in self.transformer.wte.parameters():
                    param.requires_grad = False
            # freeze the transformer layers: from start to end indices
            elif layer_rule.startswith("transformer_layers"):
                try:
                    if layer_rule == "transformer_layers":
                        start = 0
                        end = self.config.n_layer
                    else:
                        layer_range = layer_rule.split("_layers")[-1]
                        start, end = map(int, layer_range.split("-"))
                    if start < 0 or end > self.config.n_layer:
                        raise ValueError(f"Invalid frozen layer range {layer_range}, model has only {self.config.n_layer} layers")
                    for layer_idx in range(start, end):
                        for param in self.transformer.h[layer_idx].parameters():
                            param.requires_grad = False
                except (ValueError, IndexError):
                    print(f"Invalid layer format: {layer_rule}, expected 'transformer_layers' or 'transformer_layersS-E'")
            elif layer_rule == "visual_und_encoder":
                for param in self.visual_und_encoder.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Unknown frozen rule: {layer_rule}")
          
    def separate_multimodal_embedding(self, idx, image_token_mask):
        """
        Params:
            idx: input token index. [batch_size, seq_len]
            image_token_mask: image token mask. [batch_size, seq_len]
        """
        batch_size, seq_len = idx.shape
        x = torch.zeros(
            (batch_size, seq_len, self.config.n_embd),
            device=idx.device,
            dtype=self.dtype,
        )
        
        if image_token_mask.any():
            image_indices = idx[image_token_mask]
            image_indices = image_indices - self.image_token_offset
            assert (
                (image_indices >= 0).all() and (image_indices < self.image_vocab_size).all()
            ), f"Not all image target elements satisfying: n >= 0 and n < {self.image_vocab_size}."
            image_embeds = self.image_gen_wte(image_indices)
            image_embeds = self.image_gen_aligner(image_embeds)
            x[image_token_mask] = image_embeds
            
        if (~image_token_mask).any():
            regular_indices = idx[~image_token_mask]
            regular_embeds = self.transformer.wte(regular_indices)
            x[~image_token_mask] = regular_embeds

        return x
    
    def separate_multimodal_head(self, x):
        """
        Params:
            x: input features. [batch_size, seq_len, n_embd]
            image_token_mask: image token mask. [batch_size, seq_len]
        """
        image_logits = self.image_gen_head(x)
        regular_logits = self.lm_head(x)

        return image_logits, regular_logits

    def generate_image_mask(
        self,
        idx: torch.Tensor,  # (batch_size, seq_len-1)
    ) -> torch.Tensor:
        assert idx.dtype in [torch.long, torch.int], "idx must be int or long type"
        mask = idx >= self.image_token_offset
        
        return mask.to(idx.device)

    def forward(
            self,
            idx: torch.Tensor,  # batch_size x seq_len-1
            target: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            input_pos: Optional[torch.Tensor] = None,
            iw_ih_scatter_index: Optional[torch.Tensor] = None,   # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2k  (w, h)
            no_iw_ih_scatter: bool = False,  # only used in kv cache inference
            text_loss_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            image_loss_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            image_token_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            only_image_loss: bool = False,
            image_loss_weight: float = 1.0,
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            data_type: Optional[str] = None,
            imgs_input: Optional[torch.Tensor] = None,  # batch_size x n_imgs x c x h x w, used for image understanding
            image_token_id: Optional[int] = None, # required for image understanding
            eff_images_num: Optional[List[int]] = None, # required for image understanding
            return_loss: Optional[bool] = None,  # used for grpo training
    ) -> Dict[str, Optional[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, dim=0, idx=input_pos)
            sin = batched_index_select(self.sin, dim=0, idx=input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, dim=2, idx=input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        elif freqs_cos is not None and freqs_sin is not None:   # use the provided frequencies
            cos = freqs_cos[:T]
            sin = freqs_sin[:T]
            mask = attention_mask
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = attention_mask

        if not self.enable_sep_modal_adaptors:
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        else:
            if image_token_mask is None:
                image_token_mask = self.generate_image_mask(idx)
            x = self.separate_multimodal_embedding(idx, image_token_mask)
        
        if self.enable_image_und and imgs_input is not None:
            images_token_mask = idx == image_token_id
            if images_token_mask.sum() > 0:
                bs, n_imgs = imgs_input.shape[:2]
                images = rearrange(imgs_input, "b n c h w -> (b n) c h w").to(idx.device, dtype=x.dtype)
                # [b x n, T2, D]
                images_embeds = self.visual_und_aligner(self.visual_und_encoder(images))
                n_tokens_per_image = images_embeds.shape[1]
                # [b x n, T2, D] -> [b, n x T2, D]
                images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n_imgs)
                # [b, n x T2]
                if eff_images_num is not None:
                    eff_images_num_tensor = torch.tensor(eff_images_num, device=idx.device)
                    position_indices = torch.arange(n_imgs, device=idx.device)  # (max_images,)
                    # (b, n)
                    images_emb_mask = position_indices.unsqueeze(0) < eff_images_num_tensor.unsqueeze(1)
                    # (b, n x T2)
                    images_emb_mask = images_emb_mask.unsqueeze(-1).repeat(1, 1, n_tokens_per_image).view(bs, -1)
                else:
                    # (b, n x T2)
                    images_emb_mask = torch.ones(bs, images_embeds.shape[1]).bool().to(idx.device)
                
                images_token_mask = images_token_mask.to(idx.device)
                assert (
                    images_emb_mask.sum() == images_token_mask.sum()
                ), f"masked token number should be equal between images_emb_mask: {images_emb_mask.sum()} and images_token_mask: {images_token_mask.sum()}."
                x[images_token_mask] = images_embeds[images_emb_mask]

        if self.args.add_iw_ih_token and not no_iw_ih_scatter:
            assert iw_ih_scatter_index is not None and iw_ih_scatter_src is not None, "iw_ih_scatter_index and iw_ih_scatter_src are required for adding iw and ih tokens"

            batch_size = x.shape[0]
            n_embd = x.shape[-1]

            # batch_size x 2k x n_embd
            iw_ih_scatter_src = torch.cat([self.w_emb(iw_ih_scatter_src[:, 0::2].reshape(-1)).reshape(batch_size, -1, n_embd), self.h_emb(iw_ih_scatter_src[:, 1::2].reshape(-1)).reshape(batch_size, -1, n_embd)], dim=1)
            # batch_size x 2k 
            iw_ih_scatter_index = torch.cat([iw_ih_scatter_index[:, 0::2], iw_ih_scatter_index[:, 1::2]], dim=1)

            x = x.float()
            x.scatter_(
                dim=1,
                index=iw_ih_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=iw_ih_scatter_src.float(),
            )

        # For Gemma-series
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block_idx, block in enumerate(self.transformer.h):
            block_inputs = [x, cos, sin, mask, input_pos]
            if self.training and self.gradient_checkpoint and (
                    self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
            else:
                x = block(*block_inputs)

        x = self.transformer.ln_f(x)

        if not self.enable_sep_modal_adaptors:
            x = self.lm_head(x)  # (b, t, vocab_size)
            x = x.float()   # follow show-o implementation to avoid numerical issues
            # For Gemma-series
            if self.config.final_logit_softcapping is not None:
                x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
            # output
            out = {"logits": x, "loss": None}
        else:
            image_logits, text_logits = self.separate_multimodal_head(x)
            image_logits = image_logits.float()
            text_logits = text_logits.float()
            # For Gemma-series
            if self.config.final_logit_softcapping is not None:
                image_logits = torch.tanh(image_logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
                text_logits = torch.tanh(text_logits / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
            
            out = {"image_logits": image_logits, "text_logits": text_logits, "loss": None}

        if not self.training or (return_loss is not None and not return_loss):
            return out
        
        # for calculating loss
        if self.enable_sep_modal_adaptors:
            target_image_token_mask = self.generate_image_mask(target)
            image_target = target - self.image_token_offset
            image_target[~target_image_token_mask] = -100
            if image_loss_mask is not None:
                target_mask = (image_loss_mask == 0).to(image_loss_mask.device)
                image_target = image_target.masked_fill(target_mask, -100)

            text_target = target.clone()
            text_target[target_image_token_mask] = -100
            if text_loss_mask is not None:
                target_mask = (text_loss_mask == 0).to(text_loss_mask.device)
                text_target = text_target.masked_fill(target_mask, -100)
        
        # only calculate the image loss for instruction tuning
        if only_image_loss:
            if not self.enable_sep_modal_adaptors:
                loss = torch.nn.functional.cross_entropy(
                    x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
                )
            else:
                loss = torch.nn.functional.cross_entropy(
                    image_logits.view(-1, image_logits.size(-1)), image_target.view(-1), ignore_index=-100, reduction="none"
                )
            loss = image_loss_weight * (loss * image_loss_mask.view(-1)).sum() / image_loss_mask.sum()
        else:
            if image_loss_weight==1.0: # vanilla
                if not self.enable_sep_modal_adaptors:
                    loss = torch.nn.functional.cross_entropy(
                        x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
                    )
                else:
                    text_loss = torch.nn.functional.cross_entropy(
                        text_logits.view(-1, text_logits.size(-1)), text_target.view(-1), ignore_index=-100, reduction="mean"
                    )
                    image_loss = torch.nn.functional.cross_entropy(
                        image_logits.view(-1, image_logits.size(-1)), image_target.view(-1), ignore_index=-100, reduction="mean"
                    )
                    loss = image_loss + text_loss
            else: # for instruction tuning
                if not self.enable_sep_modal_adaptors:
                    loss = torch.nn.functional.cross_entropy(
                        x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
                    )
                    loss = image_loss_weight * loss * image_loss_mask.view(-1) + loss * (1 - image_loss_mask.view(-1))
                else:
                    image_loss = torch.nn.functional.cross_entropy(
                        image_logits.view(-1, image_logits.size(-1)), image_target.view(-1), ignore_index=-100, reduction="mean"
                    )
                    text_loss = torch.nn.functional.cross_entropy(
                        text_logits.view(-1, text_logits.size(-1)), text_target.view(-1), ignore_index=-100, reduction="mean"
                    )
                    loss = image_loss_weight * image_loss + text_loss

        out["loss"] = loss
        if text_loss_mask is not None or image_loss_mask is not None or (self.enable_sep_modal_adaptors and data_type == "text"):
            if not self.enable_sep_modal_adaptors:
                detach_loss = torch.nn.functional.cross_entropy(
                    x.detach().view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="none"
                )
                if text_loss_mask is not None:
                    if (n_text := text_loss_mask.sum()) > 0:
                        out["text_loss"] = (detach_loss * text_loss_mask.view(-1)).sum() / n_text
                    else:
                        out["text_loss"] = (detach_loss * text_loss_mask.view(-1)).sum()
                if image_loss_mask is not None:
                    out["image_loss"] = (detach_loss * image_loss_mask.view(-1)).sum() / image_loss_mask.sum()
            else:
                detach_text_loss = torch.nn.functional.cross_entropy(
                    text_logits.detach().view(-1, text_logits.size(-1)), text_target.view(-1), ignore_index=-100, reduction="none"
                )
                detach_image_loss = torch.nn.functional.cross_entropy(
                    image_logits.detach().view(-1, image_logits.size(-1)), image_target.view(-1), ignore_index=-100, reduction="none"
                )
                if data_type == "text" or text_loss_mask is not None:
                    if (n_text := (text_loss_mask).sum()) > 0:
                        out["text_loss"] = detach_text_loss.sum() / n_text
                    else:
                        out["text_loss"] = detach_text_loss.sum()
                if image_loss_mask is not None:
                    out["image_loss"] = (detach_image_loss * image_loss_mask.view(-1)).sum() / image_loss_mask.sum()

        return out


class ContinuousMultiModalGPT(GPT):
    def __init__(self, args: Namespace, config: Config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(args, config, **factory_kwargs)
        clip_dim = args.clip_dim
        t5_dim = args.t5_dim
        self.use_t5 = args.use_t5
        self.clip_projection = nn.Linear(clip_dim, config.n_embd)
        if self.use_t5:
            self.t5_projection = nn.Linear(t5_dim, config.n_embd)
    
    @torch.no_grad()
    def forward_inference(
        self, 
        x, 
        input_pos, 
        logits_processors, 
        cfg_enabled, 
        idx,
        total_len,
        codebook_arrangement, 
        double_vocab, 
        dac_vocab, 
        slice_manualy,
        audio_token_offset
    ):
        cos = batched_index_select(self.cos, 0, input_pos)
        sin = batched_index_select(self.sin, 0, input_pos)
        if self.mask_cache is None:
            raise TypeError("You need to call `gpt.set_kv_cache()`")
        mask = batched_index_select(self.mask_cache, 2, input_pos)
        if mask.dim() > 4:
            # the mask cache has a batch dim of 1 in addition to the one
            # we get if input_pos has a batch dimension
            mask = mask.squeeze(1)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping
        logits = x[:, -1, :]
        if slice_manualy:
            if double_vocab:
                if codebook_arrangement == "interleave":
                    if idx % 2 == 0:
                        logits = logits[:, audio_token_offset:audio_token_offset + dac_vocab]
                    else:
                        logits = logits[:, audio_token_offset+dac_vocab:audio_token_offset + 2*dac_vocab]
                else:
                    if idx < total_len // 2:
                        logits = logits[:, audio_token_offset:audio_token_offset + dac_vocab]
                    else:
                        logits = logits[:, audio_token_offset+dac_vocab:audio_token_offset + 2*dac_vocab]
            else:
                logits = logits[:, audio_token_offset:audio_token_offset + dac_vocab]
        
        logits = logits_processors(x, logits)
        if cfg_enabled:
            logits, _ = torch.chunk(logits, chunks=2, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # self.logger.info(probs.shape)
        next_token = torch.multinomial(probs, num_samples=1)
        # self.logger.info(next_token.shape)
        if cfg_enabled:
            next_token = next_token.repeat((2, 1))
        return next_token
    
    @torch.no_grad()
    def generate(
        self, 
        clip_feat, 
        t5_feat, 
        prefix_token, 
        audio_token_len, 
        logits_processors, 
        device, 
        audio_token_offset=51200,
        dac_vocab=8192,
        double_vocab=True,
        cfg_enabled=True,
        codebook_arrangement="interleave",
        slice_manualy=False,
        show_progress=False,
    ):
        """ generate audio token

        Args:
            clip_feat (torch.Tensor): (b, clip_dim), padded and cfged clip feat
            t5_feat (torch.Tensor): (b, t5_dim), padded and cfged t5 feat
            prefix_token (torch.Tensor): (b, 273), prefix sequence <bos><bov><video>*40<eov><bot><text>*227<eot><boa>
        """
        from tqdm import tqdm
        x = self.transformer.wte(prefix_token)
        clip_emb = self.clip_projection(clip_feat)  # (b, t, n_embd)
        clip_end = self.args.clip_start + clip_emb.shape[1]
        x[:, self.args.clip_start:clip_end] = clip_emb

        if self.use_t5:
            t5_emb = self.t5_projection(t5_feat)        # (b, t, n_embd) 
            t5_end = self.args.t5_start + t5_emb.shape[1]
            x[:, self.args.t5_start:t5_end] = t5_emb
        current_pos = x.shape[1]
        input_pos = torch.arange(0, x.shape[1], device=device, dtype=torch.long)
        audio_token_id = torch.empty((x.shape[0], 0), dtype=torch.long, device=device)
        # first round kv cache, rope cache
        for idx in tqdm(range(audio_token_len), disable=not show_progress):
            audio_token = self.forward_inference(
                x, 
                input_pos,
                logits_processors,
                cfg_enabled, 
                idx,
                audio_token_len,
                codebook_arrangement,
                double_vocab,
                dac_vocab, 
                slice_manualy,
                audio_token_offset
            )
            if double_vocab:
                if codebook_arrangement == "interleave":
                    if idx % 2 == 0:
                        next_token = audio_token + audio_token_offset
                    else:
                        next_token = audio_token + audio_token_offset + dac_vocab
                else:
                    if idx < audio_token_len // 2:
                        next_token = audio_token + audio_token_offset
                    else:
                        next_token = audio_token + audio_token_offset + dac_vocab
            else:
                next_token = audio_token + audio_token_offset
            audio_token_id = torch.cat([audio_token_id, audio_token], dim=-1)
            # next round input
            input_pos = torch.tensor([current_pos], device=device, dtype=torch.long)
            x = self.transformer.wte(next_token)
            current_pos += 1
        if cfg_enabled:
            audio_token_id, _ = torch.chunk(audio_token_id, chunks=2, dim=0)
        return audio_token_id

    def forward(
        self, 
        idx: torch.Tensor, 
        clip_feat: torch.Tensor,
        t5_feat: torch.Tensor,
        target: Optional[torch.Tensor] = None, 
        input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # replace feature token index with clip feature
        clip_emb = self.clip_projection(clip_feat)  # (b, t, n_embd)
        clip_end = self.args.clip_start + clip_emb.shape[1]
        x[:, self.args.clip_start:clip_end] = clip_emb

        if self.use_t5:
            t5_emb = self.t5_projection(t5_feat)        # (b, t, n_embd) 
            t5_end = self.args.t5_start + t5_emb.shape[1]
            x[:, self.args.t5_start:t5_end] = t5_emb

        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)

        x = self.lm_head(x)  # (b, t, vocab_size)
        x = x.float()
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping

        if not self.training:
            return {"logits": x, "loss": None}
        loss = torch.nn.functional.cross_entropy(
            x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
        )
        return {"logits": x, "loss": loss}


class ContinuousMultiHeadGPT(GPT):
    def __init__(self, args: Namespace, config: Config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        args.media_vocab_size = 0       # we hack here to avoid assertation error in config sanity check
        super().__init__(args, config, **factory_kwargs)
        clip_dim = args.clip_dim
        t5_dim = args.t5_dim
        self.n_codebook = args.n_codebook
        self.audio_vocab_size = args.audio_vocab_size
        self.use_t5 = args.use_t5
        self.clip_projection = nn.Linear(clip_dim, config.n_embd)
        if self.use_t5:
            self.t5_projection = nn.Linear(t5_dim, config.n_embd)
            self.t5_projection.apply(self._init_weights)
        self.audio_embds = nn.ModuleList(
            [nn.Embedding(self.audio_vocab_size, config.n_embd, **factory_kwargs) for _ in range(self.n_codebook)]
        )
        self.audio_heads = nn.ModuleList(
            [nn.Linear(config.n_embd, self.audio_vocab_size, **factory_kwargs) for _ in range(self.n_codebook)]
        )
        self.audio_embds.apply(self._init_weights)
        self.audio_heads.apply(self._init_weights)
        self.clip_projection.apply(self._init_weights)

    def forward(
        self, 
        idx: torch.Tensor,              # B, T
        audio_token: torch.Tensor,      # B, n_codebook, T_audio
        clip_feat: torch.Tensor,
        t5_feat: torch.Tensor,
        target: Optional[torch.Tensor] = None,  # B, n_codebook, T_audio
        input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # replace feature token index with clip feature
        clip_emb = self.clip_projection(clip_feat)  # (b, t, n_embd)
        clip_end = self.args.clip_start + clip_emb.shape[1]
        x[:, self.args.clip_start:clip_end] = clip_emb

        if self.use_t5:
            t5_emb = self.t5_projection(t5_feat)        # (b, t, n_embd) 
            t5_end = self.args.t5_start + t5_emb.shape[1]
            x[:, self.args.t5_start:t5_end] = t5_emb
        
        # replace audio token with new embeds
        audio_embds_list = [self.audio_embds[i](audio_token[:, i, :]) for i in range(self.n_codebook)]
        audio_embds = torch.stack(audio_embds_list, dim=3).sum(3)
        audio_ends = self.args.audio_start + audio_embds.shape[1]
        x[:, self.args.audio_start:audio_ends] = audio_embds

        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)

        # added with multi head ce loss
        gt_start = self.args.audio_start - 1        # shift left by 1 to force next token prediction
        gt_end = gt_start + audio_embds.shape[1]
        x = x[:, gt_start:gt_end]
        logits_list = [self.audio_heads[i](x) for i in range(self.n_codebook)] # [(b, t, audio_vocab_size)] * n_codebook
        x = torch.stack(logits_list, dim=1).float()       # (b, n_codebook, t, audio_vocab_size)

        # x = self.lm_head(x)  # (b, t, vocab_size)
        # x = x.float()
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping

        if not self.training:
            return {"logits": x, "loss": None}
        loss = torch.nn.functional.cross_entropy(
            x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
        )
        return {"logits": x, "loss": loss}
