# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.

Modified from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
"""

from argparse import Namespace
from typing import Any, Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from diffusers.models import ModelMixin
from typing_extensions import Self

from hymm.parallelism.parallel_states import get_parallel_state

try:
    from megatron import get_args, print_rank_0, mpu # type: ignore
    from megatron.model import GPTModel # type: ignore
except (ModuleNotFoundError, ImportError):
    print("not found ptm")

from .utils import build_mask_cache, batched_index_select, real_batched_index_select
from .model import Block
from .hunyuan import HunYuanModel
from .config import Config
from ..basic import TokenMode
from ..basic.embed_layers import TimestepEmbedder
from ..basic.patch_embed_layers import UNetDown, UNetUp, PatchEmbed, FinalLayer, LinearPatchEmbed, LinearFinalLayer
from ..basic.pos_emb_layers import build_rope_cache, get_batch_text_image_2d_rope
from ...constants import WTE_LN_F_PATH
from hymm.parallelism.utils import batch_tensor_to_obj
from hymm.ar.pipelines.cache_utils import TaylorCacheContainer, CacheWithFreqsContainer

# Type aliases
BatchRaggedImages = Union[torch.Tensor, List[Union[torch.Tensor, List[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, List[torch.Tensor]]


def at_least_2d(x: torch.Tensor, dim: int):
    return x.unsqueeze(dim) if x.ndim == 1 else x


class Transfusion(ModelMixin):
    def __init__(
            self, args: Namespace,
            config: Config,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            hf_config=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.args = args
        self.config = config
        self.total_vocab_size = config.padded_vocab_size + args.media_vocab_size

        self.use_ptm = args.get('use_ptm', False)
        self.use_hf = args.get('use_hf', False)
        self.use_pure_torch = hasattr(args, "launcher") and args.launcher == "pure_torch"

        if self.use_pure_torch:
            self.enable_pp = get_parallel_state().pp_enabled
            if self.enable_pp:
                assert args.get('use_final_time_embed', False)
            config.launcher = args.launcher
            if self.enable_pp:
                # dirty hack
                self.pre_process = dist.get_rank() == get_parallel_state().pp_mesh.mesh[0]
                self.post_process = dist.get_rank() == get_parallel_state().pp_mesh.mesh[-1]
            else:
                self.pre_process = True
                self.post_process = True
        elif self.use_ptm:
            self.enable_pp = True if mpu.get_pipeline_model_parallel_world_size() else False
            self.pre_process = mpu.is_pipeline_first_stage()
            self.post_process = mpu.is_pipeline_last_stage()
        else:
            self.enable_pp = False
            self.pre_process = True
            self.post_process = True

        self.add_timestep_token = self.args.add_timestep_token
        if self.add_timestep_token and self.pre_process:
            self.timestep_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

        self.add_timestep_r_token = self.args.get('add_timestep_r_token', False)
        if self.add_timestep_r_token and self.pre_process:
            self.timestep_r_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

        self.add_guidance_token = self.args.get('add_guidance_token', False)
        if self.add_guidance_token and self.pre_process:
            self.guidance_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.use_front_boi_token = self.args.use_front_boi_token
        if self.add_iw_ih_token and self.pre_process:
            self.w_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
            self.h_emb = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

        self.rope_type = self.args.rope_type

        # Whether to use a separated time_embed layer for final_layer. When pp parallel enabled,
        # it is useful to avoid the dependence of last stage on the first stage.
        self.use_final_time_embed = args.get('use_final_time_embed', False)
        self.patch_size = args.patch_size


        self.img_proj_type = self.args.img_proj_type
        if self.img_proj_type == "unet":
            if self.pre_process:
                self.patch_embed = UNetDown(
                    patch_size=args.patch_size,
                    emb_channels=config.n_embd,
                    in_channels=args.vae_latent_dim,
                    hidden_channels=args.patch_embed_hidden_dim,
                    out_channels=config.n_embd,
                    **factory_kwargs
                )
                self.time_embed = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

            if self.post_process:
                self.final_layer = UNetUp(
                    patch_size=args.patch_size,
                    emb_channels=config.n_embd,
                    in_channels=config.n_embd,
                    hidden_channels=args.patch_embed_hidden_dim,
                    out_channels=args.vae_latent_dim,
                    out_norm=args.get('unet_out_norm', False),
                    **factory_kwargs
                )
                if self.use_final_time_embed:
                    self.time_embed_2 = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
        elif self.img_proj_type == "linear":
            if self.pre_process:
                self.patch_embed = PatchEmbed(
                    patch_size=args.patch_size,
                    in_chans=args.vae_latent_dim,
                    embed_dim=config.n_embd,
                    act_layer=nn.SiLU,
                    **factory_kwargs
                )
                self.time_embed = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)

            if self.post_process:
                self.final_layer = FinalLayer(
                    hidden_size=config.n_embd,
                    patch_size=args.patch_size,
                    out_channels=args.vae_latent_dim,
                    act_layer=nn.SiLU,
                    **factory_kwargs
                )
                if self.use_final_time_embed:
                    self.time_embed_2 = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
        elif self.img_proj_type == "siglip2":
            in_channel = args.vision_encoder_latent_dim
            if hasattr(args, "compression_net_type"):
                from hymm.models.visual_encoders.siglip2.compression import CompressionNet
                in_channel = CompressionNet.config[args.compression_net_type]["out_channels"]
            if self.pre_process:
                self.patch_embed = LinearPatchEmbed(
                    in_chans=in_channel,
                    embed_dim=config.n_embd,
                    act_layer=nn.SiLU,
                    **factory_kwargs
                )
                self.time_embed = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
            
            if self.post_process:
                self.final_layer = LinearFinalLayer(
                    hidden_size=config.n_embd,
                    out_channels=in_channel,
                    act_layer=nn.SiLU,
                    **factory_kwargs
                )
                if self.use_final_time_embed:
                    self.time_embed_2 = TimestepEmbedder(hidden_size=config.n_embd, **factory_kwargs)
        else:
            raise ValueError(f"img_proj_type {self.img_proj_type} not supported")
        if self.post_process:
            if config.lm_head_reuse_embedding:
                if config.lm_head_bias:
                    self.lm_head_bias = nn.Parameter(torch.zeros(self.total_vocab_size, **factory_kwargs))
                else:
                    self.lm_head_bias = None
            else:
                self.lm_head = nn.Linear(config.n_embd, self.total_vocab_size, bias=config.lm_head_bias, **factory_kwargs)
                # Freeze lm_head and ln_f if no text generation during training, otherwise error occurs when loading resumed optimizer states
                if self.args.get("lm_head_ln_freeze", False):
                    self.lm_head.requires_grad_(False)

        block_kwargs = dict(
            use_compile=args.use_compile,   # torch.compile attention block online
        )

        if self.use_pure_torch:
            config.return_moe_loss = self.args.get('return_moe_loss', False)
            if self.enable_pp:
                if args.get('pp_splits', None) is not None:
                    splits = args.pp_splits.split(',')
                    splits = [int(split) for split in splits]
                    assert sum(splits) == config.n_layer, f'Invalid pp-splits argument {args.pp_splits} for a {config.n_layer}-layer model.'
                    pp_rank = get_parallel_state().pp_mesh.get_local_rank()
                    start = sum(splits[:pp_rank])
                    end = start + splits[pp_rank]
                else:
                    n_layer = config.n_layer // get_parallel_state().pp_mesh.size()
                    start = get_parallel_state().pp_mesh.get_local_rank() * n_layer
                    end = start + n_layer
            else:
                start = 0
                end = config.n_layer

            def get_block(block_idx):
                if block_idx < start or block_idx >= end:
                    return None
                return Block(config, block_idx, **factory_kwargs, **block_kwargs)

            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(self.total_vocab_size, config.n_embd, **factory_kwargs) if self.pre_process else None,
                    h=nn.ModuleList(get_block(block_idx) for block_idx in range(config.n_layer)),
                    ln_f=config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs) if self.post_process else None,
                )
            )

            if self.args.get("moe_gate_freeze", False):
                for i, block in enumerate(self.transformer.h):
                    if block is not None:
                        block.mlp.gate.requires_grad_(False)
                        print(f"rank:{torch.distributed.get_rank()}, freezed moe gate for block-{i}")

            # Freeze lm_head and ln_f if no text generation during rl training
            if self.args.get("lm_head_ln_freeze", False) and self.post_process:
                self.transformer.ln_f.requires_grad_(False)

        elif self.use_ptm:
            self.ptm_transformer = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=self.pre_process,
                post_process=False,
                pad_id=128009,
            )

            module_dict = dict()
            if self.pre_process:
                module_dict['wte'] = nn.Embedding(self.total_vocab_size, config.n_embd, **factory_kwargs)
            if self.post_process:
                module_dict['ln_f'] = config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs)
            self.transformer = nn.ModuleDict(module_dict)

            #load wte and ln_f
            self.load_wte_ln_f(args)
            print(f"rank:{torch.distributed.get_rank()}, ptm_transformer:{self.ptm_transformer}")

        elif self.use_hf:
            self.transformer = HunYuanModel(hf_config)
            if dtype is not None:
                self.transformer.to(dtype=dtype)

        else:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(self.total_vocab_size, config.n_embd, **factory_kwargs),
                    h=nn.ModuleList(Block(config, block_idx, **factory_kwargs, **block_kwargs) for block_idx in range(config.n_layer)),
                    ln_f=config.norm_class(config.n_embd, eps=config.norm_eps, **factory_kwargs),
                )
            )

        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        # RoPE cache
        self.cos_cache = None
        self.sin_cache = None

        # Gradient checkpoint
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= config.n_layer, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={config.n_layer}."
        # Gather text tokens to calculate discrete loss for saving memory
        self.gather_text_tokens = args.get('gather_text_tokens', False)
        #taylor cache
        self.use_taylor_cache = args.get('use_taylor_cache', False)        

    def load_wte_ln_f(self, args):
        device = None
        if self.pre_process:
            device = self.transformer.wte.weight.device
        if self.post_process:
            device = self.transformer.ln_f.weight.device
        if device is None:
            return

        path = WTE_LN_F_PATH[args.pretrained_ckpt]
        state_dict = torch.load(path, weights_only=True, map_location=device)
        #print("state_dict", state_dict)
        #print("before loading:", self.transformer.state_dict())
        if not self.pre_process:
            del state_dict['wte.weight']
        if not self.post_process:
            del state_dict['ln_f.weight']
        self.transformer.load_state_dict(state_dict, strict=False, assign=True)
        #print("after loading:", self.transformer.state_dict())

        # load lm_head
        if hasattr(self, 'lm_head'):
            import collections
            lm_head_state_dict = collections.OrderedDict({'weight':state_dict['lm_head.weight']})
            self.lm_head.load_state_dict(lm_head_state_dict, strict=True, assign=True)

        print("wte and ln_f loaded!")

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

    def instantiate_und_image_tokens(
            self,
            x: torch.Tensor,
            image_embeds: Union[torch.Tensor, List[torch.Tensor]],
            image_masks: torch.Tensor,
    ):
        """ In Transfusion, we only consider two image modes: gen_image and und_image.
        The `gen_image` mode means that the image tokens are generated from previous tokens in a transfusion way.
        The `und_image` mode means that the image tokens are used as embeddings, and may include a few tasks like:
            1. instruction tuning: subject-driven, inpainting, editing, etc.
            2. image understanding: VQA, image captioning, etc.

        This function is used to instantiate the image tokens in the `und_image` mode.
        """
        batch_size, seq_len, n_embd = x.shape
        index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        if isinstance(image_embeds, list):
            for i, (image_embed, mask) in enumerate(zip(image_embeds, image_masks)):
                image_scatter_index = index[i:i+1].masked_select(mask.bool()).reshape(1, -1)
                # # if image_embeds has different dtype with x, cast it to x.dtype, but we should check why it happens
                # if image_embed.dtype != x.dtype:
                #     image_embed = image_embed.to(x.dtype)
                x[i:i+1].scatter_(
                    dim=1,
                    index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_embed.reshape(1, -1, n_embd),
                )
        else:
            image_scatter_index = index.masked_select(image_masks.bool()).reshape(batch_size, -1)
            # # if image_embeds has different dtype with x, cast it to x.dtype, but we should check why it happens
            # if image_embeds.dtype != x.dtype:
            #     image_embeds = image_embeds.to(x.dtype)
            x.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_embeds,
            )

        return x

    def _get_token_size(self, image):
        assert image.shape[-2] % self.patch_size == 0
        assert image.shape[-1] % self.patch_size == 0
        token_h = int(image.shape[-2] // self.patch_size)
        token_w = int(image.shape[-1] // self.patch_size)
        return token_h, token_w

    def get_token_sizes(self, images):
        if isinstance(images, list):
            token_h, token_w = [], []
            for i, image_i in enumerate(images):
                if isinstance(image_i, torch.Tensor):
                    token_h_i, token_w_i = self._get_token_size(image_i)
                elif isinstance(image_i, list):
                    token_h_i, token_w_i = [], []
                    for j in range(len(image_i)):
                        image_ij = image_i[j]
                        token_h_ij, token_w_ij = self._get_token_size(image_ij)
                        token_h_i.append(token_h_ij)
                        token_w_i.append(token_w_ij)
                else:
                    raise ValueError(f"image_i should be a tensor or a list of tensors, got {type(image_i)}")
                token_h.append(token_h_i)
                token_w.append(token_w_i)

        else:
            token_h, token_w = self._get_token_size(images)
        return token_h, token_w

    def _instantiate_ragged_image_tokens(self, x, images, ts, image_mask, return_t_emb_and_size=False, guidance=None):
        """
        `images` can be a tensor, or a list of tensor or a list of lists of tensor.
        - If `images` is a list of tensor, the length of the list is batch_size. Each tensor is
              n_{i} x c x h x w, n_{i} is the number of images of the i-th sample.
        - If `images` is a list of lists of tensor, the length of the first list is batch_size. Each list contains
              n_{i} (c x h_{ij} x w_{ij}) tensors, n_{i} is the number of images of the i-th sample.
        `ts` is always a tensor or a list of tensors, the length of the list is batch_size. Each tensor is
            n_{i}, n_{i} is the number of images of the i-th sample.
        """
        batch_size, seq_len, n_embd = x.shape

        if isinstance(images, list):
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb, token_h, token_w = [], [], []
            for i, (image_i, t_i) in enumerate(zip(images, ts)):
                if isinstance(image_i, torch.Tensor):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)
                    if (not self.add_guidance_token) and guidance is not None:
                        t_i_emb = t_i_emb + self.guidance_emb(guidance[i:i + 1])
                    # n_{i} x one_image_seq_len x n_embd
                    image_i_seq, token_h_i, token_w_i = self.patch_embed(image_i, t_i_emb)
                    # 1 x (n_{i} * one_image_seq_len)
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x (n_{i} * one_image_seq_len) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x (n_{i} * one_image_seq_len) x n_embd
                    )
                    t_emb.append(t_i_emb)
                    token_h.append(token_h_i)
                    token_w.append(token_w_i)
                elif isinstance(image_i, list):
                    # time_embed needs a 1-D tensor as input
                    t_i_emb = self.time_embed(t_i)  # n_{i} x d
                    if (not self.add_guidance_token) and guidance is not None:
                        t_i_emb = t_i_emb + self.guidance_emb(guidance[i:i + 1])

                    image_i_seq_list, token_h_i, token_w_i = [], [], []
                    for j in range(len(image_i)):
                        image_ij = image_i[j]
                        if image_ij.dim() == 4:
                            assert image_i[j].shape[0] == 1, "image_i[j] should have a batch dimension of 1"
                        elif image_ij.dim() == 3:
                            image_ij = image_ij.unsqueeze(0)
                        else:
                            raise ValueError(f"image_i[j] should have 3 or 4 dimensions, got {image_ij.dim()}")
                        # 1 x one_image_seq_len_{j} x n_embd
                        image_i_seq_j, token_h_ij, token_w_ij = self.patch_embed(image_ij, t_i_emb[j:j + 1])
                        image_i_seq_list.append(image_i_seq_j)
                        token_h_i.append(token_h_ij)
                        token_w_i.append(token_w_ij)
                    # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    image_i_seq = torch.cat(image_i_seq_list, dim=1)
                    # 1 x sum_{j}(one_image_seq_len_{j})
                    image_i_scatter_index = index[i:i + 1].masked_select(image_mask[i:i + 1].bool()).reshape(1, -1)
                    x[i:i + 1].scatter_(
                        dim=1,
                        index=image_i_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                        # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                        src=image_i_seq.reshape(1, -1, n_embd),  # 1 x sum_{j}(one_image_seq_len_{j}) x n_embd
                    )
                    t_emb.append(t_i_emb)
                    token_h.append(token_h_i)
                    token_w.append(token_w_i)

        elif images.shape[0] > batch_size:
            # In this case, each sequence in x has multiple images
            t_emb = self.time_embed(ts)
            if (not self.add_guidance_token) and guidance is not None:
                t_emb = t_emb + self.guidance_emb(guidance)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            index = torch.arange(batch_size * seq_len, device=x.device)
            x.view(-1, n_embd).scatter_(
                dim=0,
                index=index.masked_select(image_mask.view(-1).bool()).unsqueeze(-1).repeat(1, n_embd),
                src=image_seq.reshape(-1, n_embd),
            )

        else:
            index = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
            t_emb = self.time_embed(ts)
            if (not self.add_guidance_token) and guidance is not None:
                t_emb = t_emb + self.guidance_emb(guidance)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)
            image_scatter_index = index.masked_select(image_mask.bool()).reshape(batch_size, -1)

            x.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )

        if return_t_emb_and_size:
            return x, t_emb, token_h, token_w

        return x

    def instantiate_gen_image_tokens(
            self,
            x: torch.Tensor,
            x_t: Optional[BatchRaggedImages] = None,
            t: Optional[BatchRaggedTensor] = None,
            image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            guidance: Optional[torch.Tensor] = None,
    ):
        """ In Transfusion, we only consider two image modes: gen_image and und_image.
        The `gen_image` mode means that the image tokens are generated from previous tokens in a transfusion way.
        The `und_image` mode means that the image tokens are used as embeddings, and may include a few tasks like:
            1. instruction tuning: subject-driven, inpainting, editing, etc.
            2. image understanding: VQA, image captioning, etc.

        This function is used to instantiate the image tokens in the `gen_image` mode.
        """
        if x_t is not None:
            if self.use_final_time_embed:
                x = self._instantiate_ragged_image_tokens(x, x_t, t, image_mask, guidance=guidance)
            else:
                x, t_emb, token_h, token_w = self._instantiate_ragged_image_tokens(
                    x, x_t, t, image_mask, return_t_emb_and_size=True, guidance=guidance)
        else:
            t_emb, token_h, token_w = None, None, None

        if self.use_final_time_embed:
            return x
        else:
            return x, t_emb, token_h, token_w

    def instantiate_src_image_tokens(
            self,
            x: torch.Tensor,
            src_x: BatchRaggedImages,
            src_t: BatchRaggedTensor,
            src_image_mask: torch.Tensor,
    ):
        if src_x is not None:
            x = self._instantiate_ragged_image_tokens(x, src_x, src_t, src_image_mask)
        return x

    def instantiate_scatter(
            self,
            x: torch.Tensor,
            t: Optional[BatchRaggedTensor] = None,
            src_t: Optional[BatchRaggedTensor] = None,
            iw_ih_scatter_index: Optional[BatchRaggedTensor] = None,
            iw_ih_scatter_src: Optional[BatchRaggedTensor] = None,
            timestep_scatter_index: Optional[BatchRaggedTensor] = None,
            timestep_scatter_src: Optional[BatchRaggedTensor] = None,
            guidance_scatter_index: Optional[BatchRaggedTensor] = None,
            guidance: Optional[torch.Tensor] = None,
            timestep_r_scatter_index: Optional[BatchRaggedTensor] = None,
            r: Optional[BatchRaggedTensor] = None,
            n_samples: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, n_embd = x.shape

        if self.add_guidance_token and guidance is not None and guidance_scatter_index is not None:
            if isinstance(guidance_scatter_index, list):
                assert len(guidance_scatter_index) == 1, "guidance_scatter_index should have length 1"
                for i, gs in enumerate([guidance]):
                    # 1 x n_{i} x n_embd
                    guidance_scatter_src = self.guidance_emb(gs).reshape(1, -1, n_embd)

                    x[i:i+1].scatter_(
                        dim=1,
                        index=guidance_scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                        src=guidance_scatter_src,
                    )
            elif guidance.shape[0] > batch_size:
                guidance_scatter_src = self.guidance_emb(guidance.reshape(-1))
                offset = torch.repeat_interleave(n_samples) * seq_len
                x.view(-1, n_embd).scatter_(
                    dim=0,
                    index=(guidance_scatter_index + offset).unsqueeze(-1).repeat(1, n_embd),
                    src=guidance_scatter_src,
                )
            else:
                # batch_size x n x n_embd
                guidance_scatter_src = self.guidance_emb(guidance.reshape(-1)).reshape(batch_size, -1, n_embd)
                x.scatter_(
                    dim=1,
                    index=guidance_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=guidance_scatter_src,
                )

        if self.add_timestep_r_token and r is not None and timestep_r_scatter_index is not None:
            if isinstance(r, list):
                for i, rs in enumerate(r):
                    # 1 x n_{i} x n_embd
                    timestep_r_scatter_src = self.timestep_r_emb(rs).reshape(1, -1, n_embd)

                    x[i:i+1].scatter_(
                        dim=1,
                        index=timestep_r_scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                        src=timestep_r_scatter_src,
                    )
            elif r.shape[0] > batch_size:
                timestep_r_scatter_src = self.timestep_r_emb(r.reshape(-1))
                offset = torch.repeat_interleave(n_samples) * seq_len
                x.view(-1, n_embd).scatter_(
                    dim=0,
                    index=(timestep_r_scatter_index + offset).unsqueeze(-1).repeat(1, n_embd),
                    src=timestep_r_scatter_src,
                )
            else:
                # batch_size x n x n_embd
                timestep_r_scatter_src = self.timestep_r_emb(r.reshape(-1)).reshape(batch_size, -1, n_embd)
                x.scatter_(
                    dim=1,
                    index=timestep_r_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_r_scatter_src,
                )

        if self.add_timestep_token and timestep_scatter_index is not None:
            if timestep_scatter_src is None:
                if src_t is None:
                    if isinstance(t, list):
                        for i, ts in enumerate(t):
                            # 1 x n_{i} x n_embd
                            timestep_scatter_src = self.timestep_emb(ts).reshape(1, -1, n_embd)

                            x[i:i+1].scatter_(
                                dim=1,
                                index=timestep_scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd),
                                src=timestep_scatter_src,
                            )
                    elif t.shape[0] > batch_size:
                        timestep_scatter_src = self.timestep_emb(t.reshape(-1))
                        offset = torch.repeat_interleave(n_samples) * seq_len
                        x.view(-1, n_embd).scatter_(
                            dim=0,
                            index=(timestep_scatter_index + offset).unsqueeze(-1).repeat(1, n_embd),
                            src=timestep_scatter_src,
                        )
                    else:
                        # batch_size x n x n_embd
                        timestep_scatter_src = self.timestep_emb(t.reshape(-1)).reshape(batch_size, -1, n_embd)
                        x.scatter_(
                            dim=1,
                            index=timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                            src=timestep_scatter_src,
                        )
                else:
                    if isinstance(src_t, list):
                        # in this case, timestep_scatter_index is a list of tensors, each tensor is (n_src_{i} + n_tgt_{i})
                        # src_t is a list of tensors, the length of the list is batch_size.
                        # each tensor is n_src_{i}, n_src_{i} is the number of source images of the i-th sample.
                        for i, src_ts in enumerate(src_t):
                            if t is None:
                                timestep_scatter_src = src_ts.unsqueeze(0)
                            else:
                                # 1 x (n_src_{i} + n_{j})
                                tgt_t = t[i:i+1].unsqueeze(-1) if isinstance(t, torch.Tensor) else t[i].unsqueeze(0)
                                timestep_scatter_src = torch.cat([src_ts.unsqueeze(0), tgt_t], dim=1)
                            # 1 x (n_src_{i} + n_{j}) x n_embd
                            timestep_scatter_src = self.timestep_emb(timestep_scatter_src.reshape(-1)).reshape(1, -1, n_embd)

                            x[i:i+1].scatter_(
                                dim=1,
                                index=timestep_scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd), # 1 x (n_src_{i} + n_tgt_{i}) x n_embd
                                src=timestep_scatter_src, # 1 x (n_src_{i} + n_{j}) x n_embd
                            )
                    else:
                        if t is None:
                            timestep_scatter_src = src_t.unsqueeze(-1)
                        else:
                            # batch_size x (n_src+n)
                            timestep_scatter_src = torch.cat([src_t.unsqueeze(-1), t.unsqueeze(-1)], dim=1)
                        # batch_size x (n_src+n) x n_embd
                        timestep_scatter_src = self.timestep_emb(timestep_scatter_src.reshape(-1)).reshape(batch_size, -1, n_embd)

                        x.scatter_(
                            dim=1,
                            index=timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                            src=timestep_scatter_src,
                        )
            else:
                # batch_size x k x n_embd
                timestep_scatter_src = self.timestep_emb(timestep_scatter_src.reshape(-1)).reshape(batch_size, -1, n_embd)

                x.scatter_(
                    dim=1,
                    index=timestep_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=timestep_scatter_src,
                )

        if self.add_iw_ih_token and iw_ih_scatter_index is not None:
            assert iw_ih_scatter_src is not None, "iw_ih_scatter_src is required for adding iw and ih tokens"

            if isinstance(iw_ih_scatter_index, list):
                # in this case, iw_ih_scatter_index and iw_ih_scatter_src are both a list of tensors
                # each tensor is 2(n_src_{i} + 1)
                for i, iw_ih_scatter_index_i in enumerate(iw_ih_scatter_index):
                    # 2(n_src_{i} + 1)
                    iw_ih_scatter_src_i = iw_ih_scatter_src[i]
                    # 1 x 2(n_src_{i} + 1) x n_embd
                    iw_ih_scatter_src_i = torch.cat([
                        self.w_emb(iw_ih_scatter_src_i[0::2].reshape(-1)).reshape(1, -1, n_embd),
                        self.h_emb(iw_ih_scatter_src_i[1::2].reshape(-1)).reshape(1, -1, n_embd)
                    ], dim=1)
                    # 1 x 2(n_src_{i} + 1)
                    iw_ih_scatter_index_i = torch.cat([
                        iw_ih_scatter_index_i[0::2],
                        iw_ih_scatter_index_i[1::2]
                    ], dim=0).unsqueeze(0)

                    x[i:i+1].scatter_(
                        dim=1,
                        index=iw_ih_scatter_index_i.unsqueeze(-1).repeat(1, 1, n_embd), # 1 x 2(n_src_{i} + 1) x n_embd
                        src=iw_ih_scatter_src_i, # 1 x 2(n_src_{i} + 1) x n_embd
                    )
            else:
                # batch_size x 2/2k x n_embd
                iw_ih_scatter_src = torch.cat([
                    self.w_emb(iw_ih_scatter_src[:, 0::2].reshape(-1)).reshape(batch_size, -1, n_embd),
                    self.h_emb(iw_ih_scatter_src[:, 1::2].reshape(-1)).reshape(batch_size, -1, n_embd)
                ], dim=1)
                # batch_size x 2/2k
                iw_ih_scatter_index = torch.cat([
                    iw_ih_scatter_index[:, 0::2],
                    iw_ih_scatter_index[:, 1::2]
                ], dim=1)

                x.scatter_(
                    dim=1,
                    index=iw_ih_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=iw_ih_scatter_src,
                )

        return x

    def ragged_final_layer(self, x, image_mask, t_emb, token_h, token_w):
        batch_size, seq_len, n_embd = x.shape
        if isinstance(t_emb, torch.Tensor):
            if self.img_proj_type == "siglip2":
                # siglip2 use LinearPatchEmbed which return token_h=0 & token_w=0
                image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(batch_size, -1, n_embd)
            else:
                # Here when t_emb.shape[0] > batch_size, `-1` should equal to t_emb.shape[0].
                image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(-1, token_h * token_w, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            # Multiple target images(interleave data).
            # In this case, each line of the image_mask may contain different number of Trues, leading
            # the `reshape(batch_size, ...)` is not possible.
            sections = image_mask.sum(1).tolist()
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_emb_i, token_h_i, token_w_i in zip(image_output, t_emb, token_h, token_w):
                if isinstance(token_h_i, int):
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    subsections = [token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)]
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(image_output_ij[None], t_emb_i[j:j+1], token_h_i[j], token_w_i[j])
                        pred_i.append(pred_ij)
                    pred.append(pred_i)
        return pred

    def ragged_final_layer_2(self, x, image_mask, ts, token_h, token_w):
        # When self.use_final_time_embed is True, the final layer will use a separated self.time_embed_2 layer
        # to encode t and src_t.
        batch_size, seq_len, n_embd = x.shape
        if isinstance(ts, torch.Tensor):
            # Only one target image.
            t_emb = self.time_embed_2(ts)
            if self.img_proj_type == "siglip2":
                # siglip2 use LinearPatchEmbed which return token_h=0 & token_w=0
                image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(batch_size, -1, n_embd)
            else:
                # Here when t_emb.shape[0] > batch_size, `-1` should equal to t_emb.shape[0].
                image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(-1, token_h * token_w, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            # Multiple target images(interleave data).
            # In this case, each line of the image_mask may contain different number of Trues, leading
            # the `reshape(batch_size, ...)` is not possible.
            sections = image_mask.sum(1).tolist()
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_i, token_h_i, token_w_i in zip(image_output, ts, token_h, token_w):
                t_emb_i = self.time_embed_2(t_i)
                if isinstance(token_h_i, int):
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    subsections = [token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)]
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(image_output_ij[None], t_emb_i[j:j+1], token_h_i[j], token_w_i[j])
                        pred_i.append(pred_ij)
                    pred.append(pred_i)
        return pred

    def set_rope_cache(self, seq_len, rope_image_info, device):
        cos, sin = get_batch_text_image_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            base_rescale_factor=self.config.rope_base_rescale_factor,
        )
        self.cos_cache = cos
        self.sin_cache = sin

    def clear_rope_cache(self):
        self.cos_cache = None
        self.sin_cache = None

    def get_rope_and_mask(self, seq_len, device, input_pos=None, freqs_cos=None, freqs_sin=None, attention_mask=None,
                          rope_image_info=None):
        if input_pos is not None:  # use the kv cache
            if self.rope_type == "2d":
                if self.cos_cache is None:
                    cos, sin = get_batch_text_image_2d_rope(
                        image_infos=rope_image_info,
                        seq_len=seq_len,
                        n_elem=self.config.rope_n_elem,
                        device=device,
                        condense_ratio=self.config.rope_condense_ratio,
                        base=self.config.rope_base,
                        base_rescale_factor=self.config.rope_base_rescale_factor,
                    )
                else:
                    cos, sin = self.cos_cache, self.sin_cache
                if input_pos.dim() == 1:
                    assert cos.shape[0] == 1, \
                        (f"When using input_pos with 1D shape ({input_pos.shape}), cos and sin should have a "
                         f"batch dimension of 1, but got {cos.shape}")
                    input_pos = input_pos.unsqueeze(0)
                cos = real_batched_index_select(cos, dim=1, idx=input_pos)
                sin = real_batched_index_select(sin, dim=1, idx=input_pos)
            else:
                cos = batched_index_select(self.cos, dim=0, idx=input_pos)
                sin = batched_index_select(self.sin, dim=0, idx=input_pos)
            if not self.args.kv_cache or self.use_hf:
                # use_hf 的时候, attention_mask 在 prefill 时存在, decode 时为 None. 因为 decode 时每次只有一个 token,
                # 总是会做 full attention.
                mask = attention_mask
            else:
                if self.mask_cache is None:
                    raise TypeError("You need to call `gpt.set_kv_cache()`")
                if self.mask_cache.shape[0] == 1:
                    mask = batched_index_select(self.mask_cache, dim=2, idx=input_pos)
                else:
                    mask = real_batched_index_select(self.mask_cache, dim=2, idx=input_pos)
            if mask is not None and mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        elif freqs_cos is not None and freqs_sin is not None:  # use the provided frequencies
            cos = freqs_cos[:seq_len]
            sin = freqs_sin[:seq_len]
            mask = attention_mask
        elif self.rope_type == "2d":
            cos, sin = get_batch_text_image_2d_rope(
                image_infos=rope_image_info,
                seq_len=seq_len,
                n_elem=self.config.rope_n_elem,
                device=device,
                condense_ratio=self.config.rope_condense_ratio,
                base=self.config.rope_base,
                base_rescale_factor=self.config.rope_base_rescale_factor,
            )
            mask = attention_mask
        else:
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]
            mask = attention_mask
        return cos, sin, mask

    def forward(
            self,
            idx: torch.Tensor,  # bsz x seq_len-1
            x_t: Optional[BatchRaggedImages] = None,  # batch_size x c x h x w, or bsz x (n_i x (c x h_ij x w_ij))
            t: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            target: Optional[torch.Tensor] = None,  # bsz x seq_len-1, for calculating discrete loss
            diffusion_loss_fn: Optional[nn.Module] = None,  # for calculating diffusion loss, can be None when sampling
            src_x: Optional[BatchRaggedImages] = None,  # bsz x c x h x w, or bsz x (n_src_i x (c x h_ij x w_ij))
            src_t: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_src_i)
            src_image_mask: Optional[torch.Tensor] = None,  # bsz x seq_len-1
            input_pos: Optional[torch.Tensor] = None,   # bsz x seq_len-1, used for KVCache
            iw_ih_scatter_index: Optional[BatchRaggedTensor] = None,   # bsz x 2k, or bsz x (2k_i)  (index of w, index of h)
            iw_ih_scatter_src: Optional[BatchRaggedTensor] = None,  # bsz x 2k, or bsz x (2k_i)  (w, h)
            timestep_scatter_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            timestep_scatter_src: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            timestep_r_scatter_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            guidance_scatter_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            text_mask: Optional[torch.Tensor] = None,  # bsz x seq_len-1
            image_mask: Optional[torch.Tensor] = None,  # bsz x seq_len-1
            image_loss_weight: float = 1.0,
            attention_mask: Optional[torch.Tensor] = None,  # bsz x 1 x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            data_type: Optional[str] = "image",
            und_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
            und_image_masks: Optional[torch.Tensor] = None,
            rope_image_info: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
            sample_offsets: Optional[List[torch.Tensor]] = None,
            n_samples: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,  # used for rl training
            return_moe_loss: Optional[bool] = None,  # used for pure-torch rl training
            past_key_values=None,
            guidance: Optional[torch.Tensor] = None,
            r: Optional[torch.Tensor] = None,
            cache_dic=None,
    ) -> Any:
        """
        data_type: str
            "image"/"t2i": text-to-image
            "text"/"lm": text-to-text
            "mmu": image-to-text
            "faceid": text-faceid-to-image
            "faceid": text-faceid-face_src_image-to-image (multiple conditions)
            "ti2i": text+image(s)-to-image
        """
        # if self.with_guidance_embed:
        #     assert guidance is not None, "guidance is not None, when config.guidance_embed is True"
        #     if self.add_guidance_token:
        #         assert guidance_scatter_index is not None, "guidance_scatter_index is not None, when config.add_guidance_token is True"
        if self.use_pure_torch:
            if self.enable_pp and rope_image_info is not None:
                rope_image_info = batch_tensor_to_obj(rope_image_info)
        token_mode: TokenMode = TokenMode.DUMMY
        if src_x is not None:
            token_mode |= TokenMode.SRC_IMAGE
        if x_t is not None:
            token_mode |= TokenMode.GEN_IMAGE
        if und_image_embeds is not None:
            token_mode |= TokenMode.UND_IMAGE

        assert data_type in [
            "text", "lm", "image", "t2i", "mmu", "faceid", "ti2i"
        ], f"data_type {data_type} not supported"

        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        cos, sin, mask = self.get_rope_and_mask(
            T, idx.device, input_pos=input_pos, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
            attention_mask=attention_mask, rope_image_info=rope_image_info,
        )

        if self.pre_process:
            # token embeddings of shape (b, t, n_embd)
            # If idx includes <img> tokens, these embeddings will be replaced by x_{t} patches
            x = self.transformer.wte(idx)
            #x = self.ptm_transformer.language_model.embedding.word_embeddings(idx)
            batch_size, _, n_embd = x.shape

            if TokenMode.GEN_IMAGE in token_mode:
                x = self.instantiate_gen_image_tokens(
                    x, x_t, t, image_mask, guidance=guidance,
                )
                if self.use_final_time_embed:
                    token_h, token_w = self.get_token_sizes(x_t)
                else:
                    x, t_emb, token_h, token_w = x
            else:
                t_emb, token_h, token_w = None, None, None

            if TokenMode.SRC_IMAGE in token_mode:
                x = self.instantiate_src_image_tokens(
                    x, src_x, src_t, src_image_mask,
                )

            if TokenMode.UND_IMAGE in token_mode:
                x = self.instantiate_und_image_tokens(
                    x, und_image_embeds, und_image_masks,
                )

            x = self.instantiate_scatter(
                x, t, src_t, iw_ih_scatter_index, iw_ih_scatter_src, timestep_scatter_index, timestep_scatter_src, guidance_scatter_index, guidance, timestep_r_scatter_index, r,
                n_samples=n_samples,
            )

        else:
            t_emb, token_h, token_w = None, None, None
            x = None

        out = {}

        # transformer blocks
        if self.use_ptm:
            if self.pre_process:
                # (b,s,h) -> (s, b, h)
                x = x.transpose(0, 1).contiguous()
                if self.args.sequence_parallel:
                    x = mpu.scatter_to_sequence_parallel_region(x)

                with torch.autocast(device_type="cuda", enabled=False):
                    x, *other_losses = self.ptm_transformer(idx, None, mask, encoder_input=x, transfusion_kvcache_input_pos=input_pos, custom_pos_emb=(cos, sin))
            else:
                with torch.autocast(device_type="cuda", enabled=False):
                    x, *other_losses = self.ptm_transformer(idx, None, mask, transfusion_kvcache_input_pos=input_pos, custom_pos_emb=(cos, sin))

            interval = 3
            moe_loss = [sum([other_losses[j + i] for j in range(0, len(other_losses), interval)]) for i in
                        range(interval)]

            if not self.post_process:
                return x, None, moe_loss

            if self.args.sequence_parallel:
                x = mpu.gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
            x = x.transpose(0, 1).contiguous()

        elif self.use_pure_torch:
            if not self.pre_process:
                x = idx
            if not self.use_taylor_cache:
                for block_idx, block in enumerate(self.transformer.h):
                    if block is None:
                        continue
                    block_inputs = [x, cos.to(x.dtype), sin.to(x.dtype), mask, input_pos]
                    if self.training and self.gradient_checkpoint and (
                            self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                        from torch.utils import checkpoint
                        x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
                    else:
                        x = block(*block_inputs)
            else:
                if not hasattr(self.transformer, "taylor_cache"):
                    #self.transformer.taylor_cache = TaylorCacheContainer(cache_dic['max_order'])
                    self.transformer.taylor_cache = CacheWithFreqsContainer(cache_dic['max_order'])
                if not hasattr(self.transformer, "counter"):
                    self.transformer.counter = 0

                full_computation = (cache_dic['current_step'] == 0) \
                    or (self.transformer.counter == cache_dic['cache_interval'] -1) \
                    or (cache_dic['enable_first_enhance'] and cache_dic['current_step'] < cache_dic['first_enhance_steps']) \
                    or (cache_dic['enable_tailing_enhance'] and cache_dic['current_step'] >= cache_dic['num_steps'] - cache_dic['tailing_enhance_steps'])

                if not hasattr(self.transformer, "last_full_computation_step"):
                    self.transformer.last_full_computation_step = 0

                if full_computation:
                    self.transformer.counter = 0

                    for block_idx, block in enumerate(self.transformer.h):
                        if block is None:
                            continue
                        block_inputs = [x, cos, sin, mask, input_pos]
                        x = block(*block_inputs)

                    if cache_dic['enable_first_enhance'] and (cache_dic['current_step'] < (cache_dic['first_enhance_steps']-1)):
                        pass
                    else:
                        self.transformer.taylor_cache.derivatives_computation(x, distance = cache_dic['current_step'] - self.transformer.last_full_computation_step, low_freqs_order=cache_dic['low_freqs_order'], high_freqs_order=cache_dic['high_freqs_order'])

                    self.transformer.last_full_computation_step = cache_dic['current_step']

                else:
                    self.transformer.counter += 1
                    x = self.transformer.taylor_cache.taylor_formula(distance = self.transformer.counter)

                if cache_dic['current_step'] == cache_dic['num_steps'] - 1:
                    self.transformer.taylor_cache.clear_derivatives() 
                                 
            if not self.post_process:
                if self.use_final_time_embed:
                    return x
                else:
                    return t_emb, x

        elif self.use_hf:
            transformer_out = self.transformer(
                attention_mask=mask,                # [1, 1, seqlen, seqlen]
                position_ids=input_pos,             # [1, seqlen]
                past_key_values=past_key_values,    #
                inputs_embeds=x,                    # [1, seqlen, n_embd]
                custom_pos_emb=(cos, sin),          # [1, block_size, head_dim]
                gen_timestep_scatter_index=torch.full((2, 1), -1, dtype=torch.int),     # used for masking as text tokens
            )
            x = transformer_out.last_hidden_state
            out["past_key_values"] = transformer_out.past_key_values

        else:
            for block_idx, block in enumerate(self.transformer.h):
               block_inputs = [x, cos, sin, mask, input_pos]
               if self.training and self.gradient_checkpoint and (
                       self.gradient_checkpoint_layers == -1 or block_idx < self.gradient_checkpoint_layers):
                   x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_inputs, use_reentrant=False)
               else:
                   x = block(*block_inputs)

        # out norm
        x_preln = x
        x = self.transformer.ln_f(x)

        if TokenMode.GEN_IMAGE in token_mode:
            # for diffusion prediction
            x_input = x_preln if self.args.get('image_preln', False) else x
            if self.use_final_time_embed:
                if self.enable_pp:
                    token_h, token_w = self.get_token_sizes(x_t)
                diffusion_prediction = self.ragged_final_layer_2(x_input, image_mask, t, token_h, token_w)
            else:
                if hasattr(self, "enable_pp") and self.enable_pp:
                    raise NotImplementedError
                diffusion_prediction = self.ragged_final_layer(x_input, image_mask, t_emb, token_h, token_w)
            if not self.training or return_loss is False:
                out["diffusion_prediction"] = diffusion_prediction

        # for discrete tokens
        if self.config.lm_head_reuse_embedding:
            x = F.linear(x, self.transformer.wte.weight, bias=self.lm_head_bias)
        else:
            x = self.lm_head(x)  # (b, t, vocab_size)
        # x = x.float()   # follow show-o implementation to avoid numerical issues

        # output
        if self.use_pure_torch and self.enable_pp:
            if self.training:
                if return_loss is False: # Following the argument design where the `return_loss` can be None or False.
                    out["logits"] = x
                    moe_loss = None
                    loss, loss_dict = self.loss_closure(out, moe_loss)
                    if self.enable_pp:
                        self.cache_result('loss_dict', loss_dict)
                        self.cache_result('ret_val', out)
                        return loss
                    else:
                        # TODO: support return moe loss
                        return out
            else:
                out["logits"] = x
                if self.enable_pp:
                    self.cache_result('ret_val', out)
                    return torch.tensor([1.]).cuda() # whatever
                else:
                    return out
        else:
            if not self.training or return_loss is False:
                out["logits"] = x.float()
                return out

        # Introduce a flag to determine if we should gather valid text tokens to calculating discrete loss.
        # When the sequence is long, such as 32K, the x memory size will be as large as 32K * 128000 * 2 = 8GB,
        # and therefore the CE loss will consume 24GB of memory.
        if (data_type in ["text", "lm"]) or (data_type in ["image", "t2i"] and src_x is None) or (
                data_type == "mmu") or (data_type == "ti2i") or (data_type == "faceid"):
            # Gather valid text tokens. Only support bsz=1
            if self.gather_text_tokens and x.size(0) == 1 and data_type in ["image", "t2i", "ti2i", "faceid"]:
                bool_mask = text_mask[0].bool()
                valid_x = x[0, bool_mask]
                valid_target = target[0, bool_mask]
                discrete_loss = torch.nn.functional.cross_entropy(
                    valid_x, valid_target, ignore_index=-100, reduction="mean"
                )
            else:
                discrete_loss = torch.nn.functional.cross_entropy(
                    x.view(-1, x.size(-1)), target.view(-1), ignore_index=-100, reduction="mean"
                )
        else:
            discrete_loss = 0
        loss = discrete_loss

        if TokenMode.GEN_IMAGE in token_mode:
            # TODO(jarvizhang): interleave
            diffusion_loss = diffusion_loss_fn(model_output=diffusion_prediction)["loss"].mean()
            omit_image_loss = False
            if isinstance(diffusion_prediction, torch.Tensor) and \
                    ((diffusion_prediction.ndim == 4 and diffusion_prediction.shape[2] * diffusion_prediction.shape[3] <= self.args.patch_size ** 2)
                     or (diffusion_prediction.ndim == 3 and diffusion_prediction.shape[1] <= 1)):
                # Only return the image loss of image batches. p^2-token batches are text batches.
                omit_image_loss = True
            if not omit_image_loss:
                out["image_loss"] = diffusion_loss.detach()
            loss = loss + image_loss_weight * diffusion_loss

        out["loss"] = loss

        # Only for print
        if text_mask is not None:
            if self.use_ptm:
                if data_type == "mmu":
                    out["mmu_text_loss"] = discrete_loss.detach()
                if data_type == "t2i":
                    out["t2i_text_loss"] = discrete_loss.detach()
                if data_type == "lm":
                    out["lm_text_loss"] = discrete_loss.detach()
                out["text_loss"] = discrete_loss.detach()
            else:
                out["text_loss"] = discrete_loss.detach().cpu().reshape(-1)

        if self.use_pure_torch:
            if self.enable_pp:
                loss, loss_dict = self.loss_closure(out)
                self.cache_result('ret_val', out)
                self.cache_result('loss_dict', loss_dict)
                return loss
            else:
                return out
        elif self.use_ptm:
            return x, out, moe_loss
        else:
            return out

    # we decouple training and inference forward function due to complex implementation KVCache of Transfusion
    def infer_forward(
            self,
            idx: torch.Tensor,  # batch_size x seq_len-1
            x_t: torch.Tensor, # batch_size x c x h x w
            t: torch.Tensor,  # batch_size
            src_x: Optional[torch.Tensor] = None,  # batch_size x c x h x w, only used for instruction tuning
            src_t: Optional[torch.Tensor] = None,  # batch_size, only used for instruction tuning
            src_image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1, only used for instruction tuning
            first_step=True,
            input_pos: Optional[torch.Tensor] = None,
            iw_ih_scatter_index: Optional[torch.Tensor] = None,   # batch_size x 2k  (index of w, index of h)
            iw_ih_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 2k  (w, h)
            timestep_scatter_index: Optional[torch.Tensor] = None,  # batch_size x 1
            timestep_scatter_src: Optional[torch.Tensor] = None,  # batch_size x 1
            image_mask: Optional[torch.Tensor] = None,  # batch_size x seq_len-1
            attention_mask: Optional[torch.Tensor] = None,  # batch_size x 1 x seq_len-1 x seq_len-1
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            und_image_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
            und_image_masks: Optional[torch.Tensor] = None,
            rope_image_info: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
            n_samples: Optional[torch.Tensor] = None,
            real_seqlen: Optional[torch.Tensor] = None, # for pp only
            past_key_values=None,
            gen_timestep_scatter_index: Optional[torch.Tensor] = None,  # batch_size x 1,
            cache_dic=None,
            ext_model_forward: Optional[Any] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.use_pure_torch:
            if self.enable_pp and rope_image_info is not None:
                rope_image_info = batch_tensor_to_obj(rope_image_info)

        if freqs_cos is not None and freqs_sin is not None:   # use the provided frequencies
            cos = freqs_cos
            sin = freqs_sin
        elif self.rope_type == "2d":
            if self.cos_cache is None:
                cos, sin = get_batch_text_image_2d_rope(
                    image_infos=rope_image_info,
                    seq_len=(idx.shape[1] if self.pre_process else real_seqlen) if self.use_pure_torch else idx.shape[1],
                    n_elem=self.config.rope_n_elem,
                    device=idx.device,
                    condense_ratio=self.config.rope_condense_ratio,
                    base=self.config.rope_base,
                    base_rescale_factor=self.config.rope_base_rescale_factor,
                )
            else:
                cos, sin = self.cos_cache, self.sin_cache
            cos = real_batched_index_select(cos, dim=1, idx=input_pos)
            sin = real_batched_index_select(sin, dim=1, idx=input_pos)
        else:
            cos = batched_index_select(self.cos, dim=0, idx=input_pos)
            sin = batched_index_select(self.sin, dim=0, idx=input_pos)

        mask = attention_mask

        if self.pre_process:
            # first step of KVCache
            if first_step:
                # token embeddings of shape (b, t, n_embd)
                # idx includes <img> tokens, whose embeddings will be replaced by x_{t} patches
                x = self.transformer.wte(idx)

                batch_size = x.shape[0]
                n_embd = x.shape[-1]

                token_mode: TokenMode = TokenMode.DUMMY
                if src_x is not None:
                    token_mode |= TokenMode.SRC_IMAGE
                if x_t is not None:
                    token_mode |= TokenMode.GEN_IMAGE
                if und_image_embeds is not None:
                    token_mode |= TokenMode.UND_IMAGE

                if TokenMode.GEN_IMAGE in token_mode:
                    x = self.instantiate_gen_image_tokens(
                        x, x_t, t, image_mask,
                    )
                    if self.use_final_time_embed:
                        token_h, token_w = self.get_token_sizes(x_t)
                    else:
                        x, t_emb, token_h, token_w = x
                else:
                    t_emb, token_h, token_w = None, None, None
                
                if TokenMode.SRC_IMAGE in token_mode:
                    x = self.instantiate_src_image_tokens(
                        x, src_x, src_t, src_image_mask,
                    )

                if TokenMode.UND_IMAGE in token_mode:
                    x = self.instantiate_und_image_tokens(x, und_image_embeds, und_image_masks)

                x = self.instantiate_scatter(
                    x, t, src_t, iw_ih_scatter_index, iw_ih_scatter_src, timestep_scatter_index, timestep_scatter_src,
                    n_samples=n_samples,
                )

            # following steps of KVCache
            else:
                if self.img_proj_type in ["unet", "linear", "siglip2"]:
                    t_emb = self.time_embed(t)
                    # batch_size x image_seq_len x n_embd
                    image_seq, token_h, token_w = self.patch_embed(x_t, t_emb)
                else:
                    raise ValueError(f"img_proj_type {self.img_proj_type} not supported")

                x = image_seq

                batch_size = x.shape[0]
                n_embd = x.shape[-1]

                if self.add_timestep_token:
                    if timestep_scatter_src is None:
                        timestep_scatter_src = t.unsqueeze(-1) # batch_size x 1

                    # batch_size x 1 x n_embd
                    timestep_emb = self.timestep_emb(timestep_scatter_src.reshape(-1)).reshape(batch_size, -1, n_embd)

                    if self.use_front_boi_token:
                        x = torch.cat([timestep_emb, x], dim=1)
                    else:
                        boi_idx = idx[torch.arange(batch_size), timestep_scatter_index[:, 0] + 1].unsqueeze(-1)
                        # batch_size x 1 x n_embd
                        boi_emb = self.transformer.wte(boi_idx)

                        x = torch.cat([timestep_emb, boi_emb, x], dim=1)

        out = {}
        if self.use_pure_torch:
            if not self.pre_process:
                x = idx

            batch_size = x.shape[0]
            n_embd = x.shape[-1]
            for block_idx, block in enumerate(self.transformer.h):
                if block is None:
                    continue
                block_inputs = [x, cos, sin, mask, input_pos]
                x = block(*block_inputs)
            if not self.post_process:
                return x

        elif self.use_ptm:
            if self.pre_process:
                # (b, l, d) -> (l, b, d)
                x = x.transpose(0, 1).contiguous()
                if self.args.sequence_parallel:
                    x = mpu.scatter_to_sequence_parallel_region(x)

                # ptm_transformer needs the seq length of idx aligned with that of x
                # ptm_transformer only use idx to get the seq length
                idx = idx.gather(dim=1, index=input_pos)
                with torch.autocast(device_type="cuda", enabled=False):
                    x, *moe_loss = self.ptm_transformer(idx, None, mask, encoder_input=x, transfusion_kvcache_input_pos=input_pos, custom_pos_emb=(cos, sin))
            else:
                idx = idx.gather(dim=1, index=input_pos)
                with torch.autocast(device_type="cuda", enabled=False):
                    x, *moe_loss = self.ptm_transformer(idx, None, mask, transfusion_kvcache_input_pos=input_pos, custom_pos_emb=(cos, sin))

            if not self.post_process:
                return x, None, moe_loss

            if self.args.sequence_parallel:
                x = mpu.gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
            # (l, b, d) -> (b, l, d)
            x = x.transpose(0, 1).contiguous()

        elif self.use_hf:
            input_device = x.device
            if not self.use_taylor_cache:
                # print(f"****Transfusion, not use_taylor_cache****")
                # print(f"{mask.shape=}, {input_pos=}, {x.shape=}, {cos.shape=}")
                # import pdb; pdb.set_trace()
                transformer_out = self.transformer(
                    attention_mask=mask,                # [2, 1, seqlen, seqlen], bool
                    position_ids=input_pos,             # [2, seqlen]
                    past_key_values=past_key_values,    #
                    inputs_embeds=x,                    # [2, seqlen, n_embd]
                    custom_pos_emb=(cos, sin),          # [2, block_size, head_dim]
                    first_step=first_step,
                    gen_timestep_scatter_index=gen_timestep_scatter_index,
                    ext_model_forward=ext_model_forward,
                )
                x = transformer_out.last_hidden_state.to(input_device)
                out["past_key_values"] = transformer_out.past_key_values
            else:
                if not hasattr(self.transformer, "taylor_cache"):
                    #self.transformer.taylor_cache = TaylorCacheContainer(cache_dic['max_order'])
                    self.transformer.taylor_cache = CacheWithFreqsContainer(cache_dic['max_order'])
                if not hasattr(self.transformer, "counter"):
                    self.transformer.counter = 0

                full_computation = (cache_dic['current_step'] == 0) \
                    or (self.transformer.counter == cache_dic['cache_interval'] -1) \
                    or (cache_dic['enable_first_enhance'] and cache_dic['current_step'] < cache_dic['first_enhance_steps']) \
                    or (cache_dic['enable_tailing_enhance'] and cache_dic['current_step'] >= cache_dic['num_steps'] - cache_dic['tailing_enhance_steps'])

                if not hasattr(self.transformer, "last_full_computation_step"):
                    self.transformer.last_full_computation_step = 0

                if full_computation:
                    self.transformer.counter = 0

                    # print(f"{mask.shape=}, {input_pos=}, {x.shape=}, {cos.shape=}")
                    # import pdb; pdb.set_trace()
                    transformer_out = self.transformer(
                        attention_mask=mask,                # [2, 1, seqlen, seqlen], bool
                        position_ids=input_pos,             # [2, seqlen]
                        past_key_values=past_key_values,    #
                        inputs_embeds=x,                    # [2, seqlen, n_embd]
                        custom_pos_emb=(cos, sin),          # [2, block_size, head_dim]
                        first_step=first_step,
                        gen_timestep_scatter_index=gen_timestep_scatter_index,
                        ext_model_forward=ext_model_forward,
                    )
                    x = transformer_out.last_hidden_state.to(input_device)
                    out["past_key_values"] = transformer_out.past_key_values

                    if cache_dic['enable_first_enhance'] and (cache_dic['current_step'] < (cache_dic['first_enhance_steps']-1)):
                        pass
                    else:
                        self.transformer.taylor_cache.derivatives_computation(x, distance = cache_dic['current_step'] - self.transformer.last_full_computation_step, low_freqs_order=cache_dic['low_freqs_order'], high_freqs_order=cache_dic['high_freqs_order'])

                    self.transformer.last_full_computation_step = cache_dic['current_step']

                else:
                    self.transformer.counter += 1
                    x = self.transformer.taylor_cache.taylor_formula(distance = self.transformer.counter)

                if cache_dic['current_step'] == cache_dic['num_steps'] - 1:
                    self.transformer.taylor_cache.clear_derivatives()

        else:
            for block_idx, block in enumerate(self.transformer.h):
                block_inputs = [x, cos, sin, mask, input_pos]
                x = block(*block_inputs)

        if not self.args.get('image_preln', False):
            x = self.transformer.ln_f(x)

        if first_step:
            image_output = x.masked_select(image_mask.unsqueeze(-1).bool()).reshape(batch_size, -1, n_embd)
        else:
            if self.add_timestep_token:
                if self.use_front_boi_token:
                    image_output = x[:, 1:, :] # x = <boi> [<iw> <ih>] | <timestep> <img>
                else:
                    image_output = x[:, 2:, :] # x = [<iw> <ih>] | <timestep> <boi> <img>
            else:
                image_output = x
        # final layer
        if self.use_final_time_embed:
            if self.enable_pp:
                token_h, token_w = self.get_token_sizes(x_t)
            final_layer_t_emb = self.time_embed_2(t)
        else:
            final_layer_t_emb = t_emb
        if self.img_proj_type in ["unet", "linear", "siglip2"]:
            diffusion_prediction = self.final_layer(image_output, final_layer_t_emb, token_h, token_w)
        else:
            raise ValueError(f"img_proj_type {self.img_proj_type} not supported")

        # for discrete tokens
        # x = self.lm_head(x)  # (b, t, vocab_size)
        # x = x.float()   # follow show-o implementation to avoid numerical issues

        # output
        # out = {"logits": x, "diffusion_prediction": diffusion_prediction}
        out["diffusion_prediction"] = diffusion_prediction
        if self.use_pure_torch:
            if self.enable_pp:
                self.cache_result('ret_val', out)
                return torch.tensor([1.]).cuda() # whatever
            else:
                return out
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
        image_slices: Optional[Union[List[slice], List[List[slice]]]] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.config.rope_n_elem

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        if self.use_pure_torch:
            for block in self.transformer.h:
                if block is None:
                    continue
                block.attn.kv_cache = block.attn.build_kv_cache(
                    batch_size,
                    max_seq_length,
                    rope_cache_length,
                    device,
                    dtype,
                )
        elif self.use_ptm:
            for block in self.ptm_transformer.language_model.encoder.layers:
                block.self_attention.transfusion_kvcache = block.self_attention.build_kv_cache(
                    batch_size,
                    max_seq_length,
                    rope_cache_length,
                    device,
                    dtype,
                )
        else:
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
            self.mask_cache = build_mask_cache(max_seq_length, device)
            if image_slices is not None:
                if isinstance(image_slices[0], list):
                    # image_slices are list of lists of slices, the outer list is for batch dimension
                    assert len(image_slices) == batch_size, "image_slices should have the same length with batch_size"
                    self.mask_cache = self.mask_cache.repeat(batch_size, 1, 1, 1)
                    for i, image_slice_list in enumerate(image_slices):
                        for image_slice in image_slice_list:
                            self.mask_cache[i, 0, image_slice, image_slice] = True
                else:
                    for image_slice in image_slices:
                        self.mask_cache[0, 0, image_slice, image_slice] = True

    def update_mask_cache(self, image_slices):
        if self.mask_cache is not None:
            for image_slice in image_slices:
                self.mask_cache[0, 0, image_slice, image_slice] = True

    def clear_kv_cache(self) -> None:
        if self.use_ptm:
            for block in self.ptm_transformer.language_model.encoder.layers:
                block.self_attention.transfusion_kvcache = None
        else:
            for block in self.transformer.h:
                if block is None:
                    continue
                block.attn.kv_cache = None
        self.mask_cache = None

    def enable_deterministic(self) -> None:
        pass

    def disable_deterministic(self) -> None:
        pass

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

    def set_input_tensor(self, input_tensor) -> None:
        self.ptm_transformer.set_input_tensor(input_tensor)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward
