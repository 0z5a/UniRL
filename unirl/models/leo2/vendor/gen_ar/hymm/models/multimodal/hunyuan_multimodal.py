import math
from argparse import Namespace
from contextlib import nullcontext
from typing import Any, Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask
flex_attention = torch.compile(flex_attention, dynamic=False)

from hymm.core.global_vars import get_parallel_state

from ..basic.moe_layers import HunyuanMLP, HunyuanMoE, DeepSeekMoE, FlashInferMoE, Qwen3VLSparesMoeBlock, ExpertParallelMoE
from ..basic.rope_cache import CachedRoPE
from .hunyuan_multimodal_config import HunyuanMultimodalConfig
from .hunyuan_multimodal_state import HunyuanMultimodalState, HunyuanMultimodalOutput
from ..autoregressive.custom_cache import HunyuanStaticCache
from ..basic.embed_layers import TimestepEmbedder
from ..basic.initializers import normal_weight_reset_parameters
from ..basic.patch_embed_layers import project_in_layer, project_out_layer
from ..basic.pos_emb_layers import apply_rope
from ...utils.env import is_bitwise_align_mode
from ...utils.helpers import default
from ...utils.torch_utils import PRECISION_TO_TYPE

from hy_parallelism.context_parallel.core import (
    maybe_scatter_seq,
    maybe_gather_seq,
    maybe_to_split_head,
    maybe_to_split_seq,
    maybe_to_cp_region_num_head,
    maybe_to_normal_region_num_head,
)
from hy_parallelism.models.modules.cross_entropy import ChunkedCELoss

# Type aliases
BatchRaggedMedia = Union[torch.Tensor, list[Union[torch.Tensor, list[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, list[torch.Tensor]]


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


def get_device(tensor: BatchRaggedMedia):
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    elif isinstance(tensor, list):
        return get_device(tensor[0])
    else:
        raise ValueError(f"Unsupported type for get_device: {type(tensor)}")


class CausalSelfAttention(nn.Module):
    def __init__(
            self,
            config: HunyuanMultimodalConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx

        if not config.split_qkv:
            self.qkv_proj = nn.Linear(
                config.hidden_size, (config.num_attention_heads + 2 * config.num_kv_heads) * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
        else:
            if config.attention_bias:
                raise NotImplementedError('Checkpoint conversion with bias is not implemented for TP-friendly qkv yet.')
            self.q_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
            self.k_proj = nn.Linear(
                config.hidden_size, config.num_kv_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )
            self.v_proj = nn.Linear(
                config.hidden_size, config.num_kv_heads * config.attention_head_size,
                bias=config.attention_bias, **factory_kwargs
            )

        self.o_proj = nn.Linear(
            config.attention_head_size * config.num_attention_heads, config.hidden_size,
            bias=config.attention_bias, **factory_kwargs
        )

        if config.use_qk_norm:
            self.query_layernorm = config.norm_class(config.attention_head_size, eps=config.norm_eps, **factory_kwargs)
            self.key_layernorm = config.norm_class(config.attention_head_size, eps=config.norm_eps, **factory_kwargs)

    @staticmethod
    def all2all_qkv(qkv: torch.Tensor, head_size: int, total_qkv: int) -> torch.Tensor:
        return einops.rearrange(
            maybe_to_split_head(
                einops.rearrange(
                    qkv,
                    'b s (n_kv_head total_qkv head_size) -> (b total_qkv) s n_kv_head head_size',
                    head_size=head_size, total_qkv=total_qkv,
                )
            ),
            '(b total_qkv) s n_kv_head head_size -> b s n_kv_head total_qkv head_size',
            head_size=head_size, total_qkv=total_qkv,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            past_key_values: Optional[HunyuanStaticCache] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        if not self._config.split_qkv:
            qkv = self.qkv_proj(hidden_states)
            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            qkv = qkv.view(bsz, seqlen, n_kv_head, total_qkv, head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)  # (bsz, num_kv_heads, total_qkv, T, head_size)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
        else:
            new_q = self.q_proj(hidden_states).view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
            new_k = self.k_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
            new_v = self.v_proj(hidden_states).view(bsz, seqlen, n_kv_head, 1, head_size)
            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [new_q, new_k, new_v])
 
        q = q.reshape(bsz, -1, seqlen, head_size)  # (B, n_q_head, T, hs)
        k = k.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)
        v = v.reshape(bsz, -1, seqlen, head_size)  # (B, n_kv_head, T, hs)

        # QWen VL use qk norm before rotary pos emb
        if self._config.use_qk_norm and self._config.pre_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        q = apply_rope(q, *rotary_position_embeddings, apply_rope_in_fp32=self._config.apply_rope_in_fp32)
        k = apply_rope(k, *rotary_position_embeddings, apply_rope_in_fp32=self._config.apply_rope_in_fp32)

        # Some others use qk norm after rotary pos emb
        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if input_pos is not None:
            cache_kwargs = {"cache_position": input_pos}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = k.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if n_kv_head != n_q_head and (input_pos is None or q_per_kv != 1):
            k = k.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            v = v.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        y = self.scaled_dot_product_attention(q, k, v, attention_mask)

        y = y.reshape(bsz, seqlen, head_size * n_q_head)  # re-assemble all head outputs side by side

        # output projection
        return self.o_proj(y)

    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # q, k, v: (bsz, n_head, seqlen, head_size)
        scale = 1.0 / math.sqrt(self._config.attention_head_size)

        if isinstance(mask, BlockMask):
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            y = flex_attention(q, k, v, block_mask=mask, scale=scale)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
                # If q only has one token (typically in AR model decoding stage), we should use full attention.
                is_causal=mask is None and q.size(2) > 1
            )

        return y.transpose(1, 2)


class CausalSelfAttentionMoT(CausalSelfAttention):
    def __init__(
            self,
            config: HunyuanMultimodalConfig,
            config_mot_gen: HunyuanMultimodalConfig,
            layer_idx: int,
            mot_und_frozen: bool = False,
            mot_gen_frozen: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(config, layer_idx, dtype, device)
        self.mot_und_frozen = mot_und_frozen
        self.mot_gen_frozen = mot_gen_frozen

        if not config_mot_gen.split_qkv:
            self.qkv_proj_mot_gen = nn.Linear(
                config_mot_gen.hidden_size,
                (config_mot_gen.num_attention_heads + 2 * config_mot_gen.num_kv_heads) * config_mot_gen.attention_head_size,
                bias=config_mot_gen.attention_bias, **factory_kwargs
            )
        else:
            self.q_proj_mot_gen = nn.Linear(
                config_mot_gen.hidden_size, config_mot_gen.num_attention_heads * config_mot_gen.attention_head_size,
                bias=config_mot_gen.attention_bias, **factory_kwargs
            )
            self.k_proj_mot_gen = nn.Linear(
                config_mot_gen.hidden_size, config_mot_gen.num_kv_heads * config_mot_gen.attention_head_size,
                bias=config_mot_gen.attention_bias, **factory_kwargs
            )
            self.v_proj_mot_gen = nn.Linear(
                config_mot_gen.hidden_size, config_mot_gen.num_kv_heads * config_mot_gen.attention_head_size,
                bias=config_mot_gen.attention_bias, **factory_kwargs
            )

        self.o_proj_mot_gen = nn.Linear(
            config_mot_gen.attention_head_size * config_mot_gen.num_attention_heads, config_mot_gen.hidden_size,
            bias=config_mot_gen.attention_bias, **factory_kwargs
        )

        if config_mot_gen.use_qk_norm:
            self.query_layernorm_mot_gen = config_mot_gen.norm_class(config_mot_gen.attention_head_size, eps=config_mot_gen.norm_eps, **factory_kwargs)
            self.key_layernorm_mot_gen = config_mot_gen.norm_class(config_mot_gen.attention_head_size, eps=config_mot_gen.norm_eps, **factory_kwargs)

        if self.mot_und_frozen:
            if not config_mot_gen.split_qkv:
                self.qkv_proj.eval()
                self.qkv_proj.requires_grad_(False)
            else:
                self.q_proj.eval()
                self.q_proj.requires_grad_(False)
                self.k_proj.eval()
                self.k_proj.requires_grad_(False)
                self.v_proj.eval()
                self.v_proj.requires_grad_(False)
            self.o_proj.eval()
            self.o_proj.requires_grad_(False)
            self.query_layernorm.eval()
            self.query_layernorm.requires_grad_(False)
            self.key_layernorm.eval()
            self.key_layernorm.requires_grad_(False)

        if self.mot_gen_frozen:
            if not config_mot_gen.split_qkv:
                self.qkv_proj_mot_gen.eval()
                self.qkv_proj_mot_gen.requires_grad_(False)
            else:
                self.q_proj_mot_gen.eval()
                self.q_proj_mot_gen.requires_grad_(False)
                self.k_proj_mot_gen.eval()
                self.k_proj_mot_gen.requires_grad_(False)
                self.v_proj_mot_gen.eval()
                self.v_proj_mot_gen.requires_grad_(False)
            self.o_proj_mot_gen.eval()
            self.o_proj_mot_gen.requires_grad_(False)
            self.query_layernorm_mot_gen.eval()
            self.query_layernorm_mot_gen.requires_grad_(False)
            self.key_layernorm_mot_gen.eval()
            self.key_layernorm_mot_gen.requires_grad_(False)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        past_key_values: Optional[HunyuanStaticCache] = None,
        und_token_indices: Optional[torch.Tensor] = None,
        gen_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        und_hidden_states, gen_hidden_states = hidden_states

        bsz = und_hidden_states.shape[0]
        und_seqlen, gen_seqlen = und_hidden_states.shape[1], gen_hidden_states.shape[1]
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        cp_enabled = get_parallel_state().cp_size > 1
        # if cp_enabled:
        #     # assert input_pos is None, "kv-cache with sharded heads is not handled yet"

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        if not self._config.split_qkv:
            # [bsz, seqlen_local, (n_kv_head * (q_per_kv + 2) * head_size)]
            qkv = self.qkv_proj(und_hidden_states)
            gen_qkv = self.qkv_proj_mot_gen(gen_hidden_states)
            total_qkv = q_per_kv + 2

            if cp_enabled:
                # all-to-all: [b, s_local, ...] -> [b, s_full, n_kv_local, total_qkv, hd]
                qkv = self.all2all_qkv(qkv, head_size=head_size, total_qkv=total_qkv)
                gen_qkv = self.all2all_qkv(gen_qkv, head_size=head_size, total_qkv=total_qkv)
                und_seqlen, gen_seqlen = qkv.shape[1], gen_qkv.shape[1]
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])
            else:
                qkv = qkv.view(bsz, und_seqlen, n_kv_head, total_qkv, head_size)
                gen_qkv = gen_qkv.view(bsz, gen_seqlen, n_kv_head, total_qkv, head_size)

            # (bsz, n_kv_head, q_per_kv+2, T, head_size)
            qkv = qkv.permute(0, 2, 3, 1, 4)
            gen_qkv = gen_qkv.permute(0, 2, 3, 1, 4)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            gen_q, gen_k, gen_v = gen_qkv.split((q_per_kv, 1, 1), dim=2)
        else:
            q = self.q_proj(und_hidden_states)
            k = self.k_proj(und_hidden_states)
            v = self.v_proj(und_hidden_states)
            gen_q = self.q_proj_mot_gen(gen_hidden_states)
            gen_k = self.k_proj_mot_gen(gen_hidden_states)
            gen_v = self.v_proj_mot_gen(gen_hidden_states)

            if cp_enabled:
                # Each tensor: [b, s_local, n_head*hd] -> [b, s_full, n_head_local, hd]
                q, k, v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, und_seqlen, x.shape[-1] // head_size, head_size)
                    ),
                    [q, k, v],
                )
                gen_q, gen_k, gen_v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, gen_seqlen, x.shape[-1] // head_size, head_size)
                    ),
                    [gen_q, gen_k, gen_v],
                )
                und_seqlen, gen_seqlen = q.shape[1], gen_q.shape[1]
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])

            q = q.view(bsz, und_seqlen, n_kv_head, q_per_kv, head_size)
            k = k.view(bsz, und_seqlen, n_kv_head, 1, head_size)
            v = v.view(bsz, und_seqlen, n_kv_head, 1, head_size)
            gen_q = gen_q.view(bsz, gen_seqlen, n_kv_head, q_per_kv, head_size)
            gen_k = gen_k.view(bsz, gen_seqlen, n_kv_head, 1, head_size)
            gen_v = gen_v.view(bsz, gen_seqlen, n_kv_head, 1, head_size)

            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [q, k, v])
            gen_q, gen_k, gen_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [gen_q, gen_k, gen_v])

        # [bsz, h, seqlen, head_size]
        q = q.reshape(bsz, n_q_head, und_seqlen, head_size)
        k = k.reshape(bsz, n_kv_head, und_seqlen, head_size)
        v = v.reshape(bsz, n_kv_head, und_seqlen, head_size)
        gen_q = gen_q.reshape(bsz, n_q_head, gen_seqlen, head_size)
        gen_k = gen_k.reshape(bsz, n_kv_head, gen_seqlen, head_size)
        gen_v = gen_v.reshape(bsz, n_kv_head, gen_seqlen, head_size)

        # Scatter understanding and generation tokens for rope
        und_token_indices_q = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
        gen_token_indices_q = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
        und_token_indices_kv = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
        gen_token_indices_kv = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
        
        def _scatter(und_src, gen_src, und_token_indices, gen_token_indices, n_head):
            target = torch.zeros((bsz, n_head, und_seqlen+gen_seqlen, head_size), dtype=und_src.dtype, device=und_src.device)
            target.scatter_(dim=2, index=und_token_indices, src=und_src)
            target.scatter_(dim=2, index=gen_token_indices, src=gen_src)
            return target

        q_merge = _scatter(q, gen_q, und_token_indices_q, gen_token_indices_q, n_q_head)
        k_merge = _scatter(k, gen_k, und_token_indices_kv, gen_token_indices_kv, n_kv_head)
        v_merge = _scatter(v, gen_v, und_token_indices_kv, gen_token_indices_kv, n_kv_head)

        # QWen VL use qk norm before rotary pos emb
        if self._config.use_qk_norm and self._config.pre_qk_norm:
            q_ = torch.zeros_like(q_merge)
            k_ = torch.zeros_like(k_merge)

            q_.scatter_(dim=2, index=und_token_indices_q, src=self.query_layernorm(q_merge.gather(2, und_token_indices_q)).to(q_.dtype))
            q_.scatter_(dim=2, index=gen_token_indices_q, src=self.query_layernorm_mot_gen(q_merge.gather(2, gen_token_indices_q)).to(q_.dtype))
            k_.scatter_(dim=2, index=und_token_indices_kv, src=self.key_layernorm(k_merge.gather(2, und_token_indices_kv)).to(k_.dtype))
            k_.scatter_(dim=2, index=gen_token_indices_kv, src=self.key_layernorm_mot_gen(k_merge.gather(2, gen_token_indices_kv)).to(k_.dtype))
            
            q_merge = q_
            k_merge = k_

        # apply rotary position embeddings
        q_merge = apply_rope(q_merge, *rotary_position_embeddings, apply_rope_in_fp32=self._config.apply_rope_in_fp32)
        k_merge = apply_rope(k_merge, *rotary_position_embeddings, apply_rope_in_fp32=self._config.apply_rope_in_fp32)

        # Some others use qk norm after rotary pos emb
        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            q_ = torch.zeros_like(q_merge)
            k_ = torch.zeros_like(k_merge)

            q_.scatter_(dim=2, index=und_token_indices_q, src=self.query_layernorm(q_merge.gather(2, und_token_indices_q)).to(q_.dtype))
            q_.scatter_(dim=2, index=gen_token_indices_q, src=self.query_layernorm_mot_gen(q_merge.gather(2, gen_token_indices_q)).to(q_.dtype))
            k_.scatter_(dim=2, index=und_token_indices_kv, src=self.key_layernorm(k_merge.gather(2, und_token_indices_kv)).to(k_.dtype))
            k_.scatter_(dim=2, index=gen_token_indices_kv, src=self.key_layernorm_mot_gen(k_merge.gather(2, gen_token_indices_kv)).to(k_.dtype))
            
            q_merge = q_
            k_merge = k_

        q_merge = q_merge.to(v_merge.dtype)
        k_merge = k_merge.to(v_merge.dtype)

        # Restore from kv_cache and update
        if input_pos is not None:
            cache_kwargs = {"cache_position": input_pos}
            k_merge, v_merge = past_key_values.update(k_merge, v_merge, self.layer_idx, cache_kwargs)
        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = k_merge.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if n_kv_head != n_q_head and (input_pos is None or q_per_kv != 1):
            k_merge = k_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            v_merge = v_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        y = self.scaled_dot_product_attention(q_merge, k_merge, v_merge, attention_mask)

        # re-assemble all head outputs side by side
        y = y.reshape(bsz, -1, head_size * n_q_head)

        core_attn_out = y.gather(dim=1, index=und_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))
        gen_core_attn_out = y.gather(dim=1, index=gen_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))

        if cp_enabled:
            core_attn_out = maybe_to_split_seq(
                core_attn_out.view(bsz, und_seqlen, n_q_head, head_size)
            )
            gen_core_attn_out = maybe_to_split_seq(
                gen_core_attn_out.view(bsz, gen_seqlen, n_q_head, head_size)
            )
            n_q_head = maybe_to_normal_region_num_head(n_q_head)
            und_seqlen, gen_seqlen = core_attn_out.shape[1], gen_core_attn_out.shape[1]
            core_attn_out = core_attn_out.reshape(bsz, und_seqlen, n_q_head * head_size)
            gen_core_attn_out = gen_core_attn_out.reshape(bsz, gen_seqlen, n_q_head * head_size)

        und_hidden_states = self.o_proj(core_attn_out)
        gen_hidden_states = self.o_proj_mot_gen(gen_core_attn_out)
        return und_hidden_states, gen_hidden_states


MOE_LAYER_IMPL = {
    "deepseek": DeepSeekMoE,
    "hunyuan": HunyuanMoE,
    "flashinfer": FlashInferMoE,
    "qwen3": Qwen3VLSparesMoeBlock,
    "ep_moe": ExpertParallelMoE,
}


class HunyuanMultimodalLayer(nn.Module):
    def __init__(
            self,
            config: HunyuanMultimodalConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = CausalSelfAttention(config, layer_idx, **factory_kwargs)
        self.input_layernorm = config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs)
        self.post_attention_layernorm = config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs)
        if layer_idx >= config.moe_layer_num_skipped and (
                (isinstance(config.num_experts, int) and config.num_experts > 1)
                or (isinstance(config.num_experts, list) and max(config.num_experts) > 1)
        ):
            assert config.moe_impl in MOE_LAYER_IMPL, f"moe_impl {config.moe_impl} not supported."
            self.mlp = MOE_LAYER_IMPL[config.moe_impl](config, layer_idx, **factory_kwargs)
        else:
            self.mlp = HunyuanMLP(config, layer_idx, **factory_kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            input_pos: Optional[torch.Tensor] = None,
            past_key_values: Optional[HunyuanStaticCache] = None,
            und_token_indices: Optional[torch.Tensor] = None,   # not used here, only for consistency with MoT layer
            gen_token_indices: Optional[torch.Tensor] = None,   # not used here
    ):
        # Self Attention
        residual = hidden_states
        bsz, seqlen, n_embd = hidden_states.shape
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            input_pos=input_pos,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @property
    def _materialize_and_init_state(self):
        return self.__materialize_and_init_state

    @_materialize_and_init_state.setter
    def _materialize_and_init_state(self, value):
        self.__materialize_and_init_state = value
        # Invoke hooks for moe layers
        if hasattr(self.mlp, "_materialize_and_init_state"):
            self.mlp._materialize_and_init_state = value


class HunyuanMultimodalLayerMoT(HunyuanMultimodalLayer):
    def __init__(
            self,
            config: HunyuanMultimodalConfig,
            layer_idx: int,
            mot_und_frozen: bool = False,
            mot_gen_frozen: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            gen_config: Optional[HunyuanMultimodalConfig] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        gen_config = gen_config if gen_config is not None else config
        super().__init__(config, layer_idx, dtype, device)
        self.mot_und_frozen = mot_und_frozen
        self.mot_gen_frozen = mot_gen_frozen
        self.moe_layer_num_skipped = gen_config.moe_layer_num_skipped

        # TODO(kevinkhwu): image branch and video branch have different config style / signature.
        self.self_attn = CausalSelfAttentionMoT(config, gen_config, layer_idx, mot_und_frozen, mot_gen_frozen, **factory_kwargs)
        self.input_layernorm_mot_gen = gen_config.norm_class(
            gen_config.hidden_size, eps=gen_config.norm_eps, **factory_kwargs
        )
        self.post_attention_layernorm_mot_gen = gen_config.norm_class(
            gen_config.hidden_size, eps=gen_config.norm_eps, **factory_kwargs
        )

        if layer_idx >= gen_config.moe_layer_num_skipped and (
            (isinstance(gen_config.num_experts, int) and gen_config.num_experts > 1)
            or (isinstance(gen_config.num_experts, list) and max(gen_config.num_experts) > 1)
        ):
            assert gen_config.moe_impl in MOE_LAYER_IMPL, f"moe_impl {gen_config.moe_impl} not supported."
            self.mlp_mot_gen = MOE_LAYER_IMPL[gen_config.moe_impl](gen_config, layer_idx, **factory_kwargs)
        else:
            self.mlp_mot_gen = HunyuanMLP(gen_config, layer_idx, **factory_kwargs)

        if self.mot_und_frozen:
            self.input_layernorm.eval()
            self.input_layernorm.requires_grad_(False)
            self.post_attention_layernorm.eval()
            self.post_attention_layernorm.requires_grad_(False)
            self.mlp.eval()
            self.mlp.requires_grad_(False)

        if self.mot_gen_frozen:
            self.input_layernorm_mot_gen.eval()
            self.input_layernorm_mot_gen.requires_grad_(False)
            self.post_attention_layernorm_mot_gen.eval()
            self.post_attention_layernorm_mot_gen.requires_grad_(False)
            self.mlp_mot_gen.eval()
            self.mlp_mot_gen.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        past_key_values: Optional[HunyuanStaticCache] = None,
        und_token_indices: Optional[torch.Tensor] = None,
        gen_token_indices: Optional[torch.Tensor] = None,
    ):
        und_hidden_states, gen_hidden_states = hidden_states
        und_residual, gen_residual = und_hidden_states, gen_hidden_states

        # Pre-attn norm
        und_hidden_states = self.input_layernorm(und_hidden_states)
        gen_hidden_states = self.input_layernorm_mot_gen(gen_hidden_states)

        # Self attention
        core_attn_out = self.self_attn(
            (und_hidden_states, gen_hidden_states),
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            input_pos=input_pos,
            past_key_values=past_key_values,
            und_token_indices=und_token_indices,
            gen_token_indices=gen_token_indices,
        )

        und_hidden_states, gen_hidden_states = core_attn_out
        und_hidden_states = und_residual + und_hidden_states
        gen_hidden_states = gen_residual + gen_hidden_states

        # Pre-mlp norm
        und_residual, gen_residual = und_hidden_states, gen_hidden_states
        und_hidden_states = self.post_attention_layernorm(und_hidden_states)
        gen_hidden_states = self.post_attention_layernorm_mot_gen(gen_hidden_states)

        # Mlp
        und_hidden_states = self.collective_sync_mlp(self.mlp, und_hidden_states)
        gen_hidden_states = self.collective_sync_mlp(self.mlp_mot_gen, gen_hidden_states)

        und_hidden_states = und_residual + und_hidden_states
        gen_hidden_states = gen_residual + gen_hidden_states

        return (und_hidden_states, gen_hidden_states)

    def collective_sync_mlp(self, mlp_module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """ 如果 hidden_states 为空，依然创建 dummy input 并 forward 一次，以保证 FSDP 的顺序对齐 """
        if hidden_states.nelement() != 0:
            return mlp_module(hidden_states)
        assert hidden_states.ndim == 3
        bsz = hidden_states.shape[0] if hidden_states.shape[0] > 0 else 1
        dummy = hidden_states.new_zeros(bsz, 1, hidden_states.shape[-1])
        if self.training:
            mlp_out = mlp_module(dummy)
            return hidden_states + mlp_out.sum() * 0
        else:
            return mlp_module(dummy)

    @property
    def _materialize_and_init_state(self):
        return self.__materialize_and_init_state

    @_materialize_and_init_state.setter
    def _materialize_and_init_state(self, value):
        self.__materialize_and_init_state = value
        # Invoke hooks for moe layers
        if hasattr(self.mlp, "_materialize_and_init_state"):
            self.mlp._materialize_and_init_state = value
        if hasattr(self.mlp_mot_gen, "_materialize_and_init_state"):
            self.mlp_mot_gen._materialize_and_init_state = value


class HunyuanMultimodalBase(HunyuanMultimodalState):
    def __post_init__(
            self,
            config: HunyuanMultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            args: Namespace = None,
            initialize_weights: bool = True,
            gen_config: Optional[HunyuanMultimodalConfig] = None,   # for mot
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Set as protected member to avoid conflict with potential parent classes
        self._config = config
        self._gen_config = gen_config if gen_config is not None else config
        if self._config.use_mot:
            self._config_mot_gen = self._config.to_mot_gen_config()
        self._dtype = dtype
        self.patch_size = config.patch_size

        # For inference, args can be None
        self.args = args or Namespace()
        self.vit_frozen = getattr(args, "vit_frozen", True)
        self.vit_aligner_frozen = getattr(args, "vit_aligner_frozen", True)
        self.vit_precision = PRECISION_TO_TYPE[default(getattr(args, "vit_precision", None), dtype)]
        self.moe_aux_loss_coeff = getattr(args, "moe_aux_loss_coeff", 0.0)

        self.mot_und_frozen = getattr(args, "mot_und_frozen", False)
        self.mot_embed_norm_lm_head_frozen = getattr(args, "mot_embed_norm_lm_head_frozen", False)
        self.mot_gen_frozen = getattr(args, "mot_gen_frozen", False)

        # ======================================
        #     Define vae projector modules
        # ======================================
        if config.use_vae:
            vae_hidden_size = self._config_mot_gen.hidden_size if config.use_mot else config.hidden_size
            vae_config = self._config_mot_gen if config.use_mot else config
            if config.use_timestep_token:
                self.timestep_emb = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)
            if config.use_timestep_r_token:
                self.timestep_r_emb = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)
            if config.use_guidance_token:
                self.guidance_emb = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)

            # One for patch_embed and other for final_layer
            self.time_embed = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)
            self.time_embed_2 = TimestepEmbedder(hidden_size=vae_hidden_size, **factory_kwargs)

            # Image projection layers
            if getattr(config, 'img_proj_ndim', None) == 3:
                img_proj_kwargs = dict(dims=3, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            else:
                img_proj_kwargs = dict(dims=getattr(config, 'img_proj_ndim', 2))
            self.patch_embed = project_in_layer(config.img_proj_type, vae_config, **img_proj_kwargs, **factory_kwargs)
            self.final_layer = project_out_layer(config.img_proj_type, vae_config, **img_proj_kwargs, **factory_kwargs)

            if self.mot_gen_frozen:
                if config.use_timestep_token:
                    self.timestep_emb.eval()
                    self.timestep_emb.requires_grad_(False)
                if config.use_timestep_r_token:
                    self.timestep_r_emb.eval()
                    self.timestep_r_emb.requires_grad_(False)
                if config.use_guidance_token:
                    self.guidance_emb.eval()
                    self.guidance_emb.requires_grad_(False)
                self.time_embed.eval()
                self.time_embed.requires_grad_(False)
                self.time_embed_2.eval()
                self.time_embed_2.requires_grad_(False)
                self.patch_embed.eval()
                self.patch_embed.requires_grad_(False)
                self.final_layer.eval()
                self.final_layer.requires_grad_(False)

        # ======================================
        #     Define vit and aligner modules
        # ======================================
        if config.use_vit:
            from ..visual_encoders import load_vit

            self.vit = load_vit(
                vision_model_type=config.vit_type,
                vision_model_precision=self.vit_precision,
                device=device,
                require_grad=not self.vit_frozen,
                eval_mode=self.vit_frozen,
                vision_model_params=config.vit_config,
                no_load_pretrained=True,
            )
            self.vit_context = torch.no_grad if self.vit_frozen else nullcontext

        # The vit aligner is only for HunyuanImage3 which is trained from an LLM(Hunyuan-MoE-A13B).
        # For HunyuanImage3.5(Hunyuan-Gemini-3.5), we train from a VLM Hunyuan-MoE-A3B/A30B whose vit already
        # contains the aligner, so we do not need to define another vit aligner here.
        if config.use_vit_aligner:
            from ..autoregressive.mlp_layers import load_projector

            self.vit_aligner = load_projector(
                projector_type=config.vit_aligner_type,
                projector_params=dict(
                    input_dim=self.vit.config.hidden_size,
                    n_embed=config.hidden_size,
                    **config.vit_aligner_config,
                    device=device,
                    dtype=self.vit_precision,
                )
            )
            self.vit_aligner_context = torch.no_grad if self.vit_aligner_frozen else nullcontext

        # ======================================
        #       Define language modules
        # ======================================

        if config.use_mot:
            self.model = nn.ModuleDict(
                dict(
                    embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, **factory_kwargs),
                    layers=nn.ModuleList([
                        HunyuanMultimodalLayerMoT(
                            config, block_idx, self.mot_und_frozen, self.mot_gen_frozen,
                            gen_config=self._config_mot_gen if config.use_mot else self._gen_config,
                            **factory_kwargs
                        )
                        for block_idx in range(config.num_layers)
                    ]),
                    norm=config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs),
                )
            )
        else:
            assert get_parallel_state().cp_size == 1, 'cp is not implemented for non-mot model yet.'
            self.model = nn.ModuleDict(
                dict(
                    embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, **factory_kwargs),
                    layers=nn.ModuleList([
                        HunyuanMultimodalLayer(config, block_idx, **factory_kwargs)
                        for block_idx in range(config.num_layers)
                    ]),
                    norm=config.norm_class(config.hidden_size, eps=config.norm_eps, **factory_kwargs),
                )
            )

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, **factory_kwargs)

        self.use_chunked_ce_loss = bool(getattr(self.args, "use_chunked_ce_loss", False))
        self.chunked_ce_loss: Any = None
        if self.use_chunked_ce_loss:
            if config.tie_word_embeddings:
                raise ValueError("use_chunked_ce_loss requires tie_word_embeddings=False because ChunkedCELoss needs lm_head.")
            num_chunks = int(getattr(self.args, "chunked_ce_num_chunks", 8))
            self.chunked_ce_loss = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=num_chunks))
            self.chunked_ce_loss.set_lm_head(self.lm_head)

        if self.mot_embed_norm_lm_head_frozen:
            self.model.embed_tokens.eval()
            self.model.embed_tokens.requires_grad_(False)
            self.model.norm.eval()
            self.model.norm.requires_grad_(False)
            if not config.tie_word_embeddings:
                self.lm_head.eval()
                self.lm_head.requires_grad_(False)

        # ====================== Finish model building =====================

        # Initialize cached rope, supporting automatic cache update
        self.cached_rope = CachedRoPE(config)
        self.use_rope_sample_offsets = getattr(args, "use_rope_sample_offsets", False)

        # Initialize weights if needed
        self._prepare_reset_parameters()
        if initialize_weights:
            for name, module in self.named_modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def _prepare_reset_parameters(self):
        # Globally set Linear and Embedding init methods to normal
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.reset_parameters = normal_weight_reset_parameters(
                    std=self._config.init_std, bias_type="zeros").__get__(module)
        # Set specific module init methods if available
        for name, module in self.named_modules():
            if hasattr(module, "prepare_reset_parameters"):
                module.prepare_reset_parameters()

    @property
    def dtype(self):
        """Get the dtype of the model parameters."""
        if self._dtype is not None:
            return self._dtype
        # Fallback to getting dtype from model parameters
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # If no parameters, try buffers
            try:
                return next(self.buffers()).dtype
            except StopIteration:
                # Default fallback
                return torch.float32
    
    def get_config(self):
        return self._config

    def get_printable_layers(self):
        if self._config.moe_layer_num_skipped == 0:
            return [self.model["layers"][0]]
        elif self._config.moe_layer_num_skipped > 0:
            return [self.model["layers"][0], self.model["layers"][self._config.moe_layer_num_skipped]]
        return []

    def scatter_to_hidden_states(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        hidden_states: torch.Tensor,
        gen_hidden_states: Optional[torch.Tensor] = None,
        dim: int = 1,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, _ = hidden_states.shape
        n_embd = src.shape[-1]

        # attempt to scatter `src`` into `hidden_states`` when possible
        if n_embd == hidden_states.shape[-1]:
            hidden_states.scatter_(dim=dim, index=index, src=src)
        else:
            # scatter to `gen_hidden_states`` when hidden dim of `src`` and `hidden_states` differ
            assert gen_hidden_states is not None, "gen_hidden_states is required when hidden dim of src and hidden_states differ"
            assert gen_hidden_states.shape[-1] == n_embd, \
                f"Expect gen_hidden_states and src to have same hidden_size, but got {gen_hidden_states.shape[-1]} and {n_embd}"
    
            gen_hidden_states.scatter_(dim=dim, index=index, src=src)

        return hidden_states, gen_hidden_states

    def scatter_to_hidden_states_with_slice(
        self,
        slice_idx: int,
        src: torch.Tensor,
        index: torch.Tensor,
        hidden_states: torch.Tensor,
        gen_hidden_states: Optional[torch.Tensor] = None,
        dim: int = 1,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seqlen, _ = hidden_states.shape
        n_embd = src.shape[-1]

        # attempt to scatter `src`` into `hidden_states`` if possible
        if n_embd == hidden_states.shape[-1]:
            hidden_states[slice_idx:slice_idx+1].scatter_(dim=dim, index=index, src=src)
        else:
            # scatter to `gen_hidden_states`` when hidden dim of `src`` and `hidden_states` differ
            assert gen_hidden_states is not None, "gen_hidden_states is required when hidden dim of src and hidden_states differ"
            assert gen_hidden_states.shape[-1] == n_embd, \
                f"Expect gen_hidden_states and src to have same hidden_size, but got {gen_hidden_states.shape[-1]} and {n_embd}"
    
            gen_hidden_states[slice_idx:slice_idx+1].scatter_(dim=dim, index=index, src=src)

        return hidden_states, gen_hidden_states

    def instantiate_vae_image_tokens(
            self,
            hidden_states: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
            timesteps: BatchRaggedTensor,
            medias: BatchRaggedMedia,
            media_mask: torch.Tensor,
    ):
        """
        Instantiate the VAE image embeddings into the input embedding sequence(x).
        If x is None, using ts and images to create a new input embedding sequence.

        Args:
            hidden_states (torch.Tensor): input sequence, (bsz, seqlen, n_embd)
            timesteps (BatchRaggedTensor): ts can be a 1-D tensor, or a list of 1-D tensors
            medias (BatchRaggedMedia): images can be a 4-D tensor, or a list of 4-D tensors,
                or a list of lists of 3-D tensors.
            media_mask (torch.Tensor): (bsz, seqlen)

        Returns:
            Instantiated input sequence
        """
        if hidden_states is None:
            # Only for inference in non-first step image generation
            hidden_size = self._config_mot_gen.hidden_size if self._config.use_mot else self._config.hidden_size

            if isinstance(medias, list):
                # batch多分辨率推理：需要逐sample做patch_embed再进行1D padding，避免2D空间padding导致pad token与有效token交错
                emb_list = []
                for i in range(len(medias)):
                    t_i = timesteps[i:i+1] if isinstance(timesteps, torch.Tensor) else timesteps[i]
                    t_emb_i = self.time_embed(t_i)
                    img_emb_i, _, _ = self.patch_embed(medias[i], t_emb_i)
                    emb_list.append(img_emb_i)

                max_tokens = max(e.size(1) for e in emb_list)
                padded_img_emb = torch.zeros(len(medias), max_tokens, hidden_size, device=emb_list[0].device, dtype=emb_list[0].dtype)
                for i, emb in enumerate(emb_list):
                    padded_img_emb[i, :emb.size(1), :] = emb[0]

                timestep_emb = self.timestep_emb(timesteps).reshape(len(medias), -1, hidden_size)
                hidden_states = torch.cat([timestep_emb, padded_img_emb], dim=1)
                return hidden_states

            t_emb = self.time_embed(timesteps)
            image_emb, _, _ = self.patch_embed(medias, t_emb)
            hidden_size = self._config_mot_gen.hidden_size if self._config.use_mot else self._config.hidden_size
            timestep_emb = self.timestep_emb(timesteps).reshape(medias.size(0), -1, hidden_size)
            hidden_states = torch.cat([timestep_emb, image_emb], dim=1)
            return hidden_states

        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, n_embd = hidden_states.shape
        assert isinstance(medias, (torch.Tensor, list)), f"images should be BatchRaggedMedia, got {type(medias)}"

        if isinstance(medias, torch.Tensor):
            assert medias.ndim in [4, 5], f"images should be a 4-D or 5-D tensor, got {medias.ndim}-D tensor"
            assert isinstance(timesteps, torch.Tensor), f"timesteps should be 1-D tensor, got {type(timesteps)}"

            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            t_emb = self.time_embed(timesteps)     # (bsz, n_embd)
            media_seq, *_ = self.patch_embed(medias, t_emb)   # (bsz, num_patches, n_embd)
            media_index = index.masked_select(media_mask.bool()).reshape(bsz, -1)   # (bsz, num_patches)
            assert media_seq.size(1) == media_index.size(1), \
                f"image_seq ({list(media_seq.size())}) has inconsistent shape with index ({list(media_index.size())})"
            n_embd = media_seq.shape[-1]
            index_exp = media_index.unsqueeze(-1).repeat(1, 1, n_embd)
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                media_seq.to(hidden_states.dtype), index_exp, hidden_states, gen_hidden_states
            )

        else:   # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            for i in range(len(medias)):
                media_i = medias[i]
                t_i = timesteps[i:i+1] if isinstance(timesteps, torch.Tensor) else timesteps[i]
                
                t_i_emb = self.time_embed(t_i)      # (n_i, n_embd)

                if isinstance(media_i, torch.Tensor):
                    media_i_seq, *_ = self.patch_embed(media_i, t_i_emb)  # (n_i, num_patches, n_embd)

                elif isinstance(media_i, list):
                    media_i_seq_list = []
                    for j in range(len(media_i)):
                        media_ij = media_i[j].unsqueeze(0)
                        assert media_ij.ndim in [4, 5], \
                            f"image_ij should have size of (1, C, H, W) or (1, C, D, H, W), got {list(media_ij.size())}"
                        media_ij_seq, *_ = self.patch_embed(media_ij, t_i_emb[j:j + 1])  # (1, num_patches, n_embd)
                        media_i_seq_list.append(media_ij_seq)
                    media_i_seq = torch.cat(media_i_seq_list, dim=1)    # (1, Σj num_patches_j, n_embd)

                else:
                    raise TypeError(f"image_i should be a 4-D or 5-D tensor or a list, got {type(media_i)}")

                media_i_index = index[i:i + 1].masked_select(media_mask[i:i + 1].bool()).reshape(1, -1)  # (1, img_seqlen)
                n_embd = media_i_seq.shape[-1]
                media_i_index_exp = media_i_index.unsqueeze(-1).repeat(1, 1, n_embd)
                media_i_seq_flat = media_i_seq.reshape(1, -1, n_embd)
                assert media_i_seq_flat.shape[1] == media_i_index_exp.shape[1], \
                    f"media_i_seq_flat ({list(media_i_seq_flat.size())}) has inconsistent shape with media_i_index_exp ({list(media_i_index_exp.size())})"
                hidden_states, gen_hidden_states = self.scatter_to_hidden_states_with_slice(
                    i, media_i_seq_flat.to(hidden_states.dtype), media_i_index_exp, hidden_states, gen_hidden_states
                )

        if gen_hidden_states is not None:
            return hidden_states, gen_hidden_states

        return hidden_states

    def _forward_vision_encoder(self, images, **image_kwargs):
        with self.vit_context():
            image_embeds = self.vit(images, **image_kwargs)
        
        if isinstance(image_embeds, tuple):
            image_embeds, deepstack_image_embeds = image_embeds
        else:
            deepstack_image_embeds = None
            image_embeds = image_embeds.last_hidden_state
        if self._config.use_vit_aligner:
            with self.vit_aligner_context():
                image_embeds = self.vit_aligner(image_embeds)

        return image_embeds, deepstack_image_embeds

    @staticmethod
    def _accumulate_deepstack_embeds(all_embeds, new_embeds):
        if new_embeds is None:
            return all_embeds
        if all_embeds is None:
            all_embeds = [[] for _ in range(len(new_embeds))]
        for layer_idx, layer_embeds in enumerate(new_embeds):
            all_embeds[layer_idx].append(layer_embeds)
        return all_embeds

    def instantiate_vit_image_tokens(
            self,
            hidden_states: torch.Tensor,
            images: torch.Tensor | list[torch.Tensor],
            image_masks: torch.Tensor,
            image_kwargs: dict[str, torch.Tensor],
    ):
        """
        Encode images using vision encoder(vit), and then instantiate the image embeddings into
        the input embedding sequence.

        Args:
            hidden_states (torch.Tensor): input sequence, (bsz, seqlen, n_embd)
            images (torch.Tensor | list[torch.Tensor]): images can be a 3-D or 4-D tensor, or a list of tensors.
            image_masks (torch.Tensor): mask for the images, (bsz, seqlen)
            image_kwargs (dict[str, torch.Tensor]): additional keyword arguments for the image encoder

        Returns:
            Instantiated input sequence
        """

        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, _ = hidden_states.shape
        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)

        if isinstance(images, torch.Tensor):
            assert images.ndim in [3, 4, 5], f"images should be a 3-D, 4-D, or 5-D tensor, got {images.ndim}-D tensor."
            if images.ndim in [4, 5]:
                bsz, n = images.shape[:2]
                images = images.view(bsz * n, *images.shape[2:])
                image_kwargs = image_kwargs if image_kwargs is not None else {}
                for k, v in image_kwargs.items():
                    image_kwargs[k] = v.reshape(bsz * n, *v.shape[2:])
            else:
                n = 1
            image_embeds, deepstack_image_embeds = self._forward_vision_encoder(images, **image_kwargs)
            # image_seqlen = image_embeds.size(1)

            # image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
            # hidden_states.scatter_(
            #     dim=1,
            #     index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
            #     src=image_embeds.reshape(bsz, n * image_seqlen, n_embd).to(hidden_states.dtype),
            # )

            image_seqlen, n_embd = image_embeds.size(1), image_embeds.size(-1)

            image_scatter_index = index.masked_select(image_masks.bool()).reshape(bsz, -1)
            index = image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd)
            src = image_embeds.reshape(bsz, n * image_seqlen, n_embd)
            assert src.shape[1] == index.shape[1], \
                f"src ({list(src.size())}) has inconsistent shape with index ({list(index.size())})"
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states
            )

        elif isinstance(images, list):
            # debug code
            all_deepstack_embeds = None
            for i, (image, image_mask) in enumerate(zip(images, image_masks)):
                start_index = 0
                image_scatter_index = index[i].masked_select(image_mask.bool()).reshape(1, -1)
                for j, singel_image in enumerate(image):
                    
                    cur_kwargs = {k: v[i][j:j+1] for k, v in image_kwargs.items()} if image_kwargs is not None else {}
                    if isinstance(singel_image, list):
                        image_embed_list = []
                        for _single_image in singel_image:
                            image_embed, deepstack_image_embeds = self._forward_vision_encoder(_single_image, **cur_kwargs)
                            image_embed_list.append(image_embed)
                            all_deepstack_embeds = self._accumulate_deepstack_embeds(all_deepstack_embeds, deepstack_image_embeds)
                        image_embed = torch.cat(image_embed_list, dim=1)
                        if image_embed.ndim == 3:
                            n, image_seqlen, n_embd = image_embed.shape
                            image_embed = image_embed.reshape(n * image_seqlen, n_embd)
                        else:
                            n_embd = image_embed.shape[-1]
                    else:
                        image_embed, deepstack_image_embeds = self._forward_vision_encoder(singel_image, **cur_kwargs)
                        all_deepstack_embeds = self._accumulate_deepstack_embeds(all_deepstack_embeds, deepstack_image_embeds)
                        if image_embed.ndim == 3:
                            n, image_seqlen, n_embd = image_embed.shape
                            image_embed = image_embed.reshape(n * image_seqlen, n_embd)
                        else:
                            n_embd = image_embed.shape[-1]

                    # image_scatter_index_j = image_scatter_index[:, start_index:start_index + image_embed.shape[0]]
                    # start_index += image_embed.shape[0]

                    # hidden_states[i:i+1].scatter_(
                    #     dim=1,
                    #     index=image_scatter_index_j.unsqueeze(-1).repeat(1, 1, n_embd),
                    #     src=image_embed.reshape(1, -1, n_embd).to(hidden_states.dtype),
                    # )

                    image_scatter_index_j = image_scatter_index[:, start_index:start_index + image_embed.shape[0]]
                    image_scatter_index_j = image_scatter_index_j.unsqueeze(-1).repeat(1, 1, n_embd)
                    image_embed = image_embed.reshape(1, -1, n_embd)
                    start_index += image_embed.shape[1]

                    assert image_scatter_index_j.shape[1] == image_embed.shape[1], \
                        f"image_scatter_index_j ({list(image_scatter_index_j.size())}) has inconsistent shape with image_embed ({list(image_embed.size())})"
                    hidden_states, gen_hidden_states = self.scatter_to_hidden_states_with_slice(
                        i, image_embed.to(hidden_states.dtype), image_scatter_index_j, hidden_states, gen_hidden_states
                    )
            
            # Concatenate all deepstack_embeds for each layer
            if all_deepstack_embeds is not None:
                deepstack_image_embeds = []
                for layer_embeds_list in all_deepstack_embeds:
                    # Filter out None values and ensure all elements are tensors
                    valid_embeds = [e for e in layer_embeds_list if e is not None]
                    if len(valid_embeds) == 0:
                        continue
                    # If any element is a list, flatten it first
                    flattened_embeds = []
                    for e in valid_embeds:
                        if isinstance(e, list):
                            flattened_embeds.extend([item for item in e if isinstance(item, torch.Tensor)])
                        elif isinstance(e, torch.Tensor):
                            flattened_embeds.append(e)
                    if len(flattened_embeds) == 1:
                        # If there's only one element, use it directly
                        deepstack_image_embeds.append(flattened_embeds[0])
                    elif len(flattened_embeds) > 1:
                        # If there are multiple elements, concatenate them
                        deepstack_image_embeds.append(torch.cat(flattened_embeds, dim=0))
            else:
                deepstack_image_embeds = None
        else:
            raise ValueError(f"und_images should be Tensor or List, but got {type(images)}")

        if gen_hidden_states is not None:
            return (hidden_states, gen_hidden_states), deepstack_image_embeds

        return hidden_states, deepstack_image_embeds

    def instantiate_continuous_tokens(
            self,
            hidden_states: torch.Tensor,
            emb_layer: nn.Module,
            scatter_src: Optional[BatchRaggedTensor] = None,
            scatter_index: Optional[BatchRaggedTensor] = None,
    ):
        if isinstance(hidden_states, tuple):
            hidden_states, gen_hidden_states = hidden_states
        else:
            gen_hidden_states = None

        bsz, seqlen, _ = hidden_states.shape

        if isinstance(scatter_src, list):
            for i, scatter_src_i in enumerate(scatter_src):
                src = emb_layer(scatter_src_i)  # (n, n_embd)
                n_embd = src.shape[-1]
                index = scatter_index[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, n_embd)
                src = src.reshape(1, -1, n_embd)

                assert index.shape[1] == src.shape[1], \
                    f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
                hidden_states, gen_hidden_states = self.scatter_to_hidden_states_with_slice(
                    i, src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states
                )

        else:
            src = emb_layer(scatter_src.reshape(-1))    # (bsz * n, n_embd)
            n_embd = src.shape[-1]
            index = scatter_index.unsqueeze(-1).repeat(1, 1, n_embd)
            src = src.reshape(bsz, -1, n_embd)

            assert index.shape[1] == src.shape[1], \
                f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
            hidden_states, gen_hidden_states = self.scatter_to_hidden_states(
                src.to(hidden_states.dtype), index, hidden_states, gen_hidden_states
            )

        if gen_hidden_states is not None:
            return hidden_states, gen_hidden_states

        return hidden_states

    def get_image_tokens_hw(self, images: BatchRaggedMedia):
        assert isinstance(images, (torch.Tensor, list)), f"images should be BatchRaggedMedia, got {type(images)}"
        if isinstance(images, torch.Tensor):
            token_h = images.shape[-2] // self.patch_size
            token_w = images.shape[-1] // self.patch_size
        else:
            token_h, token_w = [], []
            for image_i in images:
                assert isinstance(image_i, (torch.Tensor, list)), \
                    f"image_i should be a tensor or a list of tensors, got {type(image_i)}"
                if isinstance(image_i, torch.Tensor):
                    token_h.append(image_i.shape[-2] // self.patch_size)
                    token_w.append(image_i.shape[-1] // self.patch_size)
                else:
                    token_h.append([])
                    token_w.append([])
                    for j in range(len(image_i)):
                        token_h[-1].append(image_i[j].shape[-2] // self.patch_size)
                        token_w[-1].append(image_i[j].shape[-1] // self.patch_size)
        return token_h, token_w

    def ragged_final_layer(self, hidden_states, image_mask, timesteps, token_h, token_w, first_step=None, batch_image_sizes=None):
        n_embd = hidden_states.size(-1)

        if batch_image_sizes is not None:
            # For batch multi-resolution inference
            # bsz可能包含cond/uncond大小，而batch_image_sizes只包含实际的image大小
            bsz = hidden_states.size(0)
            actual_bsz = len(batch_image_sizes)
            pred = []
            for i in range(bsz):
                h_lat, w_lat = batch_image_sizes[i % actual_bsz]
                th_i = h_lat // self.patch_size
                tw_i = w_lat // self.patch_size
                n_tokens_i = th_i * tw_i
                assert n_tokens_i == image_mask[i].sum().item(), \
                    f"n_tokens_i ({n_tokens_i}) has inconsistent shape with image_mask[i].sum().item() ({image_mask[i].sum().item()})"

                if first_step is False:
                    # 非首步：hidden_states = [timestep_emb, img_emb(1D padded)]，因为有效 token 在前，直接按数量截取
                    image_output_i = hidden_states[i:i+1, 1:1+n_tokens_i, :]
                else:
                    # 首步：从完整序列中按 image_mask 布尔索引提取 image token
                    image_output_i = hidden_states[i, image_mask[i].bool(), :].unsqueeze(0)

                t_emb_i = self.time_embed_2(timesteps[i:i+1])
                pred_i = self.final_layer(image_output_i, t_emb_i, th_i, tw_i)
                pred.append(pred_i)
            return pred

        if isinstance(timesteps, torch.Tensor):
            # When timesteps is a tensor, images must be a 4-D tensor (B, C, H, W), which means only one target image
            t_emb = self.time_embed_2(timesteps)
            if first_step is False:
                # only for gen_image non-first-step inference
                image_output = hidden_states[:, 1:, :]
            else:   # first_step is True or None
                image_output = hidden_states.masked_select(
                    image_mask.unsqueeze(-1).bool()).reshape(-1, token_h * token_w, n_embd)
            pred = self.final_layer(image_output, t_emb, token_h, token_w)
        else:
            # When timesteps is a list, images must be a list of 4-D tensors or a list of list of 3-D tensors, and token_h and token_w must be a list of int or a list of list of int.
            # In this case, each line of the image_mask may contain different number of Trues, leading
            # the `reshape(batch_size, ...)` is not possible.
            sections = image_mask.sum(1).tolist()
            image_output = hidden_states.masked_select(
                image_mask.unsqueeze(-1).bool()).reshape(-1, n_embd).split(sections)
            pred = []
            for image_output_i, t_i, token_h_i, token_w_i in zip(image_output, timesteps, token_h, token_w):
                t_emb_i = self.time_embed_2(t_i)
                if isinstance(token_h_i, int):
                    # corresponds to image_output as a list of 4-D tensors, image_output_i as a 4-D tensor
                    image_output_i = image_output_i.reshape(-1, token_h_i * token_w_i, n_embd)
                    pred_i = self.final_layer(image_output_i, t_emb_i, token_h_i, token_w_i)
                    pred.append(pred_i)
                else:
                    # corresponds to image_output as a list of list of 3-D tensors, image_output_i as a list of 3-D tensors
                    subsections = [token_h_ij * token_w_ij for token_h_ij, token_w_ij in zip(token_h_i, token_w_i)]
                    assert sum(subsections) == image_output_i.shape[0], \
                        f"sum(subsections) ({sum(subsections)}) has inconsistent shape with image_output_i.shape[0] ({image_output_i.shape[0]})"
                    image_output_i = image_output_i.split(subsections)
                    pred_i = []
                    for j, image_output_ij in enumerate(image_output_i):
                        pred_ij = self.final_layer(image_output_ij[None], t_emb_i[j:j+1], token_h_i[j], token_w_i[j])
                        pred_i.append(pred_ij)
                    pred.append(pred_i) # a list of list of 4-D tensors [B x (N_i x [1, C, H_ij, W_ij])]
        return pred

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        # for Qwen3VL, it use features extracted from vit to enhance image hidden states
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        # Handle case where visual_embeds might be a list instead of a tensor
        if isinstance(visual_embeds, list):
            # If it's a list of tensors, concatenate them or use the single element
            if len(visual_embeds) == 0:
                raise ValueError("visual_embeds is an empty list")
            elif len(visual_embeds) == 1:
                # If there's only one element, use it directly
                visual_embeds = visual_embeds[0]
                if not isinstance(visual_embeds, torch.Tensor):
                    raise ValueError(f"visual_embeds list contains non-tensor element: {type(visual_embeds)}")
            else:
                # If there are multiple elements, concatenate them
                if all(isinstance(e, torch.Tensor) for e in visual_embeds):
                    visual_embeds = torch.cat(visual_embeds, dim=0)
                else:
                    raise ValueError(f"visual_embeds list contains non-tensor elements")
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # debug code
        bsz = hidden_states.shape[0]
        # Process each batch separately
        for i in range(bsz):
            batch_mask = visual_pos_masks[i]  # [seqlen]
            batch_hidden = hidden_states[i]  # [seqlen, n_embd]
            # Get the number of True positions in this batch
            num_visual_tokens = batch_mask.sum().item()
            if num_visual_tokens == 0:
                continue
            
            # Extract visual positions for this batch
            visual_hidden = batch_hidden[batch_mask]  # [num_visual_tokens, n_embd]
            
            # visual_embeds should match the number of visual tokens
            # If visual_embeds has more tokens than this batch, take the first num_visual_tokens
            # If visual_embeds has fewer tokens, pad or handle accordingly
            if visual_embeds.shape[0] >= num_visual_tokens:
                batch_visual_embeds = visual_embeds[:num_visual_tokens]
            else:
                # If visual_embeds has fewer tokens, we need to handle this case
                # For now, we'll pad with zeros or repeat the last token
                batch_visual_embeds = visual_embeds
                if visual_embeds.shape[0] < num_visual_tokens:
                    # Pad with the last token
                    padding = visual_embeds[-1:].repeat(num_visual_tokens - visual_embeds.shape[0], 1)
                    batch_visual_embeds = torch.cat([visual_embeds, padding], dim=0)
            
            # Update hidden states
            visual_hidden = visual_hidden + batch_visual_embeds
            batch_hidden[batch_mask] = visual_hidden
            hidden_states[i] = batch_hidden
        
        return hidden_states

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,  # bsz x seqlen
            attention_mask: Optional[torch.Tensor] = None,  # bsz x 1 x seqlen x seqlen
            rope_image_info: Optional[list[list[tuple[slice, tuple[int, int], dict]]]] = None,
            return_dict: bool = True,
            # for gen text
            target: Optional[torch.Tensor] = None,  # bsz x seqlen, for calculating discrete loss
            text_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            # for gen images
            images: Optional[BatchRaggedMedia] = None,  # bsz x c x h x w, or bsz x (n_i x (c x h_ij x w_ij))
            image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            timesteps: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (n_i)
            timesteps_index: Optional[BatchRaggedTensor] = None,  # bsz x k, or bsz x (k_i)
            guidance: Optional[torch.Tensor] = None,
            guidance_index: Optional[BatchRaggedTensor] = None,
            timestep_r: Optional[BatchRaggedTensor] = None,
            timestep_r_index: Optional[BatchRaggedTensor] = None,
            # for cond images
            cond_vae_images: Optional[BatchRaggedMedia] = None,  # bsz x c x h x w, or bsz x (m_i x (c x h_ij x w_ij))
            cond_vae_image_mask: Optional[torch.Tensor] = None,  # bsz x seqlen
            cond_timesteps: Optional[BatchRaggedTensor] = None,  # bsz, or bsz x (m_i)
            cond_timesteps_index: Optional[BatchRaggedTensor] = None,
            cond_vit_images: Optional[BatchRaggedMedia] = None,
            cond_vit_image_mask: Optional[torch.Tensor] = None,
            cond_vit_image_kwargs: Optional[dict[str, Any]] = None,
            # only for training
            diffusion_loss_fn: Optional[nn.Module] = None,  # for calculating diffusion loss, can be None when sampling
            image_loss_weight: float = 0.0,
            gather_text_tokens: bool = False,
            dataset_tag: str | None = None,     # for labeling multi-task losses
            return_loss: Optional[bool] = None,  # used for rl training
            # only for inference
            input_pos: Optional[torch.Tensor] = None,  # bsz x seq_len-1, used for KVCache
            past_key_values: Optional[HunyuanStaticCache] = None,
            mode: Optional[str] = None,
            first_step: Optional[bool] = None,
            # only for pipeline parallelism (not implemented yet, just a placeholder)
            ut: Optional[torch.Tensor] = None,  # velocity target for flow matching, passed through pipeline stages
            und_token_indices: Optional[torch.Tensor] = None,
            gen_token_indices: Optional[torch.Tensor] = None,
            sample_offsets: Optional[list[torch.Tensor]] = None,
            batch_image_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> HunyuanMultimodalOutput | tuple:
        # Sanity check
        if input_ids is None and images is None:
            raise ValueError("Either input_ids or images should be provided.")
        if input_ids is not None:
            bsz = input_ids.size(0)
            device = input_ids.device
        else:
            bsz = images.size(0) if isinstance(images, torch.Tensor) else len(images)
            device = get_device(images)
        if self.training:
            seqlen = input_ids.size(1)
        else:
            # For inference, we always set seqlen to maximum length to simplify the rope cache handling
            seqlen = self._config.max_position_embeddings
        assert self._config.max_position_embeddings >= seqlen, (
            f"Cannot forward sequence of length {seqlen}, "
            f"max position embeddings is only {self._config.max_position_embeddings}, "
            f"try set --max-position-embeddings to a larger value."
        )
        if gather_text_tokens and bsz > 1:
            raise ValueError(f"gather_text_tokens only supports batch size of 1, got {bsz}.")

        # Calculate multimodal 2d rope
        cos, sin = self.cached_rope(
            seqlen, device, rope_media_info=rope_image_info, input_pos=input_pos, sample_offsets=sample_offsets if self.use_rope_sample_offsets else None,
        )
        # QWen VL apply rope in bf16 precision, both for cos and sin and hidden states
        if not self._config.apply_rope_in_fp32:
            cos = cos.to(dtype=self.dtype)
            sin = sin.to(dtype=self.dtype)

        # === Map token ids to embeddings ===
        if input_ids is not None:
            hidden_states = self.model["embed_tokens"](input_ids)     # (bsz, seqlen, n_embd)
            if is_bitwise_align_mode():
                hidden_states = hidden_states.clone() # nn.Embedding with BackwardHook return a view, need clone.

            if self._config.use_mot and self._config_mot_gen.hidden_size != self._config.hidden_size:
                und_hidden_states = hidden_states
                gen_hidden_states = torch.zeros(
                    (hidden_states.size(0), hidden_states.size(1), self._config_mot_gen.hidden_size),
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                hidden_states = (und_hidden_states, gen_hidden_states)
        else:
            hidden_states = None    # only for non-first step inference of the image generation

        # === Input layers ===
        deepstack_image_embeds = None
        if images is not None:
            hidden_states = self.instantiate_vae_image_tokens(hidden_states, timesteps, images, image_mask)

        if cond_vae_images is not None:
            hidden_states = self.instantiate_vae_image_tokens(hidden_states, cond_timesteps, cond_vae_images, cond_vae_image_mask)

        if cond_vit_images is not None:
            hidden_states, deepstack_image_embeds = self.instantiate_vit_image_tokens(hidden_states, cond_vit_images, cond_vit_image_mask, cond_vit_image_kwargs)

        if timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.timestep_emb, scatter_src=timesteps, scatter_index=timesteps_index)

        if cond_timesteps_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.timestep_emb, scatter_src=cond_timesteps, scatter_index=cond_timesteps_index)
        
        if guidance_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.guidance_emb, scatter_src=guidance, scatter_index=guidance_index)

        if timestep_r_index is not None:
            hidden_states = self.instantiate_continuous_tokens(hidden_states, emb_layer=self.timestep_r_emb, scatter_src=timestep_r, scatter_index=timestep_r_index)

        # Split understanding and generation tokens when using MoT
        if self._config.use_mot:
            assert und_token_indices is not None and gen_token_indices is not None, \
                "und_token_indices and gen_token_indices must be provided when using MoT"
            # when mot training first_step is None, when mot sampling first_step is True or False
            if isinstance(hidden_states, tuple):
                # two cases
                # mot training, und and gen have different hidden size, first_step is None
                # mot sampling, und and gen have different hidden size, first_step is True
                und_hidden_states, gen_hidden_states = hidden_states
            else:
                # three cases
                # 1. mot training, und and gen have the same hidden size, first_step is None, und_hidden_states = hidden_states
                # 2. mot sampling, und and gen have the same hidden size, first_step is True, und_hidden_states = hidden_states
                # 3. mot sampling, und and gen have the same or different hidden size, first_step is False, hidden_states is timestep token + image tokens, they are all gen tokens, so we create a zero length und_hidden_states
                und_hidden_states = (
                    hidden_states.new_zeros(hidden_states.shape[0], 0, self._config.hidden_size)
                    if first_step is False else hidden_states
                )
                gen_hidden_states = hidden_states

            und_token_indices_ = und_token_indices.unsqueeze(-1).expand(-1, -1, und_hidden_states.shape[-1])
            gen_token_indices_ = gen_token_indices.unsqueeze(-1).expand(-1, -1, gen_hidden_states.shape[-1])

            und_hidden_states = und_hidden_states.gather(dim=1, index=und_token_indices_)
            gen_hidden_states = gen_hidden_states.gather(dim=1, index=gen_token_indices_)

            if get_parallel_state().cp_size > 1:
                und_hidden_states = maybe_scatter_seq(und_hidden_states)
                gen_hidden_states = maybe_scatter_seq(gen_hidden_states)

            hidden_states = (und_hidden_states, gen_hidden_states)
        else:
            assert get_parallel_state().cp_size == 1, 'cp is not implemented for non-mot model'


        # === Transformer blocks ===
        for layer_idx, layer in enumerate(self.model["layers"]):    # noqa
            layer_inputs = [
                hidden_states, 
                attention_mask, 
                (cos, sin), 
                input_pos, 
                past_key_values, 
                und_token_indices, 
                gen_token_indices
            ]
            hidden_states = layer(*layer_inputs)
            if deepstack_image_embeds is not None and layer_idx in range(len(deepstack_image_embeds)):
                if isinstance(hidden_states, tuple):
                    und_hs, gen_hs = hidden_states
                    und_hs = self._deepstack_process(
                        und_hs,
                        cond_vit_image_mask,
                        deepstack_image_embeds[layer_idx],
                    )
                    hidden_states = (und_hs, gen_hs)
                else:
                    hidden_states = self._deepstack_process(
                        hidden_states,
                        cond_vit_image_mask,
                        deepstack_image_embeds[layer_idx],
                    )

        if isinstance(hidden_states, tuple) and get_parallel_state().cp_size > 1:
            und_hs, gen_hs = hidden_states
            hidden_states = (maybe_gather_seq(und_hs), maybe_gather_seq(gen_hs))

        # Scatter understanding/generation tokens back to hidden_states
        if isinstance(hidden_states, tuple):
            und_hidden_states_, gen_hidden_states_ = hidden_states
            bsz = und_hidden_states_.shape[0]
            und_seqlen = und_hidden_states_.shape[1]
            gen_seqlen = gen_hidden_states_.shape[1]
            
            und_hidden_states = torch.zeros(
                (bsz, und_seqlen+gen_seqlen, und_hidden_states_.shape[-1]), 
                device=und_hidden_states_.device, dtype=und_hidden_states_.dtype
            )
            und_hidden_states.scatter_(dim=1, index=und_token_indices_.to(und_hidden_states_.device), src=und_hidden_states_)

            gen_hidden_states = torch.zeros(
                (bsz, und_seqlen+gen_seqlen, gen_hidden_states_.shape[-1]),
                device=gen_hidden_states_.device, dtype=gen_hidden_states_.dtype
            )
            gen_hidden_states.scatter_(dim=1, index=gen_token_indices_.to(gen_hidden_states_.device), src=gen_hidden_states_)

        else:
            und_hidden_states, gen_hidden_states = hidden_states, hidden_states

        # === Output layers ===
        # -- image tokens
        if images is not None:
            # images: 4-D tensor (B, C, H, W), token_h: int, token_w: int
            # images: list of 4-D tensors [B x (N, C, H, W)], token_h: list of int (all H // patch_size), token_w: list of int (all W // patch_size)
            # images: list of list of 3-D tensors [B x (N_i x [C, H_ij, W_ij])], token_h: list of list of int, token_w: list of list of int
            token_h, token_w = self.get_image_tokens_hw(images)
            gen_hidden_states = gen_hidden_states.to(device=get_device(images))
            diff_pred = self.ragged_final_layer(
                gen_hidden_states, image_mask, timesteps, token_h, token_w, first_step, batch_image_sizes=batch_image_sizes)
        else:
            diff_pred = None

        # -- text tokens
        if input_ids is None or mode == "gen_image":
            logits = None
        else:
            und_hidden_states = self.model["norm"](und_hidden_states)
            skip_logits_for_chunked_ce = (
                self.training
                and return_loss is not False
                and self.use_chunked_ce_loss
                and self.chunked_ce_loss is not None
            )
            if skip_logits_for_chunked_ce:
                logits = None
            elif not self._config.tie_word_embeddings:
                logits = self.lm_head(und_hidden_states)  # (bsz, seqlen, vocab_size)
            else:
                logits = F.linear(und_hidden_states, self.model.embed_tokens.weight)

        # -- for inference
        if not self.training or return_loss is False:
            if not return_dict:
                return logits, past_key_values, diff_pred
            return HunyuanMultimodalOutput(
                # No need to call .float(), because transformers will handle the dtype conversion.
                logits=logits,
                past_key_values=past_key_values,
                diffusion_prediction=diff_pred,
            )

        # === Calculate losses ===
        losses = {}
        loss = 0.0
        # -- text loss
        use_global_discrete_loss_average = getattr(self.args, "use_global_discrete_loss_average", False)
        if text_mask.max().item() > 0:
            use_chunked_ce_loss = self.use_chunked_ce_loss and self.chunked_ce_loss is not None
            num_valid_tokens = text_mask.sum().float().to(und_hidden_states.device)
            discrete_loss_sum = None

            if gather_text_tokens and (logits is None or logits.size(0) == 1):
                bool_mask = text_mask[0].bool()
                valid_target = target[0, bool_mask]
                if use_chunked_ce_loss:
                    valid_hidden_states = und_hidden_states[0, bool_mask].unsqueeze(0)
                    valid_target = valid_target.unsqueeze(0)
                    discrete_loss_sum = self.chunked_ce_loss(
                        valid_hidden_states, valid_target, ignore_index=-100
                    )
                else:
                    valid_logits = logits[0, bool_mask]
                    discrete_loss_sum = torch.nn.functional.cross_entropy(
                        valid_logits.float(), valid_target, ignore_index=-100, reduction="sum"
                    )
            elif use_chunked_ce_loss:
                discrete_loss_sum = self.chunked_ce_loss(
                    und_hidden_states, target, ignore_index=-100
                )
            else:
                discrete_loss_sum = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(), target.view(-1), ignore_index=-100, reduction="sum"
                )

            # When use_global_discrete_loss_average: output sum and count for trainer to all_reduce and average globally
            if use_global_discrete_loss_average:
                losses["discrete_loss_sum"] = discrete_loss_sum
                losses["discrete_loss_count"] = num_valid_tokens.to(torch.float64)
                # Do not add to loss here; trainer will all_reduce and add global average
                if dataset_tag is not None:
                    losses[f"{dataset_tag}_text_loss"] = (discrete_loss_sum / (num_valid_tokens + 1e-8)).detach()
                else:
                    losses["text_loss"] = (discrete_loss_sum / (num_valid_tokens + 1e-8)).detach()
            else:
                discrete_loss = discrete_loss_sum / (num_valid_tokens + 1e-8)
                if dataset_tag is not None:
                    losses[f"{dataset_tag}_text_loss"] = discrete_loss.detach()
                else:
                    losses["text_loss"] = discrete_loss.detach()
                loss = discrete_loss
        else:
            loss_key = f"{dataset_tag}_text_loss" if dataset_tag is not None else "text_loss"
            zero_ref = logits if logits is not None else und_hidden_states
            zero_loss = zero_ref.sum() * 0.0
            if use_global_discrete_loss_average:
                num_valid_tokens = torch.tensor(0.0, device=zero_ref.device)
                losses["discrete_loss_sum"] = zero_loss
                losses["discrete_loss_count"] = num_valid_tokens.to(torch.float64)
                loss_value = zero_loss
            else:
                loss_value = zero_loss
                loss = zero_loss
            losses[loss_key] = loss_value.detach()

        # if pred is a 4-D tensor (B, C, H, W), raw_loss is a tensor (B)
        # if pred is a list of 4-D tensors [B x (N, C, H, W)], raw_loss is a tensor (B), N is already averaged
        # if pred is a list of list of 4-D tensors [B x (N_i x [1, C, H_ij, W_ij])], raw_loss is a tensor (B), N_i is already averaged
        def _get_ragged_local_sum_and_count(
                raw_loss: torch.Tensor,
                pred,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if isinstance(pred, torch.Tensor):
                local_count = torch.tensor(float(pred.size(0)), device=raw_loss.device, dtype=torch.float64)
                return raw_loss.sum(), local_count
            num_per_batch_item = [len(item) for item in pred]
            num_per_item_tensor = torch.tensor(num_per_batch_item, device=raw_loss.device, dtype=raw_loss.dtype)
            local_loss_sum = (raw_loss * num_per_item_tensor).sum()
            local_count = torch.tensor(float(sum(num_per_batch_item)), device=raw_loss.device, dtype=torch.float64)
            return local_loss_sum, local_count

        use_global_diffusion_loss_average = getattr(
            self.args, "use_global_diffusion_loss_average", False
        )

        def _accumulate_diffusion_loss(
                pred,
                diffusion_loss_fn,
                loss_key: str,
                loss_weight: float,
                sum_key: str,
                count_key: str,
                loss: torch.Tensor,
        ) -> torch.Tensor:
            if pred is None:
                return loss

            raw_diff_loss = diffusion_loss_fn(model_output=pred)["loss"]
            if use_global_diffusion_loss_average:
                if loss_weight > 0:
                    diff_loss_sum, diff_loss_count = _get_ragged_local_sum_and_count(raw_diff_loss, pred)
                    losses[sum_key] = diff_loss_sum
                    losses[count_key] = diff_loss_count
                    losses[loss_key] = raw_diff_loss.mean().detach()
                    # The TRUE final loss will be calculated in loss_closure after all_reduce for global average.
                    # So `loss` is not incremented here.
                else:
                    # For lm and some other tasks, diffusion loss weight is set to 0.
                    # To keep consistent computation graph across ranks, we still add diffusion loss here.
                    diff_loss = raw_diff_loss.mean()
                    loss = loss + loss_weight * diff_loss
            else:
                diff_loss = raw_diff_loss.mean()
                if loss_weight > 0:
                    losses[loss_key] = diff_loss.detach()
                loss = loss + loss_weight * diff_loss
            return loss

        loss = _accumulate_diffusion_loss(
            pred=diff_pred,
            diffusion_loss_fn=diffusion_loss_fn,
            loss_key=f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss",
            loss_weight=image_loss_weight,
            sum_key="diff_loss_sum",
            count_key="diff_loss_count",
            loss=loss,
        )

        # -- moe losses
        if self._config.moe_aux_loss and self.moe_aux_loss_coeff > 0:
            moe_aux_losses = [
                block.mlp.get_balance_loss()
                for block in self.model["layers"]   # noqa
                if isinstance(block.mlp, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE, Qwen3VLSparesMoeBlock))
            ]
            assert len(moe_aux_losses) > 0, "No MoE losses found across the model layers."
            moe_aux_loss = sum(moe_aux_losses)
            loss = loss + moe_aux_loss * self.moe_aux_loss_coeff
            losses["moe_loss"] = moe_aux_loss.detach() / len(moe_aux_losses)     # noqa

            capacity_rates = [
                block.mlp.get_capacity_rate()
                for block in self.model["layers"]   # noqa
                if isinstance(block.mlp, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE, Qwen3VLSparesMoeBlock))
            ]
            assert len(capacity_rates) > 0, "No capacity losses found across the model layers."
            capacity_rates = sum(capacity_rates) / len(capacity_rates)
            losses["capacity_rate"] = capacity_rates

            if self._config.use_mot and image_loss_weight > 0:
                moe_aux_losses = [
                    block.mlp_mot_gen.get_balance_loss()
                    for block in self.model["layers"]   # noqa
                    if isinstance(block.mlp_mot_gen, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE, Qwen3VLSparesMoeBlock))
                ]
                assert len(moe_aux_losses) > 0, "No MoE losses found across the model layers."
                moe_aux_loss = sum(moe_aux_losses)
                loss = loss + moe_aux_loss * self.moe_aux_loss_coeff
                losses["moe_loss_mot_gen"] = moe_aux_loss.detach() / len(moe_aux_losses)     # noqa

                capacity_rates = [
                    block.mlp_mot_gen.get_capacity_rate()
                    for block in self.model["layers"]   # noqa
                    if isinstance(block.mlp_mot_gen, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE, Qwen3VLSparesMoeBlock))
                ]
                assert len(capacity_rates) > 0, "No capacity losses found across the model layers."
                capacity_rates = sum(capacity_rates) / len(capacity_rates)
                losses["capacity_rate_mot_gen"] = capacity_rates

        # -- total loss (for backward)
        losses["loss"] = loss

        if not return_dict:
            return loss, logits, past_key_values, diff_pred
        return HunyuanMultimodalOutput(
            losses=losses,
            logits=logits,
            past_key_values=past_key_values,
            diffusion_prediction=diff_pred,
        )


class HunyuanMultimodal(HunyuanMultimodalBase):
    def __init__(
            self,
            args: Namespace,
            config: HunyuanMultimodalConfig,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            initialize_weights: bool = True,
            gen_config: Optional[HunyuanMultimodalConfig] = None,
    ):
        super().__init__()
        self.__post_init__(config, dtype, device, args, initialize_weights, gen_config)
