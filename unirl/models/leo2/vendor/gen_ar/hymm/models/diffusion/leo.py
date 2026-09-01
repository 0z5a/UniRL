import math
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional, Any, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.nn.attention.flex_attention import BlockMask
from transformers.utils.generic import ModelOutput

from hymm.core.global_vars import get_parallel_state, get_args
from hy_parallelism.training.checkpointing import flush_pending_offloads
from hy_parallelism.context_parallel.core import (
    get_cp_info as get_cp_info_core,
    maybe_scatter_seq,
    maybe_gather_seq,
    maybe_to_split_head,
    maybe_to_split_seq,
    maybe_to_cp_region_num_head,
    maybe_to_normal_region_num_head,
    register_cp_info,
)

from .leo_cache import LeoFirstBlockCacheConfig, LeoFirstBlockCacheController
from .leo_config import LeoConfig
from ..basic.attention import SelfAttention
from ..basic.embed_layers import TimestepEmbedder, TextProjection, AudioProjection
from ..basic.modulate import ModulateDiT, modulate, apply_gate
from ..basic.moe_layers import (
    HunyuanMLP, HunyuanMoE, DeepSeekMoE, FlashInferMoE, Qwen3VLSparesMoeBlock, ExpertParallelMoE
)
from ..basic.patch_embed_layers import project_in_layer, project_out_layer
from ..basic.pos_emb_layers import apply_rope, apply_rope_qk
from ..basic.rope_cache import CachedRoPE
from ..basic.token_refiner import SingleTokenRefiner
from ..multimodal.hunyuan_multimodal_state import HunyuanMultimodalState

compiled_flex_attention = None
# Type aliases
BatchRaggedMedia = Union[torch.Tensor, list[Union[torch.Tensor, list[torch.Tensor]]]]
BatchRaggedTensor = Union[torch.Tensor, list[torch.Tensor]]

LEO_MEDIA_CP_INFO = "leo_media"
LEO_TEXT_CP_INFO = "leo_text"
LEO_AUDIO_CP_INFO = "leo_audio"


def maybe_wait_tensor(x):
    return x.wait() if hasattr(x, "wait") else x


def scatter_seq_and_register_cp_info(x: torch.Tensor, name: str) -> torch.Tensor:
    x, cp_info = maybe_scatter_seq(x, return_split_meta=True)
    register_cp_info(name, cp_info)
    return x

def get_cp_info(name: str):
    try:
        args = get_args()
    except:
        return get_cp_info_core(name)
    if args.enable_cp_info:
        return get_cp_info_core(name)
    return None

def _repeat_interleave(
    inputs,
    repeats,
    dim=None,
    *,
    output_size=None,
):
    if get_parallel_state().cp_size > 1 and dim == 1 and inputs.ndim == 3 and repeats.ndim == 1:
        sample_indices = torch.repeat_interleave(
            torch.arange(inputs.size(dim), device=inputs.device),
            repeats.to(device=inputs.device),
            output_size=output_size,
        )
        sample_indices = maybe_scatter_seq(sample_indices.view(1, -1, 1)).flatten()
        return inputs.float().index_select(dim, sample_indices)

    ret = torch.repeat_interleave(inputs.float(), repeats, dim=dim, output_size=output_size)
    if get_parallel_state().cp_size > 1:
        ret = maybe_scatter_seq(ret)
    return ret


def _apply_cond_zero_timestep_mod(gen_mod, gen_mod_t0, gen_cond_token_mask, gen_lengths_for_mod):
    """Returns a per-token gen modulation [*, total_gen, 6H] for r2v training and inference
    """
    if gen_lengths_for_mod is not None: # r2v training path(seq packing)
        gen_mod = _repeat_interleave(gen_mod, gen_lengths_for_mod, dim=1)        # [1, total_gen, 6H]
    else: # r2v inference path(single timestep per sample)
        if gen_mod.ndim == 2:
            gen_mod = gen_mod.unsqueeze(1)                                       # [bsz, 1, 6H]
        gen_mod = gen_mod.expand(-1, gen_cond_token_mask.size(1), -1)            # [bsz, total_gen, 6H]
    if gen_mod_t0.ndim == 2: # r2v inference path(single timestep per sample)
        gen_mod_t0 = gen_mod_t0.unsqueeze(1)                                     # [1, 1, 6H]
    return torch.where(gen_cond_token_mask.unsqueeze(-1), gen_mod_t0.to(gen_mod.dtype), gen_mod)


class LeoSelfAttention(SelfAttention):
    """
    Self-attention module for Leo model.
    """
    @staticmethod
    def all2all_qkv(qkv, head_size, total_qkv, cp_info=None):
        return einops.rearrange(
            maybe_to_split_head(
                einops.rearrange(
                    qkv, 
                    'b s (n_kv_head total_qkv head_size) -> (b total_qkv) s n_kv_head head_size', 
                    head_size=head_size, total_qkv=total_qkv
                ),
                cp_info=cp_info,
            ),
            '(b total_qkv) s n_kv_head head_size -> b s n_kv_head total_qkv head_size', 
            head_size=head_size, total_qkv=total_qkv
        )


class LeoDualAttention(LeoSelfAttention):
    """
    Dual attention module for Leo model.
    """
    def __init__(
            self,
            config: LeoConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            txt_config: Optional[LeoConfig] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Create the main branch attention modules
        super().__init__(config, layer_idx, dtype, device)
        self._txt_config = txt_config = txt_config or config
        self.layer_idx = layer_idx

        # Create the auxiliary branch attention modules for text hidden states
        if not txt_config.split_qkv:
            self.qkv_proj_txt = nn.Linear(
                txt_config.hidden_size,
                (txt_config.num_attention_heads + 2 * txt_config.num_kv_heads) * txt_config.attention_head_size,
                bias=txt_config.attention_bias, **factory_kwargs
            )
        else:
            self.q_proj_txt = nn.Linear(
                txt_config.hidden_size,
                txt_config.num_attention_heads * txt_config.attention_head_size,
                bias=txt_config.attention_bias, **factory_kwargs
            )
            self.k_proj_txt = nn.Linear(
                txt_config.hidden_size,
                txt_config.num_kv_heads * txt_config.attention_head_size,
                bias=txt_config.attention_bias, **factory_kwargs
            )
            self.v_proj_txt = nn.Linear(
                txt_config.hidden_size,
                txt_config.num_kv_heads * txt_config.attention_head_size,
                bias=txt_config.attention_bias, **factory_kwargs
            )

        if txt_config.skip_txt_after_last_attn and layer_idx == txt_config.num_layers - 1:
            self.o_proj_txt = None
        else:
            self.o_proj_txt = nn.Linear(
                txt_config.attention_head_size * txt_config.num_attention_heads,
                txt_config.hidden_size,
                bias=txt_config.attention_bias, **factory_kwargs
            )

        if txt_config.use_qk_norm:
            self.query_layernorm_txt = txt_config.qk_norm_class(
                txt_config.attention_head_size, **txt_config.get_norm_kwargs(txt_config.qk_norm_type), **factory_kwargs)
            self.key_layernorm_txt = txt_config.qk_norm_class(
                txt_config.attention_head_size, **txt_config.get_norm_kwargs(txt_config.qk_norm_type), **factory_kwargs)

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor] = None,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, txt_hidden_states = hidden_states
        bsz, seqlen, _ = hidden_states.size()
        txt_seqlen = txt_hidden_states.size(1)

        # For attention heads, _config and _txt_config should have the same num_attention_heads, num_kv_heads,
        # and attention_head_size.
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        assert n_q_head % n_kv_head == 0, f'{n_q_head=} {n_kv_head=}'
        media_pre_qk_norm_done = False
        if get_parallel_state().cp_size > 1:
            media_cp_info = get_cp_info(LEO_MEDIA_CP_INFO)
            text_cp_info = get_cp_info(LEO_TEXT_CP_INFO)
        else:
            media_cp_info = text_cp_info = None

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        if not self._config.split_qkv:
            qkv = self.qkv_proj(hidden_states)
            txt_qkv = self.qkv_proj_txt(txt_hidden_states)

            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value

            assert n_kv_head * total_qkv * head_size == self.qkv_proj.out_features, f'{n_kv_head * total_qkv * head_size=} {self.qkv_proj.out_features=}'

            if get_parallel_state().cp_size > 1:
                qkv = self.all2all_qkv(qkv, head_size=head_size, total_qkv=total_qkv, cp_info=media_cp_info)
                txt_qkv = self.all2all_qkv(txt_qkv, head_size=head_size, total_qkv=total_qkv, cp_info=text_cp_info)

                seqlen, txt_seqlen = map(lambda x: x.shape[1], [qkv, txt_qkv])
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])


            # (bsz, num_kv_heads, total_qkv, T, head_size)
            qkv = qkv.view(bsz, seqlen, n_kv_head, total_qkv, head_size).permute(0, 2, 3, 1, 4)
            txt_qkv = txt_qkv.view(bsz, txt_seqlen, n_kv_head, total_qkv, head_size).permute(0, 2, 3, 1, 4)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            txt_q, txt_k, txt_v = txt_qkv.split((q_per_kv, 1, 1), dim=2)
        else:
            media_pre_qk_norm_done = False

            if get_parallel_state().cp_size > 1:
                media_q_norm = self.query_layernorm if (self._config.use_qk_norm and self._config.pre_qk_norm) else None
                media_k_norm = self.key_layernorm if (self._config.use_qk_norm and self._config.pre_qk_norm) else None
                q = self.q_proj(hidden_states).reshape(bsz, seqlen, n_q_head, head_size)
                if media_q_norm is not None:
                    q = media_q_norm(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
                q_handle = maybe_to_split_head(q, cp_info=media_cp_info, async_op=True)

                k = self.k_proj(hidden_states).reshape(bsz, seqlen, n_kv_head, head_size)
                if media_k_norm is not None:
                    k = media_k_norm(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
                k_handle = maybe_to_split_head(k, cp_info=media_cp_info, async_op=True)

                v = self.v_proj(hidden_states).reshape(bsz, seqlen, n_kv_head, head_size)
                v_handle = maybe_to_split_head(v, cp_info=media_cp_info, async_op=True)

                txt_q = self.q_proj_txt(txt_hidden_states)
                txt_k = self.k_proj_txt(txt_hidden_states)
                txt_v = self.v_proj_txt(txt_hidden_states)

                q = maybe_wait_tensor(q_handle)
                k = maybe_wait_tensor(k_handle)
                v = maybe_wait_tensor(v_handle)

                txt_q, txt_k, txt_v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, txt_seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=text_cp_info,
                    ),
                    [txt_q, txt_k, txt_v],
                )

                media_pre_qk_norm_done = self._config.use_qk_norm and self._config.pre_qk_norm
                seqlen, txt_seqlen = map(lambda x: x.shape[1], [q, txt_q])
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])
            else:
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                txt_q = self.q_proj_txt(txt_hidden_states)
                txt_k = self.k_proj_txt(txt_hidden_states)
                txt_v = self.v_proj_txt(txt_hidden_states)
            
            q = q.view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
            k = k.view(bsz, seqlen, n_kv_head, 1, head_size)
            v = v.view(bsz, seqlen, n_kv_head, 1, head_size)
            txt_q = txt_q.view(bsz, txt_seqlen, n_kv_head, q_per_kv, head_size)
            txt_k = txt_k.view(bsz, txt_seqlen, n_kv_head, 1, head_size)
            txt_v = txt_v.view(bsz, txt_seqlen, n_kv_head, 1, head_size)

            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [q, k, v])
            txt_q, txt_k, txt_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [txt_q, txt_k, txt_v])

        q = q.reshape(bsz, n_q_head, seqlen, head_size)
        k = k.reshape(bsz, n_kv_head, seqlen, head_size)
        v = v.reshape(bsz, n_kv_head, seqlen, head_size)
        txt_q = txt_q.reshape(bsz, n_q_head, txt_seqlen, head_size)
        txt_k = txt_k.reshape(bsz, n_kv_head, txt_seqlen, head_size)
        txt_v = txt_v.reshape(bsz, n_kv_head, txt_seqlen, head_size)

        if self._config.use_qk_norm and self._config.pre_qk_norm:
            if not (self._config.split_qkv and get_parallel_state().cp_size > 1 and media_pre_qk_norm_done):
                q = self.query_layernorm(q)
                k = self.key_layernorm(k)
            txt_q = self.query_layernorm_txt(txt_q)
            txt_k = self.key_layernorm_txt(txt_k)

        use_packing = token_indices is not None and token_indices[0] is not None
        # For non-packing mode, the image and text order can be controled by `config.text_order`.
        text_first = not use_packing and hasattr(self._config, "text_order") and self._config.text_order == "text_first"

        raw_cos, raw_sin = rotary_position_embeddings
        apply_rope_kwargs = dict(
            apply_rope_in_fp32=self._config.apply_rope_in_fp32,
            interleave=self._config.rope_interleave,
            cast_output_to_input_dtype=self._config.pre_qk_norm,
        )

        if not use_packing:
            if text_first:
                txt_cos, cos = raw_cos.split((txt_seqlen, seqlen), dim=-2)
                txt_sin, sin = raw_sin.split((txt_seqlen, seqlen), dim=-2)
            else:
                cos, txt_cos = raw_cos.split((seqlen, txt_seqlen), dim=-2)
                sin, txt_sin = raw_sin.split((seqlen, txt_seqlen), dim=-2)
            q, k = apply_rope_qk(q, k, cos, sin, **apply_rope_kwargs)
            txt_q, txt_k = apply_rope_qk(txt_q, txt_k, txt_cos, txt_sin, **apply_rope_kwargs)

        else:
            # Merge the three branches together to apply RoPE and self attention

            def _merge_branches(
                    gen_src, und_src, gen_token_indices, und_token_indices, n_head
            ):
                bsz = gen_src.size(0)
                merged = torch.zeros(
                    bsz, n_head, seqlen + txt_seqlen, head_size,
                    dtype=gen_src.dtype, device=gen_src.device,
                )
                merged.scatter_(dim=2, index=gen_token_indices, src=gen_src)
                merged.scatter_(dim=2, index=und_token_indices, src=und_src)
                return merged

            gen_indices, und_indices = token_indices

            gen_indices_q = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            und_indices_q = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            gen_indices_kv = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            und_indices_kv = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))

            qq = _merge_branches(q, txt_q, gen_indices_q, und_indices_q, q.size(1))
            kk = _merge_branches(k, txt_k, gen_indices_kv, und_indices_kv, k.size(1))
            vv = _merge_branches(v, txt_v, gen_indices_kv, und_indices_kv, v.size(1))

            qq, kk = apply_rope_qk(qq, kk, raw_cos, raw_sin, **apply_rope_kwargs)

        # Some others use qk norm after rotary pos emb
        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            assert not use_packing, "qk norm after RoPE is not implemented when sequence pack is used."
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
            txt_q = self.query_layernorm_txt(txt_q)
            txt_k = self.key_layernorm_txt(txt_k)

        if not use_packing:
            # Merge into a single sequence
            if text_first:
                qq = torch.cat([txt_q, q], dim=-2)
                kk = torch.cat([txt_k, k], dim=-2)
                vv = torch.cat([txt_v, v], dim=-2)
            else:
                qq = torch.cat([q, txt_q], dim=-2)
                kk = torch.cat([k, txt_k], dim=-2)
                vv = torch.cat([v, txt_v], dim=-2)

        qq = qq.to(vv.dtype)
        kk = kk.to(vv.dtype)

        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = kk.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if q_per_kv != 1:
            kk = kk.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            vv = vv.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        yy = self.scaled_dot_product_attention(qq, kk, vv, attention_mask)

        yy = yy.reshape(bsz, -1, head_size * n_q_head)  # re-assemble all head outputs side by side

        # Split into two branches
        if not use_packing:
            if text_first:
                txt_y, y = yy.split((txt_seqlen, seqlen), dim=1)
            else:
                y, txt_y = yy.split((seqlen, txt_seqlen), dim=1)
        else:
            y = yy.gather(dim=1, index=gen_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))
            txt_y = yy.gather(dim=1, index=und_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))

        if get_parallel_state().cp_size > 1:
            y = maybe_to_split_seq(y.reshape(bsz, seqlen, n_q_head, head_size), cp_info=media_cp_info)
            txt_y = maybe_to_split_seq(txt_y.reshape(bsz, txt_seqlen, n_q_head, head_size), cp_info=text_cp_info)
            n_q_head = maybe_to_normal_region_num_head(n_q_head)
            seqlen, txt_seqlen = map(lambda x: x.shape[1], [y, txt_y])

            y = y.reshape(bsz, seqlen, head_size * n_q_head)  # re-assemble all head outputs side by side
            txt_y = txt_y.reshape(bsz, txt_seqlen, head_size * n_q_head)

        # output projection
        hidden_states = self.o_proj(y)
        if self.o_proj_txt is not None:
            txt_hidden_states = self.o_proj_txt(txt_y)

        return hidden_states, txt_hidden_states


class LeoTripleAttention(LeoDualAttention):
    """
    Triple attention module for Leo model.
    """
    def __init__(
            self,
            config: LeoConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            txt_config: Optional[LeoConfig] = None,
            audio_config: Optional[LeoConfig] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Create the main branch attention modules
        super().__init__(config, layer_idx, dtype, device, txt_config)
        self._audio_config = audio_config
        self.layer_idx = layer_idx

        # Create the auxiliary branch attention modules for text hidden states
        if not audio_config.split_qkv:
            self.qkv_proj_audio = nn.Linear(
                audio_config.hidden_size,
                (audio_config.num_attention_heads + 2 * audio_config.num_kv_heads) * audio_config.attention_head_size,
                bias=audio_config.attention_bias, **factory_kwargs
            )
        else:
            self.q_proj_audio = nn.Linear(
                audio_config.hidden_size,
                audio_config.num_attention_heads * audio_config.attention_head_size,
                bias=audio_config.attention_bias, **factory_kwargs
            )
            self.k_proj_audio = nn.Linear(
                audio_config.hidden_size,
                audio_config.num_kv_heads * audio_config.attention_head_size,
                bias=audio_config.attention_bias, **factory_kwargs
            )
            self.v_proj_audio = nn.Linear(
                audio_config.hidden_size,
                audio_config.num_kv_heads * audio_config.attention_head_size,
                bias=audio_config.attention_bias, **factory_kwargs
            )

        self.o_proj_audio = nn.Linear(
            audio_config.attention_head_size * audio_config.num_attention_heads,
            audio_config.hidden_size,
            bias=audio_config.attention_bias, **factory_kwargs
        )

        if audio_config.use_qk_norm:
            self.query_layernorm_audio = audio_config.qk_norm_class(
                audio_config.attention_head_size, **audio_config.get_norm_kwargs(audio_config.qk_norm_type), **factory_kwargs)
            self.key_layernorm_audio = audio_config.qk_norm_class(
                audio_config.attention_head_size, **audio_config.get_norm_kwargs(audio_config.qk_norm_type), **factory_kwargs)

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_packing = token_indices is not None and token_indices[0] is not None
        # TODO: support non-packing, non-split-qkv, and non-pre-qk-norm cases
        if get_parallel_state().cp_size > 1 and use_packing and self._config.split_qkv and self._config.pre_qk_norm:
            return self.forward_async_cp(
                hidden_states,
                attention_mask=attention_mask,
                rotary_position_embeddings=rotary_position_embeddings,
                token_indices=token_indices,
                **kwargs,
            )
        return self.forward_naive(
            hidden_states,
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            token_indices=token_indices,
            **kwargs,
        )

    def forward_naive(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, audio_hidden_states, txt_hidden_states = hidden_states
        has_audio = audio_hidden_states is not None
        bsz, seqlen, _ = hidden_states.size()
        txt_seqlen = txt_hidden_states.size(1)
        if has_audio:
            audio_seqlen = audio_hidden_states.size(1)
        else:
            audio_seqlen = 0

        # For attention heads, _config and _txt_config should have the same num_attention_heads, num_kv_heads,
        # and attention_head_size.
        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        if get_parallel_state().cp_size > 1:
            media_cp_info = get_cp_info(LEO_MEDIA_CP_INFO)
            text_cp_info = get_cp_info(LEO_TEXT_CP_INFO)
            audio_cp_info = get_cp_info(LEO_AUDIO_CP_INFO) if has_audio else None
        else:
            media_cp_info = text_cp_info = audio_cp_info = None

        if not self._config.split_qkv:
            qkv = self.qkv_proj(hidden_states)
            txt_qkv = self.qkv_proj_txt(txt_hidden_states)
            if has_audio:
                audio_qkv = self.qkv_proj_audio(audio_hidden_states)
            else:
                audio_qkv = None

            total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
            if get_parallel_state().cp_size > 1:
                qkv = self.all2all_qkv(qkv, head_size=head_size, total_qkv=total_qkv, cp_info=media_cp_info)
                txt_qkv = self.all2all_qkv(txt_qkv, head_size=head_size, total_qkv=total_qkv, cp_info=text_cp_info)
                if has_audio:
                    audio_qkv = self.all2all_qkv(
                        audio_qkv,
                        head_size=head_size,
                        total_qkv=total_qkv,
                        cp_info=audio_cp_info,
                    )

                if has_audio:
                    seqlen, txt_seqlen, audio_seqlen = map(lambda x: x.shape[1], [qkv, txt_qkv, audio_qkv])
                else:
                    seqlen, txt_seqlen = map(lambda x: x.shape[1], [qkv, txt_qkv])
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])

            # (bsz, num_kv_heads, total_qkv, T, head_size)
            qkv = qkv.view(bsz, seqlen, n_kv_head, total_qkv, head_size).permute(0, 2, 3, 1, 4)
            txt_qkv = txt_qkv.view(bsz, txt_seqlen, n_kv_head, total_qkv, head_size).permute(0, 2, 3, 1, 4)
            if has_audio:
                audio_qkv = audio_qkv.view(bsz, audio_seqlen, n_kv_head, total_qkv, head_size).permute(0, 2, 3, 1, 4)

            # split batched computation into three
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            txt_q, txt_k, txt_v = txt_qkv.split((q_per_kv, 1, 1), dim=2)
            if has_audio:
                audio_q, audio_k, audio_v = audio_qkv.split((q_per_kv, 1, 1), dim=2)
            else:
                audio_q = audio_k = audio_v = None
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            txt_q = self.q_proj_txt(txt_hidden_states)
            txt_k = self.k_proj_txt(txt_hidden_states)
            txt_v = self.v_proj_txt(txt_hidden_states)
            if has_audio:
                audio_q = self.q_proj_audio(audio_hidden_states)
                audio_k = self.k_proj_audio(audio_hidden_states)
                audio_v = self.v_proj_audio(audio_hidden_states)
            else:
                audio_q = audio_k = audio_v = None

            if get_parallel_state().cp_size > 1:
                q, k, v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=media_cp_info,
                    ),
                    [q, k, v],
                )
                txt_q, txt_k, txt_v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, txt_seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=text_cp_info,
                    ),
                    [txt_q, txt_k, txt_v],
                )
                if has_audio:
                    audio_q, audio_k, audio_v = map(
                        lambda x: maybe_to_split_head(
                            x.reshape(bsz, audio_seqlen, x.shape[-1] // head_size, head_size),
                            cp_info=audio_cp_info,
                        ),
                        [audio_q, audio_k, audio_v]
                    )

                if has_audio:
                    seqlen, txt_seqlen, audio_seqlen = map(lambda x: x.shape[1], [q, txt_q, audio_q])
                else:
                    seqlen, txt_seqlen = map(lambda x: x.shape[1], [q, txt_q])
                n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])

            q = q.view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
            k = k.view(bsz, seqlen, n_kv_head, 1, head_size)
            v = v.view(bsz, seqlen, n_kv_head, 1, head_size)
            txt_q = txt_q.view(bsz, txt_seqlen, n_kv_head, q_per_kv, head_size)
            txt_k = txt_k.view(bsz, txt_seqlen, n_kv_head, 1, head_size)
            txt_v = txt_v.view(bsz, txt_seqlen, n_kv_head, 1, head_size)
            if has_audio:
                audio_q = audio_q.view(bsz, audio_seqlen, n_kv_head, q_per_kv, head_size)
                audio_k = audio_k.view(bsz, audio_seqlen, n_kv_head, 1, head_size)
                audio_v = audio_v.view(bsz, audio_seqlen, n_kv_head, 1, head_size)

            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [q, k, v])
            txt_q, txt_k, txt_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [txt_q, txt_k, txt_v])
            if has_audio:
                audio_q, audio_k, audio_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [audio_q, audio_k, audio_v])

        q = q.reshape(bsz, n_q_head, seqlen, head_size)
        k = k.reshape(bsz, n_kv_head, seqlen, head_size)
        v = v.reshape(bsz, n_kv_head, seqlen, head_size)
        txt_q = txt_q.reshape(bsz, n_q_head, txt_seqlen, head_size)
        txt_k = txt_k.reshape(bsz, n_kv_head, txt_seqlen, head_size)
        txt_v = txt_v.reshape(bsz, n_kv_head, txt_seqlen, head_size)
        if has_audio:
            audio_q = audio_q.reshape(bsz, n_q_head, audio_seqlen, head_size)
            audio_k = audio_k.reshape(bsz, n_kv_head, audio_seqlen, head_size)
            audio_v = audio_v.reshape(bsz, n_kv_head, audio_seqlen, head_size)

        if self._config.use_qk_norm and self._config.pre_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
            txt_q = self.query_layernorm_txt(txt_q)
            txt_k = self.key_layernorm_txt(txt_k)
            if has_audio:
                audio_q = self.query_layernorm_audio(audio_q)
                audio_k = self.key_layernorm_audio(audio_k)

        use_packing = token_indices is not None and token_indices[0] is not None

        raw_cos, raw_sin = rotary_position_embeddings
        apply_rope_kwargs = dict(
            apply_rope_in_fp32=self._config.apply_rope_in_fp32,
            interleave=self._config.rope_interleave,
            cast_output_to_input_dtype=self._config.pre_qk_norm,
        )
        if not use_packing:
            if audio_seqlen == 1:
                # Dummy audio token, move it to the end to avoid affecting the original RoPE order of text and video tokens
                cos, txt_cos, audio_cos = raw_cos.split((seqlen, txt_seqlen, audio_seqlen), dim=-2)
                sin, txt_sin, audio_sin = raw_sin.split((seqlen, txt_seqlen, audio_seqlen), dim=-2)
            elif seqlen == 1:
                # Dummy video token, move it to the front to avoid affecting the original RoPE order of text and audio tokens
                audio_cos, txt_cos, cos = raw_cos.split((audio_seqlen, txt_seqlen, seqlen), dim=-2)
                audio_sin, txt_sin, sin = raw_sin.split((audio_seqlen, txt_seqlen, seqlen), dim=-2)
            else:
                cos, audio_cos, txt_cos = raw_cos.split((seqlen, audio_seqlen, txt_seqlen), dim=-2)
                sin, audio_sin, txt_sin = raw_sin.split((seqlen, audio_seqlen, txt_seqlen), dim=-2)
            q, k = apply_rope_qk(q, k, cos, sin, **apply_rope_kwargs)
            txt_q, txt_k = apply_rope_qk(txt_q, txt_k, txt_cos, txt_sin, **apply_rope_kwargs)
            if has_audio:
                audio_q, audio_k = apply_rope_qk(audio_q, audio_k, audio_cos, audio_sin, **apply_rope_kwargs)

            gen_indices = audio_indices = und_indices = None
        else:
            # Merge the three branches together to apply RoPE and self attention

            def _merge_branches(
                    gen_src, audio_src, und_src, gen_token_indices, audio_token_indices, und_token_indices, n_head
            ):
                bsz = gen_src.size(0)
                merged = torch.zeros(
                    bsz, n_head, seqlen + txt_seqlen + audio_seqlen, head_size,
                    dtype=gen_src.dtype, device=gen_src.device,
                )
                merged.scatter_(dim=2, index=gen_token_indices, src=gen_src)
                if has_audio:
                    merged.scatter_(dim=2, index=audio_token_indices, src=audio_src)
                merged.scatter_(dim=2, index=und_token_indices, src=und_src)
                return merged

            gen_indices, audio_indices, und_indices = token_indices

            gen_indices_q = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            if has_audio:
                audio_indices_q = audio_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            else:
                audio_indices_q = None
            und_indices_q = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            gen_indices_kv = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            if has_audio:
                audio_indices_kv = audio_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            else:
                audio_indices_kv = None
            und_indices_kv = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))

            qq = _merge_branches(q, audio_q, txt_q, gen_indices_q, audio_indices_q, und_indices_q, q.size(1))
            kk = _merge_branches(k, audio_k, txt_k, gen_indices_kv, audio_indices_kv, und_indices_kv, k.size(1))
            vv = _merge_branches(v, audio_v, txt_v, gen_indices_kv, audio_indices_kv, und_indices_kv, v.size(1))

            qq, kk = apply_rope_qk(qq, kk, raw_cos, raw_sin, **apply_rope_kwargs)

        # Some others use qk norm after rotary pos emb
        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            assert not use_packing, "qk norm after RoPE is not implemented when sequence pack is used."
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)
            txt_q = self.query_layernorm_txt(txt_q)
            txt_k = self.key_layernorm_txt(txt_k)
            audio_q = self.query_layernorm_audio(audio_q)
            audio_k = self.key_layernorm_audio(audio_k)

        if not use_packing:
            # Merge into a single sequence
            if audio_seqlen == 0:
                qq = torch.cat([q, txt_q], dim=-2)
                kk = torch.cat([k, txt_k], dim=-2)
                vv = torch.cat([v, txt_v], dim=-2)
            elif audio_seqlen == 1:
                qq = torch.cat([q, txt_q, audio_q], dim=-2)
                kk = torch.cat([k, txt_k, audio_k], dim=-2)
                vv = torch.cat([v, txt_v, audio_v], dim=-2)
            elif seqlen == 0:
                qq = torch.cat([audio_q, txt_q], dim=-2)
                kk = torch.cat([audio_k, txt_k], dim=-2)
                vv = torch.cat([audio_v, txt_v], dim=-2)
            elif seqlen == 1:
                qq = torch.cat([audio_q, txt_q, q], dim=-2)
                kk = torch.cat([audio_k, txt_k, k], dim=-2)
                vv = torch.cat([audio_v, txt_v, v], dim=-2)
            else:
                qq = torch.cat([q, audio_q, txt_q], dim=-2)
                kk = torch.cat([k, audio_k, txt_k], dim=-2)
                vv = torch.cat([v, audio_v, txt_v], dim=-2)

        qq = qq.to(vv.dtype)
        kk = kk.to(vv.dtype)

        # If restore from cache, kv_seqlen >= seqlen
        kv_seqlen = kk.size(2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if q_per_kv != 1:
            kk = kk.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            vv = vv.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        yy = self.scaled_dot_product_attention(qq, kk, vv, attention_mask)

        yy = yy.reshape(bsz, -1, head_size * n_q_head)  # re-assemble all head outputs side by side

        # Split into three branches
        if not use_packing:
            if audio_seqlen == 1:
                y, txt_y, audio_y = yy.split((seqlen, txt_seqlen, audio_seqlen), dim=1)
            elif seqlen == 1:
                audio_y, txt_y, y = yy.split((audio_seqlen, txt_seqlen, seqlen), dim=1)
            else:
                y, audio_y, txt_y = yy.split((seqlen, audio_seqlen, txt_seqlen), dim=1)
        else:
            y = yy.gather(dim=1, index=gen_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))
            if has_audio:
                audio_y = yy.gather(dim=1, index=audio_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))
            else:
                audio_y = None
            txt_y = yy.gather(dim=1, index=und_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))

        if get_parallel_state().cp_size > 1:
            y = maybe_to_split_seq(y.reshape(bsz, seqlen, n_q_head, head_size), cp_info=media_cp_info)
            if has_audio:
                audio_y = maybe_to_split_seq(
                    audio_y.reshape(bsz, audio_seqlen, n_q_head, head_size),
                    cp_info=audio_cp_info,
                )
            txt_y = maybe_to_split_seq(txt_y.reshape(bsz, txt_seqlen, n_q_head, head_size), cp_info=text_cp_info)
            n_q_head = maybe_to_normal_region_num_head(n_q_head)
            if has_audio:
                seqlen, audio_seqlen, txt_seqlen = map(lambda x: x.shape[1], [y, audio_y, txt_y])
            else:
                seqlen, txt_seqlen = map(lambda x: x.shape[1], [y, txt_y])

            y = y.reshape(bsz, seqlen, head_size * n_q_head)
            txt_y = txt_y.reshape(bsz, txt_seqlen, head_size * n_q_head)
            if has_audio:
                audio_y = audio_y.reshape(bsz, audio_seqlen, head_size * n_q_head)

        # output projection
        hidden_states = self.o_proj(y)
        if self.o_proj_txt is not None:
            txt_hidden_states = self.o_proj_txt(txt_y)
        if has_audio:
            audio_hidden_states = self.o_proj_audio(audio_y)

        return hidden_states, audio_hidden_states, txt_hidden_states


    def forward_async_cp(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
            **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from hy_parallelism.tools.profiling import profile_range

        hidden_states, audio_hidden_states, txt_hidden_states = hidden_states

        def _merge_branches(
                gen_src, audio_src, und_src, gen_token_indices, audio_token_indices, und_token_indices, n_head
        ):
            merged = torch.zeros(
                bsz, n_head, seqlen + txt_seqlen + audio_seqlen, head_size,
                dtype=gen_src.dtype, device=gen_src.device,
            )
            merged.scatter_(dim=2, index=gen_token_indices, src=gen_src)
            if has_audio:
                merged.scatter_(dim=2, index=audio_token_indices, src=audio_src)
            merged.scatter_(dim=2, index=und_token_indices, src=und_src)
            return merged

        has_audio = audio_hidden_states is not None
        bsz, seqlen, _ = hidden_states.size()
        txt_seqlen = txt_hidden_states.size(1)
        audio_seqlen = audio_hidden_states.size(1) if has_audio else 0

        head_size = self._config.attention_head_size
        n_q_head = self._config.num_attention_heads
        n_kv_head = self._config.num_kv_heads
        q_per_kv = n_q_head // n_kv_head
        media_pre_qk_norm_done = False

        media_cp_info = get_cp_info(LEO_MEDIA_CP_INFO)
        text_cp_info = get_cp_info(LEO_TEXT_CP_INFO)
        audio_cp_info = get_cp_info(LEO_AUDIO_CP_INFO) if has_audio else None

        if not self._config.split_qkv:
            raise NotImplementedError("split_qkv is not supported in CP mode")

        with profile_range('txt_audio_proj'):
            txt_q = self.q_proj_txt(txt_hidden_states)
            txt_k = self.k_proj_txt(txt_hidden_states)
            txt_v = self.v_proj_txt(txt_hidden_states)
            txt_q, txt_k, txt_v = map(
                lambda x: maybe_to_split_head(
                    x.reshape(bsz, txt_seqlen, x.shape[-1] // head_size, head_size),
                    cp_info=text_cp_info,
                ),
                [txt_q, txt_k, txt_v],
            )
            if has_audio:
                audio_q = self.q_proj_audio(audio_hidden_states)
                audio_k = self.k_proj_audio(audio_hidden_states)
                audio_v = self.v_proj_audio(audio_hidden_states)
                audio_q, audio_k, audio_v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, audio_seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=audio_cp_info,
                    ),
                    [audio_q, audio_k, audio_v],
                )
            else:
                audio_q = audio_k = audio_v = None

        q = self.q_proj(hidden_states).reshape(bsz, seqlen, n_q_head, head_size)
        q = maybe_to_split_head(q, cp_info=media_cp_info, async_op=True)

        k = self.k_proj(hidden_states).reshape(bsz, seqlen, n_kv_head, head_size)
        k = maybe_to_split_head(k, cp_info=media_cp_info, async_op=True)

        v = self.v_proj(hidden_states).reshape(bsz, seqlen, n_kv_head, head_size)
        v = maybe_to_split_head(v, cp_info=media_cp_info, async_op=True)


        q = maybe_wait_tensor(q)
        with profile_range('txt_audio_op'):
            if has_audio:
                seqlen, txt_seqlen, audio_seqlen = map(lambda x: x.shape[1], [q, txt_q, audio_q])
            else:
                seqlen, txt_seqlen = map(lambda x: x.shape[1], [q, txt_q])
            n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])

            txt_q = txt_q.view(bsz, txt_seqlen, n_kv_head, q_per_kv, head_size)
            txt_k = txt_k.view(bsz, txt_seqlen, n_kv_head, 1, head_size)
            txt_v = txt_v.view(bsz, txt_seqlen, n_kv_head, 1, head_size)

            txt_q, txt_k, txt_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [txt_q, txt_k, txt_v])

            txt_q = txt_q.reshape(bsz, n_q_head, txt_seqlen, head_size)
            txt_k = txt_k.reshape(bsz, n_kv_head, txt_seqlen, head_size)
            txt_v = txt_v.reshape(bsz, n_kv_head, txt_seqlen, head_size)

            if has_audio:
                audio_q = audio_q.view(bsz, audio_seqlen, n_kv_head, q_per_kv, head_size)
                audio_k = audio_k.view(bsz, audio_seqlen, n_kv_head, 1, head_size)
                audio_v = audio_v.view(bsz, audio_seqlen, n_kv_head, 1, head_size)

                audio_q, audio_k, audio_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [audio_q, audio_k, audio_v])

                audio_q = audio_q.reshape(bsz, n_q_head, audio_seqlen, head_size)
                audio_k = audio_k.reshape(bsz, n_kv_head, audio_seqlen, head_size)
                audio_v = audio_v.reshape(bsz, n_kv_head, audio_seqlen, head_size)

        use_packing = token_indices is not None and token_indices[0] is not None
        assert use_packing, "sequence pack is not supported in async CP mode"

        raw_cos, raw_sin = rotary_position_embeddings
        apply_rope_kwargs = dict(
            apply_rope_in_fp32=self._config.apply_rope_in_fp32,
            interleave=self._config.rope_interleave,
            cast_output_to_input_dtype=self._config.pre_qk_norm,
        )

        if self._config.use_qk_norm and self._config.pre_qk_norm:
            with profile_range('txt_audio_norm'):
                txt_q = self.query_layernorm_txt(txt_q)
                txt_k = self.key_layernorm_txt(txt_k)
                if has_audio:
                    audio_q = self.query_layernorm_audio(audio_q)
                    audio_k = self.key_layernorm_audio(audio_k)

        with profile_range('q_op'):
            q = maybe_wait_tensor(q)
            q = q.view(bsz, seqlen, n_kv_head, q_per_kv, head_size)
            q = q.permute(0, 2, 3, 1, 4)
            q = q.reshape(bsz, n_q_head, seqlen, head_size)
            if self._config.use_qk_norm and self._config.pre_qk_norm:
                q = self.query_layernorm(q)

            gen_indices, audio_indices, und_indices = token_indices
            gen_indices_q = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            if has_audio:
                audio_indices_q = audio_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))
            else:
                audio_indices_q = None
            und_indices_q = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, q.size(1), -1, q.size(-1))

            qq = _merge_branches(q, audio_q, txt_q, gen_indices_q, audio_indices_q, und_indices_q, q.size(1))

        with profile_range('k_op'):
            k = maybe_wait_tensor(k)
            k = k.view(bsz, seqlen, n_kv_head, 1, head_size)
            k = k.permute(0, 2, 3, 1, 4)
            k = k.reshape(bsz, n_kv_head, seqlen, head_size)

            if self._config.use_qk_norm and self._config.pre_qk_norm:
                k = self.key_layernorm(k)

            gen_indices_kv = gen_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            if has_audio:
                audio_indices_kv = audio_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            else:
                audio_indices_kv = None
            und_indices_kv = und_indices.unsqueeze(-1).unsqueeze(1).expand(-1, k.size(1), -1, k.size(-1))
            kk = _merge_branches(k, audio_k, txt_k, gen_indices_kv, audio_indices_kv, und_indices_kv, k.size(1))

        with profile_range('rope_qk'):
            qq, kk = apply_rope_qk(qq, kk, raw_cos, raw_sin, **apply_rope_kwargs)

        with profile_range('v_op'):
            v = maybe_wait_tensor(v)
            v = v.view(bsz, seqlen, n_kv_head, 1, head_size)
            v = v.permute(0, 2, 3, 1, 4)
            v = v.reshape(bsz, n_kv_head, seqlen, head_size)
            if use_packing:
                vv = _merge_branches(v, audio_v, txt_v, gen_indices_kv, audio_indices_kv, und_indices_kv, v.size(1))

        qq = qq.to(vv.dtype)
        kk = kk.to(vv.dtype)

        kv_seqlen = kk.size(2)

        if q_per_kv != 1:
            kk = kk.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)
            vv = vv.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).reshape(bsz, -1, kv_seqlen, head_size)

        flush_pending_offloads()

        yy = self.scaled_dot_product_attention(qq, kk, vv, attention_mask)
        yy = yy.reshape(bsz, -1, head_size * n_q_head)

        y = yy.gather(dim=1, index=gen_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))
        if has_audio:
            audio_y = yy.gather(dim=1, index=audio_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))
        else:
            audio_y = None
        txt_y = yy.gather(dim=1, index=und_indices.unsqueeze(-1).expand(-1, -1, yy.size(-1)))

        y = maybe_to_split_seq(y.reshape(bsz, seqlen, n_q_head, head_size), cp_info=media_cp_info)
        if has_audio:
            audio_y = maybe_to_split_seq(
                audio_y.reshape(bsz, audio_seqlen, n_q_head, head_size),
                cp_info=audio_cp_info,
            )
        txt_y = maybe_to_split_seq(txt_y.reshape(bsz, txt_seqlen, n_q_head, head_size), cp_info=text_cp_info)
        n_q_head = maybe_to_normal_region_num_head(n_q_head)
        if has_audio:
            seqlen, audio_seqlen, txt_seqlen = map(lambda x: x.shape[1], [y, audio_y, txt_y])
        else:
            seqlen, txt_seqlen = map(lambda x: x.shape[1], [y, txt_y])

        y = y.reshape(bsz, seqlen, head_size * n_q_head)
        txt_y = txt_y.reshape(bsz, txt_seqlen, head_size * n_q_head)
        if has_audio:
            audio_y = audio_y.reshape(bsz, audio_seqlen, head_size * n_q_head)

        hidden_states = self.o_proj(y)
        if self.o_proj_txt is not None:
            txt_hidden_states = self.o_proj_txt(txt_y)
        if has_audio:
            audio_hidden_states = self.o_proj_audio(audio_y)

        return hidden_states, audio_hidden_states, txt_hidden_states


MOE_LAYER_IMPL = {
    "deepseek": DeepSeekMoE,
    "hunyuan": HunyuanMoE,
    "flashinfer": FlashInferMoE,
    "qwen3": Qwen3VLSparesMoeBlock,
    "ep_moe": ExpertParallelMoE,
}


class LeoLayer(nn.Module):
    def __init__(
            self,
            config: LeoConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.deterministic = False

        if config.use_modulation:
            self.mod_proj = ModulateDiT(
                config.modulate_hidden_size, factor=6, act_layer=config.act_class,
                output_hidden_size=config.hidden_size, **factory_kwargs,
            )
        self.self_attn = LeoSelfAttention(config, layer_idx, **factory_kwargs)
        self.input_layernorm = config.norm_class(
            config.hidden_size, **config.get_norm_kwargs(config.norm_type), **factory_kwargs
        )
        self.post_attention_layernorm = config.norm_class(
            config.hidden_size, **config.get_norm_kwargs(config.norm_type), **factory_kwargs
        )
        if layer_idx >= config.moe_layer_num_skipped and (
                (isinstance(config.num_experts, int) and config.num_experts > 1)
                or (isinstance(config.num_experts, list) and max(config.num_experts) > 1)
        ):
            assert config.moe_impl in MOE_LAYER_IMPL, f"moe_impl {config.moe_impl} not supported."
            self.mlp = MOE_LAYER_IMPL[config.moe_impl](config, layer_idx, **factory_kwargs)
        else:
            self.mlp = HunyuanMLP(config, layer_idx, **factory_kwargs)

    def enable_deterministic(self) -> None:
        self.deterministic = True
        self.self_attn.enable_deterministic()

    def disable_deterministic(self) -> None:
        self.deterministic = False
        self.self_attn.disable_deterministic()

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor],
            condition: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class LeoDualLayer(LeoLayer):
    def __init__(
            self,
            config: LeoConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            txt_config: Optional[LeoConfig] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Create the main branch attention modules
        super().__init__(config, layer_idx, dtype, device)
        self._config = config
        self._txt_config = txt_config = txt_config or config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Replace the self attention with dual attention
        self.self_attn = LeoDualAttention(config, layer_idx, txt_config=self._txt_config, **factory_kwargs)

        # Create the text branch
        if txt_config.use_modulation:
            self.mod_proj_txt = ModulateDiT(
                txt_config.modulate_hidden_size, factor=6, act_layer=txt_config.act_class,
                output_hidden_size=txt_config.hidden_size, **factory_kwargs
            )
        self.input_layernorm_txt = txt_config.norm_class(
            txt_config.hidden_size, **txt_config.get_norm_kwargs(txt_config.norm_type), **factory_kwargs
        )
        if txt_config.skip_txt_after_last_attn and layer_idx == txt_config.num_layers - 1:
            self.post_attention_layernorm_txt = None
            self.mlp_txt = None
        else:
            self.post_attention_layernorm_txt = txt_config.norm_class(
                txt_config.hidden_size, **txt_config.get_norm_kwargs(txt_config.norm_type), **factory_kwargs
            )
            if layer_idx >= txt_config.moe_layer_num_skipped and (
                    (isinstance(txt_config.num_experts, int) and txt_config.num_experts > 1)
                    or (isinstance(txt_config.num_experts, list) and max(txt_config.num_experts) > 1)
            ):
                assert txt_config.moe_impl in MOE_LAYER_IMPL, f"moe_impl {txt_config.moe_impl} not supported."
                self.mlp_txt = MOE_LAYER_IMPL[txt_config.moe_impl](txt_config, layer_idx, **factory_kwargs)
            else:
                self.mlp_txt = HunyuanMLP(txt_config, layer_idx, **factory_kwargs)

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor],
            timestep_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor] = None,
            token_lengths: tuple[torch.Tensor, torch.Tensor] = None,
            gen_cond_token_mask: Optional[torch.Tensor] = None,
            zero_timestep_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gen_lengths_for_mod = token_lengths[0] if token_lengths is not None else None
        # Calculate Modulation Coefficients
        if self._config.use_modulation:
            gen_mod = self.mod_proj(timestep_states)
            if gen_cond_token_mask is not None and zero_timestep_states is not None:
                gen_mod = _apply_cond_zero_timestep_mod(
                    gen_mod, self.mod_proj(zero_timestep_states), gen_cond_token_mask, gen_lengths_for_mod)
                gen_lengths_for_mod = None
            (
                attn_mod_shift, attn_mod_scale, attn_mod_gate,
                mlp_mod_shift, mlp_mod_scale, mlp_mod_gate,
            ) = gen_mod.chunk(6, dim=-1)
        else:
            attn_mod_shift = attn_mod_scale = attn_mod_gate = None
            mlp_mod_shift = mlp_mod_scale = mlp_mod_gate = None

        if self._txt_config.use_modulation:
            (
                txt_attn_mod_shift, txt_attn_mod_scale, txt_attn_mod_gate,
                txt_mlp_mod_shift, txt_mlp_mod_scale, txt_mlp_mod_gate,
            ) = self.mod_proj_txt(timestep_states).chunk(6, dim=-1)
        else:
            txt_attn_mod_shift = txt_attn_mod_scale = txt_attn_mod_gate = None
            txt_mlp_mod_shift = txt_mlp_mod_scale = txt_mlp_mod_gate = None

        # ============== Attention ==============
        # -- img --
        residual, txt_residual = hidden_states
        hidden_states, txt_hidden_states = hidden_states
        gen_token_indices, und_token_indices = token_indices
        gen_token_lengths, und_token_lengths = token_lengths
        # Input Normalization
        hidden_states = self.input_layernorm(hidden_states)
        # Modulate (Scale & Shift, No-Op if modulation is not enabled)
        if gen_token_indices is not None and attn_mod_shift is not None and gen_lengths_for_mod is not None:
            attn_mod_shift = _repeat_interleave(attn_mod_shift, gen_lengths_for_mod, dim=1)
            attn_mod_scale = _repeat_interleave(attn_mod_scale, gen_lengths_for_mod, dim=1)
        hidden_states = modulate(hidden_states, shift=attn_mod_shift, scale=attn_mod_scale)

        # -- txt --
        # Input Normalization
        txt_hidden_states = self.input_layernorm_txt(txt_hidden_states)
        # Modulate (Scale & Shift, No-Op if modulation is not enabled)
        if und_token_indices is not None and txt_attn_mod_shift is not None and und_token_lengths is not None:
            txt_attn_mod_shift = _repeat_interleave(txt_attn_mod_shift, und_token_lengths, dim=1)
            txt_attn_mod_scale = _repeat_interleave(txt_attn_mod_scale, und_token_lengths, dim=1)
        txt_hidden_states = modulate(txt_hidden_states, shift=txt_attn_mod_shift, scale=txt_attn_mod_scale)

        # Self Attention
        hidden_states, txt_hidden_states = self.self_attn(
            (hidden_states, txt_hidden_states),
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            token_indices=token_indices,
        )

        # -- img --
        # Modulate (Gate, No-Op if modulation is not enabled)
        if gen_token_indices is not None and attn_mod_gate is not None and gen_lengths_for_mod is not None:
            attn_mod_gate = _repeat_interleave(attn_mod_gate, gen_lengths_for_mod, dim=1)
        hidden_states = apply_gate(hidden_states, gate=attn_mod_gate)
        # Attention Residual
        hidden_states = residual + hidden_states
        # ============== MLP ==============
        residual = hidden_states
        # Post-Attention Normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Modulate (Scale & Shift, No-Op if modulation is not enabled)
        if gen_token_indices is not None and mlp_mod_shift is not None and gen_lengths_for_mod is not None:
            mlp_mod_shift = _repeat_interleave(mlp_mod_shift, gen_lengths_for_mod, dim=1)
            mlp_mod_scale = _repeat_interleave(mlp_mod_scale, gen_lengths_for_mod, dim=1)
        hidden_states = modulate(hidden_states, shift=mlp_mod_shift, scale=mlp_mod_scale)
        # MLP
        hidden_states = self.mlp(hidden_states)
        # Modulate (Gate, No-Op if modulation is not enabled)
        if gen_token_indices is not None and mlp_mod_gate is not None and gen_lengths_for_mod is not None:
            mlp_mod_gate = _repeat_interleave(mlp_mod_gate, gen_lengths_for_mod, dim=1)
        hidden_states = apply_gate(hidden_states, gate=mlp_mod_gate)
        # MLP Residual
        hidden_states = residual + hidden_states

        # -- txt --
        if self.mlp_txt is not None:
            # Modulate (Gate, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_attn_mod_gate is not None and und_token_lengths is not None:
                txt_attn_mod_gate = _repeat_interleave(txt_attn_mod_gate, und_token_lengths, dim=1)
            txt_hidden_states = apply_gate(txt_hidden_states, gate=txt_attn_mod_gate)
            # Attention Residual
            txt_hidden_states = txt_residual + txt_hidden_states
            # ============== MLP ==============
            txt_residual = txt_hidden_states
            # Post-Attention Normalization
            txt_hidden_states = self.post_attention_layernorm_txt(txt_hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_mlp_mod_shift is not None and und_token_lengths is not None:
                txt_mlp_mod_shift = _repeat_interleave(txt_mlp_mod_shift, und_token_lengths, dim=1)
                txt_mlp_mod_scale = _repeat_interleave(txt_mlp_mod_scale, und_token_lengths, dim=1)
            txt_hidden_states = modulate(txt_hidden_states, shift=txt_mlp_mod_shift, scale=txt_mlp_mod_scale)
            # MLP
            txt_hidden_states = self.mlp_txt(txt_hidden_states)
            # Modulate (Gate, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_mlp_mod_gate is not None and und_token_lengths is not None:
                txt_mlp_mod_gate = _repeat_interleave(txt_mlp_mod_gate, und_token_lengths, dim=1)
            txt_hidden_states = apply_gate(txt_hidden_states, gate=txt_mlp_mod_gate)
            # MLP Residual
            txt_hidden_states = txt_residual + txt_hidden_states

        return hidden_states, txt_hidden_states

    def get_mlp_layers(self):
        layers = dict(
            mlp=self.mlp,
        )
        if self.mlp_txt is not None:
            layers["mlp_txt"] = self.mlp_txt
        return layers

    @property
    def _materialize_and_init_state(self):
        return self.__materialize_and_init_state

    @_materialize_and_init_state.setter
    def _materialize_and_init_state(self, value):
        self.__materialize_and_init_state = value
        # Invoke hooks for moe layers
        for _, mlp in self.get_mlp_layers().items():
            if hasattr(mlp, "_materialize_and_init_state"):
                mlp._materialize_and_init_state = value


class LeoTripleLayer(LeoDualLayer):
    def __init__(
            self,
            config: LeoConfig,
            layer_idx: int,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            txt_config: Optional[LeoConfig] = None,
            audio_config: Optional[LeoConfig] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Create the main branch attention modules
        super().__init__(config, layer_idx, dtype, device, txt_config)
        self._config = config
        self._audio_config = audio_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Replace the self attention with dual attention
        self.self_attn = LeoTripleAttention(
            config, layer_idx, txt_config=self._txt_config, audio_config=self._audio_config,
            **factory_kwargs
        )

        # Create the text branch
        if audio_config.use_modulation:
            self.mod_proj_audio = ModulateDiT(
                audio_config.modulate_hidden_size, factor=6, act_layer=audio_config.act_class,
                output_hidden_size=audio_config.hidden_size, **factory_kwargs
            )
        self.input_layernorm_audio = audio_config.norm_class(
            audio_config.hidden_size, **audio_config.get_norm_kwargs(audio_config.norm_type), **factory_kwargs
        )
        self.post_attention_layernorm_audio = audio_config.norm_class(
            audio_config.hidden_size, **audio_config.get_norm_kwargs(audio_config.norm_type), **factory_kwargs
        )
        if layer_idx >= audio_config.moe_layer_num_skipped and (
                (isinstance(audio_config.num_experts, int) and audio_config.num_experts > 1)
                or (isinstance(audio_config.num_experts, list) and max(audio_config.num_experts) > 1)
        ):
            assert audio_config.moe_impl in MOE_LAYER_IMPL, f"moe_impl {audio_config.moe_impl} not supported."
            self.mlp_audio = MOE_LAYER_IMPL[audio_config.moe_impl](audio_config, layer_idx, **factory_kwargs)
        else:
            self.mlp_audio = HunyuanMLP(audio_config, layer_idx, **factory_kwargs)

    def forward(
            self,
            hidden_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            timestep_states: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            token_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
            token_lengths: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
            gen_cond_token_mask: Optional[torch.Tensor] = None,
            zero_timestep_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate Modulation Coefficients
        attn_mod_shift = attn_mod_scale = attn_mod_gate = None
        mlp_mod_shift = mlp_mod_scale = mlp_mod_gate = None
        txt_attn_mod_shift = txt_attn_mod_scale = txt_attn_mod_gate = None
        txt_mlp_mod_shift = txt_mlp_mod_scale = txt_mlp_mod_gate = None
        audio_attn_mod_shift = audio_attn_mod_scale = audio_attn_mod_gate = None
        audio_mlp_mod_shift = audio_mlp_mod_scale = audio_mlp_mod_gate = None
        gen_lengths_for_mod = token_lengths[0] if token_lengths is not None else None
        if self._config.use_modulation:
            # [bsz, modulate_hidden_size] for non-packed mode
            # [1, n_samples, modulate_hidden_size] for packed mode, where n_samples is the total number of samples
            # in the packed sequence
            assert isinstance(timestep_states, tuple), \
                (f"timestep_states should be a tuple of (img/video/text timestep states, audio timestep states), "
                 f"but got {type(timestep_states)}")
            if timestep_states[0] is not None:
                gen_mod = self.mod_proj(timestep_states[0])
                if gen_cond_token_mask is not None and zero_timestep_states is not None:
                    gen_mod = _apply_cond_zero_timestep_mod(
                        gen_mod, self.mod_proj(zero_timestep_states), gen_cond_token_mask, gen_lengths_for_mod)
                    gen_lengths_for_mod = None   # already per-token; skip later per-component repeat
                (
                    attn_mod_shift, attn_mod_scale, attn_mod_gate,
                    mlp_mod_shift, mlp_mod_scale, mlp_mod_gate,
                ) = gen_mod.chunk(6, dim=-1)

            if timestep_states[1] is not None:
                (
                    audio_attn_mod_shift, audio_attn_mod_scale, audio_attn_mod_gate,
                    audio_mlp_mod_shift, audio_mlp_mod_scale, audio_mlp_mod_gate,
                ) = self.mod_proj_audio(timestep_states[1]).chunk(6, dim=-1)

        if self._txt_config.use_modulation:
            (
                txt_attn_mod_shift, txt_attn_mod_scale, txt_attn_mod_gate,
                txt_mlp_mod_shift, txt_mlp_mod_scale, txt_mlp_mod_gate,
            ) = self.mod_proj_txt(timestep_states[0]).chunk(6, dim=-1)

        # ============== Attention ==============
        # -- img/video --
        residual, audio_residual, txt_residual = hidden_states
        hidden_states, audio_hidden_states, txt_hidden_states = hidden_states
        gen_token_indices, audio_token_indices, und_token_indices = token_indices
        gen_token_lengths, audio_token_lengths, und_token_lengths = token_lengths
        # Input Normalization
        if self.training or hidden_states.size(1) > 0:
            hidden_states = self.input_layernorm(hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if gen_token_indices is not None and attn_mod_shift is not None and gen_lengths_for_mod is not None:
                attn_mod_shift = _repeat_interleave(attn_mod_shift, gen_lengths_for_mod, dim=1)
                attn_mod_scale = _repeat_interleave(attn_mod_scale, gen_lengths_for_mod, dim=1)
            hidden_states = modulate(hidden_states, shift=attn_mod_shift, scale=attn_mod_scale)

        # -- txt --
        # Input Normalization
        txt_hidden_states = self.input_layernorm_txt(txt_hidden_states)
        # Modulate (Scale & Shift, No-Op if modulation is not enabled)
        if und_token_indices is not None and txt_attn_mod_shift is not None and und_token_lengths is not None:
            txt_attn_mod_shift = _repeat_interleave(txt_attn_mod_shift, und_token_lengths, dim=1)
            txt_attn_mod_scale = _repeat_interleave(txt_attn_mod_scale, und_token_lengths, dim=1)
        txt_hidden_states = modulate(txt_hidden_states, shift=txt_attn_mod_shift, scale=txt_attn_mod_scale)

        # -- audio --
        if audio_hidden_states is not None:
            # Input Normalization
            audio_hidden_states = self.input_layernorm_audio(audio_hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if audio_token_indices is not None and audio_attn_mod_shift is not None and audio_token_lengths is not None:
                audio_attn_mod_shift = _repeat_interleave(audio_attn_mod_shift, audio_token_lengths, dim=1)
                audio_attn_mod_scale = _repeat_interleave(audio_attn_mod_scale, audio_token_lengths, dim=1)
            audio_hidden_states = modulate(audio_hidden_states, shift=audio_attn_mod_shift, scale=audio_attn_mod_scale)

        # Self Attention
        hidden_states, audio_hidden_states, txt_hidden_states = self.self_attn(
            (hidden_states, audio_hidden_states, txt_hidden_states),
            attention_mask=attention_mask,
            rotary_position_embeddings=rotary_position_embeddings,
            token_indices=token_indices,
        )

        # -- img/video --
        if self.training or hidden_states.size(1) > 0:
            # Modulate (Gate, No-Op if modulation is not enabled)
            if gen_token_indices is not None and attn_mod_gate is not None and gen_lengths_for_mod is not None:
                attn_mod_gate = _repeat_interleave(attn_mod_gate, gen_lengths_for_mod, dim=1)
            hidden_states = apply_gate(hidden_states, gate=attn_mod_gate)
            # Attention Residual
            hidden_states = residual + hidden_states
            # ============== MLP ==============
            residual = hidden_states
            # Post-Attention Normalization
            hidden_states = self.post_attention_layernorm(hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if gen_token_indices is not None and mlp_mod_shift is not None and gen_lengths_for_mod is not None:
                mlp_mod_shift = _repeat_interleave(mlp_mod_shift, gen_lengths_for_mod, dim=1)
                mlp_mod_scale = _repeat_interleave(mlp_mod_scale, gen_lengths_for_mod, dim=1)
            hidden_states = modulate(hidden_states, shift=mlp_mod_shift, scale=mlp_mod_scale)
            # MLP
            hidden_states = self.mlp(hidden_states)
            # Modulate (Gate, No-Op if modulation is not enabled)
            if gen_token_indices is not None and mlp_mod_gate is not None and gen_lengths_for_mod is not None:
                mlp_mod_gate = _repeat_interleave(mlp_mod_gate, gen_lengths_for_mod, dim=1)
            hidden_states = apply_gate(hidden_states, gate=mlp_mod_gate)
            # MLP Residual
            hidden_states = residual + hidden_states

        # -- txt --
        if self.mlp_txt is not None:
            # Modulate (Gate, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_attn_mod_gate is not None and und_token_lengths is not None:
                txt_attn_mod_gate = _repeat_interleave(txt_attn_mod_gate, und_token_lengths, dim=1)
            txt_hidden_states = apply_gate(txt_hidden_states, gate=txt_attn_mod_gate)
            # Attention Residual
            txt_hidden_states = txt_residual + txt_hidden_states
            # ============== MLP ==============
            txt_residual = txt_hidden_states
            # Post-Attention Normalization
            txt_hidden_states = self.post_attention_layernorm_txt(txt_hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_mlp_mod_shift is not None and und_token_lengths is not None:
                txt_mlp_mod_shift = _repeat_interleave(txt_mlp_mod_shift, und_token_lengths, dim=1)
                txt_mlp_mod_scale = _repeat_interleave(txt_mlp_mod_scale, und_token_lengths, dim=1)
            txt_hidden_states = modulate(txt_hidden_states, shift=txt_mlp_mod_shift, scale=txt_mlp_mod_scale)
            # MLP
            txt_hidden_states = self.mlp_txt(txt_hidden_states)
            # Modulate (Gate, No-Op if modulation is not enabled)
            if und_token_indices is not None and txt_mlp_mod_gate is not None and und_token_lengths is not None:
                txt_mlp_mod_gate = _repeat_interleave(txt_mlp_mod_gate, und_token_lengths, dim=1)
            txt_hidden_states = apply_gate(txt_hidden_states, gate=txt_mlp_mod_gate)
            # MLP Residual
            txt_hidden_states = txt_residual + txt_hidden_states

        # -- audio --
        if audio_hidden_states is not None:
            # Modulate (Gate, No-Op if modulation is not enabled)
            if audio_token_indices is not None and audio_attn_mod_gate is not None and audio_token_lengths is not None:
                audio_attn_mod_gate = _repeat_interleave(audio_attn_mod_gate, audio_token_lengths, dim=1)
            audio_hidden_states = apply_gate(audio_hidden_states, gate=audio_attn_mod_gate)
            # Attention Residual
            audio_hidden_states = audio_residual + audio_hidden_states
            # ============== MLP ==============
            audio_residual = audio_hidden_states
            # Post-Attention Normalization
            audio_hidden_states = self.post_attention_layernorm_audio(audio_hidden_states)
            # Modulate (Scale & Shift, No-Op if modulation is not enabled)
            if audio_token_indices is not None and audio_mlp_mod_shift is not None and audio_token_lengths is not None:
                audio_mlp_mod_shift = _repeat_interleave(audio_mlp_mod_shift, audio_token_lengths, dim=1)
                audio_mlp_mod_scale = _repeat_interleave(audio_mlp_mod_scale, audio_token_lengths, dim=1)
            audio_hidden_states = modulate(audio_hidden_states, shift=audio_mlp_mod_shift, scale=audio_mlp_mod_scale)
            # MLP
            audio_hidden_states = self.mlp_audio(audio_hidden_states)
            # Modulate (Gate, No-Op if modulation is not enabled)
            if audio_token_indices is not None and audio_mlp_mod_gate is not None and audio_token_lengths is not None:
                audio_mlp_mod_gate = _repeat_interleave(audio_mlp_mod_gate, audio_token_lengths, dim=1)
            audio_hidden_states = apply_gate(audio_hidden_states, gate=audio_mlp_mod_gate)
            # MLP Residual
            audio_hidden_states = audio_residual + audio_hidden_states

        return hidden_states, audio_hidden_states, txt_hidden_states

    def get_mlp_layers(self):
        layers = dict(
            mlp=self.mlp,
            mlp_audio=self.mlp_audio,
        )
        if self.mlp_txt is not None:
            layers["mlp_txt"] = self.mlp_txt
        return layers

    @property
    def _materialize_and_init_state(self):
        return self.__materialize_and_init_state

    @_materialize_and_init_state.setter
    def _materialize_and_init_state(self, value):
        self.__materialize_and_init_state = value
        # Invoke hooks for moe layers
        for _, mlp in self.get_mlp_layers().items():
            if hasattr(mlp, "_materialize_and_init_state"):
                mlp._materialize_and_init_state = value


@dataclass
class LeoOutput(ModelOutput):
    """
    Base class for leo model (generalized autoregressive) outputs.

    Args:
        losses (dict[str, torch.Tensor], optional): A dictionary of loss components.
            Language modeling loss and diffusion loss are usually included.
        diffusion_prediction (torch.Tensor, optional): The predicted noise or denoised images
            from the diffusion modeling.
        audio_diffusion_prediction (torch.Tensor, optional): The predicted noise or denoised audio
    """

    losses: Optional[dict[str, torch.Tensor]] = None
    diffusion_prediction: Optional[torch.Tensor] = None
    audio_diffusion_prediction: Optional[torch.Tensor] = None


def unwrap(module):
    if isinstance(module, CheckpointWrapper):
        return module._checkpoint_wrapped_module  # noqa
    return module


class LeoModelBase(HunyuanMultimodalState):
    _config: LeoConfig
    _txt_config: LeoConfig

    def __post_init__(
            self,
            config: LeoConfig,
            txt_config: Optional[LeoConfig] = None,
            audio_config: Optional[LeoConfig] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            args: Namespace = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._config = config
        self._txt_config = txt_config = txt_config or config
        self._audio_config = audio_config   # audio config should be explicitly defined if audio branch is used.
        self.args = args or Namespace()     # for inference, args can be None.
        self.deterministic = False

        self._config.fp32_modules = getattr(args, "fp32_modules", []) or []
        if "proj_in" in self._config.fp32_modules:
            proj_in_factory_kwargs = {"dtype": torch.float32, 'device': device}
        else:
            proj_in_factory_kwargs = factory_kwargs
        if "proj_out" in self._config.fp32_modules:
            proj_out_factory_kwargs = {"dtype": torch.float32, 'device': device}
        else:
            proj_out_factory_kwargs = factory_kwargs
        if "timestep" in self._config.fp32_modules:
            timestep_factory_kwargs = {"dtype": torch.float32, 'device': device}
        else:
            timestep_factory_kwargs = factory_kwargs
        print(proj_in_factory_kwargs, proj_out_factory_kwargs, timestep_factory_kwargs)
        # Training related
        self.repa_loss_weight = getattr(args, "repa_loss_weight", 0.05)
        self.use_repa = config.use_repa and self.repa_loss_weight > 0
        self.moe_aux_loss_coeff = getattr(args, "moe_aux_loss_coeff", 0.0)

        # Image/Video projection layer
        self.time_embed = TimestepEmbedder(
            config.modulate_hidden_size if config.use_modulation else config.hidden_size,
            act_layer=config.act_class,
            **timestep_factory_kwargs,
        )
        if config.use_timestep_token:
            self.timestep_emb = TimestepEmbedder(config.hidden_size, act_layer=config.act_class, **timestep_factory_kwargs)

        if config.img_proj_type == "conv":
            proj_kwargs = dict(dims=config.img_proj_ndim, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_type="group")
            final_kwargs = dict(dims=config.img_proj_ndim, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_type="group")
        else:   # linear
            # Leo2 doesn't use modulation in the input projection layer.
            proj_kwargs = dict(dims=config.img_proj_ndim, use_modulation=False)
            final_kwargs = dict(dims=config.img_proj_ndim, norm_type="layer_f32", modulate_hidden_size=config.modulate_hidden_size)
        self.patch_embed = project_in_layer(proj_type=config.img_proj_type, config=config, **proj_kwargs, **proj_in_factory_kwargs)
        self.final_layer = project_out_layer(proj_type=config.img_proj_type, config=config, **final_kwargs, **proj_out_factory_kwargs)

        # Audio projection layer
        if self._audio_config is not None:
            self.audio_time_embed = TimestepEmbedder(
                audio_config.modulate_hidden_size if audio_config.use_modulation else audio_config.hidden_size,
                act_layer=audio_config.act_class,
                **timestep_factory_kwargs,
            )
            self.audio_projector = AudioProjection(
                in_channels=audio_config.audio_vae_latent_dim,
                hidden_size=audio_config.hidden_size,
                act_layer=audio_config.act_class,
                **proj_in_factory_kwargs,
            )
            self.audio_final_layer = project_out_layer(
                proj_type=audio_config.audio_proj_type,
                config=audio_config,
                norm_type="layer_f32",
                **proj_out_factory_kwargs,
            )

        # Text projection layer
        if config.text_proj_type == "linear":
            self.text_projector = TextProjection(
                in_channels=txt_config.text_states_hidden_dim,
                hidden_size=txt_config.hidden_size,
                act_layer=config.act_class,
                **factory_kwargs,
            )
        elif config.text_proj_type == "single_refiner":
            self.text_projector = SingleTokenRefiner(
                depth=2,
                txt_config=txt_config,
                **factory_kwargs,
            )
        else:
            raise ValueError(f"text_proj_type {config.text_proj_type} not supported.")

        # REPA
        if config.use_repa:
            assert config.repa_proj_hidden_size is not None, \
                "repa_proj_hidden_size must be specified when use_repa is True."
            self.repa_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.repa_proj_hidden_size, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(config.repa_proj_hidden_size, config.repa_proj_hidden_size, **factory_kwargs),
                nn.SiLU(),
                nn.Linear(config.repa_proj_hidden_size, config.repa_proj_out_size, **factory_kwargs),
            )

        # Transformers
        if self._audio_config is None:
            self.layers = nn.ModuleList([
                LeoDualLayer(config, layer_idx=i, txt_config=txt_config, **factory_kwargs)
                for i in range(config.num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                LeoTripleLayer(config, layer_idx=i, txt_config=txt_config, audio_config=audio_config, **factory_kwargs)
                for i in range(config.num_layers)
            ])

        # ====================== Finish model building =====================

        # Initialize cached rope, supporting automatic cache update
        self.cached_rope = CachedRoPE(config)
        self._cache_config = None
        self._leo_cache_controller = None

        # Initialize weights if needed
        self._prepare_reset_parameters()
        for name, module in self.named_modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _prepare_reset_parameters(self):
        # Globally set Linear and Embedding init methods to normal
        # for module in self.modules():
        #     if isinstance(module, (nn.Linear, nn.Embedding)):
        #         module.reset_parameters = normal_weight_reset_parameters(
        #             std=self._config.init_std, bias_type="zeros").__get__(module)
        # Set specific module init methods if available
        for name, module in self.named_modules():
            if hasattr(module, "prepare_reset_parameters"):
                module.prepare_reset_parameters()

    def get_config(self):
        return self._config

    def get_audio_config(self):
        return self._audio_config

    def get_txt_config(self):
        return self._txt_config

    def enable_cache(self, config: object = None) -> None:
        """Enable Leo first-block caching with a Diffusers-compatible config."""
        if self.is_cache_enabled:
            raise ValueError("Cache is already enabled; call `disable_cache()` before enabling it again.")
        if config is None:
            config = LeoFirstBlockCacheConfig()
        controller = LeoFirstBlockCacheController(config)
        self._cache_config = config
        self._leo_cache_controller = controller

    @property
    def is_cache_enabled(self) -> bool:
        """Return whether a Leo cache configuration is installed."""
        return self._leo_cache_controller is not None

    def disable_cache(self) -> None:
        """Disable Leo first-block caching and release cached activations."""
        self._reset_stateful_cache()
        self._leo_cache_controller = None
        self._cache_config = None

    def cache_context(self, name: str = "default"):
        """Scope stateful cache data to one complete inference request."""
        if self._leo_cache_controller is None:
            return nullcontext()
        return self._leo_cache_controller.context(name)

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        """Release request-local Leo cache state."""
        _ = recurse
        if self._leo_cache_controller is not None:
            self._leo_cache_controller.reset()

    def cache_stats(self) -> dict[str, int | float]:
        """Return first-block cache counters from the latest inference request."""
        if self._leo_cache_controller is None:
            return {"threshold": 0.0, "full_steps": 0, "skipped_steps": 0}
        return self._leo_cache_controller.stats()

    def get_printable_layers(self):
        if self._config.moe_layer_num_skipped == 0:
            return [unwrap(self.layers[0])]
        elif self._config.moe_layer_num_skipped > 0:
            return [unwrap(self.layers[0]), unwrap(self.layers[self._config.moe_layer_num_skipped])]
        return []

    def enable_deterministic(self) -> None:
        self.deterministic = True
        if hasattr(self.text_projector, "enable_deterministic"):
            self.text_projector.enable_deterministic()
        for layer in self.layers:
            unwrap(layer).enable_deterministic()

    def disable_deterministic(self) -> None:
        self.deterministic = False
        if hasattr(self.text_projector, "disable_deterministic"):
            self.text_projector.disable_deterministic()
        for layer in self.layers:
            unwrap(layer).disable_deterministic()

    def _align_repa_features(
            self,
            hidden_state: torch.Tensor,
            tk_depth: int,
            tk_height: int,
            tk_width: int,
            bsz: int,
            encoder_type: str
            ):

        if "DINOv3" in encoder_type:
            return hidden_state
        elif "qwen-3.5-9b" in encoder_type:
            patch_base = 32
            target_h_grid = int(round(tk_height * 16 / patch_base) * patch_base / 16)
            target_w_grid = int(round(tk_width * 16 / patch_base) * patch_base / 16)

            # (B, T*H*W, D) -> (B*T, D, H, W)
            x_2d = rearrange(hidden_state, "B (T H W) D -> (B T) D H W",
                             T=tk_depth, H=tk_height, W=tk_width)
            x_interpolated = F.interpolate(
                x_2d,
                size=(target_h_grid, target_w_grid),
                mode='bilinear',
                align_corners=False
            )
            # (B*T, D, H_grid, W_grid) -> (B, T*H_grid*W_grid, D)
            x_out = rearrange(x_interpolated, "(B T) D H W -> B (T H W) D", B=bsz, T=tk_depth)
            return x_out
        else:
            raise ValueError(f"repa_encoder_type {encoder_type} not supported.")

    def _get_features_for_repa(
            self,
            hidden_states: torch.Tensor,
            rope_media_info: list[list[tuple[slice, tuple[int, int, int], dict]]],
            use_packing: bool = False,
    ) -> torch.Tensor:
        if not use_packing:
            # Sequence pack is not supported in REPA for now.
            assert all(len(infos) == 1 for infos in rope_media_info), \
                "Multiple medias per sample are not supported in REPA for now."
            assert all(rope_media_info[0][0][1] == infos[0][1] for infos in rope_media_info), \
                f"All media in the batch should have the same token sizes for REPA, got {rope_media_info}"
            tk_depth, tk_height, tk_width = rope_media_info[0][0][1]
            bsz = hidden_states.shape[0]
            x_out = self._align_repa_features(hidden_states, tk_depth, tk_height, tk_width, bsz,
                                              self._config.repa_encoder_type)
            out_feature = self.repa_projector(x_out)
        else:
            aligned_list = []
            offset = 0
            bsz = 1
            for media_info in rope_media_info[0]:
                if media_info[2]["type"] != "gen_image":
                    continue
                tk_depth, tk_height, tk_width = media_info[1]
                tk_num = tk_height * tk_width

                hidden_state = hidden_states[:, offset: offset + tk_num]
                offset += tk_num

                x_out = self._align_repa_features(
                    hidden_state, tk_depth, tk_height, tk_width, bsz,
                    self._config.repa_encoder_type,
                )
                aligned_list.append(x_out)

            if len(aligned_list) == 0:
                return []

            seq_lens = [a.shape[1] for a in aligned_list]
            aligned_concat = torch.cat(aligned_list, dim=1)             # [1, sum N_i, D]
            projected_concat = self.repa_projector(aligned_concat)      # [1, sum N_i, D']
            out_feature = list(projected_concat.split(seq_lens, dim=1)) # list[ [1, N_i, D'] ]

        return out_feature

    @staticmethod
    def instantiate_vae_media_tokens(
            patch_embed: nn.Module,
            medias: BatchRaggedMedia,
            timestep_states: BatchRaggedTensor,
            allowed_dims: list,
    ):
        """
        Apply patch embed to medias and concatenate the resulting media tokens in the sequence dimension.

        Args:
            patch_embed (nn.Module): The patch embedding module to apply to the medias.
            timestep_states (BatchRaggedTensor): timestep states can be a 1-D tensor, or a list of 1-D tensors
            medias (BatchRaggedMedia): medias can be a 5-D tensor, or a list of 5-D tensors, or a list of lists
                of 4-D tensors.
            allowed_dims (list): allowed dimensions for media tensors, e.g. [4, 5] for images and videos.

        Returns:
            Concatenated input sequence
        """
        assert isinstance(medias, (torch.Tensor, list)), f"medias should be BatchRaggedMedia, got {type(medias)}"

        if isinstance(medias, torch.Tensor):
            assert medias.ndim in allowed_dims, f"medias should be a {allowed_dims}-D tensor, got {medias.ndim}-D"

            hidden_states, *_ = patch_embed(medias, timestep_states)  # (bsz, num_patches, n_embd)

        else:  # list (packing mode)
            assert len(medias) == len(timestep_states), \
                (f"Length of medias list ({len(medias)}) should match length of "
                 f"timestep_states list ({len(timestep_states)})")

            media_seq_lst = []
            for media_i, ts_state_i in zip(medias, timestep_states):

                if isinstance(media_i, torch.Tensor):
                    media_i_seq, *_ = patch_embed(media_i, ts_state_i)  # (n_i, num_patches, n_embd)

                elif isinstance(media_i, list):
                    media_i_seq_list = []
                    for j in range(len(media_i)):
                        media_ij = media_i[j].unsqueeze(0)
                        assert media_ij.ndim in allowed_dims, \
                            f"image_ij should have size of (1, C, H, W) or (1, C, D, H, W), got {list(media_ij.size())}"
                        media_ij_seq, *_ = patch_embed(media_ij, ts_state_i[j:j + 1])  # (1, num_patches, n_embd)
                        media_i_seq_list.append(media_ij_seq)
                    media_i_seq = torch.cat(media_i_seq_list, dim=1)  # (1, Σj num_patches_j, n_embd)

                else:
                    raise TypeError(f"image_i should be a {allowed_dims}-D tensor or a list, got {type(media_i)}")

                # Flat n_samples dimension to sequence dimension for packing
                media_i_seq = media_i_seq.reshape(1, -1, media_i_seq.size(-1))  # (1, num_patches_total, n_embd)
                media_seq_lst.append(media_i_seq)

            assert all(seq.size(1) == media_seq_lst[0].size(1) for seq in media_seq_lst), \
                "All media sequences should have the same sequence length after patch embedding."
            hidden_states = torch.cat(media_seq_lst, dim=0)  # (bsz, num_patches, n_embd)

        return hidden_states

    @staticmethod
    def instantiate_vae_media_tokens_full_seqlen(
            patch_embed: nn.Module,
            time_embed: nn.Module,
            hidden_states: torch.Tensor,
            timesteps: BatchRaggedTensor,
            medias: BatchRaggedMedia,
            media_mask: torch.Tensor,
            allowed_dims: list,
    ):
        """
        Instantiate the VAE media embeddings into the input embedding sequence(x).
        If x is None, using ts and images to create a new input embedding sequence.

        Args:
            patch_embed (nn.Module): The patch embedding module to apply to the medias.
            time_embed (nn.Module): The time embedding module to apply to the medias.
            hidden_states (torch.Tensor): input sequence, (bsz, seqlen, n_embd)
            timesteps (BatchRaggedTensor): ts can be a 1-D tensor, or a list of 1-D tensors
            medias (BatchRaggedMedia): images can be a 4-D tensor, or a list of 4-D tensors,
                or a list of lists of 3-D tensors.
            media_mask (torch.Tensor): (bsz, seqlen)
            allowed_dims (list): List of allowed tensor dimensions for media tensors.

        Returns:
            Instantiated input sequence
        """
        bsz, seqlen, n_embd = hidden_states.shape
        assert isinstance(medias, (torch.Tensor, list)), f"medias should be BatchRaggedMedia, got {type(medias)}"

        if isinstance(medias, torch.Tensor):
            assert medias.ndim in allowed_dims, f"medias should be a {allowed_dims}-D tensor, got {medias.ndim}-D tensor"
            assert isinstance(timesteps, torch.Tensor), f"timesteps should be 1-D tensor, got {type(timesteps)}"

            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            t_emb = time_embed(timesteps)     # (bsz, n_embd)
            media_seq, *_ = patch_embed(medias, t_emb)   # (bsz, num_patches, n_embd)
            media_index = index.masked_select(media_mask.bool()).reshape(bsz, -1)   # (bsz, num_patches)
            assert media_seq.size(1) == media_index.size(1), \
                f"image_seq ({list(media_seq.size())}) has inconsistent shape with index ({list(media_index.size())})"
            n_embd = media_seq.shape[-1]
            index_exp = media_index.unsqueeze(-1).expand(-1, -1, n_embd)
            hidden_states.scatter_(dim=1, index=index_exp, src=media_seq.to(hidden_states.dtype))

        else:   # list
            index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
            t_emb = []
            for i in range(len(medias)):
                media_i = medias[i]
                t_i = timesteps[i:i+1] if isinstance(timesteps, torch.Tensor) else timesteps[i]

                t_i_emb = time_embed(t_i)      # (n_i, n_embd)
                t_emb.append(t_i_emb)

                if isinstance(media_i, torch.Tensor):
                    media_i_seq, *_ = patch_embed(media_i, t_i_emb)  # (n_i, num_patches, n_embd)

                elif isinstance(media_i, list):
                    media_i_seq_list = []
                    for j in range(len(media_i)):
                        media_ij = media_i[j].unsqueeze(0)
                        assert media_ij.ndim in allowed_dims, \
                            f"image_ij should have size of {allowed_dims}-D tensor, got {list(media_ij.size())}"
                        media_ij_seq, *_ = patch_embed(media_ij, t_i_emb[j:j + 1])  # (1, num_patches, n_embd)
                        media_i_seq_list.append(media_ij_seq)
                    media_i_seq = torch.cat(media_i_seq_list, dim=1)    # (1, Σj num_patches_j, n_embd)

                else:
                    raise TypeError(f"image_i should be a 4-D or 5-D tensor or a list, got {type(media_i)}")

                media_i_index = index[i:i + 1].masked_select(media_mask[i:i + 1].bool()).reshape(1, -1)  # (1, img_seqlen)
                n_embd = media_i_seq.shape[-1]
                media_i_index_exp = media_i_index.unsqueeze(-1).expand(-1, -1, n_embd)
                media_i_seq_flat = media_i_seq.reshape(1, -1, n_embd)
                assert media_i_seq_flat.shape[1] == media_i_index_exp.shape[1], \
                    f"media_i_seq_flat ({list(media_i_seq_flat.size())}) has inconsistent shape with media_i_index_exp ({list(media_i_index_exp.size())})"
                hidden_states[i:i + 1].scatter_(dim=1, index=media_i_index_exp, src=media_i_seq_flat.to(hidden_states.dtype))

        return hidden_states, t_emb

    @staticmethod
    def instantiate_continuous_tokens_full_seqlen(
            hidden_states: torch.Tensor,
            emb_layer: nn.Module,
            scatter_src: Optional[BatchRaggedTensor] = None,
            scatter_index: Optional[BatchRaggedTensor] = None,
    ):
        bsz, _, _ = hidden_states.shape

        if isinstance(scatter_src, list):
            for i, scatter_src_i in enumerate(scatter_src):
                src = emb_layer(scatter_src_i)  # (n, n_embd)
                n_embd = src.shape[-1]
                index = scatter_index[i].unsqueeze(0).unsqueeze(-1).expand(-1, -1, n_embd)
                src = src.reshape(1, -1, n_embd)

                assert index.shape[1] == src.shape[1], \
                    f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
                hidden_states[i:i + 1].scatter_(dim=1, index=index, src=src.to(hidden_states.dtype))

        else:
            src = emb_layer(scatter_src.reshape(-1))    # (bsz * n, n_embd)
            n_embd = src.shape[-1]
            index = scatter_index.unsqueeze(-1).expand(-1, -1, n_embd)
            src = src.reshape(bsz, -1, n_embd)

            assert index.shape[1] == src.shape[1], \
                f"index ({list(index.size())}) has inconsistent shape with src ({list(src.size())})"
            hidden_states.scatter_(dim=1, index=index, src=src.to(hidden_states.dtype))

        return hidden_states

    def instantiate_text_tokens_full_seqlen(
            self,
            hidden_states: torch.Tensor,
            timesteps: BatchRaggedTensor,
            cond_text_states: torch.Tensor,
            cond_text_mask: torch.Tensor,
            text_mask: torch.Tensor,
    ):
        if self._config.text_proj_type == "linear":
            if self.training:
                # In packed mode, the batch dimension is flattened to sequence dimension for linear text projection.
                cond_text_states = cond_text_states.reshape(-1, cond_text_states.size(-1))[
                    cond_text_mask.view(-1).bool()].unsqueeze(0)
            text_states = self.text_projector(cond_text_states)
        elif self._config.text_proj_type == "single_refiner":
            text_states = self.text_projector(cond_text_states, t=timesteps, mask=cond_text_mask)
            if self.training:
                # In packed mode, the batch dimension is flattened to sequence dimension after single_refiner.
                text_states = text_states.reshape(-1, text_states.size(-1))[
                    cond_text_mask.view(-1).bool()].unsqueeze(0).contiguous()
        else:
            raise ValueError(f"text_proj_type {self._config.text_proj_type} not supported.")

        # Scatter text states to the full sequence.
        bsz, seqlen, _ = hidden_states.shape
        assert hidden_states.size(-1) == text_states.size(-1), \
            f"hidden_states ({list(hidden_states.size())}) and text_states ({list(text_states.size())}) have different hidden sizes"

        index = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)   # (bsz, seqlen)
        n_embd = text_states.shape[-1]
        # For inference, text token lengths are different for batch inference. We use for loop to scatter text states.
        # For training in pack mode, bsz should be 1.
        for i in range(bsz):
            index_i = index[i:i + 1].masked_select(text_mask[i:i + 1].bool()).reshape(1, -1)  # (1, text_seqlen)
            assert text_states.size(1) >= index_i.size(1), \
                f"text_states ({list(text_states.size())}) has less tokens than index ({list(index_i.size())})"
            index_i_exp = index_i.unsqueeze(-1).expand(-1, -1, n_embd)
            hidden_states[i:i + 1].scatter_(dim=1, index=index_i_exp, src=text_states[i:i + 1].to(hidden_states.dtype))

        return hidden_states

    @staticmethod
    def apply_padding(hidden_states, txt_hidden_states, audio_hidden_states, pad_count):
        """
        Apply padding to the hidden states of media, text, and audio branches if pad_count is specified.

        Args:
            hidden_states:
            txt_hidden_states:
            audio_hidden_states:
            pad_count (dict | None): the number of padding tokens to add to the sequence length of media tokens.
                Pad tokens are added at the end of the media token sequence, and will be masked out by attention_mask,
                so they won't affect attention or loss calculation. This is used to ensure:
                1) the sum of sequence lengths of all three branches is divisible by the chunk size of flash/flex attn.
                2) the hidden state of each branch is divisible by CP size (context parallel).
        """
        if pad_count is None:
            return hidden_states, txt_hidden_states, audio_hidden_states
        assert isinstance(pad_count, dict), \
            f"pad_count should be a dict with keys 'gen', 'und', and 'audio', got {type(pad_count)}"

        if hidden_states is not None:
            pad_shape = (hidden_states.size(0), pad_count["gen"], hidden_states.size(2))
            hidden_states = torch.cat([
                hidden_states,
                torch.zeros(pad_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            ], dim=1)
        if txt_hidden_states is not None:
            pad_shape = (txt_hidden_states.size(0), pad_count["und"], txt_hidden_states.size(2))
            txt_hidden_states = torch.cat([
                txt_hidden_states,
                torch.zeros(pad_shape, dtype=txt_hidden_states.dtype, device=txt_hidden_states.device)
            ], dim=1)
        if audio_hidden_states is not None:
            pad_shape = (audio_hidden_states.size(0), pad_count["audio"], audio_hidden_states.size(2))
            audio_hidden_states = torch.cat([
                audio_hidden_states,
                torch.zeros(pad_shape, dtype=audio_hidden_states.dtype, device=audio_hidden_states.device)
            ], dim=1)
        return hidden_states, txt_hidden_states, audio_hidden_states

    @staticmethod
    def check_sizes(
            attention_mask: torch.Tensor = None,
            hidden_states: torch.Tensor = None,
            txt_hidden_states: torch.Tensor = None,
            audio_hidden_states: torch.Tensor = None,
            gen_token_indices: torch.Tensor = None,
            und_token_indices: torch.Tensor = None,
            audio_token_indices: torch.Tensor = None,
            media_len: int = None,
    ):
        if gen_token_indices is not None:
            error_msg = "The sequence length of {}hidden_states ({}) should match the size of {}token_indices ({})"
            assert hidden_states.size(1) == gen_token_indices.size(1), \
                error_msg.format("", hidden_states.size(1), "gen_", gen_token_indices.size(1))
            assert txt_hidden_states.size(1) == und_token_indices.size(1), \
                error_msg.format("txt_", txt_hidden_states.size(1), "und_", und_token_indices.size(1))
            if audio_token_indices is not None and audio_hidden_states is not None:
                assert audio_hidden_states.size(1) == audio_token_indices.size(1), \
                    error_msg.format(
                        "audio_", audio_hidden_states.size(1), "audio_", audio_token_indices.size(1)
                    )
            assert gen_token_indices.size(1) + und_token_indices.size(1) + \
                (audio_token_indices.size(1) if audio_token_indices is not None else 0) == attention_mask.size(1), \
                (f"The total sequence length of token indices "
                 f"(gen: {gen_token_indices.size(1)}, und: {und_token_indices.size(1)}, "
                 f"audio: {audio_token_indices.size(1) if audio_token_indices is not None else 0}) "
                 f"should match the attention mask ({attention_mask.size(1)})")
        else:
            assert media_len + txt_hidden_states.size(1) == attention_mask.size(1), \
                (f"The sequence length of video+audio hidden states ({media_len}) and "
                 f"text hidden states ({txt_hidden_states.size(1)}) does not match the "
                 f"attention mask ({attention_mask.size(1)})")

    @staticmethod
    def ragged_final_layer(
            final_layer,
            hidden_states: torch.Tensor,
            timestep_states: BatchRaggedTensor,
            rope_media_info: list[list[tuple[slice, tuple[int, int, int], dict]]],
            token_lengths: torch.Tensor = None,     # with pad count if in packed mode.
            pad_count: int = None,
    ) -> BatchRaggedTensor:
        if token_lengths is None:
            # non-packed mode
            assert isinstance(timestep_states, torch.Tensor), \
                f"timestep_states should be a tensor in non-packed mode, got {type(timestep_states)}"
            # If there are context medias, they should be always before the generated media.
            token_sizes = rope_media_info[0][-1][1]
            assert all(info[-1][1] == token_sizes for info in rope_media_info), \
                f"All media in the batch should have the same token sizes, got {rope_media_info}"
            diff_pred = final_layer(hidden_states, timestep_states, *token_sizes)
        else:
            diff_pred_list = []
            assert isinstance(timestep_states, list) and len(timestep_states) == 1, \
                (f"timestep_states should be a list of tensors in packed mode, "
                 f"got {type(timestep_states)} with shape {timestep_states.shape=}")
            # Remove pad count.
            token_lengths_lst = token_lengths.tolist()
            if pad_count is not None and pad_count > 0:
                # Should Only enter this branch when cp_size > 1.
                token_lengths_lst[-1] = token_lengths_lst[-1] - pad_count
                hidden_states = hidden_states[:, :sum(token_lengths_lst), :]
            # Split hidden states according to token lengths, and apply final layer to each media separately.
            media_outputs = [hidden_states.split(token_lengths_lst, dim=1)]
            for media_out_i, ts_state_i, rope_info_i in zip(media_outputs, timestep_states, rope_media_info):
                diff_preds = []
                assert len(media_out_i) == len(rope_info_i), \
                    f"Length of media outputs ({len(media_out_i)}) should match length of rope_media_info ({len(rope_info_i)})"
                for j, (media_out_ij, rope_info_ij) in enumerate(zip(media_out_i, rope_info_i)):
                    diff_pred_ij = final_layer(media_out_ij, ts_state_i[j:j+1], *rope_info_ij[1])
                    diff_preds.append(diff_pred_ij)
                diff_pred_list.append(diff_preds)
            diff_pred = diff_pred_list

        return diff_pred

    @staticmethod
    def ragged_final_layer_full_seqlen(
            final_layer: nn.Module,
            timestep_states: torch.Tensor,
            hidden_states: torch.Tensor,
            media_mask: torch.Tensor,
            rope_media_info: list[list[tuple[slice, tuple[int, int, int], dict]]],
            free_input: bool = False,
    ):
        n_embd = hidden_states.size(-1)
        flat_hidden = hidden_states.reshape(-1, n_embd)
        flat_indices = media_mask.bool().reshape(-1).nonzero(as_tuple=True)[0]

        if isinstance(timestep_states, torch.Tensor):
            # When timesteps is a tensor, images must be a 4-D tensor (B, C, H, W), which means only one target image
            media_output = flat_hidden.index_select(0, flat_indices).reshape(
                -1, math.prod(rope_media_info[0][0][1]), n_embd)
            if free_input:
                hidden_states.untyped_storage().resize_(0)
            pred = final_layer(media_output, timestep_states, *rope_media_info[0][0][1])
        else:
            # When timesteps is a list, images must be a list of 4-D tensors or a list of list of 3-D tensors, and token_h and token_w must be a list of int or a list of list of int.
            # In this case, each line of the image_mask may contain different number of Trues, leading
            # the `reshape(batch_size, ...)` is not possible.
            sections = media_mask.sum(1).tolist()
            media_output = flat_hidden.index_select(0, flat_indices).split(sections)
            if free_input:
                hidden_states.untyped_storage().resize_(0)
            pred = []
            for media_output_i, t_emb_i, info_i in zip(media_output, timestep_states, rope_media_info):
                subsections = [math.prod(info_ij[1]) for info_ij in info_i]
                assert sum(subsections) == media_output_i.shape[0], \
                    (f"sum(subsections) ({sum(subsections)}) has inconsistent shape with media_output_i.shape[0] "
                     f"({media_output_i.shape[0]})")
                media_output_i = media_output_i.split(subsections)
                pred_i = []
                for j, media_output_ij in enumerate(media_output_i):
                    pred_ij = final_layer(media_output_ij[None], t_emb_i[j:j+1], *info_i[j][1])
                    pred_i.append(pred_ij)
                pred.append(pred_i) # a list of list of 4-D tensors [B x (N_i x [1, C, H_ij, W_ij])]
        return pred

    @staticmethod
    def check_types(attention_mask, rope_media_info):
        # attention_mask:
        # - BlockMask: flex attention
        # - torch.LongTensor/torch.IntTensor: flash, flash_packed
        assert isinstance(attention_mask, (torch.Tensor, BlockMask)), \
            f"attention_mask should be a torch.Tensor or BlockMask, got {type(attention_mask)}"
        if isinstance(attention_mask, torch.Tensor):
            assert attention_mask.dtype in [torch.long, torch.int32], \
                f"attention_mask should be of dtype long or int32, got {attention_mask.dtype}"
        # rope_media_info
        assert isinstance(rope_media_info, list), f"rope_media_info should be a list, got {type(rope_media_info)}"
        assert all(isinstance(info, list) for info in rope_media_info), \
            f"Each element of rope_media_info should be a list, got {type(rope_media_info[0])}"

    def forward(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            rope_media_info: Optional[list[list[tuple[slice, tuple[int, int, int], dict]]]] = None,
            return_dict: bool = True,
            # for gen image/video
            latents: BatchRaggedMedia = None,
            visual_mask: Optional[BatchRaggedMedia] = None,
            timesteps: BatchRaggedTensor = None,
            timesteps_index: Optional[BatchRaggedTensor] = None,
            # for gen audio
            audio_latents: Optional[torch.Tensor] = None,
            audio_mask: Optional[torch.Tensor] = None,
            audio_timesteps: Optional[torch.Tensor] = None,
            # for cond text
            cond_text_states: Optional[torch.Tensor] = None,
            cond_text_mask: Optional[torch.Tensor] = None,
            text_mask: Optional[torch.Tensor] = None,
            # r2v: full-seqlen scatter target for cond_text_states. When
            # None (e.g. plain t2v), the text-only `text_mask` is used. See data_provider_dit.
            cond_text_scatter_mask: Optional[torch.Tensor] = None,
            cond_vae_latents: Optional[BatchRaggedMedia] = None,
            cond_vae_timesteps: Optional[BatchRaggedTensor] = None,
            cond_vae_mask: Optional[torch.Tensor] = None,
            # sequence packing
            und_token_indices: Optional[torch.Tensor] = None,
            gen_token_indices: Optional[torch.Tensor] = None,
            audio_token_indices: Optional[torch.Tensor] = None,
            sample_offsets: Optional[torch.Tensor] = None,
            pad_count: Optional[dict[str, int]] = None,
            dummy_count: Optional[dict[str, int]] = None,   # Not used. Only for aligning interface with ptm2.
            und_token_lengths: Optional[torch.Tensor] = None,
            gen_token_lengths: Optional[torch.Tensor] = None,
            audio_token_lengths: Optional[torch.Tensor] = None,
            # only for training
            diffusion_loss_fn: Optional[Any] = None,
            audio_diffusion_loss_fn: Optional[Any] = None,
            visual_loss_weight: float = 1.0,
            audio_loss_weight: float = 1.0,
            repa_feats: Optional[torch.Tensor] = None,
            dataset_tag: str | None = None,     # for labeling multi-task losses
            # only for pipeline parallelism (not implemented yet, just a placeholder)
            ut: Optional[torch.Tensor] = None,
            aut: Optional[torch.Tensor] = None,
    ) -> LeoOutput | tuple:
        _ = ut  # not used for now
        _ = aut  # not used for now
        self.check_types(attention_mask, rope_media_info)
        if latents is None and audio_latents is None:
            raise ValueError("At least one of latents and audio_latents should be provided.")

        # === Input layers ===
        assert self._config.patch_size == 1, "instantiate_vae_image_tokens only supports patch_size=1 for now."
        gen_cond_token_mask = None
        zero_timestep_states = None

        if input_ids is None:
            # Project hidden_states
            timestep_states = self.time_embed(timesteps)
            hidden_states = self.instantiate_vae_media_tokens(
                self.patch_embed, latents, timestep_states, allowed_dims=[4, 5]
            )
            if audio_latents is not None:
                audio_timestep_states = self.audio_time_embed(audio_timesteps)
                # audio_hidden_states = self.audio_projector(audio_latents, audio_timestep_states)
                audio_hidden_states = self.instantiate_vae_media_tokens(
                    self.audio_projector, audio_latents, audio_timestep_states, allowed_dims=[3, 4]
                )
            else:
                audio_timestep_states = None
                audio_hidden_states = None

            # Project text conditions
            if self._config.text_proj_type == "linear":
                if gen_token_indices is not None and self.training and cond_text_states.size(0) == 1:
                    # Only the genuinely packed single-sequence training case (bsz==1) needs de-padding
                    cond_text_states = cond_text_states.reshape(-1, cond_text_states.size(-1))[
                        cond_text_mask.view(-1).bool()].unsqueeze(0)
                txt_hidden_states = self.text_projector(cond_text_states)
            elif self._config.text_proj_type == "single_refiner":
                txt_hidden_states = self.text_projector(cond_text_states, t=timesteps, mask=cond_text_mask)
                if gen_token_indices is not None and self.training and txt_hidden_states.size(0) == 1:
                    # Only the genuinely packed single-sequence training case (bsz==1) needs de-padding
                    txt_hidden_states = txt_hidden_states.reshape(-1, txt_hidden_states.size(-1))[
                        cond_text_mask.view(-1).bool()].unsqueeze(0).contiguous()
            else:
                raise ValueError(f"text_proj_type {self._config.text_proj_type} not supported.")

            # Apply padding
            # A potential mismatch: indices with dummy and pad have the order of real-pad-dummy, while hidden states have
            # the order of real-dummy-pad after padding. But since both dummy and pad tokens are masked out in attention
            # and loss calculation, it should not cause issues.
            hidden_states, txt_hidden_states, audio_hidden_states = self.apply_padding(
                hidden_states, txt_hidden_states, audio_hidden_states, pad_count=pad_count
            )
            if audio_latents is not None:
                media_len = hidden_states.size(1) + audio_hidden_states.size(1)
            else:
                media_len = hidden_states.size(1)
            # Check sizes. Also works for pipeline parallelism for verifying the random status correctness of samples.
            self.check_sizes(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                txt_hidden_states=txt_hidden_states,
                audio_hidden_states=audio_hidden_states,
                gen_token_indices=gen_token_indices,
                und_token_indices=und_token_indices,
                audio_token_indices=audio_token_indices,
                media_len=media_len,
            )

            # Calculate RoPE
            seqlen = media_len + txt_hidden_states.size(1)
            device = hidden_states.device
            cos, sin = self.cached_rope(
                seqlen, device, rope_media_info=rope_media_info, sample_offsets=sample_offsets,
            )
            if get_parallel_state().cp_size > 1:
                hidden_states = scatter_seq_and_register_cp_info(hidden_states, LEO_MEDIA_CP_INFO)
                txt_hidden_states = scatter_seq_and_register_cp_info(txt_hidden_states, LEO_TEXT_CP_INFO)
                if audio_latents is not None:
                    audio_hidden_states = scatter_seq_and_register_cp_info(audio_hidden_states, LEO_AUDIO_CP_INFO)

            if isinstance(timestep_states, list):
                timestep_states_tensor = torch.stack(timestep_states, dim=0)
            else:
                timestep_states_tensor = timestep_states
            if audio_latents is not None and isinstance(audio_timestep_states, list):
                audio_timestep_states_tensor = torch.stack(audio_timestep_states, dim=0)
            else:
                audio_timestep_states_tensor = audio_timestep_states

        else:
            if "proj_in" in self._config.fp32_modules:
                in_context = torch.autocast(device_type="cuda", enabled=False)
            else:
                in_context = nullcontext()
            with in_context:
                # ----------- Visual -----------
                hidden_states_fl = torch.zeros(
                    input_ids.size(0), input_ids.size(1), self._config.hidden_size,
                    dtype=torch.float32 if "proj_in" in self._config.fp32_modules else torch.bfloat16,
                    device=input_ids.device,
                )

                if latents is not None:
                    hidden_states_fl, timestep_states = self.instantiate_vae_media_tokens_full_seqlen(
                        self.patch_embed, self.time_embed, hidden_states_fl, timesteps, latents, visual_mask,
                        allowed_dims=[4, 5],
                    )
                    # r2v: scatter noise-free cond reference (cond_vae) tokens into their own mask positions at t=0.
                    # cond_vae_latents: merged reference-image VAE latents (cond_vae_images) and
                    # source-video VAE latents (cond_vae_videos)
                    if cond_vae_latents is not None and cond_vae_mask is not None:
                        hidden_states_fl, _ = self.instantiate_vae_media_tokens_full_seqlen(
                            self.patch_embed, self.time_embed, hidden_states_fl, cond_vae_timesteps,
                            cond_vae_latents, cond_vae_mask, allowed_dims=[4, 5],
                        )
                    if isinstance(timestep_states, list):
                        timestep_states_tensor = torch.stack(timestep_states, dim=0)
                    else:
                        timestep_states_tensor = timestep_states
                else:
                    timestep_states_tensor = None

                if timesteps_index is not None:
                    hidden_states_fl = self.instantiate_continuous_tokens_full_seqlen(
                        hidden_states_fl, emb_layer=self.timestep_emb, scatter_src=timesteps, scatter_index=timesteps_index
                    )

                # full sequence --> branch sequence (include pad/dummy tokens)
                gen_token_indices_ = gen_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states_fl.shape[-1])
                hidden_states = hidden_states_fl.gather(dim=1, index=gen_token_indices_)
                del hidden_states_fl

            # cond_vae_zero_timestep: 
            if (getattr(self._config, "cond_vae_zero_timestep", False)
                    and cond_vae_latents is not None and cond_vae_mask is not None):
                gen_cond_token_mask = cond_vae_mask.to(torch.bool).gather(dim=1, index=gen_token_indices)
                zero_timestep_states = self.time_embed(torch.zeros(1, device=hidden_states.device)).unsqueeze(0)
                if timestep_states_tensor is not None:
                    zero_timestep_states = zero_timestep_states.to(timestep_states_tensor.dtype)

            # ----------- Text -----------
            txt_hidden_states_fl = torch.zeros(
                input_ids.size(0), input_ids.size(1), self._txt_config.hidden_size,
                dtype=torch.float32 if self._config.text_proj_type == "single_refiner" else torch.bfloat16,
                device=input_ids.device,
            )
            # r2v scatters text + cond_vit hidden states into (text ∪ cond_vit) positions; plain
            # t2v has no cond_vit and falls back to the text-only mask.
            scatter_mask = cond_text_scatter_mask if cond_text_scatter_mask is not None else text_mask
            txt_hidden_states_fl = self.instantiate_text_tokens_full_seqlen(
                txt_hidden_states_fl, timesteps, cond_text_states, cond_text_mask, scatter_mask
            )

            # full sequence --> branch sequence (include pad/dummy tokens)
            und_token_indices_ = und_token_indices.unsqueeze(-1).expand(-1, -1, txt_hidden_states_fl.shape[-1])
            txt_hidden_states = txt_hidden_states_fl.gather(dim=1, index=und_token_indices_)
            del txt_hidden_states_fl

            if "proj_in" in self._config.fp32_modules:
                audio_in_context = torch.autocast(device_type="cuda", enabled=False)
            else:
                audio_in_context = nullcontext()
            with audio_in_context:
                # ----------- Audio -----------
                if audio_latents is not None:
                    audio_hidden_states_fl = torch.zeros(
                        input_ids.size(0), input_ids.size(1), self._audio_config.hidden_size,
                        dtype=torch.float32 if "proj_in" in self._config.fp32_modules else torch.bfloat16,
                        device=input_ids.device,
                    )
                    audio_hidden_states_fl, audio_timestep_states = self.instantiate_vae_media_tokens_full_seqlen(
                        self.audio_projector, self.audio_time_embed,
                        audio_hidden_states_fl, audio_timesteps, audio_latents, audio_mask,
                        allowed_dims=[3, 4],
                    )
                    if isinstance(audio_timestep_states, list):
                        audio_timestep_states_tensor = torch.stack(audio_timestep_states, dim=0)
                    else:
                        audio_timestep_states_tensor = audio_timestep_states

                    # full sequence --> branch sequence (include pad/dummy tokens)
                    audio_token_indices_ = audio_token_indices.unsqueeze(-1).expand(-1, -1, audio_hidden_states_fl.shape[-1])
                    audio_hidden_states = audio_hidden_states_fl.gather(dim=1, index=audio_token_indices_)
                    del audio_hidden_states_fl
                else:
                    audio_hidden_states = None
                    audio_timestep_states_tensor = None

            if get_parallel_state().cp_size > 1:
                hidden_states = scatter_seq_and_register_cp_info(hidden_states, LEO_MEDIA_CP_INFO)
                txt_hidden_states = scatter_seq_and_register_cp_info(txt_hidden_states, LEO_TEXT_CP_INFO)
                if audio_latents is not None:
                    audio_hidden_states = scatter_seq_and_register_cp_info(audio_hidden_states, LEO_AUDIO_CP_INFO)
                if gen_cond_token_mask is not None:
                    gen_cond_token_mask = maybe_scatter_seq(
                        gen_cond_token_mask.unsqueeze(-1).to(hidden_states.dtype),
                        cp_info=get_cp_info(LEO_MEDIA_CP_INFO),
                    ).squeeze(-1).to(torch.bool)

            seqlen = input_ids.size(1)
            device = hidden_states.device
            cos, sin = self.cached_rope(
                seqlen, device, rope_media_info=rope_media_info, sample_offsets=sample_offsets,
            )

        if attention_mask is not None and self._config.attn_impl in ["flash", "flash_packed", "flash3", "flash3_packed"]:
            from hy_parallelism.models.modules.attentions.flash import FlashAttnMaskInfo
            attention_mask = FlashAttnMaskInfo.from_attention_mask(attention_mask, pack="packed" in self._config.attn_impl)

        # Prepare transformer block inputs
        middle_layer_hidden_states = None
        # === Transformer blocks ===
        cache_controller = self._leo_cache_controller
        cache_active = (
            cache_controller is not None
            and cache_controller.active
            and not self.training
            and len(self.layers) > 1
        )
        if self._audio_config is not None:
            block_head_inputs = (hidden_states, audio_hidden_states, txt_hidden_states)
        else:
            block_head_inputs = (hidden_states, txt_hidden_states)
        block_head_outputs = None
        cache_hit = False
        for layer_idx, layer in enumerate(self.layers):
            if self._audio_config is not None:
                layer_inputs = [
                    (hidden_states, audio_hidden_states, txt_hidden_states),
                    (timestep_states_tensor, audio_timestep_states_tensor),
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, audio_token_indices, und_token_indices),
                    (gen_token_lengths, audio_token_lengths, und_token_lengths),
                ]
                hidden_states, audio_hidden_states, txt_hidden_states = layer(
                    *layer_inputs,
                    gen_cond_token_mask=gen_cond_token_mask,
                    zero_timestep_states=zero_timestep_states,
                )
            else:
                layer_inputs = [
                    (hidden_states, txt_hidden_states),
                    timestep_states_tensor,
                    attention_mask,
                    (cos, sin),
                    (gen_token_indices, und_token_indices),
                    (gen_token_lengths, und_token_lengths),
                ]
                hidden_states, txt_hidden_states = layer(
                    *layer_inputs,
                    gen_cond_token_mask=gen_cond_token_mask,
                    zero_timestep_states=zero_timestep_states,
                )

            if cache_active and layer_idx == 0:
                if self._audio_config is not None:
                    block_head_outputs = (hidden_states, audio_hidden_states, txt_hidden_states)
                else:
                    block_head_outputs = (hidden_states, txt_hidden_states)
                if cache_controller.should_reuse(block_head_inputs, block_head_outputs, leader_block=layer):
                    cached_outputs = cache_controller.apply_tail(block_head_outputs)
                    if self._audio_config is not None:
                        hidden_states, audio_hidden_states, txt_hidden_states = cached_outputs
                    else:
                        hidden_states, txt_hidden_states = cached_outputs
                    cache_hit = True
                    break

            # For repa
            if self.use_repa and self.training and layer_idx == self._config.num_layers // 2:
                if get_parallel_state().cp_size > 1:
                    hidden_states = maybe_gather_seq(hidden_states, cp_info=get_cp_info(LEO_MEDIA_CP_INFO))
                    raise NotImplementedError("REPA + CP + SeqPack is not checked.")
                use_packing = layer_inputs[5] is not None and layer_inputs[5][0] is not None
                middle_layer_hidden_states = self._get_features_for_repa(hidden_states, rope_media_info, use_packing)
                if get_parallel_state().cp_size > 1:
                    hidden_states = scatter_seq_and_register_cp_info(hidden_states, LEO_MEDIA_CP_INFO)

        if cache_active and not cache_hit:
            if self._audio_config is not None:
                block_outputs = (hidden_states, audio_hidden_states, txt_hidden_states)
            else:
                block_outputs = (hidden_states, txt_hidden_states)
            cache_controller.update_tail(block_head_outputs, block_outputs)

        if get_parallel_state().cp_size > 1:
            hidden_states = maybe_gather_seq(hidden_states, cp_info=get_cp_info(LEO_MEDIA_CP_INFO))
            if audio_latents is not None:
                audio_hidden_states = maybe_gather_seq(audio_hidden_states, cp_info=get_cp_info(LEO_AUDIO_CP_INFO))

        # === Output layers ===
        # Final projection
        # free_input = not torch.is_grad_enabled()
        free_input = False  # With final_layer memory optimization, we can disable free_input.
        if latents is not None:
            visual_rope_media_info = [
                # Only pass visual media info to the image/video final layer, include dummy ones.
                [info for info in infos if info[2]['type'] in ["gen_image", "gen_video"]]
                for infos in rope_media_info
            ]
            if input_ids is None:
                diff_pred = self.ragged_final_layer(
                    self.final_layer, hidden_states, timestep_states, visual_rope_media_info,
                    token_lengths=gen_token_lengths, pad_count=pad_count["gen"] if pad_count is not None else None,
                )
            else:
                # For training, bsz should be 1. For inference, bsz should be 1 or 2(with cfg).
                gen_branch_visual_mask = visual_mask.gather(dim=1, index=gen_token_indices)
                if free_input:
                    hidden_states.untyped_storage().resize_(0)
                if "proj_out" in self._config.fp32_modules:
                    out_context = torch.autocast(device_type="cuda", enabled=False)
                    hidden_states = hidden_states.float()
                else:
                    out_context = nullcontext()
                with out_context:
                    diff_pred = self.ragged_final_layer_full_seqlen(
                        self.final_layer, timestep_states, hidden_states, gen_branch_visual_mask, visual_rope_media_info,
                        free_input=free_input,
                    )
        else:
            diff_pred = None
        if audio_latents is not None:
            audio_rope_media_info = [
                # Only pass audio media info to the audio final layer， include dummy ones.
                [info for info in infos if info[2]['type'] == "gen_audio"]
                for infos in rope_media_info
            ]
            if input_ids is None:
                audio_diff_pred = self.ragged_final_layer(
                    self.audio_final_layer, audio_hidden_states, audio_timestep_states, audio_rope_media_info,
                    token_lengths=audio_token_lengths, pad_count=pad_count["audio"] if pad_count is not None else None,
                )
            else:
                audio_branch_audio_mask = audio_mask.gather(dim=1, index=audio_token_indices)
                if free_input:
                    audio_hidden_states.untyped_storage().resize_(0)
                if "proj_out" in self._config.fp32_modules:
                    audio_out_context = torch.autocast(device_type="cuda", enabled=False)
                    audio_hidden_states = audio_hidden_states.float()
                else:
                    audio_out_context = nullcontext()
                with audio_out_context:
                    audio_diff_pred = self.ragged_final_layer_full_seqlen(
                        self.audio_final_layer, audio_timestep_states, audio_hidden_states, audio_branch_audio_mask, audio_rope_media_info,
                        free_input=free_input,
                    )
        else:
            audio_diff_pred = None

        # -- for inference
        if not self.training:
            if not return_dict:
                return diff_pred, audio_diff_pred
            return LeoOutput(
                diffusion_prediction=diff_pred,
                audio_diffusion_prediction=audio_diff_pred,
            )

        # === Calculate losses ===
        losses = {}
        global_metric_losses = {} # used for global average loss calculation for per-task metrics when use_global_diffusion_loss_average is True
        loss = torch.tensor(0.0, device=hidden_states.device)
        use_global_diffusion_loss_average = getattr(self.args, "use_global_diffusion_loss_average", False)

        # -- diffusion loss
        if latents is not None:
            raw_visual_loss = diffusion_loss_fn(model_output=diff_pred)["loss"]
            if use_global_diffusion_loss_average:
                if visual_loss_weight > 0:
                    diff_loss_sum, diff_loss_count = self._get_local_sum_and_count(
                        raw_visual_loss, diff_pred
                    )
                    losses["diff_loss_sum"] = diff_loss_sum
                    losses["diff_loss_count"] = diff_loss_count
                    losses["diff_loss_weight"] = visual_loss_weight
                    loss_key = f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss"
                    global_metric_losses[loss_key] = (diff_loss_sum.detach().clone(), diff_loss_count.detach().clone())
                else:
                    visual_diff_loss = raw_visual_loss.mean()
                    loss = loss + visual_loss_weight * visual_diff_loss
            else:
                visual_diff_loss = raw_visual_loss.mean()
                if visual_loss_weight > 0:
                    loss_key = f"{dataset_tag}_image_loss" if dataset_tag is not None else "image_loss"
                    losses[loss_key] = visual_diff_loss.detach()
                loss = loss + visual_loss_weight * visual_diff_loss

        # -- audio diffusion loss
        if audio_latents is not None:
            raw_audio_loss = audio_diffusion_loss_fn(model_output=audio_diff_pred)["loss"]
            if use_global_diffusion_loss_average:
                if audio_loss_weight > 0:
                    audio_loss_sum, audio_loss_count = self._get_local_sum_and_count(
                        raw_audio_loss, audio_diff_pred
                    )
                    losses["audio_diff_loss_sum"] = audio_loss_sum
                    losses["audio_diff_loss_count"] = audio_loss_count
                    losses["audio_diff_loss_weight"] = audio_loss_weight
                    loss_key = f"{dataset_tag}_audio_loss" if dataset_tag is not None else "audio_loss"
                    global_metric_losses[loss_key] = (audio_loss_sum.detach().clone(), audio_loss_count.detach().clone())
                else:
                    audio_diff_loss = raw_audio_loss.mean()
                    loss = loss + audio_loss_weight * audio_diff_loss
            else:
                audio_diff_loss = raw_audio_loss.mean()
                if audio_loss_weight > 0:
                    loss_key = f"{dataset_tag}_audio_loss" if dataset_tag is not None else "audio_loss"
                    losses[loss_key] = audio_diff_loss.detach()
                loss = loss + audio_loss_weight * audio_diff_loss

        if global_metric_losses:
            losses["_global_metric_losses"] = global_metric_losses

        loss, losses = self.get_aux_losses(
            loss,
            losses,
            repa_feats=repa_feats,
            middle_layer_hidden_states=middle_layer_hidden_states,
            use_global_diffusion_loss_average=use_global_diffusion_loss_average,
        )

        # -- total loss (for backward)
        losses["loss"] = loss

        if not return_dict:
            return losses, diff_pred, audio_diff_pred

        return LeoOutput(
            losses=losses,
            diffusion_prediction=diff_pred,
            audio_diffusion_prediction=audio_diff_pred,
        )

    @staticmethod
    def _get_local_sum_and_count(
            raw_loss: torch.Tensor,
            pred,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute local loss sum and sample count for global average loss.

        raw_loss is a 1-D tensor of per-sample mean losses.
        pred can be a plain tensor (non-packing) or a ragged list structure (packing).
        In packing mode, pred is list[list[tensor]], and raw_loss[i] is already the
        average over the N_i sub-samples inside batch item i, so we weight by N_i.
        """
        if isinstance(pred, torch.Tensor):
            local_count = torch.tensor(float(pred.size(0)), device=raw_loss.device, dtype=torch.float64)
            return raw_loss.sum(), local_count
        num_per_batch_item = [len(item) for item in pred]
        num_per_item_tensor = torch.tensor(num_per_batch_item, device=raw_loss.device, dtype=raw_loss.dtype)
        local_loss_sum = (raw_loss * num_per_item_tensor).sum()
        local_count = torch.tensor(float(sum(num_per_batch_item)), device=raw_loss.device, dtype=torch.float64)
        return local_loss_sum, local_count

    def get_aux_losses(self, loss, losses, repa_feats=None, middle_layer_hidden_states=None,
                       use_global_diffusion_loss_average=False):
        if self._config.moe_aux_loss and self.moe_aux_loss_coeff > 0:
            # -- image/video moe losses
            if self._config.num_experts > 0:
                moe_aux_losses = [
                    block.mlp.get_balance_loss()
                    for block in self.layers
                    if isinstance(block.mlp, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(moe_aux_losses) > 0, "No MoE losses found across the model layers."
                moe_aux_loss = sum(moe_aux_losses)
                loss = loss + moe_aux_loss * self.moe_aux_loss_coeff
                losses["image_video_moe_loss"] = moe_aux_loss.detach() / len(moe_aux_losses)     # noqa

                capacity_rates = [
                    block.mlp.get_capacity_rate()
                        for block in self.layers   # noqa
                    if isinstance(block.mlp, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(capacity_rates) > 0, "No capacity losses found across the model layers."
                capacity_rates = sum(capacity_rates) / len(capacity_rates)
                losses["image_video_capacity_rate"] = capacity_rates

            # -- audio moe losses
            if self._audio_config is not None and self._audio_config.num_experts > 0:
                audio_moe_aux_losses = [
                    block.mlp_audio.get_balance_loss()
                    for block in self.layers
                    if isinstance(block.mlp_audio, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(audio_moe_aux_losses) > 0, "No MoE losses found across the model layers."
                audio_moe_aux_loss = sum(audio_moe_aux_losses)
                loss = loss + audio_moe_aux_loss * self.moe_aux_loss_coeff
                losses["audio_moe_loss"] = audio_moe_aux_loss.detach() / len(audio_moe_aux_losses)     # noqa

                audio_capacity_rates = [
                    block.mlp_audio.get_capacity_rate()
                    for block in self.layers
                    if isinstance(block.mlp_audio, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(audio_capacity_rates) > 0, "No capacity losses found across the model layers."
                audio_capacity_rates = sum(audio_capacity_rates) / len(audio_capacity_rates)
                losses["audio_capacity_rate"] = audio_capacity_rates

            # -- text moe losses
            if self._txt_config.num_experts > 0:
                text_moe_aux_losses = [
                    block.mlp_txt.get_balance_loss()
                    for block in self.layers
                    if isinstance(block.mlp_txt, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(text_moe_aux_losses) > 0, "No MoE losses found across the model layers."
                text_moe_aux_loss = sum(text_moe_aux_losses)
                loss = loss + text_moe_aux_loss * self.moe_aux_loss_coeff
                losses["text_moe_loss"] = text_moe_aux_loss.detach() / len(text_moe_aux_losses)     # noqa

                text_capacity_rates = [
                    block.mlp_txt.get_capacity_rate()
                    for block in self.layers
                    if isinstance(block.mlp_txt, (HunyuanMoE, DeepSeekMoE, ExpertParallelMoE))
                ]
                assert len(text_capacity_rates) > 0, "No capacity losses found across the model layers."
                text_capacity_rates = sum(text_capacity_rates) / len(text_capacity_rates)
                losses["text_capacity_rate"] = text_capacity_rates

        # -- repa loss
        if self.use_repa:
            if not isinstance(repa_feats, list):
                repa_feats = F.normalize(repa_feats, dim=-1)
                mid_hidden = F.normalize(middle_layer_hidden_states, dim=-1)
                repa_loss = torch.mean(
                    -(repa_feats * mid_hidden).sum(dim=-1),
                    dim=1,
                ).mean()
                losses["repa_loss"] = repa_loss.detach()
                loss = loss + self.repa_loss_weight * repa_loss
            else:
                if use_global_diffusion_loss_average:
                    if self.repa_loss_weight > 0:
                        repa_loss = 0.0
                        for f, m in zip(repa_feats, middle_layer_hidden_states):
                            repa_loss += -F.cosine_similarity(f, m, dim=-1).mean()
                        repa_loss /= len(repa_feats)

                        repa_loss_sum, repa_loss_count = self._get_local_sum_and_count(
                            repa_loss, [repa_feats]
                        )
                        losses["repa_loss_sum"] = repa_loss_sum
                        losses["repa_loss_count"] = repa_loss_count
                        losses["repa_loss_weight"] = self.repa_loss_weight
                else:
                    repa_loss = 0.0
                    for f, m in zip(repa_feats, middle_layer_hidden_states):
                        repa_loss += -F.cosine_similarity(f, m, dim=-1).mean()
                    repa_loss /= len(repa_feats)

                    losses["repa_loss"] = repa_loss.detach()
                    loss = loss + self.repa_loss_weight * repa_loss

        return loss, losses


class LeoModel(LeoModelBase):
    def __init__(
            self,
            args: Namespace,
            config: LeoConfig,
            txt_config: Optional[LeoConfig] = None,
            audio_config: Optional[LeoConfig] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.__post_init__(config, txt_config, audio_config, dtype, device, args)
