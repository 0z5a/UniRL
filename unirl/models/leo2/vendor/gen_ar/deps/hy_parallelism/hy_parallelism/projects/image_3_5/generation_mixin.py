import os
import functools
import json
import logging
import random
import math
import re
import time
import traceback
import gc
import contextlib
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Any, Union, List, TYPE_CHECKING
from functools import cache

import loguru
import torch
import torch.distributed as dist
from PIL import Image
from accelerate import dispatch_model
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import ALL_CACHE_NAMES, GenerationMixin, GenerateOutput, GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel, GenerationConfig
try:
    from transformers.modeling_utils import PretrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import ModelOutput

from hymm.ar.pipelines.pipeline_hunyuan_multimodal import HunyuanMultimodalPipeline
from hymm.diffusion.flow.transport import compute_empirical_mu
from hymm.diffusion.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hymm.core.global_vars import get_parallel_state
from hymm.data_kits.system_prompt import get_system_prompt
from hymm.data_kits.utils.image_utils import ImageProcessor
from hymm.models.autoregressive.custom_cache import HunyuanStaticCache
from hymm.models.tokenizers.conversation import get_conversation_template
from hymm.models.utils.generation_utils import MultimodalGenerationOutputs
from hymm.utils.helpers import default
from hymm.utils.rank_log import RankPrefixedTextStreamer, rank_print, rank_print_multiline
from hymm.utils.image_base import ImageInfo, ImageTensor, CondImage
from hymm.utils.import_utils import is_package_version
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.models.multimodal.hunyuan_multimodal import HunyuanMultimodalBase
from hymm.models.multimodal.hunyuan_multimodal import CausalSelfAttention
from hy_parallelism.common.logging import trace_log
from hy_parallelism.projects.image_3_5 import cuda_graph
from hy_parallelism.tools.profiling import start_profiling, profiler_step, ProfilingConfig, profile_range, profile_func, profile_class
from hy_parallelism.training.rollout import call_forward_directly

from hy_parallelism.context_parallel.core import (
    get_cp_info,
    maybe_scatter_seq,
    maybe_gather_seq,
    maybe_to_split_head,
    maybe_to_split_seq,
    maybe_to_cp_region_num_head,
    maybe_to_normal_region_num_head,
)
from hymm.models.basic.pos_emb_layers import apply_rope, apply_rope_qk

from torch.distributed.fsdp._fully_shard import FSDPModule

try:
    import flashinfer
except Exception:
    flashinfer = None


MOT_UND_CP_INFO_KEY = "hymm_mot_und"
MOT_GEN_CP_INFO_KEY = "hymm_mot_gen"
UND_MOE_NO_EP_MESH_TAG = "no_ep"

def _is_und_moe_fqn(fqn: str) -> bool:
    return fqn.endswith(".mlp") and "mlp_mot_gen" not in fqn


def _get_unsharded_param_tensor(param: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    if isinstance(param, DTensor):
        return param.full_tensor().detach()
    if hasattr(param, "full_tensor"):
        return param.full_tensor().detach()
    return param.detach()


def _all_gather_expert_tensor(local_tensor: torch.Tensor) -> torch.Tensor:
    p_state = get_parallel_state()
    if p_state.ep_size <= 1:
        return local_tensor

    ep_group = p_state.ep_group
    full_shape = list(local_tensor.shape)
    full_shape[0] *= p_state.ep_size
    output = torch.empty(full_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    input_tensor = local_tensor if local_tensor.is_contiguous() else local_tensor.contiguous()
    dist.all_gather_into_tensor(output, input_tensor, group=ep_group)
    return output


class GatheredHunyuanFusedExpert:
    EXPERT_WEIGHT_NAMES = ("gate_proj_weights", "up_proj_weights", "down_proj_weights")

    def __init__(
        self,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
        num_experts: int,
        use_flashinfer: bool = False,
    ):
        self.use_flashinfer = use_flashinfer
        self.num_local_experts = num_experts
        self.expert_gate_and_up_weights = None
        self.expert_down_weights = None
        self._weights_rearranged = False
        self.gate = None
        self.gate_impl = None
        if use_flashinfer:
            assert flashinfer is not None, "Package 'flashinfer' is not installed."
            self.expert_gate_and_up_weights, self.expert_down_weights = (
                self._build_flashinfer_weights(up_proj_weights, gate_proj_weights, down_proj_weights)
            )
            self._weights_rearranged = True
        else:
            self.gate_proj_weights = gate_proj_weights
            self.up_proj_weights = up_proj_weights
            self.down_proj_weights = down_proj_weights

    @staticmethod
    def _build_flashinfer_weights(
        up_proj_weights: torch.Tensor,
        gate_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_experts, ffn, hidden = up_proj_weights.shape
        expert_gate_and_up_weights = torch.empty(
            num_experts,
            2 * ffn,
            hidden,
            dtype=up_proj_weights.dtype,
            device=up_proj_weights.device,
        )
        # Copy and release one projection at a time to avoid torch.cat peak memory.
        expert_gate_and_up_weights[:, :ffn].copy_(up_proj_weights)
        del up_proj_weights
        expert_gate_and_up_weights[:, ffn:].copy_(gate_proj_weights)
        del gate_proj_weights
        expert_down_weights = torch.empty_like(down_proj_weights)
        expert_down_weights.copy_(down_proj_weights)
        del down_proj_weights
        return expert_gate_and_up_weights, expert_down_weights

    @classmethod
    def from_gathered_weights(
        cls,
        gate_proj_weights: torch.Tensor,
        up_proj_weights: torch.Tensor,
        down_proj_weights: torch.Tensor,
        num_experts: int,
    ):
        assert flashinfer is not None, "Package 'flashinfer' is not installed."
        expert_gate_and_up_weights, expert_down_weights = cls._build_flashinfer_weights(
            up_proj_weights, gate_proj_weights, down_proj_weights
        )
        self = cls.__new__(cls)
        self.use_flashinfer = True
        self.num_local_experts = num_experts
        self.expert_gate_and_up_weights = expert_gate_and_up_weights
        self.expert_down_weights = expert_down_weights
        self._weights_rearranged = True
        self.gate = None
        self.gate_impl = None
        return self

    def rearrange_weights_for_flashinfer(self):
        if self._weights_rearranged:
            return

        self.expert_gate_and_up_weights, self.expert_down_weights = self._build_flashinfer_weights(
            self.up_proj_weights,
            self.gate_proj_weights,
            self.down_proj_weights,
        )
        del self.gate_proj_weights, self.up_proj_weights, self.down_proj_weights
        self._weights_rearranged = True

    def forward_flashinfer(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        assert self.use_flashinfer, "forward_flashinfer requires use_flashinfer=True"
        self.rearrange_weights_for_flashinfer()
        torch.cuda.set_device(hidden_states.device.index)

        bsz, seqlen, hdim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hdim).contiguous()
        flashinfer_dtype = torch.bfloat16
        weighted_outputs = torch.zeros(
            hidden_states.shape[0],
            hdim,
            dtype=flashinfer_dtype,
            device=hidden_states.device,
        )
        _ = flashinfer.fused_moe.cutlass_fused_moe(  # noqa
            hidden_states.to(flashinfer_dtype),
            topk_idx.to(torch.int).contiguous(),
            topk_weights.to(torch.float).contiguous(),
            self.expert_gate_and_up_weights.to(flashinfer_dtype),
            self.expert_down_weights.to(flashinfer_dtype),
            flashinfer_dtype,
            output=weighted_outputs,
            quant_scales=None,
        )
        return weighted_outputs.to(hidden_states.dtype).view(bsz, seqlen, hdim)

    def __call__(self, hidden_states, *args, **kwargs):
        if self.use_flashinfer:
            topk_idx = kwargs.pop("topk_idx", None)
            topk_weights = kwargs.pop("topk_weights", None)
            if topk_idx is None or topk_weights is None:
                raise ValueError(
                    "GatheredHunyuanFusedExpert in flashinfer mode requires topk_idx and topk_weights."
                )
            if args or kwargs:
                raise ValueError(
                    "GatheredHunyuanFusedExpert in flashinfer mode does not support "
                    "index or num_global_sum_tokens_per_local_expert."
                )
            return self.forward_flashinfer(hidden_states, topk_idx, topk_weights)

        from hymm.models.basic.moe_layers import HunyuanFusedExpert
        return HunyuanFusedExpert.forward(self, hidden_states, *args, **kwargs)


class GatheredUndMoEProxy:
    _FORWARD_NON_EP_ATTRS = (
        "gate",
        "gate_impl",
        "moe_drop_token_enabled",
        "fused_expert",
        "shared_mlp",
        "num_experts",
        "top_k",
        "_config",
        "_moe_output_container",
        "training",
    )

    def __init__(self, moe, gathered_experts):
        self.experts = gathered_experts
        for attr in self._FORWARD_NON_EP_ATTRS:
            setattr(self, attr, getattr(moe, attr))

    def forward_expert(self, chunk_hidden_states: torch.Tensor, index: int) -> torch.Tensor:
        if self.fused_expert:
            return self.experts(chunk_hidden_states, index=index)
        return self.experts[index](chunk_hidden_states)

    def forward_non_ep_flashinfer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.experts.use_flashinfer, "forward_non_ep_flashinfer requires flashinfer gathered experts"
        input_hidden_states = hidden_states

        with torch.autocast('cuda', enabled=False):
            if self.gate_impl == "ep_moe":
                (
                    _balance_loss, _router_z_loss,
                    topk_idx, topk_weights,
                    _count_top_1_rate, _count_top_k_rate, _capacity_rate
                ) = self.gate(hidden_states)
            elif self.gate_impl in ("deepseek", "flashinfer"):
                topk_weights, topk_idx, _balance_loss, _capacity_rate = self.gate(hidden_states)
            else:
                raise NotImplementedError(
                    f"Gate impl `{self.gate_impl}` not implemented in GatheredUndMoEProxy."
                )

        weighted_outputs = self.experts(
            hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        if self.shared_mlp is not None:
            return weighted_outputs + self.shared_mlp(input_hidden_states)
        return weighted_outputs

class PrefetchManager:
    # self._get_fsdp_state()._states_to_forward_prefetch = [
    #     module._get_fsdp_state() for module in modules
    # ]
    def __init__(self, model):
        self.prefetch_dict = {} # module -> _states_to_forward_prefetch
        self.model = model

    def skip_prefetch(self):
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                self.prefetch_dict[id(module)] = module._get_fsdp_state()._states_to_forward_prefetch
                module._get_fsdp_state()._states_to_forward_prefetch = []

    def restore_prefetch(self):
        for module in self.model.modules():
            if id(module) in self.prefetch_dict:
                module._get_fsdp_state()._states_to_forward_prefetch = self.prefetch_dict[id(module)]
        self.prefetch_dict.clear()


def _assert_decode_dense_mask_is_full_attention(attention_mask: torch.Tensor, *, kv_len: int) -> None:
    assert attention_mask.ndim == 4, (
        f"expected 4D attention_mask for decode, got shape {tuple(attention_mask.shape)}"
    )
    assert attention_mask.size(-2) == 1, (
        f"expected q_len=1 in attention_mask, got {attention_mask.size(-2)}"
    )
    assert attention_mask.size(-1) >= kv_len, (
        f"attention_mask kv_len {attention_mask.size(-1)} < cache kv_len {kv_len}"
    )
    mask = attention_mask[..., :1, :kv_len]
    if mask.dtype == torch.bool:
        assert mask.all(), "bool decode attention_mask must be all True before setting to None"
    elif mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        assert mask.min() >= 0 and mask.max() <= 1 and (mask != 0).all(), (
            "integer decode attention_mask must be all 1 before setting to None"
        )
    else:
        raise AssertionError(
            f"unsupported attention_mask dtype {mask.dtype} for decode mask drop; "
            "only bool or 0/1 integer masks are supported"
        )


def forward_fast(
    self,
    hidden_states: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    past_key_values: Optional[HunyuanStaticCache] = None,
    und_token_indices: Optional[torch.Tensor] = None,
    gen_token_indices: Optional[torch.Tensor] = None,
    und_can_skip: bool = False,
    gen_can_skip: bool = False,
) -> torch.Tensor:
    und_hidden_states, gen_hidden_states = hidden_states
    if und_can_skip:
        assert und_hidden_states.numel() == 0
    if gen_can_skip:
        assert gen_hidden_states.numel() == 0

    bsz = und_hidden_states.shape[0]
    und_seqlen, gen_seqlen = und_hidden_states.shape[1], gen_hidden_states.shape[1]
    head_size = self._config.attention_head_size
    n_q_head = self._config.num_attention_heads
    n_kv_head = self._config.num_kv_heads
    q_per_kv = n_q_head // n_kv_head

    cp_enabled = get_parallel_state().cp_size > 1
    und_cp_info = get_cp_info(MOT_UND_CP_INFO_KEY) if cp_enabled else None
    gen_cp_info = get_cp_info(MOT_GEN_CP_INFO_KEY) if cp_enabled else None

    # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
    if not self._config.split_qkv:
        # [bsz, seqlen_local, (n_kv_head * (q_per_kv + 2) * head_size)]
        if not und_can_skip:
            qkv = self.qkv_proj(und_hidden_states)
        if not gen_can_skip:
            gen_qkv = self.qkv_proj_mot_gen(gen_hidden_states)
        total_qkv = q_per_kv + 2

        if cp_enabled:
            # all-to-all: [b, s_local, ...] -> [b, s_full, n_kv_local, total_qkv, hd]
            if not und_can_skip:
                qkv = self.all2all_qkv(qkv, head_size=head_size, total_qkv=total_qkv, cp_info=und_cp_info)
            if not gen_can_skip:
                gen_qkv = self.all2all_qkv(gen_qkv, head_size=head_size, total_qkv=total_qkv, cp_info=gen_cp_info)
            if not und_can_skip:
                und_seqlen = qkv.shape[1]
            if not gen_can_skip:
                gen_seqlen = gen_qkv.shape[1]
            n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])
        else:
            if not und_can_skip:
                qkv = qkv.view(bsz, und_seqlen, n_kv_head, total_qkv, head_size)
            if not gen_can_skip:
                gen_qkv = gen_qkv.view(bsz, gen_seqlen, n_kv_head, total_qkv, head_size)

        # (bsz, n_kv_head, q_per_kv+2, T, head_size)
        if not und_can_skip:
            qkv = qkv.permute(0, 2, 3, 1, 4)
        if not gen_can_skip:
            gen_qkv = gen_qkv.permute(0, 2, 3, 1, 4)

        # split batched computation into three
        if not und_can_skip:
            q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
        if not gen_can_skip:
            gen_q, gen_k, gen_v = gen_qkv.split((q_per_kv, 1, 1), dim=2)
    else:

        if not und_can_skip:
            q = self.q_proj(und_hidden_states)
            k = self.k_proj(und_hidden_states)
            v = self.v_proj(und_hidden_states)
        if not gen_can_skip:
            gen_q = self.q_proj_mot_gen(gen_hidden_states)
            gen_k = self.k_proj_mot_gen(gen_hidden_states)
            gen_v = self.v_proj_mot_gen(gen_hidden_states)

        if cp_enabled:
            # Each tensor: [b, s_local, n_head*hd] -> [b, s_full, n_head_local, hd]
            if not und_can_skip:
                q, k, v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, und_seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=und_cp_info,
                    ),
                    [q, k, v],
                )
            if not gen_can_skip:
                gen_q, gen_k, gen_v = map(
                    lambda x: maybe_to_split_head(
                        x.reshape(bsz, gen_seqlen, x.shape[-1] // head_size, head_size),
                        cp_info=gen_cp_info,
                    ),
                    [gen_q, gen_k, gen_v],
                )
            if not und_can_skip:
                und_seqlen = q.shape[1]
            if not gen_can_skip:
                gen_seqlen = gen_q.shape[1]
            n_q_head, n_kv_head = map(maybe_to_cp_region_num_head, [n_q_head, n_kv_head])

        if not und_can_skip:
            q = q.view(bsz, und_seqlen, n_kv_head, q_per_kv, head_size)
            k = k.view(bsz, und_seqlen, n_kv_head, 1, head_size)
            v = v.view(bsz, und_seqlen, n_kv_head, 1, head_size)
            q, k, v = map(lambda x: x.permute(0, 2, 3, 1, 4), [q, k, v])
        if not gen_can_skip:
            gen_q = gen_q.view(bsz, gen_seqlen, n_kv_head, q_per_kv, head_size)
            gen_k = gen_k.view(bsz, gen_seqlen, n_kv_head, 1, head_size)
            gen_v = gen_v.view(bsz, gen_seqlen, n_kv_head, 1, head_size)
            gen_q, gen_k, gen_v = map(lambda x: x.permute(0, 2, 3, 1, 4), [gen_q, gen_k, gen_v])

    # [bsz, h, seqlen, head_size]
    if not und_can_skip:
        q = q.reshape(bsz, n_q_head, und_seqlen, head_size)
        k = k.reshape(bsz, n_kv_head, und_seqlen, head_size)
        v = v.reshape(bsz, n_kv_head, und_seqlen, head_size)
    else:
        q = k = v = None
    if not gen_can_skip:
        gen_q = gen_q.reshape(bsz, n_q_head, gen_seqlen, head_size)
        gen_k = gen_k.reshape(bsz, n_kv_head, gen_seqlen, head_size)
        gen_v = gen_v.reshape(bsz, n_kv_head, gen_seqlen, head_size)
    else:
        gen_q = gen_k = gen_v = None

    merged_seqlen = und_seqlen + gen_seqlen
    # KV-cache decode usually activates one MoT branch only; avoid scatter/gather round-trips.
    single_branch_kv_decode = input_pos is not None and (und_can_skip or gen_can_skip) and torch.is_grad_enabled()

    if single_branch_kv_decode:
        if gen_can_skip:
            q_merge, k_merge, v_merge = q, k, v
        else:
            q_merge, k_merge, v_merge = gen_q, gen_k, gen_v

        if self._config.use_qk_norm and self._config.pre_qk_norm:
            if gen_can_skip:
                q_merge = self.query_layernorm(q_merge)
                k_merge = self.key_layernorm(k_merge)
            else:
                q_merge = self.query_layernorm_mot_gen(q_merge)
                k_merge = self.key_layernorm_mot_gen(k_merge)

        q_merge, k_merge = apply_rope_qk(q_merge, k_merge, *rotary_position_embeddings, apply_rope_in_fp32=self._config.apply_rope_in_fp32)

        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            if gen_can_skip:
                q_merge = self.query_layernorm(q_merge)
                k_merge = self.key_layernorm(k_merge)
            else:
                q_merge = self.query_layernorm_mot_gen(q_merge)
                k_merge = self.key_layernorm_mot_gen(k_merge)
    else:
        cos, sin = rotary_position_embeddings
        rope_idx_dim = cos.size(-1)

        def _rope_pos_indices(token_indices: torch.Tensor) -> torch.Tensor:
            return token_indices.unsqueeze(-1).expand(-1, -1, rope_idx_dim)

        # QK norm and RoPE are per-token; apply on each branch before merge to avoid gather/scatter.
        if self._config.use_qk_norm and self._config.pre_qk_norm:
            if not und_can_skip:
                q = self.query_layernorm(q)
                k = self.key_layernorm(k)
            if not gen_can_skip:
                gen_q = self.query_layernorm_mot_gen(gen_q)
                gen_k = self.key_layernorm_mot_gen(gen_k)

        if not und_can_skip:
            und_cos = cos.gather(1, _rope_pos_indices(und_token_indices))
            und_sin = sin.gather(1, _rope_pos_indices(und_token_indices))
            q, k = apply_rope_qk(q, k, und_cos, und_sin, apply_rope_in_fp32=self._config.apply_rope_in_fp32)
        if not gen_can_skip:
            gen_cos = cos.gather(1, _rope_pos_indices(gen_token_indices))
            gen_sin = sin.gather(1, _rope_pos_indices(gen_token_indices))
            gen_q, gen_k = apply_rope_qk(gen_q, gen_k, gen_cos, gen_sin, apply_rope_in_fp32=self._config.apply_rope_in_fp32)

        if self._config.use_qk_norm and not self._config.pre_qk_norm:
            if not und_can_skip:
                q = self.query_layernorm(q)
                k = self.key_layernorm(k)
            if not gen_can_skip:
                gen_q = self.query_layernorm_mot_gen(gen_q)
                gen_k = self.key_layernorm_mot_gen(gen_k)

        und_token_indices_q = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, n_q_head, -1, head_size)
        und_token_indices_kv = und_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, n_kv_head, -1, head_size)
        if not gen_can_skip:
            gen_token_indices_q = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, n_q_head, -1, head_size)
            gen_token_indices_kv = gen_token_indices.unsqueeze(-1).unsqueeze(1).expand(-1, n_kv_head, -1, head_size)
        else:
            gen_token_indices_q = gen_token_indices_kv = None

        def _scatter(und_src, gen_src, und_idx, gen_idx, n_head):
            ref = und_src if und_src is not None else gen_src
            target = torch.zeros((bsz, n_head, merged_seqlen, head_size), dtype=ref.dtype, device=ref.device)
            if not und_can_skip:
                target.scatter_(dim=2, index=und_idx, src=und_src)
            if not gen_can_skip:
                target.scatter_(dim=2, index=gen_idx, src=gen_src)
            return target

        q_merge = _scatter(q, gen_q, und_token_indices_q, gen_token_indices_q, n_q_head)
        k_merge = _scatter(k, gen_k, und_token_indices_kv, gen_token_indices_kv, n_kv_head)
        v_merge = _scatter(v, gen_v, und_token_indices_kv, gen_token_indices_kv, n_kv_head)

    q_merge = q_merge.to(v_merge.dtype)
    k_merge = k_merge.to(v_merge.dtype)

    # Restore from kv_cache and update
    if input_pos is not None:
        cache_kwargs = {"cache_position": input_pos}
        k_merge, v_merge = past_key_values.update(k_merge, v_merge, self.layer_idx, cache_kwargs)

    # maybe repeat k and v if for the non multi-head attention cases
    # GQA: handled inside attention backends (flex/sdpa enable_gqa, magi native); no repeat needed.
    if self._should_repeat_kv_heads(
        attention_mask,
        n_kv_head=n_kv_head,
        n_q_head=n_q_head,
        input_pos=input_pos,
        q_per_kv=q_per_kv,
    ):
        k_merge = k_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).flatten(1, 2)
        v_merge = v_merge.unsqueeze(dim=2).expand(-1, -1, q_per_kv, -1, -1).flatten(1, 2)


    from hy_parallelism.training.checkpointing import flush_pending_offloads
    if q_merge.size(2) == 1 and input_pos is not None:
        if isinstance(attention_mask, torch.Tensor):
            _assert_decode_dense_mask_is_full_attention(attention_mask, kv_len=k_merge.size(2))
            attention_mask = None
    y = self.scaled_dot_product_attention(q_merge, k_merge, v_merge, attention_mask)
    flush_pending_offloads()

    # re-assemble all head outputs side by side
    y = y.reshape(bsz, merged_seqlen, head_size * n_q_head)

    if single_branch_kv_decode:
        if gen_can_skip:
            core_attn_out = y
            gen_core_attn_out = y.new_zeros(bsz, 0, y.size(-1))
        else:
            core_attn_out = y.new_zeros(bsz, 0, y.size(-1))
            gen_core_attn_out = y
    else:
        core_attn_out = y.gather(dim=1, index=und_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))
        gen_core_attn_out = y.gather(dim=1, index=gen_token_indices.unsqueeze(-1).expand(-1, -1, y.size(-1)))

    if cp_enabled:
        core_attn_out = maybe_to_split_seq(
            core_attn_out.view(bsz, und_seqlen, n_q_head, head_size),
            cp_info=und_cp_info,
        )
        gen_core_attn_out = maybe_to_split_seq(
            gen_core_attn_out.view(bsz, gen_seqlen, n_q_head, head_size),
            cp_info=gen_cp_info,
        )
        n_q_head = maybe_to_normal_region_num_head(n_q_head)
        und_seqlen, gen_seqlen = core_attn_out.shape[1], gen_core_attn_out.shape[1]
        core_attn_out = core_attn_out.reshape(bsz, und_seqlen, n_q_head * head_size)
        gen_core_attn_out = gen_core_attn_out.reshape(bsz, gen_seqlen, n_q_head * head_size)

    und_hidden_states = self.o_proj(core_attn_out)
    gen_hidden_states = self.o_proj_mot_gen(gen_core_attn_out)
    return und_hidden_states, gen_hidden_states



class RouterReplayMixin:
    """MoE router-replay recording helpers for generation models.

    Owns group tagging, RECORD on/off, and static buffers.
    ``__call__`` invokes ``record_router_replay_step(*args, **kwargs)`` after the
    real forward with the **same** arguments as that call.

    ``record_router_replay_step`` is **not implemented** here — subclasses must
    override it. Assemble / ``set_replay`` are also left to the caller (e.g.
    ``assemble_routing_captures`` + ``AssembledRouterReplay.set_replay``).

    Typical flow::

        model.set_router_replay_group("teacher")
        model.start_router_recording()
        model.generate(...)  # each __call__ triggers record_router_replay_step
        model.stop_router_recording()
        # caller assembles captures and calls AssembledRouterReplay.set_replay(...)
    """

    # Max sequence length for CUDA-graph-friendly static routing buffers.
    ROUTER_REPLAY_STATIC_MAX_TOKENS = int(os.environ.get("ROUTER_REPLAY_STATIC_MAX_TOKENS", 200_000))

    def _ensure_router_replay_state(self):
        if getattr(self, "_router_replay_state_ready", False):
            return
        self._router_replay_group: Optional[str] = None
        self._router_replay_capturing: bool = False
        self._router_replay_static_buffer = None
        self.router_replay_captures: List = []
        self._router_replay_state_ready = True

    def _reset_router_replay_captures(self):
        self.router_replay_captures = []

    def collect_router_replay_instances(self):
        """Collect ``RouterReplay`` instances on this model (layer order)."""
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay

        instances = []
        for _, module in self.named_modules():
            rr = getattr(module, "router_replay", None)
            if isinstance(rr, RouterReplay):
                instances.append(rr)
        return instances

    def _setup_router_replay_static_buffer(self):
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay

        bufs = []
        for _, module in self.named_modules():
            rr = getattr(module, "router_replay", None)
            if not isinstance(rr, RouterReplay):
                continue
            if not hasattr(module, "top_k"):
                raise RuntimeError(
                    f"RouterReplay on {type(module).__name__} has no top_k for static buffer"
                )
            buf = torch.empty(
                (self.ROUTER_REPLAY_STATIC_MAX_TOKENS, int(module.top_k)),
                dtype=torch.int32,
                device="cuda",
            )
            rr.set_static_buffer(buf)
            bufs.append(buf)
        if not bufs:
            raise RuntimeError("No RouterReplay instances for static buffer setup")
        self._router_replay_static_buffer = bufs

    def set_router_replay_group(self, name: str):
        # tag with model-level name
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay

        self._ensure_router_replay_state()
        instances = self.collect_router_replay_instances()
        if not instances:
            raise RuntimeError(
                "No RouterReplay instances found. Enable "
                "moe_enable_routing_replay when building MoE gates."
            )
        RouterReplay.tag_instances(name, instances)
        self._router_replay_group = name
        return instances

    def clear_router_replay_runtime(
        self,
        group_names: Optional[List[str]] = None,
        *,
        action: bool = True,
        indices: bool = True,
        static_buffers: bool = False,
    ):
        """Clear RouterReplay global state for the given groups.

        Defaults to the current model replay group when ``group_names`` is None.
        """
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay

        self._ensure_router_replay_state()
        if group_names is None:
            if self._router_replay_group is None:
                return
            group_names = [self._router_replay_group]

        for name in group_names:
            if action:
                RouterReplay.clear_global_router_replay_action(name=name)
            if indices:
                RouterReplay.clear_global_indices(name=name)
            if static_buffers:
                RouterReplay.clear_global_static_buffers(name=name)

    def start_router_recording(
        self,
        group_name: Optional[str] = None,
        *,
        clear_captures: bool = True,
    ):
        """Enable RECORD. clear_captures=False keeps captures across generate phases."""
        from hy_parallelism.models.modules.moe.routers.router_replay import (
            RouterReplay,
            RouterReplayAction,
        )

        self._ensure_router_replay_state()
        if group_name is not None:
            self.set_router_replay_group(group_name)
        elif self._router_replay_group is None:
            self.set_router_replay_group("default")

        self._setup_router_replay_static_buffer()
        # Drop prior REPLAY targets/action before RECORD.
        self.clear_router_replay_runtime()
        RouterReplay.set_global_router_replay_action(
            RouterReplayAction.RECORD, name=self._router_replay_group
        )

        if clear_captures:
            self._reset_router_replay_captures()
            self.router_replay_assembled = None
        self._router_replay_capturing = True

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        if getattr(self, "_router_replay_capturing", False):
            self.record_router_replay_step(*args, **kwargs)
        return out

    def record_router_replay_step(self, *args, **kwargs):
        """Snapshot this forward's routing using the same inputs as ``__call__``.

        Invoked automatically by ``RouterReplayMixin.__call__`` **after** the
        real forward, with identical ``*args, **kwargs`` (so overrides can
        inspect ``und_token_indices``, ``gen_token_indices``, ``input_ids``,
        etc. the same way the forward did).

        **Not implemented in the base mixin.** Subclasses must override.
        Snapshot per-stream routing from the same ``*args, **kwargs`` as the
        forward; do not assume a shared contiguous ``pos_ids`` across routers.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.record_router_replay_step(*args, **kwargs) "
            "is not implemented."
        )

    def stop_router_recording(self):
        """Stop RECORD and clear static buffers."""
        self._ensure_router_replay_state()
        self._router_replay_capturing = False
        # Keep captured indices; only drop action + static buffers.
        self.clear_router_replay_runtime(indices=False, static_buffers=True)
        self._router_replay_static_buffer = None


class Image35RouterReplayMixin(RouterReplayMixin):

    und_moe_cpu_offload_sharded_params = True

    def _ensure_router_replay_state(self):
        super()._ensure_router_replay_state()
        if not hasattr(self, "_router_replay_und_committed"):
            self._router_replay_und_committed = 0
            self._router_replay_gen_committed = 0

    def _reset_router_replay_captures(self):
        super()._reset_router_replay_captures()
        self._router_replay_und_committed = 0
        self._router_replay_gen_committed = 0

    @dataclass
    class MotRouterReplayStep:
        und_topk_idx: Optional[List[torch.Tensor]]
        gen_topk_idx: Optional[List[torch.Tensor]]
        und_token_indices: Optional[torch.Tensor]
        gen_token_indices: Optional[torch.Tensor]

    @dataclass
    class MotRouterReplayAssembled:
        # 与 RouterReplay 实例同序；某层为 None 表示该层不 replay（自由 routing）
        topk_indices: List[Optional[torch.Tensor]]
        und_token_indices: Optional[torch.Tensor]
        gen_token_indices: Optional[torch.Tensor]
        is_gen: List[bool]  # 与 topk_indices 同序；True=gen / False=und

        def to(self, device):
            self.topk_indices = [
                t.to(device) if t is not None else None for t in self.topk_indices
            ]
            if self.und_token_indices is not None:
                self.und_token_indices = self.und_token_indices.to(device)
            if self.gen_token_indices is not None:
                self.gen_token_indices = self.gen_token_indices.to(device)
            return self

        def set_replay(
            self,
            name: Optional[str] = None,
            *,
            set_action: bool = True,
            pad_to: Optional[int] = None,
            expand_batch: Union[bool, int] = False,
            und_len: Optional[int] = None,
            gen_len: Optional[int] = None,
        ):
            """Install replay tables.

            expand_batch: CFG 倍数（True=2）。capture 为 B=1 [T,K] 时扩到 B 份。
            und_len / gen_len: forward 每样本 und/gen 长度；比 capture 长则按样本 pad，
            后缀 fallback 到自然 topk（避免 flat pad 在 B>1 错位）。
            """
            assert pad_to is None, "pad_to 参数没设计好，很不好用，特别不适合 mot, 因为 mot 长度不一定对"
            from hy_parallelism.models.modules.moe.routers.router_replay import (
                RouterReplay,
                RouterReplayAction,
            )
            instances = RouterReplay._instances(name)
            if len(instances) != len(self.topk_indices):
                raise ValueError(
                    f"topk_indices ({len(self.topk_indices)}) != instances ({len(instances)})"
                )
            batch = 2 if expand_batch is True else (int(expand_batch) if expand_batch else 1)
            for i, (rr, topk, is_gen) in enumerate(zip(instances, self.topk_indices, self.is_gen)):
                rr._replay_logged = i < 2  # 只 log 前两层，避免刷屏
                rr.clear_indices()
                if topk is None:
                    rr.clear_router_replay_action()
                    continue
                mask = None
                if batch > 1:
                    # CFG B>1 且 capture 短于 forward 时：须按样本 pad。
                    # 若先 repeat 再 flat 尾部 pad，会错成「s0 全 replay、s1 全 fallback」。
                    length = gen_len if is_gen else und_len
                    if length is None:
                        topk = topk.repeat(batch, 1)
                    else:
                        cap = topk.shape[0]
                        if length < cap:
                            raise ValueError(f"replay T={cap} > forward T={length}")
                        # [B, cap, K] → pad 到 [B, length, K] → flat
                        topk = topk.unsqueeze(0).expand(batch, -1, -1)
                        if length > cap:
                            topk = torch.nn.functional.pad(topk, (0, 0, 0, length - cap))
                            mask = (torch.arange(length, device=topk.device) < cap).repeat(batch)
                        topk = topk.reshape(batch * length, -1)
                rr.set_target_indices(topk, valid_mask=mask)
                if set_action:
                    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            return self

    @staticmethod
    def _cat_mot_stream(steps, topk_attr, idx_attr):
        """按时间步拼接同一 MOT stream 的 topk / indices。

        每步 topk 为 [B*T, K]（与 gate 的 batch-major flatten 一致），indices 为 [B, T]。
        本函数先 view 成 [B, T, K] 再沿 T 维 cat，最后 reshape 回 [B*T_total, K]；
        indices 仅 cat(dim=1)，其数值不参与 topk 重排。

        正确性假设：
        - 后续一次 forward 的该 stream indices 等于各 step indices 按时间cat(dim=1)（gather 顺序 = 各步拼接顺序）。
        - 步间无位置重叠；各 step 的 B 相同，且 topk.shape[0] == B*T。
        - 同 stream 各步的 MoE 层数一致；steps 为时间序。
        """

        if not steps:
            return None, None
        idxs = [getattr(s, idx_attr) for s in steps]
        indices = torch.cat(idxs, dim=1)
        b, t = indices.shape
        layers = [
            torch.cat(
                [getattr(s, topk_attr)[i].view(b, idx.shape[1], -1) for s, idx in zip(steps, idxs)],
                dim=1,
            ).reshape(b * t, -1)
            for i in range(len(getattr(steps[0], topk_attr)))
        ]
        return layers, indices

    def _delta_mot_stream(
        self,
        idx: Optional[torch.Tensor],
        topk_layers: Optional[List[torch.Tensor]],
        committed: int,
        *,
        input_pos: Optional[torch.Tensor],
    ):
        """从本步 gate 记录里切出「尚未写入 captures」的新 token 段。

        参数含义（以 und 流、B=1 为例）::

            idx          # [B, T] 本步 gather 用的下标（有 cache 时多为 local）
            topk_layers  # 每层 [B*T, K]，与 idx 的 T 对齐
            committed    # 已经记进 captures 的 stream token 数
            input_pos    # [B, cur_len] 本步每个位置的绝对 pos

        例子 1 — 首次 prefill（L=100，und 共 80 个）::

            input_pos = [0,1,...,99]          # 整段从 0 连续 → prefill=True
            idx.shape[1] = T = 80
            committed = 0
            → start=0，整段都记；返回后 committed=80

        例子 2 — decode 一步（新 token 绝对位置 100）::

            input_pos = [100]                 # 短窗 → prefill=False
            idx = [[0]]                       # local；T=1
            committed = 80
            → start=0，记这 1 个；store_idx=[[100]]；committed=81

        例子 3 — 插图后整段 re-prefill（und 仍 81，无新 und）::

            input_pos = [0,1,...,130]         # 又是整段从 0 → prefill=True
            T = 81, committed = 81
            → start=81 >= T，本 stream 不记（避免把前缀再 append 一遍）

        例子 4 — re-prefill 时 und 变长到 90（多了 9 个）::

            T=90, committed=81, prefill=True
            → start=81，只切 idx/topk 的后缀 9 个；committed=90

        返回:
            (new_topk_layers | None, store_idx | None, new_committed)
            store_idx 尽量为绝对位置，供 finalize 按时间 cat。
        """
        # 空 stream / 本步没跑到该 stream 的 gate
        if idx is None or idx.numel() == 0 or not topk_layers:
            return None, None, committed

        b, t = idx.shape  # t = 本步该 stream 的 token 数 T
        # prefill: input_pos 形如 [0..L-1]；decode: 如 [100]（长度 1）
        prefill = input_pos is None or (
            input_pos.shape[1] > 1
            and int(input_pos[0, 0]) == 0
            and int(input_pos[0, -1]) == input_pos.shape[1] - 1
        )
        # 例3: prefill 且 committed==T → start=T → 下面直接 return
        # 例2: decode → start=0，整步都当新的
        start = committed if prefill else 0
        if start >= t:
            return None, None, committed
        if topk_layers[0].shape[0] != b * t:
            raise RuntimeError(
                f"router replay topk T={topk_layers[0].shape[0]} != B*T={b * t}"
            )

        # local: 新后缀的下标；有 input_pos 时 gather 成绝对 pos
        # 例2: local=[[0]] → store_idx=input_pos.gather → [[100]]
        local = idx[:, start:]
        if input_pos is not None:
            store_idx = input_pos.gather(1, local.to(device=input_pos.device)).detach().clone()
        else:
            store_idx = local.detach().clone()

        # topk 与 T 对齐后切 [start:]；例4: [B,90,K][:, 81:] → 9 个新 token
        if start == 0:
            new_topk = topk_layers
        else:
            new_topk = [
                tk.view(b, t, -1)[:, start:].reshape(b * (t - start), -1)
                for tk in topk_layers
            ]
        # 例1 prefill → committed=T；例2 decode → committed+1
        return new_topk, store_idx, (t if prefill else committed + t)

    def record_router_replay_step(self, *args, **kwargs):
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay

        if getattr(self, "infer_mode", None) != "text":
            return None

        self._ensure_router_replay_state()
        und_idx = kwargs.get("und_token_indices")
        gen_idx = kwargs.get("gen_token_indices")
        und_raw = [] if und_idx is not None and und_idx.numel() > 0 else None
        gen_raw = [] if gen_idx is not None and gen_idx.numel() > 0 else None

        for fqn, module in self.named_modules():
            rr = getattr(module, "router_replay", None)
            if not isinstance(rr, RouterReplay):
                continue
            recorded = rr.get_recorded_indices()
            bucket = gen_raw if "mlp_mot_gen" in fqn else und_raw
            if bucket is None or recorded is None:
                continue
            bucket.append(recorded.detach().clone())

        if und_raw is not None and not und_raw:
            und_raw = None
        if gen_raw is not None and not gen_raw:
            gen_raw = None

        input_pos = kwargs.get("input_pos")

        old_und_committed = self._router_replay_und_committed
        old_gen_committed = self._router_replay_gen_committed


        # 里面根据用了哪些 kv cache, 根据 und 和 gen 的 idx, 只把新 token 的 topk_idx 返回， 并更新 committed 指针
        und_topk, und_store, self._router_replay_und_committed = self._delta_mot_stream(
            und_idx, und_raw, self._router_replay_und_committed, input_pos=input_pos,
        )
        gen_topk, gen_store, self._router_replay_gen_committed = self._delta_mot_stream(
            gen_idx, gen_raw, self._router_replay_gen_committed, input_pos=input_pos,
        )
        if und_topk is None and gen_topk is None:
            return None

        new_und_committed = self._router_replay_und_committed
        new_gen_committed = self._router_replay_gen_committed
        und_committed_delta = new_und_committed - old_und_committed
        gen_committed_delta = new_gen_committed - old_gen_committed
        if und_committed_delta > 1 or gen_committed_delta > 1:
            trace_log(
                f'[Rank {os.getenv("RANK", "0")}]: '
                f"[record_router_replay_step] commit {und_committed_delta} und ({old_und_committed} -> {new_und_committed}) | {gen_committed_delta} gen ({old_gen_committed} -> {new_gen_committed})"
            )

        step = self.MotRouterReplayStep(und_topk, gen_topk, und_store, gen_store)
        self.router_replay_captures.append(step)
        return step

    def finalize_router_replay_captures(self):
        """Cat und/gen captures；缺 stream 的层在 flat 里为 None（set_replay 时自由 routing）。"""
        from hy_parallelism.models.modules.moe.routers.router_replay import RouterReplay
        from hy_parallelism.common.logging import trace_log
        import os

        self._ensure_router_replay_state()
        if not self.router_replay_captures:
            raise RuntimeError("finalize_router_replay_captures: no captures")

        und_topk, und_idx = self._cat_mot_stream(
            [s for s in self.router_replay_captures if s.und_topk_idx is not None],
            "und_topk_idx", "und_token_indices",
        )
        gen_topk, gen_idx = self._cat_mot_stream(
            [s for s in self.router_replay_captures if s.gen_topk_idx is not None],
            "gen_topk_idx", "gen_token_indices",
        )

        und_idx_shape = None if und_idx is None else tuple(und_idx.shape)
        gen_idx_shape = None if gen_idx is None else tuple(gen_idx.shape)
        und_topk0_T = None if not und_topk else int(und_topk[0].shape[0])
        gen_topk0_T = None if not gen_topk else int(gen_topk[0].shape[0])
        trace_log(
            f'[Rank {os.getenv("RANK", "0")}]: '
            f"[finalize_router_replay_captures] steps={len(self.router_replay_captures)} "
            f"und_idx_shape={und_idx_shape} und_topk0_T={und_topk0_T} "
            f"gen_idx_shape={gen_idx_shape} gen_topk0_T={gen_topk0_T}"
        )

        und_q, gen_q = list(und_topk or []), list(gen_topk or [])
        flat, is_gen = [], []
        for fqn, module in self.named_modules():
            if not isinstance(getattr(module, "router_replay", None), RouterReplay):
                continue
            # 记下 MOT stream，供 set_replay 选 und_len/gen_len
            gen = "mlp_mot_gen" in fqn
            q = gen_q if gen else und_q
            flat.append(q.pop(0) if q else None)
            is_gen.append(gen)

        out = self.MotRouterReplayAssembled(flat, und_idx, gen_idx, is_gen)
        self.router_replay_assembled = out
        return out

    # ================================================================

class Image35GenerationMixin(Image35RouterReplayMixin):
    min_text_forward_cnt = 5

    def set_drop_token(self, val):
        for module in self.modules():
            if hasattr(module, "moe_drop_token_enabled"):
                module.moe_drop_token_enabled = val
            if hasattr(module, "set_moe_compute_capacity_rate"):
                module.set_moe_compute_capacity_rate(val)

    def set_und_moe_no_ep_inference(self, enabled: bool):
        self._und_moe_no_ep_inference = enabled

    def set_und_moe_flashinfer_inference(self, enabled: bool):
        self._und_moe_flashinfer_inference = enabled

    @property
    def und_moe_flashinfer_enabled(self) -> bool:
        if cuda_graph.enabled():
            # num_global_sum_tokens_per_local_expert = expert_mask.sum(dim=(1, 2))
            # RuntimeError: CUDA error: operation not permitted when stream is capturing
            return True
        return os.getenv("UND_MOE_FLASHINFER_ENABLED", "1") == "1"
        if getattr(self, "_und_moe_flashinfer_inference", None) is not None:
            return self._und_moe_flashinfer_inference
        return getattr(getattr(self, "args", None), "und_moe_flashinfer_inference", False)

    @property
    def und_moe_no_ep_inference_enabled(self) -> bool:
        return os.getenv("UND_MOE_NO_EP_INFERENCE_ENABLED", "1") == "1"
        if getattr(self, "_und_moe_no_ep_inference", None) is not None:
            return self._und_moe_no_ep_inference
        return getattr(getattr(self, "args", None), "und_moe_no_ep_inference", False)

    @property
    def und_moe_gather_done(self) -> bool:
        return getattr(self, "_und_moe_gather_done", False)

    def ensure_und_moe_no_ep_mesh(self) -> str:
        from hy_parallelism.parallel_states import get_or_init_parallel_state, get_parallel_state

        p_state = get_parallel_state()
        get_or_init_parallel_state(
            dp_replicate=p_state.dp_replicate,
            dp_shard=p_state.dp_shard,
            expert_dp_shard=p_state.expert_dp_shard,
            cp=p_state.cp,
            tp=p_state.tp,
            pp=p_state.pp,
            ep=1,
            etp=1,
            enable_expert_fsdp_sharding=p_state.enable_expert_fsdp_sharding,
            world_size=p_state.world_size,
            mesh_tag=UND_MOE_NO_EP_MESH_TAG,
        )
        return UND_MOE_NO_EP_MESH_TAG

    def _iter_und_moe_modules(self):
        from hy_parallelism.engines.parallel_engine import BaseParallelEngine
        from hymm.models.basic.moe_layers import ExpertParallelMoE

        for fqn, module in BaseParallelEngine.recursive_module_generator_buttom_up(
            None, self.model, return_name=True
        ):
            if not isinstance(module, ExpertParallelMoE):
                continue
            if not _is_und_moe_fqn(fqn):
                continue
            yield fqn, module

    def gather_und_moe_for_no_ep_inference(self, ep_group=None):
        from hy_parallelism.parallel_states import device_mesh_context, get_parallel_state
        from hymm.models.basic.moe_layers import ExpertParallelMoE

        if self.und_moe_gather_done:
            return

        if ep_group is None:
            p_state = get_parallel_state()
            if p_state.ep_size <= 1 and not self.und_moe_flashinfer_enabled:
                return
            ep_group = p_state.ep_group
        elif dist.get_world_size(ep_group) <= 1 and not self.und_moe_flashinfer_enabled:
            return

        mesh_tag = self.ensure_und_moe_no_ep_mesh()
        # self.unshard()
        # 吗的，2.7.1 的 copyout 一直不释放，相当于双倍MOE内存占用。。
        # 这里手动释放
        for name, module in self.named_modules():
            if name.endswith('experts'):
                if hasattr(module, 'reshard'):
                    module.reshard()
        if self.und_moe_cpu_offload_sharded_params:
            from hy_parallelism.distributed.fsdp_util import (
                iter_sharded_param_tensors,
                move_sharded_param,
            )
            for fsdp_param in iter_sharded_param_tensors(self.model):
                move_sharded_param(fsdp_param, "cpu")

        if not hasattr(self, "und_moe_gather_state"):
            self.und_moe_gather_state = {}

        use_flashinfer = self.und_moe_flashinfer_enabled
        for fqn, module in self._iter_und_moe_modules():
            if id(module) in self.und_moe_gather_state:
                continue
            if not module.fused_expert:
                raise NotImplementedError(
                    f"und MoE no-EP inference only supports fused expert, got {fqn=}"
                )

            experts_module = module.experts
            if isinstance(experts_module, FSDPModule):
                experts_module.unshard()

            gathered_weights = {}
            device = None
            for name in GatheredHunyuanFusedExpert.EXPERT_WEIGHT_NAMES:
                local_tensor = _get_unsharded_param_tensor(getattr(experts_module, name))
                gathered = _all_gather_expert_tensor(local_tensor)
                gathered_weights[name] = gathered
                device = gathered.device

            gate_proj_weights = gathered_weights.pop("gate_proj_weights")
            up_proj_weights = gathered_weights.pop("up_proj_weights")
            down_proj_weights = gathered_weights.pop("down_proj_weights")
            if use_flashinfer:
                gathered_experts = GatheredHunyuanFusedExpert.from_gathered_weights(
                    gate_proj_weights,
                    up_proj_weights,
                    down_proj_weights,
                    module.num_experts,
                )
                gathered_weights = {
                    "expert_gate_and_up_weights": gathered_experts.expert_gate_and_up_weights,
                    "expert_down_weights": gathered_experts.expert_down_weights,
                }
            else:
                gathered_experts = GatheredHunyuanFusedExpert(
                    gate_proj_weights,
                    up_proj_weights,
                    down_proj_weights,
                    module.num_experts,
                    use_flashinfer=False,
                )
                gathered_weights = {
                    "gate_proj_weights": gate_proj_weights,
                    "up_proj_weights": up_proj_weights,
                    "down_proj_weights": down_proj_weights,
                }
            proxy = GatheredUndMoEProxy(module, gathered_experts)
            original_forward = module.forward

            def make_forward(moe_proxy, no_ep_mesh_tag, flashinfer_mode):
                def forward(hidden_states):
                    with device_mesh_context(no_ep_mesh_tag):
                        if flashinfer_mode:
                            return moe_proxy.forward_non_ep_flashinfer(hidden_states)
                        return ExpertParallelMoE.forward_non_ep(moe_proxy, hidden_states)

                return forward

            module.forward = make_forward(proxy, mesh_tag, use_flashinfer)

            fsdp_experts = experts_module if isinstance(experts_module, FSDPModule) else None
            if fsdp_experts is not None:
                fsdp_experts.cpu()

            self.und_moe_gather_state[id(module)] = {
                "module": module,
                "original_forward": original_forward,
                "gathered_weights": gathered_weights,
                "fsdp_experts": fsdp_experts,
                "device": device,
                "use_flashinfer": use_flashinfer,
            }
        if dist.get_rank() == 0:
            trace_log(f"gathered und MoE expert params on EP group for no-EP inference, "
                f"num_layers={len(self.und_moe_gather_state)}, "
                f"use_flashinfer={use_flashinfer}")

        self._und_moe_gather_done = True

    def maybe_gather_und_moe_once_for_decode(self):
        """Gather und MoE expert weights once, right after prefill (before first decode step)."""
        if self.und_moe_gather_done:
            return
        if not self.und_moe_no_ep_inference_enabled:
            return
        p_state = get_parallel_state()
        if p_state.ep_size <= 1 and not self.und_moe_flashinfer_enabled:
            return

        ep_group = p_state.ep_group
        self.gather_und_moe_for_no_ep_inference(ep_group=ep_group)

    def restore_und_moe_from_no_ep_inference(self):
        gather_state = getattr(self, "und_moe_gather_state", None)
        if not gather_state:
            return

        for state in gather_state.values():
            module = state["module"]
            module.forward = state["original_forward"]
            fsdp_experts = state["fsdp_experts"]
            state["gathered_weights"].clear()
            if fsdp_experts is not None:
                fsdp_experts.to(state["device"])

        if self.und_moe_cpu_offload_sharded_params:
            from hy_parallelism.distributed.fsdp_util import (
                iter_sharded_param_tensors,
                move_sharded_param,
            )
            device = torch.device("cuda", torch.cuda.current_device())
            for fsdp_param in iter_sharded_param_tensors(self.model):
                move_sharded_param(fsdp_param, device)

        gather_state.clear()
        self._und_moe_gather_done = False
        self.reshard()

        if dist.get_rank() == 0:
            loguru.logger.info("restored und MoE expert params from no-EP inference mode")

    def get_prefetch_manager(self):
        if not hasattr(self, '_prefetch_manager'):
            self._prefetch_manager = PrefetchManager(self)
        return self._prefetch_manager


    def reshard(self):
        from hy_parallelism.engines.parallel_engine import BaseParallelEngine
        for name, module in BaseParallelEngine.recursive_module_generator_buttom_up(None, self.model, return_name=True):
            if hasattr(module, 'reshard'):
                module.reshard()

    def unshard(self):
        from hy_parallelism.engines.parallel_engine import BaseParallelEngine
        for name, module in BaseParallelEngine.recursive_module_generator_buttom_up(None, self.model, return_name=True):
            if hasattr(module, 'unshard'):
                module.unshard()

    def to_text_generation_mode(self):
        from hy_parallelism.context_parallel.core import set_disable_cp_ops
        from hy_parallelism.tools.profiling import pause_memory_snapshot
        trace_log("Switching to text generation mode")

        # Each text session owns its own gathered weights; drop any leftover from a prior session.
        self.restore_und_moe_from_no_ep_inference()
        cuda_graph.reset_decode_cuda_graphs(self)
        self.infer_mode = 'text'
        self.text_forward_cnt = 0
        self._und_moe_gather_done = False

        if dist.get_rank() == 0:
            loguru.logger.info("Switching to text generation mode")

        self.set_drop_token(False)
        set_disable_cp_ops(True)
        # pause_memory_snapshot()
        os.environ['SKIP_CHECKPOINTING'] = '1'
        self.get_prefetch_manager().skip_prefetch()

        # if get_parallel_state().ep_size > 1:
            # TODO: 其實唔應該放喺呢度設置，因為prefetch可能導致gen都unshard
        self.reset_prefetch_plan(prefetch_gen=False, prefetch_und=True)


        if hasattr(self, 'compiled_funcs') and len(self.compiled_funcs) > 0:
            for module in self.modules():
                if id(module) in self.compiled_funcs:
                    module._compiled_call_impl = self.compiled_funcs[id(module)]
            self.compiled_funcs.clear()

        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.attn_mode = 'sdpa'

        gc.collect()
        torch.cuda.empty_cache()

    def to_image_or_training_mode(self):
        from hy_parallelism.context_parallel.core import set_disable_cp_ops
        from hy_parallelism.models.modules.moe.moe_parallel_deepep import (
            version as deepep_version,
            set_low_latency,
        )
        from hy_parallelism.tools.profiling import start_memory_snapshot
        from hymm.engines.hunyuan_multimodal_engine import HunyuanMultimodalEngine

        trace_log("Switching to image mode")
        if self.text_forward_cnt < self.min_text_forward_cnt:
            msg = f"WARN: text forward cnt {self.text_forward_cnt} < {self.min_text_forward_cnt}"
            trace_log(msg)
            raise RuntimeError(msg)

        self.infer_mode = 'image'
        is_rank0 = dist.get_rank() == 0

        if is_rank0:
            loguru.logger.info("Switching to image or training mode")

        self.restore_und_moe_from_no_ep_inference()
        cuda_graph.reset_decode_cuda_graphs(self)
        self.set_drop_token(getattr(self.args, 'drop_token_training', False))
        # start_memory_snapshot()
        self.set_reshard_after_forward(True, recurse=True)
        self.get_prefetch_manager().restore_prefetch()
        self.reset_prefetch_plan(prefetch_gen=True, prefetch_und=False)
        for name, module in self.named_modules():
            if hasattr(module, 'reshard'):
                module.reshard()
        os.environ['SKIP_CHECKPOINTING'] = '0'
        self.enable_skip_cache = False

        set_disable_cp_ops(False)
        if deepep_version == 'v2':
            set_low_latency(False)

        if is_rank0:
            loguru.logger.info(f'text forward cnt: {self.text_forward_cnt}')

        self.compiled_funcs = {}
        for module in self.modules():
            if hasattr(module, '_compiled_call_impl'):
                self.compiled_funcs[id(module)] = module._compiled_call_impl
                module._compiled_call_impl = None
            if isinstance(module, CausalSelfAttention):
                module.attn_mode = 'flash2'
        trace_log(f"To image mode. Last text forward cnt: {self.text_forward_cnt}")
        self.text_forward_cnt = 0

        gc.collect()
        torch.cuda.empty_cache()



    def generate(self, *args, **kwargs):
        self.to_text_generation_mode()

        # self.start_router_recording()
        ret = super().generate(*args, **kwargs)
        # self.stop_router_recording()

        self.to_image_or_training_mode()
        return ret

    def reset_prefetch_plan(self, prefetch_gen, prefetch_und):
        from hy_parallelism.distributed.fsdp_util import set_forward_prefetch

        def should_forward_prefetch(fqn, module):
            if "mlp_mot_gen" in fqn:
                return prefetch_gen
            if ".mlp." in fqn:
                return prefetch_und
            return True

        prefetch_factor = getattr(getattr(self, "args", None), "prefetch_factor", 1)
        set_forward_prefetch(
            self,
            self.model.layers,
            prefetch_factor=prefetch_factor,
            should_forward_prefetch=should_forward_prefetch,
        )

    def set_replay(self, assembled_replay: Image35RouterReplayMixin.MotRouterReplayAssembled, model_kwargs: dict):
        und = model_kwargs.get("und_token_indices")
        gen = model_kwargs.get("gen_token_indices")
        und_len = None if und is None else und.shape[1]
        gen_len = None if gen is None else gen.shape[1]
        expand_batch = 1
        if und is not None and assembled_replay.und_token_indices is not None:
            expand_batch = und.shape[0] // assembled_replay.und_token_indices.shape[0]
        elif gen is not None and assembled_replay.gen_token_indices is not None:
            expand_batch = gen.shape[0] // assembled_replay.gen_token_indices.shape[0]
        self.router_replay = assembled_replay.set_replay(
            name=self._router_replay_group,
            expand_batch=expand_batch,
            und_len=und_len,
            gen_len=gen_len,
        )

    def __call__(self, *args, **kwargs):
        from hy_parallelism.models.modules.moe.moe_parallel_deepep import version as deepep_version
        from hy_parallelism.models.modules.moe.moe_parallel_deepep import set_low_latency
        self.text_forward_cnt = getattr(self, 'text_forward_cnt', 0) + 1
        infer_mode = getattr(self, 'infer_mode', None)

        if infer_mode == 'text':
            print_time = random.random() < 10 / 1000
            if self.text_forward_cnt <= 10:
                print_time = True
        else:
            print_time = random.random() < 10 / 30
            print_time = True
            barrier_start = time.time()
            dist.barrier() # 确保耗时统计准确
            barrier_end = time.time()
            if print_time and os.getenv('LOCAL_RANK', '0') == '0':
                if barrier_end - barrier_start > 10:
                    trace_log(f"barrier time: {barrier_end - barrier_start} s")
        start_time = time.time()
        # if infer_mode == 'image':
        # start_profiling(ProfilingConfig(profile_freq=4, profiler_warmup=0, profiler_active=4, save_traces_folder='profile_traces/cuda_graph'))
        # profiler_step(skip_barrier=True)

        und_token_indices = kwargs.get("und_token_indices")
        gen_token_indices = kwargs.get("gen_token_indices")
        force_trace = (
            (und_token_indices is not None and und_token_indices.shape[1] > 20480)
            or (gen_token_indices is not None and gen_token_indices.shape[1] > 20480)
        )
        if force_trace:
            print_time = True

        if print_time:
            msg = f'Forwarding: fwd_cnt={self.text_forward_cnt} | infer_mode={getattr(self, "infer_mode", None)} | training: {self.training}'
            if und_token_indices is not None:
                msg += f" und_seqlen={und_token_indices.shape[1]}"
            if gen_token_indices is not None:
                msg += f" gen_seqlen={gen_token_indices.shape[1]}"
            attention_mask = kwargs.get("attention_mask")
            if hasattr(attention_mask, 'attention_mask'): # Offloaded Attention Mask
                attention_mask = attention_mask.attention_mask
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                msg += f" attention_mask shape: {attention_mask.shape}"
                # trace_log(f"attention_mask shape: {attention_mask.shape}")
            trace_log(msg)

        # Pipeline denoise: replay CoT routing only on the first image forward (KV prefill).
        # Training re-installs via MotRouterReplayAssembled.set_replay (grad enabled → skip here).
        if infer_mode == 'image' and not torch.is_grad_enabled() and getattr(self, 'replay_on_diffusion_prefill', True):
            assembled = getattr(self, 'router_replay_assembled', None)
            if assembled is not None:
                if self.text_forward_cnt == 1:
                    trace_log("Setting router replay for first diffusion prefill")
                    self.set_replay(assembled, kwargs)
                elif und_token_indices.shape[1] == 0:
                    trace_log("Clearing router replay for second diffusion step")
                    self.clear_router_replay_runtime()


        if self.text_forward_cnt > 2 and infer_mode == 'text':
            # 寫成2系為咗保證 reshard_after_forward 設置成功
            # und_moe_gather_done 可以正確 gather
            from hy_parallelism.parallel_states import device_mesh_context

            self.maybe_gather_und_moe_once_for_decode()

            # prefill 階段唔可以 inference_mode, 因為需要 fsdp unshard
            # 所以上面寫 > 2
            # After CUDA graph capture, replay never enters nested FSDP modules;
            # skip the all-module _call_impl patch to save Python overhead.
            forward_ctx = [torch.inference_mode()]
            runner = getattr(self, "_decode_cuda_graph_runner", None)
            if not (cuda_graph.enabled() and runner is not None and runner._captured):
                forward_ctx.append(call_forward_directly(self))
            if cuda_graph.enabled():
                forward_ctx.append(cuda_graph.use_decode_cuda_graph(self))
            if self.und_moe_gather_done:
                forward_ctx.append(device_mesh_context(self.ensure_und_moe_no_ep_mesh()))

            with contextlib.ExitStack() as stack:
                for ctx in forward_ctx:
                    stack.enter_context(ctx)
                ret = super().__call__(*args, **kwargs)
            if isinstance(ret, torch.Tensor):
                ret = ret.clone()
        else:
            ret = super().__call__(*args, **kwargs)
        if print_time:
            torch.cuda.synchronize()
            msg = f"step time: {time.time() - start_time} s ({infer_mode=}). nosync"
            trace_log(msg)

        if self.text_forward_cnt == 1 and infer_mode == 'text':
            # prefill 完，可以放心設置
            # self.set_reshard_after_forward(False, recurse=True)
            for name, module in self.named_modules():
                if not hasattr(module, "set_reshard_after_forward"):
                    continue
                # module.set_reshard_after_forward("mlp_mot_gen" in name, recurse=False)
                if 'experts' in name:
                    # 橫掂 moe 都要都要重新 gather 替換成 flashinfer
                    module.set_reshard_after_forward(True, recurse=False)
                else:
                    module.set_reshard_after_forward(False, recurse=False)


        # Defer setting low latency mode. Assumming prefill only once
        if self.text_forward_cnt == 2 and infer_mode == 'text':
            self.enable_skip_cache = True
            # os.environ['HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM'] = '0'
            if deepep_version == 'v2':
                set_low_latency(True)
        return ret

    @cache
    def get_sync_groups(self):
        from hy_parallelism.parallel_states import get_parallel_state
        groups = []
        for key, group in get_parallel_state().should_sync_groups_dict().items():
            if 'fsdp' in key:
                continue
            groups.append(group)
        return groups

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer = None,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        from transformers.generation.utils import logger, GenerateNonBeamOutput, GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
        from torch import nn
        import transformers

        assert transformers.__version__ == "4.57.1", (
            f"Requires transformers==4.57.1, got {transformers.__version__}"
        )
        assert not synced_gpus

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        router_replay_paused = False
        while True:
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            # 因为历史遗留原因和显存原因，现在升文必须 Forward 3次以上（第三次会触发 unshard 通信)
            # 如果部分rank只 decode 一个token就结束了，会通信对不齐，所以下方强制5次内不能停
            # 但为了避免 eos 之后多 concat 无用 token，有了下面这个 if
            # 之所以有個 if, 系避免無謂嘅 cpu sync
            if self.text_forward_cnt < self.min_text_forward_cnt:
                if this_peer_finished:
                    # 本步已 capture 到 eos routing，保留；后续 sync-only 不再记
                    # 否则会有可能多记几个 routing topk_idx 到 replay 中，导致 replay T 长于 Forward
                    if getattr(self, "_router_replay_capturing", False):
                        self._router_replay_capturing = False
                        router_replay_paused = True
                    # keep forwarding for cnt/sync, but do not sample/pad/cat (same as HF synced continue)
                    model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    if not self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                        break
                    continue
            if not (synced_gpus and this_peer_finished):
                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            # Prefetch next-step inputs before the sync in _has_unfinished_sequences so most
            # prepare_inputs_for_generation ops can be queued on CUDA first.
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if not self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                break

        if router_replay_paused:
            self._router_replay_capturing = True

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids



    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        # 避免prefill直接 eos
        if self.text_forward_cnt < self.min_text_forward_cnt:
            return True
        DECODE_SYNC_FREQ = 20 if not cuda_graph.enabled() else 60
        if synced_gpus:
            if self.text_forward_cnt % DECODE_SYNC_FREQ == 0:
                return super()._has_unfinished_sequences(this_peer_finished, synced_gpus, device)
            return True
        return super()._has_unfinished_sequences(this_peer_finished, synced_gpus, device)
        # if synced_gpus:
        #     # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
        #     # The following logic allows an early break if all peers finished generating their sequence
        #     this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=device)
        #     # send 0.0 if we finished, 1.0 otherwise
        #     for group in self.get_sync_groups():
        #         dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM, group=group)
        #     # did all peers finish? the reduced sum will be 0.0 then
        #     if this_peer_finished_flag.item() == 0.0:
        #         return False
        # elif this_peer_finished:
        #     return False
        # return True
