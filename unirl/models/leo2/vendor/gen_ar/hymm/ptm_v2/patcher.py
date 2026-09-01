from contextlib import nullcontext
from functools import wraps
from time import time
from logging import getLogger
from torch.utils.checkpoint import detach_variable
from typing import List, Optional, Union, Callable, Tuple, Dict, Any
import logging
import torch

from torch.autograd.variable import Variable
from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import _get_all_rng_states, split_tensor_into_1d_equal_chunks, gather_split_1d_tensor, _fork_rng, _set_all_rng_states
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, _KEEP_ORIGINAL_PRECISION_ATTR
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
    get_transformer_layer_offset
)
from megatron.core.ftext_utils import ftext_start_span, ftext_end_span
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor, safely_set_viewless_tensor_data, get_attr_wrapped_model, log_single_rank
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, get_updated_expert_bias
from angelptm.toolkits.multimodal_gen.bitwise_align.ptm_patcher import bitwise_align_patchers
from angelptm.toolkits.multimodal_gen.bitwise_align.utils import is_bitwise_align_mode
from angelptm.toolkits.patch import PatchesManager
from hymm.utils.helpers import multi_pattern_match
from megatron.training.global_vars import get_args, get_timers, get_tensorboard_writer, get_wandb_writer
from megatron.training.utils import is_rank0

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm

try:
    import nvidia_kitchen  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

logger = getLogger(__name__)
_SPECIAL_LR_LOG_STATE = {}
_BASE_LR_LOG_STATE = {}

# Import hymm global_vars for accessing scalar_state
try:
    import hymm.core.global_vars as hymm_global_vars
    HAS_HYMM_GLOBAL_VARS = True
except ImportError:
    HAS_HYMM_GLOBAL_VARS = False

# Import AngelPTM cache-aware checkpoint
from angelptm.megatron.core.models.gemini.cache_aware_checkpoint import (
    cache_aware_checkpoint,
    argv_has_boolean_flag,
)
from angelptm.megatron.core.models.gemini.attn_cache_checkpoint import (
    HAS_ATTN_CACHE_CHECKPOINT,
    _install_flex_attention_cache_patch,
    _install_magi_attention_cache_patch,
)
from angelptm.megatron.core.models.gemini.deepep_cache_checkpoint import (
    HAS_DEEPEP_CACHE_CHECKPOINT,
    _install_deepep_cache_patch,
)

def forward_step_calc_loss_wrapper(fn):
    # scale the aux loss when cp is enabled, as the aux_loss is attched to the forward activation
    # so we need to scale it manually
    # each time we call forward_step_calc_loss, the MTPLossAutoScaler will set loss scale,
    # so we can scale it use its main_loss_backward_scale
    # the root reason why we need this, is the loss is scaled by cp size in forward_step_calc_loss
    @wraps(fn)
    def wrapper(model, output_tensor, *args, **kwargs):
        vp_stage = args[2]
        # Handle tuple output_tensor (hidden_states, ut) from non-post-process stages
        # Extract just the tensor for forward_step_calc_loss processing

        args_obj = get_args()
        if args_obj.moe_aux_loss and args_obj.context_parallel_size > 1: 
            MoEAuxLossAutoScaler.set_loss_scale(MoEAuxLossAutoScaler.main_loss_backward_scale * args_obj.context_parallel_size)

        if isinstance(output_tensor, (tuple, list)) and len(output_tensor) > 1:
            output_tensor_only = output_tensor[0]  # Get hidden_states only
            extra_tensors = output_tensor[1:]
            result_tensor, num_tokens = fn(model, output_tensor_only, *args, **kwargs)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
                return ([result_tensor, *extra_tensors]), num_tokens
            else:
                return result_tensor, num_tokens
        else:
            result_tensor, num_tokens = fn(model, output_tensor, *args, **kwargs)
        
        return result_tensor, num_tokens
    return wrapper


class CheckpointFunction(torch.autograd.Function):
    """Checkpoint Function

    This function is adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """

    # pylint: disable=missing-function-docstring
    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        """Forward pass."""
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.rng_states = _get_all_rng_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)
        return outputs

    # pylint: disable=missing-function-docstring
    @staticmethod
    def backward(ctx, *args):
        """Backward pass."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]
        inputs = tuple(inputs)
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        with _fork_rng():
            # Set the states to what it used to be before the forward pass.
            _set_all_rng_states(*ctx.rng_states)

            # Compute the forward pass.
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args))
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
        return (None, None) + grads

# wrapper to record loading checkpoint timestamp
def load_checkpoint_wrapper(fn):
    def wrapper(*args, **kwargs) -> None:
        if is_rank0():
            logger.info(f"begin checkpoint loading")
            start_load = time()

        iteration, num_floating_point_operations_so_far = fn(*args, **kwargs)

        if is_rank0():
            end_load = time()
            logger.info(f"finish checkpoint loading step {iteration}, took {(end_load - start_load):.3f} s")
        return iteration, num_floating_point_operations_so_far

    return wrapper

# wrapper to record saving checkpoint timestamp
def save_checkpoint_wrapper(fn):
    def wrapper(iteration, *args, **kwargs) -> None:
        if is_rank0():
            logger.info(f"begin checkpoint saving step {int(iteration):7d}")
            start_save = time()

        fn(iteration, *args, **kwargs)

        if is_rank0():
            end_save = time()
            logger.info(f"finish checkpoint saving step {int(iteration):7d}, took {(end_save - start_save):.3f} s")
        return

    return wrapper

def transformer_block_forward(
    self,
    hidden_states: Union[torch.Tensor, WrappedTensor],
    attention_mask: Optional[torch.Tensor],
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    rotary_pos_cos: Optional[torch.Tensor] = None,
    rotary_pos_sin: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[torch.Tensor] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    **kwargs
):
    """
    Perform the forward pass through the transformer block.

    This method handles the core computation of the transformer, including
    self-attention, optional cross-attention, and feed-forward operations.

    Args:
        hidden_states (Union[torch.Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
            where s is the sequence length, b is the batch size, and h is the hidden size.
            Can be passed as a WrappedTensor during inference to avoid an obsolete
            reference in the calling function.
        attention_mask (torch.Tensor): Boolean tensor of shape [1, 1, s, s] for masking
            self-attention.
        context (torch.Tensor, optional): Context tensor for cross-attention.
        context_mask (torch.Tensor, optional): Mask for cross-attention context
        rotary_pos_emb (torch.Tensor, optional): Rotary positional embeddings.
        attention_bias (torch.Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
            to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
            Used as an alternative to apply attention mask for TE cuDNN attention.
        inference_context (BaseInferenceContext, optional): Parameters for inference-time
            optimizations.
        packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
            processing.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output hidden states tensor of shape
        [s, b, h], and optionally the updated context tensor if cross-attention is used.
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    # Delete the obsolete reference to the initial input tensor if necessary
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()

    # Extract additional parameters for pipeline parallelism
    ut = kwargs.pop('ut', None)
    images = kwargs.pop('images', None)
    image_h_w = kwargs.pop('image_h_w', None)
    timesteps = kwargs.pop('timesteps', None)
    und_token_indices = kwargs.pop('und_token_indices', None)
    gen_token_indices = kwargs.pop('gen_token_indices', None)
    gen_hidden_states = kwargs.pop('gen_hidden_states', None)
    und_packed_seq_params = kwargs.pop('und_packed_seq_params', None)
    gen_packed_seq_params = kwargs.pop('gen_packed_seq_params', None)

    if not self.pre_process:
        # See set_input_tensor()
        if getattr(get_args(), 'use_mot', False):
            hidden_states, ut, images, image_h_w, timesteps, gen_hidden_states = self.input_tensor
            gen_hidden_states = gen_hidden_states.to(torch.bfloat16)
        else:
            hidden_states, ut, images, image_h_w, timesteps = self.input_tensor
        hidden_states = hidden_states.to(torch.bfloat16)

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
    if getattr(get_args(), 'use_mot', False):
        gen_hidden_states = make_viewless_tensor(inp=gen_hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
    # otherwise do nothing extra at the outer level
    # if we are using other fp8 recipes, then the context manager enter&exit are free
    # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
    # control which layer will be fp8 or bf16
    use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
    use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
    outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

    with rng_context, outer_fp8_context:
        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            mot_ckpt_kwargs = {}
            if getattr(get_args(), 'use_mot', False):
                mot_ckpt_kwargs = dict(
                    und_token_indices=und_token_indices,
                    gen_token_indices=gen_token_indices,
                    gen_hidden_states=gen_hidden_states,
                    und_packed_seq_params=und_packed_seq_params,
                    gen_packed_seq_params=gen_packed_seq_params,
                )
            hidden_states, gen_hidden_states = self._checkpointed_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_inner_fp8_context=use_inner_fp8_context,
                **mot_ckpt_kwargs,
            )
        else:
            use_mot = getattr(get_args(), 'use_mot', False)
            for l_no, layer in enumerate(self.layers):
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                mot_kwargs = {}
                if use_mot:
                    mot_kwargs = dict(
                        und_token_indices=und_token_indices,
                        gen_token_indices=gen_token_indices,
                        gen_hidden_states=gen_hidden_states,
                        und_packed_seq_params=und_packed_seq_params,
                        gen_packed_seq_params=gen_packed_seq_params,
                    )
                with self.offload_context, inner_fp8_context:
                    layer_output = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        **mot_kwargs,
                    )
                    if use_mot:
                        hidden_states, gen_hidden_states, context = layer_output
                    else:
                        hidden_states, context = layer_output

                if (
                    torch.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    # Final layer norm.
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

    # If this TransformerBlock is empty, input and output hidden states will be the same node
    # on the computational graph and will lead to unexpected errors in pipeline schedules.
    if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
        hidden_states = hidden_states.clone()
    
    # 清除input_tenosr的引用，防止vpp下缓存过多input_tensor
    self.input_tensor = None
    if getattr(get_args(), 'use_mot', False):
        return hidden_states.contiguous(), ut, images, image_h_w, timesteps, gen_hidden_states
    else:
        return hidden_states.contiguous(), ut, images, image_h_w, timesteps

def transformer_block_checkpointed_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    attention_bias: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    use_inner_fp8_context: bool,
    und_token_indices: torch.Tensor = None,
    gen_token_indices: torch.Tensor = None,
    gen_hidden_states: torch.Tensor = None,
    und_packed_seq_params=None,
    gen_packed_seq_params=None,
):
    """Forward method with activation checkpointing."""
    use_mot = getattr(get_args(), 'use_mot', False)

    def custom(start: int, end: int):
        if use_mot:
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                und_token_indices,
                gen_token_indices,
                gen_hidden_states,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with inner_fp8_context:
                        hidden_states, gen_hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                            und_token_indices=und_token_indices,
                            gen_token_indices=gen_token_indices,
                            gen_hidden_states=gen_hidden_states,
                            und_packed_seq_params=und_packed_seq_params,
                            gen_packed_seq_params=gen_packed_seq_params,
                        )
                return hidden_states, gen_hidden_states, context
        else:
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    inner_fp8_context = (
                        get_fp8_context(self.config, layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    with inner_fp8_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

        return custom_forward

    def checkpoint_handler(forward_func):
        """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
        if use_mot:
            ckpt_args = (
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                und_token_indices,
                gen_token_indices,
                gen_hidden_states,
            )
        else:
            ckpt_args = (
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )
        if self.config.fp8:
            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                *ckpt_args,
            )
        else:
            # Always route through the cache-aware reentrant checkpoint:
            # a registry-driven wrapper that enters each registered cache
            # module's capture/replay envelope around ``forward_func``
            # (attn cache ...). When NO cache module is enabled it short-
            # circuits to ``tensor_parallel.checkpoint`` so baseline runs
            # pay zero overhead.
            return cache_aware_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                *ckpt_args,
            )

    if self.config.recompute_method == 'uniform':
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            ckpt_result = checkpoint_handler(
                custom(layer_idx, layer_idx + self.config.recompute_num_layers)
            )
            if use_mot:
                hidden_states, gen_hidden_states, context = ckpt_result
            else:
                hidden_states, context = ckpt_result

            layer_idx += self.config.recompute_num_layers

    elif self.config.recompute_method == 'block':
        recompute_skip_num_layers = 0

        if parallel_state.get_pipeline_model_parallel_world_size() > 0 and \
                self.config.per_stage_recompute_num_layers is not None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            recompute_num_layers = self.config.per_stage_recompute_num_layers[pp_rank]
        else:
            recompute_num_layers = self.config.recompute_num_layers

        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if self.config.fp8 and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < recompute_num_layers + recompute_skip_num_layers
            ):
                ckpt_result = checkpoint_handler(custom(layer_idx, layer_idx + 1))
            else:
                if use_mot:
                    ckpt_result = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        und_token_indices,
                        gen_token_indices,
                        gen_hidden_states,
                    )
                else:
                    ckpt_result = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
            if use_mot:
                hidden_states, gen_hidden_states, context = ckpt_result
            else:
                hidden_states, context = ckpt_result
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states, gen_hidden_states

def transformer_block_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask, *args, **kwargs):
        # Extract additional parameters for pipeline parallelism
        ut = kwargs.pop('ut', None)
        images = kwargs.pop('images', None)
        image_h_w = kwargs.pop('image_h_w', None)
        timesteps = kwargs.pop('timesteps', None)

        # If not in pre_process stage, get these values from input_tensor
        original_input_tensor = None
        if not self.pre_process:
            # See set_input_tensor()
            original_input_tensor = self.input_tensor
            hidden_states, ut, images, image_h_w, timesteps = self.input_tensor
            hidden_states = hidden_states.to(torch.bfloat16)
            # Temporarily set input_tensor to hidden_states
            self.input_tensor = hidden_states

        try:
            # Call original forward function
            result = fn(self, hidden_states, attention_mask, *args, **kwargs)
        finally:
            # Restore original input_tensor
            if original_input_tensor is not None:
                self.input_tensor = original_input_tensor

        hidden_states = result
        ut = ut.contiguous() if ut is not None and isinstance(ut, torch.Tensor) else ut
        images = images.contiguous() if images is not None and isinstance(ut, torch.Tensor) else images
        image_h_w = image_h_w.contiguous() if image_h_w is not None and isinstance(ut, torch.Tensor) else image_h_w
        timesteps = timesteps.contiguous() if timesteps is not None and isinstance(ut, torch.Tensor) else timesteps

        return hidden_states.contiguous(), ut, images, image_h_w, timesteps

    return wrapper


def transformer_layer_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, *args, **kwargs):
        # Extract additional parameters used for MoT models
        und_token_indices = kwargs.pop('und_token_indices', None)
        gen_token_indices = kwargs.pop('gen_token_indices', None)
        
        # Call original forward function
        hidden_states = fn(self, hidden_states, *args, **kwargs)
        return hidden_states
    
    return wrapper

def forward_step_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs) -> bool:
        output_tensor, num_tokens = fn(*args, **kwargs)
        vp_stage = kwargs.get("vp_stage", None)
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            return output_tensor, num_tokens
        # Only unwrap when forward_step wrapped output in a single-element list.
        if isinstance(output_tensor, (list, tuple)) and len(output_tensor) == 1:
            return output_tensor[0], num_tokens
        return output_tensor, num_tokens

    return wrapper


def _is_leo_model_pipeline(model):
    """Return True iff the (unwrapped) model in this pipeline is ``LeoModel``.
    """
    m = model[0] if isinstance(model, list) else model
    try:
        _unwrapped = get_attr_wrapped_model(m, "build_schedule_plan", return_model_obj=True)
    except Exception:
        return False
    return type(_unwrapped).__name__ == "LeoModel"


def combined_1f1b_schedule_for_no_pipelining_wrapper(fn):
    """Route the PP=1 combined-1F1B schedule to leo when the model is LeoModel."""
    @wraps(fn)
    def wrapper(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        output_tensor_grad,
        forward_data_store,
        config,
        collect_non_loss_data,
        first_val_step,
        forward_only,
        no_sync_func,
        total_num_tokens,
        check_first_val_step,
    ):
        target = fn
        if _is_leo_model_pipeline(model):
            from angelptm.megatron.core.models.leo.ep_1f1b_overlap.leo_combined_1f1b import (
                leo_combined_1f1b_schedule_for_no_pipelining,
            )
            target = leo_combined_1f1b_schedule_for_no_pipelining
        return target(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            check_first_val_step,
        )

    return wrapper


def combined_1f1b_schedule_for_interleaved_pipelining_wrapper(fn):
    """Route the interleaved (VPP) combined-1F1B schedule to leo when the model is LeoModel."""
    @wraps(fn)
    def wrapper(
        config,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        forward_data_store,
        forward_step_helper_preprocess,
        forward_step_helper_postprocess,
        backward_step_helper_preprocess,
        backward_step_helper_postprocess,
        get_microbatch_id_in_model_chunk,
        get_model_chunk_id,
        check_first_val_step,
        is_first_microbatch_for_model_chunk,
        collect_non_loss_data,
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        pre_processed_batch=None,
    ):
        target = fn
        if _is_leo_model_pipeline(model):
            from angelptm.megatron.core.models.leo.ep_1f1b_overlap.leo_combined_1f1b import (
                leo_combined_1f1b_schedule_for_interleaved_pipelining,
            )
            target = leo_combined_1f1b_schedule_for_interleaved_pipelining
        return target(
            config,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            forward_data_store,
            forward_step_helper_preprocess,
            forward_step_helper_postprocess,
            backward_step_helper_preprocess,
            backward_step_helper_postprocess,
            get_microbatch_id_in_model_chunk,
            get_model_chunk_id,
            check_first_val_step,
            is_first_microbatch_for_model_chunk,
            collect_non_loss_data,
            f_virtual_microbatch_id=f_virtual_microbatch_id,
            b_virtual_microbatch_id=b_virtual_microbatch_id,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
            pre_processed_batch=pre_processed_batch,
        )

    return wrapper


def training_log_wrapper(fn):
    @wraps(fn)
    def wrapper(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
        num_floating_point_operations_in_batch,
        start_iteration,
    ):  
        args = get_args()
        timers = get_timers()
        if HAS_HYMM_GLOBAL_VARS:            
            try:
                writer = get_tensorboard_writer()
                wandb_writer = get_wandb_writer()
                scalar_state = hymm_global_vars.get_scalar_state()

                # Advanced, skipped, and Nan iterations.
                advanced_iters_key = 'advanced iterations'
                skipped_iters_key = 'skipped iterations'
                # Advanced iterations.
                if not skipped_iter:
                    advanced_iters = total_loss_dict.get(advanced_iters_key, 0) + 1
                else:
                    if advanced_iters_key not in total_loss_dict:
                        advanced_iters = 0
                # Skipped iterations.
                skipped_iters = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
                total_iterations = advanced_iters + total_loss_dict[skipped_iters_key]
                # Don't reset timers as it will be invoked in original function call
                elapsed_time = timers('interval-time').elapsed(reset=False, barrier=True)
                elapsed_time_per_iteration = elapsed_time / total_iterations

                if writer and (iteration % args.tensorboard_log_interval == 0):
                    summary_events = [
                        ("Speed/steps_per_sec", 1 / elapsed_time_per_iteration, iteration),
                        ("Speed/seconds_per_step", elapsed_time_per_iteration, iteration),
                        ("Gradient/grad_norm", grad_norm, iteration),
                    ]
                    if _SPECIAL_LR_LOG_STATE:
                        summary_events.append(
                            ("learning-rate/special-lr-scale", _SPECIAL_LR_LOG_STATE["lr"], iteration)
                        )
                        if _SPECIAL_LR_LOG_STATE["min_lr"] != _SPECIAL_LR_LOG_STATE["max_lr"]:
                            summary_events.extend([
                                ("learning-rate/special-lr-scale-min", _SPECIAL_LR_LOG_STATE["min_lr"], iteration),
                                ("learning-rate/special-lr-scale-max", _SPECIAL_LR_LOG_STATE["max_lr"], iteration),
                            ])

                    for name, loss in loss_dict.items():
                        summary_events.append((f"Loss/{name}", loss, iteration))
                    for name, samples in scalar_state.consumed_samples_total.items():
                        summary_events.append((f"Consumed Samples/{name}", samples, iteration))
                    for name, tokens in scalar_state.consumed_tokens_total.items():
                        summary_events.append((f"Consumed Tokens/{name}", tokens, iteration))

                    for (tag, value, step) in summary_events:
                        writer.add_scalar(tag, value, step)
                        if wandb_writer:
                            wandb_writer.log({tag: value}, step)
            except Exception:
                # Silently ignore if scalar_state is not available
                pass

        # Megatron's original logging uses the last non-decoupled param group lr, which
        # can be a special scaled group. Prefer the cached base lr when available.
        logged_learning_rate = _BASE_LR_LOG_STATE.get("learning_rate", learning_rate)

        # Call original function
        report_memory_flag = fn(
            loss_dict,
            total_loss_dict,
            logged_learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            num_floating_point_operations_in_batch,
            start_iteration,
        )

        # Megatron's training_log only prints timers in a hard-coded `timers_to_log` list
        # (see Megatron-LM/megatron/training/training.py),add multimodal_gen_timers_to_log 
        # to log multimodal_gen timers to tensorboard and wandb
        multimodal_gen_timers_to_log = [
            "prepare-model-inputs",
            "model-forward",
            # Leo encode-balance/dp-balance phase timers
            "encode-balance-all",
            "prefetch-batches",
            "encode-audio-buddy",
            "dp-balance",
            "encode-intra-dp",
            "encode-cp-broadcast",
            "encode-offload",
        ]
        if iteration % args.log_interval == 0:
            writer = get_tensorboard_writer()
            wandb_writer = get_wandb_writer()
            if args.log_timers_to_tensorboard:
                timers.write(
                    multimodal_gen_timers_to_log, writer, iteration,
                    normalizer=args.log_interval, reset=False,
                )
                timers.write(
                    multimodal_gen_timers_to_log, wandb_writer, iteration,
                    normalizer=args.log_interval, reset=False,
                )
            timers.log(multimodal_gen_timers_to_log, normalizer=args.log_interval)

        return report_memory_flag

    return wrapper


def optimizer_param_scheduler_step_wrapper(fn):
    @wraps(fn)
    def wrapper(self, increment: int) -> None:
        result = fn(self, increment)

        try:
            args_obj = get_args()
            if getattr(args_obj, 'special_lr_scale_params', None):
                special_lr_mult = getattr(args_obj, 'special_lr_mult', None)
                special_lrs = []
                base_lrs = []
                default_base_lrs = []
                for param_group in self.optimizer.param_groups:
                    if len(param_group.get('params', ())) == 0:
                        continue
                    group_lr_mult = param_group.get('lr_mult', 1.0)
                    is_special_lr_group = (
                        special_lr_mult is not None
                        and group_lr_mult == special_lr_mult
                    )
                    if is_special_lr_group:
                        special_lrs.append(param_group['lr'])
                    elif not param_group.get('is_decoupled_lr', False):
                        base_lrs.append(param_group['lr'])
                        if group_lr_mult == 1.0:
                            default_base_lrs.append(param_group['lr'])

                global _SPECIAL_LR_LOG_STATE, _BASE_LR_LOG_STATE
                if special_lrs:
                    _SPECIAL_LR_LOG_STATE = {
                        "lr": special_lrs[0],
                        "min_lr": min(special_lrs),
                        "max_lr": max(special_lrs),
                    }
                else:
                    _SPECIAL_LR_LOG_STATE = {}

                logged_base_lrs = default_base_lrs or base_lrs
                if logged_base_lrs:
                    _BASE_LR_LOG_STATE = {
                        "learning_rate": logged_base_lrs[0],
                        "min_lr": min(logged_base_lrs),
                        "max_lr": max(logged_base_lrs),
                    }
                else:
                    _BASE_LR_LOG_STATE = {}
            else:
                _SPECIAL_LR_LOG_STATE = {}
                _BASE_LR_LOG_STATE = {}
        except Exception as exc:
            raise RuntimeError("Failed to update special LR log state") from exc

        return result

    return wrapper


def custom_backward_replacer(output, grad_output):
    """Custom backward function, Copid from backward_step"""
    # assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    # assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    # assert isinstance(grad_output, (torch.Tensor, type(None))), (
    #     "grad_output == '%s'." % type(grad_output).__name__
    # )
    # Handle scalar output
    if len(output) == 1:
        output = output[0]
        
    if len(grad_output) == 1 and grad_output[0] is None:
        # assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    new_output = []
    new_grad_output = []
    for i, item in enumerate(output):
        if item.grad_fn is not None:
            new_output.append(item)
            new_grad_output.append(grad_output[i])


    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=tuple(new_output),
        grad_tensors=tuple(new_grad_output),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

def backward_step(
    input_tensor,
    output_tensor,
    output_tensor_grad,
    model_type,
    config,
    pipeline_model_parallel_size=1,
    current_microbatch=None,
    vp_stage=None,
):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    _bwd_span_name = f"backward-compute-mb{current_microbatch}"
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()
    ftext_start_span(
        _bwd_span_name, enable_cpu_time=True, microstep=current_microbatch, vpp=vp_stage
    )
    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])
    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward_replacer(output_tensor, output_tensor_grad)
            # custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            # torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
            torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for _, x in enumerate(input_tensor):
            if x is None:
                input_tensor_grad.append(None)
            elif x.grad is None:
                input_tensor_grad.append(torch.zeros_like(x))
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()
    ftext_end_span(_bwd_span_name)
    return input_tensor_grad

def print_rank_last_wrapper(fn):
    """Wrapper for print_rank_last to add consumed_samples_total and consumed_tokens_total to training logs."""
    @wraps(fn)
    def wrapper(message, *args, **kwargs):
        # Check if this is a training iteration log
        if isinstance(message, str) and 'iteration' in message and 'consumed samples:' in message:
            # Try to add consumed_samples_total and consumed_tokens_total
            if HAS_HYMM_GLOBAL_VARS:
                try:
                    scalar_state = hymm_global_vars.get_scalar_state()
                    if scalar_state is not None:
                        # Calculate total consumed samples and tokens across all datasets
                        total_consumed_samples = 0
                        total_consumed_tokens = 0
                        if hasattr(scalar_state, 'consumed_samples_total'):
                            total_consumed_samples = sum(scalar_state.consumed_samples_total.values())
                        if hasattr(scalar_state, 'consumed_tokens_total'):
                            total_consumed_tokens = sum(scalar_state.consumed_tokens_total.values())
                        
                        # Insert the metrics after "consumed samples:" 
                        if total_consumed_samples > 0 or total_consumed_tokens > 0:
                            # Find the position after "consumed samples:"
                            insert_pos = message.find('consumed samples:')
                            if insert_pos != -1:
                                # Find the end of the consumed samples value (look for next |)
                                end_pos = message.find('|', insert_pos)
                                if end_pos != -1:
                                    # Insert the new metrics before the next |
                                    # Collect all metrics in a list to avoid duplicate separators
                                    metrics_list = []
                                    
                                    # Add total metrics
                                    if total_consumed_samples > 0:
                                        metrics_list.append(' consumed_samples_total: {:12d}'.format(total_consumed_samples))
                                    if total_consumed_tokens > 0:
                                        metrics_list.append(' consumed_tokens_total: {:12d}'.format(total_consumed_tokens))
                                    
                                    # Add per-dataset metrics (consumed_samples_total and consumed_tokens_total by name)
                                    if hasattr(scalar_state, 'consumed_samples_total') and scalar_state.consumed_samples_total:
                                        for name, samples in scalar_state.consumed_samples_total.items():
                                            if samples > 0:
                                                metrics_list.append(f' {name} smp: {samples:>14,}({scalar_state.consumed_epoch[name]})')
                                    
                                    if hasattr(scalar_state, 'consumed_tokens_total') and scalar_state.consumed_tokens_total:
                                        for name, tokens in scalar_state.consumed_tokens_total.items():
                                            if tokens > 0:
                                                metrics_list.append(f' {name} tks: {tokens:>18,}')
                                    
                                    # Join all metrics with ' |' separator
                                    if metrics_list:
                                        metrics_str = ' |'.join(metrics_list) + ' |'
                                        message = message[:end_pos] + metrics_str + message[end_pos:]
                except Exception:
                    # Silently ignore if scalar_state is not available
                    pass
        
        # Call original function
        return fn(message, *args, **kwargs)
    
    return wrapper

def is_muon_used_on_this_param_wrapper(fn):
    @wraps(fn)
    def wrapper(name, param):
        args_obj = get_args()
        special_adamw_params = getattr(args_obj, 'special_adamw_params', None)
        if special_adamw_params is None:
            return fn(name, param)
        return param.ndim >= 2 and not multi_pattern_match(name, special_adamw_params)

    return wrapper

def track_moe_metrics_wrapper(fn):
    """Patch track_moe_metrics to include gen_ prefixed loss keys in reduce.

    In MoT (Mixture of Tokens) mode, ``_wrap_moe_layer_with_loss_prefix``
    stores gen-branch aux losses under keys like ``gen_load_balancing_loss``.
    However the upstream ``track_names`` list only contains the original
    names (e.g. ``load_balancing_loss``), so the gen_ variants are never
    reduced across PP/DP ranks.  This wrapper deterministically derives
    gen_ keys from track_names (instead of inspecting the local tracker)
    to ensure all PP stages use identical track_names and thus perform the
    same number of collective operations in reduce_aux_losses_tracker_across_ranks.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        track_names = kwargs.get('track_names', None)
        if track_names is not None:
            args_obj = get_args()
            if getattr(args_obj, 'use_mot', False):
                gen_keys = [
                    f"gen_{name}" for name in track_names
                    if f"gen_{name}" not in track_names
                ]
                if gen_keys:
                    track_names = list(track_names) + gen_keys
                    kwargs['track_names'] = track_names
        return fn(*args, **kwargs)
    return wrapper

def update_router_expert_bias_wrapper(fn):
    @wraps(fn)
    def wrapper(model: List[torch.nn.Module], config: TransformerConfig):
        """
        Update the expert bias of the router for a global batch.
        This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
        """

        args_obj = get_args()
        mot_und_frozen = getattr(args_obj, 'mot_und_frozen', False)
        mot_gen_frozen = getattr(args_obj, 'mot_gen_frozen', False)

        tokens_per_expert_list = []
        expert_bias_list = []
        tokens_reset_only_list = []
        for model_chunk in model:
            for name, module in get_attr_wrapped_model(model_chunk, 'named_modules')():
                if hasattr(module, 'expert_bias'):
                    if mot_und_frozen and not ("mlp_mot_gen" in name.split(".")):
                        tokens_reset_only_list.append(module.local_tokens_per_expert)  # und：不更新 bias，但要清零计数
                        continue
                    if mot_gen_frozen and ("mlp_mot_gen" in name.split(".")):
                        tokens_reset_only_list.append(module.local_tokens_per_expert)  # gen：不更新 bias，但要清零计数
                        continue
                    tokens_per_expert_list.append(module.local_tokens_per_expert)
                    expert_bias_list.append(module.expert_bias)
        # For hybrid models with both MoE and Dense layers, this list can be empty.
        if len(expert_bias_list) == 0:
            return
        stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
        stacked_expert_bias = torch.stack(expert_bias_list, dim=0)

        stacked_updated_expert_bias = get_updated_expert_bias(
            stacked_tokens_per_expert,
            stacked_expert_bias,
            config.moe_router_bias_update_rate,
            config.moe_router_enable_expert_bias_zero_mean_update,
        )

        for tokens_per_expert, expert_bias, updated_expert_bias in zip(
            tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
        ):
            tokens_per_expert.zero_()
            expert_bias.copy_(updated_expert_bias)

        # Reset tokens_per_expert for und branch if mot_und_frozen is True
        for tokens_per_expert in tokens_reset_only_list:
            tokens_per_expert.zero_()

    return wrapper

def get_param_groups_wrapper(fn):
    @wraps(fn)
    def wrapper(    
        model_chunks: List[MegatronModule],
        no_weight_decay_cond: Optional[Callable],
        scale_lr_cond: Optional[Callable],
        lr_mult: float,
        lr: float,
        min_lr: float,
        decoupled_lr: Optional[float],
        decoupled_min_lr: Optional[float],
        default_skip_embedding_weight_decay: bool = False,
        extra_weight_decay_param_substrings: Tuple[str] = (),
    ):
        """
        Patch get_param_groups to add custom no_weight_decay_cond
        """

        def custom_no_weight_decay_cond(name, param):
            match_extra_substrings = any(
                substring in name for substring in extra_weight_decay_param_substrings
            )

            no_wd = not match_extra_substrings and (
                name.endswith(".bias")
                or len(param.shape) == 1
                or (
                    default_skip_embedding_weight_decay
                    and (
                        "embedding" in name
                        or "embed_tokens" in name # patch部分，embed_tokens等价embedding, no_wd
                        or ("output_layer" in name and getattr(param, 'shared', False))
                    )
                )
            )

            if no_wd:
                log_single_rank(logger, logging.INFO, f"no_wd: {name}")
            else:
                log_single_rank(logger, logging.INFO, f"with_wd: {name}")
            return no_wd

        args_obj = get_args()
        lr_mult = getattr(args_obj, 'special_lr_mult', lr_mult)
        special_lr_scale_params = getattr(args_obj, 'special_lr_scale_params', None)
        special_lr_scale_exclude_params = getattr(args_obj, 'special_lr_scale_exclude_params', None)
        special_lr_scale_max_lr = getattr(args_obj, 'special_lr_scale_max_lr', None)
        special_lr_scale_min_lr = getattr(args_obj, 'special_lr_scale_min_lr', None)

        if special_lr_scale_params:
            def custom_scale_lr_cond(name, param):
                scale_lr = multi_pattern_match(name, special_lr_scale_params)
                if special_lr_scale_exclude_params and multi_pattern_match(name, special_lr_scale_exclude_params):
                    scale_lr = False
                if scale_lr:
                    log_single_rank(logger, logging.INFO, f"scale_lr: {name}, lr_mult: {lr_mult}")
                return scale_lr

            scale_lr_cond = custom_scale_lr_cond

        param_groups = fn(
            model_chunks=model_chunks,
            no_weight_decay_cond=custom_no_weight_decay_cond,
            scale_lr_cond=scale_lr_cond,
            lr_mult=lr_mult,
            lr=lr,
            min_lr=min_lr,
            decoupled_lr=decoupled_lr,
            decoupled_min_lr=decoupled_min_lr,
            default_skip_embedding_weight_decay=default_skip_embedding_weight_decay,
            extra_weight_decay_param_substrings=extra_weight_decay_param_substrings,
        )

        if special_lr_scale_params and (
            special_lr_scale_max_lr is not None or special_lr_scale_min_lr is not None
        ):
            assert lr_mult != 0, "lr_mult must be non-zero when overriding scaled lr bounds"
            for param_group in param_groups:
                if param_group.get('lr_mult', 1.0) == lr_mult:
                    if special_lr_scale_max_lr is not None:
                        param_group['max_lr'] = special_lr_scale_max_lr / lr_mult
                    if special_lr_scale_min_lr is not None:
                        param_group['min_lr'] = special_lr_scale_min_lr / lr_mult
                    log_single_rank(
                        logger,
                        logging.INFO,
                        (
                            "override scaled lr bounds: "
                            f"max_lr={param_group.get('max_lr')}, "
                            f"min_lr={param_group.get('min_lr')}, "
                            f"lr_mult={lr_mult}"
                        ),
                    )

        return param_groups
    return wrapper


def topk_router_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        if getattr(args, "keep_router_fp32_precision", False):
            setattr(self, _KEEP_ORIGINAL_PRECISION_ATTR, True)
    return wrapper

def distributed_optimizer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # FSDP path: DistributedOptimizer.__init__ returns early (before building
        # shard_fp32_from_float16_groups / model_float16_groups), so
        # _shard_main_param_to_model_param() would AttributeError. The _param_name
        # tags set below are only read by the muon bf16-update path, which under
        # FSDP keys by orig_param instead -> nothing to do here.
        if getattr(getattr(self, "ddp_config", None), "use_megatron_fsdp", False):
            return

        model_to_shard = self._shard_main_param_to_model_param()

        param_to_name = {}
        for chunk_id, model_chunk in enumerate(self.model_chunks):
            for name, model_param in model_chunk.named_parameters():
                param_to_name[model_param] = f"{chunk_id}.{name}"

        for model_param, shard_param in model_to_shard.items():
            setattr(shard_param, "_param_name", param_to_name[model_param])
    return wrapper


def muon_adamw_update_wrapper(fn):
    @wraps(fn)
    def wrapper(self, params: List[torch.Tensor], group: Dict[str, Any]):
        args = get_args()
        if not getattr(args, "param_update_in_bf16", False):
            return fn(self, params, group)
        
        # 参数 & exp_avg & exp_avg_sq以bf16的精度更新，expert的router除外
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["weight_decay"]
        if "step" in group:
            group["step"] += 1
        else:
            group["step"] = 1
        step = group["step"]

        for p in params:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if self._adamw_moment1_state_key not in state:
                state[self._adamw_moment1_state_key] = torch.zeros_like(g)
            if self._adamw_moment2_state_key not in state:
                state[self._adamw_moment2_state_key] = torch.zeros_like(g)

            update_on_bf16 = "router.weight" not in p._param_name

            buf1 = state[self._adamw_moment1_state_key]
            buf2 = state[self._adamw_moment2_state_key]
            data = p.data

            if update_on_bf16:  
                data = data.bfloat16()   
                g    = g.bfloat16()
                buf1 = buf1.bfloat16()
                buf2 = buf2.bfloat16()
            buf1.lerp_(g, 1 - beta1)
            buf2.lerp_(g.square(), 1 - beta2)

            g = buf1 / (eps + buf2.sqrt())

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            scale = bias_correction1 / bias_correction2**0.5
            data.mul_(1 - lr * weight_decay)
            data.add_(g, alpha=-lr / scale)

            if update_on_bf16:     
                state[self._adamw_moment1_state_key].copy_(buf1.float())
                state[self._adamw_moment2_state_key].copy_(buf2.float())
                p.data.copy_(data.float())        

    return wrapper

def muon_muon_update_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        args_obj = get_args()
        if getattr(args_obj, "param_update_in_bf16", False):
            raise NotImplementedError("param_update_bf16_precision is not supported for muon_update")
        return fn(self, *args, **kwargs)
    return wrapper


# run_patches only applies once
_patches_applied = False

def run_patches():
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    PatchesManager.register_patch(
        "megatron.core.pipeline_parallel.schedules.forward_step_calc_loss", forward_step_calc_loss_wrapper
    )
    # Fix type error when recompute enabled
    PatchesManager.register_patch(
        "megatron.core.tensor_parallel.random.CheckpointFunction", CheckpointFunction
    )

    # Install the cache-aware wrappers for attn cache (both the torch
    # flex_attention BlockMask path and the magi flex_flash_attn_func ranges
    # path; each install is a no-op if its kernel symbol is unavailable).
    if HAS_ATTN_CACHE_CHECKPOINT and argv_has_boolean_flag('selective-attn-output-checkpoint'):
        _install_flex_attention_cache_patch()
        _install_magi_attention_cache_patch()

    # Install the cache-aware wrapper for the DeepEP combine cache.
    if HAS_DEEPEP_CACHE_CHECKPOINT and argv_has_boolean_flag(
        'selective-deepep-combine-checkpoint'
    ):
        _install_deepep_cache_patch()

    # Add checkpoint loading and saving timestamp logging
    PatchesManager.register_patch(
        "megatron.training.checkpoint_dispatch.load_checkpoint", load_checkpoint_wrapper
    )
    PatchesManager.register_patch(
        "megatron.training.checkpoint_dispatch.save_checkpoint", save_checkpoint_wrapper
    )

    PatchesManager.register_patch(
        "megatron.core.pipeline_parallel.schedules.forward_step", forward_step_wrapper
    )

    PatchesManager.register_patch(
        "megatron.core.pipeline_parallel.schedules.backward_step", backward_step
    )

    # Leo a2a-overlap combined_1f1b dispatch.
    PatchesManager.register_patch(
        "megatron.core.pipeline_parallel.combined_1f1b.combined_1f1b_schedule_for_no_pipelining",
        combined_1f1b_schedule_for_no_pipelining_wrapper,
    )
    PatchesManager.register_patch(
        "megatron.core.pipeline_parallel.combined_1f1b.combined_1f1b_schedule_for_interleaved_pipelining",
        combined_1f1b_schedule_for_interleaved_pipelining_wrapper,
    )

    # Patch TransformerBlock.forward to support ut, images, image_h_w, timesteps parameter for pipeline parallelism
    PatchesManager.register_patch(
        "megatron.core.transformer.transformer_block.TransformerBlock.forward", transformer_block_forward,
    )
    PatchesManager.register_patch(
        "megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward", transformer_block_checkpointed_forward,
    )

    # Patch TransformerBlock.forward to support und_token_indices, gen_token_indices paramter
    PatchesManager.register_patch(
        "megatron.core.transformer.transformer_layer.TransformerLayer.forward", transformer_layer_forward_wrapper
    )

    # Align ptmv2 tensorboard/wandb logging with pure_torch
    PatchesManager.register_patch(
        "megatron.training.training.training_log", training_log_wrapper
    )
    PatchesManager.register_patch(
        "megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.step",
        optimizer_param_scheduler_step_wrapper,
    )

    # Patch print_rank_last to add consumed_samples_total and consumed_tokens_total to training logs
    PatchesManager.register_patch(
        "megatron.training.utils.print_rank_last", print_rank_last_wrapper
    )

    # Patch track_moe_metrics to include gen_ prefixed loss keys (from MoT) in reduce
    PatchesManager.register_patch(
        "megatron.core.transformer.moe.moe_utils.track_moe_metrics", track_moe_metrics_wrapper
    )

    # Patch is_muon_used_on_this_param to support special_adamw_params args
    PatchesManager.register_patch(
        "megatron.core.optimizer.is_muon_used_on_this_param", is_muon_used_on_this_param_wrapper
    )

    # Patch update_router_expert_bias_wrapper for mot_und_frozen
    PatchesManager.register_patch(
        "megatron.core.distributed.finalize_model_grads._update_router_expert_bias", update_router_expert_bias_wrapper
    )

    # Patch get_param_groups_wrapper for _get_param_groups add custom no_weight_decay_cond
    PatchesManager.register_patch(
        "megatron.core.optimizer._get_param_groups", get_param_groups_wrapper
    )
    
    # keep router fp32 precision
    PatchesManager.register_patch(
        "megatron.core.transformer.moe.router.TopKRouter.__init__", topk_router_init_wrapper
    )

    # bf16 param update
    PatchesManager.register_patch(
        "megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__", distributed_optimizer_init_wrapper
    )
    PatchesManager.register_patch(
        "angelptm.common.optimizers.muon.Muon._adamw_update", muon_adamw_update_wrapper
    )
    PatchesManager.register_patch(
        "angelptm.common.optimizers.muon.Muon._muon_update", muon_muon_update_wrapper
    )
    PatchesManager.register_patch(
        "megatron.core.optimizer.muon.DistributedMuonAdaptor._muon_update", muon_muon_update_wrapper
    )

    if is_bitwise_align_mode():
        for orig_func_name, new_func in bitwise_align_patchers:
                PatchesManager.register_patch(orig_func_name, new_func)

    # Meta-device streaming construction for the LeoLayer backbone (unit modules).
    # Wraps the __init__ of the memory-dominant local-impl leaves so that, while
    # model_provider builds LeoLayerBlock under meta_build_active(), their
    # weight/bias allocate on meta (zero RNG, zero memory). reset_parameters is
    # injected so MegatronFSDP materializes them per-shard, bitwise-identical to
    # the non-meta path. No-op when meta-init is off (wrappers just call orig).
    # See hymm/ptm_v2/meta_init_patch.py.
    from hymm.ptm_v2.meta_init_patch import (
        column_parallel_linear_init_wrapper,
        row_parallel_linear_init_wrapper,
        grouped_mlp_init_wrapper,
        leo_layer_block_init_wrapper,
        install_meta_init_methods,
    )
    PatchesManager.register_patch(
        "megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__",
        column_parallel_linear_init_wrapper,
    )
    PatchesManager.register_patch(
        "megatron.core.tensor_parallel.layers.RowParallelLinear.__init__",
        row_parallel_linear_init_wrapper,
    )
    PatchesManager.register_patch(
        "megatron.core.transformer.moe.experts.GroupedMLP.__init__",
        grouped_mlp_init_wrapper,
    )
    PatchesManager.register_patch(
        "angelptm.megatron.core.models.leo.leo_layer_block.LeoLayerBlock.__init__",
        leo_layer_block_init_wrapper,
    )
    install_meta_init_methods()

    PatchesManager.apply_patches()
