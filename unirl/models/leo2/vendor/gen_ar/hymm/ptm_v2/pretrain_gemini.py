import argparse
import logging
import os
import time
from functools import partial
from os.path import isfile

import torch
from loguru import logger

import hymm.core.global_vars as global_vars
import megatron
from angelptm.megatron.core.models.gemini import rl_utils
from angelptm.megatron.core.models.gemini.broadcast_utils import (
    sync_batches_across_pp_cp,
    sync_vae_outputs_across_cp,
    dp_load_balance_batches,
    vpp_data_iterators,
)
from angelptm.megatron.core.models.gemini.gemini_config import GeminiConfig
from angelptm.megatron.core.models.gemini.gemini_model import GeminiModel
from angelptm.megatron.core.models.gemini.transformer_layer import TransformerLayerMoT
from angelptm.megatron.training.callbacks import TrainerCallback

from hymm.core.data_provider import (
    DatasetsProvider,
    prepare_model_inputs,
)
from hymm.core.extra_model_provider import build_scalar_state, build_vae, prerun_vae, build_denoiser, build_tkwrapper
from hymm.core.global_vars import get_mm_state, set_logger, get_combined_iterator
from hymm.models.multimodal.hunyuan_multimodal_state import ParameterSummary
from hymm.utils.helpers import print_args
from hymm.utils.torch_utils import set_manual_seed, set_reproducibility
from megatron.core import mpu
from megatron.core import parallel_state
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_decoder_block_spec
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
import megatron.core.pipeline_parallel.schedules as schedule
import megatron.core.pipeline_parallel.p2p_communication as p2p_comm
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.training import get_args
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import (
    get_checkpoint_tracker_filename,
    read_metadata,
    get_checkpoint_name
)
from megatron.training.training import build_train_valid_test_datasets
from megatron.training.utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    unwrap_model,
)
from .patcher import run_patches

run_patches()


def _replace_self_attention_module(submodules: TransformerLayerSubmodules):
    from megatron.training.global_vars import get_args
    if getattr(get_args(), 'use_mot', False):
        print_rank_0(f"use CustomAttentionMoTCp")
        from angelptm.megatron.core.models.gemini.attention import CustomAttentionMoTCp
        submodules.self_attention.module = CustomAttentionMoTCp
    else:
        print_rank_0(f"use CustomAttention")
        from angelptm.megatron.core.models.gemini.attention import CustomAttention
        submodules.self_attention.module = CustomAttention


def get_layer_spec_fn_for_hunyuan(layer_spec_build_fn):
    def _fn(*args, **kwargs):
        layer_spec = layer_spec_build_fn(*args, **kwargs)
        # _check_layer_spec(layer_spec)
        _replace_self_attention_module(layer_spec.submodules)
        return layer_spec

    return _fn


def get_decoder_block_spec_fn_for_hunyuan(decoder_block_spec_build_fn):
    def _fn(*args, **kwargs):
        use_mot = kwargs.pop("use_mot", False)
        use_transformer_engine = kwargs["use_transformer_engine"]

        block_spec: TransformerBlockSubmodules = decoder_block_spec_build_fn(
            *args, **kwargs
        )
        if not isinstance(block_spec, TransformerBlockSubmodules):
            raise ValueError(
                "block_spec_wrapper only accepts 'decoder_block_spec_build_fn' "
                f"that returns a TransformerBlockSubmodules, but got '{type(block_spec).__name__}'"
            )

        
        if block_spec.layer_specs is not None:
            for layer_spec in block_spec.layer_specs:
                # _check_layer_spec(layer_spec)
                if use_mot:
                    layer_spec.module = TransformerLayerMoT

                _replace_self_attention_module(layer_spec.submodules)   # type: ignore
                # Clear TE-compatibility key mapping since we use HF-aligned specs
                # where input_layernorm/pre_mlp_layernorm are stored directly,
                # not remapped to self_attention.linear_qkv.layer_norm_ etc.
                if not use_transformer_engine:
                    layer_spec.submodules.sharded_state_dict_keys_map = {}
        return block_spec

    return _fn


@get_decoder_block_spec_fn_for_hunyuan
def get_gpt_decoder_block_spec_for_hunyuan(*args, **kwargs):
    return get_gpt_decoder_block_spec(*args, **kwargs)


@get_layer_spec_fn_for_hunyuan
def get_gpt_layer_with_transformer_engine_spec_for_hunyuan(*args, **kwargs):
    return get_gpt_layer_with_transformer_engine_spec(*args, **kwargs)


def model_provider(pre_process, post_process, vp_stage=None, config=None, ref_model=False):
    """ Build the model. """
    _ = config
    args = get_args()

    vp_size = args.virtual_pipeline_model_parallel_size

    if is_vp_first_stage(vp_stage, vp_size) and not ref_model:
        set_logger(logger)
    
    # build and initialize main model
    set_manual_seed(args.seed)
    set_reproducibility(args.reproduce, args.seed, args.benchmark)

    config = config_from_args(args)

    if args.num_experts:
        # Define the decoder block spec
        use_te = args.transformer_impl == "transformer_engine"
        transformer_layer_spec = get_gpt_decoder_block_spec_for_hunyuan(
            config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
            use_mot=getattr(args, 'use_mot', False),
        )
    else:
        # Define the decoder layer spec
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_for_hunyuan(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )

    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        config.variable_seq_lengths = True

        def get_tensor_shapes(
            *,
            seq_length: int,
            micro_batch_size: int,
            decoder_seq_length: int,
            config,
            tp_group: torch.distributed.ProcessGroup,
            cp_group: torch.distributed.ProcessGroup,
        ):
            from megatron.training.global_vars import get_args
            use_mot = getattr(get_args(), 'use_mot', False)
            tensor_shapes = [()] * 6 if use_mot else [()] * 5

            return tensor_shapes

        schedule.get_tensor_shapes = get_tensor_shapes

        if config.pp_multi_tensor_coalesced:

            # 多tensor batch通信时，hidden_states和gen_hidden_states保持bf16, 减少显存使用
            def get_tensor_dtypes(N, pipeline_dtype):
                from megatron.training.global_vars import get_args
                use_mot = getattr(get_args(), 'use_mot', False)
                pipeline_dtypes = [torch.bfloat16] + [torch.float32] * 4
                if use_mot:
                    pipeline_dtypes = pipeline_dtypes + [torch.bfloat16]
                return pipeline_dtypes

            p2p_comm.get_tensor_dtypes = get_tensor_dtypes

    if args.record_memory_history:
        # Start recording from model construction so the snapshot captures
        # model init + first training step memory allocations.
        torch.cuda.memory._record_memory_history(
            max_entries=args.memory_snapshot_max_entries
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print("saving allocated state during OOM")
            # Ensure recording is on so the OOM snapshot has stack info
            torch.cuda.memory._record_memory_history(
                max_entries=args.memory_snapshot_max_entries
            )
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(
                    f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}",
                    "wb",
                ),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    model = GeminiModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )

    # initialize scalar states, denoiser, vae, tkwrapper as global vars, after build model
    if is_vp_first_stage(vp_stage, vp_size) and not ref_model:
        build_or_load_scalar_state()
        build_denoiser()
        # build vae for the first stage if pp is enabled and stage is 0, otherwise build vae for all stages.
        if parallel_state.get_pipeline_model_parallel_world_size() == 1 or parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            build_vae()
        # TODO: when enable benchmark, the output of vae is not deterministic. 
        if args.benchmark:
            prerun_vae()
        build_tkwrapper()

    if torch.distributed.get_rank() == 0:
        print(ParameterSummary.from_module(model).as_table())

    return model


def config_from_args(args):
    """ transfusion config """

    model_name = args.model_name.split(".")[-1]

    from hymm.models.multimodal.hunyuan_multimodal_config import HunyuanMultimodalConfig, core_model_config_from_args
    model_config_dict = core_model_config_from_args(args)
    model_config = HunyuanMultimodalConfig.from_name(model_name, **model_config_dict)

    if args.per_stage_recompute_num_layers is not None:
        per_stage_recompute_num_layers = args.per_stage_recompute_num_layers
        pp_size = args.pipeline_model_parallel_size
        assert args.per_stage_recompute_num_layers and len(per_stage_recompute_num_layers) == pp_size, (
            f'--per-stage-recompute-num-layers length ({len(args.per_stage_recompute_num_layers)}) '
            f'must equal --pipeline-model-parallel-size ({pp_size}). '
            f'Got values: {per_stage_recompute_num_layers}'
        )

    model_args = {
        'use_vae': model_config.use_vae,
        'patch_size': model_config.patch_size,
        'img_proj_type': model_config.img_proj_type,
        # "img_proj_ndim": model_config.img_proj_ndim, # TODO(kevinkhwu): added in dev_video
        'patch_embed_hidden_dim': model_config.patch_embed_hidden_dim,
        'max_position_embeddings': model_config.max_position_embeddings,
        'rope_type': model_config.rope_type,
        'vae_latent_dim': model_config.vae_latent_dim,
        'use_timestep_token': model_config.use_timestep_token,
        'use_vit': model_config.use_vit,
        'vit_type': model_config.vit_type,
        'vit_config': model_config.vit_config,
        'use_vit_aligner': model_config.use_vit_aligner,
        'vit_aligner_type': model_config.vit_aligner_type,
        'vit_aligner_config': model_config.vit_aligner_config,
        'vocab_size': model_config.vocab_size,
        'pipeline_dtype': torch.float32, # pipeline dtype is float32 for gemini model, otherwise it will cause precision mismatch.
        'rope_base_rescale_factor': model_config.rope_scaling,
        'xdrope_section': model_config.xdrope_section,
        'use_mot': args.use_mot,
        'hidden_size_mot_gen': args.hidden_size_mot_gen if args.hidden_size_mot_gen is not None else model_config.hidden_size_mot_gen,
        'moe_topk_mot_gen': args.moe_topk_mot_gen if args.moe_topk_mot_gen is not None else model_config.moe_topk_mot_gen,
        'num_kv_heads_mot_gen': model_config.num_kv_heads_mot_gen,
        'ffn_hidden_size_mot_gen': model_config.ffn_hidden_size_mot_gen,
        'num_experts_mot_gen': model_config.num_experts_mot_gen,
        'mrope_section': model_config.mrope_section, 
        'per_stage_recompute_num_layers': args.per_stage_recompute_num_layers,
        'attention_head_size': model_config.attention_head_size,
        'rope_theta': model_config.rope_theta,
        'rope_scaling': model_config.rope_scaling,
        "apply_rope_in_fp32": model_config.apply_rope_in_fp32,
    }

    # read config and mapping the config to mcore
    transformer_config = core_transformer_config_from_args(args)
    # if there are duplicate keys, model_args overwrites transformer_config.__dict__
    config = GeminiConfig(**{**transformer_config.__dict__, **model_args})

    if args.use_unified_init_method:
        assert config.init_method is not None
        config.embedding_init_method = config.init_method
        config.output_layer_init_method = config.init_method

    # hardcode config.fp4 to apdapt the latest version of megatron, commitid: bb21676
    # leo_config.fp4 = None

    # Muon optimizer requires splitting params by num_heads and num_experts. When mot_gen config are different from und, 
    # the split rules need to modify.
    if args.optimizer == "muon":
        assert config.num_kv_heads_mot_gen is None or config.num_kv_heads_mot_gen == config.num_kv_heads
        assert config.ffn_hidden_size_mot_gen is None or config.ffn_hidden_size_mot_gen == config.ffn_hidden_size
        assert config.num_experts_mot_gen is None or config.num_experts_mot_gen == config.num_experts

    if torch.distributed.get_rank() == 0:
        print_args("model config", config)
    return config


def loss_func(loss_dict: dict[str, torch.Tensor], output_tensor: torch.Tensor, consumed_metrics: dict):
    _ = output_tensor
    scalar_state = global_vars.get_scalar_state()
    args = get_args()
    loss = loss_dict['loss'].float()
    is_update_step = scalar_state.train_steps % args.gradient_accumulation_steps == 0

    # detach统计信息，避免input_tensor的grad_fn无法释放，导致last pp stage梯度随micro-batch异常增长
    loss_dict_detach = {}
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            loss_dict_detach[k] = v.detach()
        else:
            loss_dict_detach[k] = v
    scalar_state.update(loss_dict_detach, consumed_metrics, 0.0, is_update_step, 0.0)
    return loss, {}

def forward_step(data_iterator, model):
    args = get_args()

    _t0 = time.monotonic()
    batch = next(data_iterator)
    _elapsed = time.monotonic() - _t0
    if _elapsed > 1.0:
        rank = torch.distributed.get_rank()
        print(f"[WARNING] rank={rank} | get batch took {_elapsed:.3f}s")

    vp_size = args.virtual_pipeline_model_parallel_size
    vp_stage = getattr(model.module, "vp_stage", None)

    if is_vp_first_stage(vp_stage, vp_size):
        device = torch.device("cuda", args.local_rank)

        with torch.autocast(device_type="cuda", enabled=False):
            is_pp_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            cp_size = parallel_state.get_context_parallel_world_size()
            cp_rank = parallel_state.get_context_parallel_rank()

            # PP!=0 always skips VAE encoding (receives x_t via pipeline P2P).
            # When broadcast-data is enabled, CP!=0 on PP=0 also skips:
            # VAE outputs are synced from CP=0 below.
            _broadcast_data = getattr(args, 'broadcast_data', False)
            _need_dp_load_balance = getattr(args, 'dp_load_balance', False)

            skip_vae_encode = not is_pp_first or (_broadcast_data and cp_size > 1 and cp_rank != 0)
            model_input_kwargs, bsz, seqlen = prepare_model_inputs(batch, device, skip_vae_encode=skip_vae_encode)
            if _need_dp_load_balance:
                model_input_kwargs.update(dict(
                    micro_batch_image_count=batch["micro_batch_image_count"], 
                    micro_batch_token_count=batch["micro_batch_token_count"]
                ))

            if _broadcast_data and is_pp_first and cp_size > 1:
                model_input_kwargs = sync_vae_outputs_across_cp(model_input_kwargs, device)
        if vp_size is not None and vp_size > 1:
            data_iterator.publish((batch, model_input_kwargs, bsz, seqlen))
    else:
        batch, model_input_kwargs, bsz, seqlen = batch

    consumed_metrics = {
        batch["dataset_tag"][0]: {
            "samples": batch["n_samples"].sum().item(),
            "tokens": bsz * seqlen
        }
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output_tensor, loss_dict = model(**model_input_kwargs)

    return output_tensor, partial(loss_func, loss_dict, consumed_metrics=consumed_metrics)


def _add_gemini_extra_args(parser: argparse.ArgumentParser):
    # MFU monitoring args — registered here (instead of in AngelPTM's
    # arguments.py) to avoid modifying the submodule. dest 名保持不变
    # (mfu_log / mfu_detail_log / mfu_gpu_type)，所有下游 getattr(args, ...)
    # 行为完全一致。
    group = parser.add_argument_group(title="gemini-extra-args")
    group.add_argument("--keep-router-fp32-precision", action="store_true", default=False,
                       help="Keep router fp32 precision")
    group.add_argument("--param-update-in-bf16", action="store_true", default=False,
                       help="param update in bf16 precision")
    return parser


def extra_args_provider(parser: argparse.ArgumentParser):
    from ..config import add_core_args, add_evaluation_args
    from angelptm.megatron.training.arguments import add_ptm_extra_args

    parser = add_core_args(parser, ptm="v2")
    parser = _add_gemini_extra_args(parser)
    add_ptm_extra_args(parser)

    return parser


def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func):
    """Single training step."""
    _ = config
    args = get_args()
    timers = get_timers()
    mm_state = get_mm_state()

    # Define monitored metrics
    loss_names = ["loss"] + [
        f"{dataset_tag}_text_loss" for dataset_tag in mm_state.all_dataset_keys
    ] + [
        f"{dataset_tag}_image_loss" for dataset_tag in mm_state.all_dataset_keys
    ]
    if args.moe_aux_loss_coeff > 0:
        loss_names.append("moe_loss")

    loss_names.append("capacity_rate")
    loss_names.append("num_tokens_per_expert_balance_mean")
    loss_names.append("num_tokens_per_expert_balance_std")
    loss_names.append("num_tokens_per_expert_balance_max")
    loss_names.append("num_tokens_per_expert_balance_min")
    loss_names.append("num_tokens_per_expert_balance_std/mean")
    if args.use_mot:
        loss_names.append("capacity_rate_mot_gen")
        loss_names.append("num_tokens_per_expert_mot_gen_balance_mean")
        loss_names.append("num_tokens_per_expert_mot_gen_balance_std")
        loss_names.append("num_tokens_per_expert_mot_gen_balance_max")
        loss_names.append("num_tokens_per_expert_mot_gen_balance_min")
        loss_names.append("num_tokens_per_expert_mot_gen_balance_std/mean")
    
    if getattr(args, "rl_enabled", False):
        loss_names.extend(rl_utils.dpo_extra_loss_names())


    _need_batch_sync = getattr(args, 'broadcast_data', False)
    _need_dp_load_balance = getattr(args, 'dp_load_balance', False)
    if _need_dp_load_balance:
        assert not _need_batch_sync, "dp_load_balance return iterator, so broadcast_data cannot be used at the same time now."
    
    _use_global_batch_count_average = getattr(args, 'use_global_batch_count_average', False)
    if _use_global_batch_count_average:
        assert _need_dp_load_balance, "use_global_batch_count_average requires dp_load_balance now."

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        adjust_tensor_shapes_fn = None

        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            for optim_instance in optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()  # noqa

        # Broadcast batch from PP=0,CP=0 to ensure data consistency across
        # all ranks.  Must run before the pipeline schedule (PP stages call
        # forward_step at staggered times, so collective ops inside would deadlock).
        if _need_batch_sync or _need_dp_load_balance:
            num_mb = get_num_microbatches()

            # vpp的时候，data_iterator是list
            if isinstance(data_iterator, list):
                pre_batches = [next(data_iterator[0]) for _ in range(num_mb)]
            else:
                pre_batches = [next(data_iterator) for _ in range(num_mb)]

            # DP load-balance: redistribute batches across DP ranks.
            if _need_dp_load_balance:
                pre_batches = dp_load_balance_batches(
                    pre_batches,
                    task_priority_list=args.dp_load_balance_task_priority,
                    dp_group=parallel_state.get_data_parallel_group(),
                    use_global_batch_count_average=_use_global_batch_count_average,
                )

            if _need_batch_sync:
                pre_batches = sync_batches_across_pp_cp(pre_batches)

            fwd_data_iter = iter(pre_batches)
            if isinstance(data_iterator, list):
                fwd_data_iter = [fwd_data_iter] + data_iterator[1:]
        else:
            fwd_data_iter = data_iterator

        vp_size = args.virtual_pipeline_model_parallel_size
        # 开vpp的时候, 只有vp_stage=0做数据处理， 并缓存处理好的数据，后续的vp_stage直接读取预处理好的数据
        if vp_size is not None and vp_size > 1:
            fwd_data_iter = vpp_data_iterators(fwd_data_iter, vp_size)

        # Forward pass.
        losses_reduced = forward_backward_func(     # noqa
            forward_step_func=forward_step_func,
            data_iterator=fwd_data_iter,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        scalar_state = global_vars.get_scalar_state()
        loss_reduced = scalar_state.all_reduce(
            loss_names, mm_state.all_dataset_keys, parallel_state.get_data_parallel_group())
        scalar_state.reset_running_states()
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad


megatron.training.training.train_step = train_step


# Keep scalar-state persistence on the dispatcher so FSDP uses its dedicated
# checkpoint service rather than falling back to the legacy implementation.
import megatron.training.checkpoint_dispatch as checkpoint_dispatch_module
_original_save_checkpoint = checkpoint_dispatch_module.save_checkpoint


def save_checkpoint_with_scalar_states(
        iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, *extra_args, **kwargs
):
    """Wrapper for save_checkpoint that saves scalar_states separately"""
    args = get_args()
    scalar_state = global_vars.get_scalar_state()
    print_rank_0(f"[DEBUG] save_checkpoint_with_scalar_states called for iteration {iteration}")

    # Call original save_checkpoint first
    result = _original_save_checkpoint(
        iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far, *extra_args, **kwargs
    )

    # Sync data iterator state
    if args.resume_index_batch_sampler:
        combined_iterator = get_combined_iterator()
        print_rank_0(f"Synchronizing IndexBatchSampler state dict...")
        combined_iterator.sync_state_dict()

    # After checkpoint is saved, save scalar_states to a separate file
    p_state = global_vars.get_parallel_state()
    save_ss = (p_state.dp_rank == 0) and (p_state.pp_rank == p_state.pp_size - 1) and (p_state.tp_rank == 0) and (p_state.cp_rank == 0)
    if scalar_state is not None and save_ss:
        checkpoint_dir = os.path.join(args.save, f"iter_{iteration:07d}")
        training_states_path = os.path.join(checkpoint_dir, "training_states.pt")

        training_states = {'scalar_state': scalar_state.serialize()}
        torch.save(training_states, training_states_path)

        print_rank_0(f"==> Saved scalar_state to {training_states_path}: training_states:{training_states}")
    elif scalar_state is None:
        print_rank_0(f"[WARNING] scalar_state is None, not saving scalar_state")

    return result


# Training imported save_checkpoint by value; replace that alias while leaving
# the base implementation untouched for checkpoint_dispatch to delegate to.
import megatron.training.training as training_module
training_module.save_checkpoint = save_checkpoint_with_scalar_states


def build_or_load_scalar_state():
    args = get_args()

    scalar_state = None
    if hasattr(args, 'load') and args.load is not None:
        # Megatron saves checkpoints in a specific structure
        checkpoint_path = args.load
        # Look for the latest checkpoint
        iteration, release = -1, False
        if checkpoint_path is not None:
            tracker_filename = get_checkpoint_tracker_filename(checkpoint_path)
            if isfile(tracker_filename):
                iteration, release = read_metadata(tracker_filename)

        if getattr(args, "ckpt_step", None):
            iteration = args.ckpt_step
        
        if iteration != -1:
            checkpoint_dir = get_checkpoint_name(checkpoint_path, iteration, release, return_base_dir=True)
        else:
            checkpoint_dir = checkpoint_path
            print_rank_0(f"==> Warning: iteration == -1, use checkpoint_path as checkpoint_dir")
        
        print_rank_0(f"==> checkpoint_dir: {checkpoint_dir}")

        # Try to load scalar_states from checkpoint metadata
        training_states_path = os.path.join(checkpoint_dir, 'training_states.pt')
        if os.path.exists(training_states_path):
            training_states = torch.load(training_states_path, map_location='cpu')

            if 'scalar_state' in training_states:
                scalar_state = training_states['scalar_state']
            elif 'scalar_state' in training_states['client_state']:
                scalar_state = training_states['client_state']['scalar_state']  # for load torch training_states.pt

    scalar_state = build_scalar_state(scalar_state)
    scalar_state.current_run_update_steps = 0
    scalar_state.current_forward_times = 0
    print_rank_0(f"build or load scalar states: {scalar_state}")


# =========================================================================
#       Entry point for training Gemini-MoE with AngelPTM v2
# =========================================================================
def launch(frozen_args):
    from megatron.core.enums import ModelType
    from megatron.training import inprocess_restart
    from megatron.training import pretrain

    # Optionally enable in-process restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    logger = logging.getLogger("megatron.core.utils")
    logger.setLevel(logging.WARN)

    rl_enabled = frozen_args.get("rl_enabled", False)
    if rl_enabled:
        chosen_forward_step = rl_utils.forward_step_dpo
        chosen_model_provider = rl_utils._wrap_model_provider_for_dpo(model_provider)
        datasets_provider = rl_utils.DatasetsProvider()
    else:
        chosen_forward_step = forward_step
        chosen_model_provider = model_provider
        datasets_provider = DatasetsProvider()

    pretrain(
        datasets_provider,
        chosen_model_provider,
        ModelType.encoder_or_decoder,
        chosen_forward_step,
        args_defaults=frozen_args,
        extra_args_provider=extra_args_provider,
        store=store,
        callbacks=TrainerCallback(),
    )
