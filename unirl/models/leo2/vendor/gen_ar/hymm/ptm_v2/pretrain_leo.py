import argparse
import copy
import logging
import os
from functools import partial
from os.path import isfile

import torch
from loguru import logger

import hymm.core.global_vars as global_vars
import megatron
from angelptm.megatron.core.models.leo.dp_load_balance import (
    dp_load_balance,
    dp_load_balance_payload,
    dplb_overlap_enabled,
)
from angelptm.megatron.core.models.leo.leo_utils.fsdp import (
    dp_load_balance_payload_stratified,
    dp_load_balance_stratified,
    get_or_create_buddy_encode_group,
    warmup_dplb_schedule_group,
)
from angelptm.megatron.core.models.leo.ep_1f1b_overlap.leo_combine_recompute import (
    validate_combine_recompute,
)
from angelptm.megatron.core.models.leo.leo_utils.parallel_layout import (
    megatron_global_rank,
)
from angelptm.megatron.core.models.leo.leo_utils.pp_tensor_shapes import (
    assert_pp_payload_shape_matches_analytic,
)
from angelptm.megatron.core.models.leo.leo_model import LeoModel
from angelptm.megatron.core.models.leo.leo_layer_specs import get_leo_layer_block_spec
from angelptm.megatron.core.models.leo.leo_config import LeoConfig

from hymm.core.data_provider import DatasetsProvider
from hymm.core.data_provider_dit import compute_cond_vae_mask, prepare_model_inputs
from hymm.core.extra_model_provider import (
    build_scalar_state,
    build_vae,
    prerun_vae,
    build_denoiser,
    build_tkwrapper,
    build_text_encoder,
    build_repa_encoder,
    build_audio_vae,
)
from hymm.core.global_vars import set_logger, get_mm_state, get_combined_iterator
from hymm.models.multimodal.hunyuan_multimodal_state import ParameterSummary
from hymm.utils.helpers import print_args
from hymm.utils.torch_utils import set_manual_seed, set_reproducibility
from megatron.core import parallel_state, mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import (
    get_checkpoint_tracker_filename,
    read_metadata,
    get_checkpoint_name
)
from megatron.training.utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    unwrap_model,
)
from angelptm.megatron.training.callbacks import TrainerCallback

from .patcher import run_patches

run_patches()


# Text-encoder FSDP shard scopes; only meaningful when the encoder is built on
# every PP rank (intra_dp_encode_balance):
#   "block":  contiguous text_encoder_fsdp_max_group_size blocks over the encoder
#             ranks, so adjacent cards/machines form a shard group.
#   "pp-cp":  one shard group spanning pp×cp — exactly the intra-dp
#             encode group; max_group_size is ignored.
_TEXT_ENCODER_FSDP_SCOPES = ("block", "pp-cp")

# One-shot guard for the stratified DP-LB exchange-group creation (see train_step).
_dplb_group_warmed = {"done": False}


def _need_encoder_on_all_pp(args, kind):
    """Whether the given encoder ('image_vae' / 'audio_vae' / 'text_encoder') needs
    to be built on every PP rank (for encode load balancing).

    Scope rules:
      - intra_dp_encode_balance     -> needs image_vae / audio_vae / text_encoder on all PP
      - audio_buddy_encode_balance  -> needs audio_vae on all PP (t2a + helper DPs)
      - fsdp_encode_balance         -> needs image_vae / text_encoder on every DP helper
                                       (PP=1, so "all PP" is just the single stage)
    """
    intra_dp = getattr(args, "intra_dp_encode_balance", False)
    buddy = getattr(args, "audio_buddy_encode_balance", False)
    fsdp_encode = getattr(args, "fsdp_encode_balance", False)
    if kind == "audio_vae":
        return intra_dp or buddy
    return intra_dp or fsdp_encode   # image_vae / text_encoder


def _validate_fsdp_encode_balance_args(args):
    """Fail before model construction for unsupported FSDP buddy layouts.

    Every check here is a precondition of image_buddy_balance / text_buddy_balance
    (see angelptm.../leo/fsdp_encode_balance). They are validated eagerly so a bad
    config fails at startup instead of mid-collective on one rank.
    """
    if not getattr(args, "fsdp_encode_balance", False):
        return
    parallel_sizes = {
        "TP": getattr(args, "tensor_model_parallel_size", 1),
        "PP": getattr(args, "pipeline_model_parallel_size", 1),
        "CP": getattr(args, "context_parallel_size", 1),
    }
    if any(size != 1 for size in parallel_sizes.values()):
        raise ValueError(
            "fsdp-encode-balance currently requires TP=PP=CP=1, got "
            + ", ".join(f"{name}={size}" for name, size in parallel_sizes.items())
        )
    if getattr(args, "vae_encode_type", None) != "mode":
        raise ValueError(
            "fsdp-encode-balance requires --vae-encode-type mode so a helper rank's "
            "VAE output does not depend on its own RNG state"
        )
    if getattr(args, "combined_iterator_sampling_mode", None) != "fixed":
        raise ValueError(
            "fsdp-encode-balance requires --combined-iterator-sampling-mode fixed: "
            "the buddy layout is keyed on a stable task-to-DP-rank allocation"
        )
    if not getattr(args, "text_encoder_use_pack", False):
        raise ValueError(
            "fsdp-encode-balance requires --text-encoder-use-pack so every "
            "worker can issue one packed FSDP-safe forward"
        )
    if not getattr(args, "text_encoder_pack_microbatches", False):
        raise ValueError(
            "fsdp-encode-balance requires --text-encoder-pack-microbatches"
        )
    if getattr(args, "intra_dp_encode_by_pp_stage", False):
        raise ValueError(
            "fsdp-encode-balance cannot be combined with "
            "--intra-dp-encode-by-pp-stage because both own visual/text encoding"
        )
    if getattr(args, "intra_dp_encode_balance", False):
        raise ValueError(
            "fsdp-encode-balance cannot be combined with --intra-dp-encode-balance: "
            "both phases would encode the same micro-batch, producing the same "
            "precomputed keys (image_latents / last_frame_latents / text_states / "
            "text_mask). fsdp-encode-balance requires PP=CP=1, where intra-dp "
            "(a PP x CP split) has nothing to distribute anyway."
        )
    if getattr(args, "benchmark", False):
        raise ValueError(
            "fsdp-encode-balance requires deterministic VAE execution; disable --benchmark"
        )
    if getattr(args, "overlap_moe_expert_parallel_comm", False):
        raise ValueError(
            "fsdp-encode-balance requires PP=1, but Leo's PP=1 MoE A2A-overlap "
            "schedule is not supported; disable --overlap-moe-expert-parallel-comm"
        )


def _resolve_buddy_encode_group(args, batches):
    """Build a bounded, task-mixed buddy group before DP-LB reshuffles batches.

    Must be called while the fixed task-to-DP-rank allocation is still visible in
    ``batches``: DP-LB may change this rank's local task afterwards. Returns the
    full DP group when --fsdp-encode-balance-group-size is 0 (the default).
    """
    full_dp_group = parallel_state.get_data_parallel_group()
    full_dp_size = torch.distributed.get_world_size(group=full_dp_group)
    group_size = int(getattr(args, "fsdp_encode_balance_group_size", 0) or 0)
    if group_size < 0 or group_size > full_dp_size:
        raise ValueError(
            "fsdp-encode-balance-group-size must be in "
            f"[0, {full_dp_size}], got {group_size}"
        )
    if group_size in (0, full_dp_size):
        return full_dp_group
    if getattr(args, "combined_iterator_sampling_mode", None) != "fixed":
        raise RuntimeError(
            "buddy encode subgroups require --combined-iterator-sampling-mode fixed"
        )

    group_info = get_or_create_buddy_encode_group(
        full_dp_group=full_dp_group,
        batches=batches,
        max_group_size=group_size,
    )
    if torch.distributed.get_rank() == 0:
        layout = group_info.layout
        logger.info(
            f"buddy encode subgroups: max_size={group_size} "
            f"groups={len(layout.group_dp_ranks)} layout={layout.group_dp_ranks}"
        )
    return group_info.group


def _text_image_pp_stages(args):
    """Parse + validate the text/image PP-stage specs for intra-dp-encode-by-pp-stage.

    Returns (text_stages, image_stages) as sets of PP-stage indices.
    """
    from angelptm.megatron.core.models.leo.leo_utils.encode_balance import parse_pp_stage_spec
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    text_stages = parse_pp_stage_spec(args.intra_dp_text_pp_stages, pp_size)
    image_stages = parse_pp_stage_spec(args.intra_dp_image_pp_stages, pp_size)
    assert getattr(args, "intra_dp_encode_balance", False), (
        "intra_dp_encode_by_pp_stage requires intra_dp_encode_balance=True.")
    assert text_stages and not (text_stages & image_stages), (
        f"text PP stages must be non-empty and text/image PP stages must be disjoint, got "
        f"text={sorted(text_stages)} image={sorted(image_stages)}.")
    if image_stages:
        # With online image encoding, the owner runs add_image_noise and therefore
        # must also hold the image VAE (including its noise generator).
        assert 0 in image_stages, (
            f"PP stage 0 (encode owner) must be an image-PP stage; got {sorted(image_stages)}.")
    else:
        # An empty image-stage set is the all-offline-latent mode: no PP rank runs
        # the image encoder, while pp0 keeps only normalization state + RNG for
        # normalize/add-noise in prepare_model_inputs.
        assert getattr(args, "vae_norm_stats_only", False), (
            "empty intra_dp_image_pp_stages requires vae_norm_stats_only=True.")
        assert not getattr(args, "use_repa", False), (
            "empty intra_dp_image_pp_stages is incompatible with use_repa: "
            "REPA requires raw image pixels and an image encoder stage.")
    return text_stages, image_stages


def _create_text_encoder_fsdp_mesh(built_on_all_pp=False, by_pp_stage=False):
    """Build the FSDP DeviceMesh (shard groups) for the text encoder.

    `built_on_all_pp`:
      - False: pp_stage_0 only. Encoder ranks are the leading ``[0, world/pp)``.
      - True:  every rank . Encoder ranks are all ``[0, world)``.

    `by_pp_stage`: shard the encoder WITHIN each configured text-PP stage,
    in contiguous ``text_encoder_fsdp_max_group_size`` blocks

    Shard grouping (``text_encoder_fsdp_scope``; only meaningful when
    ``built_on_all_pp``):
      - "block" (default): split the encoder ranks into contiguous
        ``text_encoder_fsdp_max_group_size`` blocks.
      - "pp-cp": one shard group per spanning pp×cp — exactly the
        intra-dp encode group; ``text_encoder_fsdp_max_group_size`` is ignored.

    Returns the DeviceMesh containing the current rank, or None if the rank is
    outside all shard groups
    """
    from torch.distributed.device_mesh import DeviceMesh

    args = get_args()
    pp = parallel_state.get_pipeline_model_parallel_world_size()
    cp = args.context_parallel_size
    tp = args.tensor_model_parallel_size
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    dp = world // (tp * cp * pp)
    fsdp_scope = getattr(args, "text_encoder_fsdp_scope", "block")
    max_gsize = getattr(args, "text_encoder_fsdp_max_group_size", 0)

    if by_pp_stage:
        from angelptm.megatron.core.models.leo.leo_utils.encode_balance import parse_pp_stage_spec
        text_stages = sorted(parse_pp_stage_spec(args.intra_dp_text_pp_stages, pp))
        fsdp_scope = "by_pp_stage"
        #Shard the encoder in contiguous `gsize` blocks WITHIN each Each PP stage
        per_stage = world // pp
        gsize = min(max_gsize, per_stage) if max_gsize > 0 else per_stage
        assert per_stage % gsize == 0, (
            f"ranks per PP stage ({per_stage}) must be divisible by the text-encoder "
            f"FSDP group size ({gsize}); pick a text_encoder_fsdp_max_group_size that "
            f"divides {per_stage}.")
        groups = [list(range(st * per_stage + i, st * per_stage + i + gsize))
                  for st in text_stages
                  for i in range(0, per_stage, gsize)]
    elif built_on_all_pp and fsdp_scope == "pp-cp":
        if max_gsize > 0:
            print_rank_0("Text encoder FSDP: max_group_size ignored for scope='pp-cp'.")
        gsize = pp * cp
        # rank layout: tp + cp*tp + dp*tp*cp + pp*tp*cp*dp  (megatron default order)
        groups = [[megatron_global_rank(tp_i, cp_i, dp_i, pp_i,
                                        tp_size=tp, cp_size=cp, dp_size=dp)
                   for pp_i in range(pp) for cp_i in range(cp)]
                  for tp_i in range(tp) for dp_i in range(dp)]
    else:
        # contiguous ranges: [0, world) when built on all PP, else [0, world/pp).
        if fsdp_scope != "block":
            raise ValueError(
                f"text_encoder_fsdp_scope='{fsdp_scope}' is invalid; expected one of "
                f"{list(_TEXT_ENCODER_FSDP_SCOPES)} ('pp-cp' requires intra_dp_encode_balance).")
        universe = world if built_on_all_pp else world // pp
        gsize = max_gsize if max_gsize > 0 else universe
        assert universe % gsize == 0, (
            f"encoder rank count ({universe}) must be divisible by "
            f"text_encoder_fsdp_max_group_size ({gsize}).")
        groups = [list(range(i, i + gsize)) for i in range(0, universe, gsize)]

    # DeviceMesh -> new_group is collective: every rank must build every group in
    # the same order, even ones it isn't in. Each rank lands in exactly one group.
    group_meshes = [DeviceMesh("cuda", sorted(grp)) for grp in groups]
    my_grp = next((grp for grp in groups if rank in grp), None)
    my_mesh = next((m for m, grp in zip(group_meshes, groups) if rank in grp), None)

    # Print the full shard-group layout (once, on global rank 0) 
    groups_str = "\n".join(
        f"      shard_group[{gi:>3d}] (n={len(grp):>3d}): {sorted(grp)}"
        for gi, grp in enumerate(groups))
    print_rank_0(
        "=" * 80 + "\n"
        f"Text encoder FSDP sharding: scope='{fsdp_scope}', built_on_all_pp={built_on_all_pp}, "
        f"group_size={gsize}, num_groups={len(groups)} "
        f"(world={world}, tp={tp}, cp={cp}, pp={pp}, dp={dp})\n"
        f"{groups_str}\n" + "=" * 80)
    # Also let every rank report which shard group it belongs to
    logging.getLogger(__name__).info(
        "Text encoder FSDP: rank %d -> shard group %s", rank,
        sorted(my_grp) if my_grp is not None else None
    )
    return my_mesh


def build_extra_model():
    args = get_args()
    _validate_fsdp_encode_balance_args(args)
    validate_combine_recompute(args)

    # initialize scalar states, denoiser, vae, tkwrapper as global vars, after build model
    # align build_extra_model in MultimodalTrainer
    build_or_load_scalar_state()
    build_denoiser()

    is_pp0 = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    by_pp_stage = args.intra_dp_encode_by_pp_stage
    # by-pp-stage assigns encode by PP stage: text on text_stages, image VAE/repa on image_stages
    text_stages, image_stages = _text_image_pp_stages(args) if by_pp_stage else (None, None)

    def _build_here(kind, stages):
        return (pp_rank in stages) if by_pp_stage else (is_pp0 or _need_encoder_on_all_pp(args, kind))

    build_aud_vae = is_pp0 or _need_encoder_on_all_pp(args, "audio_vae")

    # In by-PP-stage all-offline mode image_stages is empty: no rank needs the
    # image encoder, but pp0 still needs latent normalization statistics and the
    # noise generator used by prepare_model_inputs.
    build_image_vae = (
        is_pp0 if by_pp_stage and not image_stages
        else _build_here("image_vae", image_stages)
    )
    if build_image_vae:
        build_vae(only_encoder=True, norm_stats_only=getattr(args, "vae_norm_stats_only", False))
    # TODO: when enable benchmark, the output of vae is not deterministic. 
    if args.benchmark:
        prerun_vae()
    build_tkwrapper()

    # build text encoder
    use_fsdp = getattr(args, "text_encoder_use_fsdp", False)
    built_on_all_pp = False if by_pp_stage else _need_encoder_on_all_pp(args, "text_encoder")
    # TODO: add text encoder fsdp validation (youngfyang)
    fsdp_mesh = _create_text_encoder_fsdp_mesh(
        built_on_all_pp=built_on_all_pp, by_pp_stage=by_pp_stage) if use_fsdp else None
    if _build_here("text_encoder", text_stages):
        assert args.use_text_encoder, "`use_text_encoder` must be True for Leo2Trainer."
        build_text_encoder(fsdp_mesh=fsdp_mesh)

    build_denoiser(
        denoiser_type="video",
        shift=args.flow_shift_video,
        snr_type=(args.flow_snr_type_video
                  if args.flow_snr_type_video is not None else args.flow_snr_type),
        snr_mix_uniform_ratio=(args.flow_snr_mix_uniform_ratio_video
                               if args.flow_snr_mix_uniform_ratio_video is not None
                               else args.flow_snr_mix_uniform_ratio),
    )
    if args.audio_branch_model_name is not None:
        build_denoiser(
            denoiser_type="audio",
            shift=args.flow_shift_audio,
            snr_type=(args.flow_snr_type_audio
                      if args.flow_snr_type_audio is not None else args.flow_snr_type),
            snr_mix_uniform_ratio=(args.flow_snr_mix_uniform_ratio_audio
                                   if args.flow_snr_mix_uniform_ratio_audio is not None
                                   else args.flow_snr_mix_uniform_ratio),
        )
    if build_aud_vae:
        if args.use_audio_vae:
            # Training only encodes audio; drop the decoder to save GPU memory.
            build_audio_vae(only_encoder=True)
        elif args.audio_branch_model_name is not None:
            # Audio latents are precomputed offline 
            build_audio_vae(use_audio_vae=False)
    # repa encoder (vision tower): same placement as image VAE (precomputes from raw images).
    if args.use_repa and _build_here("repa", image_stages):
        build_repa_encoder()

    # Warm up the intra-dp encode communicator
    if getattr(args, "intra_dp_encode_balance", False):
        from angelptm.megatron.core.models.leo.leo_utils.encode_balance import (
            warmup_intra_dp_encode_group,
        )
        warmup_intra_dp_encode_group()


def model_provider(pre_process, post_process, vp_stage=None, config=None):
    """ Build the model. """
    _ = config
    args = get_args()

    # VPP: when virtual pipeline is enabled, model_provider is called once per VP
    # chunk on each rank. Logger / seed / reproducibility / extra-model (VAE,
    # text-encoder, ...) must only be set up once — on the first chunk — to avoid
    # redundant builds and extra RNG consumption. vp_stage is None when VPP is off,
    # so this is a no-op for the non-VPP path (behaves exactly as before).
    is_first_chunk = (vp_stage is None or vp_stage == 0)

    if is_first_chunk:
        set_logger(logger)

        # build and initialize main model
        set_manual_seed(args.seed)
        set_reproducibility(args.reproduce, args.seed, args.benchmark)

    # Meta-device streaming construction of the LeoLayer backbone. When
    # --init-model-with-meta-device is set (megatron_fsdp path), the LeoLayerBlock
    # __init__ wrapper builds all LeoLayer unit modules on meta (zero RNG / zero
    # GPU at build time); MegatronFSDP then materializes them shard-by-shard via
    # the injected reset_parameters, cutting the iter-1 build peak (PROGRESS §9.D).
    # No-op when the flag is off. See hymm/ptm_v2/meta_init_patch.py.
    from hymm.ptm_v2.meta_init_patch import set_meta_init_enabled
    set_meta_init_enabled(
        getattr(args, "init_model_with_meta_device", False)
        and getattr(args, "use_megatron_fsdp", False)
    )

    config, text_branch_config, audio_branch_config = config_from_args(args)

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
            # pipline stage send/recv tensor is \
            # [hidden_states, txt_hidden_states, audio_hidden_states, timestep_states, audio_timestep_states, ut, aut, middle_layer_hidden_states, repa_feats, zero_timestep_states]
            # audio_hidden_states, audio_timestep_states, aut is pp dummy tensor if only t2i task
            # middle_layer_hidden_states, repa_feats is pp dummy tensor if not use_repa
            # zero_timestep_states is a pp dummy tensor unless the r2v cond-vae path is active
            tensor_shapes = [()] * 10

            return tensor_shapes

        schedule.get_tensor_shapes = get_tensor_shapes
        # When get_shapes_from_dataloader is enabled, the interleaved schedule asks
        # the model to derive the PP send/recv tensor shapes from batch metadata
        # This skips the P2P shape-negotiation round. The callback unwraps the (batch, precomputed)
        # tuple produced by the encode-balance iterator wrapper
        if getattr(config, "get_shapes_from_dataloader", False):
            from angelptm.megatron.core.models.leo.leo_utils.pp_tensor_shapes import (
                compute_pp_tensor_shapes,
            )
            _cp_size = parallel_state.get_context_parallel_world_size()
            _txt_cfg = text_branch_config
            _audio_cfg = audio_branch_config

            def _compute_pp_tensor_shapes_fn(item):
                batch = item[0] if (isinstance(item, tuple) and len(item) == 2
                                    and isinstance(item[0], dict)) else item
                return compute_pp_tensor_shapes(batch, config, _cp_size,
                            txt_config=_txt_cfg, audio_config=_audio_cfg,
                            cond_vae_mask=compute_cond_vae_mask(batch))

            config.compute_pp_tensor_shapes_fn = _compute_pp_tensor_shapes_fn

            if getattr(config, "pp_multi_tensor_concat_uint8", False):
                from megatron.core.pipeline_parallel import p2p_communication as p2p_comm
                from angelptm.megatron.core.models.leo.leo_utils.pp_tensor_shapes import (
                    compute_pp_tensor_dtypes,
                )

                def get_tensor_dtypes(N, pipeline_dtype):
                    dtypes = compute_pp_tensor_dtypes()
                    assert len(dtypes) == N, (
                        f"leo pp_tensor_dtypes length {len(dtypes)} != payload tensor count {N}"
                    )
                    return dtypes

                p2p_comm.get_tensor_dtypes = get_tensor_dtypes


    layer_spec = get_leo_layer_block_spec(
        config=config,
        txt_config=text_branch_config,
        audio_config=audio_branch_config,
        vp_stage=vp_stage,
    )


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

    model = LeoModel(
        config=config,
        layer_spec=layer_spec,
        txt_config=text_branch_config,
        audio_config=audio_branch_config,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        dtype=torch.bfloat16, # hardcode dtype to bfloat16
        device=torch.device("cuda", args.local_rank),
    )

    # Only build VAE / text-encoder / etc. once (on the first VP chunk). For the
    # non-VPP path is_first_chunk is always True, so behavior is unchanged.
    if is_first_chunk:
        build_extra_model()

    # Model Reproducibility.
    if args.reproduce and hasattr(model, "enable_deterministic"):
        model.enable_deterministic()

    if torch.distributed.get_rank() == 0:
        print(ParameterSummary.from_module(model).as_table())

    # Install MFU hook if needed by --leo-mfu-log or --dp-load-balance-flops-proxy=mfu_aligned
    # (both consume the cached LeoArchConfig). Wrapper is a no-op when leo_mfu_log=off.
    _need_mfu_arch = (
        getattr(args, 'leo_mfu_log', False)
        or getattr(args, 'dp_load_balance_flops_proxy', 'legacy_lsq') == 'mfu_aligned'
    )
    if _need_mfu_arch:
        from angelptm.megatron.core.models.leo.mfu import install_mfu_hook
        install_mfu_hook(
            config=config,
            text_branch_config=text_branch_config,
            audio_branch_config=audio_branch_config,
        )

    return model


def config_from_args(args):
    """ transfusion config """

    model_name = args.model_name.split(".")[-1]

    from hymm.models.diffusion.leo_config import LeoConfig as TorchLeoConfig, core_model_config_from_args
    model_config_dict = core_model_config_from_args(args)
    model_config = TorchLeoConfig.from_name(model_name, **model_config_dict)

    if args.per_stage_recompute_num_layers is not None and args.pipeline_model_parallel_size > 1:
        per_stage_recompute_num_layers = args.per_stage_recompute_num_layers
        pp_size = args.pipeline_model_parallel_size
        assert args.per_stage_recompute_num_layers and len(per_stage_recompute_num_layers) == pp_size, (
            f'--per-stage-recompute-num-layers length ({len(args.per_stage_recompute_num_layers)}) '
            f'must equal --pipeline-model-parallel-size ({pp_size}). '
            f'Got values: {per_stage_recompute_num_layers}'
        )
        model_config.per_stage_recompute_num_layers = per_stage_recompute_num_layers

    # read config and mapping the config to mcore
    leo_a2a_enabled = getattr(args, "overlap_moe_expert_parallel_comm", False)
    _dense_overrides = {}
    if leo_a2a_enabled:
        args.overlap_moe_expert_parallel_comm_support_full_recompute = True
        # Dense branches (text / audio) don't drain the delayed wgrad queue, so
        # disable overlap on them (legacy_gemm_delay_wgrad is stamped main-only below).
        _dense_overrides["overlap_moe_expert_parallel_comm"] = False

    transformer_config = core_transformer_config_from_args(args)
    transformer_config.pipeline_dtype = torch.float32

    # if there are duplicate keys, model_config overwrites transformer_config.__dict__
    config = LeoConfig(**{**transformer_config.__dict__, **model_config.__dict__})
    # Plain CLI flag, not a LeoConfig field: stamp it onto the main config
    # so upstream moe/experts.py + leo_layer gate on it (dense off).
    config.legacy_gemm_delay_wgrad = getattr(args, "legacy_gemm_delay_wgrad", False)

    if args.text_branch_model_name is not None:
        text_branch_config_dict = core_model_config_from_args(args, prefix="text_branch_")
        text_branch_model_name = args.text_branch_model_name.split(".")[-1]
        text_branch_model_config = TorchLeoConfig.from_name(text_branch_model_name, **text_branch_config_dict)
        text_branch_config = LeoConfig(**{
            **transformer_config.__dict__,
            **text_branch_model_config.__dict__,
            **_dense_overrides,
        })
    else:
        text_branch_config = None

    if args.audio_branch_model_name is not None:
        audio_branch_config_dict = core_model_config_from_args(args, prefix="audio_branch_")
        audio_branch_model_name = args.audio_branch_model_name.split(".")[-1]
        audio_branch_model_config = TorchLeoConfig.from_name(audio_branch_model_name, **audio_branch_config_dict)
        audio_branch_config = LeoConfig(**{
            **transformer_config.__dict__,
            **audio_branch_model_config.__dict__,
            **_dense_overrides,
        })
    else:
        audio_branch_config = None

    if torch.distributed.get_rank() == 0:
        print_args("model config", config)
    return config, text_branch_config, audio_branch_config


def _leo_scalar_update_and_pack(loss_dict, consumed_metrics):
    args = get_args()
    scalar_state = global_vars.get_scalar_state()
    loss = loss_dict["loss"].float()
    is_update_step = scalar_state.train_steps % args.gradient_accumulation_steps == 0
    scalar_state.update(loss_dict, consumed_metrics, 0.0, is_update_step, 0.0)
    return loss, {}


def loss_func(loss_dict: dict[str, torch.Tensor], output_tensor: torch.Tensor, consumed_metrics: dict):
    _ = output_tensor  # OFF path: loss_dict is partial-bound; output_tensor unused.
    return _leo_scalar_update_and_pack(loss_dict, consumed_metrics)


def forward_step(data_iterator, model, return_schedule_plan=False, pre_processed_batch=None):
    args = get_args()
    timers = get_timers()

    # timers("batch-generator", log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
    # get_shapes_from_dataloader: the schedule already advanced the iterator in
    # get_next_batch (to derive the PP comm shapes) and hands the same item back
    # here. Reusing it avoids a second next() that would desync data / RNG.
    if pre_processed_batch is not None:
        item = pre_processed_batch
    else:
        item = next(data_iterator)
    # timers("batch-generator").stop(barrier=args.leo_timing_barrier)
    device = torch.device("cuda", args.local_rank)

    # When an encode-balance flag is on, train_step wraps the iterator so that
    # each yielded element is (batch_dict, precomputed_dict). When off, the
    # iterator yields the raw batch dict as before. Detect the wrapping here.
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
        batch, precomputed = item
        # reload precomputed to device (only pp stage 0)
        if precomputed:
            from angelptm.megatron.core.models.leo.leo_utils.encode_balance import (
                reload_precomputed_to_device,
            )
            precomputed = reload_precomputed_to_device(precomputed, device)
    else:
        batch = item
        precomputed = None

    # timers("prepare-model-inputs", log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
    unwrapped_model = unwrap_model(model)
    # VPP: with --single-dataloader-multi-virtual-pipeline-stages, the interleaved
    # schedule calls forward_step once per VP chunk on the same microbatch. Only the
    # first chunk (vp_stage in {None, 0}) should run VAE encode / add-noise; the later
    # chunks must skip them to avoid consuming extra RNG (which would shift the loss)
    # and redundant compute. vp_stage is read from the model object (aligned with
    # pretrain_audio), and is None when VPP is off — so this is a no-op without VPP.
    vp_stage = getattr(unwrapped_model, "vp_stage", None)
    is_vpp_first_chunk = (vp_stage is None or vp_stage == 0)
    with torch.autocast(device_type="cuda", enabled=False):
        # don't use skip_vae_encode, prepare_model_inputs use p_state.pp_rank to determine if skip vae encode
        model_input_kwargs, bsz, seqlen = prepare_model_inputs(
            batch,
            device,
            model_config=unwrapped_model._config,
            precomputed=precomputed,
            is_vpp_first_chunk=is_vpp_first_chunk,
        )
        # Forward the precomputed grouping-invariant counts (set by
        # dp_load_balance.set_micro_batch_counts when use_global_batch_count_average is on).
        if "micro_batch_image_count" in batch:
            model_input_kwargs.update(dict(
                micro_batch_image_count=batch["micro_batch_image_count"],
                micro_batch_audio_count=batch["micro_batch_audio_count"],
                micro_batch_repa_count=batch["micro_batch_repa_count"],
            ))
    # timers("prepare-model-inputs").stop(barrier=args.leo_timing_barrier)
    consumed_metrics = {
        batch["dataset_tag"][0]: {
            "samples": batch["n_samples"].sum().item(),
            "tokens": bsz * seqlen
        }
    }

    if return_schedule_plan:
        # ON path returns a schedule plan plus a loss closure; the closure reads
        # the last-stage loss stash and reuses the OFF-path bookkeeping helper.
        schedule_plan = model.build_schedule_plan(**model_input_kwargs)

        def _overlap_loss_func(output_tensor):
            """Return OFF-compatible ``(loss_tensor, {})`` for ON path."""
            _ = output_tensor  # raw scalar; we read loss_dict from stash.
            loss_dict = getattr(
                schedule_plan._model_chunk_state, "last_losses_dict", None
            )
            if loss_dict is None:
                # PP-middle: no stash; return-shape just satisfies the contract.
                _zero = torch.zeros((), device=torch.cuda.current_device())
                return _zero.float(), {}
            return _leo_scalar_update_and_pack(loss_dict, consumed_metrics)

        return schedule_plan, _overlap_loss_func

    # timers("model-forward", log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
    if return_schedule_plan:
        schedule_plan = unwrapped_model.build_schedule_plan(**model_input_kwargs)

        def schedule_plan_loss(output_tensor):
            return loss_func(
                schedule_plan.state.last_losses_dict,
                output_tensor,
                consumed_metrics=consumed_metrics,
            )

        return schedule_plan, schedule_plan_loss

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        output_tensor, loss_dict = model(**model_input_kwargs)
    # timers("model-forward").stop(barrier=args.leo_timing_barrier)
    assert_pp_payload_shape_matches_analytic(model, batch, output_tensor)
    return output_tensor, partial(loss_func, loss_dict, consumed_metrics=consumed_metrics)


def _get_megatron_fsdp_model(model):
    current = model
    while current is not None:
        if hasattr(current, "begin_unit_major_iteration"):
            return current
        next_model = getattr(current, "module", None)
        if next_model is current:
            break
        current = next_model
    raise RuntimeError(
        "fsdp-gradient-accumulation-steps < gradient-accumulation-steps "
        "requires the Megatron-FSDP wrapper."
    )


def _get_fsdp_communication_windows(args, num_microbatches):
    """Number of communication windows the step's microbatches are split into.

    ``--fsdp-gradient-accumulation-steps`` is a WINDOW COUNT, not a microbatch
    count: each window all-gathers an FSDP unit's parameters once and reuses them
    for every microbatch in that window, so fewer windows means less parameter
    communication and a higher activation peak. Unset means one window per
    microbatch, i.e. Megatron's ordinary schedule.
    """
    communication_windows = getattr(args, "fsdp_gradient_accumulation_steps", None)
    if communication_windows is None:
        communication_windows = num_microbatches
    if not isinstance(communication_windows, int) or communication_windows < 1:
        raise ValueError(
            "fsdp-gradient-accumulation-steps must be a positive integer, got "
            f"{communication_windows!r}."
        )
    if communication_windows > num_microbatches:
        raise ValueError(
            "fsdp-gradient-accumulation-steps must not exceed the number of "
            f"microbatches, got {communication_windows} > {num_microbatches}."
        )
    return communication_windows


def _fsdp_window_bounds(num_microbatches, communication_windows):
    """``[(start, end), ...]`` half-open microbatch ranges, one per window.

    The count need not divide ``num_microbatches``. The remainder is given to the
    LAST windows, so the first window is the smallest: a step's first window runs
    while the caching allocator and the pinned pool are still growing, and that is
    where a peak is most likely to fail. 8 microbatches over 3 windows -> sizes
    ``[2, 3, 3]``.

    Peak memory is set by the LARGEST window, so an uneven split costs the same
    peak as rounding the window size up, while issuing fewer all-gathers than
    rounding it down.
    """
    base, remainder = divmod(num_microbatches, communication_windows)
    first_grown = communication_windows - remainder
    bounds = []
    start = 0
    for index in range(communication_windows):
        size = base + (1 if index >= first_grown else 0)
        bounds.append((start, start + size))
        start += size
    assert start == num_microbatches, (start, num_microbatches)
    return bounds


def _get_fsdp_communication_window_size(args, num_microbatches):
    """Largest window size — the memory-relevant one. 1 means no unit-major run."""
    windows = _get_fsdp_communication_windows(args, num_microbatches)
    return max(end - start for start, end in
               _fsdp_window_bounds(num_microbatches, windows))


def _get_raw_leo_model(model):
    current = unwrap_model(model)
    while current is not None and not hasattr(current, "build_schedule_plan"):
        next_model = getattr(current, "module", None)
        if next_model is current:
            break
        current = next_model
    if current is None or not hasattr(current, "build_schedule_plan"):
        raise RuntimeError("Unit-major scheduling requires a LeoModel with build_schedule_plan().")
    return current


def _unit_major_forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    config,
):
    """Run a PP=1 Leo update in FSDP-unit-major communication windows."""
    if isinstance(model, list):
        if len(model) != 1:
            raise ValueError("Unit-major FSDP scheduling does not support model chunking.")
        model = model[0]
    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            raise ValueError("Unit-major FSDP scheduling does not support VPP data iterators.")
        data_iterator = data_iterator[0]

    args = get_args()
    window_bounds = _fsdp_window_bounds(
        num_microbatches, _get_fsdp_communication_windows(args, num_microbatches))
    if max(end - start for start, end in window_bounds) == 1:
        raise RuntimeError("Unit-major scheduling requires a communication window larger than one.")
    if not getattr(args, "use_megatron_fsdp", False):
        raise RuntimeError("Unit-major scheduling requires --use-megatron-fsdp.")
    if getattr(args, "pipeline_model_parallel_size", 1) != 1:
        raise RuntimeError("Unit-major scheduling currently requires pipeline-model-parallel-size=1.")
    if getattr(args, "recompute_granularity", None) != "full":
        raise RuntimeError(
            "Unit-major scheduling requires --recompute-granularity full so activations "
            "are rebuilt one FSDP unit at a time during backward."
        )
    if getattr(config, "overlap_moe_expert_parallel_comm", False):
        raise RuntimeError(
            "Unit-major FSDP scheduling is incompatible with overlap-moe-expert-parallel-comm."
        )
    if getattr(config, "fp8", False):
        raise RuntimeError("Unit-major FSDP scheduling does not support FP8 parameters.")
    if getattr(args, "megatron_fsdp_enable_fine_grained_param_gather", False):
        raise RuntimeError(
            "Unit-major FSDP scheduling requires megatron-fsdp-enable-fine-grained-param-gather=false."
        )

    from megatron.core.pipeline_parallel.schedules import (
        forward_step_calc_loss,
        set_current_microbatch,
    )
    from megatron.core.pipeline_parallel.utils import set_streams

    # SchedulePlan nodes retain this stream through backward's record_stream().
    set_streams()

    fsdp_model = _get_megatron_fsdp_model(model)
    raw_model = _get_raw_leo_model(model)
    fsdp_model.begin_unit_major_iteration()
    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    for window_start, window_end in window_bounds:
        plans = []
        plan_loss_functions = []
        layer_inputs = []

        for microbatch_id in range(window_start, window_end):
            set_current_microbatch(raw_model, microbatch_id)
            plan, plan_loss = forward_step_func(
                data_iterator,
                raw_model,
                return_schedule_plan=True,
            )
            plan.record_current_stream()
            plans.append(plan)
            plan_loss_functions.append(plan_loss)
            layer_inputs.append(plan.pre_process.forward())

        layer_grads = []
        try:
            num_layers = plans[0].num_layers()
            if any(plan.num_layers() != num_layers for plan in plans[1:]):
                raise RuntimeError("All microbatches in a communication window need identical Leo layers.")

            for layer_idx in range(num_layers):
                unit = plans[0].get_layer(layer_idx).layer
                with torch.autograd.profiler.record_function(
                    f"FSDPUnitMajor::forward_unit[{layer_idx}]"
                ):
                    fsdp_model.begin_unit_major_forward(unit)
                    try:
                        if layer_idx + 1 < num_layers:
                            next_unit = plans[0].get_layer(layer_idx + 1).layer
                            with torch.autograd.profiler.record_function(
                                f"FSDPUnitMajor::prefetch_forward_unit[{layer_idx + 1}]"
                            ):
                                fsdp_model.prefetch_unit_major_forward(next_unit)
                        for microbatch_idx, f_layer in enumerate(
                            plan.get_layer(layer_idx) for plan in plans
                        ):
                            set_current_microbatch(raw_model, window_start + microbatch_idx)
                            layer_inputs[microbatch_idx], _ = type(f_layer).run(
                                f_layer,
                                None,
                                f_input=layer_inputs[microbatch_idx],
                            )
                    finally:
                        fsdp_model.end_unit_major_forward(unit)

            for plan, plan_loss, layer_output in zip(plans, plan_loss_functions, layer_inputs):
                raw_loss = plan.post_process.forward(layer_output)
                scaled_loss, num_tokens = forward_step_calc_loss(
                    model,
                    raw_loss,
                    plan_loss,
                    config,
                    None,
                    False,
                    num_microbatches,
                    forward_data_store,
                )
                total_num_tokens += num_tokens
                with fsdp_model.unit_major_loss_backward():
                    torch.autograd.backward(scaled_loss)
                output_grad = plan.post_process.get_grad()
                plan.post_process.mark_backward_consumed(output_grad)
                layer_grads.append(plan.post_process.backward(output_grad))

            for layer_idx in range(num_layers - 1, -1, -1):
                unit = plans[0].get_layer(layer_idx).layer
                with torch.autograd.profiler.record_function(
                    f"FSDPUnitMajor::backward_unit[{layer_idx}]"
                ):
                    fsdp_model.begin_unit_major_backward(unit, is_last_microbatch=False)
                    try:
                        if layer_idx > 0:
                            next_unit = plans[0].get_layer(layer_idx - 1).layer
                            with torch.autograd.profiler.record_function(
                                f"FSDPUnitMajor::prefetch_backward_unit[{layer_idx - 1}]"
                            ):
                                fsdp_model.prefetch_unit_major_backward(next_unit)
                        for microbatch_idx, plan in enumerate(plans):
                            set_current_microbatch(raw_model, window_start + microbatch_idx)
                            b_layer = plan.get_layer(layer_idx)
                            _, layer_grads[microbatch_idx] = type(b_layer).run(
                                None,
                                b_layer,
                                b_grad=layer_grads[microbatch_idx],
                                is_last_layer_in_bwd=False,
                            )
                        fsdp_model.finalize_unit_major_backward(unit)
                    finally:
                        fsdp_model.end_unit_major_backward(unit)

            fsdp_model.finish_unit_major_backward_pipeline()

            for plan, layer_grad in zip(plans, layer_grads):
                plan.pre_process.backward(layer_grad)
        finally:
            try:
                for plan in plans:
                    try:
                        plan.wait_current_stream()
                    finally:
                        plan.release_state()
            finally:
                fsdp_model.clear_unit_major_prefetches()

    if config.finalize_model_grads_func is not None:
        config.finalize_model_grads_func([model], total_num_tokens)
    return forward_data_store


def extra_args_provider(parser: argparse.ArgumentParser):
    from ..config import add_core_args, add_evaluation_args
    from angelptm.megatron.training.arguments import add_ptm_extra_args

    parser = add_core_args(parser, ptm="v2")
    parser = _add_leo_extra_args(parser)
    add_ptm_extra_args(parser)

    return parser

def _leo_extra_args_validator(args, args_defaults):
    _ = args_defaults

    # Validate Leo overlap moe expert parallel comm args
    from angelptm.megatron.core.models.leo.ep_1f1b_overlap.validation import (
        validate_leo_a2a_overlap_args,
    )
    validate_leo_a2a_overlap_args(args)

def _dp_load_balance_payload(args, data_iterator, keep_on_device_keys=None):
    """Build the VPP-aware prefetch closure and delegate the reshuffle (with
    optional --dplb-overlap-plan overlap) to
    :func:`dp_load_balance.dp_load_balance_payload`.

    The prefetch closure (``_prefetch_microbatches_from_publisher``) is training-loop
    glue — it drains the VPP DataPublisher/subscribers — so it stays here; the
    reshuffle / overlap orchestration lives in ``dp_load_balance``.

    ``keep_on_device_keys`` (encode-balance path) keeps those top-level batch keys
    on the GPU through the reshuffle so the downstream intra-dp encode can run on
    them without a re-upload.

    A stratified exchange group (``--dp-load-balance-group-size``) is honoured on
    both the inline and the ``--dplb-overlap-plan`` path;
    ``dp_load_balance_payload_stratified`` forwards to the upstream driver when no
    stratified group is configured.
    """
    num_mb = get_num_microbatches()
    prefetch = lambda: _prefetch_microbatches_from_publisher(  # noqa: E731
        data_iterator, num_mb, drain=True)
    return dp_load_balance_payload_stratified(
        args, prefetch, keep_on_device_keys=keep_on_device_keys)


def _dp_load_balance_carry_audio(args, pre_batches, audio_precomputed):
    """DP reshuffle that carries each sample's buddy audio latents along.

    Keeps :func:`dp_load_balance` pure: the buddy latents are stuffed into the
    batch dict, ride the reshuffle, then are pulled back out keyed by the new mb
    index. Returns (reshuffled_pre_batches, audio_precomputed_by_new_mb).
    """
    from hymm.utils.helpers import multi_pattern_match

    # pop audios if audio lantents are already encoded
    if args.audio_buddy_encode_balance:
        for b in pre_batches:
            if multi_pattern_match(b["dataset_tag"][0], ["t2a*"]):
                b.pop("audios", None)

    for mb, pc in audio_precomputed.items():
        pre_batches[mb]["__audio_precomputed__"] = pc

    # Keep on GPU the media that intra-dp encodes
    keep_on_device_keys = {"images"}
    if args.intra_dp_encode_balance or args.audio_buddy_encode_balance:
        keep_on_device_keys.add("audios") # TODO: audio maybe pre-computed
    if args.audio_buddy_encode_balance:
        keep_on_device_keys.add("__audio_precomputed__")

    # dp_load_balance_stratified forwards to the upstream dp_load_balance when
    # --dp-load-balance-group-size is 0 (the default), so this is behaviour-neutral
    # unless a stratified exchange group is configured.
    pre_batches = list(dp_load_balance_stratified(
        args, pre_batches, keep_on_device_keys=keep_on_device_keys))
    carried = {}
    for mb, b in enumerate(pre_batches):
        pc = b.pop("__audio_precomputed__", None)
        if pc is not None:
            carried[mb] = pc
    return pre_batches, carried


def _run_encode_load_balance(args, model, data_iterator,
                             need_audio_buddy_balance, need_intra_dp_balance,
                             need_dp_load_balance, need_fsdp_encode_balance=False):
    """Pre-fetch one step of micro-batches and distribute their encoder work as
    independent phases, ordered by their layout assumptions:
      1. audio-buddy  — raw single-task-per-DP layout (before dp-balance).
      2. dp-balance   — reshuffle across DP; buddy latents carried along.
      3. image-buddy  — raw visual payloads are now unique per DP and can be
                        encoded by idle DP helpers before the latents return.
      4. text-buddy   — packed token payloads are LPT-balanced across DP helpers.
      5. intra-dp     — per-mb encode on the (mixed-task) batches.

    Returns (pre_batches, precomputed_per_mb).
    """
    from angelptm.megatron.core.models.leo.leo_utils.encode_balance import (
        audio_buddy_balance,
        intra_dp_balance,
        merge_precomputed,
        cp_broadcast_precomputed,
        offload_media_to_cpu,
        offload_precomputed_to_cpu,
    )
    from angelptm.megatron.core.models.leo.leo_utils.fsdp import (
        image_buddy_balance,
        merge_precomputed_strict,
        offload_media_to_cpu_with_precomputed,
        text_buddy_balance,
    )

    # The overlap prefetcher runs the whole reshuffle on a background worker, so the
    # audio-buddy GPU encode — which must run on the RAW batches BEFORE the reshuffle
    # and carry its latents along — cannot be hosted there. The two are mutually
    # exclusive; audio-buddy therefore always takes the inline (non-overlap) path.
    assert not (dplb_overlap_enabled(args) and need_audio_buddy_balance), (
        "--dplb-overlap-plan is incompatible with --audio-buddy-encode-balance: "
        "the audio-buddy encode runs on raw batches before the reshuffle and cannot "
        "be overlapped in the background dp-load-balance worker."
    )
    # The buddy exchange group must be built from the PRE-reshuffle batches on the
    # main thread; the overlap worker never exposes them (it reads data_iterator
    # itself and returns already-reshuffled batches).
    assert not (dplb_overlap_enabled(args) and need_fsdp_encode_balance), (
        "--dplb-overlap-plan is incompatible with --fsdp-encode-balance: the buddy "
        "exchange group is derived from the fixed task-to-DP-rank allocation visible "
        "only in the raw pre-reshuffle batches."
    )

    timers = get_timers()
    num_mb = get_num_microbatches()

    audio_precomputed = {}
    buddy_dp_group = None
    if need_dp_load_balance and dplb_overlap_enabled(args):
        # Keep the media intra-dp encodes resident on GPU through the reshuffle.
        # The reshuffle reads straight from data_iterator and is overlap-capable.
        timers("dp-balance",
                log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
        pre_batches = _dp_load_balance_payload(args, data_iterator)
        timers("dp-balance").stop(barrier=args.leo_timing_barrier)
    else:
        timers("prefetch-batches",
               log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
        pre_batches = _prefetch_microbatches_from_publisher(data_iterator, num_mb, drain=True)
        timers("prefetch-batches").stop(barrier=args.leo_timing_barrier)

        # Create the buddy communication domain while the fixed task-to-DP-rank
        # allocation is still visible; DP-LB may change this rank's local task next.
        if need_fsdp_encode_balance:
            buddy_dp_group = _resolve_buddy_encode_group(args, pre_batches)

        if need_audio_buddy_balance:
            timers("encode-audio-buddy",
                log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
            audio_precomputed = audio_buddy_balance(pre_batches)
            timers("encode-audio-buddy").stop(barrier=args.leo_timing_barrier)

        if need_dp_load_balance:
            timers("dp-balance",
                   log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
            pre_batches, audio_precomputed = _dp_load_balance_carry_audio(
                args, pre_batches, audio_precomputed)
            timers("dp-balance").stop(barrier=args.leo_timing_barrier)

    image_precomputed = {}
    text_precomputed = {}
    if need_fsdp_encode_balance:
        full_dp_group = parallel_state.get_data_parallel_group()
        timers("encode-image-buddy",
               log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
        image_precomputed = image_buddy_balance(
            pre_batches,
            model_config=unwrap_model(model[0])._config,
            dp_group=buddy_dp_group,
            full_dp_group=full_dp_group,
        )
        timers("encode-image-buddy").stop(barrier=args.leo_timing_barrier)

        timers("encode-text-buddy",
               log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
        text_precomputed = text_buddy_balance(
            pre_batches,
            dp_group=buddy_dp_group,
            full_dp_group=full_dp_group,
        )
        timers("encode-text-buddy").stop(barrier=args.leo_timing_barrier)

    if need_intra_dp_balance:
        timers("encode-intra-dp",
               log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
        other_precomputed = intra_dp_balance(
            pre_batches,
            model_config=unwrap_model(model[0])._config,
            audio_buddy_enabled=need_audio_buddy_balance,
        )
        timers("encode-intra-dp").stop(barrier=args.leo_timing_barrier)
    else:
        other_precomputed = {}

    if need_fsdp_encode_balance:
        # Four contributing phases, and a duplicate key means the same latent was
        # encoded twice — fail instead of silently picking one.
        precomputed_per_mb = merge_precomputed_strict(
            num_mb, audio_precomputed, image_precomputed, text_precomputed, other_precomputed)
    else:
        precomputed_per_mb = merge_precomputed(
            num_mb, audio_precomputed, other_precomputed)

    # Results were gathered only to (pp0, cp0); share with the other pp0 cp ranks so
    # every cp rank feeds the model the same pre-noise latents.
    p_state = global_vars.get_parallel_state()
    if p_state.pp_rank == 0 and p_state.cp_size > 1:
        timers("encode-cp-broadcast", log_level=args.leo_timing_log_level).start(barrier=False)
        precomputed_per_mb = cp_broadcast_precomputed(precomputed_per_mb, p_state.cp_group)
        timers("encode-cp-broadcast").stop(barrier=False)

    # encode offload raw media + precomputed latents to cpu (keep mb0 resident)
    timers("encode-offload", log_level=args.leo_timing_log_level).start(barrier=False)
    if need_fsdp_encode_balance:
        # PP0's raw video_last_frames is dead too where image buddy supplied latents.
        offload_media_to_cpu_with_precomputed(pre_batches, precomputed_per_mb)
    else:
        offload_media_to_cpu(pre_batches)
    offload_precomputed_to_cpu(precomputed_per_mb, keep_first=True)
    timers("encode-offload").stop(barrier=False)
    return pre_batches, precomputed_per_mb


def _wrap_fwd_iter_for_vpp(args, payload, data_iterator, ctx):
    """Wrap `payload` into `fwd_data_iter`. VPP (list) -> one fresh iter per chunk;
    non-VPP -> a single iter."""
    if isinstance(data_iterator, list):
        assert getattr(args, "single_dataloader_multi_virtual_pipeline_stages", False), (
            f"{ctx} + VPP requires --single-dataloader-multi-virtual-pipeline-stages "
            "(else chunks 1..VPP-1 own independent dataloaders and desync from chunk-0's prefetch)."
        )
        return [iter(payload) for _ in data_iterator]
    return iter(payload)


def _drain_vpp_subscribers(data_iterator, n):
    """Drop the `n` batches the DataPublisher pushed into each VPP DataSubscriber
    during this step's balance-path prefetch.

    Under VPP + single-dataloader, balance paths feed forward from a separate reshuffled
    payload, so subscribers are never read and their ``.data`` lists (and pinned-memory
    tensors) leak by `n` per step. Pop those dead copies to mirror normal consumption.
    No-op for non-VPP iterators or subscribers without a ``.data`` list.
    """
    if not isinstance(data_iterator, list):
        return
    for sub in data_iterator[1:]:
        data = getattr(sub, "data", None)
        if not isinstance(data, list):
            continue
        # Pop from the front to mirror DataSubscriber.__next__ (FIFO pop(0)); only the
        # batches this step pushed (the trailing `n`) are ours to drop. Guard length.
        drop = min(n, len(data))
        if drop:
            del data[:drop]


def _prefetch_microbatches_from_publisher(data_iterator, n, *, drain=True):
    """Pull `n` micro-batches from chunk-0 (the VPP DataPublisher) and, by default,
    drain the copies it pushed into chunk-1..VPP-1 subscribers — they're never read
    on balance paths and would leak pinned host RAM. Funnel all balance
    prefetches through here so new paths can't forget to drain. Set ``drain=False``
    only if forward will actually consume from the subscribers.
    """
    src_iter = data_iterator[0] if isinstance(data_iterator, list) else data_iterator
    pre_batches = [next(src_iter) for _ in range(n)]
    if drain:
        _drain_vpp_subscribers(data_iterator, n)
    return pre_batches


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
    ] + [
        f"{dataset_tag}_audio_loss" 
        for dataset_tag in mm_state.all_dataset_keys
        if "2a" in dataset_tag or "2va" in dataset_tag
    ] 
    if args.use_repa:
        loss_names.append("neg_repa_loss")
    if args.moe_aux_loss_coeff > 0:
        loss_names.append("moe_loss")
    # Optionally average diffusion loss over global image sample count (all ranks)
    if getattr(args, "use_global_diffusion_loss_average", False):
        loss_names.append("global_image_video_loss")
        if args.audio_branch_model_name is not None:
            loss_names.append("global_audio_loss")
        if args.use_repa:
            loss_names.append("global_neg_repa_loss")

    _need_audio_buddy_balance = getattr(args, "audio_buddy_encode_balance", False)
    _need_intra_dp_balance = getattr(args, "intra_dp_encode_balance", False)
    _need_dp_load_balance = getattr(args, "dp_load_balance", False)
    # fsdp_encode_balance jointly enables the image-VAE and packed-text buddy phases.
    _need_fsdp_encode_balance = getattr(args, "fsdp_encode_balance", False)

    # Create the stratified DP-LB exchange group once, here on the main thread with
    # every rank synchronized. Group creation is the only collective in the resolve
    # path; doing it eagerly is what lets --dp-load-balance-group-size coexist with
    # --dplb-overlap-plan (the background worker then only hits the cached path).
    # This is the first point where the fixed task-to-rank allocation is available.
    if _need_dp_load_balance and not _dplb_group_warmed["done"]:
        warmup_dplb_schedule_group(args)
        _dplb_group_warmed["done"] = True
    # any encode-balance flag enables the (pre-fetching) encode-balance path.
    _need_encode_balance = (
        _need_audio_buddy_balance
        or _need_intra_dp_balance
        or _need_fsdp_encode_balance
    )


    # Cache the encode result so a rerun on the same step does not consume extra
    # batches; reset after the iteration that actually advances.
    encode_lb_cache: dict = {"pre_batches": None, "precomputed_per_mb": None}

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

        if _need_encode_balance:
            if encode_lb_cache["pre_batches"] is None:
                timers("encode-balance-all",
                       log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
                encode_lb_cache["pre_batches"], encode_lb_cache["precomputed_per_mb"] = (
                    _run_encode_load_balance(
                        args, model, data_iterator,
                        _need_audio_buddy_balance, _need_intra_dp_balance,
                        _need_dp_load_balance, _need_fsdp_encode_balance))
                timers("encode-balance-all").stop(barrier=args.leo_timing_barrier)
            # Serve every vp chunk from this prefetched payload.
            payload = list(zip(
                encode_lb_cache["pre_batches"], encode_lb_cache["precomputed_per_mb"]))
            fwd_data_iter = _wrap_fwd_iter_for_vpp(args, payload, data_iterator, "encode-balance")
        elif _need_dp_load_balance:
            timers("dp-balance",
               log_level=args.leo_timing_log_level).start(barrier=args.leo_timing_barrier)
            # Reshuffle across DP. With --dplb-overlap-plan the work is hidden behind
            # the previous step's forward; forward is served from `payload`.
            payload = _dp_load_balance_payload(args, data_iterator)
            fwd_data_iter = _wrap_fwd_iter_for_vpp(args, payload, data_iterator, "dp_load_balance")
            timers("dp-balance").stop(barrier=args.leo_timing_barrier)
        else:
            fwd_data_iter = data_iterator

        # Forward/backward pass. A reduced FSDP communication-window count
        # selects Leo's unit-major schedule; the default remains Megatron's
        # regular microbatch-major schedule.
        num_microbatches = get_num_microbatches()
        if _get_fsdp_communication_window_size(args, num_microbatches) > 1:
            losses_reduced = _unit_major_forward_backward_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=fwd_data_iter,
                model=model,
                num_microbatches=num_microbatches,
                config=config,
            )
        else:
            losses_reduced = forward_backward_func(     # noqa
                forward_step_func=forward_step_func,
                data_iterator=fwd_data_iter,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=False,
                adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
            )

    # encode load balancing: drop our references to the gathered precomputed tensors
    # promptly so they are freed each step.
    if _need_encode_balance:
        encode_lb_cache["pre_batches"] = None
        encode_lb_cache["precomputed_per_mb"] = None
        fwd_data_iter = None

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


def _add_leo_extra_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enable-modulate-gate-fusion", 
        action="store_true", 
        default=False, 
        help="When enabled, use fused modulate/gate kernels")
    parser.add_argument(
        "--enable-async-cp",
        action="store_true",
        default=False,
        help="When enabled, use async context-parallel all-to-all overlap")
    parser.add_argument(
        "--cp-fused-qkv-a2a-max-tokens",
        type=int,
        default=0,
        help="Send q/k/v in one context-parallel all-to-all instead of three when a branch "
             "holds at most this many tokens per rank (0 disables; requires --enable-async-cp).")
    parser.add_argument(
        "--cp-skip-dummy-branch-a2a",
        action="store_true",
        default=False,
        help="Replace both context-parallel all-to-all with a bit-for-bit local reshape on "
             "all-dummy attention branches, which no rank reads (requires --enable-async-cp).")
    parser.add_argument(
        "--enable-fused-norm-merge-rope",
        action="store_true",
        default=False,
        help="When enabled, use fused norm/merge/RoPE optimized kernels")
    parser.add_argument(
        "--recompute-preprocess",
        action="store_true",
        default=False,
        help="Recompute the pre_process input-embedding build (first stage)")
    parser.add_argument(
        "--activation-offload-min-mb",
        type=float,
        default=1.0,
        help="Minimum tensor size (MB) for saved_tensors-hook activation "
        "offload (encoder and fused-transformer callable namespace).",
    )
    parser.add_argument(
        '--recompute-input-offload',
        action='store_true',
        help='Offload each full-recompute segment activation inputs '
             '(hidden/txt/audio) to pinned host between forward and backward, '
             'reloaded with cross-layer prefetch. The stage last (first-to-'
             'backward) segment is excluded.',
    )
    parser.add_argument(
        '--leo-ep-overlap-activation-offload',
        action='store_true',
        help='Leo a2a-overlap fwd->bwd activation offload, two tensor sets on the '
             'shared pinned pool + (mb,ln) prefetch registry: (a) the attn_head '
             'output triple, D2H at forward-collapse (instead of freed) and reloaded '
             'at recompute to feed attn_tail, which lets attn_head.recompute (the '
             'big self_attn graph) be DEFERRED to just before attn_head.backward, '
             'shrinking that graph live window to head.bwd only; (b) the manual-VJP '
             'img_post_combine saved tensors (y + gate clone), D2H after the forward '
             'gate kernel and reloaded at its backward. Value-preserving (bitwise). '
             'Requires --overlap-moe-expert-parallel-comm, --recompute-granularity=full '
             'and --activation-offload-pinned-slot.',
    )
    parser.add_argument(
        "--legacy-gemm-delay-wgrad",
        action="store_true",
        default=False,
        help="Defer wgrad on the non-TE (legacy) GEMM path (expert grouped_mm + "
             "attn q/k/v/o projections) so it can overlap DeepEP A2A / PP send. "
             "Requires --overlap-moe-expert-parallel-comm.")
    # Leo timing args
    parser.add_argument('--leo-timing-log-level', type=int,
                       default=2, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       'This parameter is compared with timing-log-level. '
                       'If it is less than or equal to timing-log-level, time statistics will be printed'
                       'The default value of timing-log-level is 0.')
    parser.add_argument('--leo-timing-barrier', action='store_true',
                       help='If set, the start/stop of timers, args barrier is True')
    parser.add_argument(
        "--intra-dp-encode-balance",
        action="store_true",
        default=False,
        help="Within each DP, distribute text/image/audio encode across PP*CP ranks.",
    )
    parser.add_argument(
        "--audio-buddy-encode-balance",
        action="store_true",
        default=False,
        help="For t2a tasks, distribute audio_vae work across a buddy_group "
             "(t2a DP + helper DPs).",
    )
    parser.add_argument(
        "--fsdp-encode-balance",
        action="store_true",
        default=False,
        help="Jointly distribute post-DP-load-balance frozen VAE and packed text "
             "encoder work across DP helpers. Requires TP=PP=CP=1, fixed sampling, "
             "--vae-encode-type mode, --text-encoder-use-pack, and "
             "--text-encoder-pack-microbatches. Size the exchange group with "
             "--fsdp-encode-balance-group-size.",
    )
    parser.add_argument(
        "--leo-combine-recompute",
        action="store_true",
        default=False,
        help="Under --recompute-granularity=full, keep moe_combine / "
             "img_post_combine as plain autograd nodes and rebuild them with "
             ".recompute() in the backward chain, instead of the hand-written VJP "
             "nodes that capture _y / _gate_clone at forward. The manual nodes buy "
             "combineB||moe_mlpR overlap but hold ~33.6 GiB/rank (128 GPUs, window "
             "of 4 microbatches, 47 MoE layers) from forward to backward, which is "
             "wasted when the a2a-overlap schedule is off. Same math either way. "
             "Requires PP=1 and no --overlap-moe-expert-parallel-comm.",
    )
    parser.add_argument(
        "--intra-dp-encode-by-pp-stage",
        action="store_true",
        default=False,
        help="Assign encode work by PP stage: text-PP stages do text encode, "
             "image-PP stages do image VAE (in parallel), then audio VAE is split "
             "by sample across the whole DP. Requires intra-dp-encode-balance.",
    )
    parser.add_argument(
        "--intra-dp-text-pp-stages", type=str, default=None,
        help="Comma-separated PP stage indices that run text encode under "
             "intra-dp-encode-by-pp-stage, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--intra-dp-image-pp-stages", type=str, default=None,
        help="Comma-separated PP stage indices that run image VAE under "
             "intra-dp-encode-by-pp-stage, e.g. '4,5,6,7'.",
    )
    parser.add_argument(
        "--intra-dp-image-shape-batch", action="store_true", default=False,
        help="intra-dp-encode-by-pp-stage: batch same-shape images into one "
             "vae.encode (numerics user-validated; off = per-sample, bitwise).",
    )
    parser.add_argument('--leo-mfu-log', nargs='?', const='summary', default=None,
                       choices=['summary', 'breakdown'],
                       help='Enable real-time MFU logging (per log_interval). '
                            '"summary" (default when flag is given without value) '
                            'prints MFU only; "breakdown" additionally prints the '
                            'per-component cluster-step FLOPs breakdown '
                            '(sum over all DP replicas × all microbatches).')
    return parser


# =========================================================================
#       Entry point for training Leo with AngelPTM v2
# =========================================================================
def launch(frozen_args):
    from megatron.core.enums import ModelType
    from megatron.core.multimodal_parallel_state import maybe_initialize_model_parallel
    from megatron.training import inprocess_restart
    from megatron.training import pretrain

    # Optionally enable in-process restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    logger = logging.getLogger("megatron.core.utils")
    logger.setLevel(logging.WARN)

    # post_init_func is for compatibility with VLM
    pretrain(
        DatasetsProvider(),
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=frozen_args,
        extra_args_provider=extra_args_provider,
        extra_args_validator=_leo_extra_args_validator,
        store=store,
        post_init_func=lambda args: maybe_initialize_model_parallel(
            tp_size=args.tensor_model_parallel_size,
            pp_size=args.pipeline_model_parallel_size,
            cp_size=args.context_parallel_size,
            create_gloo_process_groups=args.enable_gloo_process_groups,
        ),
        callbacks=TrainerCallback(),
    )
