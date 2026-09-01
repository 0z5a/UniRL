import math
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from .global_vars import (
    get_args,
    get_parallel_state,
    get_logger,
    get_vae,
    get_denoiser,
    get_tkwrapper,
    get_scalar_state,
    get_mm_state,
    set_combined_iterator,
    set_mm_state,
)
from .parallel_states import ParallelState
from ..constants import VISION_ENCODER_META_INFO
from ..data_kits.combined_iterator import CombinedBatchIterator
from ..data_kits.mm_loader import load_dataloader
from ..utils.image_base import ImageInfo
from ..data_kits.utils import prepare_distributed_sampling, MultimodalTasksState
from ..models.autoencoders import vae_encode_and_add_noise
from ..models.autoregressive.flex_attn_layers import (
    create_text_image_mask_mod,
    create_batch_text_image_mask_mod,
    create_causal_mask_mod,
    create_batch_causal_mask_mod,
    create_interleave_mask_mod,
    create_batch_interleave_mask_mod,
)
from ..utils.file_utils import safe_dir
from ..utils.helpers import to_2tuple, multi_pattern_match
from ..utils.torch_utils import to_device, set_manual_seed

try:
    from megatron.core.packed_seq_params import PackedSeqParamsWithValidTokens
except ImportError:
    # For CI / no-Megatron runs
    class PackedSeqParamsWithValidTokens:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __getattr__(self, name):
            return self.kwargs[name]

def build_packed_seq_params(valid_local_token_mask, device):
    return PackedSeqParamsWithValidTokens(
        qkv_format=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        valid_local_token_mask=valid_local_token_mask.to(device),
        valid_total_num_tokens=int(valid_local_token_mask.sum().item()),
    )


def _extend_packed_seq_params_valid(packed, n_new: int, device, *, valid: bool):
    """Append n_new positions to MoE valid_local_token_mask (dummy=True, flex tail pad=False)."""
    if packed is None or n_new <= 0 or PackedSeqParamsWithValidTokens is None:
        return packed
    v = getattr(packed, "valid_local_token_mask", None)
    if v is None:
        return packed
    tail = torch.full((n_new,), valid, dtype=torch.bool, device=device)
    new_v = torch.cat([v.to(device), tail], dim=0)
    return build_packed_seq_params(new_v, device)


class DatasetsProvider:
    def __init__(self, task_info_dict=None):
        # Allowing custom task info dict for building dataloaders
        self.task_info_dict = task_info_dict
        # Used by mcore
        self.is_distributed = True

    def __call__(self, train_valid_test_num_samples):
        _ = train_valid_test_num_samples

        args = get_args()
        p_state: ParallelState = get_parallel_state()
        logger = get_logger()
        scalar_state = get_scalar_state()

        # before build datasets, set_manual_seed only used for dataloader set_worker_seed_builder  \
        # and dropout in model training. vae.sample and denoiser use the same random external generator
        set_manual_seed(args.seed + p_state.dp_rank)

        # Load sampling probabilities and dummy tokens info
        mm_state = MultimodalTasksState.from_args(args)
        # Distribute (multiple) training tasks among data parallel groups
        mm_state = prepare_distributed_sampling(args, mm_state, p_state, _logger=logger)
        # Load dataloaders
        mm_state, datasets, samplers, dataloaders = load_dataloader(
            args, mm_state, p_state, task_info_dict=self.task_info_dict
        )
        # Combine dataloaders into a single iterator
        combined_iterator = CombinedBatchIterator(
            ss=scalar_state,
            fast_shuffle=args.fast_shuffle,
            rank=p_state.dp_rank,
            world_size=p_state.dp_size,
            datasets=datasets,
            samplers=samplers,
            dataloaders=dataloaders,
            sampling_probs=mm_state.sampling_probs,
            initial_seed=args.seed,
            sampling_mode=mm_state.distributed_sampling_state.mode,
            cache_shuffle=args.cache_shuffle,
            fixed_key=mm_state.distributed_sampling_state.cur_key,
            fixed_key_group=mm_state.distributed_sampling_state.cur_key_group,
            distributed_sampling_state=mm_state.distributed_sampling_state,
            force_sync_shuffle=False,
            pack_buffer_factor=args.pack_buffer_factor,
            keys2id=mm_state.keys2id,
            dp_group=p_state.dp_group,
            resume_index_batch_sampler=args.resume_index_batch_sampler,
            first_epoch_no_shuffle=getattr(args, 'first_epoch_no_shuffle', None),
            save_pack_buffer=getattr(args, 'save_pack_buffer', False),
            save_all_ranks_training_states=getattr(args, 'save_all_ranks_training_states', False),
        )

        set_mm_state(mm_state)
        set_combined_iterator(combined_iterator)
        return combined_iterator, None, None


def add_dummy_tokens(
        tokens, target_tokens, extra, dummy_token_type, dummy_number, device
):
    """ Add dummy tokens to avoid hanging when deepspeed all-reduce gradients.

    Different modalities connect with different model parameters in the computation graph.
    If different batches correspond to different modalities, deepspeed cannot correctly perform
    all-reduce gradients. Therefore, we need to pad some dummy tokens to maintain consistent
    activated model parameters.

    ======================================== Dummy tokens in sequence ========================================
     t2i: ++++++++++++ mmu_dummy
    ti2i: ++++++++++++
     mmu: ++++++++++++
      lm: ++++++++++++ t2i_dummy mmu_dummy
    ==========================================================================================================

    ======================================== Dummy token affectation =========================================

    ts_proj: tokens, target_tokens, text_mask, image_mask, images, timesteps, diff_fn
             ^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^  ++++++++++  ++++++  +++++++++  +++++++
        vit: tokens, target_tokens, text_mask,                                        cond_vit_image_mask, cond_vit_images
             ^^^^^^  ^^^^^^^^^^^^^  ^^^^^^^^^                                         +++++++++++++++++++  +++++++++++++++

    Notice:
      ^^^: Fixed concat
      ***: Automatic extension
      *+*: Automatic extension or new ones.
      +++: New ones that cause dummy
    ==========================================================================================================
    """
    args = get_args()
    # denoiser = get_denoiser()
    tkwrapper = get_tkwrapper()

    bsz, seqlen = tokens.shape

    # ^^^^^^^^ Fixed concat ^^^^^^^^
    dummy_tokens = torch.full((bsz, dummy_number), tkwrapper.pad_token_id,
                              dtype=tokens.dtype, device=device)
    dummy_target_tokens = (-100) * torch.ones((bsz, dummy_number), dtype=tokens.dtype, device=device)
    tokens = torch.cat([tokens, dummy_tokens], dim=1)
    target_tokens = torch.cat([target_tokens, dummy_target_tokens], dim=1)

    extra['text_mask'] = torch.cat([
        extra['text_mask'],
        torch.zeros_like(dummy_tokens, dtype=extra['text_mask'].dtype, device=device)
    ], dim=1)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if extra.get("packed_seq_params") is not None:
        extra["packed_seq_params"] = _extend_packed_seq_params_valid(
            extra["packed_seq_params"], dummy_number, device, valid=True
        )
        extra["valid_local_token_mask"] = extra["packed_seq_params"].valid_local_token_mask

    if args.use_mot:
        appended_to_und = False
        appended_to_gen = False
        dummy_token_indices = torch.tensor([list(range(seqlen, seqlen+dummy_number)) for _ in range(bsz)], dtype=torch.long, device=device)
        if dummy_token_type == "ts_proj":
            if "vae" in args.und_token_type:
                extra['und_token_indices'] = torch.cat([extra['und_token_indices'], dummy_token_indices], dim=1)
                appended_to_und = True
            elif "vae" in args.gen_token_type:
                extra['gen_token_indices'] = torch.cat([extra['gen_token_indices'], dummy_token_indices], dim=1)
                appended_to_gen = True
        elif dummy_token_type == "vit":
            if "vit" in args.und_token_type:
                extra['und_token_indices'] = torch.cat([extra['und_token_indices'], dummy_token_indices], dim=1)
                appended_to_und = True
            elif "vit" in args.gen_token_type:
                extra['gen_token_indices'] = torch.cat([extra['gen_token_indices'], dummy_token_indices], dim=1)
                appended_to_gen = True

        if appended_to_und and extra.get("und_packed_seq_params") is not None:
            extra["und_packed_seq_params"] = _extend_packed_seq_params_valid(
                extra["und_packed_seq_params"], dummy_number, device, valid=True
            )
            extra["und_valid_local_token_mask"] = extra["und_packed_seq_params"].valid_local_token_mask
        elif appended_to_gen and extra.get("gen_packed_seq_params") is not None:
            extra["gen_packed_seq_params"] = _extend_packed_seq_params_valid(
                extra["gen_packed_seq_params"], dummy_number, device, valid=True
            )
            extra["gen_valid_local_token_mask"] = extra["gen_packed_seq_params"].valid_local_token_mask

    if dummy_token_type == "ts_proj":
        # ++++++++ New ones that cause dummy ++++++++
        denoiser = get_denoiser()
        p_state = get_parallel_state()
        generator = torch.Generator(device).manual_seed(args.seed + p_state.dp_rank)
        patch_size = args.patch_size
        latents = torch.randn((bsz, args.vae_latent_dim, patch_size, patch_size), device=device, generator=generator)
        t, x_0, x_1 = denoiser.sample(latents, seqlen, generator=generator)
        t, x_t, u_t = denoiser.path_sampler.plan(t, x_0, x_1)
        model_t = denoiser.get_model_t(t)  # t*1000
        extra.update(dict(
            images=x_t,
            timesteps=model_t,
            ut=u_t,
            image_mask=torch.zeros_like(tokens, dtype=torch.bool, device=device),
            diffusion_loss_fn=partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        ))
        extra['image_mask'][:, -1] = True
        if args.add_timestep_token:
            assert 'timesteps_index' not in extra or extra['timesteps_index'] is None, "timesteps_index already exists in extra."
            timesteps_index = torch.tensor([[seqlen]] * bsz, dtype=torch.long, device=device)
            extra.update(dict(timesteps_index=timesteps_index))
        # ++++++++++++++++++++++++++++++++++++++++++++

        # ******** Automatic extension ********
        for name in ['cond_vae_image_mask', 'cond_vit_image_mask']:
            if name in extra and extra[name] is not None:
                extra[name] = torch.cat([
                    extra[name],
                    torch.zeros_like(dummy_tokens, dtype=extra[name].dtype, device=device)
                ], dim=1)
        # *************************************

    elif dummy_token_type == "vit":
        # ++++++++ New ones that cause dummy ++++++++
        vision_encoder_meta_info = VISION_ENCODER_META_INFO[args.vit_type]
        downsample_factor = to_2tuple(vision_encoder_meta_info["downsample_factor"])
        if "siglip2" in args.vit_type:
            # siglip2-so400m-patch16-naflex needs und_images as 3D tensor: bsz x vit_seqlen x dim
            extra.update(dict(
                cond_vit_images=torch.rand((bsz, 1, 3 * math.prod(downsample_factor)), device=device) * 2 - 1,
                cond_vit_image_mask=torch.zeros_like(tokens, dtype=torch.bool, device=device),
                cond_vit_image_kwargs={
                    "spatial_shapes": torch.tensor([[1, 1]] * bsz, dtype=torch.long, device=device),
                    "attention_mask": torch.ones((bsz, 1), dtype=torch.int32, device=device),
                },
            ))
        elif "anyres" in args.vit_type:
            extra.update(dict(
                cond_vit_images=torch.rand((bsz, 1, 3, *downsample_factor), device=device) * 2 - 1,
                cond_vit_image_mask=torch.zeros_like(tokens, dtype=torch.bool, device=device),
            ))
        elif "qwen3vl" in args.vit_type:
            # qwen3vl needs grid_thw parameter for vision encoder
            # For dummy tokens, we use patch_size=16 (default for qwen3vl) and spatial_merge_size=2
            # But we need to ensure height and width are at least spatial_merge_size (2) to avoid division by zero
            # Calculate grid_thw: (temporal, height, width) after patch_embed
            # height and width are the feature map dimensions after patch embedding
            grid_h, grid_w = to_2tuple(vision_encoder_meta_info["spatial_merge_size"])
            patch_dim = vision_encoder_meta_info["patch_dim"]
            extra.update(dict(
                cond_vit_images=torch.rand((bsz, 1, grid_h * grid_w, patch_dim), device=device) * 2 - 1,
                cond_vit_image_mask=torch.zeros_like(tokens, dtype=torch.bool, device=device),
                cond_vit_image_kwargs={
                    "grid_thw": torch.tensor([[[1, grid_h, grid_w]]] * bsz, dtype=torch.long, device=device),
                },
            ))
        else:
            raise NotImplementedError(f"Dummy vit not implemented for vit type {args.vit_type}.")

        vit_dummy_number = vision_encoder_meta_info["dummy_number"]
        extra['cond_vit_image_mask'][:, -vit_dummy_number:] = True
        # ++++++++++++++++++++++++++++++++++++++++++++

        # ******** Automatic extension ********
        for name in ['image_mask', 'cond_vae_image_mask']:
            if name in extra and extra[name] is not None:
                extra[name] = torch.cat([
                    extra[name],
                    torch.zeros_like(dummy_tokens, dtype=extra[name].dtype, device=device)
                ], dim=1)
        # *************************************

    seqlen += dummy_number

    return tokens, target_tokens, extra, seqlen


def save_first_training_samples(batch, key_not_save=None, extra=None):
    args = get_args()
    if args.save_n_training_data == 0:
        return

    mm_state = get_mm_state()
    ss = get_scalar_state()
    p_state = get_parallel_state()
    ds_state = mm_state.distributed_sampling_state

    cur_forward_times = ss.current_forward_times
    dp_rank = p_state.dp_rank if ds_state.cur_key is None else ds_state.dataset_rank[ds_state.cur_key]

    do_save = (p_state.pp_rank == 0) and (p_state.tp_rank == 0) and (p_state.cp_rank == 0) \
        and (cur_forward_times <= args.save_n_training_data) and (dp_rank < 8)
    if not do_save:
        return

    # Save training data for debugging
    check_data_path = safe_dir(Path(args.saved_training_batch_dir) / Path(args.task_id))
    rank = torch.distributed.get_rank()
    save_path = check_data_path / f"data_batch{cur_forward_times}_{batch["dataset_tag"]}_dp{dp_rank}_global{rank}.pt"
    # Filter BlockMask

    # seqlen = batch["text_mask"].shape[1]
    # B = 1
    # H = 1
    # Q_LEN = seqlen
    # KV_LEN = seqlen
    # b_idx = torch.arange(B, device="cuda").view(B, 1, 1, 1).expand(B, H, Q_LEN, KV_LEN)
    # h_idx = torch.arange(H, device="cuda").view(1, H, 1, 1).expand(B, H, Q_LEN, KV_LEN)
    # q_idx = torch.arange(Q_LEN, device="cuda").view(1, 1, Q_LEN, 1).expand(B, H, Q_LEN, KV_LEN)
    # kv_idx = torch.arange(KV_LEN, device="cuda").view(1, 1, 1, KV_LEN).expand(B, H, Q_LEN, KV_LEN)

    # attention_mask = batch["attention_mask"].mask_mod(b_idx, h_idx, q_idx, kv_idx)
    if BlockMask is not None:
        batch = {k: v for k, v in batch.items() if not isinstance(v, BlockMask)}

    # batch["attention_mask"] = attention_mask

    if key_not_save is not None:
        if isinstance(key_not_save, str):
            key_not_save = [key_not_save]
        batch = {k: v for k, v in batch.items() if k not in key_not_save}
    if extra is not None:
        for k, v in extra.items():
            if k not in batch:
                batch[k] = v
            else:
                batch[f"extra_{k}"] = v
    torch.save(batch, save_path)


def prepare_model_t2i_inputs(batch: dict, device: int | str | torch.device, **kwargs):
    args = get_args()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")

    if not kwargs.get('skip_vae_encode'):
        vae = get_vae()
    denoiser = get_denoiser()

    tokens = batch["tokens"][:, :-1].contiguous().to(device)
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
    # image_loss is computed inplace, therefore image_mask is shifted same as tokens
    _image_mask_full = batch["image_mask"]
    if isinstance(_image_mask_full, torch.Tensor):
        assert not _image_mask_full[:, -1].to(dtype=torch.bool).any(), (
            "Last sequence token cannot be a gen image token: image_mask[:, :-1] must align with "
            "tokens[:, :-1] (e.g. end with eos/pad after the image block)."
        )
    image_mask = _image_mask_full[:, :-1].contiguous().to(device)
    bsz, seqlen = tokens.shape

    # Add dummy tokens
    extra = dict(
        dataset_tag=dataset_tag,
        text_mask=text_mask,        # [bsz, seqlen]
        image_mask=image_mask,      # [bsz, seqlen]
        timesteps_index=to_device(batch["timesteps_index"], device),  # [bsz, 1]
        rope_image_info=batch["rope_image_info"],
    )

    if batch.get('valid_local_token_mask') is not None:
        extra.update(dict(
            valid_local_token_mask=batch['valid_local_token_mask'][:-1].contiguous().to(device),
            packed_seq_params=build_packed_seq_params(batch['valid_local_token_mask'][:-1], device),
        ))

    if args.use_mot:
        extra.update(dict(
            und_token_indices=to_device(batch["und_token_indices"][:, :-1], device),
            gen_token_indices=to_device(batch["gen_token_indices"], device),
        ))

        n_und, n_gen = extra["und_token_indices"].shape[1], extra["gen_token_indices"].shape[1]
        assert n_und + n_gen == tokens.shape[1], (
            f"MoT indices length mismatch: und={n_und}, gen={n_gen}, seqlen={tokens.shape[1]}"
        )
        if batch.get('und_valid_local_token_mask') is not None and batch.get('gen_valid_local_token_mask') is not None:
            extra.update(dict(
                und_valid_local_token_mask=batch['und_valid_local_token_mask'][:-1].contiguous().to(device),
                gen_valid_local_token_mask=batch['gen_valid_local_token_mask'].contiguous().to(device),
                und_packed_seq_params=build_packed_seq_params(batch['und_valid_local_token_mask'][:-1], device),
                gen_packed_seq_params=build_packed_seq_params(batch['gen_valid_local_token_mask'], device),
            ))


    if "offsets" in batch:
        extra.update(dict(
            sample_offsets=[
                # clamp by seqlen to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                # due to seqlen = batch["tokens"].shape[1] - 1
                torch.clamp(offset, 0, seqlen) for offset in batch["offsets"]
            ],
        ))

    # t2i has the same dummy_type_dict
    dummy_type_dict = batch["dummy_type_dict"][0]
    for dummy_type, dummy_number in dummy_type_dict.items():
        tokens, target_tokens, extra, seqlen = add_dummy_tokens(
            tokens, target_tokens, extra,
            dummy_token_type=dummy_type, dummy_number=dummy_number, device=device
        )

    # Attention mask
    attn_type = task_kwargs['attn_type']
    image_slices = batch["image_slices"]
    if getattr(args, "timestep_vae_full_attn", False):
        assert "image_full_attn_slices" in batch, (
            "timestep_vae_full_attn is enabled, but image_full_attn_slices is missing from batch."
        )
        attn_image_slices = batch["image_full_attn_slices"]
    else:
        attn_image_slices = image_slices
    if attn_type == 'auto':
        attention_mask = batch["attention_mask"].to(device)
    elif attn_type == 'flex':
        bsz, seq_len = tokens.shape
        assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
        if bsz == 1:
            mask_mod = create_text_image_mask_mod(
                attn_image_slices[0], seq_len, device, offsets=extra.get("sample_offsets", [None])[0],
            )
            attention_mask = create_block_mask(
                mask_mod, 
                B=None, 
                H=None, 
                Q_LEN=seq_len, 
                KV_LEN=seq_len, 
                device=device,
                _compile=True if device.type == "cuda" else False,  # for CI / no cuda runs, disable _compile
            )
        else:
            mask_mod = create_batch_text_image_mask_mod(
                attn_image_slices, seq_len, device, batch_offsets=extra.get("sample_offsets"),
            )
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        attention_mask.dtype = torch.bool
        attention_mask.device = tokens.device
    elif attn_type == 'magi':
        from angelptm.megatron.core.models.gemini.magi_attn_utils import image_slices_to_magi_ranges

        bsz, seq_len = tokens.shape
        if bsz == 1:
            attention_mask = image_slices_to_magi_ranges(
                attn_image_slices[0], seq_len,
                offsets=extra.get("sample_offsets", [None])[0],
            )
        else:
            attention_mask = image_slices_to_magi_ranges(attn_image_slices[0], seq_len)
    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    # ===================================== prepare diffusion =====================================
    if kwargs.get('skip_vae_encode'):
        t, model_t, x_0, x_t, u_t = None, None, None, None, None
    else:
        out = vae_encode_and_add_noise(
            vae, batch["images"], device, denoiser=denoiser, sample_type="sample",
        )
        t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        input_ids=tokens,                   # [bsz, seqlen]
        target=target_tokens,               # [bsz, seqlen]
        attention_mask=attention_mask,      # [bsz, 1, seqlen, seqlen]
        images=x_t,                         # [bsz, c, h, w]
        timesteps=model_t,                  # [bsz]
        ut=u_t,                             # [bsz, c, h, w] velocity target for flow matching
        diffusion_loss_fn=partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        image_loss_weight=args.image_loss_weight,
        **extra,
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    # Save model_input_kwargs and a few more info for reconstructing the attention mask
    # Save raw images for training data visualization
    save_first_training_samples(model_input_kwargs, 
        key_not_save=["packed_seq_params", "und_packed_seq_params", "gen_packed_seq_params"],
        extra=dict(
            raw_images=batch.get("images", None),
            image_slices=batch.get("image_slices", None),
            image_full_attn_slices=batch.get("image_full_attn_slices", None),
        )
    )
    for k in ("valid_local_token_mask", "und_valid_local_token_mask", "gen_valid_local_token_mask"): 
        model_input_kwargs.pop(k, None)
    return model_input_kwargs, bsz, seqlen


def prepare_model_lm_inputs(batch: dict, device: int | str | torch.device, **kwargs):
    _ = kwargs
    args = get_args()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")

    tokens = batch["tokens"][:, :-1].contiguous().to(device)
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
    batch_size, n_tokens = tokens.shape

    # Add dummy tokens
    extra = dict(
        dataset_tag=dataset_tag,
        text_mask=text_mask,
    )

    if batch.get('valid_local_token_mask') is not None:
        extra.update(dict(
            valid_local_token_mask=batch['valid_local_token_mask'][:-1].contiguous().to(device),
            packed_seq_params=build_packed_seq_params(batch['valid_local_token_mask'][:-1], device),
        ))

    if args.use_mot:
        extra.update(dict(
            und_token_indices=to_device(batch["und_token_indices"][:, :-1], device),
            gen_token_indices=to_device(batch["gen_token_indices"], device),
        ))

        n_und, n_gen = extra["und_token_indices"].shape[1], extra["gen_token_indices"].shape[1]
        assert n_und + n_gen == tokens.shape[1], (
            f"MoT indices length mismatch: und={n_und}, gen={n_gen}, seqlen={tokens.shape[1]}"
        )
        if batch.get('und_valid_local_token_mask') is not None and batch.get('gen_valid_local_token_mask') is not None:
            extra.update(dict(
                und_valid_local_token_mask=batch['und_valid_local_token_mask'][:-1].contiguous().to(device),
                gen_valid_local_token_mask=batch['gen_valid_local_token_mask'].contiguous().to(device),
                und_packed_seq_params=build_packed_seq_params(batch['und_valid_local_token_mask'][:-1], device),
                gen_packed_seq_params=build_packed_seq_params(batch['gen_valid_local_token_mask'], device),
            ))

    if "offsets" in batch:
        extra.update(dict(
            sample_offsets=[
                # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                # due to n_tokens = batch["tokens"].shape[1] - 1
                torch.clamp(offset, 0, n_tokens)
                for offset in batch["offsets"]
            ],
        ))
    dummy_type_dict = batch["dummy_type_dict"][0]
    for dummy_type, dummy_number in dummy_type_dict.items():
        tokens, target_tokens, extra, n_tokens = add_dummy_tokens(
            tokens, target_tokens, extra,
            dummy_token_type=dummy_type, dummy_number=dummy_number, device=device
        )

    # Align to multiple of 128 for flex attention
    attn_type = task_kwargs["attn_type"]
    if attn_type == 'flex':
        align_extra = [k for k in ('image_mask', 'cond_vae_image_mask', 'cond_vit_image_mask') if k in extra]
        if args.use_mot:
            align_extra.append('und_token_indices')

    # Attention mask
    if attn_type == 'auto':
        causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device).tril(diagonal=0)
        attention_mask = causal_mask.view(1, 1, n_tokens, n_tokens).repeat(batch_size, 1, 1, 1)
    elif attn_type == 'flex':
        bsz, seq_len = tokens.shape
        assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
        if bsz == 1:
            mask_mod = create_causal_mask_mod(seq_len, device, offsets=extra.get("sample_offsets", [None])[0])
            attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        else:
            mask_mod = create_batch_causal_mask_mod(seq_len, device, batch_offsets=extra.get("sample_offsets"))
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        attention_mask.dtype = torch.bool
        attention_mask.device = tokens.device
    elif attn_type == 'magi':
        from angelptm.megatron.core.models.gemini.magi_attn_utils import causal_to_magi_ranges

        bsz, seq_len = tokens.shape
        if bsz == 1:
            attention_mask = causal_to_magi_ranges(
                seq_len, offsets=extra.get("sample_offsets", [None])[0],
            )
        else:
            attention_mask = causal_to_magi_ranges(seq_len)
    else:
        # Default to causal
        attention_mask = None

    model_input_kwargs = dict(
        input_ids=tokens,                   # [b, seqlen]
        target=target_tokens,               # [b, seqlen]
        attention_mask=attention_mask,      # [b, 1, seqlen, seqlen]
        image_loss_weight=0,                # Set to zero to avoid dummy image tokens to affect the text loss
        **extra,    # x_t, t, image_mask, und_images, und_image_masks, diffusion_loss_fn, timestep
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    save_first_training_samples(model_input_kwargs, 
        key_not_save=["packed_seq_params", "und_packed_seq_params", "gen_packed_seq_params"],
    )
    for k in ("valid_local_token_mask", "und_valid_local_token_mask", "gen_valid_local_token_mask"): 
        model_input_kwargs.pop(k, None)
    return model_input_kwargs, batch_size, n_tokens


def prepare_model_mmu_inputs(batch: dict, device: int | str | torch.device, **kwargs):
    args = get_args()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")

    if "skip_vae_encode" not in kwargs:
        kwargs["skip_vae_encode"] = task_kwargs.get('skip_vae_encode', False)
    if not kwargs.get('skip_vae_encode'):
        vae = get_vae()
        denoiser = get_denoiser()

    tokens = batch["tokens"][:, :-1].contiguous().to(device)
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
    cond_vae_image_mask = batch["cond_vae_image_mask"][:, :-1].contiguous().to(device) \
        if "cond_vae_image_mask" in batch else None
    cond_vit_image_mask = batch["cond_vit_image_mask"][:, :-1].contiguous().to(device) \
        if "cond_vit_image_mask" in batch else None
    assert cond_vae_image_mask is not None or cond_vit_image_mask is not None, \
        "At least one of cond_vae_image_mask and cond_vit_image_mask should be provided for mmu task."
    bsz, seqlen = tokens.shape

    # Add dummy tokens
    extra = dict(
        dataset_tag=dataset_tag,
        text_mask=text_mask,                        # [bsz, seqlen]
        cond_vae_image_mask=cond_vae_image_mask,    # [bsz, seqlen]
        cond_vit_image_mask=cond_vit_image_mask,    # [bsz, seqlen]
        cond_timesteps_index=to_device(batch.get("cond_timesteps_index"), device),  # [bsz, 1]
        rope_image_info=batch["rope_image_info"],
    )
    
    if batch.get('valid_local_token_mask') is not None:
        extra.update(dict(
            valid_local_token_mask=batch['valid_local_token_mask'][:-1].contiguous().to(device),
            packed_seq_params=build_packed_seq_params(batch['valid_local_token_mask'][:-1], device),
        ))

    if args.use_mot:
        extra.update(dict(
            und_token_indices=to_device(batch["und_token_indices"][:, :-1], device),
            gen_token_indices=to_device(batch["gen_token_indices"], device),
        ))

        n_und, n_gen = extra["und_token_indices"].shape[1], extra["gen_token_indices"].shape[1]
        assert n_und + n_gen == tokens.shape[1], (
            f"MoT indices length mismatch: und={n_und}, gen={n_gen}, seqlen={tokens.shape[1]}"
        )
        if batch.get('und_valid_local_token_mask') is not None and batch.get('gen_valid_local_token_mask') is not None:
            extra.update(dict(
                und_valid_local_token_mask=batch['und_valid_local_token_mask'][:-1].contiguous().to(device),
                gen_valid_local_token_mask=batch['gen_valid_local_token_mask'].contiguous().to(device),
                und_packed_seq_params=build_packed_seq_params(batch['und_valid_local_token_mask'][:-1], device),
                gen_packed_seq_params=build_packed_seq_params(batch['gen_valid_local_token_mask'], device),
            ))

    if "offsets" in batch:
        extra.update(dict(
            sample_offsets=[
                # clamp by seqlen to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                # due to seqlen = batch["tokens"].shape[1] - 1
                torch.clamp(offset, 0, seqlen)
                for offset in batch["offsets"]
            ],
        ))
    dummy_type_dict = batch["dummy_type_dict"][0]
    for dummy_type, dummy_number in dummy_type_dict.items():
        tokens, target_tokens, extra, seqlen = add_dummy_tokens(
            tokens, target_tokens, extra,
            dummy_token_type=dummy_type, dummy_number=dummy_number, device=device
        )

    # Attention mask
    attn_type = task_kwargs['attn_type']
    if attn_type == "auto":
        attention_mask = batch["attention_mask"].to(device)
    elif attn_type == "flex":
        bsz, seq_len = tokens.shape
        assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
        if bsz == 1:
            image_slices = batch["cond_full_attn_slices"][0]
            mask_mod = create_text_image_mask_mod(image_slices, seq_len, device,
                                                  offsets=extra.get("sample_offsets", [None])[0])
            attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        else:
            cond_slices = batch["cond_full_attn_slices"]
            mask_mod = create_batch_text_image_mask_mod(cond_slices, seq_len, device,
                                                        batch_offsets=extra.get("sample_offsets"))
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        attention_mask.dtype = torch.bool
        attention_mask.device = tokens.device
    elif attn_type == 'magi':
        from angelptm.megatron.core.models.gemini.magi_attn_utils import image_slices_to_magi_ranges

        bsz, seq_len = tokens.shape
        if bsz == 1:
            attention_mask = image_slices_to_magi_ranges(
                batch["cond_full_attn_slices"][0], seq_len,
                offsets=extra.get("sample_offsets", [None])[0],
            )
        else:
            attention_mask = image_slices_to_magi_ranges(
                batch["cond_full_attn_slices"][0], seq_len,
            )
    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    cond_vit_images = to_device(batch.get("cond_vit_images"), device)
    cond_vit_image_kwargs = (
        {k: to_device(v, device) for k, v in batch["cond_vit_image_kwargs"].items()}
        if "cond_vit_image_kwargs" in batch
        else None
    )

    if kwargs.get('skip_vae_encode', False) or cond_vae_image_mask is None:
        cond_vae_ts, cond_vae_latents = None, None
    else:
        sout = vae_encode_and_add_noise(
            vae, batch["cond_vae_images"], device, denoiser=denoiser, sample_type="sample_start",
        )
        cond_vae_ts, cond_vae_latents = sout.model_t, sout.x_t

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        input_ids=tokens,  # [b, 512]
        target=target_tokens,  # [b, 512]
        attention_mask=attention_mask,  # [b, 512, 512]
        cond_vae_images=cond_vae_latents,                  # [b, c, h, w]
        cond_timesteps=cond_vae_ts,                  # [b]
        cond_vit_images=cond_vit_images,
        cond_vit_image_kwargs=cond_vit_image_kwargs,
        image_loss_weight=0,    # Set to zero to avoid dummy image tokens to affect the text loss
        **extra,
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    save_first_training_samples(model_input_kwargs, 
        key_not_save=["packed_seq_params", "und_packed_seq_params", "gen_packed_seq_params"],
        extra=dict(
            image_slices=batch.get("image_slices", None),
            cond_full_attn_slices=batch.get("cond_full_attn_slices", None),
            cond_vae_image_slices=batch.get("cond_vae_image_slices", None),
            cond_vit_image_slices=batch.get("cond_vit_image_slices", None),
            raw_vae_images=batch.get("cond_vae_images"),
            raw_vit_images=batch.get("cond_vit_images"),
        )
    )
    for k in ("valid_local_token_mask", "und_valid_local_token_mask", "gen_valid_local_token_mask"): 
        model_input_kwargs.pop(k, None)
    return model_input_kwargs, bsz, seqlen


def prepare_model_interleave_inputs(batch: dict, device: int | str | torch.device, **kwargs):
    args = get_args()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")

    if not kwargs.get('skip_vae_encode'):
        vae = get_vae()
    denoiser = get_denoiser()
    tkwrapper = get_tkwrapper()

    tokens = batch["tokens"][:, :-1].contiguous().to(device)
    target_tokens = batch["target_tokens"][:, 1:].contiguous().to(device)
    text_mask = batch["text_mask"][:, 1:].contiguous().to(device)
    # image_loss is computed inplace, therefore image_mask is shifted same as tokens
    if "image_mask" in batch:
        _image_mask_full = batch["image_mask"]
        if isinstance(_image_mask_full, torch.Tensor):
            assert not _image_mask_full[:, -1].to(dtype=torch.bool).any(), (
                "Last sequence token cannot be a gen image token: image_mask[:, :-1] must align with "
                "tokens[:, :-1] (e.g. end with eos/pad after the image block)."
            )
        image_mask = _image_mask_full[:, :-1].contiguous().to(device)
    else:
        image_mask = None
    cond_vae_image_mask = batch["cond_vae_image_mask"][:, :-1].contiguous().to(device) \
        if "cond_vae_image_mask" in batch else None
    cond_vit_image_mask = batch["cond_vit_image_mask"][:, :-1].contiguous().to(device) \
        if "cond_vit_image_mask" in batch else None
    # assert cond_vae_image_mask is not None or cond_vit_image_mask is not None, \
    # "At least one of cond_vae_image_mask and cond_vit_image_mask should be provided for interleave task."
    bsz, seqlen = tokens.shape

    # Add dummy tokens
    extra = dict(
        dataset_tag=dataset_tag,
        text_mask=text_mask,  # [b, seqlen]
        image_mask=image_mask,  # [b, seqlen]
        timesteps_index=to_device(batch["timesteps_index"], device),  # [b, 1]
        cond_vae_image_mask=cond_vae_image_mask,  # [b, seqlen]
        cond_timesteps_index=to_device(batch.get("cond_timesteps_index"), device),
        cond_vit_image_mask=cond_vit_image_mask,  # [b, seqlen]
        rope_image_info=batch["rope_image_info"],
    )

    if batch.get('valid_local_token_mask') is not None:
        extra.update(dict(
            valid_local_token_mask=batch['valid_local_token_mask'][:-1].contiguous().to(device),
            packed_seq_params=build_packed_seq_params(batch['valid_local_token_mask'][:-1], device),
        ))

    if args.use_mot:
        extra.update(dict(
            und_token_indices=to_device(batch["und_token_indices"][:, :-1], device),
            gen_token_indices=to_device(batch["gen_token_indices"], device),
        ))

        n_und, n_gen = extra["und_token_indices"].shape[1], extra["gen_token_indices"].shape[1]
        assert n_und + n_gen == tokens.shape[1], (
            f"MoT indices length mismatch: und={n_und}, gen={n_gen}, seqlen={tokens.shape[1]}"
        )
        if batch.get('und_valid_local_token_mask') is not None and batch.get('gen_valid_local_token_mask') is not None:
            extra.update(dict(
                und_valid_local_token_mask=batch['und_valid_local_token_mask'][:-1].contiguous().to(device),
                gen_valid_local_token_mask=batch['gen_valid_local_token_mask'].contiguous().to(device),
                und_packed_seq_params=build_packed_seq_params(batch['und_valid_local_token_mask'][:-1], device),
                gen_packed_seq_params=build_packed_seq_params(batch['gen_valid_local_token_mask'], device),
            ))


    if "offsets" in batch:
        extra.update(dict(
            sample_offsets=[
                # clamp by n_tokens to avoid the boundary situation where offset[-1] == batch["tokens"].shape[1]
                # due to n_tokens = batch["tokens"].shape[1] - 1
                torch.clamp(offset, 0, seqlen) for offset in batch["offsets"]
            ],
        ))

    # ===================================== prepare diffusion =====================================
    if kwargs.get('skip_vae_encode'):
        t, model_t, x_0, x_t, u_t = None, None, None, None, None
        input_src_t, input_src_x = None, None
    else:
        # interleave_mmu
        if image_mask is not None:
            out = vae_encode_and_add_noise(
                vae, batch["images"], device, denoiser=denoiser, sample_type="sample",
            )
            t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t
        else:
            t, model_t, x_0, x_t, u_t = None, None, None, None, None
        # interleave_t2i
        if cond_vae_image_mask is not None:
            sout = vae_encode_and_add_noise(
                vae, batch["cond_vae_images"], device, denoiser=denoiser, sample_type="sample_start",
            )
            input_src_t, input_src_x = sout.model_t, sout.x_t
        else:
            input_src_t, input_src_x = None, None

    # vit image kwargs
    cond_vit_images = to_device(batch.get("cond_vit_images"), device) if "cond_vit_images" in batch else None
    cond_vit_image_kwargs = (
        {k: to_device(v, device) for k, v in batch["cond_vit_image_kwargs"].items()}
        if "cond_vit_image_kwargs" in batch
        else None
    )

    extra.update(dict(
        images=x_t,
        timesteps=model_t,
        ut=u_t,
        diffusion_loss_fn=partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        cond_vae_images=input_src_x,
        cond_timesteps=input_src_t,
        cond_vit_images=cond_vit_images,
        cond_vit_image_kwargs=cond_vit_image_kwargs,
    ))

    if all(batch["dummy_type_dict"][i] == batch["dummy_type_dict"][0] for i in range(len(batch["dummy_type_dict"]))):
        dummy_type_dict = batch["dummy_type_dict"][0]
        for dummy_type, dummy_number in dummy_type_dict.items():
            tokens, target_tokens, extra, seqlen = add_dummy_tokens(
                tokens, target_tokens, extra,
                dummy_token_type=dummy_type, dummy_number=dummy_number, device=device
            )
    else:
        raise NotImplementedError("Different dummy_type_dict in a batch has not been supported yet.")

    # Attention mask
    attn_type = task_kwargs["attn_type"]
    image_slices = batch["image_slices"]
    if getattr(args, "timestep_vae_full_attn", False):
        assert "image_full_attn_slices" in batch, (
            "timestep_vae_full_attn is enabled, but image_full_attn_slices is missing from batch."
        )
        attn_image_slices = batch["image_full_attn_slices"]
    else:
        attn_image_slices = image_slices
    if attn_type == 'auto':
        attention_mask = batch["attention_mask"].to(device)
    elif attn_type == 'flex':
        bsz, seq_len = tokens.shape
        assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
        if batch["dataset_tag"][0].startswith("interleave"):
            has_image = any([batch["cond_full_attn_slices"][i] for i in range(bsz)] + [image_slices[i] for i in range(bsz)])
            if has_image:
                num_image_token_prefix = ImageInfo.num_image_token_prefix()
                num_image_token_suffix = ImageInfo.num_image_token_suffix()
            else:
                num_image_token_prefix = num_image_token_suffix = 0
            if bsz == 1:
                cond_slices = batch["cond_full_attn_slices"][0]
                gen_slices = image_slices[0]
                gen_full_slices = attn_image_slices[0]
                hole_slices = []
                for full_sli, sli in zip(gen_full_slices, gen_slices):
                    # Gen images except the last one are not allowed to be attended by following tokens.
                    if tokens[0, sli.stop + 1] != tkwrapper.eos_token_id:
                        hole_slices.append(slice(
                            min(full_sli.start, sli.start - num_image_token_prefix),
                            sli.stop + num_image_token_suffix,
                        ))
                mask_mod = create_interleave_mask_mod(
                    gen_full_slices, cond_slices, hole_slices, seq_len, device,
                    offsets=extra.get("sample_offsets", [None])[0],
                )
                attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
            else:
                cond_slices = batch["cond_full_attn_slices"]
                hole_slices = []
                for gen_full_slices, gen_slices in zip(attn_image_slices, image_slices):
                    hole_slices_i = []
                    for full_sli, sli in zip(gen_full_slices, gen_slices):
                        # Gen images except the last one are not allowed to be attended by following tokens.
                        if tokens[0, sli.stop + 1] != tkwrapper.eos_token_id:
                            hole_slices_i.append(slice(
                                min(full_sli.start, sli.start - num_image_token_prefix),
                                sli.stop + num_image_token_suffix,
                            ))
                    hole_slices.append(hole_slices_i)
                mask_mod = create_batch_interleave_mask_mod(
                    attn_image_slices, cond_slices, hole_slices, seq_len, device,
                    batch_offsets=extra.get("sample_offsets"),
                )
                attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        elif bsz == 1:
            all_slices = attn_image_slices[0] + batch["cond_full_attn_slices"][0]
            mask_mod = create_text_image_mask_mod(all_slices, seq_len, device,
                                                  offsets=extra.get("sample_offsets", [None])[0])
            attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        else:
            batch_image_slices = [
                attn_image_slices[i] + batch["cond_full_attn_slices"][i]
                for i in range(bsz)
            ]
            mask_mod = create_batch_text_image_mask_mod(batch_image_slices, seq_len, device,
                                                        batch_offsets=extra.get("sample_offsets"))
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
        attention_mask.dtype = torch.bool
        attention_mask.device = tokens.device
    elif attn_type == 'magi':
        from angelptm.megatron.core.models.gemini.magi_attn_utils import interleave_slices_to_magi_ranges, image_slices_to_magi_ranges

        bsz, seq_len = tokens.shape
        if batch["dataset_tag"][0].startswith("interleave"):
            has_image = any([batch["cond_full_attn_slices"][i] for i in range(bsz)] + [image_slices[i] for i in range(bsz)])
            if has_image:
                num_image_token_prefix = ImageInfo.num_image_token_prefix()
                num_image_token_suffix = ImageInfo.num_image_token_suffix()
            else:
                num_image_token_prefix = num_image_token_suffix = 0
            if bsz == 1:
                cond_slices = batch["cond_full_attn_slices"][0]
                gen_slices = image_slices[0]
                gen_full_slices = attn_image_slices[0]
                hole_slices = []
                for full_sli, sli in zip(gen_full_slices, gen_slices):
                    if tokens[0, sli.stop + 1] != tkwrapper.eos_token_id:
                        hole_slices.append(slice(
                            min(full_sli.start, sli.start - num_image_token_prefix),
                            sli.stop + num_image_token_suffix,
                        ))
                attention_mask = interleave_slices_to_magi_ranges(
                    gen_full_slices, cond_slices, hole_slices, seq_len,
                    offsets=extra.get("sample_offsets", [None])[0],
                )
            else:
                raise NotImplementedError("Magi interleave attention with bsz > 1 not yet supported")
        elif bsz == 1:
            all_slices = attn_image_slices[0] + batch["cond_full_attn_slices"][0]
            attention_mask = image_slices_to_magi_ranges(
                all_slices, seq_len,
                offsets=extra.get("sample_offsets", [None])[0],
            )
        else:
            all_slices = attn_image_slices[0] + batch["cond_full_attn_slices"][0]
            attention_mask = image_slices_to_magi_ranges(all_slices, seq_len)
    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        input_ids=tokens,  # [b, seqlen]
        target=target_tokens,  # [b, seqlen]
        attention_mask=attention_mask,  # [b, seqlen, seqlen]
        image_loss_weight=args.image_loss_weight,
        **extra,
    )

    save_first_training_samples(model_input_kwargs, 
        key_not_save=["packed_seq_params", "und_packed_seq_params", "gen_packed_seq_params"],
        extra=dict(
            image_slices=batch.get("image_slices", None),
            image_full_attn_slices=batch.get("image_full_attn_slices", None),
            cond_full_attn_slices=batch.get("cond_full_attn_slices", None),
            cond_vae_image_slices=batch.get("cond_vae_image_slices", None),
            cond_vit_image_slices=batch.get("cond_vit_image_slices", None),
        )
    )
    for k in ("valid_local_token_mask", "und_valid_local_token_mask", "gen_valid_local_token_mask"): 
        model_input_kwargs.pop(k, None)
    return model_input_kwargs, bsz, seqlen


def prepare_model_inputs(batch: dict, device: int | str | torch.device, **kwargs):
    ss = get_scalar_state()
    ss.current_forward_times += 1
    if multi_pattern_match(batch["dataset_tag"][0], ["t2i*"]):
        return prepare_model_t2i_inputs(batch, device, **kwargs)
    elif multi_pattern_match(batch["dataset_tag"][0], ["lm*"]) or multi_pattern_match(batch["dataset_tag"][0], ["interleave_LLM*"]):
        return prepare_model_lm_inputs(batch, device, **kwargs)
    elif multi_pattern_match(batch["dataset_tag"][0], ["mmu*"]):
        return prepare_model_mmu_inputs(batch, device, **kwargs)
    elif multi_pattern_match(batch["dataset_tag"][0], ["interleave*", "pair*"]):
        return prepare_model_interleave_inputs(batch, device, **kwargs)
    else:
        raise NotImplementedError(f"dataset_tag `{batch['dataset_tag'][0]}` not recognized in prepare_model_inputs.")
