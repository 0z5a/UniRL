import math
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from .data_provider import save_first_training_samples
from .global_vars import (
    get_args,
    get_parallel_state,
    get_scalar_state,
    get_denoiser,
    get_video_denoiser,
    get_audio_denoiser,
    get_vae,
    get_audio_vae,
    get_text_encoder,
    get_repa_encoder,
)
from .parallel_states import ParallelState
from ..data_kits.system_prompt import get_system_prompt
from ..constants import SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES
from ..models.audio_encoders import (
    add_audio_noise,
    audio_vae_encode_and_add_noise,
)
from ..models.autoencoders import (
    add_image_noise,
    add_noise_to_latents,
    normalize_vae_latents,
    vae_encode_and_add_noise,
)
from ..models.autoregressive.flex_attn_layers import (
    create_text_image_mask_mod,
    create_batch_text_image_mask_mod,
)
from ..models.visual_encoders.qwen.qwen_vit import (
    flatten_qwen3vl_pixel_values as _flatten_qwen3vl_pixel_values,
    flatten_qwen3vl_grid_thw as _flatten_qwen3vl_grid_thw,
)
from ..utils.helpers import multi_pattern_match
from ..utils.torch_utils import to_device


def _is_pre_process_stage(p_state: "ParallelState", is_vpp_first_chunk: bool = True) -> bool:
    """Whether this is the first PP stage AND the first VP chunk (the real pre_process
    stage that runs VAE encode + add-noise).

    Non-VPP: ``is_vpp_first_chunk`` is always True, so this reduces to ``p_state.pp_rank == 0``.
    VPP (Virtual Pipeline Parallelism): PP rank 0 hosts multiple VP chunks. Megatron does
    NOT call ``set_virtual_pipeline_model_parallel_rank`` during schedule execution, so the
    global VP rank cannot be queried via ``get_virtual_pipeline_model_parallel_rank()``.
    Instead, forward_step derives the chunk from the model object's ``vp_stage`` and passes
    ``is_vpp_first_chunk``:
      - VP chunk 0  -> is_vpp_first_chunk=True  -> real pre_process stage (runs VAE/add-noise)
      - VP chunk 1+ -> is_vpp_first_chunk=False -> skip VAE encode / add-noise, avoiding extra
        RNG consumption that would shift the loss.
    """
    return is_vpp_first_chunk and p_state.pp_rank == 0


def create_audio_dummy_tokens(bsz, denoiser, model_t, device, dtype, dummy_number=1, packed=False):
    # Add dummy audio latents and timesteps for t2i tasks with audio branch enabled,
    # to avoid special handling in the model forward.
    args = get_args()
    if packed:
        assert bsz == 1, "Packed sequence with dummy audio tokens currently only supports batch size of 1."
        shape = (args.audio_vae_latent_dim, dummy_number)
    else:
        shape = (bsz, args.audio_vae_latent_dim, dummy_number)

    # The number of dummy video tokens don't need to be associated with that of video samples.
    a_model_t = model_t.clone() \
        if isinstance(model_t, torch.Tensor) \
        else torch.stack([model_t_i.clone()[:dummy_number] for model_t_i in model_t]).flatten()
    a_x_t = torch.zeros(shape, device=device, dtype=dtype)
    a_u_t = torch.zeros(shape, device=device, dtype=dtype)

    if packed:
        a_model_t = [a_model_t]
        a_x_t = [[a_x_t]]
        a_u_t = [[a_u_t]]
    audio_diffusion_loss_fn = partial(denoiser.training_losses_fn, t=None, x0=None, xt=a_x_t, ut=a_u_t)

    return a_model_t, a_x_t, a_u_t, audio_diffusion_loss_fn


def create_video_dummy_tokens(bsz, denoiser, model_t, device, dtype, dummy_number=1, packed=False):
    """ Similar to create_audio_dummy_tokens but for video latents and timesteps. """
    args = get_args()
    x_t_shape = (getattr(args, "img_latent_in_channels", args.vae_latent_dim), dummy_number, 1, 1)
    u_t_shape = (args.vae_latent_dim, dummy_number, 1, 1)
    if packed:
        assert bsz == 1, "Packed sequence with dummy video tokens currently only supports batch size of 1."
    else:
        x_t_shape = (bsz, *x_t_shape)
        u_t_shape = (bsz, *u_t_shape)

    v_model_t = model_t.clone() \
        if isinstance(model_t, torch.Tensor) \
        else torch.stack([model_t_i.clone()[:dummy_number] for model_t_i in model_t]).flatten()
    v_x_t = torch.zeros(x_t_shape, device=device, dtype=dtype)
    v_u_t = torch.zeros(u_t_shape, device=device, dtype=dtype)

    if packed:
        v_model_t = [v_model_t]
        v_x_t = [[v_x_t]]
        v_u_t = [[v_u_t]]
    video_diffusion_loss_fn = partial(denoiser.training_losses_fn, t=None, x0=None, xt=v_x_t, ut=v_u_t)

    return v_model_t, v_x_t, v_u_t, video_diffusion_loss_fn


def extend_seq_tensors(data, keys, extend_length, fill_value: dict | int = 0):
    for key in keys:
        fill_value_by_key = fill_value[key] if isinstance(fill_value, dict) else fill_value
        data[key] = torch.cat([
            data[key],
            torch.full(
                (data[key].size(0), extend_length),
                fill_value_by_key,
                device=data[key].device,
                dtype=data[key].dtype
            )
        ], dim=1)
    return data
def _media_to_list(media):
    if isinstance(media, list):
        return list(media)
    if isinstance(media, torch.Tensor):
        if media.dim() == 3:
            return [media]
        return [m for m in media]
    raise TypeError(f"Unsupported media type: {type(media)}")


def _pad_cond_latent_channels(latent, target_latent_channels):
    """Match cond_vae latent channels to the DiT patch embedding input channels."""
    if target_latent_channels is None:
        return latent
    channel_dim = 1 if latent.dim() == 5 else 0
    in_channels = latent.shape[channel_dim]
    if in_channels == target_latent_channels:
        return latent
    if in_channels > target_latent_channels:
        raise ValueError(
            f"cond_vae latent has {in_channels} channels, but model expects "
            f"{target_latent_channels}: shape={tuple(latent.shape)}"
        )
    pad_shape = list(latent.shape)
    pad_shape[channel_dim] = target_latent_channels - in_channels
    pad = torch.zeros(pad_shape, dtype=latent.dtype, device=latent.device)
    return torch.cat([latent, pad], dim=channel_dim)


# Tasks routed to prepare_model_t2v_inputs, the only provider that builds cond_vae_mask.
T2V_DATASET_PATTERNS = ["t2v*", "i2v*", "fl2v*", "t2va*", "r2v*", "r2va*"]


def compute_cond_vae_mask(batch):
    """Merged r2v reference-latent mask the model sees as ``cond_vae_mask``, or None.
    Runs on the raw batch so ``pp_tensor_shapes`` can predict the PP payload too."""
    if not multi_pattern_match(batch["dataset_tag"][0], T2V_DATASET_PATTERNS):
        return None
    merged = None
    for mask in (batch.get("cond_vae_image_mask"), batch.get("cond_vae_video_mask")):
        if mask is not None:
            mask = mask.bool()
            merged = mask if merged is None else (merged | mask)
    return merged.to(torch.long) if merged is not None else None


def _build_cond_text_scatter_mask(batch):
    """Full-seqlen scatter target for r2v text + cond_vit hidden states.
    """
    image_slices = (batch.get("cond_vit_image_slices") or [[]])[0]
    video_slices = (batch.get("cond_vit_video_slices") or [[]])[0]
    slices = list(image_slices) + list(video_slices)
    if not slices:
        return None
    te_mask = batch["text_mask"][0].clone().bool()
    seqlen = te_mask.size(0)
    for sli in slices:
        lo = max(sli.start - 1, 0)
        hi = min(sli.stop + 1, seqlen)
        te_mask[lo:hi] = True
    return te_mask.unsqueeze(0).to(batch["text_mask"].dtype)


def extract_cond_vit_inputs(batch, device):
    """Qwen3-VL vision inputs for r2v reference media, on ``device``; None if the batch
    has none. Images and videos stay on separate inputs (their grids differ)."""
    cond_vit_image_kwargs = batch.get("cond_vit_image_kwargs")
    cond_vit_video_kwargs = batch.get("cond_vit_video_kwargs")
    inputs = {}
    if cond_vit_image_kwargs is not None and "grid_thw" in cond_vit_image_kwargs:
        inputs["image_grid_thw"] = _flatten_qwen3vl_grid_thw(cond_vit_image_kwargs["grid_thw"])
        inputs["pixel_values"] = _flatten_qwen3vl_pixel_values(batch.get("cond_vit_images"))
    if cond_vit_video_kwargs is not None and "video_grid_thw" in cond_vit_video_kwargs:
        inputs["video_grid_thw"] = _flatten_qwen3vl_grid_thw(cond_vit_video_kwargs["video_grid_thw"])
        inputs["pixel_values_videos"] = _flatten_qwen3vl_pixel_values(batch.get("cond_vit_videos"))
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
    return inputs or None


def encode_text(text_encoder, batch, task_kwargs, device):
    sequence_pack = task_kwargs.get("sequence_pack", False)
    args = get_args()
    text_encoder_use_pack = getattr(args, "text_encoder_use_pack", False)
    if text_encoder is not None:
        cond_vit_inputs = extract_cond_vit_inputs(batch, device)
        if sequence_pack and text_encoder_use_pack and text_encoder.supports_packed_sequence:
            text_outputs = text_encoder.sequence_pack_encode(
                batch,
                system_prompt=get_system_prompt(task_kwargs["system_prompt_type"], None),
                sequence_pack=sequence_pack,
                vision_inputs=cond_vit_inputs,
            )
        else:
            cond_vit_inputs = cond_vit_inputs or {}
            text_outputs = text_encoder.batch_encode_with_sp(
                batch,
                system_prompt=get_system_prompt(task_kwargs["system_prompt_type"], None),
                sequence_pack=sequence_pack,
                pixel_values=cond_vit_inputs.get("pixel_values"),
                image_grid_thw=cond_vit_inputs.get("image_grid_thw"),
                pixel_values_videos=cond_vit_inputs.get("pixel_values_videos"),
                video_grid_thw=cond_vit_inputs.get("video_grid_thw"),
            )

        cond_text_states = text_outputs.hidden_state.contiguous().to(device)
        if sequence_pack:
            # In sequence_pack mode, batch_encode_with_sp and sequence_pack_encode both return
            # unpacked/padded text layout for downstream TokenRefiner compatibility.
            cond_text_mask = text_outputs.attention_mask.contiguous().to(device)
        else:
            cond_text_mask = batch["text_mask"].contiguous().to(device)
    else:
        cond_text_states, cond_text_mask = None, None
    return cond_text_states, cond_text_mask


def encode_text_batches(text_encoder, batches, task_kwargs_list, device):
    """Encode several packed micro-batches in ONE forward.

    Used by intra_dp_balance under --text-encoder-pack-microbatches; requires
    sequence_pack tasks with a packed-sequence-capable encoder. `task_kwargs_list`
    is aligned with `batches`, so batches of DIFFERENT tasks can be packed together
    (each sample uses its own task's system prompt) — still ONE forward (FSDP-safe).
    """
    assert text_encoder is not None, "encode_text_batches requires a built text encoder."
    args = get_args()
    for task_kwargs in task_kwargs_list:
        assert (task_kwargs.get("sequence_pack", False)
                and getattr(args, "text_encoder_use_pack", False)
                and text_encoder.supports_packed_sequence), (
            "encode_text_batches requires sequence_pack tasks, text_encoder_use_pack=True, "
            "and a packed-sequence-capable text encoder.")
    system_prompts = [get_system_prompt(tk["system_prompt_type"], None) for tk in task_kwargs_list]
    vision_inputs = [extract_cond_vit_inputs(b, device) for b in batches]
    outs = text_encoder.sequence_pack_encode_batches(
        batches, system_prompts=system_prompts, vision_inputs=vision_inputs)
    return [(o.hidden_state.contiguous().to(device), o.attention_mask.contiguous().to(device))
            for o in outs]


def compute_repa_feats(repa_encoder, batch, device, sequence_pack):
    """Run the REPA vision encoder on a batch's clean images."""
    def _feats(out):
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out

    if not sequence_pack:
        clean_pixels = batch["images"].to(device)
        if clean_pixels.dim() == 5:
            clean_pixels = clean_pixels.squeeze(dim=1)
        assert clean_pixels.ndim == 4, f"Unsupported media shape for REPA encoder: {clean_pixels.shape}"
        # map back to [0, 1] for vision encoders
        clean_pixels = (clean_pixels / 2 + 0.5).clamp(0, 1)
        with torch.no_grad():
            return _feats(repa_encoder.encode_images(clean_pixels))

    pixels_list = []
    for clean_pixel in batch["images"][0]:
        clean_pixel = clean_pixel.unsqueeze(0).to(device)
        assert clean_pixel.ndim == 4, f"Unsupported media shape for REPA encoder: {clean_pixel.shape}"
        clean_pixel = (clean_pixel / 2 + 0.5).clamp(0, 1)
        pixels_list.append(clean_pixel)
    if len(pixels_list) == 0:
        return []
    with torch.no_grad():
        repa_feats = _feats(repa_encoder.encode_images(pixels_list))
    return repa_feats if isinstance(repa_feats, list) else [repa_feats]


def prepare_model_t2i_inputs(batch: dict, device: int | str | torch.device, precomputed=None, **kwargs):
    is_vpp_first_chunk = kwargs.pop("is_vpp_first_chunk", True)
    _ = kwargs
    args = get_args()
    p_state: ParallelState = get_parallel_state()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")
    sequence_pack = task_kwargs.get("sequence_pack", False)

    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        vae = get_vae()
        # the encoder may be absent here under intra_dp_encode_by_pp_stage.
        text_encoder = None if (precomputed and "text_states" in precomputed) else get_text_encoder()
    else:
        vae = None
        text_encoder = None
    denoiser = get_denoiser()

    bsz = batch["tokens"].size(0)

    extra = {}
    if sequence_pack:
        extra["und_token_indices"] = batch["und_token_indices"].to(device)
        extra["gen_token_indices"] = batch["gen_token_indices"].to(device)
        extra["audio_token_indices"] = to_device(batch.get("audio_token_indices"), device)
        extra["sample_offsets"] = batch["offsets"]                                          # [n_samples + 1]
        extra["pad_count"] = batch["pad_count"]
        extra["und_token_lengths"] = batch["und_token_lengths"].to(device)                  # [n_samples]
        extra["gen_token_lengths"] = batch["gen_token_lengths"].to(device)                  # [n_samples]
        extra["audio_token_lengths"] = to_device(batch.get("audio_token_lengths"), device)  # [n_samples]

        # Verification
        assert extra["gen_token_lengths"].size(0) == len(batch["images"][0]), \
            (f"gen_token_lengths size {extra['gen_token_lengths'].size()} doesn't match number of "
             f"images {len(batch['images'][0])}")
        # audio is dummy
        assert extra["audio_token_lengths"].size(0) == 1, \
            f"audio_token_lengths size {extra['audio_token_lengths'].size()} should be 1 for dummy audio tokens."

        if args.use_input_ids:
            extra["input_ids"] = batch["tokens"]    # No need to move to device, because we only use its shape.
            extra["text_mask"] = batch["text_mask"].to(device)
            extra["visual_mask"] = batch["image_mask"].to(device)
            if args.add_timestep_token:
                extra["timesteps_index"] = batch["timesteps_index"].to(device)

    # ================================= prepare text condition ====================================
    # encode_load_balance: text may be pre-encoded by intra_dp_balance on the owner.
    if precomputed and "text_states" in precomputed:
        cond_text_states, cond_text_mask = precomputed["text_states"], precomputed["text_mask"]
    else:
        cond_text_states, cond_text_mask = encode_text(text_encoder, batch, task_kwargs, device)

    # ===================================== prepare diffusion =====================================
    extra['dummy_count'] = {
        'und': 0,
        'gen': 0,
        'audio': 0,
    }
    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        # Channel-concat extension type comes straight from the task's YAML kwargs; one of
        # `SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES` when `extend_latent_channels` is on.
        latent_channel_extend_type = task_kwargs.get("latent_channel_extend_type") \
            if args.extend_latent_channels else None
        assert not args.extend_latent_channels \
            or latent_channel_extend_type in SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES, (
            f"`latent_channel_extend_type` must be set in `{dataset_tag}_task_kwargs` to one of "
            f"{list(SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES)} when `extend_latent_channels=True`, "
            f"got {latent_channel_extend_type!r}."
        )
        # When precomputed image_latents are provided by encode_load_balance
        # Stage 5+ (worker already encoded), skip encode and only add noise.
        if precomputed and "image_latents" in precomputed:
            out = add_image_noise(
                vae, precomputed["image_latents"], denoiser, sample_type="sample",
                noise_dtype=args.noise_dtype, latent_channel_extend_type=latent_channel_extend_type,
            )
        elif task_kwargs.get("image_data_format", "pixels") == "latents":  # vae离线latent分支
            # batch["images"] 此时是 raw latent tensor [C, 1, H, W]，不是像素
            # 直接复用视频的 add_noise_to_latents：normalize -> add_noise
            out = add_noise_to_latents(
                vae,
                batch["images"],
                device,
                denoiser=denoiser,
                sample_type="sample",
                noise_dtype=args.noise_dtype,
                latent_channel_extend_type=latent_channel_extend_type,
            )
        else:
            out = vae_encode_and_add_noise(
                vae,
                batch["images"],
                device,
                denoiser=denoiser,
                sample_type="sample",
                keep_depth=kwargs["model_config"].img_proj_ndim == 3,
                vae_encode_type=args.vae_encode_type,
                noise_dtype=args.noise_dtype,
                latent_channel_extend_type=latent_channel_extend_type,
            )
        t, model_t, x_0, x_t, u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t
    else:
        t, model_t, x_0, x_t, u_t = None, None, None, None, None
    # x_t:
    #   [bsz, c, 1, h, w], [bsz, n, c, 1, h, w], or bsz * [n, 1, c, h, w]), or bsz * n * [1, c, h, w]
    # model_t:
    #   n == 1: [bsz], or bsz * [n]

    a_x_t, a_model_t, a_u_t = None, None, None
    audio_diffusion_loss_fn = None
    dummy_media_length = 0
    if args.audio_branch_model_name is not None:
        # all pp stages need dummy_media_length and indices/lengths
        dummy_media_length = sum(batch['dummy_type_dict'][0].values())
        if "audio_token_indices" in extra:  # packed
            if p_state.backend == 'megatron':
                assert extra["audio_token_indices"].numel() + dummy_media_length == p_state.cp_size, \
                    ("audio_token_indices should be None or empty tensor when using dummy audio tokens, "
                    f"as they are not associated with any real audio tokens. Got {extra['audio_token_indices']}.")
            start_pos = (extra["und_token_indices"].size(1) + extra["gen_token_indices"].size(1)
                         + extra["audio_token_indices"].size(1))
            extra["audio_token_indices"] = torch.cat([
                extra["audio_token_indices"],
                torch.arange(start_pos, start_pos + dummy_media_length, device=device).unsqueeze(0).expand(bsz, -1)
            ], dim=1)
            extra["audio_token_lengths"][-1] += dummy_media_length
            extra['dummy_count']['audio'] += dummy_media_length
            if args.use_input_ids:
                extra = extend_seq_tensors(
                    extra, keys=["input_ids", "text_mask", "visual_mask"], extend_length=dummy_media_length
                )
                extra["audio_mask"] = torch.zeros_like(extra["input_ids"], dtype=torch.bool, device=device)
                extra["audio_mask"][:, start_pos: start_pos + dummy_media_length] = True
        else:   # non-packed
            start_pos = None

        # NOTE(vpp): only append the dummy gen_audio on the first VP chunk. Under VPP,
        # chunk0 and chunk1 share the SAME rope_media_info list object (DataSubscriber does a
        # shallow ref copy), so appending again on chunk1 would duplicate the dummy audio
        # (1 -> 2) and break leo_model.py split_with_sizes (sum != total). chunk1 reads the
        # already-appended entry, so the guard yields the correct final state.
        if is_vpp_first_chunk:
            for infos in batch["rope_media_info"]:
                if start_pos is None:   # non-packed
                    start_pos = sum(math.prod(info[1]) for info in infos) + cond_text_mask.size(1)
                infos.append((
                    slice(start_pos, start_pos + dummy_media_length),
                    (dummy_media_length, 1, 1),
                    {"type": "gen_audio", "is_dummy": True}
                ))
        if _is_pre_process_stage(p_state, is_vpp_first_chunk):
            # Dummy audio tokens
            dtype = torch.float32 if args.noise_dtype == "fp32" else torch.bfloat16
            a_model_t, a_x_t, a_u_t, audio_diffusion_loss_fn = \
                create_audio_dummy_tokens(
                    bsz, denoiser, model_t, device, dtype, dummy_number=dummy_media_length,
                    packed=sequence_pack,
                )
        else:
            # last pp stage need to compute audio_diffusion_loss_fn
            audio_diffusion_loss_fn = denoiser.training_losses_fn

    if sequence_pack:
        real_media_length = None
        seq_len = batch["tokens"].size(1) + dummy_media_length
    else:
        real_media_length = sum([
            math.prod(info[1]) if not info[2].get("is_dummy", False) else 0
            for info in batch["rope_media_info"][0]
        ])
        if args.add_timestep_token:
            real_media_length += 1
        seq_len = cond_text_mask.size(1) + real_media_length + dummy_media_length

    # ================================== prepare attention mask ===================================
    # Attention mask
    attn_type = task_kwargs['attn_type']
    if attn_type in ['flash', 'flash3']:
        assert not sequence_pack, "Sequence packing is not supported for flash attention in t2i tasks."
        attention_mask = F.pad(cond_text_mask, (real_media_length, 0), value=1)
        if dummy_media_length > 0:
            attention_mask = F.pad(attention_mask, (0, dummy_media_length), value=0)

    elif attn_type in ['flash_packed', 'flash3_packed']:
        assert sequence_pack, f"Sequence packing must be enabled for {attn_type} attention."
        n_samples = batch["n_samples"].item()
        assert n_samples == extra["sample_offsets"].size(1) - 1, \
            f"n_samples ({n_samples} doesn't match sample_offsets - 1 ({extra['sample_offsets'].size(1) - 1})"
        # Sample lengths mask. Used for `unpad_input_for_concatenated_sequences` of FlashAttention.
        attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
        attention_mask[0, :n_samples] = extra["sample_offsets"][0, 1:] - extra["sample_offsets"][0, :-1]

    elif attn_type == "flex":
        assert sequence_pack, "Sequence packing must be enabled for flex attention in t2i tasks."
        assert seq_len % 128 == 0, f"Sequence length {seq_len} must be divisible by 128 for flex attention."
        if bsz == 1:
            image_slices = batch["image_slices"][0]
            mask_mod = create_text_image_mask_mod(
                image_slices, seq_len, device, offsets=batch["offsets"][0],
            )
            attention_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
        else:
            mask_mod = create_batch_text_image_mask_mod(
                batch["image_slices"], seq_len, device, batch_offsets=batch["offsets"],
            )
            attention_mask = create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    # ===================================== REPA Features =====================================
    repa_feats = None
    if args.use_repa and _is_pre_process_stage(p_state, is_vpp_first_chunk):
        if precomputed and "repa_feats" in precomputed:
            repa_feats = precomputed["repa_feats"]
        else:
            repa_encoder = get_repa_encoder()
            if repa_encoder is not None:
                repa_feats = compute_repa_feats(repa_encoder, batch, device, sequence_pack)

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        attention_mask=attention_mask,      # [bsz, 1, img_seqlen + txt_seqlen]
        rope_media_info=batch["rope_media_info"],
        # -- gen image/video
        latents=x_t,                        # [bsz, c, h, w]
        timesteps=model_t,                  # [bsz]
        # -- gen audio
        audio_latents=a_x_t,                # [bsz, c, l]
        audio_timesteps=a_model_t,          # [bsz]
        # -- cond text
        cond_text_states=cond_text_states,  # [bsz, txt_seqlen, te_dim]
        cond_text_mask=cond_text_mask,      # [bsz, txt_seqlen]
        # for training
        diffusion_loss_fn=partial(denoiser.training_losses_fn, t=t, x0=x_0, xt=x_t, ut=u_t),
        audio_diffusion_loss_fn=audio_diffusion_loss_fn,
        visual_loss_weight=args.image_loss_weight,
        audio_loss_weight=0,
        repa_feats=repa_feats,
        dataset_tag=dataset_tag,
        # for pp
        ut=u_t,                             # [bsz, c, h, w] velocity target for flow matching
        aut=a_u_t,                          # [bsz, c, l] velocity target for flow matching
        # extra
        **extra,
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    # Save model_input_kwargs and a few more info for reconstructing the attention mask
    # Save raw images for training data visualization
    save_first_training_samples(model_input_kwargs, extra=dict(
        tokens=batch["tokens"],
        text_mask=batch["text_mask"],
        raw_images=batch.get("images", None),
    ))
    return model_input_kwargs, bsz, seq_len


def prepare_model_t2v_inputs(batch: dict, device: int | str | torch.device, precomputed=None, **kwargs):
    is_vpp_first_chunk = kwargs.pop("is_vpp_first_chunk", True)
    model_config = kwargs.get("model_config")
    args = get_args()
    p_state: ParallelState = get_parallel_state()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")
    sequence_pack = task_kwargs.get("sequence_pack", False)
    audio_data_format = task_kwargs.get("audio_data_format", "single_slice_file")
    latent_pre_scaled = task_kwargs.get("latent_pre_scaled", False)

    audio_vae = None
    has_real_audios = "audios" in batch
    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        vae = get_vae()
        # the encoder may be absent here under intra_dp_encode_by_pp_stage.
        text_encoder = None if (precomputed and "text_states" in precomputed) else get_text_encoder()
        if has_real_audios:
            audio_vae = get_audio_vae()
    else:
        vae = None
        text_encoder = None

    video_denoiser = get_video_denoiser()
    if args.audio_branch_model_name is not None:
        audio_denoiser = get_audio_denoiser()
    else:
        audio_denoiser = None

    bsz = batch["tokens"].size(0)

    extra = {}
    if sequence_pack:
        extra["und_token_indices"] = batch["und_token_indices"].to(device)
        extra["gen_token_indices"] = batch["gen_token_indices"].to(device)
        extra["audio_token_indices"] = to_device(batch.get("audio_token_indices"), device)
        extra["sample_offsets"] = batch["offsets"]
        extra["pad_count"] = batch["pad_count"]
        extra["und_token_lengths"] = batch["und_token_lengths"].to(device)
        extra["gen_token_lengths"] = batch["gen_token_lengths"].to(device)
        extra["audio_token_lengths"] = to_device(batch.get("audio_token_lengths"), device)

        # Verification
        assert extra["gen_token_lengths"].size(0) == len(batch["videos"][0]), \
            (f"gen_token_lengths size {extra['gen_token_lengths'].size()} doesn't match number of "
             f"videos {len(batch['videos'][0])}")
        if not has_real_audios:
            # audio is dummy
            assert extra["audio_token_lengths"].size(0) == 1, \
                f"audio_token_lengths size {extra['audio_token_lengths'].size()} should be 1 for dummy audio tokens."

        if args.use_input_ids:
            extra["input_ids"] = batch["tokens"]    # No need to move to device, because we only use its shape.
            extra["text_mask"] = batch["text_mask"].to(device)
            extra["visual_mask"] = batch["video_mask"].to(device)
            extra["audio_mask"] = to_device(batch.get("audio_mask"), device)
            extra["cond_vae_mask"] = to_device(compute_cond_vae_mask(batch), device)
            if args.add_timestep_token:
                extra["video_timesteps_index"] = batch["video_timesteps_index"].to(device)

            cond_text_scatter_mask = _build_cond_text_scatter_mask(batch)
            if cond_text_scatter_mask is not None:
                extra["cond_text_scatter_mask"] = cond_text_scatter_mask.to(device)

    # ================================= prepare text condition ====================================
    # encode_load_balance: text may be pre-encoded by intra_dp_balance on the owner.
    if precomputed and "text_states" in precomputed:
        cond_text_states, cond_text_mask = precomputed["text_states"], precomputed["text_mask"]
    else:
        cond_text_states, cond_text_mask = encode_text(text_encoder, batch, task_kwargs, device)

    # ===================================== prepare diffusion =====================================
    extra['dummy_count'] = {
        'und': 0,
        'gen': 0,
        'audio': 0,
    }
    # -- video
    # Also extend channels for legacy i2v and fl2v if applicable.
    latent_channel_extend_type = kwargs.get("latent_channel_extend_type", task_kwargs.get("latent_channel_extend_type")) \
        if args.extend_latent_channels else None
    assert not args.extend_latent_channels \
        or latent_channel_extend_type in SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES, (
        f"`latent_channel_extend_type` must be set in `{dataset_tag}_task_kwargs` to one of "
        f"{list(SUPPORTED_LATENT_CHANNEL_EXTEND_TYPES)} when `extend_latent_channels=True`, "
        f"got {latent_channel_extend_type!r}."
    )
    if args.video_data_format == "latents" and _is_pre_process_stage(p_state, is_vpp_first_chunk):
        # encode_load_balance: fl2v last frame may be pre-encoded by intra_dp_balance
        out = add_noise_to_latents(
            vae,
            batch["videos"],
            device,
            denoiser=video_denoiser,
            sample_type="sample",
            last_frame=to_device(batch.get("video_last_frames"), device),
            last_frame_is_offline_latent=(task_kwargs.get("last_frame_data_format", "pixels") == "latents"), # last_frame_is_offline_latent: 是否离线提取latent，离线提取的latent不带normalization，需要做normalize
            vae_encode_type=args.vae_encode_type,
            noise_dtype=args.noise_dtype,
            latent_channel_extend_type=latent_channel_extend_type,
            last_frame_latents=precomputed.get("last_frame_latents") if precomputed else None,
            timesteps=kwargs.get("timesteps"),
            latent_pre_scaled=latent_pre_scaled,
        )
        v_t, v_model_t, v_x_0, v_x_t, v_u_t = out.t, out.model_t, out.x_0, out.x_t, out.u_t
    else:
        v_t, v_model_t, v_x_0, v_x_t, v_u_t = None, None, None, None, None

    # -- audio
    a_t, a_model_t, a_x_0, a_x_t, a_u_t = None, None, None, None, None
    audio_diffusion_loss_fn = None
    dummy_media_length = 0
    audio_loss_weight = 0.0
    if has_real_audios:
        if _is_pre_process_stage(p_state, is_vpp_first_chunk):
            # When --decouple-va-timestep is set, audio samples its own t with its own SNR
            # distribution; otherwise audio reuses video's t to keep the original coupled behavior.
            audio_timesteps = None if getattr(args, "decouple_va_timestep", False) else v_t
            # See t2i path for precomputed handling.
            if precomputed and "audio_latents" in precomputed:
                a_out = add_audio_noise(
                    audio_vae, precomputed["audio_latents"], audio_denoiser, sample_type="sample",
                    noise_dtype=args.noise_dtype, timesteps=audio_timesteps, device=device,
                )
            else:
                a_out = audio_vae_encode_and_add_noise(
                    audio_vae,
                    audios=batch["audios"],
                    device=device,
                    denoiser=audio_denoiser,    # noqa
                    sample_type="sample",
                    noise_dtype=args.noise_dtype,
                    timesteps=audio_timesteps,
                    audio_data_format=audio_data_format,
                )
            a_t, a_model_t, a_x_0, a_x_t, a_u_t = a_out.t, a_out.model_t, a_out.x_0, a_out.x_t, a_out.u_t
        # all pp stages need audio_loss_weight and audio_diffusion_loss_fn
        audio_diffusion_loss_fn = partial(audio_denoiser.training_losses_fn, t=a_t, x0=a_x_0, xt=a_x_t, ut=a_u_t)
        audio_loss_weight = args.audio_loss_weight
    elif args.audio_branch_model_name is not None:
        # all pp stages need dummy_media_length and indices/lengths
        dummy_media_length = sum(batch['dummy_type_dict'][0].values())
        if "audio_token_indices" in extra:  # packed
            if p_state.backend == 'megatron':
                assert (extra["audio_token_indices"] is None
                        or extra["audio_token_indices"].numel() + dummy_media_length == p_state.cp_size), \
                    ("audio_token_indices should be None or empty tensor when using dummy audio tokens, "
                        "as they are not associated with any real audio tokens. Got {extra['audio_token_indices']}.")
            start_pos = (extra["und_token_indices"].size(1) + extra["gen_token_indices"].size(1)
                         + extra["audio_token_indices"].size(1))
            extra["audio_token_indices"] = torch.cat([
                extra["audio_token_indices"],
                torch.arange(start_pos, start_pos + dummy_media_length, device=device).unsqueeze(0).expand(bsz, -1)
            ], dim=1)
            extra["audio_token_lengths"][-1] += dummy_media_length
            extra['dummy_count']['audio'] += dummy_media_length
            if args.use_input_ids:
                extend_keys = ["input_ids", "text_mask", "visual_mask"]
                if extra.get("cond_vae_mask") is not None:
                    extend_keys.append("cond_vae_mask")
                if extra.get("cond_text_scatter_mask") is not None:
                    extend_keys.append("cond_text_scatter_mask")
                extra = extend_seq_tensors(
                    extra, keys=extend_keys, extend_length=dummy_media_length
                )
                if extra["audio_mask"] is None:
                    extra["audio_mask"] = torch.zeros_like(extra["input_ids"], dtype=torch.bool, device=device)
                extra["audio_mask"][:, start_pos: start_pos + dummy_media_length] = True
        else:   # non-packed
            start_pos = None

        # NOTE(vpp): guard dummy gen_audio append to first VP chunk only.
        # Shared rope_media_info object across chunks -> appending on chunk1 duplicates the
        # dummy audio and breaks leo_model.py split_with_sizes.
        if is_vpp_first_chunk:
            for infos in batch["rope_media_info"]:
                if start_pos is None:   # non-packed
                    start_pos = sum(math.prod(info[1]) for info in infos) + cond_text_mask.size(1)
                infos.append((
                    slice(start_pos, start_pos + dummy_media_length),
                    (dummy_media_length, 1, 1),
                    {"type": "gen_audio", "is_dummy": True}
                ))
        if _is_pre_process_stage(p_state, is_vpp_first_chunk):
            dtype = torch.float32 if args.noise_dtype == "fp32" else torch.bfloat16
            a_model_t, a_x_t, a_u_t, audio_diffusion_loss_fn = \
                create_audio_dummy_tokens(
                    bsz, audio_denoiser, v_model_t, device, dtype, dummy_number=dummy_media_length,
                    packed=sequence_pack,
                )
        else:
            # last pp stage need to compute audio_diffusion_loss_fn
            audio_diffusion_loss_fn = audio_denoiser.training_losses_fn

    cond_vae_latents_scatter = None
    cond_vae_timesteps = None
    if args.use_input_ids and p_state.pp_rank == 0 and v_x_t is not None:
        cond_media_list = []
        # (1) reference images: pixels -> VAE encode (no noise)
        raw_cond_vae_images = batch.get("cond_vae_images")
        if raw_cond_vae_images is not None:
            cond_out = vae_encode_and_add_noise(
                vae, raw_cond_vae_images, device,
                denoiser=video_denoiser, sample_type=None,
                vae_encode_type=args.vae_encode_type,
                noise_dtype=args.noise_dtype,
            )
            img_latents = cond_out.latents
            if img_latents is not None:
                img_flat = img_latents[0] if isinstance(img_latents, list) else img_latents
                cond_media_list.extend(_media_to_list(img_flat))
        # (2) source videos: offline .npy / online-encoded VAE latents. When not pre-scaled
        #     (e.g. vae251 raw latents) normalize to the model latent space so the source-video
        #     cond matches the reference-image cond and the gen stream.
        raw_cond_vae_videos = batch.get("cond_vae_videos")
        if raw_cond_vae_videos is not None:
            vid_flat = raw_cond_vae_videos[0] if isinstance(raw_cond_vae_videos, list) else raw_cond_vae_videos
            vid_list = vid_flat if isinstance(vid_flat, list) else _media_to_list(vid_flat)
            vid_list = [v.to(device) for v in vid_list]
            if not latent_pre_scaled:
                vid_list = [normalize_vae_latents(vae, v.unsqueeze(0)).squeeze(0) for v in vid_list]
            cond_media_list.extend(vid_list)
        if cond_media_list:
            target_latent_channels = (
                getattr(model_config, "img_latent_in_channels", None)
                or getattr(args, "img_latent_in_channels", None)
                or getattr(args, "vae_latent_dim", None)
            )
            ref_t = v_model_t[0] if isinstance(v_model_t, list) else v_model_t
            cond_list = [_pad_cond_latent_channels(c, target_latent_channels) for c in cond_media_list]
            if "proj_in" in model_config.fp32_modules:
                cond_list = [c.to(torch.float32) for c in cond_list]
            cond_vae_latents_scatter = [cond_list]
            cond_vae_timesteps = [torch.zeros(len(cond_list), device=device, dtype=ref_t.dtype)]

    # t2va/r2va
    # batch["rope_media_info"][0][0] => (None, (duration1, height, width), {"type": "gen_video"}) => video
    # batch["rope_media_info"][0][1] => (None, (duration2,), {"type": "gen_audio"}) => audio
    # t2v/i2v/fl2v/r2v
    # batch["rope_media_info"][0][0] => (None, (duration1, height, width), {"type": "gen_video"})
    # batch["rope_media_info"][0][1] => (None, (dummy_media_length,), {"type": "gen_audio", "is_dummy": True})
    if sequence_pack:
        real_media_length = None
        seq_len = batch["tokens"].size(1) + dummy_media_length
    else:
        real_media_length = sum([
            math.prod(info[1]) if not info[2].get("is_dummy", False) else 0
            for info in batch["rope_media_info"][0]
        ])
        seq_len = cond_text_mask.size(1) + real_media_length + dummy_media_length

    # ================================== prepare attention mask ===================================
    # Attention mask
    attn_type = task_kwargs['attn_type']
    if attn_type in ['flash', 'flash3']:
        assert not sequence_pack, "Sequence packing is not supported for flash attention in t2i tasks."
        attention_mask = F.pad(cond_text_mask, (real_media_length, 0), value=1)
        if dummy_media_length > 0:
            attention_mask = F.pad(attention_mask, (0, dummy_media_length), value=0)

    elif attn_type in ['flash_packed', 'flash3_packed']:
        assert sequence_pack, f"Sequence packing must be enabled for {attn_type} attention."
        n_samples = batch["n_samples"].item()
        assert n_samples == extra["sample_offsets"].size(1) - 1, \
            f"n_samples ({n_samples} doesn't match sample_offsets - 1 ({extra['sample_offsets'].size(1) - 1})"
        # Sample lengths mask. Used for `unpad_input_for_concatenated_sequences` of FlashAttention.
        attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
        attention_mask[0, :n_samples] = extra["sample_offsets"][0, 1:] - extra["sample_offsets"][0, :-1]

    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        attention_mask=attention_mask,      # [bsz, 1, img_seqlen + txt_seqlen]
        rope_media_info=batch["rope_media_info"],
        # -- gen image/video
        latents=v_x_t,                      # [bsz, c, t, h, w]
        timesteps=v_model_t,                # [bsz]
        # -- gen audio
        audio_latents=a_x_t,                # [bsz, c, l]
        audio_timesteps=a_model_t,          # [bsz]
        # -- cond text
        cond_text_states=cond_text_states,  # [bsz, txt_seqlen, te_dim]
        cond_text_mask=cond_text_mask,      # [bsz, txt_seqlen]
        # -- cond reference vae for full-seqlen (use_input_ids) scatter path (r2v / r2va)
        cond_vae_latents=cond_vae_latents_scatter,
        cond_vae_timesteps=cond_vae_timesteps,
        # for training
        diffusion_loss_fn=partial(video_denoiser.training_losses_fn, t=v_t, x0=v_x_0, xt=v_x_t, ut=v_u_t),
        audio_diffusion_loss_fn=audio_diffusion_loss_fn,
        dataset_tag=dataset_tag,
        visual_loss_weight=args.image_loss_weight,
        audio_loss_weight=audio_loss_weight,
        # for pp
        ut=v_u_t,                           # [bsz, c, t, h, w] velocity target for flow matching
        aut=a_u_t,                          # [bsz, c, l] velocity target for flow matching
        # extra
        **extra,
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    # Save model_input_kwargs and a few more info for reconstructing the attention mask
    save_first_training_samples(model_input_kwargs, extra=dict(
        tokens=batch["tokens"],
        text_mask=batch["text_mask"],
        raw_videos=batch["videos"],
        raw_last_frame=batch.get("video_last_frames"),
        raw_audios=batch.get("audios", None),
        raw_cond_vae_images=batch.get("cond_vae_images"),
        raw_cond_vit_images=batch.get("cond_vit_images"),
    ))
    return model_input_kwargs, bsz, seq_len


def prepare_model_t2a_inputs(batch: dict, device: int | str | torch.device, precomputed=None, **kwargs):
    """ Prepare model inputs for T2A (text-to-audio) training.

    Uses real audio latents and dummy video latents (video_loss_weight=0).
    This is symmetric to prepare_model_t2i_inputs which uses real video + dummy audio.
    """
    is_vpp_first_chunk = kwargs.pop("is_vpp_first_chunk", True)
    _ = kwargs
    args = get_args()
    p_state: ParallelState = get_parallel_state()
    dataset_tag = batch["dataset_tag"][0]
    task_kwargs = getattr(args, f"{dataset_tag}_task_kwargs")
    sequence_pack = task_kwargs.get("sequence_pack", False)
    audio_data_format = task_kwargs.get("audio_data_format", "single_slice_file")

    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        audio_vae = get_audio_vae()
        # the encoder may be absent here under intra_dp_encode_by_pp_stage.
        text_encoder = None if (precomputed and "text_states" in precomputed) else get_text_encoder()
    else:
        audio_vae = None
        text_encoder = None
    audio_denoiser = get_audio_denoiser()

    bsz = batch["tokens"].size(0)

    extra = {}
    if sequence_pack:
        extra["und_token_indices"] = batch["und_token_indices"].to(device)
        extra["gen_token_indices"] = batch["gen_token_indices"].to(device)
        extra["audio_token_indices"] = to_device(batch.get("audio_token_indices"), device)
        extra["sample_offsets"] = batch["offsets"]
        extra["pad_count"] = batch["pad_count"]
        extra["und_token_lengths"] = batch["und_token_lengths"].to(device)
        extra["gen_token_lengths"] = batch["gen_token_lengths"].to(device)
        extra["audio_token_lengths"] = to_device(batch.get("audio_token_lengths"), device)

        # Verification
        if "audios" in batch:
            assert extra["audio_token_lengths"].size(0) == len(batch["audios"][0]), \
                (f"audio_token_lengths size {extra['audio_token_lengths'].size()} doesn't match number of "
                 f"audios {len(batch['audios'][0])}")
        # video is dummy
        assert extra["gen_token_lengths"].size(0) == 1, \
            f"gen_token_lengths size {extra['gen_token_lengths'].size()} should be 1 for dummy image/video tokens."

        if args.use_input_ids:
            extra["input_ids"] = batch["tokens"]    # No need to move to device, because we only use its shape.
            extra["text_mask"] = batch["text_mask"].to(device)
            extra["audio_mask"] = batch["audio_mask"].to(device)
            if args.add_timestep_token:
                extra["audio_timesteps_index"] = batch["audio_timesteps_index"].to(device)

    # ================================= prepare text condition ====================================
    # encode_load_balance: text may be pre-encoded by intra_dp_balance on the owner.
    if precomputed and "text_states" in precomputed:
        cond_text_states, cond_text_mask = precomputed["text_states"], precomputed["text_mask"]
    else:
        cond_text_states, cond_text_mask = encode_text(text_encoder, batch, task_kwargs, device)

    # ===================================== prepare diffusion =====================================
    extra['dummy_count'] = {
        'und': 0,
        'gen': 0,
        'audio': 0,
    }
    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        # See t2i path for precomputed handling.
        if precomputed and "audio_latents" in precomputed:
            a_out = add_audio_noise(
                audio_vae, precomputed["audio_latents"], audio_denoiser, sample_type="sample",
                noise_dtype=args.noise_dtype, device=device,
            )
        else:
            a_out = audio_vae_encode_and_add_noise(
                audio_vae,
                audios=batch["audios"],
                device=device,
                denoiser=audio_denoiser,
                sample_type="sample",
                noise_dtype=args.noise_dtype,
                audio_data_format=audio_data_format,
            )
        a_t, a_model_t, a_x_0, a_x_t, a_u_t = a_out.t, a_out.model_t, a_out.x_0, a_out.x_t, a_out.u_t
    else:
        a_t, a_model_t, a_x_0, a_x_t, a_u_t = None, None, None, None, None

    # Dummy video tokens
    dtype = torch.bfloat16 if args.bf16 and (args.fp32_modules is None or 'proj_in' not in args.fp32_modules) else torch.float32
    dummy_media_length = sum(batch['dummy_type_dict'][0].values())
    if _is_pre_process_stage(p_state, is_vpp_first_chunk):
        v_model_t, v_x_t, v_u_t, video_diffusion_loss_fn = \
            create_video_dummy_tokens(
                bsz, get_video_denoiser(), a_model_t, device, dtype, dummy_number=dummy_media_length, packed=sequence_pack,
            )
    else:
        # last pp stage only needs video_diffusion_loss_fn
        v_model_t, v_x_t, v_u_t = None, None, None
        video_diffusion_loss_fn = partial(
            get_video_denoiser().training_losses_fn, t=None, x0=None, xt=None,
        )
    if "gen_token_indices" in extra:  # packed
        if p_state.backend == 'megatron':
            assert (extra["gen_token_indices"] is None
                    or extra["gen_token_indices"].numel() + dummy_media_length == p_state.cp_size), \
                ("gen_token_indices should be None or empty tensor when using dummy video tokens, "
                f"as they are not associated with any real video tokens. Got {extra['gen_token_indices']}.")
        start_pos = (extra["und_token_indices"].size(1) + extra["gen_token_indices"].size(1)
                     + extra["audio_token_indices"].size(1))
        extra["gen_token_indices"] = torch.cat([
            extra["gen_token_indices"],
            torch.arange(start_pos, start_pos + dummy_media_length, device=device).unsqueeze(0).expand(bsz, -1)
        ], dim=1)
        extra["gen_token_lengths"][-1] += dummy_media_length
        extra['dummy_count']['gen'] += dummy_media_length
        if args.use_input_ids:
            extra = extend_seq_tensors(
                extra, keys=["input_ids", "text_mask", "audio_mask"], extend_length=dummy_media_length
            )
            extra["visual_mask"] = torch.zeros_like(extra["input_ids"], dtype=torch.bool, device=device)
            extra["visual_mask"][:, start_pos:start_pos + dummy_media_length] = True
    else:   # non-packed
        start_pos = None

    # NOTE(vpp): guard dummy gen_video append to first VP chunk only.
    # Shared rope_media_info object across chunks -> appending on chunk1 duplicates the
    # dummy entry and breaks leo_model.py split_with_sizes.
    if is_vpp_first_chunk:
        for infos in batch["rope_media_info"]:
            if start_pos is None:
                start_pos = sum(math.prod(info[1]) for info in infos) + cond_text_mask.size(1)
            infos.append((
                slice(start_pos, start_pos + dummy_media_length),
                (dummy_media_length, 1, 1),
                {"type": "gen_video", "is_dummy": True}
            ))

    if sequence_pack:
        real_media_length = None
        seq_len = batch["tokens"].size(1) + dummy_media_length
    else:
        real_media_length = sum([
            math.prod(info[1]) if not info[2].get("is_dummy", False) else 0
            for info in batch["rope_media_info"][0]
        ])
        if args.add_timestep_token:
            real_media_length += 1
        seq_len = cond_text_mask.size(1) + real_media_length + dummy_media_length

    # ================================== prepare attention mask ===================================
    # Attention mask
    attn_type = task_kwargs['attn_type']
    if attn_type in ['flash', 'flash3']:
        assert not sequence_pack, "Sequence packing is not supported for flash attention in t2a tasks."
        attention_mask = F.pad(cond_text_mask, (real_media_length, 0), value=1)
        if dummy_media_length > 0:
            attention_mask = F.pad(attention_mask, (0, dummy_media_length), value=0)

    elif attn_type in ['flash_packed', 'flash3_packed']:
        assert sequence_pack, f"Sequence packing must be enabled for {attn_type} attention."
        n_samples = batch["n_samples"].item()
        assert n_samples == extra["sample_offsets"].size(1) - 1, \
            f"n_samples ({n_samples} doesn't match sample_offsets - 1 ({extra['sample_offsets'].size(1) - 1})"
        # Sample lengths mask. Used for `unpad_input_for_concatenated_sequences` of FlashAttention.
        attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long, device=device)
        attention_mask[0, :n_samples] = extra["sample_offsets"][0, 1:] - extra["sample_offsets"][0, :-1]

    else:
        raise NotImplementedError(f"Attention type {attn_type} is not supported.")

    # ===================================== Pack model kwargs =====================================
    model_input_kwargs = dict(
        attention_mask=attention_mask,      # [bsz, 1, img_seqlen + txt_seqlen]
        rope_media_info=batch["rope_media_info"],
        # -- gen image/video
        latents=v_x_t,                      # [bsz, c, 1, h, w]
        timesteps=v_model_t,                # [bsz]
        # -- gen audio
        audio_latents=a_x_t,                # [bsz, c, l]
        audio_timesteps=a_model_t,          # [bsz]
        # -- cond text
        cond_text_states=cond_text_states,  # [bsz, txt_seqlen, te_dim]
        cond_text_mask=cond_text_mask,      # [bsz, txt_seqlen]
        # for training
        diffusion_loss_fn=video_diffusion_loss_fn,
        audio_diffusion_loss_fn=partial(audio_denoiser.training_losses_fn, t=a_t, x0=a_x_0, xt=a_x_t, ut=a_u_t),
        visual_loss_weight=0.,
        audio_loss_weight=args.audio_loss_weight,
        dataset_tag=dataset_tag,
        # for pp
        ut=v_u_t,                           # [bsz, c, 1, h, w] velocity target for flow matching
        aut=a_u_t,                          # [bsz, c, l] velocity target for flow matching
        # extra
        **extra,
    )
    # Remove None entries
    model_input_kwargs = {k: v for k, v in model_input_kwargs.items() if v is not None}

    # Save model_input_kwargs and a few more info for reconstructing the attention mask
    save_first_training_samples(model_input_kwargs, extra=dict(
        tokens=batch["tokens"],
        text_mask=batch["text_mask"],
        raw_audios=batch.get("audios", None),
    ))
    return model_input_kwargs, bsz, seq_len


def prepare_model_inputs(batch: dict, device: int | str | torch.device, precomputed=None, **kwargs):
    """Build model_input_kwargs for a single micro-batch.

    Parameters
    ----------
    precomputed : dict | None
        Optional pre-encoded results injected by the encode_load_balance path
        (Stage 5+). Possible keys:
          - "audio_latents": output of audio_vae_encode_pure (pre-noise)
          - "image_latents": output of vae_encode_pure (pre-noise)
          - "text_states", "text_mask": output of text_encoder forward
        When present on the owner rank, the corresponding encoder call is
        skipped; add_*_noise (RNG) still runs locally for reproducibility.
        Stage 1: always None or {}; sub-functions ignore via **kwargs.
    """
    ss = get_scalar_state()
    ss.current_forward_times += 1
    with torch.autocast(device_type="cuda", enabled=False):
        if multi_pattern_match(batch["dataset_tag"][0], ["t2i*"]):
            model_input_kwargs, bsz, seqlen = prepare_model_t2i_inputs(
                batch, device, precomputed=precomputed, **kwargs,
            )
        elif multi_pattern_match(batch["dataset_tag"][0], T2V_DATASET_PATTERNS):
            model_input_kwargs, bsz, seqlen = prepare_model_t2v_inputs(
                batch, device, precomputed=precomputed, **kwargs,
            )
        elif multi_pattern_match(batch["dataset_tag"][0], ["t2a*"]):
            model_input_kwargs, bsz, seqlen = prepare_model_t2a_inputs(
                batch, device, precomputed=precomputed, **kwargs,
            )
        else:
            raise NotImplementedError(f"dataset_tag `{batch['dataset_tag'][0]}` not recognized in forward_step.")

    # MFU tracking: record per-microbatch token distribution for precise FLOPs calculation
    p_state: ParallelState = get_parallel_state()
    if p_state.backend == "megatron":
        try:
            from megatron.training import get_args
            if getattr(get_args(), 'leo_mfu_log', False):
                from angelptm.megatron.core.models.leo.mfu import record_real_seq_lengths
                record_real_seq_lengths(bsz, model_input_kwargs)
        except Exception as _mfu_e:
            import traceback as _tb
            print(f"[Leo MFU] WARN prepare_model_inputs.record_real_seq_lengths failed: "
                  f"{type(_mfu_e).__name__}: {_mfu_e}\n{_tb.format_exc()}", flush=True)

    return model_input_kwargs, bsz, seqlen
