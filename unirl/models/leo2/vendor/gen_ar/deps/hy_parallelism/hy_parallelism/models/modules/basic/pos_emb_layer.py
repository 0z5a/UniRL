import bisect
import math
from typing import Optional, Tuple, List

import torch
import numpy as np

from .rope import get_meshgrid_nd, get_1d_rotary_pos_embed


# TODO(ckczzjzhang): Support n-d RoPE.
def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, base_rescale_factor: float = 1.0, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$

    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))

    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def get_text_image_2d_rope(
        image_infos: Optional[List[Tuple[slice, Tuple[int, int], dict]]], seq_len: int, n_elem: int,
        device: Optional[torch.device] = None, base: float = 10000.0, base_rescale_factor: float = 1.0, condense_ratio: int = 1,
        sample_offsets: Optional[torch.Tensor] = None, return_all_pos: bool = False,
):
    """
    Reference: https://kexue.fm/archives/10352

    (β₁ + γ₁, β₂ + γ₂),  (β₁ + γ₁, β₂ + 2γ₂), ...,  (β₁ + γ₁, β₂ + wγ₂)
    (β₁ + 2γ₁, β₂ + γ₂), (β₁ + 2γ₁, β₂ + 2γ₂), ..., (β₁ + 2γ₁, β₂ + wγ₂)
    ...
    (β₁ + hγ₁, β₂ + γ₂), (β₁ + hγ₁, β₂ + 2γ₂), ..., (β₁ + hγ₁, β₂ + wγ₂)

    We assume the last text token has position (L, L), and the next text token has position (L + hw + 1, L + hw + 1).
    We let the distance between the last text token and the first image token equals to the distance between the last
    image token and the next text token.

    (β₁ + γ₁, β₂ + γ₂) - (L, L) = (L + hw + 1, L + hw + 1) - (β₁ + hγ₁, β₂ + wγ₂)

    Let γ₁ = γ₂ = 1, then we have
        β₁ = beta_y = L + (wh - h)/2
        β₂ = beta_x = L + (wh - w)/2

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)    # [1, half_d, 2]

    # position indices
    if image_infos is None:
        image_infos = []

    # Now we require image_info to be (slice, (h, w), dict), where dict can store extra meta info.
    # We add an empty dict if not provided for backward compatibility.
    for i in range(len(image_infos)):
        if len(image_infos[i]) == 2:
            image_infos[i] = (image_infos[i][0], image_infos[i][1], {})
    # when using sequence packing
    if sample_offsets is not None:
        sample_offsets = sample_offsets.tolist()
        image_infos_list = []
        last_id = -1
        for sli, (h, w), meta in image_infos:
            sample_id = bisect.bisect_right(sample_offsets[1:], sli.stop - 1)
            # Dummy info will be treated as part of the last sample.
            if meta.get("is_dummy", False) and sample_id != last_id:
                sample_id -= 1
                assert sample_id == last_id, "Dummy media info should belong to the last sample."
            if sample_id != last_id:
                image_infos_list.append([])
            offset = sample_offsets[sample_id]
            image_infos_list[-1].append((slice(sli.start - offset, sli.stop - offset), (h, w), meta))
            last_id = sample_id
        sample_seq_lens = [sample_offsets[i + 1] - sample_offsets[i] for i in range(len(sample_offsets) - 1)]
        # Add padding for the last sample
        sample_seq_lens[-1] += seq_len - sample_offsets[-1]
    else:
        image_infos_list = [image_infos]
        sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w), meta in sample_image_infos:
            if meta.get("is_dummy", False):
                continue
            # anyres vit has a `begin` token, image token start from position 1.
            L = sec_slice.start + meta.get("start_offset", 0)
            if last_pos < L:    # previous text
                y_sections.append(np.arange(last_pos, L))
                x_sections.append(np.arange(last_pos, L))
            elif h is None:     # and L <= last_pos, this is shifted text part
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                y_sections.append(np.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(np.arange(sec_slice.start, sec_slice.stop))
                last_pos = sec_slice.stop
                continue
            else:   # and L <= last_pos, this is shifted image part
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass
            # current image
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))  # [2, h, w]
            grid = grid.reshape(2, -1)  # (y, x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            # step
            last_pos = L + w * h
        # final text
        y_sections.append(np.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(np.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = np.concatenate(x_sections).astype(np.int32)
    y_pos = np.concatenate(y_sections).astype(np.int32)
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.from_numpy(
        np.stack((y_pos, x_pos), axis=1),
    ).unsqueeze(1).to(device)    # [seq_len, 1, 2]

    # calc rope
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def get_batch_text_image_2d_rope(
        image_infos: List[List[Tuple[slice, Tuple[int, int], dict]]], seq_len: int, n_elem: int,
        device: Optional[torch.device] = None, base: float = 10000.0, base_rescale_factor: float = 1.0, condense_ratio: int = 1,
        sample_offsets: Optional[List[torch.Tensor]] = None, return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = get_text_image_2d_rope(
            image_info, seq_len, n_elem, device=device,
            base=base, base_rescale_factor=base_rescale_factor, condense_ratio=condense_ratio,
            sample_offsets=sample_offsets[i] if sample_offsets is not None else None,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def get_text_media_3d_rope(
        media_infos: Optional[List[Tuple[slice, Tuple[int, int, int], dict]]],
        seq_len: int,
        n_elem: int,
        mrope_section: list[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        sample_offsets: Optional[torch.Tensor] = None,
        return_all_pos: bool = False,
        float_position: bool = False,
        no_space: bool = False,
        fixed_space: Optional[int] = None,
):
    """
    Reference: https://kexue.fm/archives/10352

    Start from 1, we have
        beta_z = L + (whd - d)/2
        beta_y = L + (whd - h)/2
        beta_x = L + (whd - w)/2

    When ``fixed_space`` is provided (and ``no_space`` is False), the placeholder length
    ``whd`` used for centering the media span is replaced by the fixed constant ``S = fixed_space``,
    so the media occupies ``S`` positions regardless of its (d, h, w) shape:
        beta_z = L + (S - d)/2
        beta_y = L + (S - h)/2
        beta_x = L + (S - w)/2
    Requires ``S >= max(d, h, w)`` to avoid negative offsets.

    NOTE: ``fixed_space`` is applied ONLY to video / audio-video (av) media. Media
    identified as image (``type == "gen_image"`` or, when ``type`` is missing, ``d == 1``)
    or standalone audio (``type == "gen_audio"`` without ``with_video=True``) will keep
    using the original ``whd`` placeholder logic even if ``fixed_space`` is provided.

    Returns
    -------
    cos: torch.Tensor with shape of [seq_len, n_elem]
    sin: torch.Tensor with shape of [seq_len, n_elem]
    """
    # sum(mrope_section) can be either the half of length (qwen, gemini), or the full length (leo).
    # Here we support both cases for compatibility.
    if sum(mrope_section) == n_elem:
        assert all(elem % 2 == 0 for elem in mrope_section), \
            (f"When sum(mrope_section) == n_elem, each element in mrope_section must be divisible by 2, "
             f"but got {mrope_section}.")
        mrope_section = [element // 2 for element in mrope_section]
    elif sum(mrope_section) == n_elem // 2:
        pass
    else:
        raise ValueError(f"sum(mrope_section) must be equal to n_elem or n_elem//2, "
                         f"but got {n_elem} and {mrope_section}.")

    # theta
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))    # (n_elem/2,)

    # position indices
    if media_infos is None:
        media_infos = []

    # when using sequence packing
    if sample_offsets is not None:
        sample_offsets = sample_offsets.tolist()
        media_infos_list = []
        last_id = -1
        for sli, (d, h, w), meta in media_infos:
            sample_id = bisect.bisect_right(sample_offsets[1:], sli.stop - 1)
            # Dummy info will be treated as part of the last sample.
            if meta.get("is_dummy", False) and sample_id != last_id:
                sample_id -= 1
                assert sample_id == last_id, "Dummy media info should belong to the last sample."
            if sample_id != last_id:
                media_infos_list.append([])
            offset = sample_offsets[sample_id]
            media_infos_list[-1].append((slice(sli.start - offset, sli.stop - offset), (d, h, w), meta))
            last_id = sample_id
        sample_seq_lens = [sample_offsets[i + 1] - sample_offsets[i] for i in range(len(sample_offsets) - 1)]
        # Add padding for the last sample
        sample_seq_lens[-1] += seq_len - sample_offsets[-1]
    else:
        media_infos_list = [media_infos]
        sample_seq_lens = [seq_len]

    # Prepare position indices for each sample
    x_sections = []
    y_sections = []
    z_sections = []
    for sample_id, sample_media_infos in enumerate(media_infos_list):
        last_pos = 0
        shift = 0
        # Store the video position information when calculating the position of audio in `video+audio` situation,
        # where audio tokens should have the same time ranges as video tokens by interpolating the position of video
        # tokens.
        video_position_ranges = []
        for sec_slice, (d, h, w), meta in sample_media_infos:
            if meta.get("is_dummy", False):
                continue
            L = sec_slice.start + meta.get("start_offset", 0)
            # previous text
            if last_pos < L:
                z_sections.append(torch.arange(last_pos - shift, L - shift))
                y_sections.append(torch.arange(last_pos - shift, L - shift))
                x_sections.append(torch.arange(last_pos - shift, L - shift))
            elif d is None:
                # Interleave data has overlapped positions for <boi> <size> <ratio> <timestep> <eoi> tokens.
                z_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                # Interleave data has overlapped positions for noised image and the successive clean image,
                # leading to last_pos (= last text end L + noise w * h) > L (last text end L).
                pass

            # `fixed_space` should only affect video / av (video+audio) media. For other media
            # (image / standalone audio) we fall back to the `whd` placeholder logic by disabling
            # `fixed_space` locally.
            media_type = meta.get("type", None)
            if media_type is not None:
                is_video_or_av = (
                    media_type == "gen_video"
                    or (media_type == "gen_audio" and meta.get("with_video", False))
                )
            else:
                # Backward compatibility: when `type` is not provided, treat any 3D media
                # (d > 1) with valid spatial dims as a video-like input.
                is_video_or_av = (d is not None and d > 1 and h is not None and w is not None)
            local_fixed_space = fixed_space if is_video_or_av else None

            # Process the audio in AV situation.
            if meta.get("with_video", False):
                assert meta["type"] == "gen_audio", \
                    f"Only support gen_audio type to have `with_video` field, but got type {meta['type']}."
                assert len(video_position_ranges) == 1, \
                    f"Missing video position ranges, please check the media_infos: {sample_media_infos}"
                assert h == 1 and w == 1, \
                    f"Audio media's h and w should be 1, but got h={h} and w={w} in media_infos: {sample_media_infos}"
                v_coord = video_position_ranges.pop()
                beta_z = v_coord["z0"]
                mid_y = (v_coord["y0"] + v_coord["y1"] - 1) / 2
                mid_x = (v_coord["x0"] + v_coord["x1"] - 1) / 2
                pos_args = (
                    (beta_z, mid_y, mid_x),                 # start
                    (v_coord["z1"] - 1, mid_y, mid_x),      # end
                    (d, 1, 1),                              # step, audio has the same time length as video, but only 1 token in spatial dimensions
                )
                with_end = True  # Enable `with_end` with length - 1 to align the end points of audio and video.

                if no_space or (local_fixed_space is not None):
                    span_length = 0
                else:
                    span_length = w * h * d
            else:
                # current media(image/video/audio)
                # When fixed_space is provided and no_space is False, override the placeholder length
                # used for centering with the fixed constant S; otherwise fall back to whd.
                # NOTE: `local_fixed_space` is None for non video/av media, so image / standalone
                # audio always use the whd logic here.
                S = local_fixed_space if (local_fixed_space is not None and not no_space) else (w * h * d)
                if not no_space:
                    assert S >= max(d, h, w), (
                        f"fixed_space={S} is too small to host media with (d, h, w)=({d}, {h}, {w})."
                    )
                    beta_y = L + (S - h) / 2
                    beta_x = L + (S - w) / 2
                else:
                    beta_y = L - shift
                    beta_x = L - shift
                if meta.get("rope_audio_rescale_factor", 1.0) != 1.0 and meta["type"] == "gen_audio":
                    rope_audio_rescale_factor = meta.get("rope_audio_rescale_factor", 1.0)
                    assert 0 < rope_audio_rescale_factor < 1.0, \
                        f"rope_audio_rescale_factor should be (0, 1.0), but got {rope_audio_rescale_factor}"

                    if not no_space:
                        half_delta = (d - 1) * (1.0 - rope_audio_rescale_factor) * 0.5
                        pos_args = (
                            (L + half_delta, beta_y, beta_x),  # start
                            (L + (d - 1) - half_delta, beta_y + h, beta_x + w),  # end
                            (d, 1, 1),
                        )
                    else:
                        pos_args = (
                            (L - shift, beta_y, beta_x),  # start
                            (L - shift + (d - 1) * rope_audio_rescale_factor, beta_y + h, beta_x + w),  # end
                            (d, 1, 1),
                        )
                    with_end = True

                    if no_space or (local_fixed_space is not None):
                        span_length = int(math.ceil((d - 1) * rope_audio_rescale_factor)) + 1
                    else:
                        span_length = w * h * d
                else:
                    if not no_space:
                        beta_z = L + (S - d) / 2
                    else:
                        beta_z = L - shift
                    pos_args = (
                        (beta_z, beta_y, beta_x),  # start
                        (beta_z + d, beta_y + h, beta_x + w),  # end
                    )
                    with_end = False

                    if no_space:
                        span_length = max(d, h, w)
                    elif local_fixed_space is not None:
                        span_length = local_fixed_space
                    else:
                        span_length = w * h * d

            grid = get_meshgrid_nd(
                *pos_args,
                dim=3,
                with_end=with_end,
            )   # [3, d, h, w]
            grid = grid.reshape(3, -1)  # (z, y, x)
            z_sections.append(grid[0])
            y_sections.append(grid[1])
            x_sections.append(grid[2])

            # Process the video in AV situation.
            if meta.get("with_audio", False):
                assert meta["type"] == "gen_video", \
                    f"Only support gen_video type to have `with_audio` field, but got type {meta['type']}."
                assert len(video_position_ranges) == 0, \
                    f"Already exists video position ranges, please check the media_infos: {sample_media_infos}"
                if meta.get("dataset_tag", "").startswith("fl2va"):
                    video_position_ranges.append(
                        dict(z0=beta_z, y0=beta_y, x0=beta_x, z1=beta_z + (d - 1), y1=beta_y + h, x1=beta_x + w)
                    )
                else:
                    video_position_ranges.append(
                        dict(z0=beta_z, y0=beta_y, x0=beta_x, z1=beta_z + d, y1=beta_y + h, x1=beta_x + w)
                    )

            # step
            last_pos = L + w * h * d
            shift += w * h * d - span_length
        # final text
        z_sections.append(torch.arange(last_pos - shift, sample_seq_lens[sample_id] - shift))
        y_sections.append(torch.arange(last_pos - shift, sample_seq_lens[sample_id] - shift))
        x_sections.append(torch.arange(last_pos - shift, sample_seq_lens[sample_id] - shift))

    x_pos = np.concatenate(x_sections).astype(np.float32 if float_position else np.int32)
    y_pos = np.concatenate(y_sections).astype(np.float32 if float_position else np.int32)
    z_pos = np.concatenate(z_sections).astype(np.float32 if float_position else np.int32)
    # If there are overlap positions, we need to remove them.
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    z_pos = z_pos[:seq_len]
    all_pos = torch.from_numpy(
        np.stack((z_pos, y_pos, x_pos), axis=0)
    ).to(device)    # [3, seq_len]

    inv_freq_expanded = theta[None, :, None].expand(3, -1, 1)   # (3, n_elem, 1)
    with torch.autocast(device_type=device.type, enabled=False):
        freqs = (inv_freq_expanded.float() @ all_pos.unsqueeze(1).float()).transpose(1, 2)   # (3, seq_len, n_elem)
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            elem_idx = slice(offset, length, 3)
            freqs_t[..., elem_idx] = freqs[dim, ..., elem_idx]
        emb = torch.cat((freqs_t, freqs_t), dim=-1)     # non-interleave
        cos = emb.cos()
        sin = emb.sin()

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def get_batch_text_media_3d_rope(
        media_infos: List[List[Tuple[slice, Tuple[int, int, int], dict]]],
        seq_len: int,
        n_elem: int,
        mrope_section: list[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        sample_offsets: Optional[List[torch.Tensor]] = None,
        return_all_pos: bool = False,
        float_position: bool = False,
        no_space: bool = False,
        fixed_space: Optional[int] = None,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if media_infos is None:
        media_infos = [None]
    for i, media_info in enumerate(media_infos):
        res = get_text_media_3d_rope(
            media_info, seq_len, n_elem, mrope_section, device=device,
            base=base, base_rescale_factor=base_rescale_factor,
            sample_offsets=sample_offsets[i] if sample_offsets is not None else None,
            return_all_pos=return_all_pos,
            float_position=float_position,
            no_space=no_space,
            fixed_space=fixed_space,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def get_vision_position_ids(
        start_position: int,
        grid_thw: List[int],
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute 3D positional indices for vision tokens derived from a single image or video input.

    Reference: official Qwen3.5-MoE get_vision_position_ids implementation.

    The positions are generated from the input grid defined by temporal (T), height (H), and
    width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
    merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

    Args:
        start_position: Offset added to all computed positional indices.
        grid_thw: The (T, H, W) grid representing the feature layout after patch embedding.
        temp_merge_size: Factor by which the temporal dimension is reduced. Defaults to 1.
        spatial_merge_size: Factor by which the spatial dimensions (H and W) are reduced. Defaults to 1.
        time_interval: Spacing factor applied between consecutive temporal position indices. Defaults to 1.
        device: Device on which the resulting tensor is allocated.

    Returns:
        torch.LongTensor of shape (3, sequence_length):
            Positional indices for temporal, height, and width dimensions,
            flattened into sequence form and offset by `start_position`.
    """
    t, h, w = grid_thw[0], grid_thw[1], grid_thw[2]
    if isinstance(t, torch.Tensor):
        t, h, w = t.item(), h.item(), w.item()
    llm_grid_t = t // temp_merge_size
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    position_temporal = position_temporal * time_interval
    vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)
    return vision_position_ids


def get_text_position_ids(
        length: int,
        start_position: int = 0,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate 3D position IDs for text tokens (all three dimensions share the same 1D positions).

    Reference: official Qwen3.5-MoE get_rope_index — text (modality_type == 0) branch.

    Args:
        length: Number of text tokens.
        start_position: Position offset for the first text token.
        device: Device on which the resulting tensor is allocated.

    Returns:
        torch.LongTensor of shape (3, length): Position IDs with identical T/H/W values.
    """
    return torch.arange(length, device=device).view(1, -1).expand(3, -1) + start_position


def get_interleaved_mrope_index(
        image_infos: List[Optional[List[Tuple[slice, Tuple[int, int], dict]]]],
        seq_len: int,
        spatial_merge_size: int,
        sample_offsets: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build 3D MRoPE position IDs for a batch of text-image sequences.

    Reference: official Qwen3.5-MoE get_rope_index implementation.

    For text tokens, all three dimensions (T, H, W) share the same 1D position.
    For vision tokens, T/H/W get independent 3D grid positions via get_vision_position_ids.
    After each vision segment, text positions resume from ``max(llm_grid_h, llm_grid_w)``.

    Args:
        image_infos: Per-sample list of (slice, (width, height), meta) tuples.
            ``None`` or ``[]`` means pure-text.
        seq_len: Total sequence length (tokens).
        spatial_merge_size: Spatial merge factor used by the vision encoder.
        sample_offsets: Sample offsets for the batch, for sequence packing.
            ``None`` means no sample offsets.
        device: Device on which the resulting tensor is allocated.

    Returns:
        torch.LongTensor of shape (3, batch_size, seq_len): Position IDs.
    """
    if image_infos is None:
        image_infos = [None]
    batch_size = len(image_infos)
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.int64, device=device)

    for i, image_info in enumerate(image_infos):
        llm_pos_ids_list = []
        current_pos = 0
        next_seq_idx = 0
        prev_slice_stop = 0

        if image_info is None:
            image_info = []

        offset_info = []
        num_image_token_prefix_suffix = 0
        for sec_slice, (h, w), _ in image_info:
            img_start = sec_slice.start
            text_len = img_start - prev_slice_stop

            # Text tokens before this image
            if text_len > 0:
                llm_pos_ids_list.append(get_text_position_ids(text_len, current_pos, device))
                current_pos += text_len
                next_seq_idx += text_len
            elif h is None:
                # 记录发生偏移的位置和偏移量
                if sec_slice.start < prev_slice_stop:
                    # 利用第一次发生偏移时，计算出num_image_token_prefix_suffix；也可以考虑外部传入
                    if num_image_token_prefix_suffix == 0:
                        llm_positions_tmp = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                        num_image_token_prefix_suffix = current_pos - llm_positions_tmp[0][sec_slice.start].item()
                        num_image_token_prefix_suffix = num_image_token_prefix_suffix - max(llm_grid_h, llm_grid_w)
                    # 每次发生偏移时，计算出offset_constant
                    offset_constant = num_image_token_prefix_suffix + max(llm_grid_h, llm_grid_w)
                    offset_info.append((next_seq_idx, offset_constant))

                text_len = sec_slice.stop - sec_slice.start
                llm_pos_ids_list.append(get_text_position_ids(text_len, current_pos, device))
                current_pos += text_len
                next_seq_idx += text_len
                prev_slice_stop = sec_slice.stop
                continue
            else:
                pass

            # Vision tokens
            grid_thw = [1, h, w]
            llm_pos_ids_list.append(
                get_vision_position_ids(current_pos, grid_thw, spatial_merge_size=spatial_merge_size, device=device)
            )
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            current_pos += max(llm_grid_h, llm_grid_w)
            next_seq_idx += llm_grid_h * llm_grid_w
            prev_slice_stop = img_start + llm_grid_h * llm_grid_w

        # Trailing text tokens
        if next_seq_idx < seq_len:
            llm_pos_ids_list.append(get_text_position_ids(seq_len - next_seq_idx, current_pos, device))

        # Pure-text fallback
        if len(llm_pos_ids_list) == 0:
            llm_pos_ids_list.append(get_text_position_ids(seq_len, 0, device))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

        # Pad or truncate to seq_len
        if llm_positions.shape[1] < seq_len:
            padding = llm_positions[:, -1:].expand(3, seq_len - llm_positions.shape[1])
            llm_positions = torch.cat([llm_positions, padding], dim=1)
        elif llm_positions.shape[1] > seq_len:
            llm_positions = llm_positions[:, :seq_len]

        position_ids[:, i, :] = llm_positions

        # 在每次发生offset时，对position_ids进行偏移， 通过减去offset_constant来实现
        for offset_index, offset in offset_info:
            position_ids[:, i, offset_index:] = position_ids[:, i, offset_index:] - offset

        # Handle packed-sequence offsets: re-base positions within each packed segment.
        # For each segment [start, end), subtract the segment's starting position so the segment starts from 0.
        if sample_offsets is not None and sample_offsets[i] is not None:
            offsets = sample_offsets[i].tolist()
            if len(offsets) >= 2:
                # Ensure boundaries are within [0, seq_len]
                assert offsets[0] == 0, "First offset must be 0"
                assert offsets[-1] <= seq_len, "Last offset must be less than or equal to seq_len"
                for start, end in zip(offsets[:-1], offsets[1:]):
                    assert end > start, "End must be greater than start"
                    seg_base = position_ids[:, i, start].clone()
                    position_ids[:, i, start:end] = position_ids[:, i, start:end] - seg_base.view(3, 1)


    return position_ids


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.

    Args:
        freqs: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,) e.g. [11, 11, 10]

    Returns:
        freqs_t: (bs, seq_len, head_dim // 2)
    """
    # 使用 clone 避免 in-place 修改原始 tensor
    freqs_t = freqs[0].clone()  # just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def get_batch_interleaved_mrope(
        image_infos: List[List[Tuple[slice, Tuple[int, int], dict]]],
        seq_len: int,
        n_elem: int,
        mrope_section: List[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        spatial_merge_size: int = 1,
        sample_offsets: Optional[torch.Tensor] = None,
        return_all_pos: bool = False,
) -> Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute batched interleaved MRoPE cos/sin embeddings.

    Args:
        image_infos: Per-sample list of (slice, (width, height), meta) tuples.
        seq_len: Total sequence length.
        n_elem: Rotary dimension (= head_dim * partial_rotary_factor).
        mrope_section: Per-dimension frequency counts, e.g. [11, 11, 10] for T/H/W.
        device: Target device.
        base: RoPE base frequency.
        base_rescale_factor: Base rescale factor (1.0 = no rescale).
        spatial_merge_size: Spatial merge factor used by the vision encoder, default to 1 for compatibility with official implementation.
        sample_offsets: Sample offsets for sequence packing, default to None for no packing.
        return_all_pos: Whether to return all position IDs.
    Returns:
        cos: (batch_size, seq_len, n_elem) cosine embeddings.
        sin: (batch_size, seq_len, n_elem) sine embeddings.
    """
    # Step 1: Get position IDs, shape: (3, bsz, seq_len)
    position_ids = get_interleaved_mrope_index(image_infos, seq_len, spatial_merge_size, sample_offsets, device)
    position_ids = position_ids.to(device)  # (3, bsz, seq_len)

    # Step 2: Compute inv_freq (theta) using logic from build_rope_cache
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))

    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))  # (n_elem // 2,)

    # Step 3: Expand dimensions for batch matrix multiplication
    inv_freq_expanded = theta[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)  # (3, bsz, n_elem // 2, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bsz, 1, seq_len)

    # Step 4: Compute frequencies and apply interleaved MRoPE
    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)  # (3, bsz, seq_len, n_elem // 2)
        freqs = apply_interleaved_mrope(freqs, mrope_section)  # (bsz, seq_len, n_elem // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (bsz, seq_len, n_elem)
        cos = emb.cos()
        sin = emb.sin()

    # Return all position IDs if requested
    if return_all_pos:
        return cos, sin, position_ids.transpose(0, 1)

    return cos, sin


def get_xdrope_input_positions(
        image_infos: List[List[Tuple[slice, Tuple[int, int]]]],
        seq_len: int,
        sample_offsets: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    batch_position_ids = []
    if image_infos is None:
        image_infos = [[]]

    for batch_idx, image_info in enumerate(image_infos):
        if image_info is None:
            image_info = []

        p_index = torch.arange(seq_len)
        if sample_offsets is not None and sample_offsets[batch_idx] is not None:
            offsets = sample_offsets[batch_idx].tolist()
            num_samples = len(offsets) - 1
            # Reset p_index per packed sample so each starts from 0
            for i in range(num_samples):
                start = offsets[i]
                end = offsets[i + 1] if i < num_samples - 1 else seq_len
                p_index[start:end] = torch.arange(end - start)
            # Assign per-sample image_index for t_index
            per_sample_counts = [0] * num_samples # 为每个sample初始化一个计数器，记录该sample有几张图片
            image_local_indices = [] # 记录每张图片在对应sample中的局部位置
            for sli, (h, w), meta in image_info:
                sample_id = bisect.bisect_right(offsets[1:], sli.stop - 1) # 找到该图片属于哪个sample
                image_local_indices.append(per_sample_counts[sample_id]) # 记录该图片在对应sample中的局部位置
                per_sample_counts[sample_id] += 1 # 该sample的图片数量加1
        else:
            image_local_indices = list(range(len(image_info)))

        w_index = p_index.clone()
        h_index = p_index.clone()
        t_index = p_index.clone()

        for local_img_idx, (media_slice, (tk_height, tk_width), meta) in zip(image_local_indices, image_info):
            pos = media_slice.start + meta.get("start_offset", 0)     # skip the image_begin embedding
            token_num = tk_width * tk_height
            w_index[pos: pos + token_num].copy_(
                torch.arange(0, tk_width)
                .reshape(1, -1)
                .expand(tk_height, -1)
                .reshape(-1)
            )
            h_index[pos: pos + token_num].copy_(
                torch.arange(0, tk_height)
                .reshape(-1, 1)
                .expand(-1, tk_width)
                .reshape(-1)
            )
            t_index[pos: pos + token_num] = local_img_idx

        position_ids = torch.stack([p_index, w_index, h_index, t_index])
        batch_position_ids.append(position_ids)

    batch_position_ids = torch.stack(batch_position_ids)
    return batch_position_ids


def get_batch_xdrope(
        image_infos: List[List[Tuple[slice, Tuple[int, int]]]],
        seq_len: int,
        n_elem: int,
        xdrope_section: List[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        sample_offsets: Optional[List[torch.Tensor]] = None,
        return_all_pos: bool = False,
):
    position_ids = get_xdrope_input_positions(image_infos, seq_len, sample_offsets)   # (bsz, 4, seq_len)
    cos, sin = build_rope_cache(seq_len, n_elem, device, base, base_rescale_factor)

    bsz = position_ids.size(0)
    x_dim = len(xdrope_section)
    cos = cos[position_ids, ...].permute(0, 2, 1, 3).reshape(bsz, seq_len, x_dim, -1).contiguous()
    sin = sin[position_ids, ...].permute(0, 2, 1, 3).reshape(bsz, seq_len, x_dim, -1).contiguous()

    xdrope_section = xdrope_section * 2

    # for xd concat
    assert sum(xdrope_section) == cos.shape[-1], "Illegal partition for xd rope"
    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))], dim=-1)
    if return_all_pos:
        return cos, sin, position_ids
    return cos, sin


def get_leo2_3d_rope(
        media_info: List[Tuple[slice, Tuple[int, int, int] | Tuple[int], dict]],
        seq_len: int,
        n_elem: int,
        mrope_section: list[int],
        device=None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        use_scale_rope: bool = False,
        return_all_pos: bool = False,
):
    """ A simple 3d rope with following positions:
    Assume a video with shape (2, 2, 3) and 5 text tokens, where the text tokens are positioned in the
    diagonal after the maximum video position across all dimensions.
    If including audio, we assume there are 3 audio tokens.

    If use_scale_rope is False, we have
    z:  0  0  0  0  0  0  1  1  1  1  1  1  3  4  5  6  7
    y:  0  0  0  1  1  1  0  0  0  1  1  1  3  4  5  6  7
    x:  0  1  2  0  1  2  0  1  2  0  1  2  3  4  5  6  7

    If use_scale_rope is True, we have
    z:  0  0  0  0  0  0  1  1  1  1  1  1  2  3  4  5  6
    y: -1 -1 -1  0  0  0 -1 -1 -1  0  0  0  2  3  4  5  6
    x: -2 -1  0 -2 -1  0 -2 -1  0 -2 -1  0  2  3  4  5  6

    If use_scale_rope is False and has audio info, we have:
    z:  0  0  0  0  0  0  1  1  1  1  1  1  0 0.5 1  3  4  5  6  7
    y:  0  0  0  1  1  1  0  0  0  1  1  1  0  0  0  3  4  5  6  7
    x:  0  1  2  0  1  2  0  1  2  0  1  2  0  0  0  3  4  5  6  7
       +++++++++++++++++++++++++++++++++++  =======  -------------
                      video                  audio       text

    If use_scale_rope is True and has audio info, we have:
    z:  0  0  0  0  0  0  1  1  1  1  1  1  0 0.5 1  2  3  4  5  6
    y: -1 -1 -1  0  0  0 -1 -1 -1  0  0  0  0  0  0  2  3  4  5  6
    x: -2 -1  0 -2 -1  0 -2 -1  0 -2 -1  0  0  0  0  2  3  4  5  6
       +++++++++++++++++++++++++++++++++++  =======  -------------
                      video                  audio       text

    """
    assert sum(mrope_section) == n_elem, \
        f"n_elem({n_elem}) must be equal to sum(mrope_section) ({mrope_section})."
    assert len(mrope_section) == 3, f"mrope_section must have 3 dimensions for 3d rope, but got {len(mrope_section)}."
    assert len(media_info) in [1, 2], f"Only one or two medias is supported in leo2_3d_rope for now, but got {len(media_info)}."
    # TODO: Now only support single media_info. Positions of extra media conditions should be further designed.

    grid_list = []
    media_length = 0

    # Video positions
    dims = len(mrope_section)
    if (
            len(media_info) > 0
            and not media_info[0][2].get("is_dummy", False)
            and media_info[0][2]["type"] in ["gen_video", "gen_image"]
    ):
        _, (d, h, w), meta = media_info[0]
        if use_scale_rope:
            # A bug: -h//2 will cause the center shifting 1 to the negative half axis.
            # For example, if h=5, the desired positions should be [-2, -1, 0, 1, 2], but -h//2 will give -3,
            # leading to positions [-3, -2, -1, 0, 1]. We keep this for backward compatibility.
            video_start = (0, -h // 2, -w // 2)
            video_stop = (d, h // 2, w // 2)
            video_grid = get_meshgrid_nd(video_start, video_stop, dim=dims).flatten(1)
        else:
            video_grid = get_meshgrid_nd((d, h, w), dim=dims).flatten(1)
            video_start = (0, 0, 0)
            video_stop = (d, h, w)
        grid_list.append(video_grid)
        media_length += d * h * w

        # Shift
        media_info = media_info[1:]
    else:
        video_start, video_stop = None, None

    # Audio positions (if has audio info in meta)
    if len(media_info) > 0 and not media_info[0][2].get("is_dummy", False) and media_info[0][2]["type"] == "gen_audio":
        _, audio_shape, _ = media_info[0]
        if len(audio_shape) == 1:
            al = audio_shape[0]
        else:
            al, ah, aw = audio_shape
            assert ah == aw == 1, \
                f"Audio media's h and w should be 1, but got h={ah} and w={aw} in media_infos: {media_info}"
        # use_scale_rope is spatial-related, has no effect on audio temporal position.
        if video_start is not None:
            audio_start = (video_start[0], 0, 0)
            audio_stop = (video_stop[0] - 1, 0, 0)
        else:
            audio_start = (0, 0, 0)
            audio_stop = (al - 1, 0, 0)
            d, h, w = al, 1, 1
        audio_num = (al, 1, 1)
        audio_grid = get_meshgrid_nd(audio_start, audio_stop, audio_num, dim=dims, with_end=True).flatten(1)
        grid_list.append(audio_grid)
        media_length += al

        # Shift
        media_info = media_info[1:]

    assert len(media_info) <= 1, f"Wrong number of media_info: {media_info}"
    if len(media_info) == 1:
        assert media_info[0][2]["is_dummy"], f"Wrong media_info: {media_info}"

    # Text positions
    if use_scale_rope:
        txt_start = max(d, h // 2, w // 2)
    else:
        txt_start = max(d, h, w)
    txt_length = seq_len - media_length
    txt_stop = txt_start + txt_length
    txt_grid = get_meshgrid_nd(txt_start, txt_stop, dim=1)
    txt_grid = torch.cat([txt_grid, txt_grid, txt_grid], dim=0)
    grid_list.append(txt_grid)

    all_pos = torch.cat(grid_list, dim=1)     # [3, seq_len]

    cos_by_dim, sin_by_dim = [], []
    for i in range(dims):
        cos, sin = get_1d_rotary_pos_embed(
            dim=mrope_section[i],
            pos=all_pos[i], # noqa
            theta=base,
            theta_rescale_factor=base_rescale_factor,
            interleave=True,
        )
        cos_by_dim.append(cos)
        sin_by_dim.append(sin)

    cos = torch.cat(cos_by_dim, dim=1).to(device)
    sin = torch.cat(sin_by_dim, dim=1).to(device)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def get_batch_leo2_3d_rope(
        media_infos: List[List[Tuple[slice, Tuple[int, int, int], dict]]],
        seq_len: int,
        n_elem: int,
        mrope_section: list[int],
        device: Optional[torch.device] = None,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
        use_scale_rope: bool = False,
        return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if media_infos is None:
        media_infos = [None]
    for i, media_info in enumerate(media_infos):
        res = get_leo2_3d_rope(
            media_info, seq_len, n_elem, mrope_section, device=device,
            base=base, base_rescale_factor=base_rescale_factor, use_scale_rope=use_scale_rope,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleave=False,
        apply_rope_in_fp32=False,
        unsqueeze_dim=-3,
        cast_output_to_input_dtype=False,
) -> torch.Tensor:
    input_dtype = x.dtype
    if apply_rope_in_fp32:
        x = x.float()
    if interleave:
        x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack((-x2, x1), dim=-1).flatten(x.ndim - 1)  # (B, nh, T, hs)
    else:
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
        x2 = x[..., head_size // 2:]  # (B, nh, T, hs/2)
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqueeze for nh, -3 for x.shape (B, nh, T, hs) , -2 for (B, T, nh, hs)
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    roped = (x * cos) + (rotated * sin)

    if cast_output_to_input_dtype:
        return roped.to(dtype=input_dtype)
    return roped


def apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleave=False,
    apply_rope_in_fp32=False,
    unsqueeze_dim=-3,
    cast_output_to_input_dtype=False,
):
    return apply_rope(q, cos, sin, interleave, apply_rope_in_fp32, unsqueeze_dim, cast_output_to_input_dtype), apply_rope(k, cos, sin, interleave, apply_rope_in_fp32, unsqueeze_dim, cast_output_to_input_dtype)


def liger_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleave=False,
    apply_rope_in_fp32=False,
    unsqueeze_dim=-3,
    cast_output_to_input_dtype=False,
) -> torch.Tensor:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    input_dtype = x.dtype
    if apply_rope_in_fp32:
        x = x.float()
    use_liger = not interleave
    if not use_liger:
        return apply_rope(
            x, cos, sin,
            interleave=interleave,
            apply_rope_in_fp32=apply_rope_in_fp32,
            unsqueeze_dim=unsqueeze_dim,
            cast_output_to_input_dtype=cast_output_to_input_dtype,
        )
    assert x.ndim == 4
    assert cos.shape == sin.shape
    assert x.size(-1) == cos.size(-1)
    assert x.size(-1) % 2 == 0
    assert unsqueeze_dim in (-3, 1, -2, 2)
    q_out, _ = liger_rotary_pos_emb(x, x.clone(), cos, sin, unsqueeze_dim=unsqueeze_dim)
    if cast_output_to_input_dtype:
        return q_out.to(dtype=input_dtype)
    return q_out



def liger_apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleave=False,
    apply_rope_in_fp32=False,
    unsqueeze_dim=-3,
    cast_output_to_input_dtype=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_q_dtype = q.dtype
    input_k_dtype = k.dtype
    if apply_rope_in_fp32:
        q = q.float()
        k = k.float()
    rope_kwargs = dict(
        interleave=interleave,
        apply_rope_in_fp32=apply_rope_in_fp32,
        unsqueeze_dim=unsqueeze_dim,
        cast_output_to_input_dtype=cast_output_to_input_dtype,
    )
    use_liger = not interleave
    if not use_liger:
        return apply_rope_qk(q, k, cos, sin, **rope_kwargs)
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    assert q.ndim == 4
    assert k.ndim == 4
    assert cos.shape == sin.shape
    assert q.size(-1) == k.size(-1) == cos.size(-1)
    assert q.size(-1) % 2 == 0
    assert unsqueeze_dim in (-3, 1, -2, 2)
    q_out, k_out = liger_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    if cast_output_to_input_dtype:
        return q_out.to(dtype=input_q_dtype), k_out.to(dtype=input_k_dtype)
    return q_out, k_out