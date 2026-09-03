"""MagiAttention backend.

Provides:
- magi_scaled_dot_product_attention: single-GPU FFA attention, BHSD layout
- magi_dist_attention: distributed attention with context parallelism, BHSD layout
- dense_mask_to_magi_ranges: dense bool mask → magi ranges (analogous to dense_binary_mask_to_block_mask)
- Mask conversion tools: causal / text-image / interleave slices → magi ranges tuple
"""

import importlib.metadata
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from packaging import version

try:
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
except ImportError:
    class AttnMaskType:
        FULL = 0
        CAUSAL = 1

    class AttnRanges:
        def __init__(self, ranges_list):
            self._ranges = ranges_list

        @staticmethod
        def from_ranges(ranges_list):
            return AttnRanges(ranges_list)

        def to_tensor(self):
            if not self._ranges:
                return torch.zeros(0, 2, dtype=torch.int32)
            return torch.tensor(self._ranges, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Single-GPU FFA Wrapper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _uses_legacy_ffa_api() -> bool:
    """Whether the installed MagiAttention requires max sequence lengths."""
    installed_version = version.parse(
        importlib.metadata.version("magi_attention")
    )
    return installed_version < version.parse("1.0.5")


def magi_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: tuple,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Single-GPU MagiAttention (FFA) wrapper.

    Args:
        query:     (B, num_heads_q,  S_q, D)
        key:       (B, num_heads_kv, S_k, D)
        value:     (B, num_heads_kv, S_k, D)
        attn_mask: (q_ranges, k_ranges, attn_type_map) int32 CUDA tensors.
                   Non-square (S_q != S_k) supported — FFA handles CAUSAL
                   with right-aligned positions internally.
        scale:     softmax scale, defaults to 1/sqrt(D)

    Returns:
        (B, num_heads_q, S_q, D)
    """
    from magi_attention.functional import flex_flash_attn_func

    q_ranges, k_ranges, attn_type_map = attn_mask
    b, h_q, s_q, d = query.shape
    _, h_kv, s_k, _ = key.shape

    q_flat = query.transpose(1, 2).reshape(-1, h_q, d).contiguous()
    k_flat = key.transpose(1, 2).reshape(-1, h_kv, d).contiguous()
    v_flat = value.transpose(1, 2).reshape(-1, h_kv, d).contiguous()

    ffa_kwargs = {
        "q_ranges": q_ranges,
        "k_ranges": k_ranges,
        "attn_type_map": attn_type_map,
        "softmax_scale": scale,
    }

    # MagiAttention 1.0.3 requires these arguments. Starting from 1.0.5, FFA
    # derives the values from the ranges and no longer accepts them.
    if _uses_legacy_ffa_api():
        ffa_kwargs["max_seqlen_q"] = b * s_q
        ffa_kwargs["max_seqlen_k"] = b * s_k

    output, _ = flex_flash_attn_func(q_flat, k_flat, v_flat, **ffa_kwargs)

    return output.reshape(b, s_q, h_q, d).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Distributed Attention Wrapper
# ---------------------------------------------------------------------------

_INT_TO_MASK_TYPE = {0: AttnMaskType.FULL, 1: AttnMaskType.CAUSAL}


def magi_dist_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: tuple,
    cp_group,
    scale: Optional[float] = None,
    dist_attn_config=None,
) -> torch.Tensor:
    """Distributed MagiAttention wrapper with context parallelism.

    Flow: magi_attn_flex_key → dispatch → calc_attn → undispatch.

    Args:
        query:            (B, num_heads_q,  S, D) — global tensor, same on all ranks
        key:              (B, num_heads_kv, S, D)
        value:            (B, num_heads_kv, S, D)
        attn_mask:        (q_ranges, k_ranges, attn_type_map) — GLOBAL ranges
        cp_group:         dist.ProcessGroup for context parallelism
        scale:            softmax scale, defaults to 1/sqrt(D)
        dist_attn_config: DistAttnConfig, defaults to DistAttnConfig()

    Returns:
        (B, num_heads_q, S, D) — global output
    """
    from magi_attention.api import (
        calc_attn,
        dispatch as magi_dispatch,
        magi_attn_flex_key,
        undispatch as magi_undispatch,
    )
    from magi_attention.config import DistAttnConfig

    q_ranges_tensor, k_ranges_tensor, attn_type_map_tensor = attn_mask
    b, h_q, s, d = query.shape
    _, h_kv, _, _ = key.shape
    total_seqlen = b * s

    # Tensor ranges → AttnRanges objects
    q_attn_ranges = AttnRanges.from_ranges(q_ranges_tensor.cpu().tolist())
    k_attn_ranges = AttnRanges.from_ranges(k_ranges_tensor.cpu().tolist())
    mask_types = [_INT_TO_MASK_TYPE[int(v)] for v in attn_type_map_tensor.cpu()]

    if dist_attn_config is None:
        dist_attn_config = DistAttnConfig()

    import torch.distributed as dist
    cp_size = dist.get_world_size(cp_group)
    runtime_key = magi_attn_flex_key(
        q_ranges=q_attn_ranges,
        k_ranges=k_attn_ranges,
        attn_mask_type=mask_types,
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=dist_attn_config,
        chunk_size=total_seqlen // cp_size,
    )

    # (B, H, S, D) → (B*S, H, D)
    q_flat = query.transpose(1, 2).reshape(-1, h_q, d).contiguous()
    k_flat = key.transpose(1, 2).reshape(-1, h_kv, d).contiguous()
    v_flat = value.transpose(1, 2).reshape(-1, h_kv, d).contiguous()

    local_q = magi_dispatch(q_flat, runtime_key)
    local_k = magi_dispatch(k_flat, runtime_key)
    local_v = magi_dispatch(v_flat, runtime_key)

    local_out, _ = calc_attn(local_q, local_k, local_v, runtime_key)

    global_out = magi_undispatch(local_out, runtime_key)

    # (B*S, H_q, D) → (B, H_q, S, D)
    return global_out.reshape(b, s, h_q, d).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Dense Mask → Magi Ranges Conversion
# ---------------------------------------------------------------------------


def dense_mask_to_magi_ranges(dense_mask: torch.Tensor, device="cuda"):
    """Convert a dense bool attention mask to magi ranges tuple.

    Directly analyzes each row's True segments and groups consecutive rows
    with compatible patterns into (q_range, k_range, type) blocks.
    No intermediate slice reconstruction — analogous to dense_binary_mask_to_block_mask
    for BlockMask.

    Supports text-image, interleave (with holes), packing, and batching (B > 1).
    Non-square (S_q != S_k) supported — FFA handles CAUSAL with right-aligned
    positions internally, so q_ranges use relative indices [0, S_q).

    Args:
        dense_mask: bool tensor — (S_q, S_k), (1, S_q, S_k), (1, 1, S_q, S_k),
                    or (B, 1, S_q, S_k).
        device: output tensor device (default "cuda").

    Returns:
        (q_ranges, k_ranges, attn_type_map) — int32 tensors on device.
    """

    mask = dense_mask.bool().to(device=device)
    if mask.ndim == 4:
        B_mask, _, S_q, S_k = mask.shape
        mask = mask[:, 0]  # (B, S_q, S_k)
    elif mask.ndim == 3:
        B_mask, S_q, S_k = mask.shape
    elif mask.ndim == 2:
        B_mask = 1
        S_q, S_k = mask.shape
        mask = mask.unsqueeze(0)  # (1, S_q, S_k)
    else:
        raise ValueError(f"Expected 2-4D mask, got {mask.ndim}D")

    all_q, all_k, all_m = [], [], []

    for bi in range(B_mask):
        qr, kr, mt = _dense_mask_to_ranges_single(mask[bi], S_q, S_k, device)
        if bi > 0:
            qr = qr + bi * S_q
            kr = kr + bi * S_k
        all_q.append(qr)
        all_k.append(kr)
        all_m.append(mt)

    q_ranges = torch.cat(all_q, dim=0) if B_mask > 1 else all_q[0]
    k_ranges = torch.cat(all_k, dim=0) if B_mask > 1 else all_k[0]
    attn_type_map = torch.cat(all_m, dim=0) if B_mask > 1 else all_m[0]

    return (q_ranges, k_ranges, attn_type_map)


def _dense_mask_to_ranges_single(mask_2d, S_q, S_k, device):
    """Convert a single 2D mask to ranges. Internal helper for dense_mask_to_magi_ranges."""

    CHUNK = min(2048, S_q)
    k_start_gpu = torch.zeros(S_q, dtype=torch.int64, device=mask_2d.device)
    k_end_gpu = torch.zeros(S_q, dtype=torch.int64, device=mask_2d.device)
    has_seg = torch.zeros(S_q, dtype=torch.bool, device=mask_2d.device)
    n_true = torch.zeros(S_q, dtype=torch.int64, device=mask_2d.device)
    for ci in range(0, S_q, CHUNK):
        cj = min(ci + CHUNK, S_q)
        chunk = mask_2d[ci:cj]
        has_seg[ci:cj] = chunk.any(dim=1)
        cb = chunk.byte()
        k_start_gpu[ci:cj] = cb.argmax(dim=1)
        k_end_gpu[ci:cj] = S_k - cb.flip(1).argmax(dim=1)
        n_true[ci:cj] = chunk.sum(dim=1)
    k_start_gpu[~has_seg] = 0
    k_end_gpu[~has_seg] = 0
    has_gap = (((k_end_gpu - k_start_gpu) != n_true) & has_seg).any().item()

    ks = k_start_gpu.cpu().numpy().astype(np.int64)
    ke = k_end_gpu.cpu().numpy().astype(np.int64)
    hs = has_seg.cpu().numpy()

    if not has_gap:
        return _dense_to_ranges_fast(ks, ke, hs, S_q, device)

    nt = n_true.cpu().numpy().astype(np.int64)
    gap_rows = np.where(((ke - ks) != nt) & hs)[0]
    rep = int(gap_rows[np.argmax(ke[gap_rows] - ks[gap_rows])])
    rep_row = mask_2d[rep].cpu().numpy().astype(np.int8)
    transitions = np.diff(rep_row[ks[rep]:ke[rep]])
    gap_starts = np.where(transitions == -1)[0] + 1 + ks[rep]
    gap_ends = np.where(transitions == 1)[0] + 1 + ks[rep]
    G = len(gap_starts)
    return _dense_to_ranges_with_gaps(ks, ke, hs, gap_starts, gap_ends, G, S_q, device)


def _dense_to_ranges_fast(k_start, k_end, has_seg, S, device):
    """Fast path for single-segment rows (text-image, packing). Fully numpy-vectorized."""

    is_boundary = np.zeros(S, dtype=bool)
    is_boundary[0] = True
    is_boundary[1:] |= has_seg[:-1] != has_seg[1:]
    ks = k_start.copy()
    ks[~has_seg] = -1
    is_boundary[1:] |= np.diff(ks) != 0
    ke_delta = np.diff(k_end)
    valid_delta = ((ke_delta == 0) | (ke_delta == 1)) & has_seg[:-1] & has_seg[1:]
    is_boundary[1:] |= ~valid_delta & has_seg[:-1] & has_seg[1:]
    if S > 2:
        both_valid = valid_delta[:-1] & valid_delta[1:]
        is_boundary[2:] |= (np.diff(ke_delta) != 0) & both_valid

    boundary_idx = np.where(is_boundary)[0]
    q_list, k_list, m_list = [], [], []
    for g in range(len(boundary_idx)):
        gs = int(boundary_idx[g])
        ge = int(boundary_idx[g + 1]) if g + 1 < len(boundary_idx) else S
        if not has_seg[gs]:
            continue
        q_list.append([gs, ge])
        if k_end[ge - 1] == k_end[gs]:
            k_list.append([int(k_start[gs]), int(k_end[gs])])
            m_list.append(AttnMaskType.FULL)
        else:
            k_list.append([int(k_start[gs]), int(k_end[ge - 1])])
            m_list.append(AttnMaskType.CAUSAL)

    return _to_device_tuple(q_list, k_list, m_list, device=device)


def _dense_to_ranges_with_gaps(ks, ke, hs, gap_starts, gap_ends, G, S, device):
    """Interleave path: construct segments from k_start/k_end + gap positions. Numpy-vectorized."""

    all_rows = np.arange(S)

    # Which gaps are active per row: active[i, g] = row i is past gap g AND gap g is within row's range
    active = (all_rows[:, None] >= gap_ends[None, :]) & \
             (ks[:, None] < gap_starts[None, :]) & \
             (ke[:, None] > gap_ends[None, :]) & \
             hs[:, None]
    n_active = active.sum(axis=1)
    n_segs = np.where(hs, n_active + 1, 0)
    max_segs = int(n_segs.max())

    # Build (S, max_segs) segment boundary matrices.
    starts_mat = np.full((S, max_segs), -1, dtype=np.int64)
    ends_mat = np.full((S, max_segs), -1, dtype=np.int64)

    # Segment 0 always starts at ks
    starts_mat[hs, 0] = ks[hs]

    # Each active gap g ends segment k and starts segment k+1
    cum_active = np.cumsum(active, axis=1)
    for g in range(G):
        rows_g = np.where(active[:, g])[0]
        seg_k = cum_active[rows_g, g] - 1
        ends_mat[rows_g, seg_k] = gap_starts[g]
        starts_mat[rows_g, seg_k + 1] = gap_ends[g]

    # Last segment ends at ke
    rows_with_seg = np.where(hs)[0]
    last_seg_idx = n_segs[rows_with_seg] - 1
    ends_mat[rows_with_seg, last_seg_idx] = ke[rows_with_seg]

    # Boundary detection (same as fast path, per segment column).
    is_boundary = np.zeros(S, dtype=bool)
    is_boundary[0] = True
    is_boundary[1:] |= hs[:-1] != hs[1:]
    is_boundary[1:] |= np.diff(n_segs) != 0

    for s in range(max_segs):
        act = n_segs > s
        both = act[:-1] & act[1:]
        is_boundary[1:] |= (np.diff(starts_mat[:, s]) != 0) & both
        e_delta = np.diff(ends_mat[:, s])
        valid = ((e_delta == 0) | (e_delta == 1)) & both
        is_boundary[1:] |= ~valid & both
        if S > 2:
            bv = valid[:-1] & valid[1:]
            is_boundary[2:] |= (np.diff(e_delta) != 0) & bv

    boundary_idx = np.where(is_boundary)[0]
    q_list, k_list, m_list = [], [], []
    for g_idx in range(len(boundary_idx)):
        gs = int(boundary_idx[g_idx])
        ge = int(boundary_idx[g_idx + 1]) if g_idx + 1 < len(boundary_idx) else S
        ns = int(n_segs[gs])
        if ns == 0:
            continue
        for s in range(ns):
            q_list.append([gs, ge])
            if ends_mat[ge - 1, s] == ends_mat[gs, s]:
                k_list.append([int(starts_mat[gs, s]), int(ends_mat[gs, s])])
                m_list.append(AttnMaskType.FULL)
            else:
                k_list.append([int(starts_mat[gs, s]), int(ends_mat[ge - 1, s])])
                m_list.append(AttnMaskType.CAUSAL)

    return _to_device_tuple(q_list, k_list, m_list, device=device)


def dense_mask_to_slices(dense_mask: torch.Tensor):
    """Extract image_slices, hole_slices, and packing offsets from a dense bool mask.

    Utility for cross-validating dense_mask_to_magi_ranges. Recovers the semantic
    structure (image regions, holes, sample boundaries) that was used to construct
    the dense mask.

    Returns:
        (image_slices, hole_slices, offsets_tensor_or_None)
    """
    mask = dense_mask.squeeze().bool().cpu()
    assert mask.ndim == 2 and mask.shape[0] == mask.shape[1]
    S = mask.shape[0]

    offsets = [0]
    for i in range(1, S):
        if not mask[i, offsets[-1]]:
            offsets.append(i)
    offsets.append(S)

    all_image_slices = []
    hole_slices = []
    for si in range(len(offsets) - 1):
        s_start, s_end = offsets[si], offsets[si + 1]
        in_image = False
        img_start = s_start
        for i in range(s_start, s_end - 1):
            bidirectional = mask[i, i + 1].item() and mask[i + 1, i].item()
            if bidirectional and not in_image:
                img_start = i
                in_image = True
            elif not bidirectional and in_image:
                img_end = i + 1
                sli = slice(img_start, img_end)
                all_image_slices.append(sli)
                if img_end < s_end and not mask[img_end, img_start].item():
                    hole_slices.append(sli)
                in_image = False
        if in_image:
            all_image_slices.append(slice(img_start, s_end))

    offsets_tensor = torch.tensor(offsets, dtype=torch.int32) if len(offsets) > 2 else None
    return all_image_slices, hole_slices, offsets_tensor


# ---------------------------------------------------------------------------
# Mask Conversion Tools (from slices)
# ---------------------------------------------------------------------------
# Adapted from AngelPTM: angelptm/megatron/core/models/gemini/magi_attn_utils.py
# Converts hymm mask semantics to MagiAttention (q_ranges, k_ranges, attn_type_map).


def _to_device_tuple(q_list, k_list, m_list, device="cuda"):
    """Convert range lists to (q_ranges, k_ranges, attn_type_map) int32 tensors on device."""
    q_ranges = AttnRanges.from_ranges(q_list)
    k_ranges = AttnRanges.from_ranges(k_list)
    m_tensor = torch.tensor(
        [int(mt == AttnMaskType.CAUSAL) for mt in m_list], dtype=torch.int32,
    )
    return (
        q_ranges.to_tensor().to(device),
        k_ranges.to_tensor().to(device),
        m_tensor.to(device),
    )


def causal_to_magi_ranges(total_seqlen, offsets=None, device="cuda"):
    """Pure causal mask → magi ranges.

    Without packing: one CAUSAL range over the full sequence.
    With packing (offsets = tensor([0, s1, s2, ...])): one CAUSAL range per sample.
    """
    return image_slices_to_magi_ranges([], total_seqlen, offsets, device=device)


def image_slices_to_magi_ranges(
    image_slices: list,
    total_seqlen: int,
    offsets=None,
    device="cuda",
):
    """Text-image mask → magi ranges.

    Mask semantics: text regions use causal attention, image regions use full
    attention (bidirectional within the image + sees all preceding tokens).

    Args:
        image_slices: image token intervals, e.g. [slice(52, 308), slice(568, 824)]
        total_seqlen: total sequence length
        offsets:      packing boundaries, e.g. tensor([0, 300, 600])

    Returns:
        (q_ranges, k_ranges, attn_type_map) — int32 CUDA tensors
    """
    if offsets is None:
        q_list, k_list, m_list = _single_sample_image_slices_to_ranges(
            image_slices, 0, total_seqlen,
        )
    else:
        q_list, k_list, m_list = [], [], []
        for i in range(len(offsets) - 1):
            s_start = int(offsets[i])
            s_end = int(offsets[i + 1])
            sample_slices = [
                s for s in image_slices if s.start >= s_start and s.stop <= s_end
            ]
            q, k, m = _single_sample_image_slices_to_ranges(
                sample_slices, s_start, s_end,
            )
            q_list.extend(q)
            k_list.extend(k)
            m_list.extend(m)
        if offsets[-1] < total_seqlen:
            q_list.append([offsets[-1], total_seqlen])
            k_list.append([offsets[-1], total_seqlen])
            m_list.append(AttnMaskType.CAUSAL)

    return _to_device_tuple(q_list, k_list, m_list, device=device)


def interleave_slices_to_magi_ranges(
    gen_image_slices: list,
    cond_image_slices: list,
    hole_slices: list,
    total_seqlen: int,
    offsets=None,
    device="cuda",
):
    """Interleave mask (with holes) → magi ranges.

    Mask semantics: (causal | same_image) & (~hole)
      - same_image: gen + cond image tokens do full attention + see all prior tokens
      - hole: non-last gen images are invisible to subsequent tokens
      - hole tokens internally still do full attention within their image

    Without holes, degrades to image_slices_to_magi_ranges.

    Returns:
        (q_ranges, k_ranges, attn_type_map) — int32 CUDA tensors
    """
    all_image_slices = gen_image_slices + cond_image_slices

    if not hole_slices:
        return image_slices_to_magi_ranges(all_image_slices, total_seqlen, offsets, device=device)

    if offsets is None:
        q_list, k_list, m_list = _single_sample_interleave_to_ranges(
            all_image_slices, hole_slices, 0, total_seqlen,
        )
    else:
        q_list, k_list, m_list = [], [], []
        for i in range(len(offsets) - 1):
            s_start = int(offsets[i])
            s_end = int(offsets[i + 1])
            sample_imgs = [
                s for s in all_image_slices if s.start >= s_start and s.stop <= s_end
            ]
            sample_holes = [
                h for h in hole_slices if h.start >= s_start and h.stop <= s_end
            ]
            q, k, m = _single_sample_interleave_to_ranges(
                sample_imgs, sample_holes, s_start, s_end,
            )
            q_list.extend(q)
            k_list.extend(k)
            m_list.extend(m)
        if int(offsets[-1]) < total_seqlen:
            q_list.append([int(offsets[-1]), total_seqlen])
            k_list.append([int(offsets[-1]), total_seqlen])
            m_list.append(AttnMaskType.CAUSAL)

    return _to_device_tuple(q_list, k_list, m_list, device=device)


# ---------------------------------------------------------------------------
# Internal helpers for mask conversion
# ---------------------------------------------------------------------------


def _single_sample_image_slices_to_ranges(image_slices, sample_start, sample_end):
    """Single sample text-image mask → ranges.

    Splits into alternating text (CAUSAL) and image (FULL) regions:
    - text:  q=[prev_end, img_start), k=[sample_start, img_start)  → CAUSAL
    - image: q=[img_start, img_end),  k=[sample_start, img_end)    → FULL
    """
    q_range_list = []
    k_range_list = []
    mask_type_list = []

    sorted_slices = sorted(image_slices, key=lambda s: s.start)
    prev_end = sample_start

    for sli in sorted_slices:
        img_start, img_end = sli.start, sli.stop

        if img_start > prev_end:
            q_range_list.append([prev_end, img_start])
            k_range_list.append([sample_start, img_start])
            mask_type_list.append(AttnMaskType.CAUSAL)

        q_range_list.append([img_start, img_end])
        k_range_list.append([sample_start, img_end])
        mask_type_list.append(AttnMaskType.FULL)

        prev_end = img_end

    if prev_end < sample_end:
        q_range_list.append([prev_end, sample_end])
        k_range_list.append([sample_start, sample_end])
        mask_type_list.append(AttnMaskType.CAUSAL)

    return q_range_list, k_range_list, mask_type_list


def _single_sample_interleave_to_ranges(
    image_slices, hole_slices, sample_start, sample_end,
):
    """Single sample interleave mask → ranges.

    Algorithm:
      1. Collect all cut points (sample boundaries, image boundaries, hole boundaries).
      2. For each q segment [q_s, q_e):
         a) base type: q_s inside image → FULL, else CAUSAL
         b) base k_range: FULL → [sample_start, image_end), CAUSAL → [sample_start, q_e)
         c) active holes: holes closed before q_s and not containing q_s
         d) subtract active holes from k_range → emit one range per resulting segment
    """
    cuts = {sample_start, sample_end}
    for sli in image_slices:
        cuts.add(sli.start)
        cuts.add(sli.stop)
    for h in hole_slices:
        cuts.add(h.start)
        cuts.add(h.stop)
    cuts = sorted(c for c in cuts if sample_start <= c <= sample_end)

    sorted_image_slices = sorted(image_slices, key=lambda s: s.start)
    sorted_hole_slices = sorted(hole_slices, key=lambda s: s.start)

    q_list, k_list, m_list = [], [], []

    for i in range(len(cuts) - 1):
        q_s, q_e = cuts[i], cuts[i + 1]
        if q_s >= q_e:
            continue

        containing_image = None
        for sli in sorted_image_slices:
            if sli.start <= q_s < sli.stop:
                containing_image = sli
                break

        if containing_image is not None:
            base_type = AttnMaskType.FULL
            k_base_end = containing_image.stop
        else:
            base_type = AttnMaskType.CAUSAL
            k_base_end = q_e

        q_in_hole = _q_inside_hole(q_s, sorted_hole_slices)
        active_holes = [
            h for h in sorted_hole_slices
            if h.stop <= q_s and h is not q_in_hole
        ]

        k_segments = _subtract_holes_from_krange(sample_start, k_base_end, active_holes)

        for k_seg_start, k_seg_end in k_segments:
            if k_seg_start < k_seg_end:
                q_list.append([q_s, q_e])
                k_list.append([k_seg_start, k_seg_end])
                # q_start >= k_end, every position should attend.
                # Must use FULL — FFA's CAUSAL right-aligns with local indices and would
                # incorrectly mask positions in disjoint ranges.
                seg_type = AttnMaskType.FULL if (base_type == AttnMaskType.CAUSAL and q_s >= k_seg_end) else base_type
                m_list.append(seg_type)

    return q_list, k_list, m_list


def _subtract_holes_from_krange(k_start, k_end, active_holes):
    """Remove hole intervals from [k_start, k_end), returning remaining segments."""
    segments = [(k_start, k_end)]
    for h in active_holes:
        h_s, h_e = h.start, h.stop
        new_segments = []
        for s, e in segments:
            if h_e <= s or h_s >= e:
                new_segments.append((s, e))
            else:
                if s < h_s:
                    new_segments.append((s, h_s))
                if h_e < e:
                    new_segments.append((h_e, e))
        segments = new_segments
    return segments


def _q_inside_hole(q_pos, hole_slices):
    """Return the hole slice containing q_pos, or None."""
    for h in hole_slices:
        if h.start <= q_pos < h.stop:
            return h
    return None
