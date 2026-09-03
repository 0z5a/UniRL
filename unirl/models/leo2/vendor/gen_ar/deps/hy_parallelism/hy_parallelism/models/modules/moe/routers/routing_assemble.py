"""Assemble multi-step MoE routing captures into a full-sequence replay table.

Use with ``RouterReplay`` when generation has multiple forwards (prefill /
decode / multi-prefill): each forward yields a ``RoutingCapture`` with
``pos_ids``, then ``assemble_routing_captures`` packs them into
``List[[seq_len, topk]]`` for ``RouterReplay.set_replay_data``.

When some positions are intentionally left uncovered, pass
``require_full_coverage=False, return_coverage=True`` and feed the coverage
mask into ``RouterReplay.set_replay_data(..., valid_mask=coverage)`` so
holes fall back to natural top-k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union, overload

import torch


@dataclass
class RoutingCapture:
    """Routing indices from one forward, placed onto a full-sequence layout.

    Attributes:
        topk_indices: Per-layer tensors of shape ``[T, topk]``.
        pos_ids: Shape ``[T]``; ``pos_ids[j]`` is the full-sequence position
            for ``topk_indices[layer][j]``.
        tag: Optional debug label (e.g. ``"prefill0"``, ``"decode17"``).
    """

    topk_indices: List[torch.Tensor]
    pos_ids: torch.Tensor
    tag: Optional[str] = None


@dataclass
class AssembledRouterReplay:
    """Full-sequence routing tables ready for ``RouterReplay.set_replay_data``.

    Attributes:
        topk_indices: Per-layer ``[seq_len, topk]``.
        valid_mask: Optional ``[seq_len]`` bool; ``False`` positions use natural
            top-k. ``None`` means full coverage (all positions replayed).
    """

    topk_indices: List[torch.Tensor]
    valid_mask: Optional[torch.Tensor] = None

    def set_replay(
        self,
        name: Optional[str] = None,
        *,
        set_action: bool = True,
    ):
        """Load this table into RouterReplay instances and optionally enable REPLAY_FORWARD.

        Args:
            name: Named RouterReplay group. ``None`` targets all instances.
            set_action: If True, also set ``REPLAY_FORWARD`` on that group.
        """
        from hy_parallelism.models.modules.moe.routers.router_replay import (
            RouterReplay,
            RouterReplayAction,
        )

        # Lazy clear of previous targets before installing a new replay table.
        RouterReplay.clear_global_indices(name=name)
        RouterReplay.set_replay_data(
            self.topk_indices, name=name, valid_mask=self.valid_mask
        )
        if set_action:
            RouterReplay.set_global_router_replay_action(
                RouterReplayAction.REPLAY_FORWARD, name=name
            )
        return self


def make_routing_capture(
    recorded: Sequence[Optional[torch.Tensor]],
    pos_ids: Union[torch.Tensor, Sequence[int]],
    tag: Optional[str] = None,
) -> RoutingCapture:
    """Build a ``RoutingCapture`` from ``RouterReplay.get_recorded_data()``."""
    if any(t is None for t in recorded):
        raise ValueError(
            f"recorded contains None (tag={tag!r}); ensure RECORD ran for all layers"
        )
    pos = torch.as_tensor(pos_ids, dtype=torch.long)
    if pos.ndim != 1:
        raise ValueError(f"pos_ids must be 1-D, got shape {tuple(pos.shape)} (tag={tag!r})")
    return RoutingCapture(
        topk_indices=[t.detach() for t in recorded],
        pos_ids=pos,
        tag=tag,
    )


def validate_routing_capture(capture: RoutingCapture) -> None:
    """Check internal consistency of one capture."""
    tag = capture.tag
    if not capture.topk_indices:
        raise ValueError(f"capture has no layers (tag={tag!r})")
    pos = capture.pos_ids
    if not isinstance(pos, torch.Tensor):
        raise TypeError(f"pos_ids must be a Tensor (tag={tag!r})")
    if pos.ndim != 1:
        raise ValueError(f"pos_ids must be 1-D, got {tuple(pos.shape)} (tag={tag!r})")
    if pos.dtype not in (torch.int32, torch.int64, torch.long):
        raise ValueError(f"pos_ids dtype must be integer, got {pos.dtype} (tag={tag!r})")
    if (pos < 0).any():
        raise ValueError(f"pos_ids contains negative values (tag={tag!r})")

    num_tokens = pos.numel()
    ref = capture.topk_indices[0]
    if ref.ndim != 2:
        raise ValueError(
            f"topk_indices[0] must be [T, topk], got {tuple(ref.shape)} (tag={tag!r})"
        )
    topk = ref.shape[1]
    for layer_idx, tensor in enumerate(capture.topk_indices):
        if tensor is None:
            raise ValueError(f"topk_indices[{layer_idx}] is None (tag={tag!r})")
        if tensor.ndim != 2:
            raise ValueError(
                f"topk_indices[{layer_idx}] must be [T, topk], "
                f"got {tuple(tensor.shape)} (tag={tag!r})"
            )
        if tensor.shape[0] != num_tokens:
            raise ValueError(
                f"topk_indices[{layer_idx}] T={tensor.shape[0]} != "
                f"len(pos_ids)={num_tokens} (tag={tag!r})"
            )
        if tensor.shape[1] != topk:
            raise ValueError(
                f"topk_indices[{layer_idx}] topk={tensor.shape[1]} != "
                f"topk={topk} from layer 0 (tag={tag!r})"
            )


def validate_assembled_routing(
    assembled: Sequence[torch.Tensor],
    *,
    seq_len: Optional[int] = None,
    topk: Optional[int] = None,
) -> None:
    """Check assembled per-layer tables share layout ``[seq_len, topk]``."""
    if not assembled:
        raise ValueError("assembled is empty")
    ref = assembled[0]
    if ref.ndim != 2:
        raise ValueError(f"assembled[0] must be [S, topk], got {tuple(ref.shape)}")
    s, k = ref.shape
    if seq_len is not None and s != seq_len:
        raise ValueError(f"assembled seq_len={s} != expected {seq_len}")
    if topk is not None and k != topk:
        raise ValueError(f"assembled topk={k} != expected {topk}")
    for i, tensor in enumerate(assembled):
        if tensor.shape != (s, k):
            raise ValueError(
                f"assembled[{i}] shape {tuple(tensor.shape)} != {(s, k)}"
            )


@overload
def assemble_routing_captures(
    captures: Sequence[RoutingCapture],
    *,
    seq_len: Optional[int] = None,
    overlap: Literal["error", "overwrite"] = "error",
    require_full_coverage: bool = True,
    return_coverage: Literal[False] = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    ...


@overload
def assemble_routing_captures(
    captures: Sequence[RoutingCapture],
    *,
    seq_len: Optional[int] = None,
    overlap: Literal["error", "overwrite"] = "error",
    require_full_coverage: bool = True,
    return_coverage: Literal[True],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    ...


def assemble_routing_captures(
    captures: Sequence[RoutingCapture],
    *,
    seq_len: Optional[int] = None,
    overlap: Literal["error", "overwrite"] = "error",
    require_full_coverage: bool = True,
    return_coverage: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    """Pack multi-forward routing captures into full-sequence tables.

    Args:
        captures: Ordered list of per-forward captures.
        seq_len: Full sequence length. Default: ``max(pos_ids) + 1``.
        overlap: ``"error"`` if the same position is written twice;
            ``"overwrite"`` keeps the last write.
        require_full_coverage: If True, every position in ``[0, seq_len)``
            must be written at least once. Set False to allow holes.
        return_coverage: If True, also return a ``[seq_len]`` bool mask
            (``True`` = position has injected replay indices). Pass that
            mask to ``RouterReplay.set_replay_data(..., valid_mask=...)``
            so holes use natural top-k.
        dtype / device: Output tensor attrs; default from the first layer
            of the first capture.

    Returns:
        ``assembled[layer]`` with shape ``[seq_len, topk]``. Uncovered
        positions are filled with zeros (ignored when ``valid_mask`` is used).
        If ``return_coverage``, also returns coverage ``[seq_len]`` bool.
    """
    if not captures:
        raise ValueError("captures is empty")
    if overlap not in ("error", "overwrite"):
        raise ValueError(f"overlap must be 'error' or 'overwrite', got {overlap!r}")

    for capture in captures:
        validate_routing_capture(capture)

    num_layers = len(captures[0].topk_indices)
    topk = captures[0].topk_indices[0].shape[1]
    for capture in captures[1:]:
        if len(capture.topk_indices) != num_layers:
            raise ValueError(
                f"num_layers mismatch: {len(capture.topk_indices)} vs {num_layers} "
                f"(tag={capture.tag!r})"
            )
        if capture.topk_indices[0].shape[1] != topk:
            raise ValueError(
                f"topk mismatch: {capture.topk_indices[0].shape[1]} vs {topk} "
                f"(tag={capture.tag!r})"
            )

    max_pos = max(int(c.pos_ids.max().item()) for c in captures if c.pos_ids.numel() > 0)
    if all(c.pos_ids.numel() == 0 for c in captures):
        raise ValueError("all captures have empty pos_ids")
    resolved_seq_len = max_pos + 1 if seq_len is None else seq_len
    if resolved_seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {resolved_seq_len}")
    for capture in captures:
        if capture.pos_ids.numel() == 0:
            continue
        if int(capture.pos_ids.max().item()) >= resolved_seq_len:
            raise ValueError(
                f"pos_ids max={int(capture.pos_ids.max().item())} >= seq_len="
                f"{resolved_seq_len} (tag={capture.tag!r})"
            )

    ref0 = captures[0].topk_indices[0]
    out_dtype = dtype if dtype is not None else ref0.dtype
    out_device = device if device is not None else ref0.device

    # Zeros for holes: ignored when paired with valid_mask for natural routing.
    assembled = [
        torch.zeros(resolved_seq_len, topk, dtype=out_dtype, device=out_device)
        for _ in range(num_layers)
    ]
    written = torch.zeros(resolved_seq_len, dtype=torch.bool, device=out_device)

    for capture in captures:
        if capture.pos_ids.numel() == 0:
            continue
        pos = capture.pos_ids.to(device=out_device, dtype=torch.long)
        if overlap == "error":
            conflict = written[pos]
            if conflict.any():
                bad = pos[conflict].unique().tolist()
                raise ValueError(
                    f"overlapping pos_ids {bad} while overlap='error' "
                    f"(tag={capture.tag!r})"
                )
        for layer_idx in range(num_layers):
            src = capture.topk_indices[layer_idx].to(device=out_device, dtype=out_dtype)
            assembled[layer_idx][pos] = src
        written[pos] = True

    if require_full_coverage:
        missing = (~written).nonzero(as_tuple=False).flatten().tolist()
        if missing:
            preview = missing[:16]
            more = "" if len(missing) <= 16 else f" ... (+{len(missing) - 16} more)"
            raise ValueError(
                f"require_full_coverage=True but missing positions {preview}{more} "
                f"(seq_len={resolved_seq_len})"
            )

    validate_assembled_routing(assembled, seq_len=resolved_seq_len, topk=topk)
    if return_coverage:
        return assembled, written
    return assembled
