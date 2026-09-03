# Adapted from Megatron-LM: megatron/core/transformer/moe/router_replay.py
"""MoE router replay: freeze expert choices across forwards / models.

Enable via ``TransformerConfig.moe_enable_routing_replay=True`` so each
``DeepSeekMoEGate`` creates a ``RouterReplay`` instance (registered in
creation order into ``global_router_replay_instances``).

To control two coexisting models separately, tag instances with a name
after build via ``RouterReplay.tag_instances(...)``, then pass that
``name`` to the static helpers.

Typical usage (same model, two forwards)::

    from hy_parallelism.models.modules.moe.routers import (
        RouterReplay, RouterReplayAction,
    )

    # 1) Record routing on pass A
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    out_a = model(x)
    recorded = [t.detach().clone() for t in RouterReplay.get_recorded_data()]
    # recorded[i]: [num_tokens, topk] for MoE layer i

    # 2) Replay on pass B (gate params may differ; experts stay the same)
    RouterReplay.set_replay_data(recorded)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    out_b = model(x)

    # 3) Cleanup (avoid leaking into the next step)
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_indices()

Cross-model A -> B via named groups::

    RouterReplay.tag_instances(
        "teacher", [g.router_replay for g in model_a_gates]
    )
    RouterReplay.tag_instances(
        "student", [g.router_replay for g in model_b_gates]
    )

    RouterReplay.set_global_router_replay_action(
        RouterReplayAction.RECORD, name="teacher"
    )
    _ = model_a(x)
    recorded = [
        t.detach().clone()
        for t in RouterReplay.get_recorded_data(name="teacher")
    ]

    RouterReplay.set_replay_data(recorded, name="student")
    RouterReplay.set_global_router_replay_action(
        RouterReplayAction.REPLAY_FORWARD, name="student"
    )
    out_b = model_b(x)

Partial replay with holes (uncovered tokens use natural top-k)::

    assembled, coverage = assemble_routing_captures(
        captures, require_full_coverage=False, return_coverage=True
    )
    RouterReplay.set_replay_data(assembled, valid_mask=coverage)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

Notes:
- ``RECORD`` overwrites ``recorded_topk_idx`` each forward; it does not append.
  For decode-then-teacher-force, collect per-step indices yourself and concat.
- Replay only fixes ``topk_idx``; ``topk_weights`` are re-gathered from the
  current gate scores (then norm / scaling still apply in the gate).
- ``REPLAY_BACKWARD`` is for activation recompute; it consumes
  ``replay_backward_list`` in FIFO order (filled by ``set_target_indices``).
- Passing ``name=None`` keeps Megatron-style behavior over all instances.
- ``valid_mask=None`` means full-sequence replay; ``False`` positions fall
  back to natural top-k.
"""

from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


class RouterReplayAction(Enum):
    """Actions for router replay."""

    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """Record / replay MoE routing decisions (topk expert indices).

    One instance is typically created per MoE gate/router layer. Global static
    helpers control registered instances; pass ``name`` to scope to a named
    group after tagging via ``tag_instances``.
    """

    global_router_replay_instances: List["RouterReplay"] = []
    named_router_replay_instances: Dict[str, List["RouterReplay"]] = defaultdict(list)

    @staticmethod
    def tag_instances(name: str, instances: Sequence["RouterReplay"]):
        """Mark existing instances as belonging to a named group.

        Replaces any previous list under ``name``. If an instance was already
        tagged under another name, it is removed from that group first.
        """
        instances = list(instances)
        instance_ids = {id(inst) for inst in instances}

        empty_names = []
        for group_name, group in RouterReplay.named_router_replay_instances.items():
            if group_name == name:
                continue
            kept = [inst for inst in group if id(inst) not in instance_ids]
            if kept:
                RouterReplay.named_router_replay_instances[group_name] = kept
            else:
                empty_names.append(group_name)
        for group_name in empty_names:
            del RouterReplay.named_router_replay_instances[group_name]

        for inst in instances:
            inst.name = name
        RouterReplay.named_router_replay_instances[name] = instances

    @staticmethod
    def _instances(name: Optional[str] = None) -> List["RouterReplay"]:
        if name is None:
            return RouterReplay.global_router_replay_instances
        if name not in RouterReplay.named_router_replay_instances:
            raise KeyError(
                f"No RouterReplay instances registered under name={name!r}. "
                f"Known names: {sorted(RouterReplay.named_router_replay_instances)}"
            )
        return RouterReplay.named_router_replay_instances[name]

    @staticmethod
    def set_replay_data(
        all_layers_topk_indices: List[torch.Tensor],
        name: Optional[str] = None,
        valid_mask: Optional[Union[torch.Tensor, List[Optional[torch.Tensor]]]] = None,
    ):
        """Distribute per-layer topk indices to registered instances (creation order).

        Args:
            all_layers_topk_indices: Per-layer ``[T, topk]`` tables.
            name: Optional named group.
            valid_mask: Optional coverage mask. ``None`` means all positions
                are replayed. A single ``[T]`` bool tensor is shared across
                layers; a list gives a per-layer mask (``None`` entries mean
                full coverage for that layer). Uncovered positions
                (``False``) use natural top-k during replay.
        """
        instances = RouterReplay._instances(name)
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match instances ({len(instances)}"
                + (f", name={name!r}" if name is not None else "")
                + ")."
            )
        if valid_mask is None or isinstance(valid_mask, torch.Tensor):
            masks = [valid_mask] * len(instances)
        else:
            if len(valid_mask) != len(instances):
                raise ValueError(
                    f"valid_mask list length ({len(valid_mask)}) does not match "
                    f"instances ({len(instances)})."
                )
            masks = list(valid_mask)
        for i, router_instance in enumerate(instances):
            router_instance.set_target_indices(
                all_layers_topk_indices[i], valid_mask=masks[i]
            )

    @staticmethod
    def get_recorded_data(name: Optional[str] = None) -> List[Optional[torch.Tensor]]:
        """Collect recorded topk indices from instances (optionally by name)."""
        return [router.get_recorded_indices() for router in RouterReplay._instances(name)]

    @staticmethod
    def clear_global_indices(name: Optional[str] = None):
        for router in RouterReplay._instances(name):
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(
        router_replay_action: RouterReplayAction,
        name: Optional[str] = None,
    ):
        for router in RouterReplay._instances(name):
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action(name: Optional[str] = None):
        for router in RouterReplay._instances(name):
            router.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances(name: Optional[str] = None):
        """Clear registered instances.

        If ``name`` is given, remove that named group and drop those instances
        from the global list. If ``name`` is None, clear everything.
        """
        if name is None:
            RouterReplay.global_router_replay_instances.clear()
            RouterReplay.named_router_replay_instances.clear()
            return

        named = RouterReplay.named_router_replay_instances.pop(name, [])
        named_ids = {id(router) for router in named}
        RouterReplay.global_router_replay_instances = [
            router
            for router in RouterReplay.global_router_replay_instances
            if id(router) not in named_ids
        ]

    @staticmethod
    def set_global_static_buffers(
        static_buffer: torch.Tensor,
        name: Optional[str] = None,
    ):
        """Set static buffers from a combined buffer of shape [max_tokens, num_layers, topk]."""
        instances = RouterReplay._instances(name)
        num_layers = len(instances)
        assert static_buffer.shape[1] == num_layers, (
            f"Buffer has {static_buffer.shape[1]} layers but there are "
            f"{num_layers} RouterReplay instances"
            + (f" (name={name!r})" if name is not None else "")
            + "."
        )
        for layer_idx, router_instance in enumerate(instances):
            router_instance.set_static_buffer(static_buffer[:, layer_idx, :])

    @staticmethod
    def clear_global_static_buffers(name: Optional[str] = None):
        for router in RouterReplay._instances(name):
            router.clear_static_buffer()

    def __init__(self):
        self.name: Optional[str] = None
        self.target_topk_idx: Optional[torch.Tensor] = None
        self.target_valid_mask: Optional[torch.Tensor] = None
        self.recorded_topk_idx: Optional[torch.Tensor] = None
        self.router_replay_action: Optional[RouterReplayAction] = None
        # Each entry is (topk_indices, valid_mask); valid_mask may be None.
        self.replay_backward_list: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        self.static_buffer: Optional[torch.Tensor] = None
        self._replay_logged: bool = False  # mixin 可打开，避免每层都打 get_replay_topk log
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(
        self,
        topk_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        pad_to: Optional[int] = None,
    ):
        """Set target topk indices for replay.

        Args:
            topk_indices: Shape [T, topk].
            valid_mask: Optional shape [T] bool. True = use replayed indices;
                False = natural top-k. None = all True (for the provided T).
            pad_to: If set and > T, right-pad topk with zeros and extend
                valid_mask with False so the suffix uses natural top-k.
                If the forward is longer than the stored target, padding is
                also applied automatically in get_replay_topk.
        """
        if pad_to is not None:
            topk_indices, valid_mask = RouterReplay._align_replay_target(
                topk_indices, valid_mask, pad_to
            )
        if valid_mask is not None:
            if valid_mask.ndim != 1:
                raise ValueError(
                    f"valid_mask must be 1-D [T], got shape {tuple(valid_mask.shape)}"
                )
            if valid_mask.shape[0] != topk_indices.shape[0]:
                raise ValueError(
                    f"valid_mask T={valid_mask.shape[0]} != "
                    f"topk_indices T={topk_indices.shape[0]}"
                )
        self.target_topk_idx = topk_indices
        self.target_valid_mask = valid_mask
        self.replay_backward_list.append((topk_indices, valid_mask))

    @staticmethod
    def _align_replay_target(
        top_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Align top_indices to length seq_len; pad with valid_mask=False if shorter.

        Padding appends on the flat token axis [T, K]. Correct when the missing
        tokens are a suffix of that flat layout (e.g. B=1). For B>1 with a
        longer per-sample T, pad in [B, T, K] before flatten instead.
        """
        t = top_indices.shape[0]
        if t == seq_len:
            return top_indices, valid_mask
        if t > seq_len:
            raise ValueError(
                f"replay topk T={t} > forward T={seq_len}; cannot truncate"
            )
        pad_n = seq_len - t
        top_indices = torch.cat(
            [top_indices, top_indices.new_zeros(pad_n, top_indices.shape[1])],
            dim=0,
        )
        if valid_mask is None:
            valid_mask = torch.ones(t, dtype=torch.bool, device=top_indices.device)
        else:
            valid_mask = valid_mask.to(device=top_indices.device, dtype=torch.bool)
            if valid_mask.shape[0] != t:
                raise ValueError(
                    f"valid_mask T={valid_mask.shape[0]} != topk T={t} before pad"
                )
        valid_mask = torch.cat(
            [valid_mask, valid_mask.new_zeros(pad_n)],
            dim=0,
        )
        return top_indices, valid_mask

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        return self.recorded_topk_idx

    def clear_indices(self):
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.target_valid_mask = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        self.router_replay_action = None

    def _apply_replay_indices(
        self,
        scores: torch.Tensor,
        topk: int,
        top_indices: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        default_compute_topk: Callable,
        num_groups: Optional[int],
        group_topk: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_indices = top_indices.to(device=scores.device)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=scores.device, dtype=torch.bool)
        from hy_parallelism.common.logging import trace_log
        import os

        forward_T = scores.shape[0]
        target_T = top_indices.shape[0]
        try:
            top_indices, valid_mask = RouterReplay._align_replay_target(
                top_indices, valid_mask, forward_T
            )
        except ValueError as e:
            # Only happens when target_T > forward_T; dump replay context for debugging.
            trace_log(
                f'[Rank {os.getenv("RANK", "0")}]: '
                f"[RouterReplay._apply_replay_indices] name={self.name!r} "
                f"action={self.router_replay_action} "
                f"scores_T={forward_T} target_T={target_T} "
                f"valid_mask_T={None if valid_mask is None else valid_mask.shape[0]} "
                f"exc={e}"
            )
            raise
        if valid_mask is None or bool(valid_mask.all()):
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        natural_vals, natural_idx = default_compute_topk(
            scores, topk, num_groups=num_groups, group_topk=group_topk
        )
        mixed_idx = torch.where(valid_mask.unsqueeze(-1), top_indices, natural_idx)
        probs = scores.gather(1, mixed_idx)
        return probs, mixed_idx

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[
            [torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrap top-k selection according to the current replay action.

        Returns:
            (topk_values, topk_indices)
        """
        from hy_parallelism.utils import is_recomputing
        import os
        from hy_parallelism.common.logging import trace_log
        if self.router_replay_action == RouterReplayAction.REPLAY_FORWARD and self._replay_logged:
            mask = self.target_valid_mask
            # None = 整段 target 都 replay；有 mask 时 True 才是真正 replay
            if mask is None:
                replay_n = None if self.target_topk_idx is None else int(self.target_topk_idx.shape[0])
            else:
                replay_n = int(mask.sum().item())
            trace_log(
                f'[Rank {os.getenv("RANK", "0")}]: '
                f"[RouterReplay.get_replay_topk] name={self.name!r} "
                f"action={self.router_replay_action} is_recomputing={is_recomputing()} "
                f"scores_T={scores.shape[0]} "
                f"target_T={None if self.target_topk_idx is None else tuple(self.target_topk_idx.shape)} "
                f"replay_n={replay_n} "
                f"valid_mask_T={None if mask is None else mask.shape[0]}",
            )
        if self.router_replay_action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            return self._apply_replay_indices(
                scores,
                topk,
                self.target_topk_idx,
                self.target_valid_mask,
                default_compute_topk,
                num_groups,
                group_topk,
            )
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices, valid_mask = self.replay_backward_list.pop(0)
            return self._apply_replay_indices(
                scores,
                topk,
                top_indices,
                valid_mask,
                default_compute_topk,
                num_groups,
                group_topk,
            )
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)

    def set_static_buffer(self, buffer: torch.Tensor):
        self.static_buffer = buffer

    def clear_static_buffer(self):
        self.static_buffer = None

    def record_indices(self, topk_indices: torch.Tensor):
        if self.static_buffer is not None:
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices
