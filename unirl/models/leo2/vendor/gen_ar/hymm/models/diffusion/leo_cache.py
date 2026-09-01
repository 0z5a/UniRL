"""Inference-only first-block cache for Leo diffusion blocks."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

TensorStreams = tuple[torch.Tensor | None, ...]
SyncPlan = tuple[list[dist.ProcessGroup], list[dist.ProcessGroup]]


@dataclass(frozen=True)
class LeoFirstBlockCacheConfig:
    """Configure Leo first-block residual caching."""

    threshold: float = 0.05


class LeoFirstBlockCacheController:
    """Track request-local Leo block residuals without registering model state."""

    def __init__(self, config: object):
        threshold = getattr(config, "threshold", None)
        if not isinstance(threshold, (int, float)):
            raise TypeError("First-block cache config must define a numeric `threshold`.")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("First-block cache threshold must be finite and non-negative.")
        self.threshold = float(threshold)
        self._context_depth = 0
        self._signature = None
        self._previous_head_residuals: TensorStreams | None = None
        self._tail_residuals: TensorStreams | None = None
        self._cacheable_step = False
        self.full_steps = 0
        self.skipped_steps = 0

    @property
    def active(self) -> bool:
        """Return whether an inference context currently owns the cache."""
        return self._context_depth > 0 and not torch.is_grad_enabled()

    @contextmanager
    def context(self, name: str = "default") -> Iterator[None]:
        """Scope cache state to one complete denoising trajectory."""
        _ = name
        if self._context_depth == 0:
            self.reset()
            self.full_steps = 0
            self.skipped_steps = 0
        self._context_depth += 1
        try:
            yield
        finally:
            self._context_depth -= 1
            if self._context_depth == 0:
                self.reset()

    def reset(self) -> None:
        """Drop cached activations while retaining the latest counters."""
        self._signature = None
        self._previous_head_residuals = None
        self._tail_residuals = None
        self._cacheable_step = False

    @torch.compiler.disable
    def should_reuse(
        self,
        head_inputs: TensorStreams,
        head_outputs: TensorStreams,
        leader_block: object = None,
    ) -> bool:
        """Decide whether the cached tail may replace all blocks after block zero."""
        sync_plan = self._synchronization_plan(leader_block)
        self._cacheable_step = sync_plan is not None
        if sync_plan is None:
            return False
        sum_groups, max_groups = sync_plan
        decision_groups = [*sum_groups, *max_groups]

        signature = self._stream_signature(head_inputs)
        current_residuals = self._decision_residuals(head_inputs, head_outputs)
        locally_valid = (
            self._signature == signature
            and self._previous_head_residuals is not None
            and self._tail_residuals is not None
        )
        valid = torch.tensor(
            int(locally_valid), device=current_residuals[0].device, dtype=torch.int32
        )
        self._all_reduce(valid, decision_groups, dist.ReduceOp.MIN)

        if not bool(valid.item()):
            self._signature = signature
            self._previous_head_residuals = self._detach_streams(current_residuals)
            self._tail_residuals = None
            return False

        score = self._normalized_change(
            current_residuals,
            self._previous_head_residuals,
            sum_groups,
            max_groups,
        )
        reuse = score <= self.threshold
        if reuse:
            self.skipped_steps += 1
            return True
        self._previous_head_residuals = self._detach_streams(current_residuals)
        return False

    def apply_tail(self, head_outputs: TensorStreams) -> TensorStreams:
        """Reconstruct the block-stack output from cached tail residuals."""
        if self._tail_residuals is None:
            raise RuntimeError("Leo first-block cache has no tail residuals to apply.")
        self._validate_streams(head_outputs, self._tail_residuals, "apply")
        return tuple(
            None if output is None else output + residual
            for output, residual in zip(head_outputs, self._tail_residuals)
        )

    def update_tail(self, head_outputs: TensorStreams, final_outputs: TensorStreams) -> None:
        """Store the residual contributed by all blocks after block zero."""
        self._validate_streams(head_outputs, final_outputs, "update")
        self.full_steps += 1
        if not self._cacheable_step:
            return
        self._tail_residuals = tuple(
            None if output is None else (final - output).detach()
            for output, final in zip(head_outputs, final_outputs)
        )

    def stats(self) -> dict[str, int | float]:
        """Return counters for the most recently entered cache context."""
        return {
            "threshold": self.threshold,
            "full_steps": self.full_steps,
            "skipped_steps": self.skipped_steps,
        }

    @staticmethod
    def _decision_residuals(head_inputs: TensorStreams, head_outputs: TensorStreams) -> TensorStreams:
        """Build visual and optional audio first-block residuals."""
        visual = head_outputs[0] - head_inputs[0]
        if len(head_inputs) == 3 and head_inputs[1] is not None:
            audio = head_outputs[1] - head_inputs[1]
            return visual, audio
        return visual, None

    @staticmethod
    def _detach_streams(streams: TensorStreams) -> TensorStreams:
        """Detach cache tensors from any accidental autograd graph."""
        return tuple(None if stream is None else stream.detach() for stream in streams)

    @staticmethod
    def _stream_signature(streams: TensorStreams) -> tuple:
        """Describe stream layouts that must remain stable while reusing residuals."""
        return tuple(
            None if stream is None else (tuple(stream.shape), stream.dtype, stream.device)
            for stream in streams
        )

    @staticmethod
    def _validate_streams(left: TensorStreams, right: TensorStreams, operation: str) -> None:
        """Reject incompatible stream tuples before residual arithmetic."""
        if len(left) != len(right):
            raise RuntimeError(
                f"Cannot {operation} Leo cache streams with lengths {len(left)} and {len(right)}."
            )
        for index, (left_stream, right_stream) in enumerate(zip(left, right)):
            if (left_stream is None) != (right_stream is None):
                raise RuntimeError(
                    f"Cannot {operation} Leo cache stream {index}: one side is None."
                )

    @staticmethod
    def _all_reduce(value: torch.Tensor, groups: list[dist.ProcessGroup], op: dist.ReduceOp) -> None:
        """Reduce across the model-sharding group."""
        for group in groups:
            dist.all_reduce(value, op=op, group=group)

    @classmethod
    def _normalized_change(
        cls,
        current_streams: TensorStreams,
        previous_streams: TensorStreams,
        sum_groups: list[dist.ProcessGroup],
        max_groups: list[dist.ProcessGroup],
    ) -> float:
        """Compute the global normalized first-block residual change."""
        if len(current_streams) != len(previous_streams):
            raise RuntimeError("Cannot compare Leo cache decision streams with different lengths.")
        reference = current_streams[0]
        rows = []
        for current, previous in zip(current_streams, previous_streams):
            if (current is None) != (previous is None):
                raise RuntimeError("Cannot compare Leo cache decision streams with mismatched presence.")
            if current is None:
                zero = reference.new_tensor(0, dtype=torch.float32)
                rows.append(torch.stack((zero, zero, zero)))
            else:
                rows.append(torch.stack((
                    (current - previous).float().abs().sum(),
                    previous.float().abs().sum(),
                    current.new_tensor(current.numel(), dtype=torch.float32),
                )))
        values = torch.stack(rows)
        cls._all_reduce(values, sum_groups, dist.ReduceOp.SUM)

        scores = []
        for index in range(len(rows)):
            count = values[index][2].clamp_min(1)
            numerator = values[index][0] / count
            denominator = (values[index][1] / count).clamp_min(torch.finfo(torch.float32).eps)
            scores.append(numerator / denominator)
        scores = torch.stack(scores)
        cls._all_reduce(scores, max_groups, dist.ReduceOp.MAX)
        return max(float(scores[index].item()) for index in range(len(rows)))

    @staticmethod
    def _actual_fsdp_group(leader_block: object) -> tuple[bool, dist.ProcessGroup | None]:
        """Resolve the shard group from the block's real FSDP2 wrapper."""
        if leader_block is None or not hasattr(leader_block, "modules"):
            return False, None
        try:
            from torch.distributed.fsdp import FSDPModule
        except ImportError:
            return False, None

        fsdp_modules = [module for module in leader_block.modules() if isinstance(module, FSDPModule)]
        if not fsdp_modules:
            for parameter in leader_block.parameters():
                mesh = getattr(parameter, "device_mesh", None)
                placements = getattr(parameter, "placements", ())
                shard_dims = [
                    index
                    for index, placement in enumerate(placements)
                    if getattr(placement, "is_shard", lambda: False)()
                ]
                if mesh is None or not shard_dims:
                    continue
                if len(shard_dims) != 1:
                    return True, None
                try:
                    return True, mesh.get_group(shard_dims[0])
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return True, None
            return False, None
        for module in fsdp_modules:
            try:
                state = module._get_fsdp_state()
                plural_groups = getattr(state, "_fsdp_param_groups", None)
                if plural_groups is not None:
                    if len(plural_groups) > 1:
                        return True, None
                    param_groups = list(plural_groups)
                else:
                    param_groups = [getattr(state, "_fsdp_param_group", None)]
                for param_group in param_groups:
                    mesh_info = getattr(param_group, "mesh_info", None)
                    if mesh_info is None:
                        continue
                    group = getattr(mesh_info, "shard_process_group", None)
                    if group is not None:
                        return True, group
                    mesh = getattr(mesh_info, "mesh", None)
                    shard_dim = getattr(mesh_info, "shard_mesh_dim", None)
                    if mesh is not None and shard_dim is not None:
                        return True, mesh.get_group(shard_dim)
            except (AssertionError, AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return True, None

    @classmethod
    def _synchronization_plan(cls, leader_block: object) -> SyncPlan | None:
        """Resolve CP and actual FSDP groups for the supported topology."""
        if not dist.is_available() or not dist.is_initialized():
            return [], []
        if dist.get_world_size() == 1:
            return [], []

        from hy_parallelism import parallel_states as hy_ps

        if not hy_ps.is_parallel_state_initialized():
            return None
        parallel_state = hy_ps.get_parallel_state()
        if any(getattr(parallel_state, dim, 1) != 1 for dim in ("ep", "etp", "tp", "pp")):
            return None

        sum_groups = []
        if getattr(parallel_state, "cp", 1) > 1:
            try:
                cp_mesh = parallel_state.get_mesh("cp")
            except (AttributeError, ValueError):
                cp_mesh = getattr(parallel_state, "_meshes", {}).get("cp")
            if cp_mesh is None:
                return None
            try:
                sum_groups.append(cp_mesh.get_group())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return None

        is_fsdp, fsdp_group = cls._actual_fsdp_group(leader_block)
        if is_fsdp and fsdp_group is None:
            return None

        max_groups = []
        if fsdp_group is not None:
            try:
                if dist.get_world_size(group=fsdp_group) > 1:
                    max_groups.append(fsdp_group)
            except (RuntimeError, TypeError, ValueError):
                return None
        return sum_groups, max_groups
