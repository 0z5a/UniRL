"""Inference-only feature caches for Leo diffusion blocks."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

TensorStreams = tuple[torch.Tensor | None, ...]
SyncPlan = tuple[list[dist.ProcessGroup], list[dist.ProcessGroup]]


def _unique_tensor_bytes(*stream_groups: TensorStreams | None) -> int:
    """Count request-local cache tensors once per Python object."""
    seen: set[int] = set()
    total = 0
    for streams in stream_groups:
        if streams is None:
            continue
        for tensor in streams:
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass(frozen=True)
class LeoFirstBlockCacheConfig:
    """Configure Leo first-block residual caching."""

    threshold: float = 0.05


@dataclass(frozen=True)
class LeoTaylorCacheConfig(LeoFirstBlockCacheConfig):
    """Configure first-block caching with first-order tail prediction."""

    max_extrapolation: float = 1.0


@dataclass(frozen=True)
class LeoMagCacheConfig:
    """Configure full-stack residual reuse from a calibrated magnitude profile."""

    threshold: float = 0.24
    max_skip_steps: int = 6
    retention_ratio: float = 0.2
    ratios: tuple[float, ...] = ()
    expected_timesteps: tuple[float, ...] = ()
    calibrate: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate the immutable MagCache profile."""
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise TypeError("MagCache threshold must be numeric.")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("MagCache threshold must be finite and non-negative.")
        if isinstance(self.max_skip_steps, bool) or not isinstance(self.max_skip_steps, int):
            raise TypeError("MagCache max_skip_steps must be an integer.")
        if self.max_skip_steps < 0:
            raise ValueError("MagCache max_skip_steps must be non-negative.")
        if isinstance(self.retention_ratio, bool) or not isinstance(self.retention_ratio, (int, float)):
            raise TypeError("MagCache retention_ratio must be numeric.")
        if not math.isfinite(self.retention_ratio) or not 0 <= self.retention_ratio <= 1:
            raise ValueError("MagCache retention_ratio must be finite and in [0, 1].")
        if not isinstance(self.calibrate, bool):
            raise TypeError("MagCache calibrate must be a bool.")

        try:
            ratios = tuple(self.ratios)
            expected_timesteps = tuple(self.expected_timesteps)
        except TypeError as exc:
            raise TypeError("MagCache ratios and expected_timesteps must be sequences.") from exc
        for index, ratio in enumerate(ratios):
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise TypeError(f"MagCache ratio {index} must be numeric.")
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"MagCache ratio {index} must be finite and positive.")
        for index, timestep in enumerate(expected_timesteps):
            if isinstance(timestep, bool) or not isinstance(timestep, (int, float)):
                raise TypeError(f"MagCache expected timestep {index} must be numeric.")
            if not math.isfinite(timestep):
                raise ValueError(f"MagCache expected timestep {index} must be finite.")
        if self.calibrate:
            if ratios or expected_timesteps:
                raise ValueError("MagCache calibration requires empty ratios and expected_timesteps.")
        elif not ratios or len(ratios) != len(expected_timesteps):
            raise ValueError("MagCache ratios and expected_timesteps must have the same non-zero length.")
        elif ratios[0] != 1.0:
            raise ValueError("MagCache ratio 0 must be exactly 1.0.")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "retention_ratio", float(self.retention_ratio))
        object.__setattr__(self, "ratios", tuple(float(value) for value in ratios))
        object.__setattr__(self, "expected_timesteps", tuple(float(value) for value in expected_timesteps))


class LeoFirstBlockCacheController:
    """Track request-local Leo block residuals without registering model state."""

    def __init__(self, config: object):
        threshold = getattr(config, "threshold", None)
        if not isinstance(threshold, (int, float)):
            raise TypeError("First-block cache config must define a numeric `threshold`.")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("First-block cache threshold must be finite and non-negative.")
        self.threshold = float(threshold)
        self.method = "taylor" if isinstance(config, LeoTaylorCacheConfig) else "first_block"
        max_extrapolation = getattr(config, "max_extrapolation", 0.0)
        if not isinstance(max_extrapolation, (int, float)):
            raise TypeError("Taylor cache config must define a numeric `max_extrapolation`.")
        if not math.isfinite(max_extrapolation) or max_extrapolation < 0:
            raise ValueError("Taylor cache max_extrapolation must be finite and non-negative.")
        self.max_extrapolation = float(max_extrapolation)
        self._context_depth = 0
        self._signature = None
        self._previous_head_residuals: TensorStreams | None = None
        self._previous_tail_residuals: TensorStreams | None = None
        self._tail_residuals: TensorStreams | None = None
        self._previous_tail_timestep: float | None = None
        self._tail_timestep: float | None = None
        self._pending_timestep: float | None = None
        self._pending_alpha: float | None = None
        self._cacheable_step = False
        self.full_steps = 0
        self.skipped_steps = 0
        self.predicted_steps = 0
        self.prediction_warmup_steps = 0
        self.alpha_sum = 0.0
        self.alpha_max = 0.0
        self._peak_cache_bytes = 0

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
            self.predicted_steps = 0
            self.prediction_warmup_steps = 0
            self.alpha_sum = 0.0
            self.alpha_max = 0.0
            self._peak_cache_bytes = 0
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
        self._previous_tail_residuals = None
        self._tail_residuals = None
        self._previous_tail_timestep = None
        self._tail_timestep = None
        self._pending_timestep = None
        self._pending_alpha = None
        self._cacheable_step = False

    @torch.compiler.disable
    def should_reuse(
        self,
        head_inputs: TensorStreams,
        head_outputs: TensorStreams,
        leader_block: object = None,
        timestep: torch.Tensor | None = None,
    ) -> bool:
        """Decide whether the cached tail may replace all blocks after block zero."""
        sync_plan = self._synchronization_plan(leader_block)
        self._cacheable_step = sync_plan is not None
        if sync_plan is None:
            return False
        sum_groups, max_groups = sync_plan
        decision_groups = [*sum_groups, *max_groups]
        self._pending_alpha = None
        self._pending_timestep = None

        if self.method == "taylor":
            if len(head_inputs) == 3 and head_inputs[1] is not None:
                raise RuntimeError(
                    "Leo Taylor cache currently supports video-only inference; "
                    "audio requires an independent scheduler timestep."
                )
            self._pending_timestep = self._uniform_timestep(timestep, decision_groups)

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
            self._previous_tail_residuals = None
            self._tail_residuals = None
            self._previous_tail_timestep = None
            self._tail_timestep = None
            self._record_cache_bytes()
            return False

        if self.method == "taylor":
            current_timestep = self._pending_timestep
            predictor_ready = (
                current_timestep is not None
                and self._previous_tail_residuals is not None
                and self._previous_tail_timestep is not None
                and self._tail_timestep is not None
                and self._tail_timestep != self._previous_tail_timestep
            )
            predictor_valid = torch.tensor(
                int(predictor_ready), device=current_residuals[0].device, dtype=torch.int32
            )
            self._all_reduce(predictor_valid, decision_groups, dist.ReduceOp.MIN)
            if not bool(predictor_valid.item()):
                self._previous_head_residuals = self._detach_streams(current_residuals)
                self._pending_timestep = current_timestep
                self.prediction_warmup_steps += 1
                self._record_cache_bytes()
                return False
            self._pending_timestep = current_timestep

        score = self._normalized_change(
            current_residuals,
            self._previous_head_residuals,
            sum_groups,
            max_groups,
        )
        reuse = score <= self.threshold
        if reuse:
            if self.method == "taylor":
                alpha = (self._pending_timestep - self._tail_timestep) / (
                    self._tail_timestep - self._previous_tail_timestep
                )
                if not math.isfinite(alpha) or alpha < 0:
                    self._previous_head_residuals = self._detach_streams(current_residuals)
                    self._record_cache_bytes()
                    return False
                self._pending_alpha = min(alpha, self.max_extrapolation)
                self.predicted_steps += 1
                self.alpha_sum += self._pending_alpha
                self.alpha_max = max(self.alpha_max, self._pending_alpha)
            self.skipped_steps += 1
            return True
        self._previous_head_residuals = self._detach_streams(current_residuals)
        self._record_cache_bytes()
        return False

    def apply_tail(self, head_outputs: TensorStreams) -> TensorStreams:
        """Reconstruct the block-stack output from cached tail residuals."""
        if self._tail_residuals is None:
            raise RuntimeError("Leo first-block cache has no tail residuals to apply.")
        tail_residuals = self._tail_residuals
        if self.method == "taylor":
            if self._previous_tail_residuals is None or self._pending_alpha is None:
                raise RuntimeError("Leo Taylor cache has no valid residual history to apply.")
            self._validate_streams(tail_residuals, self._previous_tail_residuals, "predict")
            tail_residuals = tuple(
                None if current is None else current + (current - previous) * self._pending_alpha
                for current, previous in zip(tail_residuals, self._previous_tail_residuals)
            )
        self._validate_streams(head_outputs, tail_residuals, "apply")
        return tuple(
            None if output is None else output + residual
            for output, residual in zip(head_outputs, tail_residuals)
        )

    def update_tail(
        self,
        head_outputs: TensorStreams,
        final_outputs: TensorStreams,
        timestep: torch.Tensor | None = None,
    ) -> None:
        """Store the residual contributed by all blocks after block zero."""
        self._validate_streams(head_outputs, final_outputs, "update")
        self.full_steps += 1
        if not self._cacheable_step:
            return
        next_tail = tuple(
            None if output is None else (final - output).detach()
            for output, final in zip(head_outputs, final_outputs)
        )
        if self.method == "taylor":
            current_timestep = self._pending_timestep
            if current_timestep is None:
                self._previous_tail_residuals = None
                self._tail_residuals = None
                self._previous_tail_timestep = None
                self._tail_timestep = None
                return
            self._previous_tail_residuals = self._tail_residuals
            self._previous_tail_timestep = self._tail_timestep
            self._tail_timestep = current_timestep
        self._tail_residuals = next_tail
        self._record_cache_bytes()

    def stats(self) -> dict[str, object]:
        """Return counters for the most recently entered cache context."""
        return {
            "method": self.method,
            "threshold": self.threshold,
            "full_steps": self.full_steps,
            "skipped_steps": self.skipped_steps,
            "predicted_steps": self.predicted_steps,
            "prediction_warmup_steps": self.prediction_warmup_steps,
            "taylor_max_extrapolation": self.max_extrapolation,
            "taylor_alpha_mean": self.alpha_sum / max(self.predicted_steps, 1),
            "taylor_alpha_max": self.alpha_max,
            "cache_bytes": self._peak_cache_bytes,
        }

    def _record_cache_bytes(self) -> None:
        """Track the peak logical bytes held by persistent cache tensors."""
        current = _unique_tensor_bytes(
            self._previous_head_residuals,
            self._previous_tail_residuals,
            self._tail_residuals,
        )
        self._peak_cache_bytes = max(self._peak_cache_bytes, current)

    @classmethod
    def _uniform_timestep(
        cls,
        timestep: torch.Tensor | None,
        groups: list[dist.ProcessGroup],
    ) -> float | None:
        """Return one globally uniform finite timestep or fail closed."""
        if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
            return None
        values = timestep.detach().float()
        bounds = torch.stack((values.amin(), values.amax()))
        low = bounds[0].clone()
        high = bounds[1].clone()
        cls._all_reduce(low, groups, dist.ReduceOp.MIN)
        cls._all_reduce(high, groups, dist.ReduceOp.MAX)
        low_value = float(low.item())
        high_value = float(high.item())
        if not math.isfinite(low_value) or not math.isfinite(high_value) or low_value != high_value:
            return None
        return low_value

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


class LeoMagCacheController(LeoFirstBlockCacheController):
    """Reuse the full Leo block-stack residual from a calibrated magnitude profile."""

    def __init__(self, config: LeoMagCacheConfig):
        if not isinstance(config, LeoMagCacheConfig):
            raise TypeError("LeoMagCacheController requires LeoMagCacheConfig.")
        self.threshold = config.threshold
        self.max_skip_steps = config.max_skip_steps
        self.retention_ratio = config.retention_ratio
        self.ratios = config.ratios
        self.expected_timesteps = config.expected_timesteps
        self.calibrate = config.calibrate
        self.method = "magcache"
        self._context_depth = 0
        self._signature = None
        self._full_residuals: TensorStreams | None = None
        self._previous_full_residuals: TensorStreams | None = None
        self._pending_timestep: float | None = None
        self._pending_sync_plan: SyncPlan | None = None
        self._distributed_config_validated = False
        self._step_index = 0
        self._accumulated_error = 0.0
        self._accumulated_ratio = 1.0
        self._accumulated_steps = 0
        self.full_steps = 0
        self.skipped_steps = 0
        self.observed_ratios: list[float] = []
        self.observed_timesteps: list[float] = []
        self._peak_cache_bytes = 0

    @contextmanager
    def context(self, name: str = "default") -> Iterator[None]:
        """Scope MagCache state and profile validation to one denoising trajectory."""
        _ = name
        if self._context_depth == 0:
            self.reset()
            self._step_index = 0
            self.full_steps = 0
            self.skipped_steps = 0
            self.observed_ratios = []
            self.observed_timesteps = []
            self._distributed_config_validated = False
            self._peak_cache_bytes = 0
        self._context_depth += 1
        completed = False
        try:
            yield
            completed = True
        finally:
            self._context_depth -= 1
            if self._context_depth == 0:
                consumed_steps = self._step_index
                self.reset()
                if completed and not self.calibrate and consumed_steps != len(self.ratios):
                    raise RuntimeError(
                        "MagCache profile length does not match the denoising trajectory: "
                        f"consumed {consumed_steps} steps, expected {len(self.ratios)}."
                    )

    def reset(self) -> None:
        """Drop cached activations while retaining request counters and profile output."""
        self._signature = None
        self._full_residuals = None
        self._previous_full_residuals = None
        self._pending_timestep = None
        self._pending_sync_plan = None
        self._reset_window()

    @torch.compiler.disable
    def should_reuse(
        self,
        block_inputs: TensorStreams,
        leader_block: object = None,
        timestep: torch.Tensor | None = None,
    ) -> bool:
        """Decide before block zero whether the cached full-stack residual may be reused."""
        if self._pending_timestep is not None or self._pending_sync_plan is not None:
            raise RuntimeError("Leo MagCache received a second decision before completing the previous step.")
        sync_plan = self._synchronization_plan(leader_block)
        if sync_plan is None:
            raise RuntimeError("MagCache cannot establish a CP/FSDP-consistent decision group.")
        sum_groups, max_groups = sync_plan
        decision_groups = [*sum_groups, *max_groups]
        reference = block_inputs[0]
        self._validate_distributed_config(reference, decision_groups)

        current_timestep = self._uniform_timestep(timestep, decision_groups)
        if current_timestep is None:
            raise RuntimeError("MagCache requires one finite timestep shared by every model-sharding rank.")
        step_index = self._step_index
        profile_ratio = 1.0
        if not self.calibrate:
            profile_ratio = self._profile_value(self.ratios, step_index, reference, decision_groups, "ratio")
            expected_timestep = self._profile_value(
                self.expected_timesteps,
                step_index,
                reference,
                decision_groups,
                "expected timestep",
            )
            if current_timestep != expected_timestep:
                raise RuntimeError(
                    f"MagCache timestep mismatch at step {step_index}: "
                    f"got {current_timestep}, expected {expected_timestep}."
                )

        self._pending_timestep = current_timestep
        self._pending_sync_plan = sync_plan
        self._step_index += 1

        signature = self._stream_signature(block_inputs)
        locally_valid = self._signature == signature and self._full_residuals is not None
        valid = reference.new_tensor(int(locally_valid), dtype=torch.int32)
        self._all_reduce(valid, decision_groups, dist.ReduceOp.MIN)
        if not bool(valid.item()):
            self._signature = signature
            self._full_residuals = None
            self._previous_full_residuals = None
            self._reset_window()
            return False
        if self.calibrate:
            return False

        retention_steps = int(len(self.ratios) * self.retention_ratio)
        if step_index < retention_steps:
            self._reset_window()
            return False

        accumulated_ratio = self._accumulated_ratio * profile_ratio
        accumulated_steps = self._accumulated_steps + 1
        accumulated_error = self._accumulated_error + abs(1.0 - accumulated_ratio)
        reuse = accumulated_error < self.threshold and accumulated_steps <= self.max_skip_steps
        reuse = self._consistent_decision(reuse, reference, decision_groups)
        if not reuse:
            self._reset_window()
            return False
        self._accumulated_ratio = accumulated_ratio
        self._accumulated_steps = accumulated_steps
        self._accumulated_error = accumulated_error
        self.skipped_steps += 1
        return True

    def apply_full(self, block_inputs: TensorStreams) -> TensorStreams:
        """Apply the cached residual contributed by the entire transformer stack."""
        if self._full_residuals is None:
            raise RuntimeError("Leo MagCache has no full-stack residual to apply.")
        self._validate_streams(block_inputs, self._full_residuals, "apply")
        outputs = tuple(
            None if block_input is None else block_input + residual
            for block_input, residual in zip(block_inputs, self._full_residuals)
        )
        self._pending_timestep = None
        self._pending_sync_plan = None
        return outputs

    def update_full(self, block_inputs: TensorStreams, final_outputs: TensorStreams) -> None:
        """Store one exact full-stack residual and record its calibration ratio."""
        if self._pending_timestep is None or self._pending_sync_plan is None:
            raise RuntimeError("Leo MagCache full update has no matching pre-block decision.")
        self._validate_streams(block_inputs, final_outputs, "update")
        next_residuals = tuple(
            None if block_input is None else (final_output - block_input).detach()
            for block_input, final_output in zip(block_inputs, final_outputs)
        )
        self.full_steps += 1
        if self.calibrate:
            if self._previous_full_residuals is None:
                ratio = 1.0
            else:
                sum_groups, max_groups = self._pending_sync_plan
                ratio = self._magnitude_ratio(
                    next_residuals[0],
                    self._previous_full_residuals[0],
                    sum_groups,
                    max_groups,
                )
            self.observed_ratios.append(ratio)
            self.observed_timesteps.append(self._pending_timestep)
            self._previous_full_residuals = next_residuals
        self._full_residuals = next_residuals
        self._record_cache_bytes()
        self._pending_timestep = None
        self._pending_sync_plan = None

    def stats(self) -> dict[str, object]:
        """Return counters and the calibrated or configured magnitude profile."""
        ratios = self.observed_ratios if self.calibrate else self.ratios
        timesteps = self.observed_timesteps if self.calibrate else self.expected_timesteps
        return {
            "method": self.method,
            "threshold": self.threshold,
            "full_steps": self.full_steps,
            "skipped_steps": self.skipped_steps,
            "magcache_calibrate": self.calibrate,
            "magcache_max_skip_steps": self.max_skip_steps,
            "magcache_retention_ratio": self.retention_ratio,
            "magcache_ratios": list(ratios),
            "magcache_expected_timesteps": list(timesteps),
            "cache_bytes": self._peak_cache_bytes,
        }

    def _record_cache_bytes(self) -> None:
        """Track the peak logical bytes held by persistent cache tensors."""
        current = _unique_tensor_bytes(self._full_residuals, self._previous_full_residuals)
        self._peak_cache_bytes = max(self._peak_cache_bytes, current)

    def _reset_window(self) -> None:
        """Reset accumulated approximation error after an exact step."""
        self._accumulated_error = 0.0
        self._accumulated_ratio = 1.0
        self._accumulated_steps = 0

    @classmethod
    def _profile_value(
        cls,
        values: tuple[float, ...],
        index: int,
        reference: torch.Tensor,
        groups: list[dist.ProcessGroup],
        label: str,
    ) -> float:
        """Read one profile value and reject exhaustion or cross-rank disagreement."""
        local_value = values[index] if index < len(values) else float("nan")
        value = cls._uniform_timestep(reference.new_tensor(local_value, dtype=torch.float32), groups)
        if value is None:
            raise RuntimeError(f"MagCache {label} is missing or differs across ranks at step {index}.")
        return value

    def _validate_distributed_config(
        self,
        reference: torch.Tensor,
        groups: list[dist.ProcessGroup],
    ) -> None:
        """Reject MagCache control values that differ across model-sharding ranks."""
        if self._distributed_config_validated:
            return
        values = (
            self.threshold,
            float(self.max_skip_steps),
            self.retention_ratio,
            float(len(self.ratios)),
            float(self.calibrate),
        )
        for value in values:
            shared = self._uniform_timestep(reference.new_tensor(value, dtype=torch.float32), groups)
            if shared is None:
                raise RuntimeError("MagCache control configuration differs across model-sharding ranks.")
        self._distributed_config_validated = True

    @classmethod
    def _consistent_decision(
        cls,
        decision: bool,
        reference: torch.Tensor,
        groups: list[dist.ProcessGroup],
    ) -> bool:
        """Reject a skip decision that differs across model-sharding ranks."""
        bounds = reference.new_tensor([int(decision), int(decision)], dtype=torch.int32)
        low = bounds[0].clone()
        high = bounds[1].clone()
        cls._all_reduce(low, groups, dist.ReduceOp.MIN)
        cls._all_reduce(high, groups, dist.ReduceOp.MAX)
        if int(low.item()) != int(high.item()):
            raise RuntimeError("MagCache skip decision differs across model-sharding ranks.")
        return bool(low.item())

    @classmethod
    def _magnitude_ratio(
        cls,
        current: torch.Tensor | None,
        previous: torch.Tensor | None,
        sum_groups: list[dist.ProcessGroup],
        max_groups: list[dist.ProcessGroup],
    ) -> float:
        """Compute the global visual full-stack residual magnitude ratio."""
        if current is None or previous is None:
            raise RuntimeError("MagCache calibration requires a visual residual on every step.")
        current_norm = current.float().norm(dim=-1)
        previous_norm = previous.float().norm(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
        local_ratios = current_norm / previous_norm
        values = torch.stack((local_ratios.sum(), local_ratios.new_tensor(local_ratios.numel())))
        cls._all_reduce(values, sum_groups, dist.ReduceOp.SUM)
        ratio = values[0] / values[1].clamp_min(1)
        cls._all_reduce(ratio, max_groups, dist.ReduceOp.MAX)
        ratio_value = float(ratio.item())
        if not math.isfinite(ratio_value) or ratio_value <= 0:
            raise RuntimeError(f"MagCache calibration produced an invalid ratio: {ratio_value}.")
        return ratio_value
