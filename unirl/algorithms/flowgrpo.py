"""Stage-driven ``FlowGRPO`` over a ``LatentSegment``."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _normalize_reference_loss_type,
    _reference_kl_loss,
    _reference_replay_means,
    _reference_velocity_loss,
    _resolve_clip_range_from_schedule,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    rollout_replay_logp_absdiff,
    typed_conditions,
)

logger = logging.getLogger(__name__)


@dataclass
class FlowGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    beta: float = 0.0
    reference_loss_type: str = "transition_kl"
    old_logp_source: str = "rollout"
    max_rollout_replay_logp_absdiff: Optional[float] = None
    rollout_replay_parity_action: str = "raise"
    use_grpo_guard: bool = False
    adv_clip_max: Optional[float] = None
    params: Any = dc_field(default=None)


class FlowGRPO(StageAlgorithm):
    """GRPO over a diffusion ``LatentSegment`` via ``DiffusionStage.replay``."""

    supports_multi_update = True
    requires_backend = True
    anchor_fields = ("sde_logp",)

    def recomputes_anchor(self) -> bool:
        return self.old_logp_source == "replay"

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        beta: float = 0.0,
        reference_loss_type: str = "transition_kl",
        old_logp_source: str = "rollout",
        max_rollout_replay_logp_absdiff: Optional[float] = None,
        rollout_replay_parity_action: str = "raise",
        use_grpo_guard: bool = False,
        adv_clip_max: Optional[float] = None,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("FlowGRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.params = params
        self.clip_range = float(clip_range)
        self.clip_schedule = str(clip_schedule)
        self.beta = float(beta)
        self.reference_loss_type = _normalize_reference_loss_type(reference_loss_type, algo="FlowGRPO")
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowGRPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        self.max_rollout_replay_logp_absdiff = (
            None if max_rollout_replay_logp_absdiff is None else float(max_rollout_replay_logp_absdiff)
        )
        if self.max_rollout_replay_logp_absdiff is not None and (
            not math.isfinite(self.max_rollout_replay_logp_absdiff)
            or self.max_rollout_replay_logp_absdiff < 0
        ):
            raise ValueError("FlowGRPO.max_rollout_replay_logp_absdiff must be finite and non-negative")
        self.rollout_replay_parity_action = str(rollout_replay_parity_action).strip().lower()
        if type(use_grpo_guard) is not bool:
            raise TypeError(f"FlowGRPO.use_grpo_guard must be bool, got {type(use_grpo_guard).__name__}.")
        self.use_grpo_guard = use_grpo_guard
        self.adv_clip_max = None if adv_clip_max is None else float(adv_clip_max)
        if self.adv_clip_max is not None and (
            not math.isfinite(self.adv_clip_max) or self.adv_clip_max <= 0
        ):
            raise ValueError("FlowGRPO.adv_clip_max must be finite and positive")
        require(
            self.rollout_replay_parity_action in ("raise", "warn"),
            f"FlowGRPO.rollout_replay_parity_action must be 'raise' or 'warn'; got {rollout_replay_parity_action!r}",
        )
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowGRPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Establish the frozen π_old anchor (``segment.sde_logp``) before the ``num_updates_per_batch`` loop."""
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout":
            if segment.sde_logp is None:
                raise RuntimeError(
                    "FlowGRPO.prepare_segment: old_logp_source='rollout' but the "
                    "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                    "None). Pin a rollout build that emits trajectory log-probs, or set "
                    "old_logp_source='replay'."
                )
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        anchors = []
        with torch.no_grad():
            for step_index in target_steps:
                result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=[step_index])
                anchors.append(result.log_probs.detach().cpu())
        segment.sde_logp = torch.cat(anchors, dim=1)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)

        step_results = []
        new_logps = []
        old_logps = []
        for step_index in target_steps:
            result, new_logp, old_logp = self._compute_step_and_backward(
                typed_conds=typed_conds,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=float(loss_scale) / len(target_steps),
                step_index=step_index,
            )
            step_results.append(result)
            new_logps.append(new_logp)
            old_logps.append(old_logp)

        metrics: Dict[str, Any] = {}
        for result in step_results:
            for key, value in result.metrics.items():
                if "_step_" in key:
                    metrics[key] = value
                elif key.endswith("_max"):
                    metrics[key] = max(metrics.get(key, value), value)
                elif key.endswith("_min"):
                    metrics[key] = min(metrics.get(key, value), value)
                else:
                    metrics[key] = metrics.get(key, 0.0) + value / len(target_steps)
        if not self.use_grpo_guard:
            new_logp = torch.cat(new_logps, dim=1)
            old_logp = torch.cat(old_logps, dim=1)
            _, ratio_metrics = _grpo_clip_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=advantages.detach().to(new_logp).reshape(-1, 1).expand_as(new_logp),
                clip_range=metrics["clip_range"],
            )
            metrics.update({key: float(value.item()) for key, value in ratio_metrics.items()})
        return AlgorithmStepResult(
            loss=sum(result.loss for result in step_results) / len(target_steps),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _compute_step_and_backward(
        self,
        *,
        typed_conds: Any,
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
        step_index: int,
    ) -> tuple[AlgorithmStepResult, torch.Tensor, torch.Tensor]:
        """Backpropagate one transition and return only detached log-probs and scalar diagnostics."""
        target_steps = [step_index]
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
        if self.adv_clip_max is not None:
            adv_b = torch.clamp(adv_b, -self.adv_clip_max, self.adv_clip_max)
        drift_metrics = rollout_replay_logp_absdiff(new_logp, old_logp)
        old_means = None
        for column, step_index in enumerate(target_steps):
            step_drift = rollout_replay_logp_absdiff(new_logp[:, column], old_logp[:, column])
            drift_metrics[f"rollout_replay_logp_absdiff_step_{step_index}_mean"] = step_drift[
                "rollout_replay_logp_absdiff_mean"
            ]
            drift_metrics[f"rollout_replay_logp_absdiff_step_{step_index}_max"] = step_drift[
                "rollout_replay_logp_absdiff_max"
            ]
        if segment.sde_means is not None and new_means is not None:
            old_means = gather_sde_field(
                segment.sde_means,
                segment.sde_indices,
                target_steps,
                field_name="sde_means",
            ).to(dtype=torch.float32, device=new_means.device)
            with torch.no_grad():
                replay_means = new_means.detach().to(torch.float32)
                mean_absdiff = (replay_means - old_means).abs()
                drift_metrics["rollout_replay_transition_mean_absdiff_mean"] = float(mean_absdiff.mean().item())
                drift_metrics["rollout_replay_transition_mean_absdiff_max"] = float(mean_absdiff.max().item())
                is_cps = getattr(getattr(self.stage, "strategy", None), "canonical_name", None) == "cps"
                sigmas = segment.sigmas.to(device=new_means.device, dtype=torch.float32) if is_cps else None
                for column, step_index in enumerate(target_steps):
                    step_mean_absdiff = mean_absdiff[:, column]
                    drift_metrics[f"rollout_replay_transition_mean_absdiff_step_{step_index}_mean"] = float(
                        step_mean_absdiff.mean().item()
                    )
                    drift_metrics[f"rollout_replay_transition_mean_absdiff_step_{step_index}_max"] = float(
                        step_mean_absdiff.max().item()
                    )
                    if sigmas is None:
                        continue
                    sigma = sigmas[step_index]
                    sigma_next = sigmas[step_index + 1]
                    transition_std = self.stage.strategy.transition_std(
                        sigma=sigma,
                        sigma_next=sigma_next,
                        eta=float(self.params.eta),
                    ).to(torch.float32)
                    root = torch.sqrt(torch.clamp(sigma_next.square() - transition_std.square(), min=0.0))
                    sample_coefficient = (1 - sigma_next) + root
                    velocity_coefficient = -sigma * (1 - sigma_next) + (1 - sigma) * root
                    if float(velocity_coefficient.abs().item()) <= torch.finfo(torch.float32).eps:
                        continue
                    sample = segment.latents_at(step_index).to(device=new_means.device, dtype=torch.float32)
                    rollout_velocity = -(old_means[:, column] - sample * sample_coefficient) / velocity_coefficient
                    replay_velocity = -(replay_means[:, column] - sample * sample_coefficient) / velocity_coefficient
                    velocity_diff = replay_velocity - rollout_velocity
                    rollout_rms = torch.sqrt(rollout_velocity.square().mean())
                    replay_rms = torch.sqrt(replay_velocity.square().mean())
                    diff_rms = torch.sqrt(velocity_diff.square().mean())
                    prefix = f"rollout_replay_velocity_step_{step_index}"
                    drift_metrics[f"{prefix}_absdiff_mean"] = float(velocity_diff.abs().mean().item())
                    drift_metrics[f"{prefix}_absdiff_max"] = float(velocity_diff.abs().max().item())
                    drift_metrics[f"{prefix}_diff_rms"] = float(diff_rms.item())
                    drift_metrics[f"{prefix}_rollout_rms"] = float(rollout_rms.item())
                    drift_metrics[f"{prefix}_replay_rms"] = float(replay_rms.item())
                    drift_metrics[f"{prefix}_relative_rms"] = float(
                        (diff_rms / rollout_rms.clamp_min(torch.finfo(torch.float32).eps)).item()
                    )
        parity_gate_active = self.max_rollout_replay_logp_absdiff is not None and self.is_first_optimizer_update
        drift_metrics["rollout_replay_parity_gate_active"] = float(parity_gate_active)
        drift_mean = drift_metrics["rollout_replay_logp_absdiff_mean"]
        drift_max = drift_metrics["rollout_replay_logp_absdiff_max"]
        if (
            parity_gate_active
            and (
                not math.isfinite(drift_mean)
                or not math.isfinite(drift_max)
                or drift_max > self.max_rollout_replay_logp_absdiff
            )
        ):
            per_step = ", ".join(
                (
                    f"{step}:"
                    f"{drift_metrics[f'rollout_replay_logp_absdiff_step_{step}_mean']:.6g}/"
                    f"{drift_metrics[f'rollout_replay_logp_absdiff_step_{step}_max']:.6g}"
                )
                for step in target_steps
            )
            velocity_detail = ", ".join(
                (
                    f"{step}:"
                    f"{drift_metrics[f'rollout_replay_velocity_step_{step}_diff_rms']:.6g}/"
                    f"{drift_metrics[f'rollout_replay_velocity_step_{step}_relative_rms']:.6g}"
                )
                for step in target_steps
                if f"rollout_replay_velocity_step_{step}_diff_rms" in drift_metrics
            )
            message = (
                "FlowGRPO rollout/replay parity failed: "
                f"mean/max |Δlogp|={drift_mean:.6g}/{drift_max:.6g}; expected finite values and "
                f"max <= {self.max_rollout_replay_logp_absdiff:.6g}; "
                f"per-step mean/max |Δlogp|=[{per_step}]"
                + (f"; per-step velocity diff_rms/relative_rms=[{velocity_detail}]" if velocity_detail else "")
            )
            if self.rollout_replay_parity_action == "raise":
                raise RuntimeError(message)
            logger.warning(message)

        if self.use_grpo_guard:
            if new_means is None or old_means is None:
                raise RuntimeError("FlowGRPO.use_grpo_guard requires rollout and replay transition means.")
            sigmas = segment.sigmas.to(device=new_logp.device, dtype=torch.float32)
            sigma = sigmas[step_index]
            sigma_next = sigmas[step_index + 1]
            transition_std = self.stage.strategy.transition_std(
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(self.params.eta),
                sigma_max=sigmas[1] if int(sigmas.shape[0]) > 1 else 0.99,
            ).to(new_logp.dtype)
            jacobian = torch.abs(
                (sigma_next - sigma)
                - 0.5 * transition_std.square() * ((1 - sigma) / sigma)
            ).to(new_logp.dtype)
            ratio_coefficient = transition_std.clamp(min=1e-6)
            loss_coefficient = jacobian.reciprocal()
            mean_bias = (new_means - old_means).square().mean(
                dim=tuple(range(2, new_means.ndim))
            )
            mean_bias = mean_bias / (2 * ratio_coefficient.square())
            log_diff = new_logp - old_logp
            ratio = torch.exp((log_diff + mean_bias) * ratio_coefficient)
            unclipped = -adv_b * ratio
            clipped = -adv_b * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            loss_per_elem = torch.maximum(unclipped, clipped) * loss_coefficient
            gt = (ratio - 1.0 > clip_range).float()
            lt = (1.0 - ratio > clip_range).float()
            ratio_metrics = {
                "ratio_mean": ratio.mean().detach(),
                "ratio_std": (
                    ratio.std()
                    if ratio.numel() > 1
                    else torch.zeros((), dtype=ratio.dtype, device=ratio.device)
                ).detach(),
                "ratio_min": ratio.min().detach(),
                "ratio_max": ratio.max().detach(),
                "clip_fraction": torch.maximum(gt, lt).mean().detach(),
                "clipfrac_gt_one": gt.mean().detach(),
                "clipfrac_lt_one": lt.mean().detach(),
                "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
            }
        else:
            loss_per_elem, ratio_metrics = _grpo_clip_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_b,
                clip_range=clip_range,
            )
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "clip_range": float(clip_range),
            **drift_metrics,
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }

        if self.beta > 0.0 and self.reference_loss_type == "velocity_mse":
            reference_loss, reference_metrics = _reference_velocity_loss(
                replay_result=replay_result,
                stage=self.stage,
                ref_model=self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            )
            loss = loss + self.beta * reference_loss
            metrics.update(reference_metrics)
            metrics["beta"] = float(self.beta)
            metrics["reference_loss"] = float(reference_loss.detach().item())
        elif self.beta > 0.0:
            if new_means is None:
                raise RuntimeError(
                    "FlowGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
                    "but got None. Ensure the stage's replay method produces means."
                )
            sigma_t = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=target_steps,
                eta=float(self.params.eta),
                device=new_logp.device,
                add_coefficient=True,
            )
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_ref = (
                (new_means - ref_means).square().mean(dim=tuple(range(2, new_means.ndim))).mean()
                if self.use_grpo_guard
                else _reference_kl_loss(new_means, ref_means, sigma_t)
            )
            loss = loss + self.beta * kl_ref
            metrics["beta"] = float(self.beta)
            metrics["kl_ref_mean"] = float(kl_ref.detach().item())

        (loss * loss_scale).backward()

        result = AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=1,
            has_backward=True,
        )
        return result, new_logp.detach(), old_logp.detach()

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment."""
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]


__all__ = ["FlowGRPO", "FlowGRPOConfig"]
