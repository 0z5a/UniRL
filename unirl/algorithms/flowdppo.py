"""FlowDPPO: KL-divergence-based masking for diffusion RL."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _gaussian_kl_div,
    _normalize_reference_loss_type,
    _reference_kl_loss,
    _reference_replay_means,
    _reference_velocity_loss,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    rollout_replay_logp_absdiff,
    typed_conditions,
)

logger = logging.getLogger(__name__)


@dataclass
class FlowDPPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    kl_mask_threshold: float = 1e-5
    add_kl_coefficient: bool = True
    beta: float = 0.0
    reference_loss_type: str = "transition_kl"
    old_logp_source: str = "rollout"
    max_rollout_replay_logp_absdiff: Optional[float] = None
    rollout_replay_parity_action: str = "raise"
    params: Any = dc_field(default=None)


def _flowdppo_kl_adv_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    advantages: torch.Tensor,
    sigma_t: torch.Tensor,
    kl_mask_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FlowDPPO KL-ADV masking loss; log-probs and advantages ``[B, S']``, means ``[B, S', *latent_shape]``."""
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped_loss = -adv * ratio

    kl_per_elem = _gaussian_kl_div(new_means, old_means, sigma_t)
    kl_per_sample = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))

    # KL mask: keep samples where KL < threshold (low divergence → safe to update)
    kl_mask = kl_per_sample < kl_mask_threshold

    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (adv > 0)
    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (adv < 0)
    rm_mask = pos_rm_mask | neg_rm_mask
    keep_adv_mask = (~rm_mask).detach()

    # Use torch.where for numerical safety: avoids inf * 0 = nan when ratio overflows
    zero = torch.zeros((), dtype=unclipped_loss.dtype, device=unclipped_loss.device)
    loss_per_elem = torch.where(keep_adv_mask, unclipped_loss, zero)

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
        "kl_new_old_mean": kl_per_sample.mean().detach(),
        "kl_new_old_max": kl_per_sample.max().detach(),
        "kl_mask_fraction": (~kl_mask).float().mean().detach(),
        "pos_rm_fraction": pos_rm_mask.float().mean().detach(),
        "neg_rm_fraction": neg_rm_mask.float().mean().detach(),
        "masked_fraction": rm_mask.float().mean().detach(),
        "unmasked_fraction": keep_adv_mask.float().mean().detach(),
    }
    return loss_per_elem, metrics


class FlowDPPO(StageAlgorithm):
    """FlowDPPO: KL-divergence-based masking for diffusion RL."""

    supports_multi_update = True
    requires_backend = True
    anchor_fields = ("sde_logp", "sde_means")

    def recomputes_anchor(self) -> bool:
        return True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        kl_mask_threshold: float = 1e-5,
        add_kl_coefficient: bool = True,
        beta: float = 0.0,
        reference_loss_type: str = "transition_kl",
        old_logp_source: str = "rollout",
        max_rollout_replay_logp_absdiff: Optional[float] = None,
        rollout_replay_parity_action: str = "raise",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("FlowDPPO: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.kl_mask_threshold = float(kl_mask_threshold)
        self.add_kl_coefficient = bool(add_kl_coefficient)
        self.beta = float(beta)
        self.reference_loss_type = _normalize_reference_loss_type(reference_loss_type, algo="FlowDPPO")
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowDPPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        self.max_rollout_replay_logp_absdiff = (
            None if max_rollout_replay_logp_absdiff is None else float(max_rollout_replay_logp_absdiff)
        )
        if self.max_rollout_replay_logp_absdiff is not None and (
            not math.isfinite(self.max_rollout_replay_logp_absdiff)
            or self.max_rollout_replay_logp_absdiff < 0
        ):
            raise ValueError("FlowDPPO.max_rollout_replay_logp_absdiff must be finite and non-negative")
        self.rollout_replay_parity_action = str(rollout_replay_parity_action).strip().lower()
        require(
            self.rollout_replay_parity_action in ("raise", "warn"),
            f"FlowDPPO.rollout_replay_parity_action must be 'raise' or 'warn'; got {rollout_replay_parity_action!r}",
        )
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowDPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Freeze the π_old log-prob and transition-mean anchors before optimizer updates."""
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout" and segment.sde_logp is None:
            raise RuntimeError(
                "FlowDPPO.prepare_segment: old_logp_source='rollout' but the "
                "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                "None). Pin a rollout build that emits trajectory log-probs, or set "
                "old_logp_source='replay'."
            )
        if self.old_logp_source == "rollout" and segment.sde_means is not None:
            if segment.sde_means.shape[1] != len(target_steps):
                raise RuntimeError(
                    "FlowDPPO.prepare_segment: rollout sde_means has "
                    f"{segment.sde_means.shape[1]} steps, expected {len(target_steps)}"
                )
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        logp_anchors = []
        mean_anchors = []
        with torch.no_grad():
            for step_index in target_steps:
                result = self.stage.replay(
                    typed_conds,
                    segment=segment,
                    params=self.params,
                    step_indices=[step_index],
                )
                if self.old_logp_source == "replay":
                    logp_anchors.append(result.log_probs.detach().cpu())
                if result.prev_sample_means is None:
                    raise RuntimeError(
                        "FlowDPPO.prepare_segment: stage.replay() returned "
                        "prev_sample_means=None. Ensure the stage's replay method "
                        "produces means (required for KL-ADV masking)."
                    )
                mean_anchors.append(result.prev_sample_means.detach().cpu())
        if self.old_logp_source == "replay":
            segment.sde_logp = torch.cat(logp_anchors, dim=1)
        segment.sde_means = torch.cat(mean_anchors, dim=1)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del training_progress
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
        new_logp = torch.cat(new_logps, dim=1)
        old_logp = torch.cat(old_logps, dim=1)
        log_diff = new_logp - old_logp
        ratio = torch.exp(log_diff)
        ratio_std = ratio.std() if ratio.numel() > 1 else torch.zeros((), dtype=ratio.dtype, device=ratio.device)
        metrics.update(
            {
                "ratio_mean": float(ratio.mean().item()),
                "ratio_std": float(ratio_std.item()),
                "ratio_min": float(ratio.min().item()),
                "ratio_max": float(ratio.max().item()),
                "approx_kl": float((0.5 * log_diff.pow(2)).mean().item()),
            }
        )
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
        loss_scale: float,
        step_index: int,
    ) -> tuple[AlgorithmStepResult, torch.Tensor, torch.Tensor]:
        """Backpropagate one transition and return detached log-probs and scalar diagnostics."""
        target_steps = [step_index]
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means
        if new_means is None:
            raise RuntimeError(
                "FlowDPPO requires stage.replay() to return prev_sample_means, "
                "but got None. Ensure the stage's replay method produces means."
            )

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        old_means = gather_sde_field(
            segment.sde_means,
            segment.sde_indices,
            target_steps,
            field_name="sde_means",
        ).to(dtype=new_means.dtype, device=new_means.device)
        sigma_t = self._compute_sigma_t(segment, target_steps, device=new_logp.device)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)

        drift_metrics = rollout_replay_logp_absdiff(new_logp, old_logp)
        step_drift = rollout_replay_logp_absdiff(new_logp[:, 0], old_logp[:, 0])
        drift_metrics[f"rollout_replay_logp_absdiff_step_{step_index}_mean"] = step_drift[
            "rollout_replay_logp_absdiff_mean"
        ]
        drift_metrics[f"rollout_replay_logp_absdiff_step_{step_index}_max"] = step_drift[
            "rollout_replay_logp_absdiff_max"
        ]
        with torch.no_grad():
            replay_means = new_means.detach().to(torch.float32)
            anchor_means = old_means.detach().to(torch.float32)
            mean_absdiff = (replay_means - anchor_means).abs()
            drift_metrics["rollout_replay_transition_mean_absdiff_mean"] = float(mean_absdiff.mean().item())
            drift_metrics["rollout_replay_transition_mean_absdiff_max"] = float(mean_absdiff.max().item())
            step_mean_absdiff = mean_absdiff[:, 0]
            drift_metrics[f"rollout_replay_transition_mean_absdiff_step_{step_index}_mean"] = float(
                step_mean_absdiff.mean().item()
            )
            drift_metrics[f"rollout_replay_transition_mean_absdiff_step_{step_index}_max"] = float(
                step_mean_absdiff.max().item()
            )
            is_cps = getattr(getattr(self.stage, "strategy", None), "canonical_name", None) == "cps"
            if is_cps:
                sigmas = segment.sigmas.to(device=new_means.device, dtype=torch.float32)
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
                if float(velocity_coefficient.abs().item()) > torch.finfo(torch.float32).eps:
                    sample = segment.latents_at(step_index).to(device=new_means.device, dtype=torch.float32)
                    rollout_velocity = -(anchor_means[:, 0] - sample * sample_coefficient) / velocity_coefficient
                    replay_velocity = -(replay_means[:, 0] - sample * sample_coefficient) / velocity_coefficient
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
            velocity_detail = ""
            velocity_key = f"rollout_replay_velocity_step_{step_index}_diff_rms"
            if velocity_key in drift_metrics:
                velocity_detail = (
                    "; velocity diff_rms/relative_rms="
                    f"{drift_metrics[velocity_key]:.6g}/"
                    f"{drift_metrics[f'rollout_replay_velocity_step_{step_index}_relative_rms']:.6g}"
                )
            message = (
                "FlowDPPO rollout/replay parity failed: "
                f"mean/max |Δlogp|={drift_mean:.6g}/{drift_max:.6g}; expected finite values and "
                f"max <= {self.max_rollout_replay_logp_absdiff:.6g}; "
                f"step {step_index} mean/max |Δlogp|="
                f"{drift_metrics[f'rollout_replay_logp_absdiff_step_{step_index}_mean']:.6g}/"
                f"{drift_metrics[f'rollout_replay_logp_absdiff_step_{step_index}_max']:.6g}"
                + velocity_detail
            )
            if self.rollout_replay_parity_action == "raise":
                raise RuntimeError(message)
            logger.warning(message)

        loss_per_elem, ratio_metrics = _flowdppo_kl_adv_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            new_means=new_means,
            old_means=old_means,
            advantages=adv_b,
            sigma_t=sigma_t,
            kl_mask_threshold=self.kl_mask_threshold,
        )
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "kl_mask_threshold": float(self.kl_mask_threshold),
            **drift_metrics,
            **{key: float(value.item()) for key, value in ratio_metrics.items()},
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
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_sigma_t = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=target_steps,
                eta=float(self.params.eta),
                device=new_logp.device,
                add_coefficient=True,
            )
            kl_ref = _reference_kl_loss(new_means, ref_means, kl_sigma_t)
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

    def _compute_sigma_t(
        self,
        segment: "LatentSegment",
        target_steps: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Per-step KL-normalization sigma_t ``[1, S', 1, 1, 1]``; ones when ``add_kl_coefficient=False``."""
        return _transition_sigma(
            self.stage,
            segment=segment,
            target_steps=target_steps,
            eta=float(self.params.eta),
            device=device,
            add_coefficient=self.add_kl_coefficient,
        )


__all__ = ["FlowDPPO", "FlowDPPOConfig"]
