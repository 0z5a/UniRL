from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.algorithms.base import AlgorithmStepResult
from unirl.algorithms.flowdppo import FlowDPPO, _flowdppo_kl_adv_loss
from unirl.types.segments.latent import LatentSegment


class _ReplayStage:
    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.logp_shift = 0.0
        self.calls: list[tuple[int, ...]] = []
        self.strategy = SimpleNamespace(transition_std=lambda **_kwargs: torch.tensor(1.0))

    def replay(self, _conditions, *, segment, params, step_indices):
        del params
        self.calls.append(tuple(step_indices))
        batch_size = segment.sde_logp.shape[0]
        step_values = torch.stack([self.weight + 0.1 * step for step in step_indices])
        log_probs = (step_values + self.logp_shift).reshape(1, -1).expand(batch_size, -1)
        means = step_values.reshape(1, -1, 1, 1, 1).expand(batch_size, -1, 1, 1, 1)
        return SimpleNamespace(log_probs=log_probs, prev_sample_means=means)


class _EquivalenceStage:
    def __init__(self, weight: float) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(weight))
        self.logp_base = torch.tensor([[0.10, -0.20], [-0.10, 0.20]])
        self.logp_coefficient = torch.tensor([[0.50, -0.25], [0.40, 0.30]])
        self.mean_base = torch.tensor([[0.00, 0.30], [0.18, 0.00]])
        self.mean_coefficient = torch.tensor([[0.20, -0.10], [0.10, 0.25]])
        self.strategy = SimpleNamespace(transition_std=lambda **kwargs: torch.full_like(kwargs["sigma"], 0.5))
        self.calls: list[tuple[int, ...]] = []

    def values(self, step_indices: list[int], *, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = [0 if step == 0 else 1 for step in step_indices]
        log_probs = self.logp_base[:, positions] + weight * self.logp_coefficient[:, positions]
        means = self.mean_base[:, positions] + weight * self.mean_coefficient[:, positions]
        return log_probs, means[..., None, None, None]

    def replay(self, _conditions, *, segment, params, step_indices):
        del segment, params
        self.calls.append(tuple(step_indices))
        log_probs, means = self.values(step_indices, weight=self.weight)
        return SimpleNamespace(log_probs=log_probs, prev_sample_means=means)


def _segment() -> LatentSegment:
    return LatentSegment(
        sde_indices=torch.tensor([0, 2]),
        sde_logp=torch.tensor([[0.0, 0.2]]),
        sde_means=torch.tensor([0.0, 0.2]).reshape(1, 2, 1, 1, 1),
    )


def _algorithm(stage: _ReplayStage, **kwargs) -> FlowDPPO:
    return FlowDPPO(
        params=SimpleNamespace(eta=0.5),
        stage=stage,
        add_kl_coefficient=False,
        kl_mask_threshold=1.0,
        **kwargs,
    )


def _compute(algorithm: FlowDPPO, segment: LatentSegment, *, loss_scale: float = 1.0) -> AlgorithmStepResult:
    return algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.ones(segment.sde_logp.shape[0]),
        training_progress=0.0,
        loss_scale=loss_scale,
    )


def test_flowdppo_replays_and_backpropagates_one_timestep_at_a_time() -> None:
    stage = _ReplayStage()
    result = _compute(_algorithm(stage), _segment(), loss_scale=2.0)

    ratios = torch.exp(torch.tensor([0.0, 0.0]))
    assert stage.calls == [(0,), (2,)]
    assert stage.weight.grad is not None
    assert stage.weight.grad.item() == pytest.approx(-2.0)
    assert result.loss == pytest.approx(-1.0)
    assert result.metrics["ratio_mean"] == pytest.approx(ratios.mean().item())
    assert result.metrics["ratio_std"] == pytest.approx(ratios.std().item())
    assert result.metrics["kl_new_old_max"] == pytest.approx(0.0)
    assert "rollout_replay_logp_absdiff_step_0_max" in result.metrics
    assert "rollout_replay_logp_absdiff_step_2_max" in result.metrics


def test_flowdppo_per_step_loss_and_gradient_match_batched_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _EquivalenceStage(weight=0.2)
    segment = LatentSegment(
        sde_indices=torch.tensor([0, 2]),
        sde_logp=torch.zeros(2, 2),
        sde_means=torch.zeros(2, 2, 1, 1, 1),
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4]),
    )
    advantages = torch.tensor([1.0, -0.5])
    reference_means = torch.tensor([[0.05, 0.10], [0.00, -0.05]])[..., None, None, None]

    def reference_replay_means(_stage, _model, *, target_steps, **_kwargs):
        positions = [0 if step == 0 else 1 for step in target_steps]
        return reference_means[:, positions]

    monkeypatch.setattr("unirl.algorithms.flowdppo._reference_replay_means", reference_replay_means)
    algorithm = FlowDPPO(
        params=SimpleNamespace(eta=0.5),
        stage=stage,
        add_kl_coefficient=False,
        kl_mask_threshold=0.01,
    )
    algorithm.beta = 0.3
    algorithm._ref_model = object()
    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.7,
    )

    batched_weight = torch.tensor(0.2, requires_grad=True)
    new_logp, new_means = stage.values([0, 2], weight=batched_weight)
    loss_per_elem, expected_metrics = _flowdppo_kl_adv_loss(
        new_logp=new_logp,
        old_logp=segment.sde_logp,
        new_means=new_means,
        old_means=segment.sde_means,
        advantages=advantages[:, None].expand_as(new_logp),
        sigma_t=torch.ones(1, 2, 1, 1, 1),
        kl_mask_threshold=0.01,
    )
    kl_ref = ((new_means - reference_means) ** 2 / (2 * 0.5**2)).mean()
    expected_loss = loss_per_elem.mean() + 0.3 * kl_ref
    (expected_loss * 1.7).backward()

    assert stage.calls == [(0,), (2,)]
    assert stage.weight.grad is not None
    assert batched_weight.grad is not None
    torch.testing.assert_close(stage.weight.grad, batched_weight.grad)
    assert result.loss == pytest.approx(expected_loss.detach().item())
    assert result.metrics["masked_fraction"] == pytest.approx(expected_metrics["masked_fraction"].item())
    assert result.metrics["kl_new_old_mean"] == pytest.approx(expected_metrics["kl_new_old_mean"].item())
    assert result.metrics["kl_ref_mean"] == pytest.approx(kl_ref.detach().item())


def test_flowdppo_prepare_segment_uses_matching_single_timestep_geometry() -> None:
    stage = _ReplayStage()
    stage.weight.data.fill_(0.4)
    segment = _segment()
    algorithm = _algorithm(stage, old_logp_source="replay")

    algorithm.prepare_segment(conditions={}, segment=segment)

    assert stage.calls == [(0,), (2,)]
    torch.testing.assert_close(segment.sde_logp, torch.tensor([[0.4, 0.6]]))
    torch.testing.assert_close(segment.sde_means.flatten(), torch.tensor([0.4, 0.6]))


def test_flowdppo_prepare_segment_reuses_rollout_anchors() -> None:
    stage = _ReplayStage()
    segment = _segment()
    logp = segment.sde_logp.clone()
    means = segment.sde_means.clone()

    _algorithm(stage, old_logp_source="rollout").prepare_segment(conditions={}, segment=segment)

    assert stage.calls == []
    torch.testing.assert_close(segment.sde_logp, logp)
    torch.testing.assert_close(segment.sde_means, means)


def test_flowdppo_prepare_segment_replays_only_missing_rollout_means() -> None:
    stage = _ReplayStage()
    stage.weight.data.fill_(0.4)
    segment = _segment()
    rollout_logp = segment.sde_logp.clone()
    segment.sde_means = None

    _algorithm(stage, old_logp_source="rollout").prepare_segment(conditions={}, segment=segment)

    assert stage.calls == [(0,), (2,)]
    torch.testing.assert_close(segment.sde_logp, rollout_logp)
    torch.testing.assert_close(segment.sde_means.flatten(), torch.tensor([0.4, 0.6]))


def test_flowdppo_parity_gate_raises_only_on_first_optimizer_update() -> None:
    stage = _ReplayStage()
    algorithm = _algorithm(
        stage,
        old_logp_source="rollout",
        max_rollout_replay_logp_absdiff=0.1,
        rollout_replay_parity_action="raise",
    )
    segment = _segment()

    stage.logp_shift = 0.05
    algorithm.begin_optimizer_update(update_index=0)
    first = _compute(algorithm, segment)
    assert first.metrics["rollout_replay_parity_gate_active"] == 1.0

    stage.logp_shift = 0.25
    algorithm.begin_optimizer_update(update_index=1)
    later = _compute(algorithm, segment)
    assert later.metrics["rollout_replay_parity_gate_active"] == 0.0
    assert later.metrics["rollout_replay_logp_absdiff_max"] == pytest.approx(0.25)

    algorithm.begin_optimizer_update(update_index=0)
    with pytest.raises(RuntimeError, match="FlowDPPO rollout/replay parity failed"):
        _compute(algorithm, segment)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_flowdppo_parity_threshold_must_be_finite_and_non_negative(threshold: float) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _algorithm(_ReplayStage(), max_rollout_replay_logp_absdiff=threshold)


def test_flowdppo_parity_action_is_validated() -> None:
    with pytest.raises(ValueError, match="must be 'raise' or 'warn'"):
        _algorithm(_ReplayStage(), rollout_replay_parity_action="ignore")


@pytest.mark.parametrize("logp", [float("nan"), float("inf")])
@pytest.mark.parametrize("action", ["raise", "warn"])
def test_flowdppo_first_update_parity_gate_fails_closed_on_non_finite_drift(
    logp: float,
    action: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stage = _ReplayStage()
    stage.logp_shift = logp
    algorithm = _algorithm(
        stage,
        max_rollout_replay_logp_absdiff=0.1,
        rollout_replay_parity_action=action,
    )
    algorithm.begin_optimizer_update(update_index=0)

    if action == "raise":
        with pytest.raises(RuntimeError, match="FlowDPPO rollout/replay parity failed"):
            _compute(algorithm, _segment())
    else:
        with caplog.at_level("WARNING", logger="unirl.algorithms.flowdppo"):
            _compute(algorithm, _segment())
        assert "FlowDPPO rollout/replay parity failed" in caplog.text
