from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm
from unirl.algorithms.flowgrpo import FlowGRPO
from unirl.train.stack.base import TrainStack, TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack
from unirl.types.sample import Part
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.latent import LatentSegment


class _DriftingReplayStage:
    def __init__(self, logp: float) -> None:
        self.logp = float(logp)

    def replay(self, _conditions, *, segment, params, step_indices):
        del params
        return SimpleNamespace(
            log_probs=torch.full(
                (segment.sde_logp.shape[0], len(step_indices)),
                self.logp,
                dtype=torch.float32,
                requires_grad=True,
            ),
            prev_sample_means=None,
        )


def _compute(algorithm: FlowGRPO, segment: LatentSegment) -> AlgorithmStepResult:
    return algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.ones(segment.sde_logp.shape[0]),
        training_progress=0.0,
        loss_scale=1.0,
    )


def test_flowgrpo_parity_gate_raises_only_on_first_optimizer_update() -> None:
    stage = _DriftingReplayStage(logp=0.05)
    algorithm = FlowGRPO(
        params=SimpleNamespace(),
        stage=stage,
        old_logp_source="rollout",
        max_rollout_replay_logp_absdiff=0.1,
        rollout_replay_parity_action="raise",
    )
    segment = LatentSegment(
        sde_indices=torch.tensor([0]),
        sde_logp=torch.zeros(1, 1),
    )

    algorithm.begin_optimizer_update(update_index=0)
    first = _compute(algorithm, segment)
    assert first.metrics["rollout_replay_parity_gate_active"] == 1.0

    stage.logp = 0.25
    algorithm.begin_optimizer_update(update_index=1)
    later = _compute(algorithm, segment)
    assert later.metrics["rollout_replay_parity_gate_active"] == 0.0
    assert later.metrics["rollout_replay_logp_absdiff_max"] == pytest.approx(0.25)

    algorithm.begin_optimizer_update(update_index=0)
    with pytest.raises(RuntimeError, match="FlowGRPO rollout/replay parity failed"):
        _compute(algorithm, segment)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_flowgrpo_parity_threshold_must_be_finite_and_non_negative(threshold: float) -> None:
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        FlowGRPO(
            params=SimpleNamespace(),
            stage=_DriftingReplayStage(logp=0.0),
            max_rollout_replay_logp_absdiff=threshold,
        )


@pytest.mark.parametrize("logp", [float("nan"), float("inf")])
@pytest.mark.parametrize("action", ["raise", "warn"])
def test_flowgrpo_first_update_parity_gate_fails_closed_on_non_finite_drift(
    logp: float,
    action: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    algorithm = FlowGRPO(
        params=SimpleNamespace(),
        stage=_DriftingReplayStage(logp=logp),
        old_logp_source="rollout",
        max_rollout_replay_logp_absdiff=0.1,
        rollout_replay_parity_action=action,
    )
    segment = LatentSegment(
        sde_indices=torch.tensor([0]),
        sde_logp=torch.zeros(1, 1),
    )
    algorithm.begin_optimizer_update(update_index=0)

    if action == "raise":
        with pytest.raises(RuntimeError, match="FlowGRPO rollout/replay parity failed"):
            _compute(algorithm, segment)
    else:
        with caplog.at_level("WARNING", logger="unirl.algorithms.flowgrpo"):
            _compute(algorithm, segment)
        assert "FlowGRPO rollout/replay parity failed" in caplog.text


class _RecordingAlgorithm(StageAlgorithm):
    supports_multi_update = True

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[int] = []

    def compute_loss_and_backward(self, **_kwargs) -> AlgorithmStepResult:
        self.observed.append(self.optimizer_update_index)
        return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=1, has_backward=True)


def test_train_stack_marks_every_micro_of_only_the_first_update() -> None:
    algorithm = _RecordingAlgorithm()
    backend = SimpleNamespace(
        _rank=0,
        zero_grad=lambda: None,
        set_grad_sync=lambda _enabled: None,
        grad_sync_deferred=False,
        optimizer_step=lambda *, max_grad_norm: 0.0,
        all_reduce_loss_sums=lambda values: values,
        gradient_average_world_size=lambda: 1,
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
        scheduler=None,
    )
    stack = TrainStack(
        fsdp_backend=backend,
        algorithm=algorithm,
        micro_batch_size=1,
        max_grad_norm=1.0,
        num_updates_per_batch=2,
    )
    part = Part(sample_ids=["0", "1", "2", "3"], advantages=torch.ones(4))
    plans = [[(0, 1), (1, 2)], [(2, 3), (3, 4)]]

    stack._run_updates(part, plans=plans, training_progress=0.0)
    stack._run_updates(part, plans=plans, training_progress=0.0)

    assert algorithm.observed == [0, 0, 1, 1, 0, 0, 1, 1]


def test_unified_model_stack_aligns_update_index_for_both_algorithms() -> None:
    ar_algorithm = _RecordingAlgorithm()
    image_algorithm = _RecordingAlgorithm()
    stack = object.__new__(UnifiedModelTrainStack)
    stack.ar_algorithm = ar_algorithm
    stack.image_algorithm = image_algorithm
    stack.fsdp_backend = SimpleNamespace(_device=torch.device("cpu"))
    stack.micro_batch_size = 1
    stack.num_updates_per_batch = 2
    observed: list[tuple[int, int]] = []

    stack.prepare_segment = MethodType(lambda self, algorithm, part: None, stack)
    stack.on_rollout_end = MethodType(lambda self: None, stack)

    def train_one_step(
        self,
        ar_part,
        image_part,
        *,
        ar_slices,
        image_slices,
        training_progress,
    ):
        del self, ar_part, image_part, ar_slices, image_slices, training_progress
        observed.append((ar_algorithm.optimizer_update_index, image_algorithm.optimizer_update_index))
        result = TrainStepResult(
            loss=0.0,
            grad_norm=0.0,
            lr=0.0,
            has_backward=True,
            micros=[],
            metrics={},
            optimizer_updates=1,
        )
        return {"ar": result, "image": result}

    stack._train_one_step = MethodType(train_one_step, stack)
    ar_part = Part(sample_ids=["ar0", "ar1"], advantages=torch.ones(2))
    image_part = Part(sample_ids=["image0", "image1"], advantages=torch.ones(2))
    sample = SimpleNamespace(
        gen_part=lambda params_cls: ar_part if params_cls is ARSamplingParams else image_part,
    )

    stack.train_track(sample, training_progress=0.0)

    assert observed == [(0, 0), (1, 1)]
