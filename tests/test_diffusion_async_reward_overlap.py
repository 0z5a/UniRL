from __future__ import annotations

from types import SimpleNamespace

from unirl.trainer import diffusion as diffusion_module
from unirl.trainer.async_diffusion import AsyncDiffusionTrainer
from unirl.trainer.diffusion import DiffusionTrainer


def test_async_diffusion_refills_rollout_before_reward() -> None:
    trainer = object.__new__(AsyncDiffusionTrainer)

    assert trainer._refill_before_score() is True


def test_eval_reward_pipeline_launches_before_resolving_previous(
    monkeypatch,
) -> None:
    events: list[str] = []

    class EvalInputs:
        batch_size = 3

        def slice(self, start: int, end: int):
            return SimpleNamespace(batch_size=end - start, index=start)

    class DataSource:
        def get_eval_samples(self, _num_prompts: int) -> EvalInputs:
            return EvalInputs()

    class Pending:
        def __init__(self, index: int) -> None:
            self.index = index

        def result(self):
            events.append(f"result:{self.index}")
            return object()

    class Reward:
        dp_size = 1

        def launch_nowait(self, method: str, generated):
            assert method == "score_and_attach"
            events.append(f"launch:{generated.index}")
            return Pending(generated.index)

    trainer = object.__new__(DiffusionTrainer)
    trainer.eval_chunk_prompts = 1
    trainer.eval_reward_async = True
    trainer.rollout = SimpleNamespace(dp_size=1)
    trainer.reward = Reward()
    trainer._build_request_sample = (
        lambda sub, _step, sampling: sub
    )

    def generate(request, **_kwargs):
        events.append(f"generate:{request.index}")
        return request

    trainer._generate_with_residency = generate

    def resolve(pending, **_kwargs):
        generated, calls, _media_prefix = pending
        events.append(f"resolve:{generated.index}")
        for _name, call in calls:
            call.result()
        return 0.0

    trainer._resolve_eval_reward_calls = resolve
    monkeypatch.setattr(
        diffusion_module,
        "_flatten_reward_rows",
        lambda generated: generated,
    )
    monkeypatch.setattr(
        diffusion_module,
        "_validate_prompt_tree_dp_geometry",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        diffusion_module,
        "total_samples_per_prompt",
        lambda _sampling: 1,
    )

    trainer._eval_pass(
        DataSource(),
        3,
        [("reward", trainer.reward)],
        {"diffusion": object()},
        0,
        sync_weights=False,
        resync_after_sleep=False,
        sleep_rollout=False,
    )

    assert events == [
        "generate:0",
        "launch:0",
        "generate:1",
        "launch:1",
        "resolve:0",
        "result:0",
        "generate:2",
        "launch:2",
        "resolve:1",
        "result:1",
        "resolve:2",
        "result:2",
    ]
