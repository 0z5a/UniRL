from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from unirl.models.leo2.pipeline import Leo2Pipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.train.stack import CountPlanner
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment


class _RecordingLeo2Pipeline:
    def __init__(self) -> None:
        self.diffusion = _FakeStage()
        self.forward_packs: list[tuple[str, ...]] = []

    def generate(self, sample: Sample) -> Sample:
        gen = sample.parts[-1]
        forward_pack = tuple(gen.sample_ids)
        self.forward_packs.append(forward_pack)
        return sample.replace_frontier(
            dataclasses.replace(
                gen,
                forward_pack_sample_ids=[forward_pack] * gen.batch_size,
            )
        )


class _UnmarkedPipeline(_RecordingLeo2Pipeline):
    def generate(self, sample: Sample) -> Sample:
        self.forward_packs.append(tuple(sample.parts[-1].sample_ids))
        return sample


class _FakeStage:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)

    def trainable_module(self) -> torch.nn.Module:
        return self.model


def _replay_packs(part: Part, *, num_updates: int, micro_batch_size: int = 2) -> list[tuple[str, ...]]:
    arranged, plan = CountPlanner().arrange(
        part,
        num_updates=num_updates,
        micro_batch_size=micro_batch_size,
    )
    return [tuple(arranged.slice(start, end).sample_ids) for update in plan for start, end in update]


def test_count_planner_replays_rollout_packs_as_contiguous_units() -> None:
    root = Part.input([f"prompt-{index}" for index in range(4)])
    request = Sample.request(root).fork(2, sampling_params=ARSamplingParams())
    pipeline = _RecordingLeo2Pipeline()
    rollout = TrainsideRolloutEngine(
        pipeline=pipeline,
        forward_batch_size=2,
    )

    generated = rollout._generate_locked(request)

    assert generated.parts[-1].forward_pack_sample_ids == [pack for pack in pipeline.forward_packs for _ in pack]
    assert generated.parts[-1].output_version == 0
    assert _replay_packs(generated.parts[-1], num_updates=2) == pipeline.forward_packs


def test_whole_rollout_packs_may_reorder_without_changing_inner_order() -> None:
    sample_ids = [f"sample-{index}" for index in range(8)]
    pack_ids = [tuple(sample_ids[start : start + 2]) for start in range(0, len(sample_ids), 2)]
    part = Part(
        sample_ids=sample_ids,
        forward_pack_sample_ids=[pack for pack in pack_ids for _ in pack],
    )
    rollout_packs = [part.slice(start, start + 2) for start in range(0, part.batch_size, 2)]
    pack_order = [2, 0, 3, 1]
    reordered = Part.concat([rollout_packs[index] for index in pack_order])

    replay_packs = _replay_packs(reordered, num_updates=2)

    assert replay_packs == [tuple(rollout_packs[index].sample_ids) for index in pack_order]


def test_count_planner_rejects_splitting_one_b4_pack_across_two_updates() -> None:
    root = Part.input(["prompt-0", "prompt-1"])
    request = Sample.request(root).fork(2, sampling_params=ARSamplingParams())
    rollout = TrainsideRolloutEngine(pipeline=_RecordingLeo2Pipeline(), forward_batch_size=4)
    generated = rollout._generate_core(request)

    with pytest.raises(ValueError, match="1 forward packs.*2 optimizer updates"):
        CountPlanner().arrange(generated.parts[-1], num_updates=2, micro_batch_size=4)


def test_count_planner_keeps_two_b4_packs_exact_across_two_updates() -> None:
    root = Part.input([f"prompt-{index}" for index in range(4)])
    request = Sample.request(root).fork(2, sampling_params=ARSamplingParams())
    pipeline = _RecordingLeo2Pipeline()
    rollout = TrainsideRolloutEngine(pipeline=pipeline, forward_batch_size=4)
    generated = rollout._generate_core(request)

    assert _replay_packs(generated.parts[-1], num_updates=2, micro_batch_size=4) == pipeline.forward_packs


@pytest.mark.parametrize("mutation", ["inner-reorder", "marker-reorder", "dp-truncate"])
def test_count_planner_rejects_changed_or_truncated_forward_pack(mutation: str) -> None:
    root = Part.input(["prompt-0", "prompt-1"])
    request = Sample.request(root).fork(2, sampling_params=ARSamplingParams())
    rollout = TrainsideRolloutEngine(pipeline=_RecordingLeo2Pipeline(), forward_batch_size=4)
    part = rollout._generate_core(request).parts[-1]
    if mutation == "inner-reorder":
        part = part.select(torch.tensor([1, 0, 2, 3]))
    elif mutation == "marker-reorder":
        reversed_pack = tuple(reversed(part.sample_ids))
        part = dataclasses.replace(
            part,
            forward_pack_sample_ids=[reversed_pack] * part.batch_size,
        )
    else:
        part = part.slice(1, 4)

    with pytest.raises(ValueError, match="rollout forward pack"):
        CountPlanner().arrange(part, num_updates=1, micro_batch_size=4)


def test_default_trainside_rollout_keeps_legacy_unmarked_planning() -> None:
    root = Part.input(["prompt-0", "prompt-1"])
    request = Sample.request(root).fork(2, sampling_params=ARSamplingParams())
    rollout = TrainsideRolloutEngine(pipeline=_UnmarkedPipeline(), forward_batch_size=2)
    part = rollout._generate_core(request).parts[-1]

    assert part.forward_pack_sample_ids == []
    _, plan = CountPlanner().arrange(part, num_updates=2, micro_batch_size=4)
    assert plan == [[(0, 2)], [(2, 4)]]


def test_leo2_pipeline_records_the_exact_generate_call_as_one_forward_pack(monkeypatch) -> None:
    batch_size = 2
    conditions = SimpleNamespace(
        hymm=[{"image_size": (16, 16), "video_duration": 1} for _ in range(batch_size)],
        to_dict=lambda: {},
    )

    class FakeNoiseRecipe:
        def resolve(self, *, device, latent_shape, **kwargs):
            return torch.zeros(batch_size, *latent_shape, device=device)

    class FakeDiffusion:
        def generate(self, _conditions, *, params, initial_latents, **kwargs):
            return LatentSegment(
                latents=initial_latents[:, None],
                indices=torch.tensor([params.num_inference_steps]),
                sigmas=params.sigmas,
            )

    class FakeVideoDecode:
        def decode(self, latents):
            return Videos.from_list([Video(frames=torch.zeros(1, 3, 16, 16)) for _ in range(int(latents.shape[0]))])

    monkeypatch.setattr(
        "unirl.models.leo2.pipeline.NoiseRecipe.from_sample",
        lambda _sample: FakeNoiseRecipe(),
    )
    pipeline = Leo2Pipeline(
        bundle=SimpleNamespace(device=torch.device("cpu")),
        cond_stage=SimpleNamespace(build=lambda *args, **kwargs: conditions),
        diffusion=FakeDiffusion(),
        video_decode=FakeVideoDecode(),
        audio_decode=None,
        config=SimpleNamespace(enable_audio=False),
    )
    root = Part.input(
        [f"prompt-{index}" for index in range(batch_size)],
        primitives={"text": Texts([f"text-{index}" for index in range(batch_size)])},
    )
    request = Sample.request(root).fork(
        1,
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=1,
            height=16,
            width=16,
            num_frames=1,
            sigmas=torch.tensor([1.0, 0.0]),
        ),
    )

    generated = pipeline.generate(request).parts[-1]
    expected_pack = tuple(request.parts[-1].sample_ids)

    assert generated.forward_pack_sample_ids == [expected_pack] * batch_size


def test_leo2_flowgrpo_recipes_pin_matching_contiguous_batching() -> None:
    examples_dir = Path(__file__).parents[1] / "examples"
    recipes = (
        "diffusion/leo2/leo2_t2v_trainside",
        "diffusion/leo2/leo2_t2v_flowgrpo_cached",
        "diffusion/leo2/leo2_t2v_flowgrpo_context_ir",
        "diffusion/leo2/leo2_t2v_flowgrpo_motion_bilingual",
        "diffusion/leo2/leo2_t2v_flowgrpo_separate",
        "diffusion/leo2/leo2_t2v_flowgrpo_colocated",
    )

    with initialize_config_dir(version_base=None, config_dir=str(examples_dir)):
        for recipe in recipes:
            config = compose(config_name=recipe)
            assert config.rollout.forward_batch_size == 2
            assert config.stack.micro_batch_size == 2
            assert config.stack.micro_planner._target_ == "unirl.train.stack.CountPlanner"
