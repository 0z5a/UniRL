from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.models.leo2.diffusion import Leo2DiffusionStage
from unirl.models.leo2.packing import (
    as_packed_media,
    pack_hymm_conditions,
    unpack_packed_prediction,
)


def _blob(*, text_tokens: int, pad_text_to: int, marker: int = 0) -> dict:
    video_tokens = 4
    audio_tokens = 2
    sequence_length = text_tokens + video_tokens + audio_tokens
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0) + marker * 100
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[0, 0] = sequence_length

    text_mask = torch.zeros_like(input_ids)
    text_mask[:, :text_tokens] = 1
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    visual_mask[:, text_tokens : text_tokens + video_tokens] = True
    audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    audio_mask[:, -audio_tokens:] = True

    text_states = torch.arange(pad_text_to * 3, dtype=torch.float32).reshape(1, pad_text_to, 3).add(marker * 1000)
    cond_text_mask = torch.zeros((1, pad_text_to), dtype=torch.int32)
    cond_text_mask[:, :text_tokens] = 1
    return {
        "input_ids": input_ids,
        "model_kwargs": {
            "attention_mask": attention_mask,
            "rope_media_info": [
                [
                    (
                        slice(text_tokens, text_tokens + video_tokens),
                        (1, 2, 2),
                        {"type": "gen_video"},
                    ),
                    (
                        slice(text_tokens + video_tokens, sequence_length),
                        (2, 1, 1),
                        {"type": "gen_audio"},
                    ),
                ]
            ],
            "cond_vae_images": None,
            "cond_text_states": text_states,
            "cond_text_mask": cond_text_mask,
            "visual_mask": visual_mask,
            "text_mask": text_mask,
            "timesteps_index": None,
            "audio_mask": audio_mask,
            "cond_vae_mask": None,
            "cond_timesteps": None,
            "cond_text_scatter_mask": text_mask.clone(),
            "und_token_indices": torch.arange(text_tokens).unsqueeze(0),
            "gen_token_indices": torch.arange(text_tokens, text_tokens + video_tokens).unsqueeze(0),
            "audio_token_indices": torch.arange(text_tokens + video_tokens, sequence_length).unsqueeze(0),
        },
        "image_size": (16, 16),
        "video_duration": 1,
        "audio_duration": 1000,
        "audio_token_length": audio_tokens,
        "training_audio_noise": torch.ones(1, 2, audio_tokens),
    }


def test_pack_hymm_conditions_builds_native_concatenated_sequence_contract() -> None:
    first = _blob(text_tokens=2, pad_text_to=4, marker=1)
    second = _blob(text_tokens=3, pad_text_to=5, marker=2)

    packed = pack_hymm_conditions([first, second])

    assert packed.batch_size == 2
    assert packed.sequence_lengths == (8, 9)
    torch.testing.assert_close(
        packed.input_ids,
        torch.cat([first["input_ids"], second["input_ids"]], dim=1),
    )
    torch.testing.assert_close(
        packed.model_kwargs["attention_mask"],
        torch.tensor([[8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    )
    torch.testing.assert_close(packed.packing_kwargs["sample_offsets"], torch.tensor([[0, 8, 17]]))
    assert packed.packing_kwargs["pad_count"] == {"und": 0, "gen": 0, "audio": 0}
    torch.testing.assert_close(packed.model_kwargs["und_token_indices"], torch.tensor([[0, 1, 8, 9, 10]]))
    torch.testing.assert_close(
        packed.model_kwargs["gen_token_indices"],
        torch.tensor([[2, 3, 4, 5, 11, 12, 13, 14]]),
    )
    torch.testing.assert_close(packed.model_kwargs["audio_token_indices"], torch.tensor([[6, 7, 15, 16]]))
    torch.testing.assert_close(packed.packing_kwargs["und_token_lengths"], torch.tensor([2, 3]))
    torch.testing.assert_close(packed.packing_kwargs["gen_token_lengths"], torch.tensor([4, 4]))
    torch.testing.assert_close(packed.packing_kwargs["audio_token_lengths"], torch.tensor([2, 2]))

    # Padded text states are compacted because LeoModel runs the RL forward in eval mode.
    expected_text = torch.cat(
        [
            first["model_kwargs"]["cond_text_states"][:, :2],
            second["model_kwargs"]["cond_text_states"][:, :3],
        ],
        dim=1,
    )
    torch.testing.assert_close(packed.model_kwargs["cond_text_states"], expected_text)
    torch.testing.assert_close(packed.model_kwargs["cond_text_mask"], torch.ones((1, 5), dtype=torch.int32))

    rope = packed.model_kwargs["rope_media_info"]
    assert [info[0] for info in rope[0]] == [
        slice(2, 6),
        slice(6, 8),
        slice(11, 15),
        slice(15, 17),
    ]


def test_pack_hymm_conditions_uses_the_same_contract_for_batch_one() -> None:
    blob = _blob(text_tokens=2, pad_text_to=4)

    packed = pack_hymm_conditions([blob])

    assert packed.batch_size == 1
    assert packed.sequence_lengths == (8,)
    torch.testing.assert_close(packed.input_ids, blob["input_ids"])
    torch.testing.assert_close(packed.packing_kwargs["sample_offsets"], torch.tensor([[0, 8]]))
    torch.testing.assert_close(packed.packing_kwargs["gen_token_lengths"], torch.tensor([4]))


def test_pack_hymm_conditions_pads_each_branch_for_context_parallelism() -> None:
    first = _blob(text_tokens=2, pad_text_to=2)
    second = _blob(text_tokens=3, pad_text_to=3)

    packed = pack_hymm_conditions(
        [first, second],
        sequence_parallel_size=2,
    )

    # The five text tokens need one tail token; video/audio totals are already even.
    assert packed.sequence_lengths == (8, 9)
    assert packed.input_ids.shape == (1, 18)
    torch.testing.assert_close(
        packed.model_kwargs["attention_mask"],
        torch.tensor([[8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    )
    torch.testing.assert_close(packed.packing_kwargs["sample_offsets"], torch.tensor([[0, 8, 17]]))
    assert packed.packing_kwargs["pad_count"] == {"und": 1, "gen": 0, "audio": 0}
    torch.testing.assert_close(packed.model_kwargs["und_token_indices"], torch.tensor([[0, 1, 8, 9, 10, 17]]))
    torch.testing.assert_close(packed.packing_kwargs["und_token_lengths"], torch.tensor([2, 4]))
    assert all(
        int(packed.model_kwargs[key].shape[1]) % 2 == 0
        for key in ("und_token_indices", "gen_token_indices", "audio_token_indices")
    )
    for key in ("visual_mask", "text_mask", "audio_mask", "cond_text_scatter_mask"):
        assert packed.model_kwargs[key].shape == (1, 18)


def test_pack_hymm_conditions_rejects_non_partitioning_indices() -> None:
    first = _blob(text_tokens=2, pad_text_to=2)
    second = _blob(text_tokens=2, pad_text_to=2)
    second["model_kwargs"]["gen_token_indices"][0, 0] = 0

    with pytest.raises(ValueError, match="partition every input token"):
        pack_hymm_conditions([first, second])


def test_pack_hymm_conditions_normalizes_int32_token_indices() -> None:
    blob = _blob(text_tokens=2, pad_text_to=2)
    for key in ("und_token_indices", "gen_token_indices", "audio_token_indices"):
        blob["model_kwargs"][key] = blob["model_kwargs"][key].to(torch.int32)

    packed = pack_hymm_conditions([blob])

    for key in ("und_token_indices", "gen_token_indices", "audio_token_indices"):
        assert packed.model_kwargs[key].dtype is torch.int64


@pytest.mark.parametrize("batch_size", [1, 3])
def test_packed_media_keeps_logical_batch_inside_one_physical_row(batch_size: int) -> None:
    media = torch.randn(batch_size, 4, 2, 2)
    packed = as_packed_media(media)

    assert len(packed) == 1
    assert packed[0] is media


def test_packed_media_keeps_ragged_samples_inside_one_physical_row() -> None:
    media = [torch.randn(1, 4, 2, 2), torch.randn(1, 4, 3, 5)]

    packed = as_packed_media(media)

    assert len(packed) == 1
    assert isinstance(packed[0], list)
    assert packed[0][0].data_ptr() == media[0].data_ptr()
    assert packed[0][1].data_ptr() == media[1].data_ptr()
    assert [tuple(row.shape) for row in packed[0]] == [(4, 2, 2), (4, 3, 5)]


def test_unpack_packed_prediction_preserves_order_and_gradients() -> None:
    samples = [torch.randn(1, 3, requires_grad=True) for _ in range(2)]
    dense = unpack_packed_prediction([samples], batch_size=2, name="video")

    torch.testing.assert_close(dense, torch.cat(samples))
    dense.sum().backward()
    for sample in samples:
        torch.testing.assert_close(sample.grad, torch.ones_like(sample))


class _FakeLeoModel:
    training = False

    def __init__(self) -> None:
        self.prepared: dict | None = None
        self.forwarded: dict | None = None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        self.prepared = {"input_ids": input_ids, **kwargs}
        return dict(self.prepared)

    def __call__(self, **kwargs):
        self.forwarded = kwargs
        latents = kwargs["latents"]
        audio = kwargs["audio_latents"]
        if isinstance(latents, list):
            logical_video = latents[0]
            logical_audio = audio[0]
            if isinstance(logical_video, list):
                return {
                    "diffusion_prediction": [[row.unsqueeze(0) * 2 for row in logical_video]],
                    "audio_diffusion_prediction": [[row.unsqueeze(0) * 3 for row in logical_audio]],
                }
            return {
                "diffusion_prediction": [
                    [logical_video[index : index + 1] * 2 for index in range(logical_video.shape[0])]
                ],
                "audio_diffusion_prediction": [
                    [logical_audio[index : index + 1] * 3 for index in range(logical_audio.shape[0])]
                ],
            }
        return {
            "diffusion_prediction": latents * 2,
            "audio_diffusion_prediction": audio * 3,
        }


def _forward_stage(model: _FakeLeoModel) -> Leo2DiffusionStage:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(model=model)
    stage.profile_forward = False
    return stage


def test_predict_joint_noise_uses_packed_model_contract_for_batch_two(monkeypatch) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    stage = _forward_stage(model)
    blobs = [_blob(text_tokens=2, pad_text_to=2), _blob(text_tokens=3, pad_text_to=3)]
    video = torch.randn(2, 4, 1, 2, 2, requires_grad=True)
    audio = torch.randn(2, 2, 2, requires_grad=True)

    video_pred, audio_pred = stage.predict_joint_noise(
        blobs,
        sample=video,
        sigma=torch.tensor([0.25, 0.75]),
        channel_cond=(None, None),
        audio_sample=audio,
        audio_sigma=torch.tensor([0.1, 0.2]),
    )

    assert isinstance(model.prepared["latents"], list)
    assert len(model.prepared["latents"]) == 1
    assert model.prepared["latents"][0] is video
    assert isinstance(model.prepared["timesteps"], list)
    torch.testing.assert_close(model.prepared["timesteps"][0], torch.tensor([250.0, 750.0]))
    torch.testing.assert_close(model.forwarded["sample_offsets"], torch.tensor([[0, 8, 17]]))
    torch.testing.assert_close(model.forwarded["gen_token_lengths"], torch.tensor([4, 4]))
    torch.testing.assert_close(video_pred, video.detach() * 2)
    torch.testing.assert_close(audio_pred, audio.detach() * 3)
    (video_pred.sum() + audio_pred.sum()).backward()
    torch.testing.assert_close(video.grad, torch.full_like(video, 2))
    torch.testing.assert_close(audio.grad, torch.full_like(audio, 3))


def test_predict_joint_noise_preserves_shared_timestep_for_batch_two(monkeypatch) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    stage = _forward_stage(model)
    blobs = [_blob(text_tokens=2, pad_text_to=2), _blob(text_tokens=3, pad_text_to=3)]

    stage.predict_joint_noise(
        blobs,
        sample=torch.randn(2, 4, 1, 2, 2),
        sigma=torch.tensor(0.25),
        channel_cond=(None, None),
        audio_sample=torch.randn(2, 2, 2),
        audio_sigma=torch.tensor([0.1]),
    )

    torch.testing.assert_close(model.prepared["timesteps"][0], torch.tensor([250.0]))
    torch.testing.assert_close(model.prepared["audio_timesteps"][0], torch.tensor([100.0]))


def test_predict_joint_noise_uses_unified_packed_contract_for_batch_one(monkeypatch) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    stage = _forward_stage(model)
    blob = _blob(text_tokens=2, pad_text_to=2)
    video = torch.randn(1, 4, 1, 2, 2)
    audio = torch.randn(1, 2, 2)

    video_pred, audio_pred = stage.predict_joint_noise(
        blob,
        sample=video,
        sigma=torch.tensor(0.5),
        channel_cond=(None, None),
        audio_sample=audio,
        audio_sigma=torch.tensor(0.25),
    )

    assert isinstance(model.prepared["latents"], list)
    assert len(model.prepared["latents"]) == 1
    assert model.prepared["latents"][0] is video
    assert isinstance(model.prepared["timesteps"], list)
    torch.testing.assert_close(model.prepared["timesteps"][0], torch.tensor([500.0]))
    torch.testing.assert_close(model.forwarded["sample_offsets"], torch.tensor([[0, 8]]))
    torch.testing.assert_close(model.forwarded["gen_token_lengths"], torch.tensor([4]))
    torch.testing.assert_close(video_pred, video * 2)
    torch.testing.assert_close(audio_pred, audio * 3)


def test_predict_joint_noise_keeps_ragged_media_and_gradients(monkeypatch) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    stage = _forward_stage(model)
    blobs = [_blob(text_tokens=2, pad_text_to=2), _blob(text_tokens=3, pad_text_to=3)]
    videos = [
        torch.randn(1, 4, 1, 2, 2, requires_grad=True),
        torch.randn(1, 4, 2, 2, 3, requires_grad=True),
    ]
    audios = [
        torch.randn(1, 2, 2, requires_grad=True),
        torch.randn(1, 2, 5, requires_grad=True),
    ]

    video_pred, audio_pred = stage.predict_joint_noise(
        blobs,
        sample=videos,
        sigma=torch.tensor([0.25, 0.75]),
        channel_cond=(None, None),
        audio_sample=audios,
        audio_sigma=torch.tensor([0.1, 0.2]),
    )

    assert isinstance(model.prepared["latents"][0], list)
    assert [tuple(row.shape) for row in model.prepared["latents"][0]] == [
        (4, 1, 2, 2),
        (4, 2, 2, 3),
    ]
    assert isinstance(video_pred, list)
    assert isinstance(audio_pred, list)
    for prediction, source in zip(video_pred, videos):
        torch.testing.assert_close(prediction, source * 2)
    for prediction, source in zip(audio_pred, audios):
        torch.testing.assert_close(prediction, source * 3)
    sum(row.sum() for row in video_pred + audio_pred).backward()
    for source in videos:
        torch.testing.assert_close(source.grad, torch.full_like(source, 2))
    for source in audios:
        torch.testing.assert_close(source.grad, torch.full_like(source, 3))


def test_predict_joint_noise_rejects_nonpacked_effective_inference_attention(
    monkeypatch,
) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    model._config = SimpleNamespace(
        attn_impl="flash3_packed",
        inference_attn_impl="flash3",
    )
    stage = _forward_stage(model)
    blob = _blob(text_tokens=2, pad_text_to=2)

    with torch.no_grad(), pytest.raises(ValueError, match="requires flash_packed or flash3_packed"):
        stage.predict_joint_noise(
            blob,
            sample=torch.randn(1, 4, 1, 2, 2),
            sigma=torch.tensor(0.5),
            channel_cond=(None, None),
        )


def test_predict_joint_noise_rejects_single_refiner_for_packed_batch(monkeypatch) -> None:
    monkeypatch.setattr("unirl.models.leo2.bundle.ensure_hy_parallel_state", lambda: True)
    model = _FakeLeoModel()
    model._config = SimpleNamespace(
        attn_impl="flash3_packed",
        inference_attn_impl=None,
        text_proj_type="single_refiner",
    )
    stage = _forward_stage(model)
    blobs = [_blob(text_tokens=2, pad_text_to=2), _blob(text_tokens=3, pad_text_to=3)]

    with pytest.raises(ValueError, match="requires text_proj_type='linear'"):
        stage.predict_joint_noise(
            blobs,
            sample=torch.randn(2, 4, 1, 2, 2),
            sigma=torch.tensor(0.5),
            channel_cond=(None, None),
        )
