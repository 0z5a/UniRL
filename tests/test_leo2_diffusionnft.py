from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from unirl.algorithms.diffusionnft import DiffusionNFT
from unirl.distributed.group.remote import RankInfo
from unirl.models.leo2.bundle import (
    _configure_grouped_gemm_fallback,
    _dcp_load_into,
    _make_inference_cache_config,
    _patch_router_dtype,
)
from unirl.models.leo2.conditions import Leo2Conditions
from unirl.models.leo2.diffusion import Leo2DiffusionStage, _combine_modality_logp
from unirl.models.leo2.pipeline import Leo2Pipeline
from unirl.models.leo2.preprocess import _records
from unirl.models.leo2.sde import Leo2FlowSDEStrategy
from unirl.models.leo2.text_embed import Leo2CondStage, _to_transport_tree
from unirl.models.leo2.vae import Leo2AudioDecodeStage, Leo2VideoDecodeStage
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.sde.kernels import FlowSDEStrategy
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.configs import EmaLoraConfig
from unirl.train.ema import EMA, Shadow, make_decay_fn
from unirl.train.stack.base import TrainStack, TrainStepResult
from unirl.trainer.diffusion import _compute_chunked_advantages, _resolve_rollout_chunk_prompts
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.segments.latent import LatentSegment


def _fake_stage() -> tuple[Leo2DiffusionStage, dict[str, object]]:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"))
    stage.trajectory_dtype = torch.float32
    stage.enable_audio = False
    observed: dict[str, object] = {}

    def prep(self, blob, sample):
        observed["blob"] = blob
        observed["prep_shape"] = tuple(sample.shape)
        return "cond", "mask"

    def autocast(self):
        return nullcontext()

    def predict(self, blob, *, sample, sigma, channel_cond):
        observed["predict_blob"] = blob
        observed["sigma"] = float(sigma)
        observed["channel_cond"] = channel_cond
        return sample + 1

    stage._prep_channel_cond = MethodType(prep, stage)
    stage._autocast = MethodType(autocast, stage)
    stage.predict_noise = MethodType(predict, stage)
    return stage, observed


def test_leo2_predict_noise_at_step_forwards_batch_one() -> None:
    stage, observed = _fake_stage()
    sample = torch.zeros(1, 48, 2, 3, 4)
    blob = {"id": "sample-0"}
    output = stage.predict_noise_at_step(
        SimpleNamespace(hymm=[blob]),
        sample=sample,
        sigma=torch.tensor([0.5]),
        params=SimpleNamespace(guidance_scale=1.0),
    )

    torch.testing.assert_close(output, torch.ones_like(sample))
    assert observed == {
        "blob": blob,
        "prep_shape": tuple(sample.shape),
        "predict_blob": [blob],
        "sigma": 0.5,
        "channel_cond": ("cond", "mask"),
    }


def test_leo2_av_single_step_uses_shifted_deterministic_audio_prior() -> None:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"))
    stage.trajectory_dtype = torch.float32
    stage.enable_audio = True
    stage.video_shift = 9.0
    stage.audio_shift = 3.0
    stage._prep_channel_cond = MethodType(
        lambda self, blob, sample: (None, None),
        stage,
    )
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    observed = {}

    def predict_joint(
        self,
        blob,
        *,
        sample,
        sigma,
        channel_cond,
        audio_sample,
        audio_sigma,
    ):
        observed["audio_sample"] = audio_sample
        observed["audio_sigma"] = float(audio_sigma)
        return torch.ones_like(sample), torch.zeros_like(audio_sample)

    stage.predict_joint_noise = MethodType(predict_joint, stage)
    output = stage.predict_noise_at_step(
        SimpleNamespace(
            hymm=[
                {
                    "training_audio_noise": torch.ones(1, 96, 4),
                }
            ]
        ),
        sample=torch.zeros(1, 48, 2, 3, 4),
        sigma=torch.tensor([0.9]),
        params=SimpleNamespace(guidance_scale=1.0),
    )

    assert observed["audio_sigma"] == pytest.approx(0.75)
    torch.testing.assert_close(
        observed["audio_sample"],
        torch.full((1, 96, 4), 0.75),
    )
    torch.testing.assert_close(output, torch.ones_like(output))


def test_leo2_av_single_step_packs_per_sample_sigmas_and_audio_priors() -> None:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"))
    stage.trajectory_dtype = torch.float32
    stage.enable_audio = True
    stage.video_shift = 9.0
    stage.audio_shift = 3.0
    stage._prep_channel_cond = MethodType(lambda self, blob, sample: (None, None), stage)
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    observed = {}

    def predict_joint(self, blobs, **kwargs):
        observed["blobs"] = blobs
        observed["sigma"] = kwargs["sigma"]
        observed["audio_sigma"] = kwargs["audio_sigma"]
        observed["audio_sample"] = kwargs["audio_sample"]
        return torch.ones_like(kwargs["sample"]), torch.zeros_like(kwargs["audio_sample"])

    stage.predict_joint_noise = MethodType(predict_joint, stage)
    blobs = [
        {"training_audio_noise": torch.ones(1, 2, 3)},
        {"training_audio_noise": torch.full((1, 2, 3), 2.0)},
    ]
    output = stage.predict_noise_at_step(
        SimpleNamespace(hymm=blobs),
        sample=torch.zeros(2, 4, 1, 2, 2),
        sigma=torch.tensor([0.9, 0.5]),
        params=SimpleNamespace(guidance_scale=1.0),
    )

    assert observed["blobs"] is blobs
    torch.testing.assert_close(observed["sigma"], torch.tensor([0.9, 0.5]))
    torch.testing.assert_close(observed["audio_sigma"], torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(
        observed["audio_sample"],
        torch.stack([torch.full((2, 3), 0.75), torch.full((2, 3), 0.5)]),
    )
    torch.testing.assert_close(output, torch.ones_like(output))


def test_leo2_audio_decode_returns_length_first_stereo() -> None:
    class FakeAudioVAE:
        def to(self, device):
            return self

        def decode(self, latents):
            assert tuple(latents.shape) == (1, 96, 4)
            return torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    bundle = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(audio_vae_on_gpu=True),
        model=SimpleNamespace(model_dict={"audio_vae": FakeAudioVAE()}),
    )
    decoded = Leo2AudioDecodeStage(bundle).decode(torch.zeros(1, 96, 4))

    items = decoded.to_list()
    assert len(items) == 1
    assert tuple(items[0].waveform.shape) == (8, 2)


def test_leo2_video_decode_preserves_native_output_dtype(monkeypatch) -> None:
    autoencoders = ModuleType("hymm.models.autoencoders")
    autoencoders.denormalize_vae_latents = lambda vae, latents: latents
    monkeypatch.setitem(sys.modules, "hymm", ModuleType("hymm"))
    monkeypatch.setitem(sys.modules, "hymm.models", ModuleType("hymm.models"))
    monkeypatch.setitem(sys.modules, "hymm.models.autoencoders", autoencoders)

    class FakeVideoVAE:
        def to(self, device):
            return self

        def decode(self, latents, return_dict=False):
            assert not return_dict
            return (latents.to(torch.float16),)

    class FakeVideoProcessor:
        @staticmethod
        def postprocess_video(visuals, output_type):
            assert output_type == "pt"
            return (visuals * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 1, 3, 4)

    bundle = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(vae_on_gpu=True),
        model=SimpleNamespace(
            model_dict={"vae": FakeVideoVAE()},
            diffusion_pipeline=SimpleNamespace(
                vae_autocast_dtype=torch.float16,
                video_processor=FakeVideoProcessor(),
            ),
        ),
    )
    decoded = Leo2VideoDecodeStage(bundle).decode_to_tensor(
        torch.full((1, 3, 1, 2, 2), 0.1),
    )

    assert decoded.dtype is torch.float16
    torch.testing.assert_close(
        decoded,
        (torch.full_like(decoded, 0.1) * 0.5 + 0.5).clamp(0, 1),
    )


def test_leo2_joint_log_prob_matches_native_modality_sum() -> None:
    video = torch.tensor([2.0, 4.0])
    audio = torch.tensor([8.0, 10.0])

    actual = _combine_modality_logp(
        video,
        audio,
        n_video=12,
        n_audio=4,
    )

    torch.testing.assert_close(actual, video + audio)


def test_leo2_grouped_gemm_fallback_defaults_inside_ray_worker(monkeypatch) -> None:
    key = "HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM"
    monkeypatch.delenv(key, raising=False)

    _configure_grouped_gemm_fallback()

    assert os.environ[key] == "0"


def test_leo2_grouped_gemm_fallback_preserves_explicit_policy(monkeypatch) -> None:
    key = "HY_PARALLELISM_USE_CUTLASS_GROUPED_GEMM"
    monkeypatch.setenv(key, "1")

    _configure_grouped_gemm_fallback()

    assert os.environ[key] == "1"


@pytest.mark.parametrize(
    ("video", "audio", "n_video", "n_audio", "exception", "message"),
    [
        (torch.ones(2), torch.ones(2), True, 4, ValueError, "positive int n_video"),
        (torch.ones(2), torch.ones(2), 12, 0, ValueError, "positive int n_audio"),
        (torch.ones(2, 1), torch.ones(2, 1), 12, 4, ValueError, r"shaped \[B\]"),
        (torch.ones(2), torch.ones(3), 12, 4, ValueError, "matching per-sample"),
        (torch.ones(2), [1.0, 1.0], 12, 4, TypeError, "expected Tensor"),
    ],
)
def test_leo2_joint_log_prob_rejects_invalid_component_contract(
    video,
    audio,
    n_video,
    n_audio,
    exception,
    message,
) -> None:
    with pytest.raises(exception, match=message):
        _combine_modality_logp(
            video,
            audio,
            n_video=n_video,
            n_audio=n_audio,
        )


def test_leo2_av_generate_stores_audio_trajectory() -> None:
    class FakeModel:
        training = False

        @staticmethod
        def cache_context(name):
            return nullcontext()

        @staticmethod
        def cache_stats():
            return {"method": "none"}

    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"), model=FakeModel())
    stage.strategy = SimpleNamespace(
        denoise=lambda **kwargs: (kwargs["sample"], None, None),
    )
    stage.trajectory_dtype = torch.float32
    stage.logprob_dtype = torch.float32
    stage.enable_audio = True
    stage.audio_joint_sde = False
    stage.video_shift = 9.0
    stage.audio_shift = 3.0
    stage._mem_reported = True
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    stage._prep_channel_cond = MethodType(
        lambda self, blob, sample: (None, None),
        stage,
    )
    stage.predict_joint_noise = MethodType(
        lambda self, blob, **kwargs: (
            torch.zeros_like(kwargs["sample"]),
            torch.zeros_like(kwargs["audio_sample"]),
        ),
        stage,
    )
    segment = stage.generate(
        SimpleNamespace(hymm=[{}]),
        params=SimpleNamespace(num_inference_steps=1, eta=0.0),
        sigmas=torch.tensor([1.0, 0.0]),
        initial_latents=torch.zeros(1, 48, 2, 3, 4),
        initial_audio_latents=torch.zeros(1, 96, 5),
    )

    assert tuple(segment.latents.shape) == (1, 2, 48, 2, 3, 4)
    assert tuple(segment.aux_latents.shape) == (1, 2, 96, 5)
    torch.testing.assert_close(segment.aux_latents_at(1), torch.zeros(1, 96, 5))


def test_leo2_bf16_av_rollout_log_probs_match_replay() -> None:
    torch.manual_seed(7)
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(
        device=torch.device("cpu"),
        model=SimpleNamespace(cache_stats=lambda: {"method": "none"}),
    )
    stage.strategy = FlowSDEStrategy()
    stage.trajectory_dtype = torch.bfloat16
    stage.logprob_dtype = torch.float32
    stage.enable_audio = True
    stage.audio_joint_sde = True
    stage.video_shift = 12.0
    stage.audio_shift = 3.0
    stage._mem_reported = True
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    stage._prep_channel_cond = MethodType(lambda self, blob, sample: (None, None), stage)
    stage.predict_joint_noise = MethodType(
        lambda self, blob, **kwargs: (
            kwargs["sample"].float().square() * 0.07 + 0.13,
            kwargs["audio_sample"].float().square() * 0.03 - 0.09,
        ),
        stage,
    )
    params = SimpleNamespace(num_inference_steps=2, eta=0.5)

    segment = stage.generate(
        SimpleNamespace(hymm=[{}, {}]),
        params=params,
        sigmas=torch.tensor([0.8, 0.5, 0.2]),
        initial_latents=torch.randn(2, 3, 2, 2, dtype=torch.bfloat16),
        initial_audio_latents=torch.randn(2, 2, 3, dtype=torch.bfloat16),
        sde_indices=[0, 1],
    )
    replay = stage.replay(SimpleNamespace(hymm=[{}, {}]), segment=segment, params=params)

    torch.testing.assert_close(replay.log_probs, segment.sde_logp, rtol=0.0, atol=0.0)
    assert segment.sde_means.dtype is torch.float32
    torch.testing.assert_close(replay.prev_sample_means, segment.sde_means, rtol=0.0, atol=0.0)


def test_flow_sde_matches_native_leo2_transition_order() -> None:
    strategy = Leo2FlowSDEStrategy()
    sample = torch.randn(2, 4, 3, 5, dtype=torch.bfloat16)
    prediction = torch.randn_like(sample)
    previous = torch.randn_like(sample).float()
    sigma = torch.tensor(1.0)
    sigma_next = torch.tensor(0.9950980544090271)
    sigma_max = sigma_next
    eta = 0.5

    actual_previous, actual_logp, actual_mean = strategy.denoise(
        noise_pred=prediction,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=eta,
        prev_sample=previous,
        sigma_max=sigma_max,
    )

    prediction_fp32 = prediction.float()
    sample_fp32 = sample.float()
    dt = sigma_next - sigma
    base_std = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    std = torch.clamp(base_std * torch.sqrt(-dt), min=1e-6)
    original = sample_fp32 - sigma * prediction_fp32
    score = -(sample_fp32 - original * (1 - sigma)) / (sigma**2)
    expected_mean = sample_fp32 + dt * prediction_fp32 + 0.5 * (std**2) * score
    expected_previous = previous.float()
    expected_logp = (
        -((expected_previous.detach() - expected_mean) ** 2) / (2 * (std**2))
        - torch.log(std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    ).mean(dim=tuple(range(1, previous.ndim)))

    torch.testing.assert_close(actual_previous, expected_previous, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_mean, expected_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_logp, expected_logp, rtol=0.0, atol=0.0)


def test_leo2_flow_sde_batches_distinct_timesteps_like_native_row_loop() -> None:
    strategy = Leo2FlowSDEStrategy()
    sample = torch.randn(3, 4, 3, 5, dtype=torch.bfloat16)
    prediction = torch.randn_like(sample)
    previous = torch.randn(3, 4, 3, 5, dtype=torch.float32)
    sigma = torch.tensor([1.0, 0.8, 0.4])
    sigma_next = torch.tensor([0.95, 0.7, 0.2])

    actual = strategy.denoise(
        noise_pred=prediction,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=0.5,
        prev_sample=previous,
        sigma_max=sigma_next[0],
    )
    expected_rows = [
        strategy.denoise(
            noise_pred=prediction[index : index + 1],
            sample=sample[index : index + 1],
            sigma=sigma[index],
            sigma_next=sigma_next[index],
            eta=0.5,
            prev_sample=previous[index : index + 1],
            sigma_max=sigma_next[0],
        )
        for index in range(sample.shape[0])
    ]
    expected = tuple(torch.cat([row[field] for row in expected_rows]) for field in range(3))

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_leo2_av_sde_noise_is_reproducible_across_worker_rng_states() -> None:
    class FakeModel:
        training = False

        @staticmethod
        def cache_context(name):
            return nullcontext()

        @staticmethod
        def cache_stats():
            return {"method": "none"}

    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"), model=FakeModel())
    stage.strategy = FlowSDEStrategy()
    stage.trajectory_dtype = torch.float32
    stage.logprob_dtype = torch.float32
    stage.enable_audio = True
    stage.audio_joint_sde = True
    stage.video_shift = 12.0
    stage.audio_shift = 3.0
    stage._mem_reported = True
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    stage._prep_channel_cond = MethodType(lambda self, blob, sample: (None, None), stage)
    stage.predict_joint_noise = MethodType(
        lambda self, blob, **kwargs: (
            torch.zeros_like(kwargs["sample"]),
            torch.zeros_like(kwargs["audio_sample"]),
        ),
        stage,
    )
    params = SimpleNamespace(num_inference_steps=2, eta=0.5)
    generate_kwargs = dict(
        conditions=SimpleNamespace(hymm=[{}]),
        params=params,
        sigmas=torch.tensor([0.8, 0.5, 0.2]),
        initial_latents=torch.zeros(1, 3, 2, 2),
        initial_audio_latents=torch.zeros(1, 2, 3),
        sde_indices=[0, 1],
        denoise_seed_keys=["sample-0"],
        denoise_base_seed=1234,
    )

    torch.manual_seed(1)
    first = stage.generate(**generate_kwargs)
    torch.manual_seed(2)
    second = stage.generate(**generate_kwargs)

    torch.testing.assert_close(first.latents, second.latents, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.aux_latents, second.aux_latents, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.sde_logp, second.sde_logp, rtol=0.0, atol=0.0)


def test_leo2_rollout_enters_request_scoped_cache_context() -> None:
    events: list[str] = []

    class FakeModel:
        def cache_context(self, name: str):
            class Context:
                def __enter__(self):
                    events.append(f"enter:{name}")

                def __exit__(self, exc_type, exc, traceback):
                    events.append(f"exit:{name}")

            return Context()

        @staticmethod
        def cache_stats():
            return {
                "method": "first_block",
                "threshold": 0.1,
                "full_steps": 1,
                "skipped_steps": 0,
                "cache_bytes": 0,
            }

    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"), model=FakeModel())
    stage.strategy = SimpleNamespace(
        denoise=lambda **kwargs: (kwargs["sample"], None, None),
    )
    stage.trajectory_dtype = torch.float32
    stage.logprob_dtype = torch.float32
    stage.enable_audio = False
    stage._mem_reported = True
    stage._autocast = MethodType(lambda self: nullcontext(), stage)
    stage._prep_channel_cond = MethodType(lambda self, blob, sample: (None, None), stage)

    def predict(self, blob, *, sample, sigma, channel_cond):
        assert events == ["enter:unirl_rollout"]
        return torch.zeros_like(sample)

    stage.predict_noise = MethodType(predict, stage)
    stage.generate(
        SimpleNamespace(hymm=[{}]),
        params=SimpleNamespace(num_inference_steps=1, eta=0.0),
        sigmas=torch.tensor([1.0, 0.0]),
        initial_latents=torch.zeros(1, 48, 2, 3, 4),
    )

    assert events == ["enter:unirl_rollout", "exit:unirl_rollout"]


@pytest.mark.parametrize(
    ("conditions", "sample", "sigma", "guidance_scale", "message"),
    [
        (SimpleNamespace(hymm=[]), torch.zeros(1, 48, 2, 3, 4), torch.tensor([0.5]), 1.0, "condition blob"),
        (
            SimpleNamespace(hymm=[{}]),
            torch.zeros(2, 48, 2, 3, 4),
            torch.tensor([0.5, 0.6]),
            1.0,
            "one condition per sample",
        ),
        (
            SimpleNamespace(hymm=[{}]),
            torch.zeros(1, 48, 2, 3, 4),
            torch.tensor([0.5, 0.6]),
            1.0,
            "one shared sigma or one per sample",
        ),
        (
            SimpleNamespace(hymm=[{}]),
            torch.zeros(1, 48, 2, 3, 4),
            torch.tensor([0.5]),
            2.0,
            "guidance_scale=1.0",
        ),
    ],
)
def test_leo2_predict_noise_at_step_rejects_invalid_contract(
    conditions: object,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    guidance_scale: float,
    message: str,
) -> None:
    stage, _ = _fake_stage()
    with pytest.raises((TypeError, ValueError), match=message):
        stage.predict_noise_at_step(
            conditions,
            sample=sample,
            sigma=sigma,
            params=SimpleNamespace(guidance_scale=guidance_scale),
        )


def test_leo2_channel_condition_follows_trajectory_dtype() -> None:
    stage = object.__new__(Leo2DiffusionStage)
    pipeline = SimpleNamespace(
        prepare_channel_cond_latents=lambda _condition, latents: (
            torch.zeros_like(latents, dtype=torch.float32),
            torch.ones(
                latents.shape[0],
                1,
                *latents.shape[2:],
                dtype=torch.float32,
                device=latents.device,
            ),
            "t2v",
        )
    )
    stage.bundle = SimpleNamespace(
        hymm_args=SimpleNamespace(extend_latent_channels=True),
        model=SimpleNamespace(diffusion_pipeline=pipeline),
    )
    latents = torch.zeros(1, 48, 2, 3, 4, dtype=torch.bfloat16)

    cond_latents, cond_mask = stage._prep_channel_cond({}, latents)

    assert cond_latents.dtype == torch.bfloat16
    assert cond_mask.dtype == torch.bfloat16


@pytest.mark.parametrize("layer_name", ["FinalLayer", "AudioFinalLayer"])
def test_leo2_final_layer_casts_fp32_norm_output_to_weight_dtype(
    layer_name: str,
) -> None:
    final_layer_type = type(layer_name, (nn.Module,), {})
    final_layer = final_layer_type()
    final_layer.linear = nn.Linear(4, 2, dtype=torch.bfloat16)
    model = nn.Module()
    model.final_layer = final_layer
    _patch_router_dtype(model)

    output = model.final_layer.linear(torch.ones(1, 4, dtype=torch.float32))

    assert output.dtype == torch.bfloat16


def test_leo2_dcp_loader_selects_local_expert_slice(tmp_path: Path) -> None:
    class ExpertBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.Module()
            self.experts.weight = nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))

    class TinyLeo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([ExpertBlock()])
            self.dense = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

    checkpoint_dir = tmp_path / "weights"
    full_experts = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    dense = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    dcp.save(
        {
            "model": {
                "layers.0.experts.weight": full_experts,
                "dense.weight": dense,
            }
        },
        checkpoint_id=str(checkpoint_dir),
    )
    model = TinyLeo()

    _dcp_load_into(
        model,
        str(checkpoint_dir),
        model_dtype=torch.bfloat16,
        expert_parallel_size=2,
        expert_parallel_rank=1,
    )

    torch.testing.assert_close(model.layers[0].experts.weight, full_experts[2:4])
    torch.testing.assert_close(model.dense.weight, dense)


def test_leo2_nft_recipe_matches_requested_contract() -> None:
    path = Path(__file__).parents[1] / "examples/diffusion/leo2/leo2_t2v_nft.yaml"
    config = OmegaConf.load(path)

    assert config.algorithm._target_ == "unirl.algorithms.diffusionnft.DiffusionNFT"
    assert config.backend.ema_lora_cfg.rank == 64
    assert config.backend.ema_lora_cfg.alpha == 128
    assert config.backend.ema_lora_cfg.ema_decay_schedule == "piecewise_linear"
    assert config.backend.ema_lora_cfg.flat_steps == 0
    assert config.backend.ema_lora_cfg.ramp_rate == pytest.approx(0.001)
    assert config.backend.ema_lora_cfg.ema_decay == pytest.approx(0.5)
    assert config.backend.ema_lora_cfg.ema_update_interval == 1
    assert config.backend.ema_lora_cfg.ema_device == "cuda"
    assert config.backend.optimizer_cfg.learning_rate == pytest.approx(3e-4)
    assert config.batch_size == 16
    assert config.rollout_chunk_prompts == 4
    assert config.sampling.num_inference_steps == 10
    assert config.sampling.guidance_scale == pytest.approx(1.0)
    assert (config.sampling.height, config.sampling.width, config.sampling.num_frames) == (464, 848, 17)
    assert config.sampling.samples_per_prompt == 8
    assert config.sampling.scheduler.num_sde_steps == 0
    assert config.algorithm.beta == pytest.approx(1.0)
    assert config.algorithm.train_timestep_mode == "random"
    assert config.algorithm.num_train_timesteps == 2
    assert config.algorithm.timestep_sampling == "logit_normal"
    assert config.algorithm.timestep_shift == pytest.approx(9.0)
    assert config.algorithm.training_timestep_fraction == pytest.approx(1.0)
    assert config.logging.log_media is True
    assert config.bundle.config.context_parallel_size == 2
    assert config.bundle.config.expert_parallel_size == 8
    assert config.bundle.config.enable_deepep is True
    assert config.bundle.config.text_encoder_gpu_transient is False
    assert config.bundle.config.condition_cache_size == 1
    assert config.bundle.config.profile_forward is False
    assert config.bundle.config.inference_cache_method == "first_block"
    assert config.bundle.config.inference_cache_threshold == pytest.approx(0.1)
    assert config.bundle.config.enable_audio is True
    assert config.bundle.config.audio_shift == pytest.approx(3.0)
    assert config.bundle.config.audio_joint_sde is False
    assert config.bundle.config.audio_vae_on_gpu is True
    assert config.backend.fsdp_cfg.sp_size == 2
    assert config.backend.fsdp_cfg.ep_size == 8


def test_leo2_context_ir_flowgrpo_recipe_matches_requested_contract() -> None:
    path = Path(__file__).parents[1] / "examples/diffusion/leo2/leo2_t2v_flowgrpo_context_ir.yaml"
    config = OmegaConf.load(path)

    assert config.num_devices == 64
    assert config.num_groups == 32
    assert config.group_size == 16
    assert config.bundle.config.context_parallel_size == 2
    assert config.bundle.config.expert_parallel_size == 1
    assert config.backend.optimizer_cfg.learning_rate == pytest.approx(3e-4)
    assert config.backend.lora_cfg.rank == 128
    assert config.backend.lora_cfg.alpha == 256
    assert config.sampling.num_inference_steps == 12
    assert config.sampling.num_frames == 9
    assert config.sampling.sde_indices is None

    scheduler = instantiate(config.sampling.scheduler)
    for rollout_id in range(8):
        selected = scheduler.get_sde_indices(rollout_id)
        assert len(selected) == 2
        assert selected <= set(range(5))


def test_leo2_motion_bilingual_flowgrpo_recipe_matches_requested_contract() -> None:
    path = Path(__file__).parents[1] / "examples/diffusion/leo2/leo2_t2v_flowgrpo_motion_bilingual.yaml"
    config = OmegaConf.load(path)

    assert config.num_devices == 64
    assert config.num_groups == 32
    assert config.group_size == 16
    assert config.save_interval == 20
    assert config.bundle.config.video_shift == pytest.approx(12.0)
    assert config.bundle.config.enable_audio is True
    assert config.bundle.config.audio_joint_sde is True
    assert config.bundle.config.context_parallel_size == 2
    assert config.bundle.config.expert_parallel_size == 1
    assert config.backend.optimizer_cfg.learning_rate == pytest.approx(3e-4)
    assert config.backend.lora_cfg.rank == 128
    assert config.backend.lora_cfg.alpha == 256
    assert config.backend.fsdp_cfg.activation_checkpointing is True
    assert config.stack.num_updates_per_batch == 2
    assert config.algorithm.old_logp_source == "rollout"
    assert config.sampling.guidance_scale == pytest.approx(1.0)
    assert config.sampling.num_inference_steps == 12
    assert config.sampling.num_frames == 9
    assert config.sampling.sde_indices is None

    scheduler = instantiate(config.sampling.scheduler)
    for rollout_id in range(16):
        selected = scheduler.get_sde_indices(rollout_id)
        assert len(selected) == 2
        assert selected <= set(range(5))


def test_leo2_native_parity_recipe_matches_native_grpo_contract() -> None:
    path = Path(__file__).parents[1] / "examples/diffusion/leo2/leo2_t2v_native_parity.yaml"
    config = OmegaConf.load(path)

    assert config.num_devices == 32
    assert config.num_groups == 8
    assert config.group_size == 8
    assert config.batch_size == 8
    assert config.rollout.forward_batch_size == 2
    assert config.bundle.config.video_shift == pytest.approx(7.0)
    assert config.bundle.config.audio_shift == pytest.approx(1.0)
    assert config.bundle.config.context_parallel_size == 4
    assert config.bundle.config.expert_parallel_size == 1
    assert config.bundle.config.audio_stochastic_rollout is True
    assert config.bundle.config.audio_joint_sde is False
    assert config.bundle.config.native_rng_compat is True
    assert config.bundle.config.reproduce is True
    assert config.bundle.config.attention_impl == "flash_packed"
    assert config.bundle.config.uniform_bf16 is False
    assert config.bundle.config.full_model_training is True
    assert config.backend.lora_cfg is None
    assert config.backend.optimizer_cfg.type == "leo2_native_muon"
    assert config.backend.optimizer_cfg.momentum == pytest.approx(0.95)
    assert "*final_layer*" in config.backend.optimizer_cfg.special_adamw_params
    assert config.backend.fsdp_cfg.fsdp_mode == "full"
    assert config.backend.fsdp_cfg.sp_size == 4
    assert config.backend.fsdp_cfg.master_dtype == "fp32"
    assert list(config.backend.fsdp_cfg.fp32_module_suffixes) == ["mlp.gate"]
    assert config.backend.fsdp_cfg.checkpoint_format == "dcp"
    assert config.sampling.num_inference_steps == 30
    assert config.sampling.guidance_scale == pytest.approx(1.0)
    assert (config.sampling.height, config.sampling.width, config.sampling.num_frames) == (480, 848, 121)
    assert config.sampling.eta == pytest.approx(0.5)
    assert list(config.sampling.sde_indices) == [0, 1, 2, 3, 4]
    assert config.sampling.samples_per_prompt == 8
    assert config.sampling.init_same_noise is True
    assert config.algorithm.clip_range == pytest.approx(1.0e-4)
    assert config.algorithm.use_grpo_guard is True
    assert config.algorithm.adv_clip_max == pytest.approx(5.0)
    assert config.algorithm.old_logp_source == "rollout"
    assert config.algorithm.max_rollout_replay_logp_absdiff == pytest.approx(0.0)


def test_leo2_native_optimizer_spec_matches_muon_adamw_partition() -> None:
    from unirl.models.leo2.optimizer import _native_optimizer_spec

    class FakeMuon:
        pass

    config = SimpleNamespace(
        learning_rate=1.0e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        weight_decay=0.01,
        momentum=0.95,
        special_adamw_params=("*final_layer*", "*embedding*"),
        special_weight_decay_params=(),
    )
    matrix = torch.nn.Parameter(torch.zeros(8, 4))
    bias = torch.nn.Parameter(torch.zeros(8))

    optimizer_cls, kwargs = _native_optimizer_spec("layers.0.self_attn.q_proj.weight", matrix, config, FakeMuon)
    assert optimizer_cls is FakeMuon
    assert kwargs == {
        "lr": pytest.approx(1.0e-5),
        "weight_decay": pytest.approx(0.01),
        "momentum": pytest.approx(0.95),
    }

    optimizer_cls, kwargs = _native_optimizer_spec("final_layer.linear.weight", matrix, config, FakeMuon)
    assert optimizer_cls is torch.optim.AdamW
    assert kwargs["weight_decay"] == pytest.approx(0.01)
    assert kwargs["betas"] == pytest.approx((0.9, 0.999))
    assert kwargs["eps"] == pytest.approx(1.0e-8)

    optimizer_cls, kwargs = _native_optimizer_spec("layers.0.self_attn.q_proj.bias", bias, config, FakeMuon)
    assert optimizer_cls is torch.optim.AdamW
    assert kwargs["weight_decay"] == pytest.approx(0.0)


def test_optimizer_factory_delegates_to_model_bundle() -> None:
    from unirl.train.backend.base import OptimizerConfig
    from unirl.train.optim import build_optimizer

    sentinel = object()

    class Bundle:
        def build_optimizer(self, *, config, model):
            assert config.learning_rate == pytest.approx(1.0e-5)
            assert model == "model"
            return sentinel

    config = OptimizerConfig(
        learning_rate=1.0e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1.0e-8,
        weight_decay=0.01,
    )

    assert build_optimizer(config, params=(), actor=Bundle(), model="model") is sentinel


def test_leo2_preprocess_preserves_multiline_prompt_jsonl(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text('{"prompt_id":"zh:1","prompt":"first line\\nsecond line","language":"zh"}\n')

    records = _records([str(manifest)], encode_targets=False)

    assert len(records) == 1
    assert records[0]["prompt_id"] == "zh:1"
    assert records[0]["prompt"] == "first line\nsecond line"


def test_ema_piecewise_linear_matches_flow_factory_schedule() -> None:
    decay = make_decay_fn(
        EmaLoraConfig(
            ema_decay_schedule="piecewise_linear",
            flat_steps=0,
            ramp_rate=0.001,
            ema_decay=0.5,
        )
    )

    assert decay(0) == pytest.approx(0.0)
    assert decay(1) == pytest.approx(0.001)
    assert decay(499) == pytest.approx(0.499)
    assert decay(500) == pytest.approx(0.5)
    assert decay(1000) == pytest.approx(0.5)


def test_ema_update_interval_uses_zero_based_flow_factory_steps() -> None:
    live = torch.tensor([2.0])
    shadow = torch.tensor([0.0])
    ema = EMA(
        shadow=Shadow(
            iter_pairs=lambda: iter([(live, shadow)]),
            swap_in=lambda: None,
            swap_out=lambda: None,
        ),
        decay_fn=lambda step: 0.5,
        timing="rollout_end",
        update_interval=2,
    )

    ema.on_rollout_end(0)
    torch.testing.assert_close(shadow, torch.tensor([0.0]))
    ema.on_rollout_end(1)
    torch.testing.assert_close(shadow, torch.tensor([1.0]))


def test_rollout_end_ema_uses_last_committed_optimizer_index() -> None:
    observed: list[int] = []
    backend = object.__new__(BaseFSDP2Backend)
    backend.ema = SimpleNamespace(on_rollout_end=observed.append)
    backend._optimizer_step_count = 1

    backend.on_rollout_end()

    assert observed == [0]


def test_chunked_global_advantages_match_full_batch() -> None:
    branch = 4
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 1.0, 4.0, 2.0, 6.0, 8.0, 5.0, 7.0, 9.0])
    sample_ids = [f"group-{group}/{sample}" for group in range(4) for sample in range(branch)]
    full = Part(sample_ids=sample_ids, rewards=rewards)
    expected = full.compute_advantages(normalize=True, use_global_std=True)
    chunks = [
        Part(sample_ids=sample_ids[: 2 * branch], rewards=rewards[: 2 * branch]),
        Part(sample_ids=sample_ids[2 * branch :], rewards=rewards[2 * branch :]),
    ]

    actual = _compute_chunked_advantages(
        chunks,
        use_global_std=True,
        min_group_std=0.0,
    )

    torch.testing.assert_close(
        torch.cat([part.advantages for part in actual]),
        expected.advantages,
    )


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (True, TypeError, "int or None"),
        (0, ValueError, r"\[1, batch_size=48\]"),
        (5, ValueError, "must be divisible"),
        (2, ValueError, "rollout dp_size=4"),
    ],
)
def test_rollout_chunk_prompt_validation_fails_fast(
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _resolve_rollout_chunk_prompts(
            value,
            batch_size=48,
            samples_per_prompt=16,
            rollout_dp_size=4,
            reward_dp_size=8,
            train_dp_size=4,
            num_updates_per_batch=1,
        )


def test_trainside_rollout_transports_only_dp_collect_head() -> None:
    sample = Sample(parts=[])

    non_head = object.__new__(TrainsideRolloutEngine)
    non_head.rank_info = RankInfo(sp_rank=1, sp_size=2)
    non_head._generate_locked = MethodType(lambda self, value: value, non_head)
    assert non_head.generate(sample) is None

    head = object.__new__(TrainsideRolloutEngine)
    head.rank_info = RankInfo(sp_rank=0, sp_size=2)
    head._generate_locked = MethodType(lambda self, value: value, head)
    assert head.generate(sample) is sample


def test_train_stack_chunk_window_steps_only_on_final_part() -> None:
    stack = object.__new__(TrainStack)
    stack.fsdp_backend = SimpleNamespace(zero_grad=lambda: None)
    stack._align_track_inputs = MethodType(lambda self, part: part, stack)
    stack._prepare_for_training = MethodType(lambda self, part, plans: part, stack)
    step_flags: list[bool] = []

    def run_update(
        self,
        part,
        *,
        micros,
        training_progress,
        zero_grad,
        do_optimizer_step,
        loss_weight,
        prior_backward,
    ):
        step_flags.append(do_optimizer_step)
        return TrainStepResult(
            loss=1.0,
            grad_norm=1.0 if do_optimizer_step else 0.0,
            lr=3e-4,
            has_backward=True,
            micros=[],
            metrics={},
            optimizer_updates=int(do_optimizer_step),
        )

    stack._run_update = MethodType(run_update, stack)
    part = Part(sample_ids=["group/0"], advantages=torch.ones(1))
    result = stack._run_window(
        [(part, (((0, 1),),)), (part, (((0, 1),),))],
        training_progress=0.0,
    )

    assert step_flags == [False, True]
    assert result.optimizer_updates == 1
    assert result.grad_norm == pytest.approx(1.0)


def test_leo2_first_block_cache_config_uses_requested_threshold() -> None:
    config = _make_inference_cache_config(
        SimpleNamespace(
            inference_cache_method="first_block",
            inference_cache_threshold=0.1,
        )
    )

    assert type(config).__name__ == "FirstBlockCacheConfig"
    assert config.threshold == pytest.approx(0.1)


def test_leo2_condition_cache_reuses_sibling_prompt_without_sharing_containers() -> None:
    stage = object.__new__(Leo2CondStage)
    stage._cache_size = 1
    stage._cache = OrderedDict()
    stage.cache_hits = 0
    stage.cache_misses = 0
    calls = 0
    tensor = torch.ones(1, 2)

    def build_uncached(self, texts, *, height, width, num_frames, seeds):
        nonlocal calls
        calls += 1
        return Leo2Conditions(
            text=TextEmbedCondition(embeds=tensor),
            hymm=[
                {
                    "input_ids": tensor,
                    "model_kwargs": {
                        "attention_mask": tensor,
                        "rope_media_info": [],
                        "cond_text_states": tensor,
                        "cond_text_mask": tensor,
                        "visual_mask": tensor,
                        "text_mask": tensor,
                        "timesteps_index": None,
                        "audio_mask": None,
                        "und_token_indices": tensor,
                        "gen_token_indices": tensor,
                        "audio_token_indices": None,
                    },
                    "image_size": (height, width),
                    "video_duration": num_frames,
                }
            ],
        )

    stage._build_uncached = MethodType(build_uncached, stage)
    kwargs = dict(height=464, width=848, num_frames=17, seeds=[42])
    first = stage.build(Texts(texts=["prompt"]), **kwargs)
    second = stage.build(Texts(texts=["prompt"]), **kwargs)
    first.hymm[0]["_device"] = "cuda:0"

    assert calls == 1
    assert stage.cache_misses == 1
    assert stage.cache_hits == 1
    assert "_device" not in second.hymm[0]
    assert first.hymm[0] is not second.hymm[0]
    assert first.hymm[0]["input_ids"].untyped_storage().data_ptr() == (
        second.hymm[0]["input_ids"].untyped_storage().data_ptr()
    )


def test_diffusion_nft_draws_two_shifted_logit_normal_training_steps() -> None:
    algorithm = DiffusionNFT(
        params=SimpleNamespace(),
        stage=SimpleNamespace(),
        nft_lora_policy=SimpleNamespace(use_shadow=lambda: nullcontext()),
        train_timestep_mode="random",
        num_train_timesteps=2,
        timestep_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        timestep_shift=3.0,
        sigma_min=1e-4,
        shuffle_train_timesteps=False,
        training_timestep_fraction=1.0,
    )
    torch.manual_seed(17)
    actual = algorithm._resolve_timesteps(
        LatentSegment(),
        torch.device("cpu"),
        torch.float32,
    )

    torch.manual_seed(17)
    unit = torch.sigmoid(torch.randn(2, dtype=torch.float32))
    expected = (3.0 * unit / (1.0 + 2.0 * unit)).clamp(min=1e-4, max=1.0 - 1e-4)

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_train_timesteps", "exception", "message"),
    [
        (True, TypeError, "expected int"),
        (0, ValueError, "must be positive"),
    ],
)
def test_diffusion_nft_rejects_invalid_training_step_count(
    num_train_timesteps: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        DiffusionNFT(
            params=SimpleNamespace(),
            stage=SimpleNamespace(),
            nft_lora_policy=SimpleNamespace(use_shadow=lambda: nullcontext()),
            train_timestep_mode="random",
            num_train_timesteps=num_train_timesteps,
        )


def test_leo2_transport_tree_normalizes_numpy_slice_bounds() -> None:
    result = _to_transport_tree(
        {"rope_media_info": [[slice(np.int32(1), np.int64(5), None)]]},
        path="model_kwargs",
    )

    bound = result["rope_media_info"][0][0]
    assert bound == slice(1, 5, None)
    assert type(bound.start) is int
    assert type(bound.stop) is int


def test_leo2_transport_tree_rejects_numpy_arrays_with_path() -> None:
    with pytest.raises(
        TypeError,
        match=r"model_kwargs\.rope_media_info contains non-transportable numpy\.ndarray",
    ):
        _to_transport_tree(
            {"rope_media_info": np.array([1, 2], dtype=np.int32)},
            path="model_kwargs",
        )


def test_leo2_conditions_allow_optional_timestep_index() -> None:
    tensor = torch.zeros(1, 1)
    conditions = Leo2Conditions.from_dict(
        {
            "hymm": [
                {
                    "input_ids": tensor,
                    "model_kwargs": {
                        "attention_mask": tensor,
                        "rope_media_info": [],
                        "cond_text_states": tensor,
                        "cond_text_mask": tensor,
                        "visual_mask": tensor,
                        "text_mask": tensor,
                        "timesteps_index": None,
                        "audio_mask": None,
                        "und_token_indices": tensor,
                        "gen_token_indices": tensor,
                        "audio_token_indices": None,
                    },
                    "image_size": (192, 336),
                    "video_duration": 49,
                }
            ]
        }
    )

    assert conditions.hymm[0]["model_kwargs"]["timesteps_index"] is None


def test_leo2_pipeline_uses_effective_hymm_geometry() -> None:
    conditions = SimpleNamespace(
        hymm=[
            {"image_size": (192, 336), "video_duration": 49},
            {"image_size": (192, 336), "video_duration": 49},
        ]
    )

    assert Leo2Pipeline._latent_shape_from_conditions(conditions) == (48, 13, 12, 21)


def test_leo2_pipeline_rejects_mixed_effective_geometry() -> None:
    conditions = SimpleNamespace(
        hymm=[
            {"image_size": (192, 336), "video_duration": 49},
            {"image_size": (464, 848), "video_duration": 121},
        ]
    )

    with pytest.raises(ValueError, match="one effective media geometry"):
        Leo2Pipeline._latent_shape_from_conditions(conditions)


def test_leo2_single_condition_preserves_native_layout() -> None:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=4),
    )
    input_ids = torch.arange(12).reshape(1, 12)
    model_kwargs = {"attention_mask": torch.tensor([[12, *([0] * 11)]])}
    blob = {"input_ids": input_ids, "model_kwargs": model_kwargs}

    packed_ids, packed_kwargs, packing_kwargs = stage._model_conditions(
        [blob],
        device=torch.device("cpu"),
    )

    assert packed_ids is input_ids
    torch.testing.assert_close(packed_kwargs["attention_mask"], model_kwargs["attention_mask"])
    assert packed_ids.shape[1] == 12
    assert packing_kwargs == {}


def test_leo2_native_seed_expansion_matches_group_branch_order() -> None:
    root = Part.input(
        ["prompt-0", "prompt-1"],
        primitives={"text": Texts(texts=["a", "b"])},
        metadata=[{"seed": 100}, {"seed": 200}],
    )
    sample = Sample.request(root).fork(
        3,
        sampling_params=SimpleNamespace(init_same_noise=False),
    )

    assert Leo2Pipeline._native_seeds(sample, base_seed=42, same_noise=False) == [
        100,
        101,
        102,
        200,
        201,
        202,
    ]
    assert Leo2Pipeline._native_seeds(sample, base_seed=42, same_noise=True) == [
        100,
        100,
        100,
        200,
        200,
        200,
    ]
