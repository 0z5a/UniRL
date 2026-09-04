from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from omegaconf import OmegaConf

from unirl.algorithms.diffusionnft import DiffusionNFT
from unirl.models.leo2.bundle import _dcp_load_into, _patch_router_dtype
from unirl.models.leo2.conditions import Leo2Conditions
from unirl.models.leo2.diffusion import Leo2DiffusionStage
from unirl.models.leo2.pipeline import Leo2Pipeline
from unirl.models.leo2.text_embed import _to_transport_tree
from unirl.types.segments.latent import LatentSegment


def _fake_stage() -> tuple[Leo2DiffusionStage, dict[str, object]]:
    stage = object.__new__(Leo2DiffusionStage)
    stage.bundle = SimpleNamespace(device=torch.device("cpu"))
    stage.trajectory_dtype = torch.float32
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
        "predict_blob": blob,
        "sigma": 0.5,
        "channel_cond": ("cond", "mask"),
    }


@pytest.mark.parametrize(
    ("conditions", "sample", "sigma", "guidance_scale", "message"),
    [
        (SimpleNamespace(hymm=[]), torch.zeros(1, 48, 2, 3, 4), torch.tensor([0.5]), 1.0, "condition blob"),
        (
            SimpleNamespace(hymm=[{}]),
            torch.zeros(2, 48, 2, 3, 4),
            torch.tensor([0.5, 0.6]),
            1.0,
            "batch-1",
        ),
        (
            SimpleNamespace(hymm=[{}]),
            torch.zeros(1, 48, 2, 3, 4),
            torch.tensor([0.5, 0.6]),
            1.0,
            "one sigma",
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


def test_leo2_final_layer_casts_fp32_norm_output_to_weight_dtype() -> None:
    final_layer_type = type("FinalLayer", (nn.Module,), {})
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
    assert config.backend.optimizer_cfg.learning_rate == pytest.approx(3e-4)
    assert config.batch_size == 48
    assert config.sampling.num_inference_steps == 10
    assert config.sampling.guidance_scale == pytest.approx(1.0)
    assert (config.sampling.height, config.sampling.width, config.sampling.num_frames) == (464, 848, 121)
    assert config.sampling.samples_per_prompt == 16
    assert config.sampling.scheduler.num_sde_steps == 0
    assert config.algorithm.beta == pytest.approx(1.0)
    assert config.algorithm.train_timestep_mode == "random"
    assert config.algorithm.num_train_timesteps == 2
    assert config.algorithm.timestep_sampling == "logit_normal"
    assert config.algorithm.timestep_shift == pytest.approx(3.0)
    assert config.algorithm.training_timestep_fraction == pytest.approx(1.0)
    assert config.logging.log_media is True
    assert config.bundle.config.context_parallel_size == 2
    assert config.bundle.config.expert_parallel_size == 8
    assert config.bundle.config.enable_deepep is True
    assert config.backend.fsdp_cfg.sp_size == 2
    assert config.backend.fsdp_cfg.ep_size == 8


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
