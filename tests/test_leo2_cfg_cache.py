from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

VENDOR_ROOT = Path(__file__).parents[1] / "unirl/models/leo2/vendor/gen_ar"
for path in (VENDOR_ROOT, VENDOR_ROOT / "deps/hy_parallelism", VENDOR_ROOT / "deps/IndexKits"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hymm.models.diffusion.leo_cache import (  # noqa: E402
    LeoCFGCacheConfig,
    LeoCFGCacheController,
    LeoCombinedCacheConfig,
    LeoFasterCacheConfig,
    LeoFasterCacheController,
)


def _controller(*, steps: int = 6, interval: int = 5) -> LeoCFGCacheController:
    return LeoCFGCacheController(
        LeoCFGCacheConfig(
            start_step=1,
            end_step=steps,
            interval=interval,
            low_frequency_weight=1.0,
            high_frequency_weight=1.0,
            low_frequency_start_step=1,
            low_frequency_end_step=steps,
            high_frequency_start_step=1,
            high_frequency_end_step=steps,
        )
    )


def test_cfg_cache_reconstructs_conditional_first_exact_delta() -> None:
    controller = _controller()
    reference = torch.zeros(2, 1, 2, 4, 4)
    conditional = torch.randn(1, 3, 2, 4, 4, dtype=torch.bfloat16)
    unconditional = torch.randn_like(conditional)

    with torch.no_grad(), controller.context("test"):
        assert controller.begin_step(guidance_enabled=True, reference=reference) is False
        controller.record_exact(conditional, unconditional)
        assert controller.begin_step(guidance_enabled=True, reference=reference) is True
        reconstructed = controller.reconstruct_unconditional(conditional)

    torch.testing.assert_close(reconstructed, unconditional, atol=2e-2, rtol=2e-2)
    stats = controller.stats()
    assert stats["cfg_compute_calls"] == 1
    assert stats["cfg_reuse_calls"] == 1
    assert stats["cache_bytes"] > 0


def test_cfg_cache_six_step_accounting() -> None:
    controller = _controller(steps=6, interval=5)
    reference = torch.zeros(2, 1, 2, 4, 4)
    conditional = torch.randn(1, 2, 2, 4, 4)
    unconditional = torch.randn_like(conditional)

    with torch.no_grad(), controller.context("test"):
        for _ in range(6):
            reuse = controller.begin_step(guidance_enabled=True, reference=reference)
            if reuse:
                controller.reconstruct_unconditional(conditional)
            else:
                controller.record_exact(conditional, unconditional)

    stats = controller.stats()
    assert stats["cfg_compute_calls"] == 2
    assert stats["cfg_reuse_calls"] == 4


def test_cfg_cache_guidance_one_is_a_strict_bypass() -> None:
    controller = _controller()
    with controller.context("test"):
        for _ in range(6):
            assert (
                controller.begin_step(
                    guidance_enabled=False,
                    reference=torch.zeros(1, 1, 2, 4, 4),
                )
                is False
            )
    assert controller.stats()["cfg_compute_calls"] == 0
    assert controller.stats()["cfg_reuse_calls"] == 0


def test_cfg_cache_selects_branch_zero_from_model_inputs() -> None:
    inputs = {
        "latents": torch.arange(4).reshape(2, 2),
        "timesteps": torch.tensor([9.0, 9.0]),
        "cond_text_states": torch.arange(8).reshape(2, 2, 2),
        "rope_media_info": [["conditional"], ["unconditional"]],
        "shared": torch.ones(3),
        "optional": None,
    }

    selected = LeoCFGCacheController.conditional_inputs(inputs)

    torch.testing.assert_close(selected["latents"], inputs["latents"][:1])
    torch.testing.assert_close(selected["timesteps"], inputs["timesteps"][:1])
    torch.testing.assert_close(selected["cond_text_states"], inputs["cond_text_states"][:1])
    assert selected["rope_media_info"] == [["conditional"]]
    assert selected["shared"] is inputs["shared"]
    assert selected["optional"] is None


def test_fastercache_canonicalizes_full_cfg_histories_to_conditional_branch() -> None:
    controller = LeoFasterCacheController(
        LeoFasterCacheConfig(start_step=0, end_step=6, interval=2),
        num_layers=1,
    )
    controller.set_cfg_mode(enabled=True, conditional_only=False)
    conditional = torch.randn(1, 3, 4)
    unconditional = torch.randn_like(conditional)
    canonical = controller._canonical_cfg_streams((torch.cat([conditional, unconditional]), None))

    torch.testing.assert_close(canonical[0], conditional)
    assert canonical[1] is None


def test_combined_cache_requires_dfr_and_cfg_configs() -> None:
    config = LeoCombinedCacheConfig(
        feature=LeoFasterCacheConfig(start_step=1, end_step=5, interval=2),
        cfg=LeoCFGCacheConfig(start_step=1, end_step=6, interval=5),
    )
    assert config.feature.interval == 2
    assert config.cfg.interval == 5

    with pytest.raises(TypeError, match="feature"):
        LeoCombinedCacheConfig(feature=object(), cfg=config.cfg)
