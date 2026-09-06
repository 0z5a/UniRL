from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS = REPO_ROOT / "examples/diffusion/leo2/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cache_benchmark_cases import TSV_FIELDS, load_cases  # noqa: E402
from cache_benchmark_entry import (  # noqa: E402
    BenchmarkOptions,
    PromptIdentity,
    _resolve_prompt_identity,
    _validate_cache_stats,
)
from compute_pixel_metrics import _load_prompts  # noqa: E402
from build_magcache_profile import _load_prompts as _load_magcache_prompts  # noqa: E402
from prepare_context_ir_pilot_cases import rows as _pilot_rows  # noqa: E402


def _case_row(**updates: object) -> dict[str, object]:
    row = {field: "" for field in TSV_FIELDS}
    row.update(
        name="case",
        method="off",
        cache_threshold="off",
        flow_shift_video=9.0,
        guidance_scale=5.0,
        baseline_case="exact",
    )
    row.update(updates)
    return row


def test_cfg_case_schema_supports_standalone_and_combined(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv"
    rows = [
        _case_row(
            name="cfg",
            method="cfg_cache",
            cfg_start_step=1,
            cfg_end_step=6,
            cfg_interval=5,
        ),
        _case_row(
            name="combined",
            method="fastercache_dfr+cfg_cache",
            dfr_start_step=1,
            dfr_end_step=5,
            dfr_interval=2,
            cfg_start_step=1,
            cfg_end_step=6,
            cfg_interval=5,
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cfg, combined = load_cases(path)
    assert cfg.method == "cfg_cache"
    assert cfg.cfg_end_step == 6
    assert cfg.cfg_low_frequency_weight == pytest.approx(1.1)
    assert combined.method == "fastercache_dfr+cfg_cache"
    assert combined.dfr_end_step == 5


def test_cfg_counter_validation_uses_actual_step_count() -> None:
    stats = {
        "full_steps": 0,
        "skipped_steps": 0,
        "tail_compute_steps": 0,
        "tail_reuse_steps": 0,
        "predicted_steps": 0,
        "prediction_warmup_steps": 0,
        "static_fallback_steps": 0,
        "attention_compute_calls": 0,
        "attention_reuse_calls": 0,
        "cfg_compute_calls": 2,
        "cfg_reuse_calls": 4,
        "cache_bytes": 1,
    }
    _validate_cache_stats(
        stats,
        method="cfg_cache",
        expected_steps=6,
        guidance_scale=5.0,
    )
    with pytest.raises(RuntimeError, match="5/6"):
        _validate_cache_stats(
            {**stats, "cfg_reuse_calls": 3},
            method="cfg_cache",
            expected_steps=6,
            guidance_scale=5.0,
        )


def test_duplicate_bilingual_seed_resolves_by_prompt_hash() -> None:
    en = PromptIdentity(0, "unused", "en", "pair", "vbench", "source")
    zh = PromptIdentity(1, "unused", "zh", "pair", "vbench", "source")
    import hashlib

    en = PromptIdentity(en.index, hashlib.sha256(b"hello").hexdigest(), en.language, en.pair_id, en.source_dataset, en.source_id)
    zh = PromptIdentity(zh.index, hashlib.sha256("你好".encode()).hexdigest(), zh.language, zh.pair_id, zh.source_dataset, zh.source_id)
    options = object.__new__(BenchmarkOptions)
    object.__setattr__(options, "prompts_by_seed", {52000: (en, zh)})

    actual = _resolve_prompt_identity(
        options,
        seed=52000,
        args=(),
        kwargs={"prompt": "你好"},
    )
    assert actual.language == "zh"
    assert actual.index == 1


def test_committed_bilingual_manifests_preserve_pair_seed_identity() -> None:
    data_dir = REPO_ROOT / "examples/diffusion/leo2/data"
    import json

    contract = json.loads(
        (data_dir / "context_ir_bilingual_cache_benchmark.manifest.json").read_text()
    )["video_contract"]
    assert contract == {
        "duration_seconds": 8,
        "fps": 24,
        "frame_formula": "duration_seconds * fps + 1",
        "num_frames": 193,
    }
    expected = {
        "context_ir_bilingual_smoke_1.csv": 1,
        "context_ir_bilingual_pilot_32.csv": 32,
        "context_ir_bilingual_calibration_16.csv": 16,
        "context_ir_bilingual_full_200.csv": 200,
    }
    for filename, count in expected.items():
        with (data_dir / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count
        by_pair: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], []).append(row)
            assert row["language"] in {"en", "zh"}
            assert row["prompt"]
        for pair_rows in by_pair.values():
            if count > 1:
                assert {row["language"] for row in pair_rows} == {"en", "zh"}
            assert len({row["seed"] for row in pair_rows}) == 1

    pixel_prompts = _load_prompts(data_dir / "context_ir_bilingual_pilot_32.csv")
    assert len(pixel_prompts) == 32
    assert len({prompt.seed for prompt in pixel_prompts}) == 16
    magcache_prompts = _load_magcache_prompts(
        data_dir / "context_ir_bilingual_pilot_32.csv"
    )
    assert len(magcache_prompts) == 32
    assert len({seed for seed, _ in magcache_prompts}) == 16


def test_formal_pilot_cases_use_external_baselines_and_hifi_cfg_window() -> None:
    baseline = Path("/shared/baseline")
    generated = _pilot_rows(6, baseline, Path("/shared/profiles"))
    assert len(generated) == 10
    assert all(row["reference_root"] == str(baseline) for row in generated.values())
    assert generated["cfg_g5"]["cfg_start_step"] == 3
    assert generated["cfg_g5"]["cfg_end_step"] == 5
    assert generated["cfg_g5"]["cfg_interval"] == 2
    assert generated["dfr_cfg_g5"]["dfr_start_step"] == 1
    assert generated["dfr_cfg_g5"]["dfr_layers"] == "24-47"
    assert generated["dfr_cfg_g5"]["cfg_interval"] == 3
