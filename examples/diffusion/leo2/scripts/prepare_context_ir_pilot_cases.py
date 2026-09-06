#!/usr/bin/env python3
"""Generate one external-baseline case CSV per formal Leo2 pilot candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cache_benchmark_cases import TSV_FIELDS
from prepare_context_ir_cache_cases import (
    CFG_HIFI_WINDOWS,
    DFR_WINDOWS,
    STEPS,
    _cfg_values,
    _dfr_values,
    _row,
)


def _write(path: Path, row: dict[str, object], *, steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row["diff_infer_steps"] = steps
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _cfg_candidate(steps: int, baseline: str, baseline_root: Path) -> dict[str, object]:
    start, end = CFG_HIFI_WINDOWS[steps]
    values = _cfg_values(steps, interval=2, weight=1.0)
    values.update(
        cfg_start_step=start,
        cfg_end_step=end,
        cfg_low_frequency_start_step=start,
        cfg_low_frequency_end_step=end,
        cfg_high_frequency_start_step=start,
        cfg_high_frequency_end_step=end,
    )
    return _row(
        f"cfg_hifi_s{steps}_g5",
        "cfg_cache",
        5.0,
        baseline,
        reference_root=str(baseline_root),
        **values,
    )


def rows(steps: int, baseline_root: Path, profile_root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for guidance, suffix in ((1.0, "g1"), (5.0, "g5")):
        baseline = f"exact_s{steps}_{suffix}"
        common = {"baseline": baseline, "reference_root": str(baseline_root)}
        result[f"first_block_{suffix}"] = _row(
            f"first_block_t010_s{steps}_{suffix}",
            "first_block",
            guidance,
            common["baseline"],
            reference_root=common["reference_root"],
            cache_threshold=0.10,
        )
        result[f"taylor_{suffix}"] = _row(
            f"taylor_t010_m050_s{steps}_{suffix}",
            "taylor",
            guidance,
            common["baseline"],
            reference_root=common["reference_root"],
            cache_threshold=0.10,
            taylor_max_extrapolation=0.50,
        )
        result[f"magcache_{suffix}"] = _row(
            f"magcache_t012_k4_s{steps}_{suffix}",
            "magcache",
            guidance,
            common["baseline"],
            reference_root=common["reference_root"],
            magcache_profile=str(profile_root / f"magcache_s{steps}_{suffix}.json"),
            magcache_threshold=0.12,
            magcache_max_skip_steps=4,
            magcache_retention_ratio=0.2,
        )
        result[f"dfr_{suffix}"] = _row(
            f"dfr_s{steps}_{suffix}",
            "fastercache_dfr",
            guidance,
            common["baseline"],
            reference_root=common["reference_root"],
            **_dfr_values(
                steps,
                layers="36-47" if guidance > 1 else None,
            ),
        )
    baseline = f"exact_s{steps}_g5"
    result["cfg_g5"] = _cfg_candidate(steps, baseline, baseline_root)
    composite_cfg = {
        key: value
        for key, value in _cfg_candidate(steps, baseline, baseline_root).items()
        if key.startswith("cfg_")
    }
    if steps == 6:
        composite_cfg["cfg_interval"] = 3
    result["dfr_cfg_g5"] = _row(
        f"dfr_cfg_hifi_s{steps}_g5",
        "fastercache_dfr+cfg_cache",
        5.0,
        baseline,
        reference_root=str(baseline_root),
        **_dfr_values(steps, layers="24-47"),
        **composite_cfg,
    )
    for row in result.values():
        row["diff_infer_steps"] = steps
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=STEPS, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.baseline_root.is_absolute() or not args.profile_root.is_absolute():
        raise ValueError("baseline and profile roots must be absolute")
    for name, row in rows(args.steps, args.baseline_root, args.profile_root).items():
        _write(args.output_dir / f"s{args.steps}_{name}.csv", row, steps=args.steps)


if __name__ == "__main__":
    main()
