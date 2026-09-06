#!/usr/bin/env python3
"""Generate the fixed shift-9 Context-IR cache benchmark case matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cache_benchmark_cases import TSV_FIELDS

STEPS = (6, 28, 50)
DFR_WINDOWS = {6: (1, 5), 28: (6, 27), 50: (10, 49)}
CFG_HIFI_WINDOWS = {6: (3, 5), 28: (20, 27), 50: (35, 49)}


def _row(name: str, method: str, guidance: float, baseline: str, **values: object) -> dict[str, object]:
    row = {field: "" for field in TSV_FIELDS}
    row.update(
        name=name,
        method=method,
        cache_threshold="off",
        flow_shift_video=9.0,
        guidance_scale=guidance,
        baseline_case=baseline,
    )
    row.update(values)
    return row


def _cfg_values(steps: int, *, interval: int = 5, weight: float = 1.1) -> dict[str, object]:
    return {
        "cfg_start_step": 1,
        "cfg_end_step": steps,
        "cfg_interval": interval,
        "cfg_low_frequency_weight": weight,
        "cfg_high_frequency_weight": weight,
        "cfg_low_frequency_start_step": 1,
        "cfg_low_frequency_end_step": steps,
        "cfg_high_frequency_start_step": 1,
        "cfg_high_frequency_end_step": steps,
    }


def _dfr_values(steps: int) -> dict[str, object]:
    start, end = DFR_WINDOWS[steps]
    return {
        "dfr_start_step": start,
        "dfr_end_step": end,
        "dfr_interval": 2,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _matrix(steps: int, profile_root: Path) -> list[dict[str, object]]:
    rows = []
    for guidance, suffix in ((1.0, "g1"), (5.0, "g5")):
        baseline = f"exact_s{steps}_{suffix}"
        rows.extend(
            [
                _row(baseline, "off", guidance, baseline),
                _row(
                    f"first_block_t010_s{steps}_{suffix}",
                    "first_block",
                    guidance,
                    baseline,
                    cache_threshold=0.10,
                ),
                _row(
                    f"taylor_t010_m050_s{steps}_{suffix}",
                    "taylor",
                    guidance,
                    baseline,
                    cache_threshold=0.10,
                    taylor_max_extrapolation=0.50,
                ),
                _row(
                    f"magcache_t012_k4_s{steps}_{suffix}",
                    "magcache",
                    guidance,
                    baseline,
                    magcache_profile=str(profile_root / f"magcache_s{steps}_{suffix}.json"),
                    magcache_threshold=0.12,
                    magcache_max_skip_steps=4,
                    magcache_retention_ratio=0.2,
                ),
                _row(
                    f"dfr_s{steps}_{suffix}",
                    "fastercache_dfr",
                    guidance,
                    baseline,
                    **_dfr_values(steps),
                ),
            ]
        )
        if guidance > 1:
            rows.extend(
                [
                    _row(
                        f"cfg_cache_s{steps}_{suffix}",
                        "cfg_cache",
                        guidance,
                        baseline,
                        **_cfg_values(steps),
                    ),
                    _row(
                        f"dfr_cfg_s{steps}_{suffix}",
                        "fastercache_dfr+cfg_cache",
                        guidance,
                        baseline,
                        **_dfr_values(steps),
                        **_cfg_values(steps),
                    ),
                ]
            )
    return rows


def _cfg_grid(steps: int) -> list[dict[str, object]]:
    baseline = f"exact_s{steps}_g5"
    rows = [_row(baseline, "off", 5.0, baseline)]
    for interval in (2, 3, 5):
        for weight in (1.0, 1.1):
            rows.append(
                _row(
                    f"cfg_i{interval}_w{int(weight * 100):03d}_s{steps}_g5",
                    "cfg_cache",
                    5.0,
                    baseline,
                    **_cfg_values(steps, interval=interval, weight=weight),
                )
            )
    return rows


def _cfg_window_grid(steps: int) -> list[dict[str, object]]:
    baseline = f"exact_s{steps}_g5"
    start, end = DFR_WINDOWS[steps]
    rows = [_row(baseline, "off", 5.0, baseline)]
    for interval in (2, 3, 5):
        values = _cfg_values(steps, interval=interval, weight=1.0)
        values.update(
            cfg_start_step=start,
            cfg_end_step=end,
            cfg_low_frequency_start_step=start,
            cfg_low_frequency_end_step=end,
            cfg_high_frequency_start_step=start,
            cfg_high_frequency_end_step=end,
        )
        rows.append(
            _row(
                f"cfg_window_i{interval}_w100_s{steps}_g5",
                "cfg_cache",
                5.0,
                baseline,
                **values,
            )
        )
    return rows


def _cfg_hifi_case(steps: int) -> list[dict[str, object]]:
    baseline = f"exact_s{steps}_g5"
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
    return [
        _row(baseline, "off", 5.0, baseline),
        _row(
            f"cfg_hifi_i2_w100_s{steps}_g5",
            "cfg_cache",
            5.0,
            baseline,
            **values,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/"
            "leo2_acceleration_benchmark/profiles"
        ),
    )
    args = parser.parse_args()

    for steps in STEPS:
        _write(args.output_dir / f"context_ir_cache_matrix_s{steps}.csv", _matrix(steps, args.profile_root))
        _write(args.output_dir / f"context_ir_cfg_grid_s{steps}.csv", _cfg_grid(steps))
        _write(args.output_dir / f"context_ir_cfg_window_grid_s{steps}.csv", _cfg_window_grid(steps))
        _write(args.output_dir / f"context_ir_cfg_hifi_s{steps}.csv", _cfg_hifi_case(steps))
        _write(
            args.output_dir / f"context_ir_pilot_baselines_s{steps}.csv",
            [
                _row(f"exact_s{steps}_g1", "off", 1.0, f"exact_s{steps}_g1"),
                _row(f"exact_s{steps}_g5", "off", 5.0, f"exact_s{steps}_g5"),
            ],
        )
        calibration = [
            _row(
                f"magcache_calibrate_s{steps}_{suffix}",
                "magcache_calibrate",
                guidance,
                f"magcache_calibrate_s{steps}_{suffix}",
                magcache_threshold=0.12,
                magcache_max_skip_steps=4,
                magcache_retention_ratio=0.2,
            )
            for guidance, suffix in ((1.0, "g1"), (5.0, "g5"))
        ]
        _write(args.output_dir / f"context_ir_magcache_calibration_s{steps}.csv", calibration)

    smoke = [
        _row("exact_s6_g1", "off", 1.0, "exact_s6_g1"),
        _row("exact_s6_g5", "off", 5.0, "exact_s6_g5"),
        _row("cfg_cache_s6_g5", "cfg_cache", 5.0, "exact_s6_g5", **_cfg_values(6)),
        _row(
            "dfr_cfg_s6_g5",
            "fastercache_dfr+cfg_cache",
            5.0,
            "exact_s6_g5",
            **_dfr_values(6),
            **_cfg_values(6),
        ),
    ]
    _write(args.output_dir / "context_ir_cfg_cache_smoke_s6.csv", smoke)


if __name__ == "__main__":
    main()
