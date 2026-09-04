#!/usr/bin/env python3
"""Validate and summarize one Leo2 vid_prompt topology run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


GENERATION_RE = re.compile(r"Generation completed in ([0-9]+(?:\.[0-9]+)?) seconds")
CHECKPOINT_RE = re.compile(r"Finished loading checkpoint in ([0-9]+(?:\.[0-9]+)?) seconds")


def _read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"expected benchmark metadata file, got {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"expected key=value at {path}:{line_number}, got {raw!r}")
        values[key] = value
    return values


def _csv_indices(path: Path, limit: int) -> list[int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < limit:
        raise ValueError(f"expected at least {limit} rows in {path}, got {len(rows)}")
    return [int(row["index"]) for row in rows[:limit]]


def _result_indices(case_dir: Path) -> list[int]:
    all_results = list(case_dir.glob("samples/**/results/all_results.csv"))
    if len(all_results) == 1:
        with all_results[0].open(encoding="utf-8", newline="") as handle:
            return [int(row["index"]) for row in csv.DictReader(handle)]
    if len(all_results) > 1:
        raise ValueError(
            f"expected at most one all_results.csv below {case_dir}, got {len(all_results)}"
        )
    result_files = list(case_dir.glob("samples/**/results/results_*.csv"))
    indices: list[int] = []
    for path in result_files:
        with path.open(encoding="utf-8", newline="") as handle:
            indices.extend(int(row["index"]) for row in csv.DictReader(handle))
    return indices


def _gpu_metrics(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        raise FileNotFoundError(f"expected GPU metrics CSV, got {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"expected at least one GPU sample in {path}, got 0")
    utils = [int(row["utilization_pct"]) for row in rows]
    memories = [int(row["memory_used_mib"]) for row in rows]
    return {
        "gpu_observations": len(rows),
        "gpu_util_mean_pct": sum(utils) / len(utils),
        "gpu_util_min_pct": min(utils),
        "gpu_util_gt90_fraction": sum(value > 90 for value in utils) / len(utils),
        "gpu_memory_peak_mib": max(memories),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    if args.expected < 1:
        raise ValueError(f"expected --expected >= 1, got {args.expected}")

    case_dir = args.case_dir.resolve()
    metadata = _read_env(case_dir / "case.env")
    run_log_path = case_dir / "run.log"
    if not run_log_path.is_file():
        raise FileNotFoundError(f"expected topology run log, got {run_log_path}")
    run_log = run_log_path.read_text(encoding="utf-8", errors="replace")

    expected_indices = _csv_indices(Path(metadata["csv"]), args.expected)
    result_indices = _result_indices(case_dir)
    video_paths = [path for path in case_dir.glob("samples/**/*.mp4") if path.is_file()]
    empty_videos = [str(path) for path in video_paths if path.stat().st_size == 0]
    generation_seconds = [float(value) for value in GENERATION_RE.findall(run_log)]
    checkpoint_seconds = [float(value) for value in CHECKPOINT_RE.findall(run_log)]
    wall_seconds = int(metadata["process_wall_seconds"])
    exit_code = int(metadata["exit_code"])
    effective_tp = 1 if "tensor_model_parallel_size is not supported" in run_log else int(
        metadata["tp_requested"]
    )

    unique_results = sorted(set(result_indices))
    complete = (
        exit_code == 0
        and unique_results == sorted(expected_indices)
        and len(result_indices) == args.expected
        and len(video_paths) == args.expected
        and not empty_videos
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "case_name": metadata["case_name"],
        "hostname": metadata["hostname"],
        "complete": complete,
        "exit_code": exit_code,
        "expected": args.expected,
        "result_rows": len(result_indices),
        "unique_result_indices": len(unique_results),
        "video_count": len(video_paths),
        "empty_videos": empty_videos,
        "process_wall_seconds": wall_seconds,
        "throughput_videos_per_hour_including_load": (
            len(video_paths) * 3600.0 / wall_seconds if wall_seconds > 0 else None
        ),
        "checkpoint_load_seconds_max": max(checkpoint_seconds) if checkpoint_seconds else None,
        "generation_rank_observations": len(generation_seconds),
        "generation_rank_seconds_mean": (
            sum(generation_seconds) / len(generation_seconds) if generation_seconds else None
        ),
        "topology": {
            "dp_shard": int(metadata["dp_shard"]),
            "cp": int(metadata["cp"]),
            "ep": int(metadata["ep"]),
            "tp_requested": int(metadata["tp_requested"]),
            "tp_effective": effective_tp,
        },
        **_gpu_metrics(case_dir / "gpu_metrics.csv"),
    }
    output = case_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not complete:
        raise RuntimeError(
            "topology case incomplete: "
            f"case={metadata['case_name']}, expected_indices={sorted(expected_indices)}, "
            f"result_indices={unique_results}, videos={len(video_paths)}, exit_code={exit_code}"
        )


if __name__ == "__main__":
    main()
