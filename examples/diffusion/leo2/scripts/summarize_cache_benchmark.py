"""Summarize paired Leo2 cache timings, latents and encoded videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import imageio_ffmpeg

REQUEST_MARKER = "LEO2_CACHE_BENCH_REQUEST_JSON="
CONFIG_MARKER = "LEO2_CACHE_BENCH_CONFIG_JSON="
EXPECTED_WIDTH = 848
EXPECTED_HEIGHT = 464
EXPECTED_FRAMES = 121
EXPECTED_FPS = 24.0


def _read_markers(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract the rank-zero config and one final record per request."""
    config: dict[str, Any] = {}
    requests: dict[int, dict[str, Any]] = {}
    if not log_path.is_file():
        return config, []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if CONFIG_MARKER in line:
            config = json.loads(line.split(CONFIG_MARKER, 1)[1])
        elif REQUEST_MARKER in line:
            record = json.loads(line.split(REQUEST_MARKER, 1)[1])
            requests[int(record["request"])] = record
    return config, [requests[key] for key in sorted(requests)]


def _percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile without external dependencies."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _mean(values: list[float]) -> float | None:
    """Return a mean or None for an empty sample."""
    return statistics.fmean(values) if values else None


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None]:
    """Return a normal-approximation 95% confidence interval for a mean."""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - margin, mean + margin


def _exit_code(case_dir: Path) -> int | None:
    """Read the torchrun exit code when the launcher recorded one."""
    path = case_dir / "exit_code.txt"
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _case_metadata(case_dir: Path) -> dict[str, str]:
    """Read simple key-value metadata emitted by the shell launcher."""
    path = case_dir / "case.env"
    if not path.is_file():
        return {}
    rows = (line.partition("=") for line in path.read_text(encoding="utf-8").splitlines())
    return {key: value for key, separator, value in rows if separator}


def _sha256(path: Path) -> str:
    """Hash a benchmark artifact without loading the complete file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path: Path) -> dict[str, Any]:
    """Validate one encoded video with the packaged imageio-ffmpeg binary."""
    try:
        reader = imageio_ffmpeg.read_frames(path)
        try:
            metadata = next(reader)
        finally:
            reader.close()
        frames, _ = imageio_ffmpeg.count_frames_and_secs(path)
        width, height = (int(value) for value in metadata["source_size"])
        fps = float(metadata["fps"])
        valid = (
            width == EXPECTED_WIDTH
            and height == EXPECTED_HEIGHT
            and frames == EXPECTED_FRAMES
            and math.isclose(fps, EXPECTED_FPS, rel_tol=0, abs_tol=1e-6)
        )
        return {
            "codec": metadata.get("codec"),
            "error": None,
            "fps": fps,
            "frames": frames,
            "height": height,
            "path": str(path),
            "valid": valid,
            "width": width,
        }
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path), "valid": False}


def _validate_latents(case_dir: Path, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check every request's latent file and recorded content digest."""
    validation = []
    for request in requests:
        filename = request.get("latent_file")
        path = case_dir / "latents" / filename if filename else None
        exists = path is not None and path.is_file()
        digest = _sha256(path) if exists else None
        validation.append(
            {
                "path": str(path) if path else None,
                "prompt_hash": request.get("prompt_hash"),
                "sha256": digest,
                "valid": exists and digest == request.get("latent_sha256"),
            }
        )
    return validation


def _case_summary(case_dir: Path, expected_videos: int) -> dict[str, Any]:
    """Build one case summary from its log, latents and videos."""
    config, records = _read_markers(case_dir / "run.log")
    successful = [record for record in records if record.get("status") == "ok"]
    elapsed = [float(record["elapsed_seconds"]) for record in successful]
    steady = elapsed[1:]
    full_steps = sum(int(record.get("full_steps", 0)) for record in successful)
    skipped_steps = sum(int(record.get("skipped_steps", 0)) for record in successful)
    cache_steps = full_steps + skipped_steps
    videos = sorted((case_dir / "samples").rglob("*.mp4"))
    video_validation = [_probe_video(path) for path in videos]
    valid_video_count = sum(bool(record["valid"]) for record in video_validation)
    latent_validation = _validate_latents(case_dir, successful)
    valid_latent_count = sum(bool(record["valid"]) for record in latent_validation)
    exit_code = _exit_code(case_dir)
    metadata = _case_metadata(case_dir)
    complete = (
        exit_code == 0
        and len(successful) == expected_videos
        and len(videos) == expected_videos
        and valid_video_count == expected_videos
        and valid_latent_count == expected_videos
    )
    mean_ci95_low, mean_ci95_high = _mean_ci95(elapsed)
    steady_ci95_low, steady_ci95_high = _mean_ci95(steady)
    return {
        "_case_dir": str(case_dir),
        "cache_enabled": config.get("cache_enabled"),
        "cache_threshold": config.get("cache_threshold"),
        "case": case_dir.name,
        "complete": complete,
        "config": config,
        "diff_infer_steps": config.get("diff_infer_steps"),
        "exit_code": exit_code,
        "expected_videos": expected_videos,
        "flow_shift_video": config.get("flow_shift_video"),
        "full_steps": full_steps,
        "generation_mean_seconds": _mean(elapsed),
        "generation_mean_seconds_ci95_high": mean_ci95_high,
        "generation_mean_seconds_ci95_low": mean_ci95_low,
        "generation_median_seconds": statistics.median(elapsed) if elapsed else None,
        "generation_p90_seconds": _percentile(elapsed, 0.90),
        "generation_steady_mean_seconds": _mean(steady),
        "generation_steady_mean_seconds_ci95_high": steady_ci95_high,
        "generation_steady_mean_seconds_ci95_low": steady_ci95_low,
        "generation_total_seconds": sum(elapsed),
        "latent_valid_count": valid_latent_count,
        "latent_validation": latent_validation,
        "max_memory_allocated_bytes": max(
            (int(record.get("max_memory_allocated_bytes", 0)) for record in successful), default=0
        ),
        "max_memory_reserved_bytes": max(
            (int(record.get("max_memory_reserved_bytes", 0)) for record in successful), default=0
        ),
        "process_wall_seconds": int(metadata["process_wall_seconds"]) if "process_wall_seconds" in metadata else None,
        "request_count": len(successful),
        "requests": successful,
        "skip_ratio": skipped_steps / cache_steps if cache_steps else 0.0,
        "skipped_steps": skipped_steps,
        "video_count": len(videos),
        "video_valid_count": valid_video_count,
        "video_validation": video_validation,
        "world_size": config.get("world_size"),
    }


def _load_latent(row: dict[str, Any], request: dict[str, Any]) -> Any:
    """Load and identity-check one saved final latent on CPU."""
    import torch

    path = Path(row["_case_dir"]) / "latents" / request["latent_file"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["prompt_hash"] != request["prompt_hash"]:
        raise ValueError(f"Latent prompt hash mismatch: {path}")
    return payload["latent"].double()


def _latent_drift(reference: Any, candidate: Any) -> dict[str, float]:
    """Compute paired final-latent drift metrics."""
    import torch

    if reference.shape != candidate.shape:
        raise ValueError(f"Latent shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    reference = reference.reshape(-1)
    candidate = candidate.reshape(-1)
    difference = candidate - reference
    epsilon = torch.finfo(reference.dtype).eps
    reference_l1 = reference.abs().sum().clamp_min(epsilon)
    reference_l2 = torch.linalg.vector_norm(reference).clamp_min(epsilon)
    candidate_l2 = torch.linalg.vector_norm(candidate).clamp_min(epsilon)
    cosine = (torch.dot(reference, candidate) / (reference_l2 * candidate_l2)).clamp(-1.0, 1.0)
    return {
        "cosine": float(cosine),
        "max_abs": float(difference.abs().max()),
        "rel_l1": float(difference.abs().sum() / reference_l1),
        "rel_l2": float(torch.linalg.vector_norm(difference) / reference_l2),
    }


def _empty_paired_metrics(row: dict[str, Any]) -> None:
    """Initialize unavailable paired metrics for a case without a baseline."""
    row.update(
        {
            "latent_cosine_mean": None,
            "latent_max_abs_max": None,
            "latent_pair_count": 0,
            "latent_rel_l1_mean": None,
            "latent_rel_l2_mean": None,
            "paired_speedup_mean": None,
            "paired_speedup_mean_ci95_high": None,
            "paired_speedup_mean_ci95_low": None,
            "paired_steady_speedup_mean": None,
            "zero_threshold_exact": None,
        }
    )


def _add_paired_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair timings and latents with the cache-off baseline at each flow shift."""
    baselines = {row["flow_shift_video"]: row for row in rows if row["cache_enabled"] is False and row["complete"]}
    paired_rows = []
    for row in rows:
        baseline = baselines.get(row["flow_shift_video"])
        if baseline is None:
            _empty_paired_metrics(row)
            continue
        baseline_requests = {request["prompt_hash"]: request for request in baseline["requests"]}
        ratios = []
        steady_ratios = []
        drift_rows = []
        for request in row["requests"]:
            reference = baseline_requests.get(request["prompt_hash"])
            if reference is None:
                continue
            ratio = float(reference["elapsed_seconds"]) / float(request["elapsed_seconds"])
            ratios.append(ratio)
            if int(request["request"]) > 1 and int(reference["request"]) > 1:
                steady_ratios.append(ratio)
            if row is baseline:
                drift = {"cosine": 1.0, "max_abs": 0.0, "rel_l1": 0.0, "rel_l2": 0.0}
            else:
                drift = _latent_drift(_load_latent(baseline, reference), _load_latent(row, request))
            paired = {
                "baseline_case": baseline["case"],
                "baseline_seconds": float(reference["elapsed_seconds"]),
                "cache_threshold": row["cache_threshold"],
                "candidate_seconds": float(request["elapsed_seconds"]),
                "case": row["case"],
                "flow_shift_video": row["flow_shift_video"],
                "prompt_hash": request["prompt_hash"],
                "prompt_index": request["prompt_index"],
                "seed": request["seed"],
                "speedup": ratio,
                **drift,
            }
            drift_rows.append(paired)
            paired_rows.append(paired)
        speedup_ci95_low, speedup_ci95_high = _mean_ci95(ratios)
        row.update(
            {
                "latent_cosine_mean": _mean([item["cosine"] for item in drift_rows]),
                "latent_max_abs_max": max((item["max_abs"] for item in drift_rows), default=None),
                "latent_pair_count": len(drift_rows),
                "latent_rel_l1_mean": _mean([item["rel_l1"] for item in drift_rows]),
                "latent_rel_l2_mean": _mean([item["rel_l2"] for item in drift_rows]),
                "paired_speedup_mean": _mean(ratios),
                "paired_speedup_mean_ci95_high": speedup_ci95_high,
                "paired_speedup_mean_ci95_low": speedup_ci95_low,
                "paired_steady_speedup_mean": _mean(steady_ratios),
                "zero_threshold_exact": None,
            }
        )
        row["complete"] = row["complete"] and len(drift_rows) == row["expected_videos"]
        if row["cache_enabled"] is True and row["cache_threshold"] == 0:
            exact = len(drift_rows) == len(row["requests"]) and all(item["max_abs"] == 0.0 for item in drift_rows)
            row["zero_threshold_exact"] = exact
            row["complete"] = row["complete"] and exact
    return paired_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable tabular projection of all case summaries."""
    fields = [
        "case",
        "complete",
        "cache_enabled",
        "cache_threshold",
        "flow_shift_video",
        "diff_infer_steps",
        "world_size",
        "request_count",
        "video_count",
        "video_valid_count",
        "latent_valid_count",
        "generation_total_seconds",
        "generation_mean_seconds",
        "generation_mean_seconds_ci95_low",
        "generation_mean_seconds_ci95_high",
        "generation_steady_mean_seconds",
        "generation_steady_mean_seconds_ci95_low",
        "generation_steady_mean_seconds_ci95_high",
        "generation_median_seconds",
        "generation_p90_seconds",
        "paired_speedup_mean",
        "paired_speedup_mean_ci95_low",
        "paired_speedup_mean_ci95_high",
        "paired_steady_speedup_mean",
        "full_steps",
        "skipped_steps",
        "skip_ratio",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
        "process_wall_seconds",
        "latent_pair_count",
        "latent_rel_l1_mean",
        "latent_rel_l2_mean",
        "latent_cosine_mean",
        "latent_max_abs_max",
        "zero_threshold_exact",
        "exit_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write prompt-paired timing and final-latent metrics."""
    fields = [
        "case",
        "baseline_case",
        "cache_threshold",
        "flow_shift_video",
        "prompt_index",
        "prompt_hash",
        "seed",
        "candidate_seconds",
        "baseline_seconds",
        "speedup",
        "rel_l1",
        "rel_l2",
        "cosine",
        "max_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Summarize every case directory immediately below one benchmark root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=16)
    args = parser.parse_args()
    case_dirs = sorted(path for path in args.root.iterdir() if path.is_dir() and (path / "run.log").exists())
    rows = [_case_summary(path, args.expected_videos) for path in case_dirs]
    paired_rows = _add_paired_metrics(rows)
    for row, case_dir in zip(rows, case_dirs):
        row.pop("_case_dir", None)
        (case_dir / "summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.root / "summary.csv", rows)
    _write_paired_csv(args.root / "paired_metrics.csv", paired_rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
