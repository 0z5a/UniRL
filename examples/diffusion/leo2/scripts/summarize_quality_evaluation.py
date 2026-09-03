#!/usr/bin/env python3
"""Aggregate Leo2 cache fidelity and video-quality metrics."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from cache_benchmark_cases import CacheBenchmarkCase, load_cases

VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
VIDEOSCORE_DIMENSIONS = ("visual_quality", "text_alignment", "physical_consistency")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def latent_tensor(path: Path) -> torch.Tensor:
    """Load one final latent on CPU as float64."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensor = payload["latent"] if isinstance(payload, dict) else payload
    return tensor.to(dtype=torch.float64)


def latent_mse(root: Path, case: str, baseline_root: Path, baseline: str, expected: int) -> tuple[float, float]:
    """Return mean per-video MSE and RMSE against the same-shift baseline."""
    if case == baseline:
        return 0.0, 0.0
    candidate_paths = sorted((root / case / "latents").glob("*.pt"))
    baseline_by_name = {path.name: path for path in (baseline_root / baseline / "latents").glob("*.pt")}
    if len(candidate_paths) != expected:
        raise ValueError(f"expected {expected} candidate latents for {case}, got {len(candidate_paths)}")
    values = []
    for candidate_path in candidate_paths:
        baseline_path = baseline_by_name.get(candidate_path.name)
        if baseline_path is None:
            raise ValueError(f"baseline lacks latent {candidate_path.name}: {baseline_root / baseline}")
        difference = latent_tensor(candidate_path) - latent_tensor(baseline_path)
        values.append(difference.square().mean().item())
    mean_mse = sum(values) / len(values)
    return mean_mse, math.sqrt(mean_mse)


def find_vbench_result(root: Path, case: str) -> dict:
    """Load the unique VBench result JSON for one case."""
    matches = sorted((root / "quality_eval" / "vbench" / case).glob("*_eval_results.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one VBench result for {case}, found {matches}")
    return json.loads(matches[0].read_text())


def videoscore_means(root: Path, case: str, expected: int) -> dict[str, float]:
    """Average the VideoScore2 ordinal scores for one case."""
    path = root / "quality_eval" / "videoscore2" / f"{case}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} VideoScore2 rows for {case}, got {len(rows)}")
    return {key: sum(row[key] for row in rows) / len(rows) for key in VIDEOSCORE_DIMENSIONS}


def _baseline(case: CacheBenchmarkCase, root: Path) -> tuple[str, Path]:
    """Resolve an explicit or legacy cache-off baseline."""
    name = case.baseline_case or f"cache_off_shift{int(case.flow_shift_video)}"
    return name, Path(case.reference_root).resolve() if case.reference_root else root


def aggregate(args: argparse.Namespace) -> list[dict[str, object]]:
    """Combine timing, paired-error, VBench, and VideoScore2 results."""
    cases = load_cases(args.cases_csv)
    summaries = {row["case"]: row for row in read_csv(args.benchmark_root / "summary.csv")}
    paired_rows = read_csv(args.benchmark_root / "paired_metrics.csv")
    paired = defaultdict(list)
    for row in paired_rows:
        paired[row["case"]].append(row)
    pixel_rows = read_csv(args.benchmark_root / "pixel_metrics_pairs.csv")
    pixels = defaultdict(list)
    for row in pixel_rows:
        pixels[row["case"]].append(row)

    results = []
    for case in cases:
        name = case.name
        baseline, baseline_root = _baseline(case, args.benchmark_root)
        summary = summaries[name]
        cache_enabled = case.cache_enabled
        sample_count = int(summary["request_count"])
        row: dict[str, object] = {
            "case": name,
            "baseline_case": baseline,
            "baseline_root": str(baseline_root),
            "cache_method": case.method,
            "flow_shift_video": case.flow_shift_video,
            "guidance_scale": case.guidance_scale,
            "cache_threshold": case.cache_threshold,
            "cache_disabled": not cache_enabled,
            "quality_source_root": str(args.benchmark_root),
            "sample_count": sample_count,
            "generation_mean_seconds": float(summary["generation_mean_seconds"]),
            "paired_speedup_mean": float(summary["paired_speedup_mean"]),
            "skip_ratio": float(summary["skip_ratio"]),
            "latent_relative_l1": sum(float(item["rel_l1"]) for item in paired[name]) / sample_count,
            "latent_relative_l2": sum(float(item["rel_l2"]) for item in paired[name]) / sample_count,
            "latent_cosine": sum(float(item["cosine"]) for item in paired[name]) / sample_count,
        }
        row["latent_mse"], row["latent_rmse"] = latent_mse(
            args.benchmark_root, name, baseline_root, baseline, sample_count
        )
        if cache_enabled:
            case_pixels = pixels[name]
            if len(case_pixels) != sample_count:
                raise ValueError(f"expected {sample_count} pixel pairs for {name}, got {len(case_pixels)}")
            row.update(
                pixel_mse=sum(float(item["rmse"]) ** 2 for item in case_pixels) / sample_count,
                pixel_mae=sum(float(item["mae"]) for item in case_pixels) / sample_count,
                pixel_rmse=math.sqrt(sum(float(item["rmse"]) ** 2 for item in case_pixels) / sample_count),
                pixel_relative_l1=sum(float(item["relative_l1"]) for item in case_pixels) / sample_count,
                pixel_relative_l2=sum(float(item["relative_l2"]) for item in case_pixels) / sample_count,
            )
        else:
            row.update(
                pixel_mse=0.0,
                pixel_mae=0.0,
                pixel_rmse=0.0,
                pixel_relative_l1=0.0,
                pixel_relative_l2=0.0,
            )
        vbench = find_vbench_result(args.benchmark_root, name)
        row.update({f"vbench_{key}": float(vbench[key][0]) for key in VBENCH_DIMENSIONS})
        row.update(
            {
                f"videoscore2_{key}": value
                for key, value in videoscore_means(args.benchmark_root, name, sample_count).items()
            }
        )
        results.append(row)
    present = {str(row["case"]) for row in results}
    for case in cases:
        baseline, baseline_root = _baseline(case, args.benchmark_root)
        if baseline in present or baseline_root == args.benchmark_root:
            continue
        path = baseline_root / "quality_eval" / "summary" / "quality_metrics_cases.json"
        rows = json.loads(path.read_text())
        matches = [row for row in rows if row.get("case") == baseline]
        if len(matches) != 1:
            raise ValueError(f"expected one quality row for {baseline} in {path}")
        baseline_row = dict(matches[0])
        if not math.isclose(float(baseline_row["flow_shift_video"]), case.flow_shift_video):
            raise ValueError(f"external quality baseline shift mismatch: {path}")
        baseline_row.setdefault("guidance_scale", case.guidance_scale)
        baseline_row.setdefault("cache_method", "first_block" if baseline_row.get("cache_disabled") is False else "off")
        baseline_row["quality_source_root"] = str(baseline_root)
        results.append(baseline_row)
        present.add(baseline)
    return results


def write_results(results: list[dict[str, object]], output_dir: Path) -> None:
    """Write machine-readable aggregate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_metrics_cases.csv"
    json_path = output_dir / "quality_metrics_cases.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite results in {output_dir}")
    with csv_path.open("x", newline="") as handle:
        fields = list(dict.fromkeys(key for row in results for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    """Aggregate and save all quality metrics."""
    args = parse_args()
    write_results(aggregate(args), args.output_dir)


if __name__ == "__main__":
    main()
