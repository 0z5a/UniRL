#!/usr/bin/env python3
"""Build one immutable manifest for the Leo2 Acceleration Lab."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, artifact_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"published artifact {path} is outside artifact root {artifact_root}") from exc


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _load_prompts(path: Path) -> dict[int, dict[str, Any]]:
    prompts: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            try:
                index = int(row["index"])
                seed = int(row["seed"])
                prompt = row["prompt"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prompt row at {path}:{line_number}: {exc}") from exc
            if index in prompts or not prompt:
                raise ValueError(f"duplicate index or empty prompt at {path}:{line_number}")
            prompts[index] = {
                "index": index,
                "seed": seed,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "language": (row.get("language") or "").strip(),
                "pair_id": (row.get("pair_id") or "").strip(),
                "source_dataset": (row.get("source_dataset") or "").strip(),
                "source_id": (row.get("source_id") or "").strip(),
                "videos": [],
            }
    if not prompts:
        raise ValueError(f"prompt CSV is empty: {path}")
    return prompts


def _pixel_groups(root: Path) -> dict[tuple[str, str], dict[str, float]]:
    path = root / "pixel_metrics_pairs.csv"
    if not path.is_file():
        return {}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["case"], "overall")].append(row)
            language = row.get("language") or "overall"
            if language != "overall":
                groups[(row["case"], language)].append(row)
    result = {}
    for key, rows in groups.items():
        result[key] = {
            f"pixel_{metric}": _mean([float(row[metric]) for row in rows])
            for metric in ("mae", "rmse", "relative_l1", "relative_l2")
        }
    return result


def _quality_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "quality_eval/summary/quality_metrics_cases.json"
    if not path.is_file():
        return {}
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise TypeError(f"expected quality JSON list in {path}, got {type(rows).__name__}")
    return {str(row["case"]): row for row in rows}


def _paired_rows(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = root / "paired_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"paired metrics are missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["case"], int(row["prompt_index"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"paired metrics contain duplicate case/prompt rows: {path}")
    return result


def _video_by_index(summary: dict[str, Any]) -> dict[int, Path]:
    result = {}
    for item in summary.get("video_validation", []):
        path = Path(item["path"])
        try:
            index = int(path.stem.rsplit("_", 1)[0])
        except ValueError as exc:
            raise ValueError(f"video filename does not encode prompt index: {path}") from exc
        result[index] = path
    return result


def _metric_row(
    summary: dict[str, Any],
    *,
    language: str,
    requests: list[dict[str, Any]],
    pixel: dict[str, float],
    quality: dict[str, Any],
) -> dict[str, Any]:
    language_metrics = summary.get("language_metrics", {}).get(language, {}) if language != "overall" else {}
    speedup = (
        language_metrics.get("paired_speedup_mean")
        if language != "overall"
        else summary.get("paired_speedup_mean")
    )
    latent_l2 = (
        language_metrics.get("latent_rel_l2_mean")
        if language != "overall"
        else summary.get("latent_rel_l2_mean")
    )
    latent_l1 = (
        language_metrics.get("latent_rel_l1_mean")
        if language != "overall"
        else summary.get("latent_rel_l1_mean")
    )
    row = {
        "case": summary["case"],
        "method": summary["cache_method"],
        "steps": int(summary["diff_infer_steps"]),
        "guidance": float(summary["guidance_scale"]),
        "language": language,
        "sample_count": len(requests),
        "latency_seconds": _mean([float(request["elapsed_seconds"]) for request in requests]),
        "paired_speedup": speedup,
        "latent_relative_l1": latent_l1,
        "latent_relative_l2": latent_l2,
        "latent_cosine": (
            language_metrics.get("latent_cosine_mean")
            if language != "overall"
            else summary.get("latent_cosine_mean")
        ),
        "tail_reuse_ratio": summary.get("tail_reuse_ratio", 0.0),
        "attention_reuse_ratio": summary.get("attention_reuse_ratio", 0.0),
        "cfg_reuse_ratio": summary.get("cfg_reuse_ratio", 0.0),
        "cache_residency_mib": (
            float(summary["cache_bytes"]) / 2**20
            if summary.get("cache_bytes") is not None
            else None
        ),
        "peak_allocated_gib": float(summary["max_memory_allocated_bytes"]) / 2**30,
        "peak_reserved_gib": float(summary["max_memory_reserved_bytes"]) / 2**30,
        "parameters": summary.get("cache_method_options", {}),
        **pixel,
    }
    for name in (
        "vbench_total",
        "videoscore2_total",
        "latent_mse",
        "latent_rmse",
    ):
        if name in quality:
            row[name] = quality[name]
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    prompts = _load_prompts(args.prompts_csv.resolve())
    metrics: list[dict[str, Any]] = []
    files: dict[str, Path] = {}
    roots = [root.resolve() for root in args.benchmark_root]
    for root in roots:
        summaries = _load_json(root / "summary.json")
        if not isinstance(summaries, list):
            raise TypeError(f"expected summary list in {root / 'summary.json'}")
        pixel = _pixel_groups(root)
        quality = _quality_rows(root)
        paired = _paired_rows(root)
        for summary in summaries:
            if summary.get("complete") is not True:
                raise ValueError(f"refusing to publish incomplete case {summary.get('case')!r} from {root}")
            if (
                int(summary.get("num_frames", -1)) != 193
                or not math.isclose(float(summary.get("video_fps", -1)), 24.0)
                or not math.isclose(float(summary.get("flow_shift_video", -1)), 9.0)
            ):
                raise ValueError(f"case {summary['case']} violates the fixed 193-frame/24-fps/shift-9 contract")
            requests = summary["requests"]
            metrics.append(
                _metric_row(
                    summary,
                    language="overall",
                    requests=requests,
                    pixel=pixel.get((summary["case"], "overall"), {}),
                    quality=quality.get(summary["case"], {}),
                )
            )
            for language in ("en", "zh"):
                subset = [request for request in requests if request.get("language") == language]
                if subset:
                    metrics.append(
                        _metric_row(
                            summary,
                            language=language,
                            requests=subset,
                            pixel=pixel.get((summary["case"], language), {}),
                            quality={},
                        )
                    )
            video_paths = _video_by_index(summary)
            for request in requests:
                index = int(request["prompt_index"])
                prompt = prompts.get(index)
                if prompt is None:
                    raise ValueError(f"case {summary['case']} references unknown prompt index {index}")
                for field in ("prompt_hash", "language", "pair_id", "seed"):
                    if str(request.get(field, "")) != str(prompt.get(field, "")):
                        raise ValueError(
                            f"case {summary['case']} prompt {index} disagrees on {field}: "
                            f"{request.get(field)!r} != {prompt.get(field)!r}"
                        )
                video_path = video_paths[index]
                relative = _relative(video_path, artifact_root)
                files[relative] = video_path
                pair = paired.get((summary["case"], index))
                if pair is None:
                    raise ValueError(
                        f"paired metrics lack case={summary['case']!r}, prompt_index={index}"
                    )
                prompt["videos"].append(
                    {
                        "case": summary["case"],
                        "method": summary["cache_method"],
                        "exact": summary["cache_method"] == "off",
                        "steps": int(summary["diff_infer_steps"]),
                        "guidance": float(summary["guidance_scale"]),
                        "path": relative,
                        "latency_seconds": float(request["elapsed_seconds"]),
                        "speedup": float(pair["speedup"]),
                        "latent_relative_l2": float(pair["rel_l2"]),
                    }
                )
    figures = []
    for raw in args.figure:
        title, separator, path_text = raw.partition("=")
        if not separator or not title.strip():
            raise ValueError(f"expected --figure TITLE=PATH, got {raw!r}")
        path = Path(path_text).resolve()
        relative = _relative(path, artifact_root)
        files[relative] = path
        figures.append({"title": title.strip(), "path": relative})
    file_rows = [
        {
            "path": relative,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for relative, path in sorted(files.items())
    ]
    published_prompts = [prompts[index] for index in sorted(prompts) if prompts[index]["videos"]]
    for prompt in published_prompts:
        prompt["worst_score"] = max(
            (
                float(video["latent_relative_l2"])
                for video in prompt["videos"]
                if not video["exact"]
            ),
            default=0.0,
        )
    return {
        "schema_version": 1,
        "release_id": args.release_id,
        "artifact_root": str(artifact_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "width": 848,
            "height": 464,
            "duration_seconds": 8,
            "num_frames": 193,
            "fps": 24,
            "shift": 9,
            "steps": sorted({int(row["steps"]) for row in metrics}),
            "guidance": sorted({float(row["guidance"]) for row in metrics}),
            "batch": 1,
            "cp": 8,
            "fsdp": 8,
            "ep": 1,
        },
        "filters": {
            "steps": sorted({int(row["steps"]) for row in metrics}),
            "guidance": sorted({float(row["guidance"]) for row in metrics}),
            "methods": sorted({str(row["method"]) for row in metrics}),
        },
        "benchmark_roots": [str(root) for root in roots],
        "metrics": metrics,
        "prompts": published_prompts,
        "figures": figures,
        "files": file_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, action="append", required=True)
    parser.add_argument("--figure", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite release manifest: {args.output}")
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
