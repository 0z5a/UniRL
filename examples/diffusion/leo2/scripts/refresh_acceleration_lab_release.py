#!/usr/bin/env python3
"""Incrementally publish completed pilot roots to the Acceleration Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from argparse import Namespace
from pathlib import Path

from publish_acceleration_release import _atomic_write, _load_json, _sha256, build


def _complete_roots(runs_root: Path) -> list[Path]:
    candidates = sorted(path for path in runs_root.glob("pilot_*") if path.is_dir())
    roots = []
    baseline_keys = set()
    pending_candidates: list[tuple[Path, list[dict]]] = []
    for root in candidates:
        summary_path = root / "summary.json"
        paired_path = root / "paired_metrics.csv"
        if not summary_path.is_file() or not paired_path.is_file():
            continue
        rows = _load_json(summary_path)
        if not isinstance(rows, list) or not rows or not all(
            isinstance(row, dict) and row.get("complete") is True for row in rows
        ):
            continue
        if all(row.get("cache_method") == "off" for row in rows):
            roots.append(root)
            baseline_keys.update(
                (
                    str(row["case"]),
                    int(row["diff_infer_steps"]),
                    float(row["guidance_scale"]),
                )
                for row in rows
            )
        else:
            pending_candidates.append((root, rows))
    for root, rows in pending_candidates:
        if all(
            (
                str(row["baseline_case"]),
                int(row["diff_infer_steps"]),
                float(row["guidance_scale"]),
            )
            in baseline_keys
            for row in rows
        ):
            roots.append(root)
    case_names: dict[str, Path] = {}
    for root in sorted(roots):
        for row in _load_json(root / "summary.json"):
            case = str(row["case"])
            case_names[case] = root
    selected = set(case_names.values())
    return sorted(selected)


def _fingerprint(roots: list[Path], prompts: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_sha256(prompts).encode())
    for script in (
        Path(__file__).resolve(),
        Path(__file__).with_name("publish_acceleration_release.py").resolve(),
    ):
        digest.update(_sha256(script).encode())
    for root in roots:
        digest.update(str(root).encode())
        for relative in (
            "summary.json",
            "paired_metrics.csv",
            "pixel_metrics_pairs.csv",
            "quality_eval/summary/quality_metrics_cases.json",
        ):
            path = root / relative
            if path.is_file():
                digest.update(relative.encode())
                digest.update(_sha256(path).encode())
    return digest.hexdigest()


def refresh(args: argparse.Namespace) -> Path:
    roots = _complete_roots(args.runs_root.resolve())
    if not roots:
        raise ValueError(f"no completed publishable roots under {args.runs_root}")
    fingerprint = _fingerprint(roots, args.prompts_csv.resolve())
    release_id = f"pilot-auto-{fingerprint[:12]}"
    output = args.releases_dir.resolve() / f"{release_id}.json"
    current = args.releases_dir.resolve() / "current.json"
    if output.is_file():
        payload = _load_json(output)
    else:
        payload = build(
            Namespace(
                release_id=release_id,
                artifact_root=args.artifact_root,
                prompts_csv=args.prompts_csv,
                benchmark_root=roots,
                figure=[],
            )
        )
        _atomic_write(output, payload, replace=False)
    current_release = (
        _load_json(current).get("release_id")
        if current.is_file()
        else None
    )
    if current_release != release_id:
        _atomic_write(current, payload, replace=True)
    print(
        json.dumps(
            {
                "release_id": release_id,
                "roots": len(roots),
                "metrics": len(payload["metrics"]),
                "prompts": len(payload["prompts"]),
                "current_updated": current_release != release_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--releases-dir", type=Path, required=True)
    parser.add_argument("--watch-interval", type=float, default=0)
    args = parser.parse_args()
    if args.watch_interval < 0:
        raise ValueError(f"watch interval must be non-negative, got {args.watch_interval}")
    while True:
        refresh(args)
        if args.watch_interval == 0:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
