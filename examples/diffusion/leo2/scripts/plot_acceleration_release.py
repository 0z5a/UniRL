#!/usr/bin/env python3
"""Render reproducible SVG/PNG figures from an acceleration release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    return float(value) if value is not None else None


def _save(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    draw: Callable[[plt.Axes, list[dict[str, Any]]], None],
) -> None:
    paths = [output_dir / f"{name}.{suffix}" for suffix in ("json", "svg", "png")]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figure artifacts: {existing}")
    paths[0].write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    draw(axis, rows)
    axis.grid(alpha=0.2)
    figure.savefig(paths[1])
    figure.savefig(paths[2], dpi=180)
    plt.close(figure)


def pareto(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    for guidance, marker in ((1.0, "o"), (5.0, "s")):
        group = [
            row
            for row in rows
            if float(row["guidance"]) == guidance
            and _number(row, "paired_speedup") is not None
            and _number(row, "latent_relative_l2") is not None
        ]
        axis.scatter(
            [row["latent_relative_l2"] for row in group],
            [row["paired_speedup"] for row in group],
            marker=marker,
            label=f"guidance={guidance:g}",
        )
    axis.set_title("Leo2 speed–quality Pareto candidates")
    axis.set_xlabel("Final latent relative L2 (lower is better)")
    axis.set_ylabel("Prompt-paired speedup (×, higher is better)")
    axis.legend()


def step_scaling(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        group = sorted(
            [row for row in rows if row["method"] == method and float(row["guidance"]) == 5.0],
            key=lambda row: int(row["steps"]),
        )
        axis.plot(
            [row["steps"] for row in group],
            [row["latency_seconds"] for row in group],
            marker="o",
            label=method,
        )
    axis.set_title("Leo2 latency scaling across denoising schedules")
    axis.set_xlabel("Denoising steps")
    axis.set_ylabel("End-to-end latency (seconds)")
    axis.legend(fontsize=8)


def guidance_compare(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    exact = [row for row in rows if row["method"] == "off"]
    keys = sorted({int(row["steps"]) for row in exact})
    width = 0.36
    for offset, guidance in ((-width / 2, 1.0), (width / 2, 5.0)):
        values = [
            next(
                float(row["latency_seconds"])
                for row in exact
                if int(row["steps"]) == steps and float(row["guidance"]) == guidance
            )
            for steps in keys
        ]
        axis.bar([index + offset for index in range(len(keys))], values, width, label=f"guidance={guidance:g}")
    axis.set_xticks(range(len(keys)), [str(value) for value in keys])
    axis.set_title("Exact inference latency: guidance 1 versus 5")
    axis.set_xlabel("Denoising steps")
    axis.set_ylabel("End-to-end latency (seconds)")
    axis.legend()


def language_compare(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    methods = sorted({row["method"] for row in rows})
    for language, marker in (("en", "o"), ("zh", "s")):
        group = {
            row["method"]: row
            for row in rows
            if row["language"] == language and _number(row, "paired_speedup") is not None
        }
        axis.plot(
            range(len(methods)),
            [group.get(method, {}).get("paired_speedup") for method in methods],
            marker=marker,
            label=language.upper(),
        )
    axis.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    axis.set_title("English versus Chinese paired speedup")
    axis.set_xlabel("Acceleration method")
    axis.set_ylabel("Prompt-paired speedup (×)")
    axis.legend()


def reuse(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['method']} / {row['steps']}s / g{row['guidance']:g}" for row in rows]
    bottom = [0.0] * len(rows)
    for field, label in (
        ("tail_reuse_ratio", "whole-tail"),
        ("attention_reuse_ratio", "attention"),
        ("cfg_reuse_ratio", "CFG output"),
    ):
        values = [100 * float(row.get(field, 0) or 0) for row in rows]
        axis.bar(range(len(rows)), values, bottom=bottom, label=label)
        bottom = [left + right for left, right in zip(bottom, values)]
    axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right", fontsize=7)
    axis.set_title("Acceleration reuse ratios by cache family")
    axis.set_xlabel("Method / steps / guidance")
    axis.set_ylabel("Reuse ratio (%)")
    axis.legend()


def memory(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['method']} / {row['steps']} / g{row['guidance']:g}" for row in rows]
    axis.bar(range(len(rows)), [row["peak_allocated_gib"] for row in rows])
    axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right", fontsize=7)
    axis.set_title("Peak CUDA allocation by Leo2 inference case")
    axis.set_xlabel("Method / steps / guidance")
    axis.set_ylabel("Peak allocated memory (GiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.release.read_text(encoding="utf-8"))
    rows = [row for row in payload["metrics"] if row.get("language") == "overall"]
    language_rows = [row for row in payload["metrics"] if row.get("language") in {"en", "zh"}]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save(args.output_dir, "speedup_quality_pareto", rows, pareto)
    _save(args.output_dir, "steps_latency_scaling", rows, step_scaling)
    _save(args.output_dir, "guidance_latency_comparison", rows, guidance_compare)
    _save(args.output_dir, "language_speedup_comparison", language_rows, language_compare)
    _save(args.output_dir, "cache_reuse_ratios", rows, reuse)
    _save(args.output_dir, "peak_cuda_memory", rows, memory)


if __name__ == "__main__":
    main()
