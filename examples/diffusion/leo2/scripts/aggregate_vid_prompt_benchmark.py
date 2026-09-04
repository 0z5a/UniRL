#!/usr/bin/env python3
"""Aggregate Leo2 topology pilots and the 256-prompt production run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"expected benchmark summary JSON, got {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(value).__name__}")
    return value


def _failure_reason(pilot_dir: Path, case_name: str, complete: bool) -> str | None:
    if complete:
        return None
    run_log = pilot_dir / f"{case_name}.run.log"
    if not run_log.is_file():
        return "failed; run log was not collected"
    content = run_log.read_text(encoding="utf-8", errors="replace")
    if "OutOfMemoryError" in content:
        return "CUDA OOM"
    if "ChildFailedError" in content:
        return "distributed child failure"
    return "failed; inspect collected run log"


def _pilot_rows(pilot_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(pilot_dir.glob("*.summary.json")):
        item = _load_json(path)
        topology = item["topology"]
        case_name = str(item["case_name"])
        rows.append(
            {
                "case": case_name,
                "complete": bool(item["complete"]),
                "expected": int(item["expected"]),
                "dp_shard": int(topology["dp_shard"]),
                "cp": int(topology["cp"]),
                "ep": int(topology["ep"]),
                "tp_requested": int(topology["tp_requested"]),
                "tp_effective": int(topology["tp_effective"]),
                "wall_seconds": int(item["process_wall_seconds"]),
                "throughput_videos_per_hour_including_load": float(
                    item["throughput_videos_per_hour_including_load"]
                ),
                "generation_rank_seconds_mean": item["generation_rank_seconds_mean"],
                "checkpoint_load_seconds_max": item["checkpoint_load_seconds_max"],
                "peak_memory_gib": float(item["gpu_memory_peak_mib"]) / 1024.0,
                "gpu_util_mean_pct": float(item["gpu_util_mean_pct"]),
                "failure_reason": _failure_reason(
                    pilot_dir, case_name, bool(item["complete"])
                ),
            }
        )
    if not rows:
        raise ValueError(f"expected pilot summaries in {pilot_dir}, got 0")
    return rows


def _full_summary(full_dir: Path) -> dict[str, Any]:
    paths = sorted((full_dir / "node_summaries").glob("node*.json"))
    if len(paths) != 8:
        raise ValueError(f"expected 8 full-run node summaries in {full_dir}, got {len(paths)}")
    nodes = [_load_json(path) for path in paths]
    incomplete = [node["case_name"] for node in nodes if not node["complete"]]
    if incomplete:
        raise RuntimeError(f"expected all full-run nodes complete, got failures={incomplete}")
    videos = [path for path in (full_dir / "videos").glob("*_0.mp4") if path.is_file()]
    indices = sorted(int(path.stem.split("_", 1)[0]) for path in videos if path.stat().st_size > 0)
    if indices != list(range(256)):
        raise RuntimeError(
            f"expected consolidated non-empty video indices 0..255, got count={len(indices)}"
        )

    wall_seconds = max(int(node["process_wall_seconds"]) for node in nodes)
    generation_round_seconds = [
        float(node["generation_rank_seconds_mean"]) for node in nodes
    ]
    return {
        "complete": True,
        "node_count": len(nodes),
        "gpu_count": len(nodes) * 8,
        "video_count": len(videos),
        "resolution": "848x464",
        "num_frames": 121,
        "diffusion_steps": 50,
        "topology": {
            "per_node": "dp_shard=8, cp=2, ep=1, tp=1",
            "data_parallel_groups_per_node": 4,
            "inter_node_strategy": "8 independent replicas over round-robin prompt shards",
            "cluster_concurrent_videos": 32,
        },
        "cluster_wall_seconds_including_load": wall_seconds,
        "cluster_throughput_videos_per_hour_including_load": 256 * 3600.0 / wall_seconds,
        "generation_round_seconds_mean": statistics.mean(generation_round_seconds),
        "generation_round_seconds_std": statistics.pstdev(generation_round_seconds),
        "estimated_steady_cluster_throughput_videos_per_hour": (
            len(nodes) * 4 * 3600.0 / statistics.mean(generation_round_seconds)
        ),
        "node_throughput_videos_per_hour_mean_including_load": statistics.mean(
            float(node["throughput_videos_per_hour_including_load"]) for node in nodes
        ),
        "checkpoint_load_seconds_mean": statistics.mean(
            float(node["checkpoint_load_seconds_max"]) for node in nodes
        ),
        "peak_memory_gib_max": max(
            float(node["gpu_memory_peak_mib"]) / 1024.0 for node in nodes
        ),
        "gpu_util_mean_pct_across_nodes": statistics.mean(
            float(node["gpu_util_mean_pct"]) for node in nodes
        ),
        "gpu_util_gt90_fraction_across_nodes": statistics.mean(
            float(node["gpu_util_gt90_fraction"]) for node in nodes
        ),
        "consolidated_video_bytes": sum(path.stat().st_size for path in videos),
        "nodes": nodes,
    }


def _write_pilot_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, pilots: list[dict[str, Any]], full: dict[str, Any]) -> None:
    successful = [row for row in pilots if row["complete"] and row["expected"] >= 4]
    best = max(successful, key=lambda row: row["throughput_videos_per_hour_including_load"])
    lines = [
        "# Leo2 vid_prompt topology benchmark",
        "",
        "## Pilot topology results",
        "",
        "| Case | Status | DP shard | CP | EP | TP requested/effective | Videos | Wall (s) | Videos/h | Peak GiB | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(pilots, key=lambda item: item["case"]):
        result = "PASS" if row["complete"] else str(row["failure_reason"])
        lines.append(
            f"| {row['case']} | {'PASS' if row['complete'] else 'FAIL'} | "
            f"{row['dp_shard']} | {row['cp']} | {row['ep']} | "
            f"{row['tp_requested']}/{row['tp_effective']} | {row['expected']} | "
            f"{row['wall_seconds']} | {row['throughput_videos_per_hour_including_load']:.2f} | "
            f"{row['peak_memory_gib']:.2f} | {result} |"
        )
    lines.extend(
        [
            "",
            "## Selected topology",
            "",
            f"Best successful four-video pilot: `{best['case']}` at "
            f"{best['throughput_videos_per_hour_including_load']:.2f} videos/hour including model load.",
            "",
            "TP=8 is not supported by the native FSDP sampler and was reset to TP=1. "
            "EP=2 and EP=4 failed with CUDA OOM even when the loading keepalive was disabled.",
            "",
            "## Full 256-prompt run",
            "",
            f"- Complete: {full['complete']}",
            f"- Output: {full['video_count']} non-empty videos, 848x464, 121 frames, 50 steps",
            f"- Cluster: {full['node_count']} nodes / {full['gpu_count']} GPUs",
            f"- Topology: {full['topology']['per_node']}; "
            f"{full['topology']['inter_node_strategy']}",
            f"- Wall time including model load: {full['cluster_wall_seconds_including_load']} s",
            f"- Actual cluster throughput including load: "
            f"{full['cluster_throughput_videos_per_hour_including_load']:.2f} videos/hour",
            f"- Estimated steady cluster throughput: "
            f"{full['estimated_steady_cluster_throughput_videos_per_hour']:.2f} videos/hour",
            f"- Mean checkpoint load: {full['checkpoint_load_seconds_mean']:.2f} s",
            f"- Max peak GPU memory: {full['peak_memory_gib_max']:.2f} GiB",
            f"- Mean sampled GPU utilization: {full['gpu_util_mean_pct_across_nodes']:.2f}%",
            f"- GPU observations above 90%: "
            f"{full['gpu_util_gt90_fraction_across_nodes'] * 100:.2f}%",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilots = _pilot_rows(args.pilot_dir)
    full = _full_summary(args.full_dir)
    report = {"schema_version": 1, "pilots": pilots, "full_run": full}
    (args.output_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_pilot_csv(args.output_dir / "pilot_topologies.csv", pilots)
    _write_markdown(args.output_dir / "BENCHMARK_REPORT.md", pilots, full)
    print(json.dumps(full, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
