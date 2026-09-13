"""Summarize Leo2 end-to-end phase JSONL and node-level GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_phase_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "phase").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_gpu_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("gpu-node*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "node": int(row["node"]),
                        "gpu": int(row["gpu"]),
                        "memory_mib": float(row["memory_mib"]),
                        "util_pct": float(row["util_pct"]),
                        "power_w": float(row["power_w"]),
                    }
                )
    return rows


def summarize(run_dir: Path) -> dict[str, Any]:
    """Build phase, memory-ledger, and GPU telemetry summaries for one run."""
    phase_rows = _load_phase_rows(run_dir)
    elapsed: dict[str, list[float]] = defaultdict(list)
    for row in phase_rows:
        if "elapsed_s" in row:
            elapsed[row["name"]].append(float(row["elapsed_s"]))
    phase_summary = {
        name: {
            "records": len(values),
            "mean_s": statistics.fmean(values),
            "max_s": max(values),
            "min_s": min(values),
            "max_allocated_gib": max(
                (
                    float(memory.get("max_allocated_gib", 0.0))
                    for row in phase_rows
                    if row["name"] == name
                    for memory in (row.get("after", {}),)
                ),
                default=0.0,
            ),
            "max_reserved_gib": max(
                (
                    float(memory.get("max_reserved_gib", 0.0))
                    for row in phase_rows
                    if row["name"] == name
                    for memory in (row.get("after", {}),)
                ),
                default=0.0,
            ),
            "max_device_used_gib": max(
                (
                    float(memory.get("device_used_gib", 0.0))
                    for row in phase_rows
                    if row["name"] == name
                    for memory in (row.get("after", {}),)
                ),
                default=0.0,
            ),
        }
        for name, values in sorted(elapsed.items())
    }
    ledgers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in phase_rows:
        if row["name"] in {"leo2.rollout.conditions", "leo2.rollout.trajectory", "leo2.rollout.decoded", "train.memory_ledger"}:
            ledgers[row["name"]].append(row)
    ledger_summary = {
        name: {
            key: max(float(row.get(key, 0.0)) for row in rows)
            for key in sorted({key for row in rows for key in row if key.endswith("_bytes")})
        }
        for name, rows in sorted(ledgers.items())
    }
    gpu_rows = _load_gpu_rows(run_dir)
    gpu_summary = {}
    if gpu_rows:
        active = [row for row in gpu_rows if row["memory_mib"] > 8192]
        gpu_summary = {
            "samples": len(gpu_rows),
            "peak_memory_mib": max(row["memory_mib"] for row in gpu_rows),
            "mean_util_pct_all": statistics.fmean(row["util_pct"] for row in gpu_rows),
            "mean_util_pct_active": (
                statistics.fmean(row["util_pct"] for row in active) if active else 0.0
            ),
            "mean_power_w_active": (
                statistics.fmean(row["power_w"] for row in active) if active else 0.0
            ),
        }
    exit_path = run_dir / "exit_code"
    return {
        "run_dir": str(run_dir),
        "exit_code": int(exit_path.read_text()) if exit_path.is_file() else None,
        "phase": phase_summary,
        "ledger": ledger_summary,
        "gpu": gpu_summary,
    }


def main() -> None:
    """Write one JSON summary beside a completed profiling run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    output = args.output or args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
