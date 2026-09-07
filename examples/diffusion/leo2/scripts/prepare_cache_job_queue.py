#!/usr/bin/env python3
"""Create a longest-job-first queue for sharded Leo2 exact inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CONFIGS = (
    (28, 5, 412.0),
    (28, 1, 222.0),
    (6, 5, 109.0),
    (6, 1, 68.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/root/UniRL"))
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    if args.queue.exists():
        raise FileExistsError(f"refusing to overwrite queue: {args.queue}")
    manifest = json.loads(args.shard_manifest.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"shard manifest has no shards: {args.shard_manifest}")

    jobs = []
    for steps, guidance, seconds_per_prompt in CONFIGS:
        case_file = args.repo_root / f"examples/diffusion/leo2/data/context_ir_exact_s{steps}_g{guidance}.csv"
        if not case_file.is_file():
            raise FileNotFoundError(f"exact case file not found: {case_file}")
        for shard in shards:
            shard_index = int(shard["index"])
            prompts = Path(shard["path"])
            rows = int(shard["rows"])
            job_id = f"exact_s{steps}_g{guidance}_shard{shard_index:02d}"
            jobs.append(
                {
                    "job_id": job_id,
                    "output_root": str(
                        args.artifact_root / "runs/full_exact_sharded_20260907_v1" / job_id
                    ),
                    "cases_csv": str(case_file),
                    "prompts_csv": str(prompts),
                    "steps": steps,
                    "estimated_seconds": rows * seconds_per_prompt,
                }
            )
    jobs.sort(key=lambda job: (-job["estimated_seconds"], job["job_id"]))
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    with args.queue.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "job_id",
                "output_root",
                "cases_csv",
                "prompts_csv",
                "steps",
                "estimated_seconds",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(jobs)
    print(f"{args.queue}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
