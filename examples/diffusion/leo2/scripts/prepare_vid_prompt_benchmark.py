#!/usr/bin/env python3
"""Prepare deterministic Leo2 CSV inputs from Flow-Factory vid_prompt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"expected prompt source file, got {path}")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"expected at least one non-empty prompt in {path}, got 0")
    return prompts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "seed", "prompt"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/Flow-Factory/dataset/vid_prompt/train.txt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()

    prompts = _prompts(args.source)
    if args.limit < 1 or args.limit > len(prompts):
        raise ValueError(
            f"expected limit in [1, {len(prompts)}] for {args.source}, got {args.limit}"
        )
    if args.shards < 1 or args.shards > args.limit:
        raise ValueError(
            f"expected shards in [1, {args.limit}] for limit={args.limit}, got {args.shards}"
        )

    rows = [
        {"index": index, "seed": args.seed_base + index, "prompt": prompt}
        for index, prompt in enumerate(prompts[: args.limit])
    ]
    output = args.output_dir / f"vid_prompt_train_first{args.limit}.csv"
    _write_csv(output, rows)

    shard_paths: list[Path] = []
    for shard_index in range(args.shards):
        shard_path = args.output_dir / (
            f"vid_prompt_train_first{args.limit}.shard{shard_index:02d}-of-{args.shards:02d}.csv"
        )
        _write_csv(shard_path, rows[shard_index :: args.shards])
        shard_paths.append(shard_path)

    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "limit": args.limit,
        "seed_base": args.seed_base,
        "shards": args.shards,
        "sharding": "round_robin_by_global_index",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "shard_files": [
            {
                "path": str(path.resolve()),
                "rows": len(rows[index :: args.shards]),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(shard_paths)
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
