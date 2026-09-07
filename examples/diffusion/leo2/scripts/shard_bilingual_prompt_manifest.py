#!/usr/bin/env python3
"""Split a bilingual prompt manifest without separating language pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.shards <= 0:
        raise ValueError(f"expected positive shard count, got {args.shards}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shard directory: {args.output_dir}")

    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    required = {"index", "seed", "prompt", "language", "pair_id"}
    if fieldnames is None or not required.issubset(fieldnames):
        raise ValueError(f"prompt manifest must contain {sorted(required)}, got {fieldnames}")
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    for pair_id, pair_rows in by_pair.items():
        languages = {row["language"] for row in pair_rows}
        seeds = {row["seed"] for row in pair_rows}
        if len(pair_rows) != 2 or languages != {"en", "zh"} or len(seeds) != 1:
            raise ValueError(
                f"expected one EN/ZH same-seed pair for pair_id={pair_id!r}, "
                f"got rows={len(pair_rows)}, languages={languages}, seeds={seeds}"
            )

    shards: list[list[dict[str, str]]] = [[] for _ in range(args.shards)]
    ordered_pairs = sorted(
        by_pair.values(),
        key=lambda pair: min(int(row["index"]) for row in pair),
    )
    for pair_position, pair_rows in enumerate(ordered_pairs):
        shards[pair_position % args.shards].extend(
            sorted(pair_rows, key=lambda row: int(row["index"]))
        )

    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "shard_count": args.shards,
        "pair_policy": "English and Chinese rows remain together",
        "shards": [],
    }
    all_indices = set()
    for shard_index, shard_rows in enumerate(shards):
        path = args.output_dir / f"full_200_shard{shard_index:02d}.csv"
        shard_rows.sort(key=lambda row: int(row["index"]))
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shard_rows)
        indices = [int(row["index"]) for row in shard_rows]
        if all_indices.intersection(indices):
            raise ValueError(f"duplicate prompt index across shards: {indices}")
        all_indices.update(indices)
        manifest["shards"].append(
            {
                "index": shard_index,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": len(shard_rows),
                "pairs": len(shard_rows) // 2,
                "prompt_indices": indices,
            }
        )
    expected_indices = {int(row["index"]) for row in rows}
    if all_indices != expected_indices:
        raise ValueError(
            f"shard coverage mismatch: expected {len(expected_indices)} indices, "
            f"got {len(all_indices)}"
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
