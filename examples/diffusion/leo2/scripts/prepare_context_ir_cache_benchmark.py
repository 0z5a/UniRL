#!/usr/bin/env python3
"""Build deterministic bilingual prompt manifests for Leo2 cache benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_FILES = {
    "pairs.jsonl": "63aec52997c9689207678d954ed02ae0de9d87658d95f4ce1c4bb089352e0464",
    "prompts.en.jsonl": "7bf0784cacfafd954b4abf0d2bc7ad15024366e9aec63225f201f832e3119592",
    "prompts.zh.jsonl": "2e5f1aca5b65f6659d81ee1a7d1a8582f9a35a33045d2a0490ef13de3112a1c4",
}
CSV_FIELDS = (
    "index",
    "seed",
    "prompt",
    "language",
    "pair_id",
    "source_dataset",
    "source_id",
    "prompt_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pairs(release: Path) -> list[dict[str, Any]]:
    for filename, expected in EXPECTED_FILES.items():
        path = release / filename
        if not path.is_file():
            raise FileNotFoundError(f"expected Context-IR release file, got {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Context-IR release digest mismatch for {path}: expected {expected}, got {actual}"
            )

    rows = []
    path = release / "pairs.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object at {path}:{line_number}, got {type(value).__name__}")
            required = {
                "pair_id",
                "prompt_en",
                "prompt_zh",
                "source_dataset",
                "source_id",
                "duration_seconds",
                "ratio",
                "prompt_format",
                "translation_review",
            }
            missing = sorted(required - set(value))
            if missing:
                raise ValueError(f"missing fields at {path}:{line_number}: {missing}")
            if value["pair_id"] != value.get("id"):
                raise ValueError(
                    f"pair identity mismatch at {path}:{line_number}: "
                    f"id={value.get('id')!r}, pair_id={value['pair_id']!r}"
                )
            if value["duration_seconds"] != 8.0 or value["ratio"] != "16:9":
                raise ValueError(
                    f"expected 8-second 16:9 prompt at {path}:{line_number}, "
                    f"got duration={value['duration_seconds']!r}, ratio={value['ratio']!r}"
                )
            if value["prompt_format"] != "av-inference-prompt/1":
                raise ValueError(
                    f"unexpected prompt_format at {path}:{line_number}: {value['prompt_format']!r}"
                )
            review = value["translation_review"]
            if not isinstance(review, dict) or review.get("approved") is not True:
                raise ValueError(f"translation is not approved at {path}:{line_number}")
            for field in ("prompt_en", "prompt_zh"):
                if not isinstance(value[field], str) or not value[field].strip():
                    raise TypeError(
                        f"expected non-empty string for {field} at {path}:{line_number}, "
                        f"got {type(value[field]).__name__}: {value[field]!r}"
                    )
            rows.append(value)
    if len(rows) != 100:
        raise ValueError(f"expected exactly 100 Context-IR bilingual pairs, got {len(rows)}")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Context-IR release contains duplicate pair_id values")
    return rows


def _quantile_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(rows):
        raise ValueError(f"expected count in [1, {len(rows)}], got {count}")
    ordered = sorted(
        rows,
        key=lambda row: (
            len(row["prompt_en"]) + len(row["prompt_zh"]),
            row["pair_id"],
        ),
    )
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(positions)) != count:
        raise RuntimeError(f"quantile selection produced duplicate positions: {positions}")
    return [ordered[position] for position in positions]


def _balanced_selection(rows: list[dict[str, Any]], per_source: int) -> list[dict[str, Any]]:
    sources = sorted({row["source_dataset"] for row in rows})
    if sources != ["vbench", "vidprom_50k"]:
        raise ValueError(f"expected vbench and vidprom_50k sources, got {sources}")
    selected = []
    for source in sources:
        selected.extend(_quantile_pick([row for row in rows if row["source_dataset"] == source], per_source))
    return sorted(selected, key=lambda row: row["pair_id"])


def _manifest_rows(
    pairs: list[dict[str, Any]],
    *,
    pair_positions: dict[str, int],
    seed_base: int,
) -> list[dict[str, object]]:
    output = []
    for pair in sorted(pairs, key=lambda row: pair_positions[row["pair_id"]]):
        pair_position = pair_positions[pair["pair_id"]]
        seed = seed_base + pair_position
        for language, prompt_field, language_offset in (
            ("en", "prompt_en", 0),
            ("zh", "prompt_zh", 1),
        ):
            prompt = pair[prompt_field]
            output.append(
                {
                    "index": 2 * pair_position + language_offset,
                    "seed": seed,
                    "prompt": prompt,
                    "language": language,
                    "pair_id": pair["pair_id"],
                    "source_dataset": pair["source_dataset"],
                    "source_id": pair["source_id"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/"
            "context_ir_prompt_data/release/context-ir-video-prompts-bilingual-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--seed-base", type=int, default=52000)
    args = parser.parse_args()

    release = args.release.resolve()
    pairs = _read_pairs(release)
    pair_positions = {row["pair_id"]: index for index, row in enumerate(pairs)}

    pilot = _balanced_selection(pairs, per_source=8)
    pilot_ids = {row["pair_id"] for row in pilot}
    remaining = [row for row in pairs if row["pair_id"] not in pilot_ids]
    calibration = _balanced_selection(remaining, per_source=4)

    outputs = {
        "context_ir_bilingual_smoke_1.csv": _manifest_rows(
            pilot[:1], pair_positions=pair_positions, seed_base=args.seed_base
        )[:1],
        "context_ir_bilingual_pilot_32.csv": _manifest_rows(
            pilot, pair_positions=pair_positions, seed_base=args.seed_base
        ),
        "context_ir_bilingual_calibration_16.csv": _manifest_rows(
            calibration, pair_positions=pair_positions, seed_base=args.seed_base
        ),
        "context_ir_bilingual_full_200.csv": _manifest_rows(
            pairs, pair_positions=pair_positions, seed_base=args.seed_base
        ),
    }
    for filename, rows in outputs.items():
        _write_csv(args.output_dir / filename, rows)

    manifest = {
        "schema_version": 1,
        "release": str(release),
        "release_files": {
            filename: {"sha256": digest, "path": str(release / filename)}
            for filename, digest in EXPECTED_FILES.items()
        },
        "seed_base": args.seed_base,
        "same_seed_policy": "English and Chinese rows of one pair share the same seed",
        "index_policy": "2 * release pair position + language offset (en=0, zh=1)",
        "video_contract": {
            "duration_seconds": 8,
            "fps": 24,
            "num_frames": 193,
            "frame_formula": "duration_seconds * fps + 1",
        },
        "pilot_selection": {
            "policy": "8 length-quantile pairs per source dataset",
            "pair_ids": [row["pair_id"] for row in pilot],
        },
        "calibration_selection": {
            "policy": "4 length-quantile pairs per source dataset, disjoint from pilot",
            "pair_ids": [row["pair_id"] for row in calibration],
        },
        "outputs": {
            filename: {
                "path": str((args.output_dir / filename).resolve()),
                "rows": len(rows),
                "sha256": _sha256(args.output_dir / filename),
            }
            for filename, rows in outputs.items()
        },
    }
    manifest_path = args.output_dir / "context_ir_bilingual_cache_benchmark.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
