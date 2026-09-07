from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
SCRIPTS = REPO_ROOT / "examples/diffusion/leo2/scripts"
DATA = REPO_ROOT / "examples/diffusion/leo2/data"


def test_bilingual_shards_and_exact_queue_cover_every_prompt(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    subprocess.run(
        (
            sys.executable,
            str(SCRIPTS / "shard_bilingual_prompt_manifest.py"),
            "--input",
            str(DATA / "context_ir_bilingual_full_200.csv"),
            "--shards",
            "8",
            "--output-dir",
            str(shard_dir),
        ),
        check=True,
    )
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    assert sum(shard["rows"] for shard in manifest["shards"]) == 200
    assert sum(shard["pairs"] for shard in manifest["shards"]) == 100
    indices = [
        prompt_index
        for shard in manifest["shards"]
        for prompt_index in shard["prompt_indices"]
    ]
    assert sorted(indices) == list(range(200))
    for shard in manifest["shards"]:
        with Path(shard["path"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_pair: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_pair.setdefault(row["pair_id"], []).append(row)
        assert all(
            {row["language"] for row in pair} == {"en", "zh"}
            and len({row["seed"] for row in pair}) == 1
            for pair in by_pair.values()
        )

    queue = tmp_path / "queue.tsv"
    subprocess.run(
        (
            sys.executable,
            str(SCRIPTS / "prepare_cache_job_queue.py"),
            "--shard-manifest",
            str(shard_dir / "manifest.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--repo-root",
            str(REPO_ROOT),
            "--queue",
            str(queue),
        ),
        check=True,
    )
    with queue.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    assert len(jobs) == 32
    assert len({job["job_id"] for job in jobs}) == 32
    estimates = [float(job["estimated_seconds"]) for job in jobs]
    assert estimates == sorted(estimates, reverse=True)
    assert {int(job["steps"]) for job in jobs} == {6, 28}
