#!/usr/bin/env python3
"""Build a replayable Leo2 MagCache profile from a completed calibration case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, NoReturn

from cache_benchmark_cases import load_cases

REQUEST_MARKER = "LEO2_CACHE_BENCH_REQUEST_JSON="
CONFIG_MARKER = "LEO2_CACHE_BENCH_CONFIG_JSON="
DIGEST = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
HARNESS_FILES = (
    "examples/diffusion/leo2/CACHE_BENCHMARK.md",
    "examples/diffusion/leo2/data/cache_benchmark_16.csv",
    "examples/diffusion/leo2/data/cache_benchmark_pilot_4.csv",
    "examples/diffusion/leo2/data/cache_benchmark_cases.csv",
    "examples/diffusion/leo2/scripts/cache_benchmark_cases.py",
    "examples/diffusion/leo2/scripts/cache_benchmark.sh",
    "examples/diffusion/leo2/scripts/cache_benchmark_entry.py",
    "examples/diffusion/leo2/scripts/build_magcache_profile.py",
    "examples/diffusion/leo2/scripts/summarize_cache_benchmark.py",
)


def _die(message: str) -> NoReturn:
    """Raise a profile-validation error."""
    raise ValueError(message)


def _sha256(path: Path) -> str:
    """Hash one file without loading it completely."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_env(path: Path) -> dict[str, str]:
    """Read a unique-key shell environment record."""
    if not path.is_file():
        _die(f"Missing metadata file: {path}")
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            _die(f"Malformed or duplicate entry at {path}:{line_number}")
        result[key] = value
    return result


def _required(metadata: dict[str, str], key: str, *, source: Path) -> str:
    """Read one required non-empty metadata field."""
    value = metadata.get(key)
    if not value:
        _die(f"Missing {key}= in {source}")
    return value


def _digest(metadata: dict[str, str], key: str, *, source: Path) -> str:
    """Read one required lowercase SHA-256 value."""
    value = _required(metadata, key, source=source)
    if DIGEST.fullmatch(value) is None:
        _die(f"Invalid {key}= digest in {source}: {value!r}")
    return value


def _verify_file(metadata: dict[str, str], path_key: str, digest_key: str, *, source: Path) -> Path:
    """Resolve and digest-check one provenance input."""
    path = Path(_required(metadata, path_key, source=source)).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        _die(f"Missing or symbolic provenance file for {path_key}: {path}")
    expected = _digest(metadata, digest_key, source=source)
    actual = _sha256(path)
    if actual != expected:
        _die(f"{path_key} digest mismatch: expected {expected}, got {actual} ({path})")
    return path


def _git_output(repo: Path, *args: str) -> bytes:
    """Run a read-only Git provenance query."""
    return subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _shard_inventory_digest(checkpoint: Path) -> str:
    """Reproduce the launcher's sorted checkpoint shard inventory digest."""
    rows = [
        f"{path.name} {path.stat().st_size}\n"
        for path in checkpoint.glob("*.distcp")
        if path.is_file() and not path.is_symlink()
    ]
    if not rows:
        _die(f"Checkpoint has no .distcp shards: {checkpoint}")
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()


def _harness_digest(repo: Path) -> str:
    """Reproduce the launcher's ordered benchmark-harness digest."""
    records = []
    for relative in HARNESS_FILES:
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            _die(f"Missing or symbolic harness file: {path}")
        records.append(f"{_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(records).encode()).hexdigest()


def _harness_digest_at_revision(repo: Path, revision: str) -> str:
    """Reproduce the harness digest from the exact recorded Git revision."""
    records = []
    for relative in HARNESS_FILES:
        content = _git_output(repo, "show", f"{revision}:{relative}")
        records.append(f"{hashlib.sha256(content).hexdigest()}  {relative}\n")
    return hashlib.sha256("".join(records).encode()).hexdigest()


def _load_prompts(path: Path) -> dict[tuple[int, str], int]:
    """Load unique index/hash identities while allowing bilingual shared seeds."""
    identities = {}
    indices = set()
    hashes = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"index", "seed", "prompt"}.issubset(reader.fieldnames or ()):
            _die(f"Prompt CSV lacks index, seed or prompt: {path}")
        for line_number, row in enumerate(reader, 2):
            try:
                index = int(row["index"])
                seed = int(row["seed"])
                prompt = row["prompt"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prompt at {path}:{line_number}") from exc
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if index in indices or prompt_hash in hashes:
                _die(f"Duplicate prompt identity at {path}:{line_number}")
            identities[(seed, prompt_hash)] = index
            indices.add(index)
            hashes.add(prompt_hash)
    if not identities:
        _die(f"Prompt CSV is empty: {path}")
    return identities


def _read_markers(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read exactly one config marker and unique request markers."""
    if not path.is_file() or path.is_symlink():
        _die(f"Missing or symbolic calibration log: {path}")
    configs = []
    requests = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if CONFIG_MARKER in line:
            configs.append(json.loads(line.split(CONFIG_MARKER, 1)[1]))
        elif REQUEST_MARKER in line:
            request = json.loads(line.split(REQUEST_MARKER, 1)[1])
            number = request.get("request")
            if not isinstance(number, int) or number in requests:
                _die(f"Invalid or duplicate request marker at {path}:{line_number}")
            requests[number] = request
    if len(configs) != 1 or not isinstance(configs[0], dict):
        _die(f"Expected exactly one config marker in {path}, found {len(configs)}")
    return configs[0], [requests[number] for number in sorted(requests)]


def _finite_array(value: Any, *, field: str, length: int, positive: bool) -> list[float]:
    """Validate one fixed-size JSON numeric array."""
    if not isinstance(value, list) or len(value) != length:
        _die(f"{field} must contain exactly {length} values")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            _die(f"{field}[{index}] is not numeric: {item!r}")
        item = float(item)
        if not math.isfinite(item) or (positive and item <= 0):
            _die(f"{field}[{index}] is not {'positive and ' if positive else ''}finite: {item!r}")
        result.append(item)
    return result


def _same_float(left: Any, right: Any, *, field: str) -> None:
    """Require two serialized configuration values to match exactly."""
    try:
        matches = float(left) == float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} values: {left!r}, {right!r}") from exc
    if not matches:
        _die(f"Mismatched {field}: {left!r} != {right!r}")


def _validate_provenance(root: Path, case_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate benchmark, Git, checkpoint, config and prompt provenance."""
    if not root.is_dir() or root.is_symlink():
        _die(f"Benchmark root is missing or symbolic: {root}")
    env_path = root / "benchmark.env"
    metadata = _read_env(env_path)
    if metadata.get("benchmark_schema_version") != "2":
        _die(f"MagCache profiles require benchmark schema version 2: {env_path}")
    try:
        expected_steps = int(metadata["diff_infer_steps"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"benchmark.env diff_infer_steps must be a positive integer, "
            f"got {metadata.get('diff_infer_steps')!r}"
        ) from exc
    if expected_steps <= 0:
        _die(f"benchmark.env diff_infer_steps must be positive, got {expected_steps}")

    repo = Path(_required(metadata, "repo_root", source=env_path)).resolve()
    if not repo.is_dir() or repo.is_symlink():
        _die(f"Recorded repo_root is missing or symbolic: {repo}")
    git_head = _required(metadata, "git_head", source=env_path)
    if GIT_COMMIT.fullmatch(git_head) is None:
        _die(f"Invalid git_head in {env_path}: {git_head!r}")
    current_head = _git_output(repo, "rev-parse", "HEAD").decode().strip()
    ancestor = subprocess.run(
        ("git", "-C", str(repo), "merge-base", "--is-ancestor", git_head, current_head),
        check=False,
    )
    if ancestor.returncode != 0:
        _die(f"Recorded git_head {git_head} is not an ancestor of current HEAD {current_head}")
    expected_diff = _digest(metadata, "git_diff_sha256", source=env_path)
    actual_diff = hashlib.sha256(_git_output(repo, "diff", "--binary", "HEAD", "--")).hexdigest()
    expected_harness = _digest(metadata, "benchmark_harness_sha256", source=env_path)
    harness_revision = "working-tree"
    if current_head == git_head and actual_diff == expected_diff:
        actual_harness = _harness_digest(repo)
    else:
        revisions = _git_output(repo, "rev-list", "--reverse", f"{git_head}..{current_head}").decode().splitlines()
        matching = [
            revision
            for revision in revisions
            if _harness_digest_at_revision(repo, revision) == expected_harness
        ]
        if not matching:
            _die(
                "Recorded benchmark harness cannot be reproduced from the current "
                f"descendant history of {git_head}"
            )
        harness_revision = matching[0]
        actual_harness = expected_harness
    if actual_harness != expected_harness:
        _die(f"Recorded benchmark harness digest no longer matches {repo}")
    _digest(metadata, "artifact_manifest_sha256", source=env_path)

    prompts_path = _verify_file(metadata, "prompts_csv", "prompts_sha256", source=env_path)
    cases_path = _verify_file(metadata, "cases_csv", "cases_sha256", source=env_path)
    model_config = _verify_file(metadata, "model_config", "model_config_sha256", source=env_path)
    generation_config = _verify_file(metadata, "generation_config", "generation_config_sha256", source=env_path)
    artifact_manifest = _verify_file(metadata, "artifact_manifest", "artifact_manifest_sha256", source=env_path)

    checkpoint = Path(_required(metadata, "checkpoint_dir", source=env_path)).resolve()
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        _die(f"Checkpoint directory is missing or symbolic: {checkpoint}")
    checkpoint_metadata = checkpoint / ".metadata"
    if not checkpoint_metadata.is_file() or checkpoint_metadata.is_symlink():
        _die(f"Checkpoint metadata is missing or symbolic: {checkpoint_metadata}")
    if _sha256(checkpoint_metadata) != _digest(metadata, "checkpoint_metadata_sha256", source=env_path):
        _die(f"Checkpoint metadata digest mismatch: {checkpoint_metadata}")
    inventory = _shard_inventory_digest(checkpoint)
    if inventory != _digest(metadata, "checkpoint_shard_inventory_sha256", source=env_path):
        _die(f"Checkpoint shard inventory digest mismatch: {checkpoint}")

    cases = {case.name: case for case in load_cases(cases_path)}
    case_spec = cases.get(case_name)
    if case_spec is None or case_spec.method != "magcache_calibrate":
        _die(f"Case {case_name!r} is not a magcache_calibrate row in {cases_path}")
    case_dir = root / case_name
    if not case_dir.is_dir() or case_dir.is_symlink() or case_dir.resolve().parent != root:
        _die(f"Unsafe or missing calibration case directory: {case_dir}")
    case_env_path = case_dir / "case.env"
    case_env = _read_env(case_env_path)
    if case_env.get("case") != case_name or case_env.get("method") != "magcache_calibrate":
        _die(f"Case metadata does not identify magcache_calibrate: {case_env_path}")
    expected_requests = int(_required(case_env, "expected_videos", source=case_env_path))
    if int(_required(metadata, "expected_videos_per_case", source=env_path)) != expected_requests:
        _die("Root and case expected-video counts differ")
    exit_code = (case_dir / "exit_code.txt").read_text(encoding="utf-8").strip()
    if exit_code != "0":
        _die(f"Calibration case did not exit successfully: {case_dir / 'exit_code.txt'}")
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("complete") is not True
        or summary.get("request_count") != expected_requests
        or summary.get("case") != case_name
        or summary.get("cache_method") != "magcache_calibrate"
        or summary.get("diff_infer_steps") != expected_steps
    ):
        _die(f"Calibration case summary is incomplete: {summary_path}")

    config, requests = _read_markers(case_dir / "run.log")
    if len(requests) != expected_requests:
        _die(f"Expected {expected_requests} request markers, found {len(requests)}")
    request_numbers = [request["request"] for request in requests]
    if request_numbers != list(range(1, expected_requests + 1)):
        _die(f"Calibration request numbers are not contiguous: {request_numbers}")
    prompts = _load_prompts(prompts_path)
    if expected_requests > len(prompts):
        _die("Calibration request count exceeds the prompt CSV")
    config_options = config.get("cache_method_options")
    if config.get("cache_method") != "magcache_calibrate" or not isinstance(config_options, dict):
        _die("Calibration config marker lacks MagCache method options")
    if config_options.get("calibrate") is not True:
        _die("Calibration config marker does not set calibrate=true")
    _same_float(config_options.get("threshold"), case_spec.magcache_threshold, field="magcache_threshold")
    if config_options.get("max_skip_steps") != case_spec.magcache_max_skip_steps:
        _die("Calibration max_skip_steps disagrees with the cases CSV")
    _same_float(
        config_options.get("retention_ratio"),
        case_spec.magcache_retention_ratio,
        field="magcache_retention_ratio",
    )
    if config.get("diff_infer_steps") != expected_steps:
        _die(
            f"Calibration config marker does not contain {expected_steps} denoising steps"
        )
    _same_float(config.get("flow_shift_video"), case_spec.flow_shift_video, field="flow_shift_video")
    _same_float(config.get("guidance_scale"), case_spec.guidance_scale, field="guidance_scale")
    if str(config.get("image_size")) != metadata.get("image_size"):
        _die("Config marker and benchmark.env image sizes differ")
    if int(config.get("num_frames")) != int(metadata.get("num_frames", -1)):
        _die("Config marker and benchmark.env frame counts differ")

    provenance = {
        "benchmark_env": str(env_path),
        "benchmark_env_sha256": _sha256(env_path),
        "benchmark_harness_sha256": metadata["benchmark_harness_sha256"],
        "benchmark_root": str(root),
        "case": case_name,
        "case_summary_sha256": _sha256(summary_path),
        "checkpoint_dir": str(checkpoint),
        "checkpoint_metadata_sha256": metadata["checkpoint_metadata_sha256"],
        "checkpoint_shard_inventory_sha256": inventory,
        "git_diff_sha256": expected_diff,
        "git_head": git_head,
        "harness_revision": harness_revision,
        "repo_root": str(repo),
        "run_log_sha256": _sha256(case_dir / "run.log"),
        "prompts_csv": str(prompts_path),
        "prompts_sha256": metadata["prompts_sha256"],
        "cases_csv": str(cases_path),
        "cases_sha256": metadata["cases_sha256"],
        "model_config": str(model_config),
        "model_config_sha256": metadata["model_config_sha256"],
        "generation_config": str(generation_config),
        "generation_config_sha256": metadata["generation_config_sha256"],
        "artifact_manifest": str(artifact_manifest),
        "artifact_manifest_sha256": metadata["artifact_manifest_sha256"],
    }
    return config, requests, {
        "expected_steps": expected_steps,
        "identities": prompts,
        "source": provenance,
    }


def _build_profile(root: Path, case_name: str) -> dict[str, Any]:
    """Validate one calibration case and aggregate its per-step ratios."""
    config, requests, context = _validate_provenance(root, case_name)
    identities = context["identities"]
    expected_steps = context["expected_steps"]
    ratio_rows = []
    timesteps = None
    seen = {"prompt_index": set(), "prompt_hash": set()}
    prompt_identities = []
    for position, request in enumerate(requests):
        label = f"request[{position}]"
        if request.get("status") != "ok" or request.get("cache_method") != "magcache_calibrate":
            _die(f"{label} is not a successful MagCache calibration request")
        if request.get("diff_infer_steps") != expected_steps:
            _die(f"{label} does not report {expected_steps} denoising steps")
        if request.get("cache_method_options") != config["cache_method_options"]:
            _die(f"{label} MagCache options disagree with the config marker")
        for field in ("flow_shift_video", "guidance_scale"):
            _same_float(request.get(field), config.get(field), field=f"{label}.{field}")
        if str(request.get("image_size")) != str(config.get("image_size")):
            _die(f"{label} image_size disagrees with the config marker")
        if request.get("num_frames") != config.get("num_frames"):
            _die(f"{label} num_frames disagrees with the config marker")
        full_steps = int(request.get("full_steps", -1))
        skipped_steps = int(request.get("skipped_steps", -1))
        if full_steps != expected_steps or skipped_steps != 0:
            _die(f"{label} must compute all {expected_steps} steps during calibration")
        ratios = _finite_array(
            request.get("magcache_ratios"),
            field=f"{label}.magcache_ratios",
            length=expected_steps,
            positive=True,
        )
        current_timesteps = _finite_array(
            request.get("magcache_expected_timesteps"),
            field=f"{label}.magcache_expected_timesteps",
            length=expected_steps,
            positive=False,
        )
        if ratios[0] != 1.0:
            _die(f"{label}.magcache_ratios[0] must equal 1.0")
        if timesteps is None:
            timesteps = current_timesteps
        elif current_timesteps != timesteps:
            _die(f"{label} timesteps differ from the first calibration request")
        seed = request.get("seed")
        prompt_index = request.get("prompt_index")
        prompt_hash = request.get("prompt_hash")
        if not isinstance(seed, int) or not isinstance(prompt_index, int) or not isinstance(prompt_hash, str):
            _die(f"{label} has an invalid prompt identity")
        if DIGEST.fullmatch(prompt_hash) is None:
            _die(f"{label} has an invalid prompt hash")
        expected = identities.get((seed, prompt_hash))
        if expected != prompt_index:
            _die(f"{label} identity disagrees with the prompt CSV")
        for field, value in (("prompt_index", prompt_index), ("prompt_hash", prompt_hash)):
            if value in seen[field]:
                _die(f"Duplicate {field} in calibration requests: {value!r}")
            seen[field].add(value)
        prompt_identities.append({"prompt_hash": prompt_hash, "prompt_index": prompt_index, "seed": seed})
        ratio_rows.append(ratios)
    if timesteps is None:
        _die("Calibration case contains no requests")
    expected_indices = set(list(identities.values())[: len(requests)])
    if seen["prompt_index"] != expected_indices:
        _die("Calibration requests do not match the prompt CSV's selected rows")

    columns = list(zip(*ratio_rows))
    means = [statistics.fmean(column) for column in columns]
    standard_deviations = [statistics.stdev(column) if len(column) > 1 else 0.0 for column in columns]
    p95 = [sorted(column)[max(0, math.ceil(0.95 * len(column)) - 1)] for column in columns]
    return {
        "schema_version": 1,
        "profile_type": "leo2_magcache",
        "ratios": means,
        "expected_timesteps": timesteps,
        "ratio_statistics": {
            "aggregation": {
                "mean": "arithmetic mean across prompts at each denoising step",
                "std": "sample standard deviation across prompts at each step (ddof=1; zero for n=1)",
                "p95": "nearest-rank 95th percentile across prompts at each step",
            },
            "mean": means,
            "std": standard_deviations,
            "p95": p95,
            "sample_count": len(ratio_rows),
        },
        "sampling": {
            "cache_method_options": config["cache_method_options"],
            "diff_infer_steps": config["diff_infer_steps"],
            "flow_shift_video": config["flow_shift_video"],
            "guidance_scale": config["guidance_scale"],
            "image_size": config["image_size"],
            "num_frames": config["num_frames"],
        },
        "prompt_identities": sorted(prompt_identities, key=lambda item: item["prompt_index"]),
        "source": context["source"],
    }


def _atomic_create(path: Path, document: dict[str, Any]) -> None:
    """Atomically create JSON without replacing an existing path."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite MagCache profile: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Build and atomically save one MagCache calibration profile."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    profile = _build_profile(root, args.case)
    _atomic_create(args.output, profile)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": _sha256(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
