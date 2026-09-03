"""Build a validated cross-root Leo2 acceleration comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
VIDEOSCORE_DIMENSIONS = ("visual_quality", "text_alignment", "physical_consistency")
FINGERPRINT_FIELDS = (
    "prompts_sha256",
    "artifact_manifest_sha256",
    "model_config_sha256",
    "generation_config_sha256",
    "checkpoint_metadata_sha256",
    "checkpoint_shard_inventory_sha256",
    "image_size",
    "num_frames",
    "diff_infer_steps",
    "expected_videos_per_case",
)


@dataclass(frozen=True)
class Metric:
    """Describe one row in the output matrix."""

    key: str
    description: str
    unit: str
    display: str = "number"


@dataclass(frozen=True)
class CaseResult:
    """Hold one validated case and its report values."""

    label: str
    name: str
    root: Path
    source_root: Path
    summary_path: Path
    quality_path: Path
    paired_path: Path
    pixel_path: Path
    env: Mapping[str, str]
    summary: Mapping[str, Any]
    identities: frozenset[tuple[int, str, int]]
    paired: tuple[Mapping[str, Any], ...]
    values: Mapping[str, Any]


METRICS = (
    Metric("case", "Artifact case name", "text", "text"),
    Metric("cache_method", "Acceleration method", "text", "text"),
    Metric("method_options", "Acceleration method options", "JSON", "text"),
    Metric("cache_threshold", "Cache threshold", "setting"),
    Metric("sample_count", "Paired prompt/video count", "count", "integer"),
    Metric("generation_mean_seconds", "Mean end-to-end generation latency", "seconds", "seconds"),
    Metric("generation_steady_mean_seconds", "Mean generation latency excluding first request", "seconds", "seconds"),
    Metric("paired_speedup_mean", "Mean prompt-paired speedup versus exact baseline", "x", "speedup"),
    Metric("paired_speedup_ci95_low", "Paired speedup mean 95% CI lower bound", "x", "speedup"),
    Metric("paired_speedup_ci95_high", "Paired speedup mean 95% CI upper bound", "x", "speedup"),
    Metric("tail_reuse_ratio", "Whole-tail residual reuse ratio", "ratio", "ratio"),
    Metric("attention_reuse_ratio", "Managed-attention reuse ratio", "ratio", "ratio"),
    Metric("cache_residency_mib", "Rank-maximum method cache residency", "MiB", "memory"),
    Metric("peak_allocated_gib", "Rank-maximum peak CUDA allocation", "GiB", "memory"),
    Metric("latent_relative_l1", "Mean per-video relative latent L1", "ratio"),
    Metric("latent_relative_l2", "Mean per-video relative latent L2", "ratio"),
    Metric("latent_mse", "Global final-latent mean squared error", "latent^2"),
    Metric("latent_global_rmse", "Global final-latent root mean squared error", "latent"),
    Metric("latent_cosine", "Mean per-video latent cosine similarity", "score"),
    Metric("latent_max_abs", "Worst-pair latent maximum absolute error", "latent"),
    Metric("pixel_mae", "Global decoded RGB mean absolute error", "RGB [0,1]"),
    Metric("pixel_relative_l1", "Mean per-video relative decoded RGB L1", "ratio"),
    Metric("pixel_relative_l2", "Mean per-video relative decoded RGB L2", "ratio"),
    Metric("pixel_mse", "Global decoded RGB mean squared error", "RGB^2 [0,1]"),
    Metric("pixel_global_rmse", "Global decoded RGB root mean squared error", "RGB [0,1]"),
    *(
        Metric(
            f"vbench_{name}",
            "VBench dynamic degree (descriptive motion-present rate)"
            if name == "dynamic_degree"
            else f"VBench {name.replace('_', ' ')}",
            "score",
        )
        for name in VBENCH_DIMENSIONS
    ),
    *(
        Metric(f"videoscore2_{name}", f"VideoScore2 {name.replace('_', ' ')}", "score")
        for name in VIDEOSCORE_DIMENSIONS
    ),
)


def _die(message: str) -> NoReturn:
    raise ValueError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _die(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    """Load one JSON file with finite numbers and unique keys."""
    if not path.is_file():
        _die(f"Missing JSON input: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: _die(f"Non-finite JSON number {value!r} in {path}"),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _read_env(path: Path) -> dict[str, str]:
    """Read a strict key-value benchmark metadata file."""
    if not path.is_file():
        _die(f"Missing benchmark metadata: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.lstrip().startswith("#"):
            continue
        key, separator, value = raw.partition("=")
        if not separator or not key or key in result:
            _die(f"Malformed or duplicate key at {path}:{line_number}: {raw!r}")
        result[key] = value
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while rejecting missing and duplicate headers."""
    if not path.is_file():
        _die(f"Missing CSV input: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            _die(f"Missing or duplicate CSV headers in {path}")
        return list(reader)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: Any, context: str) -> Path:
    """Resolve one required path relative to the manifest directory."""
    if not isinstance(value, str) or not value:
        _die(f"{context} must be a non-empty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _number(row: Mapping[str, Any], key: str, context: str) -> float:
    """Read one required finite number."""
    value = row.get(key)
    if isinstance(value, bool):
        _die(f"{context}: {key} must be numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid {key}={value!r}") from exc
    if not math.isfinite(result):
        _die(f"{context}: {key} must be finite, got {result!r}")
    return result


def _integer(row: Mapping[str, Any], key: str, context: str) -> int:
    """Read one required exact integer."""
    value = row.get(key)
    if isinstance(value, bool):
        _die(f"{context}: {key} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid {key}={value!r}") from exc
    if str(value).strip() not in {str(result), f"{result}.0"} and not (isinstance(value, float) and value == result):
        _die(f"{context}: {key} is not an exact integer: {value!r}")
    return result


def _same(left: float, right: float) -> bool:
    """Compare serialized settings or aggregate metrics."""
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _require_same(actual: float, expected: float, context: str) -> None:
    """Reject a numerical disagreement with context."""
    if not _same(actual, expected):
        _die(f"{context}: expected {expected!r}, got {actual!r}")


def _schema_version(summary: Mapping[str, Any], context: str) -> int:
    """Read the benchmark schema, defaulting historical records to v1."""
    value = summary.get("benchmark_schema_version", 1)
    version = _integer({"value": value}, "value", context)
    if version not in {1, 2}:
        _die(f"{context}: unsupported benchmark schema {version}")
    return version


def _guidance(row: Mapping[str, Any], schema_version: int, context: str) -> float:
    """Resolve guidance with the schema-v1 fixed-value contract."""
    value = row.get("guidance_scale")
    config = row.get("config")
    if value in (None, "") and isinstance(config, dict):
        value = config.get("guidance_scale")
    if value in (None, ""):
        if schema_version == 1:
            return 1.0
        _die(f"{context}: schema v2 record is missing guidance_scale")
    return _number({"value": value}, "value", context)


def _method(summary: Mapping[str, Any], context: str) -> str:
    """Resolve an explicit method or infer the schema-v1 method."""
    value = summary.get("cache_method")
    if value is None:
        enabled = summary.get("cache_enabled")
        if not isinstance(enabled, bool):
            _die(f"{context}: missing cache_method and boolean cache_enabled")
        return "first_block" if enabled else "off"
    if not isinstance(value, str) or not value:
        _die(f"{context}: invalid cache_method={value!r}")
    return value


def _method_options(summary: Mapping[str, Any], method: str) -> str:
    """Serialize method settings while retaining schema-v1 thresholds."""
    options = summary.get("cache_method_options")
    if options is None:
        threshold = summary.get("cache_threshold")
        options = {"threshold": threshold} if method != "off" and threshold is not None else {}
    if not isinstance(options, dict):
        _die(f"{summary.get('case')}: cache_method_options must be an object")
    return json.dumps(options, sort_keys=True, separators=(",", ":"))


def _baseline_root(row: Mapping[str, Any], schema_version: int, expected: Path, context: str) -> None:
    """Validate an explicit baseline root or the schema-v1 local-root convention."""
    value = row.get("baseline_root")
    if value in (None, ""):
        if schema_version != 1:
            _die(f"{context}: schema v2 row is missing baseline_root")
        return
    if not isinstance(value, str) or Path(value).expanduser().resolve() != expected:
        _die(f"{context}: baseline_root={value!r}, expected {expected}")


def _reference_root(row: Mapping[str, Any], source_root: Path, expected: Path, context: str) -> None:
    """Validate a case-level reference root."""
    value = row.get("reference_root")
    if value in (None, ""):
        if source_root != expected:
            _die(f"{context}: external baseline is missing reference_root")
        return
    if not isinstance(value, str) or Path(value).expanduser().resolve() != expected:
        _die(f"{context}: reference_root={value!r}, expected {expected}")


def _validate_source_paths(summary: Mapping[str, Any], source_root: Path, name: str, count: int, context: str) -> None:
    """Verify recorded latent and video artifacts belong to the declared source root."""
    expected_root = source_root / name
    for key in ("latent_validation", "video_validation"):
        records = summary.get(key)
        if not isinstance(records, list) or len(records) != count:
            _die(f"{context}: {key} must contain {count} rows")
        for record in records:
            if (
                not isinstance(record, dict)
                or record.get("valid") is not True
                or not isinstance(record.get("path"), str)
            ):
                _die(f"{context}: {key} has an invalid record")
            try:
                Path(record["path"]).expanduser().resolve().relative_to(expected_root)
            except ValueError as exc:
                raise ValueError(f"{context}: {key} path is outside {expected_root}: {record['path']}") from exc


def _validate_config(
    summary: Mapping[str, Any],
    env: Mapping[str, str],
    schema_version: int,
    expected_shift: float,
    expected_guidance: float,
    context: str,
) -> Mapping[str, Any]:
    """Cross-check benchmark.env, summary fields, and the recorded effective config."""
    config = summary.get("config")
    if not isinstance(config, dict):
        _die(f"{context}: summary config must be an object")
    for key in FINGERPRINT_FIELDS:
        if not env.get(key):
            _die(f"{context}: benchmark.env is missing {key}")
    if config.get("image_size") != env["image_size"]:
        _die(f"{context}: config image_size disagrees with benchmark.env")
    numeric_matches = (
        (config, "num_frames", env, "num_frames"),
        (summary, "diff_infer_steps", env, "diff_infer_steps"),
        (config, "diff_infer_steps", env, "diff_infer_steps"),
        (config, "world_size", summary, "world_size"),
    )
    for left, left_key, right, right_key in numeric_matches:
        if _integer(left, left_key, context) != _integer(right, right_key, context):
            _die(f"{context}: {left_key} disagrees with {right_key}")
    _require_same(_number(config, "flow_shift_video", context), expected_shift, f"{context}.config.flow_shift_video")
    _require_same(_guidance(config, schema_version, context), expected_guidance, f"{context}.config.guidance_scale")
    if config.get("cache_enabled") is not summary.get("cache_enabled"):
        _die(f"{context}: config and summary cache_enabled disagree")
    if schema_version == 2:
        method = _method(summary, context)
        if config.get("cache_method") != method:
            _die(f"{context}: config and summary cache_method disagree")
        if config.get("cache_method_options") != summary.get("cache_method_options"):
            _die(f"{context}: config and summary cache_method_options disagree")
        summary_threshold = summary.get("cache_threshold")
        config_threshold = config.get("cache_threshold")
        if (summary_threshold is None) != (config_threshold is None) or (
            summary_threshold is not None
            and not _same(
                _number({"value": summary_threshold}, "value", context),
                _number({"value": config_threshold}, "value", context),
            )
        ):
            _die(f"{context}: config and summary cache_threshold disagree")
    return config


def _latent_signature(summary: Mapping[str, Any], context: str) -> tuple[tuple[int, ...], str]:
    """Require one final-latent shape and dtype across all requests."""
    signatures = set()
    for request in summary["requests"]:
        shape = request.get("latent_shape")
        dtype = request.get("latent_dtype")
        if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value > 0 for value in shape):
            _die(f"{context}: invalid latent_shape={shape!r}")
        if not isinstance(dtype, str) or not dtype:
            _die(f"{context}: invalid latent_dtype={dtype!r}")
        signatures.add((tuple(shape), dtype))
    if len(signatures) != 1:
        _die(f"{context}: requests have mixed latent shapes or dtypes")
    return signatures.pop()


def _find_case(rows: Any, name: str, path: Path) -> Mapping[str, Any]:
    """Find exactly one named case in a JSON list."""
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        _die(f"Expected a JSON list of objects in {path}")
    matches = [row for row in rows if row.get("case") == name]
    if len(matches) != 1:
        _die(f"Expected exactly one case {name!r} in {path}, found {len(matches)}")
    return matches[0]


def _request_identities(
    summary: Mapping[str, Any],
    expected_count: int,
    expected_shift: float,
    expected_guidance: float,
    schema_version: int,
    context: str,
) -> frozenset[tuple[int, str, int]]:
    """Validate successful requests and return their pairing identities."""
    requests = summary.get("requests")
    if not isinstance(requests, list) or len(requests) != expected_count:
        _die(f"{context}: requests must contain exactly {expected_count} rows")
    identities: set[tuple[int, str, int]] = set()
    indices: set[int] = set()
    hashes: set[str] = set()
    seeds: set[int] = set()
    for position, request in enumerate(requests):
        item = f"{context}.requests[{position}]"
        if not isinstance(request, dict) or request.get("status") != "ok":
            _die(f"{item}: request is not a successful object")
        prompt_index = _integer(request, "prompt_index", item)
        seed = _integer(request, "seed", item)
        prompt_hash = request.get("prompt_hash")
        if not isinstance(prompt_hash, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_hash) is None:
            _die(f"{item}: invalid prompt_hash={prompt_hash!r}")
        _require_same(_number(request, "flow_shift_video", item), expected_shift, f"{item}.flow_shift_video")
        guidance = _guidance(request, schema_version, item)
        _require_same(guidance, expected_guidance, f"{item}.guidance_scale")
        identity = (prompt_index, prompt_hash, seed)
        if identity in identities or prompt_index in indices or prompt_hash in hashes or seed in seeds:
            _die(f"{context}: duplicate request index, hash, seed, or identity {identity}")
        identities.add(identity)
        indices.add(prompt_index)
        hashes.add(prompt_hash)
        seeds.add(seed)
        if guidance <= 0:
            _die(f"{item}: guidance_scale must be positive")
    return frozenset(identities)


def _paired_records(
    path: Path,
    name: str,
    expected_count: int,
    expected_shift: float,
    expected_guidance: float,
    expected_baseline: str,
    expected_baseline_root: Path,
    schema_version: int,
) -> tuple[list[dict[str, str]], frozenset[tuple[int, str, int]]]:
    """Validate prompt-paired timing and latent rows for one case."""
    rows = [row for row in _read_csv(path) if row.get("case") == name]
    if len(rows) != expected_count:
        _die(f"{path}: expected {expected_count} paired rows for {name}, found {len(rows)}")
    identities: set[tuple[int, str, int]] = set()
    for position, row in enumerate(rows):
        context = f"{path}:{name}[{position}]"
        identity = (_integer(row, "prompt_index", context), str(row.get("prompt_hash")), _integer(row, "seed", context))
        if re.fullmatch(r"[0-9a-f]{64}", identity[1]) is None or identity in identities:
            _die(f"{context}: invalid or duplicate prompt identity {identity}")
        identities.add(identity)
        baseline = row.get("baseline_case")
        if baseline != expected_baseline:
            _die(f"{context}: baseline_case={baseline!r}, expected {expected_baseline!r}")
        _baseline_root(row, schema_version, expected_baseline_root, context)
        _require_same(_number(row, "flow_shift_video", context), expected_shift, f"{context}.flow_shift_video")
        _require_same(_guidance(row, schema_version, context), expected_guidance, f"{context}.guidance_scale")
        for key in ("candidate_seconds", "baseline_seconds", "speedup"):
            if _number(row, key, context) <= 0:
                _die(f"{context}: {key} must be positive")
        for key in ("rel_l1", "rel_l2", "max_abs"):
            if _number(row, key, context) < 0:
                _die(f"{context}: {key} must be non-negative")
        cosine = _number(row, "cosine", context)
        if not -1.0 <= cosine <= 1.0:
            _die(f"{context}: cosine must be in [-1, 1]")
    return rows, frozenset(identities)


def _pixel_records(
    path: Path,
    name: str,
    expected_count: int,
    expected_shift: float,
    expected_guidance: float,
    expected_baseline: str,
    expected_baseline_root: Path,
    candidate_root: Path,
    expected_image_size: str,
    expected_frames: int,
    schema_version: int,
) -> tuple[list[dict[str, str]], frozenset[tuple[int, str, int]]]:
    """Validate paired decoded-pixel rows for one accelerated case."""
    rows = [row for row in _read_csv(path) if row.get("case") == name]
    if len(rows) != expected_count:
        _die(f"{path}: expected {expected_count} pixel rows for {name}, found {len(rows)}")
    try:
        expected_height, expected_width = (int(value) for value in expected_image_size.split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid benchmark image_size={expected_image_size!r}") from exc
    identities: set[tuple[int, str, int]] = set()
    for position, row in enumerate(rows):
        context = f"{path}:{name}[{position}]"
        identity = (_integer(row, "prompt_index", context), str(row.get("prompt_hash")), _integer(row, "seed", context))
        if re.fullmatch(r"[0-9a-f]{64}", identity[1]) is None or identity in identities:
            _die(f"{context}: invalid or duplicate prompt identity {identity}")
        identities.add(identity)
        if row.get("baseline_case") != expected_baseline:
            _die(f"{context}: baseline_case={row.get('baseline_case')!r}, expected {expected_baseline!r}")
        _baseline_root(row, schema_version, expected_baseline_root, context)
        _require_same(_number(row, "flow_shift_video", context), expected_shift, f"{context}.flow_shift_video")
        _require_same(_guidance(row, schema_version, context), expected_guidance, f"{context}.guidance_scale")
        if _integer(row, "rgb_sample_count", context) <= 0:
            _die(f"{context}: rgb_sample_count must be positive")
        width = _integer(row, "width", context)
        height = _integer(row, "height", context)
        frames = _integer(row, "frames", context)
        if (width, height, frames) != (expected_width, expected_height, expected_frames):
            _die(f"{context}: decoded geometry disagrees with benchmark.env")
        if _number(row, "fps", context) <= 0:
            _die(f"{context}: FPS must be positive")
        if _integer(row, "rgb_sample_count", context) != width * height * frames * 3:
            _die(f"{context}: rgb_sample_count disagrees with decoded geometry")
        mae, rmse, max_abs = (_number(row, key, context) for key in ("mae", "rmse", "max_abs"))
        if not 0.0 <= mae <= rmse <= max_abs <= 1.0:
            _die(f"{context}: expected 0 <= MAE <= RMSE <= max_abs <= 1")
        for key in ("relative_l1", "relative_l2"):
            if _number(row, key, context) < 0:
                _die(f"{context}: {key} must be non-negative")
        paths = (
            ("candidate_video", "candidate_video_sha256", candidate_root / name),
            ("baseline_video", "baseline_video_sha256", expected_baseline_root / expected_baseline),
        )
        for path_key, digest_key, expected_root in paths:
            video = Path(str(row.get(path_key))).expanduser().resolve()
            try:
                video.relative_to(expected_root)
            except ValueError as exc:
                raise ValueError(f"{context}: {path_key} is outside {expected_root}: {video}") from exc
            if video.name != f"{identity[0]}_0.mp4" or not video.is_file():
                _die(f"{context}: missing video: {video}")
            recorded_digest = row.get(digest_key)
            if recorded_digest in (None, ""):
                if schema_version != 1:
                    _die(f"{context}: schema v2 row is missing {digest_key}")
            elif recorded_digest != _sha256(video):
                _die(f"{context}: mismatched {digest_key}: {video}")
    return rows, frozenset(identities)


def _mean(rows: Sequence[Mapping[str, Any]], key: str, context: str) -> float:
    """Return an arithmetic mean of a required numeric column."""
    return sum(_number(row, key, context) for row in rows) / len(rows)


def _validate_quality(
    quality: Mapping[str, Any],
    summary: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    pixels: Sequence[Mapping[str, Any]],
    expected_count: int,
    expected_shift: float,
    expected_guidance: float,
    schema_version: int,
    baseline: bool,
    context: str,
) -> dict[str, Any]:
    """Cross-check aggregate quality values against timing and pair artifacts."""
    method = _method(summary, context)
    enabled = summary.get("cache_enabled")
    if not isinstance(enabled, bool) or enabled is (method == "off"):
        _die(f"{context}: cache_enabled and cache_method disagree")
    quality_disabled = quality.get("cache_disabled")
    if not isinstance(quality_disabled, bool) or quality_disabled is not (not enabled):
        _die(f"{context}: quality cache_disabled and summary cache_enabled disagree")
    quality_method = quality.get("cache_method")
    if quality_method not in (None, "", method):
        _die(f"{context}: quality cache_method={quality_method!r}, expected {method!r}")
    if _integer(quality, "sample_count", context) != expected_count:
        _die(f"{context}: sample_count differs from the manifest")
    _require_same(_number(quality, "flow_shift_video", context), expected_shift, f"{context}.flow_shift_video")
    _require_same(_guidance(quality, schema_version, context), expected_guidance, f"{context}.guidance_scale")
    cross_checks = {
        "generation_mean_seconds": "generation_mean_seconds",
        "paired_speedup_mean": "paired_speedup_mean",
        "skip_ratio": "skip_ratio",
        "latent_relative_l1": "latent_rel_l1_mean",
        "latent_relative_l2": "latent_rel_l2_mean",
        "latent_cosine": "latent_cosine_mean",
    }
    for quality_key, summary_key in cross_checks.items():
        _require_same(_number(quality, quality_key, context), _number(summary, summary_key, context), quality_key)
    pair_checks = {
        "paired_speedup_mean": "speedup",
        "latent_relative_l1": "rel_l1",
        "latent_relative_l2": "rel_l2",
        "latent_cosine": "cosine",
    }
    for quality_key, pair_key in pair_checks.items():
        _require_same(_number(quality, quality_key, context), _mean(paired, pair_key, context), quality_key)
    _require_same(
        _number(summary, "latent_max_abs_max", context),
        max(_number(row, "max_abs", context) for row in paired),
        f"{context}.latent_max_abs_max",
    )

    latent_mse = _number(quality, "latent_mse", context)
    latent_rmse = _number(quality, "latent_rmse", context)
    if latent_mse < 0 or latent_rmse < 0:
        _die(f"{context}: latent MSE/RMSE must be non-negative")
    _require_same(latent_rmse * latent_rmse, latent_mse, f"{context}.latent_rmse^2")

    if baseline:
        pixel = {
            key: _number(quality, key, context)
            for key in ("pixel_mse", "pixel_rmse", "pixel_mae", "pixel_relative_l1", "pixel_relative_l2")
        }
        if (
            any(value != 0.0 for value in pixel.values())
            or latent_mse != 0.0
            or _number(summary, "latent_max_abs_max", context) != 0.0
        ):
            _die(f"{context}: exact baseline drift must be zero")
    else:
        total_samples = sum(_integer(row, "rgb_sample_count", context) for row in pixels)
        pixel_mse = (
            sum(_number(row, "rmse", context) ** 2 * _integer(row, "rgb_sample_count", context) for row in pixels)
            / total_samples
        )
        pixel_mae = (
            sum(_number(row, "mae", context) * _integer(row, "rgb_sample_count", context) for row in pixels)
            / total_samples
        )
        _require_same(_number(quality, "pixel_mse", context), pixel_mse, f"{context}.pixel_mse")
        _require_same(_number(quality, "pixel_rmse", context), math.sqrt(pixel_mse), f"{context}.pixel_rmse")
        _require_same(_number(quality, "pixel_mae", context), pixel_mae, f"{context}.pixel_mae")
        for quality_key, pair_key in (
            ("pixel_relative_l1", "relative_l1"),
            ("pixel_relative_l2", "relative_l2"),
        ):
            _require_same(_number(quality, quality_key, context), _mean(pixels, pair_key, context), quality_key)

    for key in VBENCH_DIMENSIONS:
        value = _number(quality, f"vbench_{key}", context)
        if not 0.0 <= value <= 1.0:
            _die(f"{context}: VBench {key} must be in [0, 1]")
    for key in VIDEOSCORE_DIMENSIONS:
        value = _number(quality, f"videoscore2_{key}", context)
        if not 1.0 <= value <= 5.0:
            _die(f"{context}: VideoScore2 {key} must be in [1, 5]")

    cache_bytes = summary.get("cache_bytes")
    peak_bytes = _number(summary, "max_memory_allocated_bytes", context)
    mean_seconds = _number(summary, "generation_mean_seconds", context)
    steady_seconds = _number(summary, "generation_steady_mean_seconds", context)
    speedup = _number(summary, "paired_speedup_mean", context)
    speedup_low = _number(summary, "paired_speedup_mean_ci95_low", context)
    speedup_high = _number(summary, "paired_speedup_mean_ci95_high", context)
    tail_reuse = (
        _number(summary, "tail_reuse_ratio", context)
        if summary.get("tail_reuse_ratio") is not None
        else _number(summary, "skip_ratio", context)
    )
    attention_reuse = (
        _number(summary, "attention_reuse_ratio", context) if summary.get("attention_reuse_ratio") is not None else 0.0
    )
    if mean_seconds <= 0 or steady_seconds <= 0 or speedup <= 0:
        _die(f"{context}: latency and speedup values must be positive")
    if not speedup_low <= speedup <= speedup_high:
        _die(f"{context}: paired speedup mean is outside its confidence interval")
    if not 0.0 <= tail_reuse <= 1.0 or not 0.0 <= attention_reuse <= 1.0:
        _die(f"{context}: reuse ratios must be in [0, 1]")
    if peak_bytes < 0 or (cache_bytes not in (None, "") and _number({"value": cache_bytes}, "value", context) < 0):
        _die(f"{context}: memory byte counts must be non-negative")
    if any(_number(quality, key, context) < 0 for key in ("latent_relative_l1", "latent_relative_l2")):
        _die(f"{context}: relative latent errors must be non-negative")
    latent_cosine = _number(quality, "latent_cosine", context)
    if not -1.0 <= latent_cosine <= 1.0:
        _die(f"{context}: latent cosine must be in [-1, 1]")
    return {
        "case": summary["case"],
        "cache_method": method,
        "method_options": _method_options(summary, method),
        "cache_threshold": summary.get("cache_threshold"),
        "sample_count": expected_count,
        "generation_mean_seconds": mean_seconds,
        "generation_steady_mean_seconds": steady_seconds,
        "paired_speedup_mean": speedup,
        "paired_speedup_ci95_low": speedup_low,
        "paired_speedup_ci95_high": speedup_high,
        "tail_reuse_ratio": tail_reuse,
        "attention_reuse_ratio": attention_reuse,
        "cache_residency_mib": _number({"value": cache_bytes}, "value", context) / 2**20
        if cache_bytes not in (None, "")
        else None,
        "peak_allocated_gib": peak_bytes / 2**30,
        "latent_relative_l1": _number(quality, "latent_relative_l1", context),
        "latent_relative_l2": _number(quality, "latent_relative_l2", context),
        "latent_mse": latent_mse,
        "latent_global_rmse": latent_rmse,
        "latent_cosine": latent_cosine,
        "latent_max_abs": _number(summary, "latent_max_abs_max", context),
        "pixel_mae": _number(quality, "pixel_mae", context),
        "pixel_relative_l1": _number(quality, "pixel_relative_l1", context),
        "pixel_relative_l2": _number(quality, "pixel_relative_l2", context),
        "pixel_mse": _number(quality, "pixel_mse", context),
        "pixel_global_rmse": _number(quality, "pixel_rmse", context),
        **{f"vbench_{key}": _number(quality, f"vbench_{key}", context) for key in VBENCH_DIMENSIONS},
        **{f"videoscore2_{key}": _number(quality, f"videoscore2_{key}", context) for key in VIDEOSCORE_DIMENSIONS},
    }


def _load_case(
    entry: Mapping[str, Any],
    manifest_dir: Path,
    expected_count: int,
    expected_shift: float,
    expected_guidance: float,
    baseline_label: str,
    baseline_name: str,
    baseline_source_root: Path,
) -> CaseResult:
    """Load and internally validate one manifest case."""
    label = entry.get("label")
    name = entry.get("case")
    if not isinstance(label, str) or not label or not isinstance(name, str) or not name:
        _die("Every manifest case needs non-empty label and case strings")
    root = _resolve(manifest_dir, entry.get("artifact_root"), f"cases[{label}].artifact_root")
    if not root.is_dir():
        _die(f"Artifact root is not a directory: {root}")
    source_root = _resolve(
        manifest_dir, entry.get("source_root", entry.get("artifact_root")), f"cases[{label}].source_root"
    )

    def artifact_path(key: str, default: str) -> Path:
        value = entry.get(key)
        return _resolve(manifest_dir, value, f"cases[{label}].{key}") if value is not None else root / default

    summary_path = artifact_path("summary_json", "summary.json")
    quality_path = artifact_path("quality_metrics_json", "quality_eval/summary/quality_metrics_cases.json")
    paired_path = artifact_path("paired_metrics_csv", "paired_metrics.csv")
    pixel_path = artifact_path("pixel_metrics_pairs_csv", "pixel_metrics_pairs.csv")
    env = _read_env(root / "benchmark.env")
    summary = _find_case(_read_json(summary_path), name, summary_path)
    quality = _find_case(_read_json(quality_path), name, quality_path)
    context = f"case {label!r} ({name})"
    schema_version = _schema_version(summary, context)
    if summary.get("complete") is not True or _integer(summary, "exit_code", context) != 0:
        _die(f"{context}: benchmark is not complete and successful")
    for key in ("expected_videos", "request_count", "video_count", "video_valid_count", "latent_valid_count"):
        if _integer(summary, key, context) != expected_count:
            _die(f"{context}: {key} differs from expected sample count {expected_count}")
    if _integer(summary, "latent_pair_count", context) != expected_count:
        _die(f"{context}: latent_pair_count differs from expected sample count {expected_count}")
    if env.get("expected_videos_per_case") != str(expected_count):
        _die(f"{context}: benchmark.env expected_videos_per_case differs from {expected_count}")
    _validate_source_paths(summary, source_root, name, expected_count, context)
    config = _validate_config(summary, env, schema_version, expected_shift, expected_guidance, context)
    _require_same(_number(summary, "flow_shift_video", context), expected_shift, f"{context}.flow_shift_video")
    _require_same(_guidance(summary, schema_version, context), expected_guidance, f"{context}.guidance_scale")
    identities = _request_identities(
        summary, expected_count, expected_shift, expected_guidance, schema_version, context
    )
    paired, paired_identities = _paired_records(
        paired_path,
        name,
        expected_count,
        expected_shift,
        expected_guidance,
        baseline_name,
        baseline_source_root,
        schema_version,
    )
    baseline = label == baseline_label
    if schema_version == 2:
        quality_source = quality.get("quality_source_root")
        if not isinstance(quality_source, str) or Path(quality_source).expanduser().resolve() != source_root:
            _die(f"{context}.quality: quality_source_root must be {source_root}")
    if baseline:
        pixels: list[dict[str, str]] = []
    else:
        if schema_version == 2:
            for record, record_context in (
                (summary, context),
                (config, f"{context}.config"),
                (quality, f"{context}.quality"),
            ):
                if record.get("baseline_case") != baseline_name:
                    _die(f"{record_context}: baseline_case must be {baseline_name!r}")
        _reference_root(summary, source_root, baseline_source_root, context)
        _reference_root(config, source_root, baseline_source_root, f"{context}.config")
        _baseline_root(quality, schema_version, baseline_source_root, f"{context}.quality")
        pixels, pixel_identities = _pixel_records(
            pixel_path,
            name,
            expected_count,
            expected_shift,
            expected_guidance,
            baseline_name,
            baseline_source_root,
            source_root,
            env["image_size"],
            int(env["num_frames"]),
            schema_version,
        )
        if pixel_identities != identities:
            _die(f"{context}: pixel-pair prompt identities differ from summary requests")
    if paired_identities != identities:
        _die(f"{context}: latent/timing-pair prompt identities differ from summary requests")
    values = _validate_quality(
        quality,
        summary,
        paired,
        pixels,
        expected_count,
        expected_shift,
        expected_guidance,
        schema_version,
        baseline,
        context,
    )
    return CaseResult(
        label=label,
        name=name,
        root=root,
        source_root=source_root,
        summary_path=summary_path,
        quality_path=quality_path,
        paired_path=paired_path,
        pixel_path=pixel_path,
        env=env,
        summary=summary,
        identities=identities,
        paired=tuple(paired),
        values=values,
    )


def _sampling_signature(case: CaseResult) -> dict[str, Any]:
    """Extract settings that must match the exact baseline."""
    config = case.summary.get("config")
    if not isinstance(config, dict):
        _die(f"{case.label}: summary config must be an object")
    parallelism = config.get("parallelism")
    if not isinstance(parallelism, dict):
        _die(f"{case.label}: summary config lacks parallelism")
    return {
        **{key: case.env.get(key) for key in FINGERPRINT_FIELDS},
        "world_size": _integer(case.summary, "world_size", case.label),
        "parallelism": parallelism,
        "timing_scope": config.get("timing_scope"),
        "latent": _latent_signature(case.summary, case.label),
    }


def _request_times(case: CaseResult) -> dict[tuple[int, str, int], float]:
    """Index validated request timings by prompt identity."""
    result = {}
    for request in case.summary["requests"]:
        identity = (int(request["prompt_index"]), str(request["prompt_hash"]), int(request["seed"]))
        result[identity] = _number(request, "elapsed_seconds", case.label)
    return result


def _validate_cross_case(cases: Sequence[CaseResult], baseline_label: str) -> CaseResult:
    """Validate baseline identity, prompts, sampling, and artifact fingerprints."""
    by_label = {case.label: case for case in cases}
    if len(by_label) != len(cases):
        _die("Manifest case labels must be unique")
    baseline = by_label.get(baseline_label)
    if baseline is None:
        _die(f"baseline_label {baseline_label!r} is absent from cases")
    if _method(baseline.summary, baseline.label) != "off" or baseline.summary.get("cache_enabled") is not False:
        _die(f"Baseline {baseline.label!r} must be a cache-off exact case")
    signature = _sampling_signature(baseline)
    baseline_times = _request_times(baseline)
    if any(value in (None, "") for value in signature.values()):
        _die(f"Baseline {baseline.label!r} has an incomplete sampling/fingerprint signature")
    for case in cases:
        method = _method(case.summary, case.label)
        if case is not baseline and (method == "off" or case.summary.get("cache_enabled") is not True):
            _die(f"{case.label}: non-baseline report case must enable an acceleration method")
        if case.identities != baseline.identities:
            _die(f"{case.label}: prompt index/hash/seed set differs from exact baseline")
        if _sampling_signature(case) != signature:
            _die(f"{case.label}: sampling topology or artifact fingerprints differ from exact baseline")
        if _schema_version(case.summary, case.label) == 1 and case.source_root != baseline.source_root:
            _die(f"{case.label}: schema-v1 local baseline must share the exact source root")
        recorded_baseline = case.summary.get("baseline_case")
        if recorded_baseline not in (None, "", baseline.name):
            _die(f"{case.label}: summary baseline_case={recorded_baseline!r}, expected {baseline.name!r}")
        candidate_times = _request_times(case)
        for row in case.paired:
            identity = (int(row["prompt_index"]), str(row["prompt_hash"]), int(row["seed"]))
            candidate_seconds = _number(row, "candidate_seconds", case.label)
            baseline_seconds = _number(row, "baseline_seconds", case.label)
            speedup = _number(row, "speedup", case.label)
            _require_same(candidate_seconds, candidate_times[identity], f"{case.label}.candidate_seconds")
            _require_same(baseline_seconds, baseline_times[identity], f"{case.label}.baseline_seconds")
            _require_same(speedup, baseline_seconds / candidate_seconds, f"{case.label}.speedup")
    return baseline


def _format_csv(value: Any) -> str:
    """Serialize a matrix value without presentation rounding."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _format_markdown(value: Any, display: str) -> str:
    """Format one human-readable table cell."""
    if value is None:
        return "N/A"
    if display == "text":
        return str(value).replace("|", "\\|")
    if display == "integer":
        return str(value)
    number = float(value)
    if display == "seconds":
        return f"{number:.3f}"
    if display == "speedup":
        return f"{number:.4f}x"
    if display == "ratio":
        return f"{number:.2%}"
    if display == "memory":
        return f"{number:.3f}"
    return f"{number:.8g}"


def _markdown_text(value: str) -> str:
    """Escape one plain-text Markdown table value."""
    return value.replace("|", "\\|").replace("\n", " ")


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace one text output."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _slug_number(value: float) -> str:
    """Create a stable filename token for one numerical setting."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _render_csv(cases: Sequence[CaseResult]) -> str:
    """Render the metric-by-setting CSV matrix."""
    rows = [["metric", "description", "unit", *(case.label for case in cases)]]
    rows.extend(
        [metric.key, metric.description, metric.unit, *(_format_csv(case.values[metric.key]) for case in cases)]
        for metric in METRICS
    )
    output = []
    for row in rows:
        buffer = []
        for value in row:
            escaped = str(value).replace('"', '""')
            buffer.append(f'"{escaped}"' if any(character in escaped for character in ',"\n') else escaped)
        output.append(",".join(buffer))
    return "\n".join(output) + "\n"


def _render_markdown(cases: Sequence[CaseResult], shift: float, guidance: float) -> str:
    """Render one metric-by-setting Markdown table."""
    lines = [
        f"# Leo2 acceleration results: flow shift {shift:g}, guidance {guidance:g}",
        "",
        "MSE is mean squared error over all equal-size samples; global RMSE is its square root.",
        "Relative L1/L2 rows are means of per-video norm ratios. `N/A` means the source schema did not record the value.",
        "",
        f"| Metric | Unit | {' | '.join(_markdown_text(case.label) for case in cases)} |",
        f"|---|---|{'|'.join('---:' for _ in cases)}|",
    ]
    lines.extend(
        f"| {metric.description} | {metric.unit} | "
        f"{' | '.join(_format_markdown(case.values[metric.key], metric.display) for case in cases)} |"
        for metric in METRICS
    )
    return "\n".join(lines) + "\n"


def _render_json(
    cases: Sequence[CaseResult], manifest_path: Path, baseline: CaseResult, shift: float, guidance: float, count: int
) -> str:
    """Render the complete machine-readable report."""
    document = {
        "schema_version": 1,
        "flow_shift_video": shift,
        "guidance_scale": guidance,
        "sample_count": count,
        "baseline_label": baseline.label,
        "baseline_case": baseline.name,
        "metric_semantics": {
            "latent_mse": "mean of per-video final-latent elementwise MSE; latents have identical shape",
            "latent_global_rmse": "sqrt(latent_mse)",
            "pixel_mse": "sum squared RGB24 error / decoded RGB sample count after normalization to [0,1]",
            "pixel_global_rmse": "sqrt(pixel_mse)",
            "relative_l1_l2": "arithmetic mean of per-video candidate-reference norm / reference norm",
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "sources": [
            {
                "label": case.label,
                "case": case.name,
                "artifact_root": str(case.root),
                "source_root": str(case.source_root),
                "benchmark_env": str(case.root / "benchmark.env"),
                "benchmark_env_sha256": _sha256(case.root / "benchmark.env"),
                "summary_json": str(case.summary_path),
                "summary_sha256": _sha256(case.summary_path),
                "quality_metrics_json": str(case.quality_path),
                "quality_metrics_sha256": _sha256(case.quality_path),
                "paired_metrics_csv": str(case.paired_path),
                "paired_metrics_sha256": _sha256(case.paired_path),
                "pixel_metrics_pairs_csv": str(case.pixel_path) if case.pixel_path.is_file() else None,
                "pixel_metrics_pairs_sha256": _sha256(case.pixel_path) if case.pixel_path.is_file() else None,
            }
            for case in cases
        ],
        "metrics": [
            {
                "key": metric.key,
                "description": metric.description,
                "unit": metric.unit,
                "values": {case.label: case.values[metric.key] for case in cases},
            }
            for metric in METRICS
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_manifest(path: Path) -> tuple[float, float, int, str, list[Mapping[str, Any]]]:
    """Load and validate the top-level report manifest."""
    manifest = _read_json(path)
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "expected", "cases"}:
        _die("Manifest must contain exactly schema_version, expected and cases")
    if _integer(manifest, "schema_version", "manifest") != 1:
        _die("Only report manifest schema_version=1 is supported")
    expected = manifest.get("expected")
    if not isinstance(expected, dict) or set(expected) != {
        "flow_shift_video",
        "guidance_scale",
        "sample_count",
        "baseline_label",
    }:
        _die("Manifest expected must contain flow_shift_video, guidance_scale, sample_count and baseline_label")
    shift = _number(expected, "flow_shift_video", "manifest.expected")
    guidance = _number(expected, "guidance_scale", "manifest.expected")
    count = _integer(expected, "sample_count", "manifest.expected")
    baseline_label = expected.get("baseline_label")
    entries = manifest.get("cases")
    if shift <= 0 or guidance <= 0 or count <= 0:
        _die("Manifest shift, guidance and sample count must be positive")
    if not isinstance(baseline_label, str) or not baseline_label:
        _die("Manifest baseline_label must be a non-empty string")
    if not isinstance(entries, list) or len(entries) < 2 or not all(isinstance(entry, dict) for entry in entries):
        _die("Manifest cases must be a list of at least two objects")
    allowed = {
        "label",
        "case",
        "artifact_root",
        "source_root",
        "summary_json",
        "quality_metrics_json",
        "paired_metrics_csv",
        "pixel_metrics_pairs_csv",
    }
    for position, entry in enumerate(entries):
        extra = set(entry) - allowed
        if extra:
            _die(f"Manifest cases[{position}] has unsupported keys: {sorted(extra)}")
        if not all(isinstance(entry.get(key), str) and entry[key] for key in ("label", "case", "artifact_root")):
            _die(f"Manifest cases[{position}] needs non-empty label, case and artifact_root strings")
    if len({entry["label"] for entry in entries}) != len(entries):
        _die("Manifest case labels must be unique")
    if sum(entry["label"] == baseline_label for entry in entries) != 1:
        _die(f"Manifest must contain exactly one baseline_label={baseline_label!r} entry")
    return shift, guidance, count, baseline_label, entries


def main() -> None:
    """Validate explicit artifact roots and write one comparison matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Explicit JSON report manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", help="Output basename; defaults to shiftN_guidanceN")
    parser.add_argument("--force", action="store_true", help="Atomically replace existing outputs")
    args = parser.parse_args()

    manifest_path = args.spec.expanduser().resolve()
    shift, guidance, count, baseline_label, entries = _parse_manifest(manifest_path)
    baseline_entry = next(entry for entry in entries if entry["label"] == baseline_label)
    baseline_name = baseline_entry["case"]
    baseline_source_root = _resolve(
        manifest_path.parent,
        baseline_entry.get("source_root", baseline_entry["artifact_root"]),
        f"cases[{baseline_label}].source_root",
    )
    cases = [
        _load_case(
            entry,
            manifest_path.parent,
            count,
            shift,
            guidance,
            baseline_label,
            baseline_name,
            baseline_source_root,
        )
        for entry in entries
    ]
    baseline = _validate_cross_case(cases, baseline_label)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"shift{_slug_number(shift)}_guidance{_slug_number(guidance)}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        _die(f"Unsafe output name: {name!r}")
    outputs = {suffix: output_dir / f"{name}.{suffix}" for suffix in ("csv", "md", "json")}
    inputs = {manifest_path}
    for case in cases:
        inputs.update(
            {
                case.root / "benchmark.env",
                case.summary_path,
                case.quality_path,
                case.paired_path,
                case.pixel_path,
            }
        )
    collisions = set(outputs.values()) & {path.resolve() for path in inputs}
    if collisions:
        _die(f"Output paths collide with report inputs: {sorted(collisions)}")
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        _die(f"Refusing to overwrite outputs without --force: {existing}")
    contents = {
        "csv": _render_csv(cases),
        "md": _render_markdown(cases, shift, guidance),
        "json": _render_json(cases, manifest_path, baseline, shift, guidance, count),
    }
    for key, path in outputs.items():
        _atomic_write(path, contents[key])
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
