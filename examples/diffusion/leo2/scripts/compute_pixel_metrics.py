#!/usr/bin/env python3
"""Compute streaming RGB metrics for a completed Leo2 cache benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn, Sequence

import imageio_ffmpeg
import numpy as np
from cache_benchmark_cases import CacheBenchmarkCase, load_cases

EXPECTED_WIDTH = 848
EXPECTED_HEIGHT = 464
EXPECTED_FRAMES = 121
EXPECTED_FPS = 24.0
FPS_ABS_TOLERANCE = 1e-6
VIDEO_NAME = re.compile(r"^(?P<prompt_index>[0-9]+)_0[.]mp4$")
METRIC_NAMES = ("mae", "rmse", "relative_l1", "relative_l2", "max_abs")

PAIR_FIELDS = (
    "case",
    "baseline_case",
    "baseline_root",
    "cache_method",
    "cache_threshold",
    "flow_shift_video",
    "guidance_scale",
    "prompt_index",
    "prompt_hash",
    "seed",
    "candidate_video",
    "candidate_video_sha256",
    "baseline_video",
    "baseline_video_sha256",
    "width",
    "height",
    "frames",
    "fps",
    "rgb_sample_count",
    *METRIC_NAMES,
)

CASE_FIELDS = (
    "case",
    "baseline_case",
    "baseline_root",
    "cache_method",
    "cache_threshold",
    "flow_shift_video",
    "guidance_scale",
    "pair_count",
    *(f"{metric}_mean" for metric in METRIC_NAMES),
    *(f"{metric}_max" for metric in METRIC_NAMES),
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_index: int
    seed: int
    prompt_hash: str


@dataclass(frozen=True)
class CheckedCase:
    spec: CacheBenchmarkCase
    root: Path
    summary: Mapping[str, Any]
    requests: Mapping[tuple[float, str, int], Mapping[str, Any]]
    videos: Mapping[int, Path]


def _die(message: str) -> NoReturn:
    raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        _die(f"Missing benchmark metadata: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.lstrip().startswith("#"):
            continue
        key, separator, value = raw.partition("=")
        if not separator or not key:
            _die(f"Malformed {path}:{line_number}: {raw!r}")
        if key in result:
            _die(f"Duplicate key {key!r} in {path}")
        result[key] = value
    return result


def _resolve_input(path: str | None, metadata: Mapping[str, str], key: str) -> Path:
    value = path if path is not None else metadata.get(key)
    if not value:
        _die(f"No --{key.replace('_', '-')} was supplied and benchmark.env has no {key}= entry")
    result = Path(value).expanduser().resolve()
    if not result.is_file():
        _die(f"Input file does not exist: {result}")
    # The benchmark keys are prompts_sha256 and cases_sha256.
    recorded_digest = metadata.get(key.replace("_csv", "_sha256"))
    if recorded_digest is not None:
        actual_digest = _sha256(result)
        if actual_digest != recorded_digest:
            _die(f"{key} digest differs from benchmark.env: expected {recorded_digest}, got {actual_digest} ({result})")
    return result


def _load_prompts(path: Path) -> tuple[PromptSpec, ...]:
    prompts: list[PromptSpec] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"index", "seed", "prompt"}.issubset(reader.fieldnames):
            _die(f"Prompt CSV must contain index, seed and prompt columns: {path}")
        for line_number, row in enumerate(reader, 2):
            try:
                prompt_index = int(row["index"])
                seed = int(row["seed"])
                prompt = row["prompt"]
            except (KeyError, TypeError, ValueError) as exc:
                _die(f"Invalid prompt CSV row at {path}:{line_number}: {exc}")
            if prompt is None or not prompt.strip():
                _die(f"Empty prompt at {path}:{line_number}")
            prompts.append(
                PromptSpec(
                    prompt_index=prompt_index,
                    seed=seed,
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                )
            )
    if not prompts:
        _die(f"Prompt CSV is empty: {path}")
    for field, values in (
        ("index", [prompt.prompt_index for prompt in prompts]),
        ("seed", [prompt.seed for prompt in prompts]),
        ("prompt hash", [prompt.prompt_hash for prompt in prompts]),
    ):
        if len(set(values)) != len(values):
            _die(f"Duplicate {field} in {path}")
    return tuple(sorted(prompts, key=lambda prompt: prompt.prompt_index))


def _load_summary(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        _die(f"Missing final summary: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid or partially-written JSON in {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        _die(f"Expected a JSON list of case objects in {path}")
    return tuple(value)


def _require_int(row: Mapping[str, Any], key: str, context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        _die(f"{context}: {key} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid integer {key}={value!r}") from exc
    if isinstance(value, float) and value != result:
        _die(f"{context}: non-integral {key}={value!r}")
    return result


def _require_float(row: Mapping[str, Any], key: str, context: str) -> float:
    try:
        result = float(row.get(key))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid float {key}={row.get(key)!r}") from exc
    if not math.isfinite(result):
        _die(f"{context}: {key} must be finite, got {result!r}")
    return result


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _paths_under(directory: Path, paths: Sequence[Path], context: str) -> None:
    resolved_directory = directory.resolve()
    for path in paths:
        try:
            path.resolve().relative_to(resolved_directory)
        except ValueError as exc:
            raise ValueError(f"{context}: path escapes {resolved_directory}: {path}") from exc


def _video_map(root: Path, case_dir: Path, expected_indices: set[int]) -> dict[int, Path]:
    sample_dir = case_dir / "samples"
    if not sample_dir.is_dir():
        _die(f"Missing samples directory: {sample_dir}")
    videos = sorted(sample_dir.rglob("*.mp4"))
    if len(videos) != len(expected_indices):
        _die(f"{case_dir.name}: expected {len(expected_indices)} MP4 files, found {len(videos)}")
    _paths_under(root, videos, case_dir.name)
    result: dict[int, Path] = {}
    for video in videos:
        if video.is_symlink():
            _die(f"{case_dir.name}: video must not be a symbolic link: {video}")
        match = VIDEO_NAME.fullmatch(video.name)
        if match is None:
            _die(f"{case_dir.name}: unexpected MP4 filename (wanted <index>_0.mp4): {video}")
        prompt_index = int(match.group("prompt_index"))
        if prompt_index in result:
            _die(f"{case_dir.name}: duplicate video for prompt index {prompt_index}")
        result[prompt_index] = video.resolve()
    if set(result) != expected_indices:
        _die(f"{case_dir.name}: video indices differ from successful request indices")
    return result


def _validate_summary_video_records(row: Mapping[str, Any], videos: Mapping[int, Path], context: str) -> None:
    records = row.get("video_validation")
    if not isinstance(records, list) or len(records) != len(videos):
        _die(f"{context}: video_validation must contain {len(videos)} records")
    recorded_paths: set[Path] = set()
    for position, record in enumerate(records):
        item_context = f"{context}.video_validation[{position}]"
        if not isinstance(record, dict) or record.get("valid") is not True:
            _die(f"{item_context}: summary did not mark video valid")
        try:
            path = Path(record["path"]).resolve()
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{item_context}: missing/invalid path") from exc
        if path in recorded_paths:
            _die(f"{item_context}: duplicate path {path}")
        recorded_paths.add(path)
        if _require_int(record, "width", item_context) != EXPECTED_WIDTH:
            _die(f"{item_context}: summary width is not {EXPECTED_WIDTH}")
        if _require_int(record, "height", item_context) != EXPECTED_HEIGHT:
            _die(f"{item_context}: summary height is not {EXPECTED_HEIGHT}")
        if _require_int(record, "frames", item_context) != EXPECTED_FRAMES:
            _die(f"{item_context}: summary frame count is not {EXPECTED_FRAMES}")
        if not math.isclose(
            _require_float(record, "fps", item_context),
            EXPECTED_FPS,
            rel_tol=0.0,
            abs_tol=FPS_ABS_TOLERANCE,
        ):
            _die(f"{item_context}: summary FPS is not {EXPECTED_FPS:g}")
    actual_paths = {path.resolve() for path in videos.values()}
    if recorded_paths != actual_paths:
        missing = sorted(str(path) for path in actual_paths - recorded_paths)
        extra = sorted(str(path) for path in recorded_paths - actual_paths)
        _die(f"{context}: summary/actual video paths differ; missing={missing}, extra={extra}")


def _check_case(
    root: Path,
    spec: CacheBenchmarkCase,
    row: Mapping[str, Any],
    prompts: Sequence[PromptSpec],
    *,
    allow_prompt_superset: bool = False,
) -> CheckedCase:
    context = f"case {spec.name}"
    if row.get("case") != spec.name:
        _die(f"{context}: summary case name mismatch: {row.get('case')!r}")
    if row.get("complete") is not True or _require_int(row, "exit_code", context) != 0:
        _die(f"{context}: benchmark case is not complete and successful")
    summary_method = row.get("cache_method") or ("first_block" if row.get("cache_enabled") is True else "off")
    if summary_method != spec.method:
        _die(f"{context}: cache_method disagrees with cases CSV: {summary_method!r} != {spec.method!r}")
    if row.get("cache_enabled") is not spec.cache_enabled:
        _die(f"{context}: cache_enabled disagrees with cases CSV")
    threshold = row.get("cache_threshold")
    if spec.method == "off":
        if threshold is not None:
            _die(f"{context}: cache-off summary has threshold {threshold!r}")
    elif spec.cache_threshold is not None and not _same_float(
        _require_float(row, "cache_threshold", context), spec.cache_threshold
    ):
        _die(f"{context}: cache threshold disagrees with cases CSV")
    summary_shift = _require_float(row, "flow_shift_video", context)
    if not _same_float(summary_shift, spec.flow_shift_video):
        _die(f"{context}: flow shift disagrees with cases CSV")
    summary_guidance = float(row.get("guidance_scale", 1.0))
    if not _same_float(summary_guidance, spec.guidance_scale):
        _die(f"{context}: guidance_scale disagrees with cases CSV")
    count = _require_int(row, "expected_videos", context)
    if (allow_prompt_superset and count < len(prompts)) or (not allow_prompt_superset and count != len(prompts)):
        _die(f"{context}: expected_videos={count} is incompatible with {len(prompts)} selected prompts")
    for field in ("request_count", "video_count", "video_valid_count"):
        if _require_int(row, field, context) != count:
            _die(f"{context}: {field} is not {count}")

    raw_requests = row.get("requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != count:
        _die(f"{context}: requests must contain exactly {count} records")
    requests: dict[tuple[float, str, int], Mapping[str, Any]] = {}
    seen_indices: set[int] = set()
    expected_by_index = {prompt.prompt_index: prompt for prompt in prompts}
    for position, request in enumerate(raw_requests):
        request_context = f"{context}.requests[{position}]"
        if not isinstance(request, dict) or request.get("status") != "ok":
            _die(f"{request_context}: request is not a successful object")
        prompt_index = _require_int(request, "prompt_index", request_context)
        seed = _require_int(request, "seed", request_context)
        prompt_hash = request.get("prompt_hash")
        if not isinstance(prompt_hash, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_hash) is None:
            _die(f"{request_context}: invalid SHA-256 prompt hash {prompt_hash!r}")
        expected = expected_by_index.get(prompt_index)
        if expected is not None and (expected.seed != seed or expected.prompt_hash != prompt_hash):
            _die(f"{request_context}: prompt index/hash/seed disagree with prompts CSV")
        if prompt_index in seen_indices:
            _die(f"{context}: duplicate request for prompt index {prompt_index}")
        seen_indices.add(prompt_index)
        if not _same_float(_require_float(request, "flow_shift_video", request_context), spec.flow_shift_video):
            _die(f"{request_context}: flow shift disagrees with case")
        request_threshold = request.get("cache_threshold")
        if spec.method == "off":
            if request_threshold is not None:
                _die(f"{request_context}: cache-off request has a threshold")
        elif spec.cache_threshold is not None and not _same_float(
            _require_float(request, "cache_threshold", request_context), spec.cache_threshold
        ):
            _die(f"{request_context}: cache threshold disagrees with case")
        key = (spec.flow_shift_video, prompt_hash, seed)
        if key in requests:
            _die(f"{context}: duplicate pairing key {key}")
        requests[key] = request

    selected_indices = set(expected_by_index)
    if not selected_indices.issubset(seen_indices):
        _die(f"{context}: selected prompt indices are missing")
    if not allow_prompt_superset and seen_indices != selected_indices:
        _die(f"{context}: request prompt indices differ from the prompt CSV")
    case_dir = root / spec.name
    if not case_dir.is_dir() or case_dir.resolve().parent != root:
        _die(f"Missing or unsafe case directory: {case_dir}")
    videos = _video_map(root, case_dir, seen_indices)
    _validate_summary_video_records(row, videos, context)
    return CheckedCase(spec=spec, root=root, summary=row, requests=requests, videos=videos)


def _validate_inputs(
    root: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[CacheBenchmarkCase],
    prompts: Sequence[PromptSpec],
) -> tuple[tuple[CheckedCase, ...], dict[str, CheckedCase]]:
    summary_by_name: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(summary_rows):
        name = row.get("case")
        if not isinstance(name, str) or not name:
            _die(f"summary[{position}] has no valid case name")
        if name in summary_by_name:
            _die(f"Duplicate case {name!r} in summary.json")
        summary_by_name[name] = row
    expected_names = {case.name for case in cases}
    if set(summary_by_name) != expected_names:
        missing = sorted(expected_names - set(summary_by_name))
        extra = sorted(set(summary_by_name) - expected_names)
        _die(f"Final summary/cases CSV differ; missing={missing}, extra={extra}")
    checked = tuple(_check_case(root, case, summary_by_name[case.name], prompts) for case in cases)
    local = {case.spec.name: case for case in checked}
    external: dict[tuple[Path, str], CheckedCase] = {}
    baselines: dict[str, CheckedCase] = {}
    for candidate in (case for case in checked if case.spec.cache_enabled):
        baseline_name = candidate.spec.baseline_case
        baseline_root = Path(candidate.spec.reference_root).resolve() if candidate.spec.reference_root else root
        if baseline_name is None:
            controls = [
                case
                for case in checked
                if not case.spec.cache_enabled
                and _same_float(case.spec.flow_shift_video, candidate.spec.flow_shift_video)
                and _same_float(case.spec.guidance_scale, candidate.spec.guidance_scale)
            ]
            if len(controls) != 1:
                _die(f"{candidate.spec.name}: expected one same-setting cache-off baseline")
            baseline = controls[0]
        elif baseline_root == root and baseline_name in local:
            baseline = local[baseline_name]
        else:
            key = (baseline_root, baseline_name)
            baseline = external.get(key)
            if baseline is None:
                rows = _load_summary(baseline_root / "summary.json")
                matches = [row for row in rows if row.get("case") == baseline_name]
                if len(matches) != 1:
                    _die(f"Expected one {baseline_name!r} row in {baseline_root / 'summary.json'}")
                baseline_row = matches[0]
                method = baseline_row.get("cache_method") or (
                    "first_block" if baseline_row.get("cache_enabled") is True else "off"
                )
                baseline_spec = CacheBenchmarkCase(
                    name=baseline_name,
                    method=str(method),
                    cache_threshold=(
                        float(baseline_row["cache_threshold"])
                        if baseline_row.get("cache_threshold") is not None
                        else None
                    ),
                    flow_shift_video=float(baseline_row["flow_shift_video"]),
                    guidance_scale=float(baseline_row.get("guidance_scale", 1.0)),
                    baseline_case=baseline_name,
                    reference_root=str(baseline_root),
                )
                baseline = _check_case(
                    baseline_root,
                    baseline_spec,
                    baseline_row,
                    prompts,
                    allow_prompt_superset=True,
                )
                external[key] = baseline
        if not _same_float(candidate.spec.flow_shift_video, baseline.spec.flow_shift_video):
            _die(f"{candidate.spec.name}: candidate/baseline flow shifts differ")
        if not _same_float(candidate.spec.guidance_scale, baseline.spec.guidance_scale):
            _die(f"{candidate.spec.name}: candidate/baseline guidance scales differ")
        if not set(candidate.requests).issubset(baseline.requests):
            missing = sorted(set(candidate.requests) - set(baseline.requests))
            _die(f"{candidate.spec.name}: baseline lacks candidate pairs: {missing}")
        baselines[candidate.spec.name] = baseline
    return checked, baselines


def _open_rgb_reader(path: Path) -> tuple[Iterator[bytes], Mapping[str, Any]]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    except BaseException:
        reader.close()
        raise
    if not isinstance(metadata, dict):
        reader.close()
        _die(f"Decoder returned invalid metadata for {path}: {type(metadata).__name__}")
    context = str(path)
    try:
        width, height = (int(value) for value in metadata["source_size"])
        fps = float(metadata["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        reader.close()
        raise ValueError(f"Decoder metadata lacks valid source_size/fps for {path}: {metadata}") from exc
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        reader.close()
        _die(f"{context}: decoded size is {width}x{height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
    if not math.isfinite(fps) or not math.isclose(fps, EXPECTED_FPS, rel_tol=0.0, abs_tol=FPS_ABS_TOLERANCE):
        reader.close()
        _die(f"{context}: decoded FPS is {fps!r}, expected {EXPECTED_FPS:g}")
    return reader, metadata


def _next_frame(reader: Iterator[bytes], path: Path, frame_index: int) -> bytes:
    try:
        frame = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{path}: decoder stopped at frame {frame_index}; expected {EXPECTED_FRAMES} frames") from exc
    if not isinstance(frame, bytes):
        _die(f"{path}: frame {frame_index} has unexpected type {type(frame).__name__}")
    expected_bytes = EXPECTED_WIDTH * EXPECTED_HEIGHT * 3
    if len(frame) != expected_bytes:
        _die(f"{path}: RGB frame {frame_index} has {len(frame)} bytes, expected {expected_bytes}")
    return frame


def _require_eof(reader: Iterator[bytes], path: Path) -> None:
    try:
        next(reader)
    except StopIteration:
        return
    _die(f"{path}: decoder produced more than {EXPECTED_FRAMES} frames")


def _pixel_metrics(reference_path: Path, candidate_path: Path) -> dict[str, int | float]:
    reference_reader, reference_metadata = _open_rgb_reader(reference_path)
    try:
        candidate_reader, candidate_metadata = _open_rgb_reader(candidate_path)
    except BaseException:
        reference_reader.close()
        raise
    try:
        reference_fps = float(reference_metadata["fps"])
        candidate_fps = float(candidate_metadata["fps"])
        if not math.isclose(reference_fps, candidate_fps, rel_tol=0.0, abs_tol=FPS_ABS_TOLERANCE):
            _die(
                f"Paired videos have different FPS: {reference_path}={reference_fps}, {candidate_path}={candidate_fps}"
            )

        raw_abs_sum = 0
        raw_squared_sum = 0
        raw_reference_sum = 0
        raw_reference_squared_sum = 0
        raw_max_abs = 0
        sample_count = 0
        for frame_index in range(EXPECTED_FRAMES):
            reference_frame = _next_frame(reference_reader, reference_path, frame_index)
            candidate_frame = _next_frame(candidate_reader, candidate_path, frame_index)
            reference = np.frombuffer(reference_frame, dtype=np.uint8)
            candidate = np.frombuffer(candidate_frame, dtype=np.uint8)
            if reference.size != candidate.size:
                _die(f"Frame sample-count mismatch at frame {frame_index}")
            difference = np.subtract(candidate, reference, dtype=np.int16)
            absolute = np.abs(difference)
            raw_abs_sum += int(absolute.sum(dtype=np.int64))
            raw_squared_sum += int(np.square(difference, dtype=np.int64).sum(dtype=np.int64))
            raw_reference_sum += int(reference.sum(dtype=np.uint64))
            raw_reference_squared_sum += int(np.square(reference, dtype=np.uint64).sum(dtype=np.uint64))
            raw_max_abs = max(raw_max_abs, int(absolute.max()))
            sample_count += int(reference.size)

        _require_eof(reference_reader, reference_path)
        _require_eof(candidate_reader, candidate_path)
    finally:
        reference_reader.close()
        candidate_reader.close()

    expected_samples = EXPECTED_WIDTH * EXPECTED_HEIGHT * 3 * EXPECTED_FRAMES
    if sample_count != expected_samples:
        _die(f"Decoded {sample_count} RGB samples, expected {expected_samples}")

    scale = 255.0
    difference_l1 = float(raw_abs_sum) / scale
    difference_l2 = math.sqrt(float(raw_squared_sum)) / scale
    reference_l1 = float(raw_reference_sum) / scale
    reference_l2 = math.sqrt(float(raw_reference_squared_sum)) / scale
    epsilon = sys.float_info.epsilon
    return {
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "frames": EXPECTED_FRAMES,
        "fps": EXPECTED_FPS,
        "rgb_sample_count": sample_count,
        "mae": float(raw_abs_sum) / (scale * sample_count),
        "rmse": math.sqrt(float(raw_squared_sum) / sample_count) / scale,
        "relative_l1": difference_l1 / max(reference_l1, epsilon),
        "relative_l2": difference_l2 / max(reference_l2, epsilon),
        "max_abs": float(raw_max_abs) / scale,
    }


def _pair_rows(
    checked: Sequence[CheckedCase],
    baselines: Mapping[str, CheckedCase],
    prompts: Sequence[PromptSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in (case for case in checked if case.spec.cache_enabled):
        baseline = baselines[candidate.spec.name]
        for prompt in prompts:
            key = (candidate.spec.flow_shift_video, prompt.prompt_hash, prompt.seed)
            candidate_request = candidate.requests.get(key)
            baseline_request = baseline.requests.get(key)
            if candidate_request is None or baseline_request is None:
                _die(f"{candidate.spec.name}: missing pair {key}")
            if _require_int(candidate_request, "prompt_index", candidate.spec.name) != prompt.prompt_index:
                _die(f"{candidate.spec.name}: candidate prompt index mismatch for {key}")
            if _require_int(baseline_request, "prompt_index", baseline.spec.name) != prompt.prompt_index:
                _die(f"{baseline.spec.name}: baseline prompt index mismatch for {key}")
            candidate_video = candidate.videos[prompt.prompt_index]
            baseline_video = baseline.videos[prompt.prompt_index]
            metrics = _pixel_metrics(baseline_video, candidate_video)
            rows.append(
                {
                    "case": candidate.spec.name,
                    "baseline_case": baseline.spec.name,
                    "baseline_root": str(baseline.root),
                    "cache_method": candidate.spec.method,
                    "cache_threshold": candidate.spec.cache_threshold,
                    "flow_shift_video": candidate.spec.flow_shift_video,
                    "guidance_scale": candidate.spec.guidance_scale,
                    "prompt_index": prompt.prompt_index,
                    "prompt_hash": prompt.prompt_hash,
                    "seed": prompt.seed,
                    "candidate_video": str(candidate_video),
                    "candidate_video_sha256": _sha256(candidate_video),
                    "baseline_video": str(baseline_video),
                    "baseline_video_sha256": _sha256(baseline_video),
                    **metrics,
                }
            )
    return rows


def _aggregate_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_names = list(dict.fromkeys(str(row["case"]) for row in rows))
    aggregates: list[dict[str, Any]] = []
    for case_name in case_names:
        group = [row for row in rows if row["case"] == case_name]
        if not group:
            _die(f"{case_name}: no computed pairs")
        aggregate: dict[str, Any] = {
            "case": case_name,
            "baseline_case": group[0]["baseline_case"],
            "baseline_root": group[0]["baseline_root"],
            "cache_method": group[0]["cache_method"],
            "cache_threshold": group[0]["cache_threshold"],
            "flow_shift_video": group[0]["flow_shift_video"],
            "guidance_scale": group[0]["guidance_scale"],
            "pair_count": len(group),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_max"] = max(values)
        aggregates.append(aggregate)
    return aggregates


def _json_document(kind: str, rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": kind,
        "reference": "explicit baseline_case/reference_root with identical settings, prompt_hash and seed",
        "decoded_pixel_domain": "RGB24 normalized to [0, 1]",
        "expected_video": {
            "width": EXPECTED_WIDTH,
            "height": EXPECTED_HEIGHT,
            "frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
        },
        "relative_l1": "L1(candidate-reference) / max(L1(reference), float64_epsilon)",
        "relative_l2": "L2(candidate-reference) / max(L2(reference), float64_epsilon)",
        "inputs": dict(inputs),
        kind: list(rows),
    }


def _atomic_write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed benchmark root")
    parser.add_argument(
        "--prompts-csv",
        help="Override prompts CSV; default is prompts_csv from <root>/benchmark.env",
    )
    parser.add_argument(
        "--cases-csv",
        help="Override cases CSV; default is cases_csv from <root>/benchmark.env",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: benchmark root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace existing pixel-metric outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        _die(f"Benchmark root is not a directory: {root}")
    metadata = _read_env(root / "benchmark.env")
    expected_metadata = {
        "image_size": "464x848",
        "num_frames": str(EXPECTED_FRAMES),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            _die(f"benchmark.env {key} must be {expected!r}, got {metadata.get(key)!r}")

    prompts_csv = _resolve_input(args.prompts_csv, metadata, "prompts_csv")
    cases_csv = _resolve_input(args.cases_csv, metadata, "cases_csv")
    summary_path = root / "summary.json"
    prompts = _load_prompts(prompts_csv)
    cases = load_cases(cases_csv)
    summary = _load_summary(summary_path)
    expected_videos = int(metadata.get("expected_videos_per_case", 0))
    if expected_videos != len(prompts):
        _die(f"benchmark.env expected_videos_per_case={expected_videos} but prompts CSV has {len(prompts)} rows")
    checked, baselines = _validate_inputs(root, summary, cases, prompts)

    output_dir = (args.output_dir or root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pair_csv": output_dir / "pixel_metrics_pairs.csv",
        "pair_json": output_dir / "pixel_metrics_pairs.json",
        "case_csv": output_dir / "pixel_metrics_cases.csv",
        "case_json": output_dir / "pixel_metrics_cases.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        _die(f"Refusing to overwrite existing outputs without --force: {existing}")

    pairs = _pair_rows(checked, baselines, prompts)
    expected_pair_count = sum(case.spec.cache_enabled for case in checked) * len(prompts)
    if len(pairs) != expected_pair_count:
        _die(f"Expected {expected_pair_count} candidate/baseline pairs, computed {len(pairs)}")
    aggregates = _aggregate_cases(pairs)
    inputs = {
        "benchmark_root": str(root),
        "summary_json": str(summary_path),
        "prompts_csv": str(prompts_csv),
        "prompts_sha256": _sha256(prompts_csv),
        "cases_csv": str(cases_csv),
        "cases_sha256": _sha256(cases_csv),
    }

    _atomic_write_csv(outputs["pair_csv"], PAIR_FIELDS, pairs)
    _atomic_write_json(outputs["pair_json"], _json_document("pairs", pairs, inputs))
    _atomic_write_csv(outputs["case_csv"], CASE_FIELDS, aggregates)
    _atomic_write_json(outputs["case_json"], _json_document("cases", aggregates, inputs))
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
