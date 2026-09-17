"""Compact result validation and four-deployment comparison."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEPLOYMENT_MODES = (
    "dedicated",
    "colocated-no-mps",
    "colocated-mps",
    "consolidated",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _omit_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _omit_none(child)
            for key, child in value.items()
            if child is not None
        }
    if isinstance(value, list):
        return [_omit_none(child) for child in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _omit_none(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(run.get("rewards", [])),
        run.get("concurrency"),
        run.get("requests"),
        run.get("batch_size"),
        run.get("label"),
        run.get("repetition", 1),
    )


def _expected_run_keys(workload: dict[str, Any]) -> set[tuple[Any, ...]]:
    if workload["per_reward_isolated"]:
        reward_sets = [
            ([name], f"isolated:{name}") for name in workload["rewards"]
        ]
    else:
        reward_sets = [(workload["rewards"], None)]
    return {
        (
            tuple(rewards),
            concurrency,
            workload["requests_per_run"],
            batch_size,
            label,
            repetition,
        )
        for rewards, label in reward_sets
        for concurrency in workload["concurrency_levels"]
        for batch_size in workload["batch_sizes"]
        for repetition in range(1, workload["repetitions"] + 1)
    }


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: result root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    if payload.get("benchmark") != "reward-service-concurrent":
        raise ValueError(f"{path}: unexpected benchmark type")

    deployment = payload.get("deployment")
    workload = payload.get("workload")
    runs = payload.get("runs")
    if not isinstance(deployment, dict):
        raise ValueError(f"{path}: missing deployment")
    if deployment.get("mode") not in _DEPLOYMENT_MODES:
        raise ValueError(f"{path}: invalid deployment mode")
    if not deployment.get("physical_gpu"):
        raise ValueError(f"{path}: missing physical GPU identity")
    if not isinstance(workload, dict) or not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: missing workload or runs")

    required_workload = (
        "prompt",
        "rewards",
        "image",
        "requests_per_run",
        "concurrency_levels",
        "batch_sizes",
        "repetitions",
        "per_reward_isolated",
    )
    if any(key not in workload for key in required_workload):
        raise ValueError(f"{path}: incomplete workload identity")

    seen: set[tuple[Any, ...]] = set()
    for index, run in enumerate(runs):
        context = f"{path}: runs[{index}]"
        if not isinstance(run, dict):
            raise ValueError(f"{context} must be an object")
        for key in (
            "wall_clock_s",
            "requests_per_second",
            "items_per_second",
        ):
            value = run.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{context}.{key} must be finite and non-negative")
        requests = run.get("requests")
        succeeded = run.get("successful_requests")
        failed = run.get("failed_requests")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (requests, succeeded, failed)
        ) or succeeded + failed != requests:
            raise ValueError(f"{context}: invalid request counts")

        scores = run.get("scores")
        if not isinstance(scores, dict):
            raise ValueError(f"{context}.scores must be an object")
        for metric, score in scores.items():
            values = score.get("values") if isinstance(score, dict) else None
            if (
                not isinstance(metric, str)
                or not isinstance(values, list)
                or score.get("count") != len(values)
                or values != sorted(values)
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"{context}.scores.{metric} is invalid")

        key = _run_key(run)
        if key in seen:
            raise ValueError(f"{path}: duplicate run key")
        seen.add(key)

    if seen != _expected_run_keys(workload):
        raise ValueError(f"{path}: incomplete run matrix")
    return payload


def _workload_identity(document: dict[str, Any]) -> str:
    workload = document["workload"]
    comparable = {
        "prompt": workload["prompt"],
        "rewards": workload["rewards"],
        "image_sha256": workload["image"]["sha256"],
        "requests_per_run": workload["requests_per_run"],
        "concurrency_levels": workload["concurrency_levels"],
        "batch_sizes": workload["batch_sizes"],
        "repetitions": workload["repetitions"],
        "per_reward_isolated": workload["per_reward_isolated"],
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def _score_parity(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    absolute_tolerance: float,
) -> dict[str, Any]:
    baseline_scores = baseline.get("scores", {})
    candidate_scores = candidate.get("scores", {})
    baseline_metrics = set(baseline_scores)
    candidate_metrics = set(candidate_scores)
    if not baseline_metrics:
        return {"status": "fail", "reason": "dedicated baseline has no scores"}
    if baseline_metrics != candidate_metrics:
        return {
            "status": "fail",
            "reason": "score metric sets differ",
            "missing_metrics": sorted(baseline_metrics - candidate_metrics),
            "unexpected_metrics": sorted(candidate_metrics - baseline_metrics),
        }

    max_delta = 0.0
    for metric in baseline_metrics:
        expected = baseline_scores[metric]["values"]
        actual = candidate_scores[metric]["values"]
        if len(expected) != len(actual):
            return {"status": "fail", "reason": f"{metric} score counts differ"}
        max_delta = max(
            max_delta,
            max((abs(left - right) for left, right in zip(expected, actual)), default=0),
        )
    return {
        "status": "pass" if max_delta <= absolute_tolerance else "fail",
        "max_absolute_delta": max_delta,
    }


def _compare_result_files(
    paths: list[Path],
    output: Path | None,
    score_absolute_tolerance: float,
) -> int:
    documents = [_load_result(path) for path in paths]
    by_mode = {
        document["deployment"]["mode"]: (path, document)
        for path, document in zip(paths, documents)
    }
    if len(by_mode) != len(paths) or set(by_mode) != set(_DEPLOYMENT_MODES):
        raise ValueError("comparison requires exactly one document per mode")
    if len({_workload_identity(document) for document in documents}) != 1:
        raise ValueError("result files describe incompatible workloads")

    shared_gpu_ids = {
        by_mode[mode][1]["deployment"]["physical_gpu"]
        for mode in ("colocated-no-mps", "colocated-mps")
    }
    same_gpu = len(shared_gpu_ids) == 1
    baselines = {
        _run_key(run): run for run in by_mode["dedicated"][1]["runs"]
    }

    groups: dict[tuple[Any, ...], list[tuple[dict, dict]]] = {}
    parity_failed = False
    request_failed = False
    for mode, (_, document) in by_mode.items():
        for run in document["runs"]:
            request_failed |= run["failed_requests"] > 0
            parity = _score_parity(
                baselines[_run_key(run)],
                run,
                score_absolute_tolerance,
            )
            parity_failed |= parity["status"] != "pass"
            key = (
                mode,
                tuple(run["rewards"]),
                run["concurrency"],
                run["batch_size"],
                run.get("label"),
            )
            groups.setdefault(key, []).append((run, parity))

    aggregates = []
    for key, entries in sorted(groups.items(), key=lambda item: repr(item[0])):
        runs = [entry[0] for entry in entries]
        parities = [entry[1] for entry in entries]
        throughput = [run["items_per_second"] for run in runs]
        latency = [
            run.get("latency_ms", {}).get("p95")
            for run in runs
            if run.get("latency_ms")
        ]
        aggregates.append(
            {
                "deployment_mode": key[0],
                "rewards": list(key[1]),
                "concurrency": key[2],
                "batch_size": key[3],
                "label": key[4],
                "repetitions": len(runs),
                "items_per_second_mean": statistics.fmean(throughput),
                "items_per_second_stdev": (
                    statistics.stdev(throughput) if len(throughput) > 1 else None
                ),
                "latency_p95_ms_mean": (
                    statistics.fmean(latency) if latency else None
                ),
                "failed_requests": sum(run["failed_requests"] for run in runs),
                "parity_passed": all(
                    parity["status"] == "pass" for parity in parities
                ),
                "max_absolute_score_delta": max(
                    parity.get("max_absolute_delta", 0.0) for parity in parities
                ),
            }
        )

    sources = [
        {
            "source": path.name,
            "deployment": document["deployment"],
            "environment": document.get("environment"),
            "workload": document["workload"],
            "gpu_observations": document.get("gpu_observations"),
            "mps_log_observations": document.get("mps_log_observations"),
            "successful_requests": sum(
                run["successful_requests"] for run in document["runs"]
            ),
            "failed_requests": sum(
                run["failed_requests"] for run in document["runs"]
            ),
        }
        for path, document in zip(paths, documents)
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "reward-service-deployment-comparison",
        "created_at": _utc_now(),
        "same_physical_gpu_verified_for_shared_modes": same_gpu,
        "score_absolute_tolerance": score_absolute_tolerance,
        "sources": sources,
        "aggregates": aggregates,
    }
    if output is not None:
        _write_json(output, payload)
        print(f"wrote comparison JSON: {output}")
    return 3 if request_failed or parity_failed or not same_gpu else 0
