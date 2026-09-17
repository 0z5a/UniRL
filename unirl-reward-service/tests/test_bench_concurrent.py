from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image


_SCRIPT = Path(__file__).parents[1] / "scripts" / "bench_concurrent.py"
_SPEC = importlib.util.spec_from_file_location("bench_concurrent", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def _document(mode: str, run: dict, physical_gpu: str = "GPU-0") -> dict:
    complete_run = {
        "label": None,
        "repetition": 1,
        "rewards": ["clip"],
        "concurrency": 2,
        "requests": 2,
        "batch_size": 1,
        "items": 2,
        "wall_clock_s": 0.5,
        "successful_requests": 2,
        "failed_requests": 0,
        "requests_per_second": 4.0,
        "items_per_second": 4.0,
        "transport_errors": {},
        "per_reward_errors": {},
        "latency_ms": {
            "min": 8.0,
            "mean": 10.0,
            "max": 13.0,
            "p50": 10.0,
            "p90": 11.0,
            "p95": 12.0,
            "p99": 13.0,
        },
        "scores": {
            "clip.clip": {
                "count": 2,
                "min": 0.5,
                "mean": 0.5,
                "max": 0.5,
                "values": [0.5, 0.5],
            }
        },
    }
    complete_run.update(run)
    return {
        "schema_version": 1,
        "benchmark": "reward-service-concurrent",
        "deployment": {
            "mode": mode,
            "physical_gpu": physical_gpu,
            "gpu_count": 1,
        },
        "workload": {
            "prompt": "test",
            "rewards": ["clip"],
            "image": {"sha256": "a" * 64},
            "requests_per_run": complete_run["requests"],
            "concurrency_levels": [complete_run["concurrency"]],
            "batch_sizes": [complete_run["batch_size"]],
            "repetitions": 1,
            "warmup_requests_per_reward_set_and_batch": 0,
            "per_reward_isolated": False,
        },
        "runs": [complete_run],
    }


def test_stats_json_contains_latency_errors_and_scores() -> None:
    stats = bench._RunStats(
        concurrency=2,
        total=2,
        wall_s=0.5,
        outcomes=[
            bench._Outcome(
                ok=True,
                latency_s=0.1,
                score_values={"clip.clip": [0.25, 0.75]},
            ),
            bench._Outcome(
                ok=False,
                latency_s=0.2,
                err="TimeoutError: late",
                per_reward_errs=Counter({"clip": 1}),
            ),
        ],
    )

    result = bench._stats_to_dict(
        stats, rewards=["clip"], batch_size=2, label=None
    )

    assert result["successful_requests"] == 1
    assert result["failed_requests"] == 1
    assert result["items_per_second"] == 4.0
    assert result["latency_ms"]["p99"] == 100.0
    assert result["transport_errors"] == {"TimeoutError": 1}
    assert result["scores"]["clip.clip"]["mean"] == 0.5


def test_gpu_monitor_summary_uses_peak_and_mean() -> None:
    monitor = bench._GpuMonitor(1.0)
    monitor.samples = [
        {
            "timestamp": "a",
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-0",
                    "name": "H20",
                    "memory_total_mib": 97871.0,
                    "memory_used_mib": 100.0,
                    "gpu_utilization_percent": 20.0,
                    "memory_utilization_percent": 10.0,
                }
            ],
        },
        {
            "timestamp": "b",
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-0",
                    "name": "H20",
                    "memory_total_mib": 97871.0,
                    "memory_used_mib": 300.0,
                    "gpu_utilization_percent": 80.0,
                    "memory_utilization_percent": 30.0,
                }
            ],
        },
    ]

    device = monitor.summary()["devices"][0]
    assert device["peak_memory_used_mib"] == 300.0
    assert device["steady_memory_used_mib"] == 300.0
    assert device["mean_gpu_utilization_percent"] == 50.0
    assert device["mean_memory_utilization_percent"] == 20.0


def test_comparison_reports_score_delta_against_dedicated(
    tmp_path: Path,
) -> None:
    run = {
        "scores": {
            "clip.clip": {
                "count": 2,
                "min": 0.5,
                "mean": 0.5,
                "max": 0.5,
                "values": [0.5, 0.5],
            }
        },
    }
    candidate = json.loads(json.dumps(run))
    candidate["scores"]["clip.clip"].update(
        {"mean": 0.5002, "max": 0.5004, "values": [0.5, 0.5004]}
    )
    dedicated_path = tmp_path / "dedicated.json"
    no_mps_path = tmp_path / "no-mps.json"
    mps_path = tmp_path / "mps.json"
    consolidated_path = tmp_path / "consolidated.json"
    output_path = tmp_path / "comparison.json"
    dedicated_path.write_text(
        json.dumps(_document("dedicated", run)), encoding="utf-8"
    )
    no_mps_path.write_text(
        json.dumps(_document("colocated-no-mps", candidate)), encoding="utf-8"
    )
    mps_path.write_text(
        json.dumps(_document("colocated-mps", candidate)), encoding="utf-8"
    )
    consolidated_path.write_text(
        json.dumps(_document("consolidated", candidate)), encoding="utf-8"
    )

    assert (
        bench._compare_result_files(
            [dedicated_path, no_mps_path, mps_path, consolidated_path],
            output_path,
            0.001,
        )
        == 0
    )
    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert comparison["same_physical_gpu_verified_for_shared_modes"] is True
    mps = next(
        row
        for row in comparison["aggregates"]
        if row["deployment_mode"] == "colocated-mps"
    )
    assert mps["parity_passed"] is True
    assert mps["max_absolute_score_delta"] == pytest.approx(0.0004)


def test_json_writer_is_deterministic_and_omits_unavailable_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    bench._write_json(path, {"z": None, "b": 2, "a": {"missing": None, "x": 1}})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "x": 1\n  },\n  "b": 2\n}\n'
    )


def test_short_server_response_is_protocol_failure() -> None:
    class ShortClient:
        def score(self, requests):
            return [{"clip": {"clip": 0.5}}]

    bench._thread_local.client = ShortClient()
    try:
        outcome = bench._fire_once(
            "unused",
            None,
            False,
            "test",
            Image.new("RGB", (2, 2)),
            ["clip"],
            2,
        )
    finally:
        del bench._thread_local.client
    assert not outcome.ok
    assert outcome.err.startswith("ProtocolError:")


def test_non_finite_reward_is_counted_as_request_failure() -> None:
    class NonFiniteClient:
        def score(self, requests):
            return [{"clip": {"clip": float("nan")}}]

    bench._thread_local.client = NonFiniteClient()
    try:
        outcome = bench._fire_once(
            "unused",
            None,
            False,
            "test",
            Image.new("RGB", (2, 2)),
            ["clip"],
            1,
        )
    finally:
        del bench._thread_local.client
    assert not outcome.ok
    assert outcome.per_reward_errs == Counter({"clip": 1})
    assert outcome.err is None


def test_comparison_rejects_incompatible_workloads(tmp_path: Path) -> None:
    dedicated = _document("dedicated", {})
    no_mps = _document("colocated-no-mps", {})
    mps = _document("colocated-mps", {})
    consolidated = _document("consolidated", {})
    mps["workload"]["prompt"] = "different"
    dedicated_path = tmp_path / "dedicated.json"
    no_mps_path = tmp_path / "no-mps.json"
    mps_path = tmp_path / "mps.json"
    consolidated_path = tmp_path / "consolidated.json"
    dedicated_path.write_text(json.dumps(dedicated), encoding="utf-8")
    no_mps_path.write_text(json.dumps(no_mps), encoding="utf-8")
    mps_path.write_text(json.dumps(mps), encoding="utf-8")
    consolidated_path.write_text(json.dumps(consolidated), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible workloads"):
        bench._compare_result_files(
            [dedicated_path, no_mps_path, mps_path, consolidated_path],
            None,
            1e-5,
        )


def test_comparison_rejects_duplicate_dedicated_baseline(tmp_path: Path) -> None:
    document = _document("dedicated", {})
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(document), encoding="utf-8")
    second.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one document per mode"):
        bench._compare_result_files([first, second], None, 1e-5)


def test_comparison_fails_when_any_request_failed(tmp_path: Path) -> None:
    paths = []
    for mode in bench._DEPLOYMENT_MODES:
        run = (
            {"successful_requests": 1, "failed_requests": 1}
            if mode == "colocated-mps"
            else {}
        )
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(_document(mode, run)), encoding="utf-8")
        paths.append(path)

    assert bench._compare_result_files(paths, None, 1e-5) == 3


def test_score_parity_fails_on_missing_metric() -> None:
    baseline = _document("dedicated", {})["runs"][0]
    candidate = json.loads(json.dumps(baseline))
    candidate["scores"] = {}

    parity = bench._score_parity(baseline, candidate, 1e-5)
    assert parity["status"] == "fail"
    assert parity["missing_metrics"] == ["clip.clip"]


def test_result_validation_rejects_malformed_run(tmp_path: Path) -> None:
    document = _document("dedicated", {})
    document["runs"][0]["wall_clock_s"] = float("nan")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="wall_clock_s"):
        bench._load_result(path)


def test_gpu_monitor_stop_waits_for_sampler(monkeypatch) -> None:
    def slow_query():
        time.sleep(0.03)
        return []

    monkeypatch.setattr(bench, "_query_gpus", slow_query)
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monitor = bench._GpuMonitor(0.01)
    monitor.start()
    monitor.stop()
    assert monitor._thread is not None
    assert not monitor._thread.is_alive()


def test_mps_log_observations_are_explicit_mentions(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(
        "set_default_active_thread_percentage 100\n"
        "MPS FAULT\nout of memory\nrestart client\n",
        encoding="utf-8",
    )

    observations = bench._collect_mps_log_observations(tmp_path)

    assert observations["available"] is True
    assert observations["fault_mentions"] == 1
    assert observations["oom_mentions"] == 1
    assert observations["restart_mentions"] == 1
