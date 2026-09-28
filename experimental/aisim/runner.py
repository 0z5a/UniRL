"""Single-engine open-loop replay against the published AISimulate 0.12.0.dev1 API."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, TypedDict

from experimental.aisim.convert import to_open_loop
from experimental.aisim.schema import Manifest, WorkloadRequest

if TYPE_CHECKING:
    from aisimulate.config.engine import EnginePredictionConfig

SIMULATOR_VERSION = "0.12.0.dev1"


class MooncakeRequest(TypedDict):
    request_id: str
    session_id: str
    input_tokens: int
    output_tokens: int
    timestamp: float
    hash_ids: list[int]


@dataclass(frozen=True)
class PredictedRequest:
    request_id: str
    submit_ms: float
    first_token_ms: float
    complete_ms: float


def lower_open_loop(manifest: Manifest, requests: Sequence[WorkloadRequest], block_size: int) -> list[MooncakeRequest]:
    converted = to_open_loop(manifest, requests)
    if manifest.capture_status != "complete":
        raise ValueError("replay requires a complete capture")
    if manifest.engine_count != 1 or len({request.engine_id for request in requests}) != 1:
        raise ValueError("replay supports exactly one engine")
    if {request.policy_identity for request in requests} != {manifest.policy_identity}:
        raise ValueError("replay requires the manifest's frozen policy")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("block_size must be positive")
    rows: list[MooncakeRequest] = []
    next_block = 0
    for request in sorted(converted.requests, key=lambda row: row["arrival_ms"]):
        if request["output_tokens"] == 0:
            raise ValueError("zero-output requests are not supported by this replay adapter")
        blocks = (request["input_tokens"] + block_size - 1) // block_size
        rows.append(
            MooncakeRequest(
                request_id=request["request_id"],
                session_id=request["request_id"],
                input_tokens=request["input_tokens"],
                output_tokens=request["output_tokens"],
                timestamp=request["arrival_ms"],
                hash_ids=list(range(next_block, next_block + blocks)),
            )
        )
        next_block += blocks
    return rows


def read_predictions(path: Path, expected: Sequence[MooncakeRequest]) -> list[PredictedRequest]:
    by_id = {row["request_id"]: row for row in expected}
    results: dict[str, PredictedRequest] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        request_id = row["session_id"]
        if request_id not in by_id or request_id in results:
            raise ValueError(f"unknown or repeated simulator request: {request_id}")
        source = by_id[request_id]
        if row["turn_index"] != 0 or row["terminal_status"] != "completed":
            raise ValueError(f"request did not complete as one independent turn: {request_id}")
        if (row["input_length"], row["output_length"], row["arrival_time_ms"]) != (
            source["input_tokens"],
            source["output_tokens"],
            source["timestamp"],
        ):
            raise ValueError(f"simulator changed request semantics: {request_id}")
        times = (row["arrival_time_ms"], row["first_token_ms"], row["terminal_time_ms"])
        if (
            not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in times
            )
            or not 0 <= times[0] <= times[1] <= times[2]
        ):
            raise ValueError(f"invalid simulator timestamps: {request_id}")
        results[request_id] = PredictedRequest(request_id, *times)
    if results.keys() != by_id.keys():
        raise ValueError("simulator output is missing requests")
    return [results[row["request_id"]] for row in expected]


def run_open_loop(
    manifest: Manifest,
    requests: Sequence[WorkloadRequest],
    engine: EnginePredictionConfig,
    output_dir: Path,
) -> list[PredictedRequest]:
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.output import write_prediction_report, write_requests
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import ReplayOutputRequirements

    if version("aisimulate") != SIMULATOR_VERSION:
        raise ValueError(f"this adapter requires aisimulate=={SIMULATOR_VERSION}")
    if manifest.simulator_commit_or_version != f"aisimulate=={SIMULATOR_VERSION}":
        raise ValueError("manifest simulator version does not match the runner")
    worker = engine.workers.aggregated
    if engine.mode != "aggregated" or worker is None or worker.parallelism.replicas != 1:
        raise ValueError("replay supports one aggregated replica")
    if engine.backend != manifest.backend_name or worker.parallelism.tensor != manifest.tp_size:
        raise ValueError("engine backend or tensor parallelism differs from the manifest")
    if worker.kv_cache.prefix_caching or worker.kv_cache.block_size is None:
        raise ValueError("set an explicit KV block size and disable prefix caching")
    rows = lower_open_loop(manifest, requests, worker.kv_cache.block_size)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=1) + "\n")
    (output_dir / "workload.jsonl").write_text("".join(json.dumps(asdict(request)) + "\n" for request in requests))
    trace = output_dir / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = CorePredictionConfig.model_validate(
        {
            "engine": engine.model_dump(mode="json", exclude_none=True),
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": [str(trace.resolve())],
                    "format": "mooncake",
                    "block_size": worker.kv_cache.block_size,
                },
                "load": {"type": "trace_timestamps", "speedup": 1.0},
            },
        }
    )
    spec = prediction_to_replay_spec(config)
    (output_dir / "replay-spec.json").write_text(json.dumps(asdict(spec), indent=1) + "\n")
    runner = EngineReplayRunnerFactory().create(0)
    try:
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
            ),
        )
    finally:
        runner.close()
    native = report.metadata["native_report"]
    if not isinstance(native, dict) or not isinstance(native["per_request"], list):
        raise ValueError("simulator did not return request records")
    write_prediction_report(output_dir, native)
    write_requests(output_dir, native["per_request"])
    predicted = read_predictions(output_dir / "requests.jsonl", rows)
    (output_dir / "simulated_requests.jsonl").write_text("".join(json.dumps(asdict(row)) + "\n" for row in predicted))
    return predicted
