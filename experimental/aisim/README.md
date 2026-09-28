# experimental/aisim — offline rollout replay adapter

> **Where it fits:** an optional, default-off offline tool. It observes a real UniRL rollout
> window, converts it into an existing AISimulate / Dynamo Replay input, and compares the
> replay against the matched real execution. It does not schedule, score, train, or feed the
> optimizer, and normal training must run with this package absent.

## Layout

| File | Role |
| --- | --- |
| `schema.py` | Versioned trace contract: manifest, workload, measured outcomes. Stdlib only. |
| `convert.py` | Workload envelope → upstream replay input. Reads no measured outcomes. |
| `runner.py` | Optional single-engine open-loop lowering and CPU replay, pinned to AISimulate `0.12.0.dev1`. |

There is deliberately no `requirements.txt`: the adapter has no dependency of its own, and the
simulator belongs to a separate environment (see the repository rule against adding
non-additive requirements under `experimental/`).

## Boundaries that are load-bearing

- **Three inputs, one direction.** `manifest.json` and `workload.jsonl` are predictor input;
  `measured_requests.jsonl` is report input only. `convert.py` cannot read a measured result
  because no function accepts one, and `assert_no_outcome_leakage` re-checks the converted rows.
- **Observed arrivals are preserved, not reconstructed.** The open-loop contract keeps the
  real arrival offsets (rebased so the first submission is 0) and rejects a window that also
  carries dependency arrivals. It never adds a `session_id` or a synthesised tool wait.
- **A missing wait is not a zero wait.** `external_delay_ms` is required for a dependency
  arrival; `0.0` has to be written explicitly.
- **Unsupported lifecycles are refused, not trimmed.** `failed` / `aborted` / `partial` /
  `retry` terminations raise, and `capture_status="incomplete"` cannot produce a fidelity run.
  Dropping the failures and calling the remainder a complete rollout is the failure mode this
  check exists to prevent.
- **A GRPO group is not a barrier, and a trajectory slot is not an in-flight request.** Neither
  is modelled here; `to_dependency` labels both `not_modelled` in its sidecar rather than
  inventing a representation.

## Status of this package

The contract, converters and optional `runner.run_open_loop` are implemented. Install
`aisimulate==0.12.0.dev1` in the separate replay environment and pass its validated
`EnginePredictionConfig`, the manifest/workload, and a new output directory. Normal
training does not import this optional runner.

The runner requires a complete, frozen-policy capture, one aggregated engine, explicit
KV block size, and disabled prefix caching. It rejects zero-output requests and dependency
replay. Unique synthetic block IDs encode lengths only; they do not claim real prefix
identity. One distinct single-turn session per request preserves the source ID in the
published runtime's `session_id` output, without introducing dependencies between requests.
The runtime's generated UUID and completion order are never used to guess source identity.

The original manifest/workload, lowered trace, compiled replay specification and native
request records are retained. A normalized result is written only after checking exact
ID coverage, arrival/token semantics, completion status and finite ordered timestamps.
Exceptions propagate. Existing output directories are not overwritten. This in-process
entry point does not implement a wall-clock deadline or process supervision.

External tests exercised the real native CPU runner with fixed synthetic timing, including
reordered input and a context-length rejection. These establish parser/runner correctness,
not H100/H20 fidelity or speedup. A matched hardware profile, real UniRL capture, timeout
supervision and the GPU comparison remain pending. Fixed timing must not be reported as
calibrated inference performance.
