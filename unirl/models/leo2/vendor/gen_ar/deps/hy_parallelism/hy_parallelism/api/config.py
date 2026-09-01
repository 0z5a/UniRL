from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """One replica = ``world_size`` ranks; ``ep > 1`` implies fused multi-request HTTP (see :meth:`use_request_batching`)."""

    world_size: int
    dp_replicate: int = 1
    dp_shard: int = -1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    etp: int = 1

    batch_timeout_s: float = 2.0
    """When not broadcasting, wait at most this long (after the first request) to fill a batch."""

    # request_batching_size: int = 1
    # """Max HTTP ``batching_run`` waiters to merge per flush."""

    num_gpus_per_worker: float = 0.0 # default 0, i.e., without overriding
    num_cpus_per_worker: float = 1.0

    max_ongoing_requests: int = 5

    # def use_request_batching(self) -> bool:
    #     return self.request_batching_size > 1

    def init_parallel_state_kwargs(self) -> dict:
        return {
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "sp": 1,
            "cp": self.cp,
            "etp": self.etp,
            "tp": self.tp,
            "pp": self.pp,
            "ep": self.ep,
        }