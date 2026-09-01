"""
Ray Serve + FastAPI print demo: ``/generate`` and ``/health``.

Dependencies::

    pip install "ray[serve]" fastapi uvicorn pydantic

Usage::

    python -m hy_parallelism.api.example.example_run --tp 4 --world-size 8

With ``ep > 1``, HTTP fuses multiple requests (EP-style). Example::

    python -m hy_parallelism.api.example.example_run --ep 4 --tp 1 --dp-shard 4 --world-size 4

``POST /generate`` is only ``{"payload": {...}}``; same payload to all ranks when ``ep == 1``, batching when ``ep > 1``.

From another shell, hit the same bind address with the stdlib client::

    python -m hy_parallelism.api.example.example_client --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import ray

from pydantic import BaseModel, Field

from hy_parallelism.api.config import RuntimeConfig
from hy_parallelism.api.core import Server
from hy_parallelism.api.ray_worker import DistributedWorkerBase
from hy_parallelism.api.core import BaseIngress
from hy_parallelism.api.core import build_app

logger = logging.getLogger(__name__)


class GenerateBody(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)





def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num-replicas", type=int, default=1)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--dp-replicate", type=int, default=1)
    p.add_argument("--dp-shard", type=int, default=-1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--etp", type=int, default=1)
    p.add_argument("--batch-timeout-s", type=float, default=2.0)
    p.add_argument("--request-batching-size", type=int, default=1)
    p.add_argument("--gpus-per-worker", type=float, default=0.0)
    p.add_argument("--cpus-per-worker", type=float, default=1.0)
    args = p.parse_args()


    app = build_app()



    @ray.remote
    class DemoDistributedWorker(DistributedWorkerBase):

        def __init__(self, custom_attr, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_attr = custom_attr

        def handle_same(self, req: dict[str, Any]) -> dict[str, Any]:
            print(f"[rank {self.rank}] TP same request: {req} {self.custom_attr}", flush=True)
            return {"rank": self.rank, "echo": req}

        def handle_rank(self, req: dict[str, Any]) -> dict[str, Any]:
            print(f"[rank {self.rank}] EP slot: {req} {self.custom_attr}", flush=True)
            return {"rank": self.rank, "echo": req}

    class ExampleIngress(BaseIngress):

        @app.post("/generate")
        async def generate(self, body: GenerateBody) -> dict[str, Any]:
            if not self.request_batching:
                ranks = await self.runtime.broadcast_run("handle_same", kwargs=body.payload)
                return {"request_batching": False, "request": body.payload, "ranks": ranks}
            rep = await self.runtime.batching_run("handle_rank", kwargs=body.payload)
            return {"request_batching": True, **rep}

        @app.get("/health")
        async def health(self) -> dict[str, str]:
            return {"status": "ok"}


    cfg = RuntimeConfig(
        world_size=args.world_size,
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=args.cp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        batch_timeout_s=args.batch_timeout_s,
        request_batching_size=args.request_batching_size,
        num_gpus_per_worker=args.gpus_per_worker,
        num_cpus_per_worker=args.cpus_per_worker,
    )
    server = Server(
        app=app,
        ingress_cls=ExampleIngress,
        worker_cls=DemoDistributedWorker,
        worker_kwargs={"custom_attr": "example_custom_attr"},
        num_replicas=args.num_replicas,
        runtime_config=cfg,
        host=args.host,
        port=args.port,
    )
    server.run()


if __name__ == "__main__":
    main()
