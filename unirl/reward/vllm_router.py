"""OpenAI-compatible, live-load-aware router for replicated vLLM servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger("unirl.vllm_router")


@dataclass(frozen=True)
class BackendLoad:
    url: str
    running: float
    waiting: float
    kv_cache_usage: float
    healthy: bool = True
    error: str | None = None

    @property
    def score(self) -> float:
        return self.running + 4.0 * self.waiting + self.kv_cache_usage


def _metric_sum(lines: Iterable[str], name: str) -> float:
    values: list[float] = []
    prefix_plain = f"{name} "
    prefix_labeled = f"{name}{{"
    for line in lines:
        if not (
            line.startswith(prefix_plain) or line.startswith(prefix_labeled)
        ):
            continue
        raw_value = line.rsplit(" ", 1)[-1]
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"expected numeric Prometheus value for {name!r}, got "
                f"{raw_value!r} in line {line!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"expected finite Prometheus value for {name!r}, got {value!r}"
            )
        values.append(value)
    if not values:
        raise ValueError(f"required vLLM metric {name!r} was not present")
    return sum(values)


def parse_vllm_load(url: str, metrics_text: str) -> BackendLoad:
    if not isinstance(metrics_text, str):
        raise TypeError(
            f"expected str metrics_text for backend {url!r}, got "
            f"{type(metrics_text).__name__}: {metrics_text!r}"
        )
    lines = metrics_text.splitlines()
    return BackendLoad(
        url=url,
        running=_metric_sum(lines, "vllm:num_requests_running"),
        waiting=_metric_sum(lines, "vllm:num_requests_waiting"),
        kv_cache_usage=_metric_sum(lines, "vllm:kv_cache_usage_perc"),
    )


def select_backend_index(
    snapshots: Sequence[BackendLoad],
    local_active: Sequence[int],
    tie_cursor: int,
) -> int:
    if not snapshots:
        raise ValueError("expected at least one vLLM backend snapshot")
    if len(snapshots) != len(local_active):
        raise ValueError(
            "expected one local_active count per backend, got "
            f"snapshots={len(snapshots)} local_active={len(local_active)}"
        )
    candidates = [
        index for index, snapshot in enumerate(snapshots) if snapshot.healthy
    ]
    if not candidates:
        candidates = list(range(len(snapshots)))
    scores = {
        index: snapshots[index].score + float(local_active[index])
        for index in candidates
    }
    minimum = min(scores.values())
    tied = [
        index
        for index in candidates
        if math.isclose(scores[index], minimum, rel_tol=0.0, abs_tol=1e-9)
    ]
    return tied[tie_cursor % len(tied)]


class LiveVLLMBackendPool:
    def __init__(
        self,
        backends: Sequence[str],
        *,
        metrics_timeout: float = 2.0,
    ) -> None:
        normalized = tuple(
            dict.fromkeys(url.strip().rstrip("/") for url in backends if url.strip())
        )
        if not normalized:
            raise ValueError(
                f"expected at least one non-empty vLLM backend, got {backends!r}"
            )
        if any(not url.startswith(("http://", "https://")) for url in normalized):
            raise ValueError(
                f"expected HTTP(S) vLLM backend URLs, got {normalized!r}"
            )
        if not math.isfinite(metrics_timeout) or metrics_timeout <= 0:
            raise ValueError(
                f"expected metrics_timeout > 0, got {metrics_timeout!r}"
            )
        self.backends = normalized
        self.metrics_timeout = float(metrics_timeout)
        self.local_active = [0] * len(normalized)
        self.tie_cursor = 0
        self._lock = asyncio.Lock()
        self._client: Any = None
        self.last_snapshots: list[BackendLoad] = [
            BackendLoad(
                url=url,
                running=0.0,
                waiting=0.0,
                kv_cache_usage=0.0,
                healthy=False,
                error="not yet scraped",
            )
            for url in normalized
        ]

    async def start(self) -> None:
        import httpx

        if self._client is not None:
            raise RuntimeError("LiveVLLMBackendPool.start called more than once")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800.0, connect=30.0),
            trust_env=False,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "expected initialized vLLM router HTTP client; call start() first"
            )
        return self._client

    async def _scrape_one(self, url: str) -> BackendLoad:
        import httpx

        client = self._require_client()
        try:
            response = await client.get(
                f"{url}/metrics",
                timeout=self.metrics_timeout,
            )
            response.raise_for_status()
            return parse_vllm_load(url, response.text)
        except (httpx.HTTPError, ValueError) as exc:
            return BackendLoad(
                url=url,
                running=0.0,
                waiting=0.0,
                kv_cache_usage=0.0,
                healthy=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def snapshots(self) -> list[BackendLoad]:
        snapshots = list(
            await asyncio.gather(
                *(self._scrape_one(url) for url in self.backends)
            )
        )
        self.last_snapshots = snapshots
        return snapshots

    async def acquire(self) -> tuple[int, str, BackendLoad]:
        snapshots = await self.snapshots()
        async with self._lock:
            index = select_backend_index(
                snapshots,
                self.local_active,
                self.tie_cursor,
            )
            self.tie_cursor += 1
            self.local_active[index] += 1
        return index, self.backends[index], snapshots[index]

    async def release(self, index: int) -> None:
        async with self._lock:
            if index < 0 or index >= len(self.local_active):
                raise IndexError(
                    f"backend index {index} outside [0, {len(self.local_active)})"
                )
            if self.local_active[index] <= 0:
                raise RuntimeError(
                    f"backend {self.backends[index]!r} has no active reservation"
                )
            self.local_active[index] -= 1

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backends": [asdict(snapshot) for snapshot in self.last_snapshots],
            "router_active": list(self.local_active),
        }


def create_app(pool: LiveVLLMBackendPool) -> Any:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.on_event("startup")
    async def _startup() -> None:
        await pool.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await pool.close()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        await pool.snapshots()
        return pool.health_payload()

    @app.get("/v1/models")
    async def models() -> Response:
        index, backend, _snapshot = await pool.acquire()
        try:
            response = await pool._require_client().get(f"{backend}/v1/models")
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers={"x-vllm-backend": backend},
            )
        finally:
            await pool.release(index)

    async def _stream_and_release(response: Any, index: int):
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await pool.release(index)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.body()
        try:
            document = await request.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"expected JSON chat completion body, got {body[:200]!r}"
            ) from exc
        if not isinstance(document, dict):
            raise TypeError(
                "expected JSON object chat completion body, got "
                f"{type(document).__name__}: {document!r}"
            )
        stream = bool(document.get("stream"))
        index, backend, snapshot = await pool.acquire()
        headers = {
            "content-type": request.headers.get(
                "content-type", "application/json"
            )
        }
        authorization = request.headers.get("authorization")
        if authorization:
            headers["authorization"] = authorization
        request_headers = {
            "x-vllm-backend": backend,
            "x-vllm-running": str(snapshot.running),
            "x-vllm-waiting": str(snapshot.waiting),
        }
        client = pool._require_client()
        if stream:
            try:
                upstream_request = client.build_request(
                    "POST",
                    f"{backend}/v1/chat/completions",
                    content=body,
                    headers=headers,
                )
                response = await client.send(upstream_request, stream=True)
            except BaseException:
                await pool.release(index)
                raise
            return StreamingResponse(
                _stream_and_release(response, index),
                status_code=response.status_code,
                media_type=response.headers.get(
                    "content-type", "text/event-stream"
                ),
                headers=request_headers,
            )
        try:
            response = await client.post(
                f"{backend}/v1/chat/completions",
                content=body,
                headers=headers,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers=request_headers,
            )
        finally:
            await pool.release(index)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        required=True,
        help="comma-separated vLLM server roots, e.g. http://host1:8000,http://host2:8000",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--metrics-timeout", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = _parse_args()
    pool = LiveVLLMBackendPool(
        args.backends.split(","),
        metrics_timeout=args.metrics_timeout,
    )
    uvicorn.run(
        create_app(pool),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
