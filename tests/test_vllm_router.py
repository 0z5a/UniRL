from __future__ import annotations

import asyncio

from unirl.reward.vllm_router import (
    BackendLoad,
    LiveVLLMBackendPool,
    parse_vllm_load,
    select_backend_index,
)


def test_parse_vllm_load_metrics() -> None:
    snapshot = parse_vllm_load(
        "http://judge-a:8000",
        "\n".join(
            (
                'vllm:num_requests_running{engine="0"} 12',
                'vllm:num_requests_waiting{engine="0"} 3',
                'vllm:kv_cache_usage_perc{engine="0"} 0.25',
            )
        ),
    )
    assert snapshot == BackendLoad(
        url="http://judge-a:8000",
        running=12.0,
        waiting=3.0,
        kv_cache_usage=0.25,
    )


def test_select_backend_uses_live_load_and_rotating_ties() -> None:
    snapshots = [
        BackendLoad("http://a", running=12, waiting=0, kv_cache_usage=0.2),
        BackendLoad("http://b", running=0, waiting=0, kv_cache_usage=0.0),
    ]
    assert select_backend_index(snapshots, [0, 0], 0) == 1

    tied = [
        BackendLoad("http://a", running=0, waiting=0, kv_cache_usage=0),
        BackendLoad("http://b", running=0, waiting=0, kv_cache_usage=0),
    ]
    assert select_backend_index(tied, [0, 0], 0) == 0
    assert select_backend_index(tied, [0, 0], 1) == 1


def test_pool_reserves_equal_load_backends_without_pin() -> None:
    async def run() -> None:
        pool = LiveVLLMBackendPool(["http://a", "http://b"])
        snapshots = [
            BackendLoad("http://a", running=0, waiting=0, kv_cache_usage=0),
            BackendLoad("http://b", running=0, waiting=0, kv_cache_usage=0),
        ]

        async def fixed_snapshots() -> list[BackendLoad]:
            return snapshots

        pool.snapshots = fixed_snapshots  # type: ignore[method-assign]
        first = await pool.acquire()
        second = await pool.acquire()
        assert (first[0], second[0]) == (0, 1)
        await pool.release(first[0])
        await pool.release(second[0])

    asyncio.run(run())
