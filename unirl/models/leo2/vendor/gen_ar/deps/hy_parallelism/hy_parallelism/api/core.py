from __future__ import annotations

import time
import asyncio
import random
import socket
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI
import loguru
import ray
from ray import serve
from loguru import logger

from hy_parallelism.api.config import RuntimeConfig

try:
    from ray.serve.context import get_replica_context
except Exception:  # pragma: no cover
    get_replica_context = None  # type: ignore[misc, assignment]


@dataclass
class SingleRPCTask:
    method: str
    kwargs: dict[str, Any]
    fut: asyncio.Future[Any]


@dataclass(frozen=True)
class BatchGroupKey:
    batch_key: str
    method: str


@dataclass
class PendingBatchTask:
    method: str
    kwargs: dict[str, Any]
    futs: list[asyncio.Future[Any]]

    @classmethod
    def from_single_rpc_tasks(cls, tasks: list[SingleRPCTask], kwargs_collate_fn: Callable) -> PendingBatchTask:
        assert kwargs_collate_fn is not None
        assert len(tasks) > 0
        assert all(t.method == tasks[0].method for t in tasks), "all tasks must have the same method"
        return cls(
            method=tasks[0].method,
            kwargs=kwargs_collate_fn([task.kwargs for task in tasks]),
            futs=[task.fut for task in tasks],
        )

    @classmethod
    def from_pending_batch_tasks(cls, tasks: list[PendingBatchTask], kwargs_collate_fn: Callable) -> PendingBatchTask:
        assert kwargs_collate_fn is not None
        assert len(tasks) > 0
        assert all(t.method == tasks[0].method for t in tasks), "all tasks must have the same method"
        return cls(
            method=tasks[0].method,
            kwargs=kwargs_collate_fn([task.kwargs for task in tasks]),
            futs=sum([task.futs for task in tasks], []),
        )


@dataclass(frozen=True)
class BatchingOptions:
    kwargs_collate_fn: Callable | None
    output_split_fn: Callable | None
    static_micro_batch_size: int | None
    dynamic_micro_batch_size: int | None
    batch_key_fn: Callable[[str, dict[str, Any]], str] | None = None

    def is_static(self) -> bool:
        return self.static_micro_batch_size is not None and self.static_micro_batch_size > 1

    def is_dynamic(self) -> bool:
        return self.dynamic_micro_batch_size is not None and self.dynamic_micro_batch_size > 1

    def enable_micro_batching(self) -> bool:
        return self.is_static() or self.is_dynamic()

    def __post_init__(self) -> None:
        if self.static_micro_batch_size:
            assert self.static_micro_batch_size > 1
        if self.dynamic_micro_batch_size:
            assert self.dynamic_micro_batch_size > 1
        if self.is_static() or self.is_dynamic():
            assert self.kwargs_collate_fn is not None
            assert self.output_split_fn is not None

        if self.is_static() and self.is_dynamic():
            raise ValueError("static and dynamic micro batching cannot be enabled at the same time")


    @classmethod
    def default(cls) -> BatchingOptions:
        """No micro-batching; collate/split only used when sizes enable them."""
        return cls(
            kwargs_collate_fn=None,
            output_split_fn=None,
            static_micro_batch_size=None,
            dynamic_micro_batch_size=None,
            batch_key_fn=None,
        )

    def batch_key(self, method: str, kwargs: dict[str, Any]) -> str:
        if self.batch_key_fn is None:
            return method
        return self.batch_key_fn(method, kwargs)


def default_replica_slug() -> str:
    if get_replica_context is None:
        return uuid.uuid4().hex[:12]
    try:
        return get_replica_context().replica_tag.replace("/", "_").replace(":", "_")
    except Exception:
        return uuid.uuid4().hex[:12]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p: int = s.getsockname()[1]
    s.close()
    return p


class ServeReplicaRuntime:

    def __init__(
        self,
        *,
        workers: list[Any],
        cfg: RuntimeConfig,
    ) -> None:
        self._w = workers
        self._cfg = cfg

        # 有非常多个 queue, 每一个可以拼在一起的请求共用一个队列
        # 可以拼在一起：forward 次数相同（因为 fsdp 依赖这个）
        self._qs: dict[BatchGroupKey, asyncio.Queue[SingleRPCTask]] = {}

        self._batchers: dict[BatchGroupKey, asyncio.Task[None]] = {}
        self._batching_options: dict[BatchGroupKey, BatchingOptions] = {}
        self._batcher_lock = asyncio.Lock()

        dp_sizes = [ray.get(w.dp_size.remote()) for w in workers]
        assert all(x == dp_sizes[0] for x in dp_sizes), "dp_sizes are not the same"
        self.dp_size:int = dp_sizes[0]
        self.dp_ranks: list[int] = [ray.get(w.dp_rank.remote()) for w in workers]


    @classmethod
    def from_config(
        cls,
        worker_cls: Any,
        cfg: RuntimeConfig,
        *,
        replica_slug: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        name_prefix: str = "w",
        extra_actor_options: dict[str, Any] | None = None,
        worker_kwargs: dict[str, Any] | None = None,
    ) -> ServeReplicaRuntime:
        slug = replica_slug if replica_slug is not None else default_replica_slug()
        n = cfg.world_size
        opts: dict[str, Any] = {"num_cpus": cfg.num_cpus_per_worker}
        if cfg.num_gpus_per_worker > 0:
            opts["num_gpus"] = cfg.num_gpus_per_worker
        if extra_actor_options:
            opts.update(extra_actor_options)

        addr = master_addr or ray.util.get_node_ip_address()
        port = master_port if master_port is not None else _free_port()
        pk = cfg.init_parallel_state_kwargs()
        logger.info(f"{worker_cls=} options: {opts}, name_prefix: {name_prefix}, slug: {slug}, n: {n}.\n {worker_kwargs=} {pk=}")

        workers = [
            worker_cls.options(**opts, name=f"{name_prefix}_{slug}_r{r}").remote(
                rank=r, master_addr=addr, master_port=port, world_size=n, parallel_kwargs=pk, 
                **(worker_kwargs or {}),
            )
            for r in range(n)
        ]
        return cls(workers=workers, cfg=cfg)

    @property
    def workers(self) -> list[Any]:
        return self._w

    @property
    def world_size(self) -> int:
        return len(self._w)

    def same_rank_indices(self) -> list[int]:
        return list(range(len(self._w)))

    def maybe_pad_batch(self, batch: list[Any]) -> list[Any]:
        """Pad to ``world_size`` by repeating the last request (for collectives); HTTP replies stay length ``len(batch)``."""
        n = len(self._w)
        if len(batch) > n:
            raise ValueError(f"batch {len(batch)} > world_size {n}")
        if not batch:
            raise ValueError("empty gather batch")
        if len(batch) == n:
            return list(batch)
        last = batch[-1]
        return list(batch) + [last] * (n - len(batch))

    async def single_run(self, method: str, *, kwargs: dict[str, Any]) -> Any:
        w = random.choice(self._w)
        return await asyncio.to_thread(ray.get, getattr(w, method).remote(**kwargs))

    async def broadcast_run(self, method: str, *, kwargs: dict[str, Any]) -> Any:
        idx = self.same_rank_indices() # 同样，必须所有 worker 启动了才能调用, 因为 fsdp
        refs = [getattr(self._w[i], method).remote(**kwargs) for i in idx]
        ret = await asyncio.to_thread(ray.get, refs)
        return ret[0]

    async def batching_run(
        self,
        method: str,
        *,
        kwargs: dict[str, Any],
        batching_options: BatchingOptions | None = None,
    ) -> Any:
        batching_options = batching_options if batching_options is not None else BatchingOptions.default()
        key = BatchGroupKey(
            batch_key=batching_options.batch_key(method, kwargs),
            method=method,
        )

        # 给 key 初始化 queue 和 batcher
        # 如果 key 已经存在 queue 或者 batcher, 则直接返回
        await self._ensure_batcher(
            key,
            batching_options=batching_options,
        )

        # 任务加进队列
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._qs[key].put(SingleRPCTask(method=method, kwargs=kwargs, fut=fut))
        loguru.logger.info(f'任务已加入队列, 等待当前 key 对应 batcher 在现有 denoise 完成后从队列取出并执行')


        return await fut

    async def run(
        self,
        method: str,
        *,
        kwargs: dict[str, Any],
        batching_options: BatchingOptions | None = None,
    ) -> Any:
        if self.dp_size > 1:
            return await self.batching_run(
                method,
                kwargs=kwargs,
                batching_options=batching_options,
            )
        else:
            return await self.broadcast_run(method, kwargs=kwargs)

    async def _ensure_batcher(
        self,
        key: BatchGroupKey,
        batching_options: BatchingOptions,
    ) -> None:
        """
        调用这个表示把请求加队，并在一个时间窗口内或者设定的最大bs下组 batch

        每个 batching key 维护一个独立 queue + batcher(一个 batcher 为一个 asyncio.Task)。

        static_micro_batch_size: 表示先自己 microbatch 按预定大小组好，再在 dp_rank 中分配
        dynamic_micro_batch_size: 表示先分配到 dp_rank 中，再自己内部组 microbatch
        """
        async with self._batcher_lock:
            existing = self._batchers.get(key)
            if existing is not None and not existing.done():
                if self._batching_options.get(key) != batching_options:
                    raise ValueError(f"inconsistent batching options for key={key}")
                return
            self._batching_options[key] = batching_options
            q = self._qs.setdefault(key, asyncio.Queue())

            async def loop() -> None:
                while True:
                    first = await q.get()
                    logger.info(f'batch_key={key.batch_key} method={key.method} 收到新请求，等待新请求组 batch. [1/{self.dp_size}]')
                    buf: list[PendingBatchTask] = []
                    static_batch: list[SingleRPCTask] = []
                    max_tasks_per_flush = self.dp_size
                    if batching_options.is_dynamic():
                        max_tasks_per_flush = self.dp_size * batching_options.dynamic_micro_batch_size
                    if not batching_options.is_static():
                        buf.append(PendingBatchTask(method=first.method, kwargs=first.kwargs, futs=[first.fut]))
                    else:
                        static_batch.append(first)
                        loguru.logger.info(f'静态组 microbatch, [{len(static_batch)}/{batching_options.static_micro_batch_size}]')
                    loop = asyncio.get_running_loop()
                    t1 = loop.time() + self._cfg.batch_timeout_s

                    while len(buf) < max_tasks_per_flush:

                        def flush_static_batch():
                            if static_batch:
                                assert batching_options.kwargs_collate_fn is not None
                                buf.append(PendingBatchTask.from_single_rpc_tasks(static_batch, batching_options.kwargs_collate_fn))
                                loguru.logger.info(f'静态组 microbatch 超时或组满, [{len(static_batch)}/{batching_options.static_micro_batch_size}]')
                                static_batch.clear()

                        rem = t1 - loop.time()
                        if rem <= 0:
                            logger.info(f'batch_key={key} 组batch超时，直接执行')
                            flush_static_batch()
                            break
                        try:
                            if not batching_options.is_static():
                                new_task = await asyncio.wait_for(q.get(), timeout=rem)
                                buf.append(PendingBatchTask(method=new_task.method, kwargs=new_task.kwargs, futs=[new_task.fut]))
                                logger.info(f'batch_key={key} 收到新请求，组 batch ing. [{len(buf)}/{max_tasks_per_flush}]')
                            else:
                                if batching_options.kwargs_collate_fn is None:
                                    raise ValueError("kwargs_collate_fn must be provided when static_micro_batch_size > 1")
                                static_bs = batching_options.static_micro_batch_size
                                assert static_bs is not None
                                need = static_bs - len(static_batch)
                                for _ in range(need):
                                    new_task = await asyncio.wait_for(q.get(), timeout=rem)
                                    static_batch.append(new_task)
                                    loguru.logger.info(f'静态组 microbatch, [{len(static_batch)}/{batching_options.static_micro_batch_size}]')
                                flush_static_batch()

                        except asyncio.TimeoutError:
                            logger.debug(f'batch_key={key} 组batch超时，直接执行: {len(buf)=}, dp_size={self.dp_size}')
                            break
                        finally:
                            flush_static_batch()

                    if len(buf) == max_tasks_per_flush:
                        logger.info(f'batch_key={key} 组batch满了，执行. [{len(buf)}/{max_tasks_per_flush}]')
                    await self.flush_batch(
                        key,
                        buf,
                        batching_options=batching_options,
                    )

            self._batchers[key] = asyncio.create_task(loop())


    def _unique_dp_ranks(self) -> list[int]:
        uniq_dp = []
        seen = set()
        for r in self.dp_ranks:
            if r not in seen:
                seen.add(r)
                uniq_dp.append(r)
        return uniq_dp
        
    def assign_tasks(self, tasks: list[PendingBatchTask], *, dynamic: bool) -> dict[int, list[PendingBatchTask]]:
        """
        所有 worker 必须参与一次 forward（FSDP 等 collective 依赖 world 全参与）。

        先按 dp_rank 分组；若某 dp_rank 本轮没有分到请求，则用 tasks[0]
        占位，使每个 uniq_dp 至少有一条 PendingBatchTask（kwargs 与首条一致，
        同一对象可能被多个 dp 引用，仅用于算子侧对齐；HTTP future 仍只属于真实请求）。

        example（dp_ranks 与 uniq_dp 顺序一致时）:
            任务 [t0, t1, t2]，4 个 dp：先分配得 {dp0:[t0], dp1:[t1], dp2:[t2], dp3:[]}，
            再补空档得 {dp0:[t0], dp1:[t1], dp2:[t2], dp3:[t0]}。

            dynamic=True 时按轮询把新请求优先摊到不同 dp；任务数多于 dp 时再轮转。
       
        """
        assert len(tasks) > 0
        if not tasks:
            raise ValueError("tasks must not be empty")
        if not self.dp_ranks:
            raise ValueError("dp_ranks must not be empty")

        uniq_dp = self._unique_dp_ranks()
        groups: dict[int, list[PendingBatchTask]] = {dp_rank: [] for dp_rank in uniq_dp}
        for i, task in enumerate(tasks):
            if dynamic:
                target_dp = uniq_dp[i % len(uniq_dp)]
            else:
                target_dp = uniq_dp[min(i, len(uniq_dp) - 1)]
            groups[target_dp].append(task)

        # Add dummy tasks to each dp to avoid empty list / KeyError in flush.
        for dp_rank, task_list in groups.items():
            if not task_list:
                groups[dp_rank] = [tasks[0]]
        return groups

    async def flush_batch(
        self,
        key: BatchGroupKey,
        buf: list[PendingBatchTask],
        *,
        batching_options: BatchingOptions,
    ) -> None:

        dp_task_list_mapping = self.assign_tasks(buf, dynamic=batching_options.is_dynamic())
        dp_task_mapping: dict[int, PendingBatchTask] = {}
        if batching_options.enable_micro_batching():
            dp_task_mapping = {r: PendingBatchTask.from_pending_batch_tasks(tasks, batching_options.kwargs_collate_fn) for r, tasks in dp_task_list_mapping.items()}
        else:
            for r, tasks in dp_task_list_mapping.items():
                assert len(tasks) == 1
                dp_task_mapping[r] = tasks[0]

        try:
            refs = [getattr(self._w[r], dp_task_mapping[self.dp_ranks[r]].method).remote(**dp_task_mapping[self.dp_ranks[r]].kwargs) for r in range(len(self._w))]
            out: list[Any] = await asyncio.to_thread(ray.get, refs)
            for worker_id in range(len(self._w)):
                dp_rank = self.dp_ranks[worker_id]
                futs = dp_task_mapping[dp_rank].futs
                if batching_options.enable_micro_batching():
                    assert batching_options.output_split_fn is not None

                    split_out = batching_options.output_split_fn(out[worker_id])
                    assert len(split_out) == len(futs)
                    for fut, split_out_item in zip(futs, split_out):
                        if not fut.done():
                            fut.set_result(split_out_item)
                else:
                    for fut in futs:
                        if not fut.done():
                            fut.set_result(out[worker_id])
        except Exception as e:
            for task in dp_task_mapping.values():
                for fut in task.futs:
                    if not fut.done():
                        fut.set_exception(e)


class BaseIngress:
    def __init__(
        self,
        worker_cls: Any,
        cfg: RuntimeConfig,
        worker_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._worker_cls = worker_cls
        # from hy_parallelism.api.ray_worker import DistributedWorkerBase
        # assert issubclass(worker_cls, DistributedWorkerBase)
        self._cfg = cfg
        self._worker_kwargs = dict(worker_kwargs or {})

        self.runtime = ServeReplicaRuntime.from_config(
            self._worker_cls,
            self._cfg,
            replica_slug=default_replica_slug(),
            name_prefix="demo",
            worker_kwargs=self._worker_kwargs,
        )

def build_app() -> FastAPI:
    return FastAPI(title="hy_parallelism print demo", version="0.1.0")

class Server:

    def __init__(self, app: FastAPI, ingress_cls: type[BaseIngress], worker_cls: Any, worker_kwargs: dict[str, Any] | None = None, num_replicas: int = 2, runtime_config: RuntimeConfig | None = None, name: str = "hy_parallel_serve_demo", route_prefix: str = "/", host: str = "0.0.0.0", port: int = 8000):
        self._worker_cls = worker_cls
        self._num_replicas = num_replicas
        self._runtime_config = runtime_config
        self._name = name
        self._route_prefix = route_prefix
        self._host = host
        self._port = port
        self._worker_kwargs = worker_kwargs

        import os
        pythonpath = os.environ.get('PYTHONPATH', '')
        pythonpath = f'{pythonpath}:{os.getcwd()}'
        env = dict(os.environ).copy()
        env.update({
            'PYTHONPATH': pythonpath,
            # Instead of setting working_dir in `runtime_env` of ray.init (which would copy the entire project and be too slow),
            # we set an extra environment variable here. Our base ray worker will check for this variable
            # in its __init__ and change to that directory if it exists.
            'HY_RAY_WORKDIR': os.getcwd(), 
            'HY_PARALLELISM_LOGGING_DISABLE_FILENAME_FLUSH': "1", # ray 会 capture log, 然后拼东西，长度检测会失效
        })
        runtime_env = dict(
            env_vars=env,
            # working_dir=os.getcwd(), # will copy the whole project...
        )
        ray.init(ignore_reinit_error=True, log_to_driver=True, runtime_env=runtime_env)
        serve.start(http_options={"host": host, "port": port})
        self.dep = self.create_demo_serve_ingress(app=app, ingress_cls=ingress_cls, worker_cls=worker_cls, worker_kwargs=worker_kwargs, num_replicas=num_replicas, runtime_config=runtime_config)

    def run(self):
        serve.run(self.dep, name=self._name, route_prefix=self._route_prefix)
        logger.info(f"Ready http://{self._host}:{self._port}")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            serve.shutdown()
            ray.shutdown()


    @staticmethod
    def create_demo_serve_ingress(*, app, ingress_cls: type[BaseIngress], worker_cls: Any, worker_kwargs: dict[str, Any] | None = None, num_replicas: int = 2, runtime_config: RuntimeConfig | None = None):
        cfg = runtime_config or RuntimeConfig(world_size=4, tp=4, ep=1, dp_replicate=1, dp_shard=-1)
        assert issubclass(ingress_cls, BaseIngress)

        deployment_cls = serve.deployment(
            num_replicas=num_replicas, ray_actor_options={"num_cpus": 1},
            max_ongoing_requests=runtime_config.max_ongoing_requests,
        )(serve.ingress(app)(ingress_cls))

        return deployment_cls.bind(
            worker_cls=worker_cls,
            cfg=cfg,
            worker_kwargs=dict(worker_kwargs or {}),
        )