from __future__ import annotations

import os
from typing import Any

import ray


class DistributedWorkerBase:

    def __init__(
        self,
        *,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        parallel_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.rank = rank
        self._device_str = "cpu"
        self._world_size = world_size

        self.parallel_kwargs = parallel_kwargs or {}

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        working_dir = os.environ.get('HY_RAY_WORKDIR', None)
        if working_dir is not None:
            os.chdir(working_dir)

        from loguru import logger
        local_host = os.environ.get('LOCAL_IP', 'unknown')
        pythonpath = os.environ.get('PYTHONPATH', 'unknown')
        work_dir = os.getcwd()
        pid = os.getpid()
        logger.info(f'local_host={local_host}, pythonpath={pythonpath}, work_dir={work_dir}, pid={pid} {parallel_kwargs=} {world_size=}')

    def ping(self):
        return 'ok'

    def dp_rank(self) -> int:
        from hy_parallelism.parallel_states import get_parallel_state
        return get_parallel_state().dp_rank

    def dp_size(self) -> int:
        from hy_parallelism.parallel_states import get_parallel_state
        return get_parallel_state().dp_size

    def init_parallel_state(self) -> None:
        import torch
        import torch.distributed as dist

        import hy_parallelism.parallel_states as ps

        use_nccl = torch.cuda.is_available()
        backend = "nccl" if use_nccl else "gloo"
        if use_nccl:
            torch.cuda.set_device(torch.cuda.current_device())
            self._device_str = f"cuda:{torch.cuda.current_device()}"
        else:
            self._device_str = "cpu"

        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
        )
        ps.init_parallel_state(**self.parallel_kwargs)


@ray.remote
class DistributedWorker(DistributedWorkerBase):
    ...
