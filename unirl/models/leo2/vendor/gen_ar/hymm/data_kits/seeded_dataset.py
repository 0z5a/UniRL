from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
import struct
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


def _hash_task_tag(tag: str) -> int:
    """Stable 64-bit unsigned hash of a dataset_tag string."""
    h = hashlib.blake2b(tag.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def _stable_hash(base_seed: int, dp_rank: int, task_tag_hash: int,
                 epoch: int, idx: int) -> int:
    """Cross-process stable 63-bit non-negative hash."""
    h = hashlib.blake2b(digest_size=8)
    # task_tag_hash uses unsigned Q because _hash_task_tag may return up to 2**64-1.
    h.update(struct.pack("<qqQqq",
                         int(base_seed), int(dp_rank), int(task_tag_hash),
                         int(epoch), int(idx)))
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)



@contextmanager
def _seeded_rng_ctx(seed: int):
    from hy_parallelism.utils import isolate_rng
    with isolate_rng(include_cuda=False):
        random.seed(seed & 0xFFFFFFFFFFFFFFFF)
        np.random.seed(seed & 0xFFFFFFFF)              # np.random.seed needs uint32
        torch.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
        yield


class DeterministicSeededDatasetWrapper:
    """
    此 dataset 让 __getitem__ 的随机性固定，仅根据 (base_seed, dp_rank, task_tag, epoch, idx) 
    选择 seed，从而保证 reproducibility


    使用场景：
        原本的 Dataset 类的 __getitem__ 存在随机性，因此会受 dataloader worker 随机状态的影响。
        resume 时要么 resume worker 的随机状态，要么使用此类保证 reproducibility。
        前者会因 resume 时 worker 数量变化导致随机状态不一致，后者则不会。
        本 wrapper 为后者解决方案。

    为什么 Dataset 类的 __getitem__ 存在随机性：
        caption 字段从 ["caption_qwen", "caption_intern", ...] 里 random.choice 出一条，不同字段的文本长度差很多 → token_len 差。

    ``epoch_value`` is an ``mp.Value`` so the main process and worker children
    (fork or spawn) share one source of truth for the current epoch.
    ``task_tag`` is mixed into the seed so different datasets at the same
    ``idx`` don't collide.
    """

    def __init__(self, dataset: Any, base_seed: int, dp_rank: int,
                 epoch_value: "mp.Value", task_tag: str = ""):
        self.dataset = dataset
        self.base_seed = int(base_seed)
        self.dp_rank = int(dp_rank)
        self._epoch_value = epoch_value
        self.task_tag = str(task_tag)
        self._task_tag_hash = _hash_task_tag(self.task_tag)

    def __getitem__(self, idx):
        idx = int(idx)
        epoch = int(self._epoch_value.value)
        seed = _stable_hash(
            self.base_seed, self.dp_rank, self._task_tag_hash, epoch, idx
        )
        with _seeded_rng_ctx(seed):
            return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)
