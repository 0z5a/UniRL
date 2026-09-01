import math
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, DistributedSampler

T_co = TypeVar('T_co', covariant=True)


class DistributedSamplerFix(Sampler[T_co]):
    """
    The original DistributedSampler always align the num_samples among replicas when drop_last=False,
    This feature causes the total num_samples could be more than the actual dataset size, leading to
    vary total number of samples when using different number of GPUs, which affects the evaluation score
    such as FID, CLIP score, etc.

    This class implements a modified version of DistributedSampler that allows each replica to have different
    num_samples when drop_last=False.

    If drop_last=False and add_extra_samples=True, we will add extra samples to make it evenly divisible
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, add_extra_samples: bool | str = False,
                 repeat_times: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_world_size()` is dangerous.')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            raise RuntimeError('Using `dist.get_rank()` is dangerous.')
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.add_extra_samples = add_extra_samples
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
            self.total_size = self.num_samples * self.num_replicas
        elif self.add_extra_samples: # drop_last=False and add_extra_samples=True
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            self.total_size = self.num_samples * self.num_replicas
        else: # drop_last=False and add_extra_samples=False
            total_size = len(self.dataset)  # type: ignore[arg-type]
            self.num_samples = total_size // self.num_replicas + int(
                rank < total_size % self.num_replicas
            )
            self.total_size = total_size
        self.shuffle = shuffle
        self.seed = seed
        self.repeat_times = repeat_times

    def __iter__(self) -> Iterator[T_co]:
        dataset_length = len(self.dataset)  # type: ignore[arg-type]
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(dataset_length, generator=g).tolist()
        else:
            indices = list(range(dataset_length))

        if not self.drop_last:
            if not self.add_extra_samples:
                pass
            elif self.add_extra_samples == "extend":
                # Add extra samples to make it evenly divisible.
                # Extra samples are assigned with extended indices beyond dataset length.
                padding_size = self.total_size - len(indices)
                indices += [index + dataset_length for index in range(padding_size)]
            else:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Repeat multiple times
        if self.repeat_times:
            new_indices = indices.copy()
            for i in range(1, self.repeat_times):
                new_indices += [index + dataset_length * i for index in indices]
            indices = new_indices
            self.num_samples = self.num_samples * self.repeat_times
            self.total_size = self.total_size * self.repeat_times

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
