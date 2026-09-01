import math
from typing import Union, Iterator, List, Callable, Optional
import random
import warnings
from loguru import logger
from copy import deepcopy
from collections import deque

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler

from .indexer import ArrowIndexV2
from .bucket import MultiResolutionBucketIndexV2, MultiIndexV2, MultiMultiResolutionBucketIndexV2
from .utils import EmptyLogger, _arange


class BlockDistributedSampler(TorchDistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False,
                 batch_size=-1, start_index=0, align=1, use_numpy_indices=False):
        """
        Args:
            dataset: Dataset used for sampling.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process within num_replicas.
            shuffle: If True, the sampler will shuffle the indices.
            seed: Random seed.
            drop_last: If True, the sampler will drop the last batch if its size would be less than batch_size.
            batch_size: Size of mini-batch. If callable, it should accept a tuple of (w, h) as input and return an integer
                value as the batch size. It is useful for mix-scale(e.g., 256, 512, 1024) training.
            start_index: Start index for the sampler.
            align: Align the indices to the multiple of align for each dp.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if batch_size != -1:
            align = batch_size
            warnings.warn("batch_size is deprecated, please use `align` instead.")
        if align <= 0:
            raise ValueError(f"align should be a positive integer, but got {align}.")

        # When enabled, the sampler will use numpy to serve as indices, which significantly reduce the
        # memory usage.
        # Warning: If use_numpy_indices is True, the returned indices will be numpy.int64 type, which
        # is different from the default python int type.
        self.use_numpy_indices = use_numpy_indices

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.align = align
        self._start_index = start_index
        self.recompute_sizes()

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if self._start_index != value:
            self._start_index = value
            self.recompute_sizes()

    def recompute_sizes(self):
        self.num_samples = len(self.dataset) // self.align * self.align // self.num_replicas \
                           - self._start_index
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.use_numpy_indices:
                indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
            else:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            if self.use_numpy_indices:
                indices = _arange(len(self.dataset))  # type: ignore[arg-type]
            else:
                indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        raw_num_samples = len(indices) // self.align * self.align // self.num_replicas
        raw_total_size = raw_num_samples * self.num_replicas
        indices = indices[:raw_total_size]

        # subsample with start_index
        if self.use_numpy_indices:
            # Copy to release the memory of the original indices.
            indices = indices[self.rank * raw_num_samples + self.start_index:(self.rank + 1) * raw_num_samples].copy()
        else:
            indices = indices[self.rank * raw_num_samples + self.start_index:(self.rank + 1) * raw_num_samples]
        assert len(indices) + self.start_index == raw_num_samples, \
            f"{len(indices) + self.start_index} vs {raw_num_samples}"

        print(f"Iterator of BlockDistributedSampler created.")
        # This is a sequential sampler. The shuffle operation is done by the dataset itself.
        return iter(indices)


class DistributedSampler(TorchDistributedSampler):
    """
    A distributed sampler that supports moving the start index of the dataset to a specific position.
    This feature is useful when we want to resume training from a specific position in the dataset.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False,
                 start_index=0, raise_start_index_expired=False, batch_size=1, use_numpy_indices=False,
                 verbose=0, info_repr=None, low_cpu_memory=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.verbose = verbose
        if self.verbose <= 0:
            self.logger = EmptyLogger()
        else:
            self.logger = logger
        if info_repr is None:
            self.info_repr = f"[{self.__class__.__name__}] "
        else:
            self.info_repr = f"[{info_repr}] [{self.__class__.__name__}] "

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self._start_index = start_index
        self.shuffle = shuffle
        self.seed = seed
        self.low_cpu_memory = low_cpu_memory

        # When enabled, the sampler will use numpy to serve as indices, which significantly reduce the
        # memory usage.
        # Warning: If use_numpy_indices is True, the returned indices will be numpy.int64 type, which
        # is different from the default python int type.
        self.use_numpy_indices = use_numpy_indices

        # Define a flag to indicate whether the start_index is expired. The start_index is expired if and only
        # if the start_index is not 0 and the first epoch (i.e., the first time to create the iterator in the
        # current run) is finished. This flag will warn (or raise error) the user if the second epoch is started
        # but the start_index is not reset to 0.
        self.start_index_expired = False
        self.raise_start_index_expired = raise_start_index_expired

        # Used with MultiResolutionBucketIndexV2. When batch_size > 1, the interleaved indices will become
        # interleaved batches.
        self.batch_size = batch_size
        if self.batch_size > 1:
            assert drop_last is True, "When batch_size > 1, drop_last should be True."
            assert len(dataset) % self.batch_size == 0, \
                f"The dataset length({len(dataset)}) should be divisible by batch_size({self.batch_size})."
        elif self.batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer, but got {self.batch_size}.")

        self.recompute_sizes()

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if value % self.batch_size != 0:
            new_value = value // self.batch_size * self.batch_size
            message = f"start_index should be divisible by batch_size({self.batch_size}). Reset start_index from {value} to {new_value}."
            logger.warning(message)
            self._start_index = new_value
        else:
            self._start_index = value
        self.recompute_sizes()

    def recompute_sizes(self):
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and (len(self.dataset) - self._start_index) % (self.num_replicas * self.batch_size) != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            divider = self.num_replicas * self.batch_size
            self.num_samples = math.ceil(
                ((len(self.dataset) - self._start_index) - divider) / divider  # type: ignore[arg-type]
            ) * self.batch_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - self._start_index) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def blockwise_sample_iter(self, indices):
        # This function is used to create a blockwise iterator for the indices.
        # The indices are divided into blocks with size of self.batch_size.
        # The indices in the same block are from the same rank.

        total_len = len(indices)
        step = self.num_replicas * self.batch_size
        current_block_start = self.rank * self.batch_size
        while current_block_start < total_len:
            current_block_end = min(current_block_start + self.batch_size, total_len)

            # Get the indices in the current block
            for i in range(current_block_start, current_block_end):
                yield indices[i]

            # Move to the next block
            current_block_start += step

    def __iter__(self):
        if self.start_index_expired and self._start_index != 0:
            message = "The start_index is expired. Please reset the start_index to 0 before starting next epoch."
            if self.raise_start_index_expired:
                raise ValueError(message)
            else:
                logger.warning(message)

        self.logger.info(f"{self.info_repr}Getting dataset length")
        dataset_length = len(self.dataset)
        self.logger.info(f"{self.info_repr}dataset length: {dataset_length}")
        if self.shuffle:
            # low cpu memory not work for shuffle
            if self.low_cpu_memory:
                self.logger.info(f"Low cpu memory not work for shuffle.")
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.use_numpy_indices:
                indices = torch.randperm(dataset_length, generator=g).numpy()
            else:
                indices = torch.randperm(dataset_length, generator=g).tolist()  # type: ignore[arg-type]
            indices = indices[self._start_index:]
        else:
            if self.use_numpy_indices:
                self.logger.info(f"{self.info_repr}Creating numpy indices list.")
                if self.low_cpu_memory and self.drop_last and self.batch_size == 1:
                    # lazy init for saving memory. Now we only support drop_last=True and batch_size=1
                    indices = {'start': self._start_index, 'end': dataset_length, 'as_tensor': True}
                else:
                    indices = _arange(self._start_index, dataset_length, as_tensor=True)
                self.logger.info(f"{self.info_repr}Create numpy indices done.")
            else:
                self.logger.info(f"{self.info_repr}Creating python indices list.")
                if self.low_cpu_memory and self.drop_last and self.batch_size == 1:
                    # lazy init for saving memory. Now we only support drop_last=True and batch_size=1
                    indices = {'start': self._start_index, 'end': dataset_length, 'as_tensor': False}
                else:
                    indices = list(range(self._start_index, dataset_length))  # type: ignore[arg-type]
                self.logger.info(f"{self.info_repr}Create python indices done.")

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                if self.use_numpy_indices:
                    # indices = np.concatenate((indices, indices[:padding_size]))
                    indices = torch.cat((indices, indices[:padding_size]))
                else:
                    indices += indices[:padding_size]
            else:
                if self.use_numpy_indices:
                    indices = torch.cat((
                        indices,
                        torch.tile(indices, (math.ceil(padding_size / len(indices)),))[:padding_size],
                    ))
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            if self.low_cpu_memory and self.batch_size == 1:
                indices = {
                    'start': indices['start'],
                    'end': min(indices['start'] + self.total_size, indices['end']),
                    'as_tensor': indices['as_tensor']
                }
            else:
                indices = indices[:self.total_size]
        if isinstance(indices, dict):
            assert indices['end'] - indices['start'] == self.total_size
        else:
            assert len(indices) == self.total_size

        # subsample with start_index
        if self.batch_size == 1:
            if self.use_numpy_indices:
                self.logger.info(f"{self.info_repr}subsample according to rank and num_replicas (numpy)")
                if self.low_cpu_memory:
                    indices = _arange(indices['start'] + self.rank, indices['start'] + self.total_size, self.num_replicas)
                else:
                    # Copy to release the memory of the original indices.
                    indices = indices[self.rank:self.total_size:self.num_replicas].clone().numpy()
                # indices = indices[self.rank:self.total_size:self.num_replicas].copy()
            else:
                self.logger.info(f"{self.info_repr}subsample according to rank and num_replicas (list)")
                if self.low_cpu_memory:
                    indices = range(indices['start'] + self.rank, indices['start'] + self.total_size, self.num_replicas)
                else:
                    indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            self.start_index_expired = True
            logger.info(f"{self.info_repr}Iterator of DistributedSamplerWithStartIndex created.")
            return iter(indices)

        else:
            if self.use_numpy_indices:
                indices = indices.numpy()
            self.start_index_expired = True
            logger.info(f"{self.info_repr}Iterator of DistributedSamplerWithStartIndex created.")
            return self.blockwise_sample_iter(indices)


# For backward compatibility
DistributedSamplerWithStartIndex = DistributedSampler


def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r


class IndexBatchSampler(object):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        index_manager (ArrowIndexV2 or List[ArrowIndexV2]): Index manager.
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int or Callable or dict): Size of mini-batch. If callable, it should accept
            a tuple of (w, h) as input and return an integer value as the batch size. It is useful
            for mix-scale(e.g., 256, 512, 1024) training.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        multireso (bool): If ``True``, the sampler will use online resolution buckets.
        multidar (bool): If ``True``, the sampler will use online DurationAndResolution(DAR) buckets.
    """

    def __init__(self,
                 index_manager: Union[ArrowIndexV2, MultiIndexV2, List[ArrowIndexV2]],
                 sampler,
                 batch_size: Union[int, Callable] = 1,
                 drop_last: bool = False,
                 multireso: bool = False,
                 multidar: bool = False,
                 weights: list | tuple = None,
                 num_workers: int = 0,
                 prefetch_factor: Optional[int] = None,
                 history_buffer_size: int = 0,
                 ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.list_dataset = False
        if isinstance(index_manager, (list, tuple)):
            assert isinstance(sampler, (list, tuple)), \
                f"index_manager is a list of ArrowIndexV2|MultiIndexV2, but got sampler={type(sampler)}."
            assert isinstance(weights, (list, tuple)), \
                f"index_manager is a list of ArrowIndexV2|MultiIndexV2, but got probability={type(weights)}."
            assert len(index_manager) == len(sampler), \
                f"index_manager and sampler should have the same length, but got {len(index_manager)} vs {len(sampler)}."
            assert len(index_manager) == len(weights), \
                f"index_manager and probability should have the same length, but got {len(index_manager)} vs {len(weights)}."
            if not all(isinstance(idx, (ArrowIndexV2, MultiIndexV2)) for idx in index_manager):
                raise ValueError(f"index_manager should be a list of ArrowIndexV2|MultiIndexV2, but got {type(index_manager)}.")
            self.list_dataset = True
            self.cum_length = cumsum(index_manager)

        elif not isinstance(index_manager, (ArrowIndexV2, MultiIndexV2)):
            raise ValueError(f"index_manager should be an instance of ArrowIndexV2 or MultiIndexV2, "
                             f"but got {type(index_manager)}.")

        if multireso and multidar:
            raise ValueError("multireso and multidar cannot be both True.")

        if multireso or multidar:
            multi_bucket_error_message = (
                f"When use multireso/multidar for ArrowIndexV2/MultiIndexV2, please first set the resolutions by "
                f"calling index_manager.set_resolution_buckets or set the durations_and_resolutions by "
                f"calling index_manager.set_duration_and_resolution_buckets."
            )
            if isinstance(index_manager, (list, tuple)):
                for im in index_manager:
                    if isinstance(im, ArrowIndexV2) and (
                            multireso and im.resolutions is None or multidar and im.durations_and_resolutions is None
                    ):
                        raise ValueError(multi_bucket_error_message)
            else:
                if isinstance(index_manager, ArrowIndexV2) and (
                        multireso and index_manager.resolutions is None
                        or multidar and index_manager.durations_and_resolutions is None
                ):
                    raise ValueError(multi_bucket_error_message)
                elif isinstance(index_manager, MultiIndexV2) and (
                        multireso and any(bkt.resolutions is None for bkt in index_manager.buckets)
                        or multidar and any(bkt.durations_and_resolutions is None for bkt in index_manager.buckets)
                ):
                    raise ValueError(multi_bucket_error_message)

        if (
                not isinstance(batch_size, (int, Callable, dict))
                or (isinstance(batch_size, int) and batch_size <= 0)
                or (isinstance(batch_size, dict) and any([v <= 0 for v in batch_size.values()]))
        ):
            raise ValueError(
                f"batch_size should be a positive integer, or a callable, or a dict of positive integers, "
                f"but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

        self.index_manager = index_manager
        self.use_bucket = (
                isinstance(index_manager, (MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2))
                or (
                        isinstance(index_manager, (ArrowIndexV2, MultiIndexV2))
                        and (multireso or multidar)
                )
                or (
                        isinstance(index_manager, (list, tuple))
                        and all([isinstance(im, ArrowIndexV2) for im in index_manager])
                        and (multireso or multidar)
                )
        )
        self.sampler = sampler
        self.drop_last = drop_last
        self.multireso = multireso
        self.multidar = multidar
        self.cum_weights = [x / sum(weights) for x in weights] if self.list_dataset else None

        # Define online resolution buckets
        self.sampler_buckets = {}
        if isinstance(index_manager, ArrowIndexV2) and multireso:
            for item in index_manager.resolutions:
                self.sampler_buckets[item.size] = []
        elif isinstance(index_manager, MultiIndexV2) and multireso:
            for item in index_manager.buckets[0].resolutions:
                self.sampler_buckets[item.size] = []
        elif isinstance(index_manager, ArrowIndexV2) and multidar:
            for item in index_manager.durations_and_resolutions:
                self.sampler_buckets[item.size] = []
        elif isinstance(index_manager, MultiIndexV2) and multidar:
            for item in index_manager.buckets[0].durations_and_resolutions:
                self.sampler_buckets[item.size] = []
        elif isinstance(index_manager, MultiResolutionBucketIndexV2):
            for item in index_manager.resolutions:
                self.sampler_buckets[item.size] = []
        elif isinstance(index_manager, MultiMultiResolutionBucketIndexV2):
            for bucket in index_manager.buckets:
                for item in bucket.resolutions:
                    self.sampler_buckets[item.size] = []
        else:
            for im in index_manager:
                for item in im.resolutions:
                    self.sampler_buckets[item.size] = []

        # Precalculate batch size
        if self.use_bucket:
            if isinstance(batch_size, Callable):
                # When batch_size is callable, we can calculate the batch size for each resolution bucket.
                self.batch_size = {reso: batch_size(reso) for reso in self.sampler_buckets}
            elif isinstance(batch_size, dict):
                assert set(batch_size.keys()) == set(self.sampler_buckets.keys()), \
                    f"batch_size keys should be the same as the resolutions in index_manager, " \
                    f"but got {set(batch_size.keys())} vs {set(self.sampler_buckets.keys())}"
                self.batch_size = batch_size
            else:
                self.batch_size = {reso: batch_size for reso in self.sampler_buckets}
        else:
            self.batch_size = batch_size

        # Unique iter
        self._unique_iter = False
        self.sampler_iter = None

        # For resume training
        self.input_number = 0
        self.candidate_key_list = []
        self.num_prefetch_batches = num_workers * (prefetch_factor if prefetch_factor is not None else 2)
        # Store the history batches for state_dict.
        # `num_prefetch_batches` is for the not yet consumed prefetched batches in DataLoader.
        # `1` is for the last consumed batch (used for resuming).
        # `history_buffer_size` is for additional history batches (used for debugging).
        self.history_batches = deque(maxlen=self.num_prefetch_batches + 1 + history_buffer_size)

    def __iter__(self) -> Iterator[List[int]]:
        if not self._unique_iter:
            self._unique_iter = True
        else:
            raise RuntimeError("IndexBatchSampler is not re-iterable.")
        print(f"Iterator for IndexBatchSampler created.")

        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            if self.list_dataset:
                self.sampler_iter = [iter(s) for s in self.sampler]
            else:
                self.sampler_iter = iter(self.sampler)
            while True:
                try:
                    if self.use_bucket:
                        if self.list_dataset:
                            iter_i = random.choices(range(len(self.sampler)), weights=self.cum_weights)[0]
                            ind = next(self.sampler_iter[iter_i])
                            size = self.index_manager[iter_i].get_target_size(ind)
                            # Add cum_length to ind for ConcatDataset
                            if iter_i == 0:
                                pass
                            else:
                                ind += self.cum_length[iter_i - 1]
                        else:
                            # First clear the existed batches in the buckets
                            if len(self.candidate_key_list) > 0:
                                for key in self.candidate_key_list:
                                    bucket = self.sampler_buckets[key]
                                    if len(bucket) >= self.batch_size[key]:
                                        batch = bucket[:self.batch_size[key]]
                                        self.history_batches.append((key, batch))
                                        self.sampler_buckets[key] = bucket[self.batch_size[key]:]
                                        yield batch
                                self.candidate_key_list = []

                            ind = next(self.sampler_iter)
                            self.input_number += 1
                            size = self.index_manager.get_target_size(ind)

                        if size in self.sampler_buckets:
                            self.sampler_buckets[size].append(ind)
                            bsz = self.batch_size[size]
                            if len(self.sampler_buckets[size]) >= bsz:
                                batch = self.sampler_buckets[size][:bsz]
                                self.history_batches.append((size, batch))
                                self.sampler_buckets[size] = self.sampler_buckets[size][bsz:]
                                yield batch
                        else:
                            print(f"Error: invalid resolution {size} for index {ind}.")
                    else:
                        batch = [next(self.sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                except StopIteration:
                    break
        elif self.list_dataset:
            raise NotImplementedError(
                "Drop last is not supported for IndexBatchSampler with multiple datasets."
            )
        else:
            if self.use_bucket:
                for idx in self.sampler:
                    size = self.index_manager.get_target_size(idx)
                    if size in self.sampler_buckets:
                        self.sampler_buckets[size].append(idx)
                        if len(self.sampler_buckets[size]) == self.batch_size[size]:
                            yield self.sampler_buckets[size][:]
                            self.sampler_buckets[size] = []
                    else:
                        print(f"Error: invalid resolution {size} for index {idx}.")
                for k, v in self.sampler_buckets.items():
                    if len(v) > 0:
                        yield v
            else:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for idx in self.sampler:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                if idx_in_batch > 0:
                    yield batch[:idx_in_batch]

        # Clear sampler_buckets
        if self.use_bucket:
            for k, v in self.sampler_buckets.items():
                v.clear()
        # Reset unique iter
        self._unique_iter = False

    def __len__(self) -> int:
        if self.use_bucket:
            print(f"When using online bucket, length of dataloader is not well defined. We return the total number "
                  f"of samples across all the devices, instead of the number of batches in current device.")
            if self.list_dataset:
                return sum([len(im) for im in self.index_manager])
            else:
                return len(self.index_manager)
        else:
            # Can only be called if self.sampler has __len__ implemented
            # We cannot enforce this condition, so we turn off typechecking for the
            # implementation below.
            # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
            if self.drop_last:
                return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
            else:
                return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

    @property
    def required_samples(self):
        return self.input_number

    def state_dict(self, max_input_number: int, last_batch_index: Optional[list[int]]) -> dict:
        if self.list_dataset:
            raise NotImplementedError("Sync is not supported for IndexBatchSampler with multiple datasets.")

        # Request more samples from the sampler to fill the state up to max_input_number
        remain_number = max_input_number - self.input_number
        assert remain_number >= 0, \
            (f"max_input_number({max_input_number}) should be greater than or equal to "
             f"input_number({self.input_number}).")
        for _ in range(remain_number):
            try:
                ind = next(self.sampler_iter)
                self.input_number += 1
                size = self.index_manager.get_target_size(ind)

                if size in self.sampler_buckets:
                    self.sampler_buckets[size].append(ind)
                    self.candidate_key_list.append(size)
                else:
                    print(f"Error: invalid resolution ({size}) for index {ind}.")
            except StopIteration:
                break

        # tuple cannot be serialized in json format, so we join them with 'x',
        def size2str(size_tuple):
            return 'x'.join(map(str, size_tuple))

        def int_list(indices):
            return list(map(int, indices))

        states = {
            "input_number": self.input_number,
            "buckets": {},
            "candidates_key_list": [],
            "history_batches": [],
        }

        # When using num_workers > 0 in DataLoader, the prefetched batches may not be consumed yet.
        # We need to store these prefetched batches to the state_dict.
        if last_batch_index is not None and self.num_prefetch_batches > 0:
            # Search the last_batch_index from history_batches, and add the successive batches to the state_dict.
            found = False
            for _ in range(len(self.history_batches)):
                size, history_batch = self.history_batches.popleft()
                if found:
                    size_str = size2str(size)
                    if size_str not in states["buckets"]:
                        states["buckets"][size_str] = []
                    states["buckets"][size_str].extend(int_list(history_batch))
                    states["candidates_key_list"].append(size_str)
                else:
                    states["history_batches"].append(int_list(history_batch))
                    if history_batch == last_batch_index:
                        found = True
            if not found:
                warnings.warn("The last_batch_index is not found in the history_batches. It maybe caused by "
                              "inconsistent input num_workers and prefetch_factor compared with the DataLoader.")

        for size, v in self.sampler_buckets.items():
            size_str = size2str(size)
            if size_str not in states["buckets"]:
                states["buckets"][size_str] = []
            # int64 needed to be mapped to int.
            states["buckets"][size_str].extend(int_list(v))
        states["candidates_key_list"].extend([size2str(size) for size in self.candidate_key_list])

        return states

    def load_state_dict(self, state_dict):
        self.input_number = state_dict["input_number"]
        buckets = {tuple(map(int, k.split('x'))): v for k, v in state_dict["buckets"].items()}
        # Check the keys of buckets
        state_dict_keys = set(buckets.keys())
        sampler_bucket_keys = set(self.sampler_buckets.keys())
        if state_dict_keys != sampler_bucket_keys:
            raise ValueError(f"The keys of buckets in state_dict do not match the current sampler_buckets:\n"
                             f"Missing keys: {sampler_bucket_keys - state_dict_keys}\n"
                             f"Unexpected keys: {state_dict_keys - sampler_bucket_keys}")
        self.sampler_buckets = deepcopy(buckets)

        if "candidates_key_list" in state_dict:
            candidates_key_list = [tuple(map(int, k.split('x'))) for k in state_dict["candidates_key_list"]]
            self.candidate_key_list = deepcopy(candidates_key_list)
