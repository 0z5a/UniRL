import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from index_kits.sampler import BlockDistributedSampler, DistributedSampler, IndexBatchSampler
from loguru import logger
from torch._C._distributed_c10d import ProcessGroup

from ..data_kits.samplers import RepeatRandomDistributedSampler
from ..data_kits.utils.multimodal_states import DistributedSamplingState
from ..trainers.helpers import MultiModalScalarStates, MultiModalGRPOScalarStates
from ..utils.helpers import readable_time
from ..utils.import_utils import is_index_kits_version
from .utils.multimodal_states import fixed_dataloader_allocation


class CombinedBatchIterator(object):
    def __init__(self,
                 ss: Union[MultiModalScalarStates, MultiModalGRPOScalarStates],
                 fast_shuffle: bool,
                 rank: int,
                 world_size: int,
                 datasets: Dict[str, Any],
                 samplers: Dict[str, Any],
                 dataloaders: Dict[str, Any],
                 sampling_probs: Dict[str, float],
                 initial_seed: int,
                 seed_factor: int = 10000,
                 sampling_mode: str = "random",
                 cache_shuffle: bool = False,
                 fixed_key: str = None,
                 fixed_key_group: List[int] = None,
                 distributed_sampling_state: DistributedSamplingState = None,
                 use_ptm: bool = False,
                 ptm_dp_size: int = -1,
                 ptm_dp_group: Optional[ProcessGroup] = None,
                 force_sync_shuffle: bool = True,
                 pack_buffer_factor: int = 4,
                 keys2id: Optional[Dict[str, int]] = None,
                 infinity_iterator: bool = True,
                 dp_group: Optional[ProcessGroup] = None,
                 resume_index_batch_sampler: bool = False,
                 first_epoch_no_shuffle: Optional[bool] = None,
                 save_pack_buffer: bool = False,
                 save_all_ranks_training_states: bool = False,
                 ):
        """
        CombinedBatchIterator is a data iterator that combines multiple datasets with different sampling probabilities.

        Parameters
        ----------
        ss: MultiModalScalarStates
            Scalar states object that contains the epoch and consumed samples information.
        fast_shuffle: bool
            Whether to use fast shuffle.
        rank: int
            Rank of the current process.
        world_size: int
            Number of processes.
        datasets: Dict[str, Any]
            Dictionary of datasets. The keys should be the same as the keys of samplers, dataloaders, and sampling_probs.
        samplers: Dict[str, Any]
            Dictionary of samplers.
        dataloaders: Dict[str, Any]
            Dictionary of dataloaders.
        sampling_probs: Dict[str, float]
            Sampling probabilities of each dataset.
        initial_seed: int
            Initial seed for random shuffle.
        seed_factor: int
            Seed factor to distinguish different keys.
        sampling_mode: str
            Sampling mode. Default is "random". Valid values are ["random", "fixed"].
            - random: Each rank samples a dataset with the given probability.
            - fixed: Dataloaders are assigned to fixed ranks, following the sampling_probs as near as possible.
        cache_shuffle: bool
            Whether to cache the shuffle results. Default is False.
        fixed_key: str
            Fixed key for fixed sampling mode.
        fixed_key_group: List[int]
            The rank group contains the same `fixed key` with the current rank.
        use_ptm: bool
            whether ptm or torch
        ptm_dp_size: int
            if use_ptm, data_parallel_world_size
        ptm_dp_group: ProcessGroup or None
            if use_ptm, data_parallel_group
        force_sync_shuffle: bool
            Whether to force synchronizing the shuffle across all ranks. Default is True.
        pack_buffer_factor: int
            Capacity factor of the buffer for sequence packing. Default is 4x max_sequence_length of the dataset.
            A large buffer can improve the packing efficiency, but will also increase the memory usage.
        keys2id: Optional[Dict[str, int]]
            A mapping from dataset keys to their corresponding IDs. If None, it will be generated from datasets.keys().
        infinity_iterator: bool
            Whether to make the iterator infinite. Default is True.
        resume_index_batch_sampler: bool
            Whether to resume the IndexBatchSampler state. Default is False.
        save_pack_buffer: bool
            Whether pack_buffer is saved to checkpoint for bitwise-aligned resume
            in sequence_pack mode. Default is False.
        save_all_ranks_training_states: bool
            Whether per-rank training states are saved to checkpoint. Default is False.
            Pack-mode resume info (Y_min / per-rank prefix skip) is only computed
            when BOTH ``save_pack_buffer`` and ``save_all_ranks_training_states``
            are enabled; otherwise the default per-key ``epoch_consumed_samples``
            resume path is used.
        """
        self.logger = logger

        self.is_distributed = dist.is_available() and dist.is_initialized()

        # Only MultiModalScalarStates support accumulating multiple custom datasets at the same time.
        assert isinstance(ss, (MultiModalScalarStates, MultiModalGRPOScalarStates)), (
            f"`ss` must be an instance of MultiModalScalarStates or MultiModalGRPOScalarStates, but got {type(ss)}")
        self.ss = ss

        self.fast_shuffle = fast_shuffle
        self.rank = rank
        self.world_size = world_size
        self.dp_group = dp_group
        self.datasets = datasets
        self.samplers = samplers
        self.dataloaders = dataloaders
        self.sampling_probs = sampling_probs
        self.cache_shuffle = cache_shuffle
        self.fixed_key = fixed_key
        self.fixed_key_group = fixed_key_group
        self.distributed_sampling_state = distributed_sampling_state
        self.use_ptm = use_ptm
        self.ptm_dp_size = ptm_dp_size
        self.ptm_dp_group = ptm_dp_group
        self.force_sync_shuffle = force_sync_shuffle
        self.sampling_mode = sampling_mode
        self.infinity_iterator = infinity_iterator
        self.first_epoch_no_shuffle = first_epoch_no_shuffle if first_epoch_no_shuffle is not None else False
        # Pack dp and pp info for logging.
        if use_ptm:
            from megatron import mpu
            dp_rank = mpu.get_data_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            self.info = f"dp{dp_rank}, pp{pp_rank}"
        else:
            # Not implemented
            self.info = None
        if is_index_kits_version(">=", "0.5.14"):
            self.shuffle_kwargs = dict(info=self.info)
        else:
            self.shuffle_kwargs = dict()

        # Assign each key an id for setting shuffle seed.
        if keys2id is None:
            self.all_keys = list(self.datasets.keys())
            self.keys2id = {key: i for i, key in enumerate(self.all_keys)}
        else:
            self.keys2id = keys2id
        self.check_consistency()
        self.is_save_shuffle_rank = {}

        if sampling_mode == "random":
            self.keys = [key for key in self.datasets.keys() if self.sampling_probs[key] > 0]
            self.weights = [sampling_probs[key] for key in self.keys]

            # When `cache_shuffle` enabled, assign the save task to different nodes for speedup
            device_count = max(min(torch.cuda.device_count(), self.world_size), 1)
            num_nodes = max(self.world_size // device_count, 1)
            order = [j * device_count + i for i in range(device_count) for j in range(num_nodes)]
            for key, id_ in self.keys2id.items():
                self.is_save_shuffle_rank[key] = order[id_ % len(order)]

        elif sampling_mode == "fixed":
            # key_index should be pre-determined by CombinedBatchIterator.fixed_dataloader_allocation
            assert fixed_key is not None and fixed_key_group is not None, \
                "`fixed_key` and `fixed_key_group` should be provided in fixed sampling mode."
            # Assign the exact dataset to the rank
            self.keys = [fixed_key]
            self.weights = [1.0]

            # When `cache_shuffle` enabled, assign the save task to different nodes for speedup
            # Always save the shuffle cache on the first rank of current key group.
            self.is_save_shuffle_rank = {fixed_key: fixed_key_group[0]}

            self.logger.info(f"[Rank{self.rank}] Fixed sampling mode: {self.keys[0]}")

        else:
            raise ValueError(f"Invalid sampling mode: {sampling_mode}. Valid values are ['random', 'fixed'].")

        # ======= Define dataset shuffle seeds =======
        # initial_seed and seed_factor is used to assign different seeds for different datasets.
        # If dataset.task_kwargs contains 'shuffle_seed', it will be used instead.
        self.initial_seed = initial_seed
        self.seed_factor = seed_factor
        self.shuffle_base_seed = {}
        for key in self.keys:
            if getattr(self.datasets[key], "task_kwargs", None) is not None and \
                    "shuffle_seed" in self.datasets[key].task_kwargs and \
                    self.datasets[key].task_kwargs["shuffle_seed"] is not None:
                self.shuffle_base_seed[key] = self.datasets[key].task_kwargs["shuffle_seed"]
            else:
                self.shuffle_base_seed[key] = initial_seed + self.keys2id[key] * seed_factor
        self.logger.info(f"[Rank{self.rank}] Initial seed: {self.shuffle_base_seed}")

        # Pack-mode resume per-rank skip: for keys that need per-rank
        # prefix skipping after resume because pack-mode causes per-rank yield imbalance.
        self._resume_skip_per_key: Dict[str, int] = {}

        self.iterators = {key: self.get_iterator(key) for key in self.keys}

        # Prefetch the first batch of all dataloaders to ensure all dataloaders are ready at the beginning of training.
        self.first_flag = {key: True for key in self.keys}
        self.first_cache = {}

        # Sequence pack buffer
        self.pack_buffer_factor = pack_buffer_factor
        self.pack_buffer = {key: [] for key in self.keys}

        # Save last batch index for resuming
        self.last_batch_index = None
        self.resume_index_batch_sampler = resume_index_batch_sampler
        self.save_pack_buffer = save_pack_buffer
        self.save_all_ranks_training_states = save_all_ranks_training_states

        # shuffle before first get
        self.shuffle_and_initialize_index()

    @staticmethod
    def fixed_dataloader_allocation(weights, keys, rank, world_size, _logger=None):
        # bc
        return fixed_dataloader_allocation(weights, keys, rank, world_size, _logger=_logger)

    def check_consistency(self):
        """
        Make sure datasets, samplers, dataloaders, and sampling_probs have the same keys.
        These keys represent the data modalities, such as t2i, lm, mmu.
        """
        dataset_keys = set(self.datasets.keys())
        assert dataset_keys == set(self.samplers.keys()), \
            f"Datasets and samplers have different keys: {self.datasets.keys()} != {self.samplers.keys()}"
        assert dataset_keys == set(self.dataloaders.keys()), \
            f"Datasets and dataloaders have different keys: {self.datasets.keys()} != {self.dataloaders.keys()}"
        if self.sampling_mode == "random":
            assert dataset_keys == set(self.sampling_probs.keys()), \
                f"Datasets and sampling_probs have different keys: {self.datasets.keys()} != {self.sampling_probs.keys()}"
        for key in dataset_keys:
            assert key in self.keys2id, f"Key {key} is not in keys2id: {self.keys2id.keys()}"

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_flag:
            for key in self.keys:
                self.first_cache[key] = self.get_batch(key)
            self.first_flag = False

        emit_shuffle = {key: False for key in self.keys}
        # Get the batch with the given sampling probability
        cur_key = random.choices(self.keys, weights=self.weights, k=1)[0]
        if cur_key in self.first_cache:
            batch, cur_emit_shuffle = self.first_cache.pop(cur_key)
        else:
            batch, cur_emit_shuffle = self.get_batch(cur_key)
        if not self.infinity_iterator and cur_emit_shuffle:
            raise StopIteration(f"Dataset {cur_key} is exhausted.")

        # If sync_shuffle is disabled, we only shuffle the current rank if it emits shuffle, and
        # return the batch immediately.
        # !!!Note that in the end of dataset epoch, shuffle may be triggered in successive multiple steps
        # by different ranks, which may cause the speed of training to be slow for a long time
        # if the shuffle is time-consuming.
        if not self.force_sync_shuffle:
            if cur_emit_shuffle:
                self.shuffle(cur_key)
                batch, _ = self.get_batch(cur_key)
            return batch

        emit_shuffle[cur_key] = cur_emit_shuffle

        if self.is_distributed:
            # Globally synchronize shuffle flags
            if self.use_ptm:
                assert self.ptm_dp_size > 0 and self.ptm_dp_group, f'before synchronize shuffle flags, error self.ptm_dp_size:{self.ptm_dp_size}, self.ptm_dp_group:{self.ptm_dp_group}'
                shuffle_flags: List[Optional[Dict[str, bool]]] = [None for _ in range(self.ptm_dp_size)]
                torch.distributed.all_gather_object(shuffle_flags, emit_shuffle, group=self.ptm_dp_group)
            else:
                # shuffle_flags: List[Optional[Dict[str, bool]]] = [None for _ in range(self.world_size)]
                shuffle_flags: List[Optional[Dict[str, bool]]] = [None for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(shuffle_flags, emit_shuffle)
        else:
            shuffle_flags = [emit_shuffle]

        # Emit shuffle if any worker emits shuffle.
        for key in self.keys:
            if any([flag[key] for flag in shuffle_flags if key in flag]):
                self.shuffle(key)
                # After shuffle, get the new batch of current key
                if key == cur_key:
                    batch, _ = self.get_batch(key)

        return batch

    def __len__(self):
        raise NotImplementedError("Length of TextImageBatchIterator is undefined.")

    def shuffle_and_initialize_index(self):
        self.logger.info(f"[Rank{self.rank}] Start random shuffle")
        start = time.time()
        shuffle_times = {}

        # In distributed pack-mode, gather consumed_samples_per_dp from all ranks to get Y_min and prefix skip,
        # ensuring ranks resume correctly aligned after imbalance.
        pack_resume_enabled = self.save_pack_buffer and self.save_all_ranks_training_states
        pack_resume_info = self._compute_pack_resume_info() if pack_resume_enabled else {}

        for key in self.keys:
            seed = self.shuffle_base_seed[key] + self.ss.consumed_epoch[key]
            sampler = self.samplers[key]
            if isinstance(sampler, BlockDistributedSampler):
                consumed_samples = self.ss.consumed_samples_per_dp[key]
            elif isinstance(sampler, (DistributedSampler, RepeatRandomDistributedSampler)):
                # Check order: epoch_required_samples -> pack-mode buffer skip ->
                # epoch_consumed_samples.
                if self.resume_index_batch_sampler and \
                        (required_samples := self.ss.epoch_required_samples.get(key, -1)) >= 0:
                    consumed_samples = required_samples
                    # Load state_dict for IndexBatchSampler. Also handle the redistributed case.
                    self.load_state_dict(key)
                elif key in pack_resume_info:
                    info = pack_resume_info[key]
                    consumed_samples = info["num_replicas"] * info["Y_min"]
                    if info["skip"] > 0:
                        self._resume_skip_per_key[key] = info["skip"]
                    self.logger.info(
                        f"[Rank{self.rank}] {key} pack-mode resume: "
                        f"start_index = N({info['num_replicas']}) "
                        f"x Y_min({info['Y_min']:,}) = {consumed_samples:,}; "
                        f"local Y_R = {self.ss.consumed_samples_per_dp[key]:,}; "
                        f"skip {info['skip']} prefix yield(s) silently in wrapper."
                    )
                else:
                    # Default path: per-rank yields are uniform, so global ``epoch_consumed_samples`` works, or fresh start (everything is 0).
                    consumed_samples = self.ss.epoch_consumed_samples[key]
            else:
                consumed_samples = 0
                self.logger.warning(f"{self.samplers[key].__class__.__name__} doesn't support set start_index."
                                    f"The sampler will be reset to the beginning of the dataset.")

            cur_epoch = self.ss.consumed_epoch[key]
            do_shuffle = False if (self.first_epoch_no_shuffle and cur_epoch == 0) else True
            self.logger.info(
                f"[Rank{self.rank}] {key} dataset epoch: {cur_epoch}. "
                f"Shuffle with seed={seed}, fast={self.fast_shuffle}, cache={self.cache_shuffle}. " if do_shuffle else f"No shuffle. "
                f"Initialize with consumed_samples={consumed_samples:,}"
            )
            try:
                self.shuffle_and_reset_index(
                    self.datasets[key], self.samplers[key], seed=seed, start_index=consumed_samples,
                    save_cache=self.cache_shuffle and self.rank == self.is_save_shuffle_rank[key],
                    do_shuffle=do_shuffle,
                )
            except Exception as e:
                self.logger.error(f"[Rank{self.rank}] Error in shuffling {key}: {e}")
                raise e
            if hasattr(self.datasets[key], "index_manager"):
                shuffled_indices_examples = self.datasets[key].index_manager.ind_mapper[:10] \
                    if hasattr(self.datasets[key].index_manager, "ind_mapper") \
                    else self.datasets[key].index_manager.indices[:10]
                if isinstance(shuffled_indices_examples, torch.Tensor):
                    shuffled_indices_examples = shuffled_indices_examples.tolist()
                elif isinstance(shuffled_indices_examples, np.ndarray):
                    shuffled_indices_examples = shuffled_indices_examples.tolist()
                self.logger.info(f"[Rank{self.rank}] {key} dataset first 10 shuffled indices: {shuffled_indices_examples}")
            # Set current epoch in dataset for correct worker seeding
            self._set_dataset_epoch(key, self.ss.consumed_epoch[key])
            shuffle_times[key] = time.time() - start
            start = time.time()
        for key, time_cost in shuffle_times.items():
            self.logger.info(f"[Rank{self.rank}] Shuffle {key} time cost: {readable_time(time_cost)}")
        self.logger.info(f"[Rank{self.rank}] End of random shuffle")

    def _set_dataset_epoch(self, key: str, epoch: int) -> None:
        """Push current epoch into the wrapped dataset's ``mp.Value`` so worker
        processes derive the right per-sample seed.
        """
        # _epoch_value 是 seeded_dataset 维护的一个属性
        epoch_value = getattr(self.datasets[key], "_epoch_value", None)
        if epoch_value is None:
            return
        epoch_value.value = int(epoch)

    def _compute_pack_resume_info(self) -> Dict[str, Dict[str, int]]:
        """Gather per-rank ``consumed_samples_per_dp[key]`` (= ``Y_R``, this
        rank's total yields from the underlying DataLoader for ``key`` in the
        current epoch) across the dp group, then derive per-key ``Y_min`` and
        per-rank prefix ``skip``.

        Returns
        -------
        dict[key] -> {'Y_min': int, 'skip': int, 'num_replicas': int}
            Only keys that need buffer skip adjustment on this rank are present.
            Empty dict means buffer skip is not needed for this rank/init.
        """
        # Eligibility only depends on sampler type and dataset.sequence_pack; don't check Y_R > 0, to ensure all replicas stay in sync.
        eligible: Dict[str, int] = {}  # key -> num_replicas
        for key in self.keys:
            sampler = self.samplers[key]
            if not isinstance(sampler, (DistributedSampler, RepeatRandomDistributedSampler)):
                continue
            ds = self.datasets.get(key)
            if not (hasattr(ds, "sequence_pack") and ds.sequence_pack):
                continue
            num_replicas = int(getattr(sampler, "num_replicas", 0) or 0)
            if num_replicas <= 0:
                continue
            eligible[key] = num_replicas

        local_Y: Dict[str, int] = {
            key: int(self.ss.consumed_samples_per_dp.get(key, 0) or 0)
            for key in self.keys
        }
        if self.is_distributed and self.dp_group is not None:
            world = dist.get_world_size(group=self.dp_group)
            gathered: List[Optional[Dict[str, int]]] = [None] * world
            dist.all_gather_object(gathered, local_Y, group=self.dp_group)
        else:
            gathered = [local_Y]

        info: Dict[str, Dict[str, int]] = {}
        for key, num_replicas in eligible.items():
            my_Y_R = int(local_Y.get(key, 0))
            peer_values: List[int] = []
            for d in gathered:
                if isinstance(d, dict) and key in d:
                    peer_values.append(int(d[key]))
            if not peer_values:
                continue
            if all(v == 0 for v in peer_values):
                continue
            Y_min = min(peer_values)
            info[key] = {
                "Y_min": Y_min,
                "skip": my_Y_R - Y_min,
                "num_replicas": num_replicas,
            }
        return info

    def shuffle(self, key):
        cur_epoch = self.ss.consumed_epoch_per_dp[key] + 1
        seed = self.shuffle_base_seed[key] + cur_epoch

        self.logger.info(
            f"[Rank{self.rank}] {key} dataset epoch: {cur_epoch}. "
            f"Shuffle with seed={seed}, fast={self.fast_shuffle}, cache={self.cache_shuffle}."
        )

        try:
            self.shuffle_and_reset_index(
                self.datasets[key], self.samplers[key], seed=seed, start_index=0,
                save_cache=self.cache_shuffle and self.rank == self.is_save_shuffle_rank[key],
            )
        except Exception as e:
            self.logger.error(f"[Rank{self.rank}] Error in shuffling {key}: {e}")
            raise e
        if hasattr(self.datasets[key], "index_manager"):
            shuffled_indices_examples = self.datasets[key].index_manager.ind_mapper[:10] \
                if hasattr(self.datasets[key].index_manager, "ind_mapper") \
                else self.datasets[key].index_manager.indices[:10]
            if isinstance(shuffled_indices_examples, torch.Tensor):
                shuffled_indices_examples = shuffled_indices_examples.tolist()
            elif isinstance(shuffled_indices_examples, np.ndarray):
                shuffled_indices_examples = shuffled_indices_examples.tolist()
            self.logger.info(f"[Rank{self.rank}] {key} dataset first 10 shuffled indices: {shuffled_indices_examples}")

        # Shuffle 后重建 IndexBatchSampler，使分桶与新的 index→分辨率映射一致。
        # 同时清除 dl._iterator，因为 persistent_workers 模式下 DataLoader 会复用旧
        # iterator，而旧 iterator 缓存了旧的 IndexBatchSampler，不会读取新替换的。
        # TODO: 最好后续在 IndexBatchSampler 中支持 reset()，这里是一个临时的 hack。
        dl = self.dataloaders[key]
        old_bs = getattr(dl, 'batch_sampler', None)
        if isinstance(old_bs, IndexBatchSampler):
            new_bs = IndexBatchSampler(
                index_manager=self.datasets[key].index_manager,
                sampler=self.samplers[key],
                batch_size=old_bs.batch_size,
                drop_last=old_bs.drop_last,
                multireso=getattr(old_bs, 'multireso', False),
                multidar=getattr(old_bs, 'multidar', False),
                num_workers=getattr(old_bs, 'num_workers', 0),
                prefetch_factor=getattr(old_bs, 'prefetch_factor', None),
                history_buffer_size=getattr(old_bs, 'history_buffer_size', 64),
            )
            object.__setattr__(dl, 'batch_sampler', new_bs)
            if getattr(dl, '_iterator', None) is not None:
                object.__setattr__(dl, '_iterator', None)
            self.logger.info(f"[Rank{self.rank}] Rebuilt IndexBatchSampler for {key} after shuffle.")

        self.iterators[key] = self.get_iterator(key)

        # After shuffle, reset the consumed samples and epoch for the current key.
        self.ss.consumed_samples_per_dp[key] = 0
        self.ss.epoch_consumed_samples[key] = 0
        self.ss.consumed_epoch_per_dp[key] += 1
        # Set current epoch in dataset for correct worker seeding
        self._set_dataset_epoch(key, self.ss.consumed_epoch_per_dp[key])

    @staticmethod
    def solve_bin_packing(sizes, n_bins, max_length):
        bins = [[] for _ in range(n_bins)]
        remain_capacity = [max_length for _ in range(n_bins)]

        # Assign IDs to each item based on their index in the sizes list
        id_sizes = list(enumerate(sizes))

        id_flag = [True] * len(id_sizes)
        # FF(First-Fit) algorithm for bin packing
        for id_, size in id_sizes:
            for i in range(n_bins):
                if size <= remain_capacity[i]:
                    bins[i].append(id_)
                    remain_capacity[i] -= size
                    id_flag[id_] = False
                    break

        return bins, id_flag

    def get_batch(self, key):
        try:
            batch = next(self.iterators[key])
            bsz = len(batch)
            if hasattr(self.datasets[key], "sequence_pack") and self.datasets[key].sequence_pack:
                # When sequence_pack is enabled, batch will be a list of samples instead of a single batch dict.
                assert bsz == 1, \
                    f"When sequence_pack is enabled, the dataloader should have batch_size=1, but got {bsz}."
                # We set a buffer to store multiple samples for sequence packing.
                # The buffer capacity is pack_buffer_factor * max_sequence_length of the dataset.
                # Larger buffer can improve the packing efficiency, but will also increase the memory usage.
                capacity = self.pack_buffer_factor * self.datasets[key].max_sequence_length
                self.pack_buffer[key].extend(batch)
                remain_len = capacity - sum([item["tokens"].shape[0] for item in self.pack_buffer[key]])
                while remain_len > 0:
                    batch = next(self.iterators[key])
                    self.pack_buffer[key].extend(batch)
                    remain_len -= sum([item["tokens"].shape[0] for item in batch])
                # Sequence pack the batch. shape[0] is the sequence length.
                sizes = [item["tokens"].shape[0] for item in self.pack_buffer[key]]
                id_bins, remain_flag = self.solve_bin_packing(
                    sizes, bsz, self.datasets[key].max_sequence_length
                )
                id_bin = id_bins[0]  # only use the first bin to form the batch. We assume bsz=1.
                if len(id_bin) == 0:
                    raise RuntimeError("No samples can be packed into the batch. The buffer items have lengths: "
                                       f"{sizes}, max_sequence_length={self.datasets[key].max_sequence_length}. ")
                item_bin = [self.pack_buffer[key][index] for index in id_bin]
                self.pack_buffer[key] = [self.pack_buffer[key][index] for index, flag in enumerate(remain_flag) if flag]
                batch = self.datasets[key].seq_collate_fn(item_bin)

            else:
                # If `index` is in batch, save the last index for resuming.
                if "index" in batch:
                    self.last_batch_index = batch["index"]

            emit_global_shuffle = False
        except StopIteration:
            batch = None
            emit_global_shuffle = True
        return batch, emit_global_shuffle

    def get_iterator(self, key):
        # Skip prefix batches already yielded by this rank before resume, to align per-rank cursor. 
        # These are dropped silently and not counted again. These skipped batches are already 
        # yielded by this rank in baseline and pushed into buffer or have been consumed by the model.
        skip = int(self._resume_skip_per_key.pop(key, 0) or 0)
        if skip > 0:
            self.logger.info(
                f"[Rank{self.rank}] {key} resume-skip: silently consuming "
                f"{skip} prefix yield(s) from underlying DataLoader to align "
                f"per-rank cursor with baseline (Y_R - Y_min)."
            )
        for idx, batch in enumerate(self.dataloaders[key]):
            if skip > 0:
                skip -= 1
                continue
            if isinstance(batch, list):
                self.ss.consumed_samples_per_dp[key] += sum([item["n_samples"] for item in batch])
            else:
                self.ss.consumed_samples_per_dp[key] += batch["n_samples"].sum().item()
            yield batch

    def shuffle_and_reset_index(self, dataset, sampler, seed, start_index, save_cache=False, do_shuffle=True):
        if do_shuffle and hasattr(dataset, "shuffle"):
            dataset.shuffle(seed, fast=self.fast_shuffle, use_cache=self.cache_shuffle, save_cache=save_cache,
                            **self.shuffle_kwargs)
        sampler.start_index = start_index
        self.logger.info(f"Update {sampler.__class__.__name__} states: {start_index=:,}, {len(sampler)=:,}")

    def sync_state_dict(self):
        assert self.sampling_mode == "fixed", "sync_batch_sampler is only available in fixed sampling mode."
        key = self.keys[0]
        batch_sampler = self.dataloaders[key].batch_sampler
        if isinstance(batch_sampler, IndexBatchSampler):
            required_samples = batch_sampler.required_samples
        else:
            required_samples = -1
        # Sync the required samples across all the dp ranks and then grouped by dataset_tag.
        all_dataset_required_samples: list[dict | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            all_dataset_required_samples,
            {"dataset_tag": key, "required_samples": required_samples},
            group=self.dp_group
        )
        self.ss.epoch_required_samples.clear()
        unique_keys = sorted(list(set([rs["dataset_tag"] for rs in all_dataset_required_samples])))
        max_required_samples = {}
        for u_key in unique_keys:
            u_required_samples = [rs for rs in all_dataset_required_samples if rs["dataset_tag"] == u_key]
            max_required_samples[u_key] = max(rs["required_samples"] for rs in u_required_samples)
            self.ss.epoch_required_samples[u_key] = max_required_samples[u_key] * len(u_required_samples)

        # Sync DistributedSampler states by maximum required_samples in the same dataset_tag group,
        # and return the synced state_dict.
        if isinstance(batch_sampler, IndexBatchSampler):
            synced_state_dict = batch_sampler.state_dict(max_required_samples[key], self.last_batch_index)
        else:
            synced_state_dict = {}
        all_dataset_batch_sampler_state_dict: list[dict | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            all_dataset_batch_sampler_state_dict,
            {"dataset_tag": key, "state_dict": synced_state_dict},
            group=self.dp_group
        )
        batch_sampler_state_dict_group_by_key = {}
        for u_key in unique_keys:
            u_state_dicts = [rs for rs in all_dataset_batch_sampler_state_dict if rs["dataset_tag"] == u_key]
            # Just take the first one as they are the same after sync.
            batch_sampler_state_dict_group_by_key[u_key] = u_state_dicts
        self.ss.grouped_batch_sampler_state_dict = batch_sampler_state_dict_group_by_key

    def get_pack_buffer_state(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """
        Return a serializable state of pack_buffer for checkpoint saving.
        Only includes keys where sequence_pack is enabled and buffer is non-empty.
        Tensors are moved to CPU for serialization.
        """
        state = {}
        for key in self.keys:
            if not (hasattr(self.datasets[key], "sequence_pack") and self.datasets[key].sequence_pack):
                continue
            if not self.pack_buffer[key]:
                continue
            state[key] = []
            for item in self.pack_buffer[key]:
                state[key].append({
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in item.items()
                })
        return state if state else None

    def get_pack_buffer_state_for_checkpoint(self, dp_size: int) -> Optional[Dict[str, Any]]:
        """
        Return this rank's pack_buffer state for checkpoint. No gather: each rank
        returns only its local buffer to avoid OOM when dp_size is large (gather
        or all_gather would put full buffer on one or all ranks).

        The trainer should save this to a per-rank file (e.g. pack_buffer_rank{r}.pt)
        so no process ever holds more than its local buffer. When loading, each rank
        loads only its own file and calls load_pack_buffer_state(..., loaded_dp_size=1).

        Returns:
            None if no buffer to save; else dict with keys:
            - "pack_buffer_state": {key: [this_rank_items]} (one list per key)
            - "pack_buffer_dp_size": dp_size at save time
        """
        local_state = self.get_pack_buffer_state()
        if local_state is None:
            return None
        # Same shape as before: per key a list (here of length 1 = this rank only)
        return {
            "pack_buffer_state": {k: [v] for k, v in local_state.items()},
            "pack_buffer_dp_size": dp_size,
        }

    def get_pack_buffer_metadata_for_checkpoint(self, dp_size: int) -> Optional[Dict[str, Any]]:
        """
        Gather only per-rank counts (no buffer data) for DCP-style redistribute on load.
        Used to save pack_buffer_metadata.pt so that when dp_size changes, each rank can
        read old per-rank files one by one and keep only its slice (global_index % new_dp_size == rank).

        Returns:
            None if no buffer; else dict with "pack_buffer_dp_size" and "counts": {key: [c0, c1, ...]}.
            All ranks get the same metadata (from all_gather_object of counts).
        """
        local_state = self.get_pack_buffer_state()
        if local_state is None:
            return None
        counts_local = {k: len(v) for k, v in local_state.items()}
        if self.dp_group is None or not self.is_distributed:
            return {
                "pack_buffer_dp_size": 1,
                "counts": {k: [len(v)] for k, v in local_state.items()},
            }
        gathered: List[Optional[Dict[str, int]]] = [None] * dp_size
        dist.all_gather_object(gathered, counts_local, group=self.dp_group)
        all_keys: set = set()
        for d in gathered:
            if d is not None:
                all_keys |= set(d.keys())
        if not all_keys:
            return None
        counts = {
            key: [gathered[r].get(key, 0) if gathered[r] is not None else 0 for r in range(dp_size)]
            for key in all_keys
        }
        return {"pack_buffer_dp_size": dp_size, "counts": counts}

    @staticmethod
    def load_pack_buffer_from_per_rank_files_with_redistribute(
        load_path: Union[Path, str],
        metadata: Dict[str, Any],
        rank: int,
        new_dp_size: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        dp_size 变化时从 per-rank 文件恢复 pack_buffer 并做 redistribute。
        按「全局下标 % new_dp_size == rank」把 item 分给当前 rank；每次只读一个旧 rank 的
        文件，只保留本 rank 的切片后即释放，避免全量 gather 导致 OOM。

        Args:
            load_path: checkpoint 目录（含 pack_buffer_rank{r}.pt 的目录）。
            metadata: 含 "pack_buffer_dp_size" 与 "counts"（{key: [c0, c1, ...]}）的 dict。
            rank: 当前 rank（用于 global_index % new_dp_size == rank）。
            new_dp_size: 当前 dp_size。

        Returns:
            本 rank 的 state：{key: [item, ...]}，可直接传给 load_pack_buffer_state(..., loaded_dp_size=1)。
        """
        load_path = Path(load_path)
        saved_dp_size = metadata.get("pack_buffer_dp_size", 1)
        counts_by_key = metadata.get("counts", {}) or {}
        my_state = {key: [] for key in counts_by_key}
        # prefix_by_key[key][r] = 旧 rank 0..r-1 上该 key 的 item 总数，用于算 global_index
        prefix_by_key = {}
        for key in counts_by_key:
            counts = counts_by_key[key]
            prefix_by_key[key] = [0]
            for c in counts:
                prefix_by_key[key].append(prefix_by_key[key][-1] + c)
        for r in range(saved_dp_size):
            fpath = load_path / f"pack_buffer_rank{r}.pt"
            if not fpath.exists():
                continue
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            state_r = ckpt.get("pack_buffer_state") or {}
            for key in counts_by_key:
                if key not in state_r or not state_r[key]:
                    continue
                list_r = state_r[key][0]
                prefix = prefix_by_key[key]
                for i in range(len(list_r)):
                    global_index = prefix[r] + i
                    if global_index % new_dp_size == rank:
                        my_state[key].append(list_r[i])
            del ckpt
        return my_state

    def load_pack_buffer_state(
        self,
        state: Optional[Dict[str, Any]],
        loaded_dp_size: Optional[int] = None,
    ):
        """
        Restore pack_buffer from checkpoint for accurate resume with sequence pack.

        Supports three checkpoint formats:
        - Single-rank file (per-rank files, no OOM): state[key] = [this_rank_items],
          loaded_dp_size=1. Each rank loads only its own file. When dp_size changes,
          DCP-style redistribute is supported if pack_buffer_metadata.pt exists:
          each rank reads old files one-by-one and keeps only its slice (global_index % new_dp_size == rank).

        Args:
            state: Pack buffer state from checkpoint, or None to skip.
            loaded_dp_size: Dp size when checkpoint was saved. If None, assumes same as world_size.
        """
        if state is None:
            return

        # Single-rank file format: state[key] = [this rank's list], loaded_dp_size=1
        assert loaded_dp_size == 1, "Single-rank file format only supports loaded_dp_size=1."
        for key in state:
            if key not in self.pack_buffer:
                continue
            rank_lists = state[key]
            self.pack_buffer[key] = list(rank_lists[0]) if rank_lists else []
        restored_keys = [k for k in state if k in self.pack_buffer and self.pack_buffer[k]]
        if restored_keys:
            self.logger.info(
                f"[Rank{self.rank}] Restored pack_buffer state for keys: {restored_keys} (single-rank file)"
            )
        return

    def load_state_dict(self, key):
        assert self.sampling_mode == "fixed", "load_state_dict is only available in fixed sampling mode."
        assert self.distributed_sampling_state is not None, "distributed_sampling_state must be provided when " \
                                                            "loading state_dict in fixed sampling mode."
        batch_sampler = self.dataloaders[key].batch_sampler

        if isinstance(batch_sampler, IndexBatchSampler):
            dss = self.distributed_sampling_state
            # Determine whether we need to redistribute the batch_sampler_state_dict
            loaded_dataset_world_size = len(self.ss.grouped_batch_sampler_state_dict[key])
            if loaded_dataset_world_size == dss.cur_size:
                state_dict = self.ss.grouped_batch_sampler_state_dict[key][dss.cur_rank]["state_dict"]
            else:
                raise NotImplementedError()

            batch_sampler.load_state_dict(state_dict)
