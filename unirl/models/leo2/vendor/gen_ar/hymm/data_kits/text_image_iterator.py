import random
from typing import Optional

import torch
import torch.distributed as dist


class TextImageBatchIterator(object):
    def __init__(self,
                 ss,
                 fast_shuffle: bool,
                 rank: int, # is used to set up seed for data shuffling
                 world_size: int, # should not be used.
                 text_dataset,
                 text_sampler,
                 text_dataloader,
                 image_dataset,
                 image_sampler,
                 image_dataloader,
                 text_sampling_prob,
                 initial_seed,
                 logger=None,
                 ):
        self.ss = ss
        self.fast_shuffle = fast_shuffle
        self.rank = rank

        # To avoid making further mistake, this line is commented.
        # __init__ signature is kept to avoid breaking existing code.
        # self.world_size = world_size

        self.text_dataset = text_dataset
        self.text_sampler = text_sampler
        self.text_dataloader = text_dataloader
        self.image_dataset = image_dataset
        self.image_sampler = image_sampler
        self.image_dataloader = image_dataloader

        self.text_sampling_prob = text_sampling_prob
        self.initial_seed = initial_seed
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.text_iterator = self.get_text_iterator()
        self.image_iterator = self.get_image_iterator()

        # Prefetch the first batch of all dataloaders to ensure all dataloaders are ready at the beginning of training.
        self.first_flag = True
        self.first_text_cache = None
        self.first_image_cache = None

        # shuffle before first get
        self.shuffle_and_initialize_index()

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_flag:
            self.first_text_cache = self.get_text_batch()
            self.first_image_cache = self.get_image_batch()
            self.first_flag = False

        emit_text_shuffle = False
        emit_image_shuffle = False
        if random.random() < self.text_sampling_prob:
            if self.first_text_cache is not None:
                batch, emit_text_shuffle = self.first_text_cache
                self.first_text_cache = None
            else:
                batch, emit_text_shuffle = self.get_text_batch()
        else:
            if self.first_image_cache is not None:
                batch, emit_image_shuffle = self.first_image_cache
                self.first_image_cache = None
            else:
                batch, emit_image_shuffle = self.get_image_batch()

        # Globally synchronize shuffle flags
        shuffle_flags = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(shuffle_flags, (emit_text_shuffle, emit_image_shuffle))

        # Emit shuffle if any worker emits shuffle.
        if not emit_text_shuffle and any([flag[0] for flag in shuffle_flags]):
            self.shuffle_text()
        if not emit_image_shuffle and any([flag[1] for flag in shuffle_flags]):
            self.shuffle_image()

        return batch

    def __len__(self):
        raise NotImplementedError("Length of TextImageBatchIterator is undefined.")

    def shuffle_and_initialize_index(self):
        # == text dataset == Each rank should have different shuffle seed
        text_seed = self.initial_seed + self.ss.consumed_text_epoch + self.rank * 10000
        self.logger.info(
            f"[Rank{self.rank}] Shuffle text dataset with seed={text_seed}, fast={self.fast_shuffle}. "
            f"Text dataset is entirely assigned to every worker, thereby using different seeds. "
            f"Initialize with {self.ss.consumed_text_samples_per_dp=}")
        self.shuffle_and_reset_index(
            self.text_dataset, self.text_sampler,
            seed=text_seed,
            start_index=self.ss.consumed_text_samples_per_dp
        )

        # == image dataset == Makesure all processors use the same seed to shuffle dataset.
        image_seed = self.initial_seed + self.ss.consumed_image_epoch
        self.logger.info(
            f"Shuffle image dataset with seed={image_seed}, fast={self.fast_shuffle}. "
            f"Image dataset is split by all workers, thereby using the same seeds. "
            f"Initialize with {self.ss.consumed_image_samples_per_dp=}")
        self.shuffle_and_reset_index(
            self.image_dataset, self.image_sampler,
            seed=image_seed,
            start_index=self.ss.consumed_image_samples_per_dp,
        )
        self.logger.info(f"End of random shuffle")

    def shuffle_text(self):
        self.ss.consumed_text_epoch += 1
        self.logger.info(f"Text dataset epoch: {self.ss.consumed_text_epoch}")

        self.ss.consumed_text_samples_per_dp = 0
        self.shuffle_and_reset_index(
            self.text_dataset, self.text_sampler, self.initial_seed + self.ss.consumed_text_epoch + self.rank * 10000,
            self.ss.consumed_text_samples_per_dp
        )
        self.text_iterator = self.get_text_iterator()

    def shuffle_image(self):
        self.ss.consumed_image_epoch += 1
        self.logger.info(f"Image dataset epoch: {self.ss.consumed_image_epoch}")

        self.ss.consumed_image_samples_per_dp = 0
        self.shuffle_and_reset_index(
            self.image_dataset, self.image_sampler, self.initial_seed + self.ss.consumed_image_epoch,
            self.ss.consumed_image_samples_per_dp
        )
        self.image_iterator = self.get_image_iterator()

    def get_text_batch(self):
        try:
            batch = next(self.text_iterator)
            emit_global_shuffle = False
        except StopIteration:
            self.shuffle_text()
            batch = next(self.text_iterator)
            emit_global_shuffle = True
        return batch, emit_global_shuffle

    def get_image_batch(self):
        try:
            batch = next(self.image_iterator)
            emit_global_shuffle = False
        except StopIteration:
            self.shuffle_image()
            batch = next(self.image_iterator)
            emit_global_shuffle = True
        return batch, emit_global_shuffle

    def get_text_iterator(self):
        for idx, batch in enumerate(self.text_dataloader):
            self.ss.consumed_text_samples_per_dp += batch["n_samples"].sum().item()
            yield batch

    def get_image_iterator(self):
        for idx, batch in enumerate(self.image_dataloader):
            self.ss.consumed_image_samples_per_dp += batch["tokens"].shape[0]
            yield batch

    def shuffle_and_reset_index(self, dataset, sampler, seed, start_index):
        # Log shuffle information of all ranks for ensuring the wanted shuffle order.
        print(f"[Rank {self.rank}] Shuffle {dataset.__class__.__name__} with {seed=}")
        dataset.shuffle(seed, fast=self.fast_shuffle)
        sampler.start_index = start_index
        self.logger.info(f"Update {sampler.__class__.__name__} states: {start_index=}, {len(sampler)=:,}")
