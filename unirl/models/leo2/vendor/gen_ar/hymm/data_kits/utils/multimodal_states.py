import bisect
import fnmatch
import math
import json
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from ...core.parallel_states import ParallelState


@dataclass
class DistributedSamplingState:
    # Sampling mode: "random" or "fixed"
    # - random: Each rank build all the datasets and randomly samples a dataset to get_batch
    #           with the given probability.
    # - fixed: Each dataset is assigned to fixed ranks, following the sampling_probs as near as possible.
    #          In this mode, each rank only builds the assigned dataset.
    mode: str

    # =========================
    #   Common for both modes
    # =========================
    # Number of replicas for each dataset
    dataset_num_replicas: dict[str, int] = field(default_factory=dict)
    # Rank of each dataset in the current process. It should be in [0, dataset_num_replicas[key]-1]
    dataset_rank: dict[str, int] = field(default_factory=dict)

    # =========================
    #     For "fixed" mode
    # =========================
    # The assigned dataset key of current rank.
    cur_key: str = None
    # The list of ranks that are assigned to the same dataset with current rank.
    cur_key_group: list[int] = None
    # The dp rank of current process within the assigned dataset.
    cur_rank: int = 0
    # The size of the assigned dataset (number of ranks assigned to the same dataset).
    cur_size: int = 1


@dataclass
class MultimodalTasksState:
    sampling_probs: dict[str, float]
    all_dataset_keys: list[str]
    keys2id: dict[str, int]
    # =========================
    # distributed datasets, samplers, and dataloaders
    distributed_sampling_state: Optional[DistributedSamplingState] = None

    @staticmethod
    def from_args(args):
        sampling_probs_dict = json.loads(args.sampling_probs)
        # The id should be consistent with the order of sampling_probs_dict.
        all_dataset_keys = list(sampling_probs_dict.keys())
        keys2id = {key: i for i, key in enumerate(all_dataset_keys)}
        print(f"Dataset keys-id mapping: {keys2id}")

        return MultimodalTasksState(
            sampling_probs=sampling_probs_dict,
            all_dataset_keys=all_dataset_keys,
            keys2id=keys2id,
        )


def allocate_gpus(weights, N):
    count = len(weights)
    if N < count:
        raise ValueError("N must be greater than or equal to the number of weights")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Weights sum is 0")
    else:
        norm_weights = [w / total_weight for w in weights]

    allocated = [1] * count
    remaining_N = N - count

    if remaining_N == 0:
        return allocated

    targets = [w * remaining_N for w in norm_weights]

    integer_parts = [math.floor(t) for t in targets]

    remainders = [t - i for t, i in zip(targets, integer_parts)]
    
    for i in range(count):
        allocated[i] += integer_parts[i]
    
    current_sum = sum(allocated)
    leftover = N - current_sum

    indexed_remainders = sorted(enumerate(remainders), key=lambda x: x[1], reverse=True)

    for i in range(leftover):
        idx, _ = indexed_remainders[i]
        allocated[idx] += 1

    return allocated


def fixed_dataloader_allocation(weights, keys, rank, world_size, _logger=None):
    """
    Calculate the dataloader allocation for each rank in fixed sampling mode.

    Args:
        weights (list): A list of weights for each dataset.
        keys (list): A list of dataset keys.
        rank (int): The dp rank of the current process.
        world_size (int): The total number of dp processes.
        _logger: A logger instance.

    Returns:
        index (int): The index of the dataset assigned to the current rank.
        rank_bias (int): The starting dp rank of the assigned dataset.
        num_replicas_per_key (list): A list of number of replicas for each dataset.
        cum_ranks (list): A list of cumulative ranks for each dataset.

    Examples:
        >>> weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        >>> keys = ['t2i', 't2i_long_long', 'lm', 'mmu_caption', 'mmu']
        >>> dp_size = 8
        >>> for dp_rank in range(dp_size):
        ...     index, rank_bias, num_replicas_per_key, cum_ranks = fixed_dataloader_allocation(weights, keys, dp_rank, dp_size)
        ...     print(f"Rank {dp_rank}: Rank bias: {rank_bias}, Dataset index {index} ({keys[index]})"
        Rank 0: Rank bias: 0, Dataset index 0 (t2i)
        Rank 1: Rank bias: 0, Dataset index 0 (t2i)
        Rank 2: Rank bias: 2, Dataset index 1 (t2i_long_long)
        Rank 3: Rank bias: 2, Dataset index 1 (t2i_long_long)
        Rank 4: Rank bias: 4, Dataset index 2 (lm)
        Rank 5: Rank bias: 4, Dataset index 2 (lm)
        Rank 6: Rank bias: 6, Dataset index 3 (mmu_caption)
        Rank 7: Rank bias: 7, Dataset index 4 (mmu)
        >>> print(f"{num_replicas_per_key=}")
        num_replicas_per_key=[2, 2, 2, 1, 1]
        >>> print(f"{cum_ranks=}")
        cum_ranks=[0, 2, 4, 6, 7, 8]
    """
    assert len(weights) == len(keys), \
        f"Length of weights and keys should be the same, but got {len(weights)} != {len(keys)}"
    if _logger is None:
        _logger = logger
    # Fixed sampling mode. Each rank is assigned to a fixed dataset.
    num_replicas_per_key = allocate_gpus(weights, world_size)
    # Determine the dataset for the current rank
    cum_ranks = [0] + list(accumulate(num_replicas_per_key))
    if rank == 0:
        dataset_dispatch_repr = ""
        key_max_length = max([len(key) for key in keys])
        for i, weight in enumerate(weights):
            dataset_dispatch_repr += (
                f"\n    {i}. {keys[i]:<{key_max_length}} ({weight:.2f}): Replica {cum_ranks[i]} ~ {cum_ranks[i + 1] - 1}"
            )
        _logger.info(
            f"In fixed distributed sampling mode:"
            f"\n    Number of replicas per key: {num_replicas_per_key}"
            f"\n    Dataset dispatch: {dataset_dispatch_repr}"
        )
    index = bisect.bisect_right(cum_ranks, rank) - 1
    rank_bias = cum_ranks[index]
    return index, rank_bias, num_replicas_per_key, cum_ranks


def prepare_distributed_sampling(args, mm_state: MultimodalTasksState, p_state: "ParallelState", _logger=None):
    sampling_probs = mm_state.sampling_probs
    assert isinstance(sampling_probs, dict), f"`sampling_probs` must be a dict, but got {type(sampling_probs)}"
    # Remove zero weight datasets
    sampling_probs = {key: prob for key, prob in sampling_probs.items() if prob > 0}
    dataset_keys = list(sampling_probs.keys())

    # Determine the dataloader allocation if combined_iterator_sampling_mode is not `random`.
    if args.combined_iterator_sampling_mode == 'random':
        mm_state.distributed_sampling_state = DistributedSamplingState(
            mode='random',
            dataset_num_replicas={key: p_state.dp_size for key in dataset_keys},
            dataset_rank={key: p_state.dp_rank for key in dataset_keys},
        )

    elif args.combined_iterator_sampling_mode == 'fixed':
        # Dataloaders are assigned to fixed ranks, following the sampling_probs as near as possible.
        # For example, if sampling_probs = {t2i: 0.4, lm: 0.2, mmu: 0.4}, and dp_size = 32, ranks 0-12
        # will be assigned to t2i, ranks 13-18 to lm, and ranks 19-31 to mmu. In the other side,
        # all the samples in the t2i dataset will be evently distributed among the ranks 0-12, and so on.

        # Make sure the dp_size is greater than or equal to the number of valid datasets.
        assert len(dataset_keys) <= p_state.dp_size, (
            f"Number of dp_size({p_state.dp_size}) should be greater than or equal to "
            f"valid datasets ({len(dataset_keys)}, {dataset_keys}) in fixed mode. "
            f"Try to increase number of GPUs, or remove some datasets."
        )
        weights = [sampling_probs[key] for key in dataset_keys]
        key_index, rank_bias, num_replicas, cum_ranks = \
            fixed_dataloader_allocation(
                weights=weights, keys=dataset_keys, rank=p_state.dp_rank, world_size=p_state.dp_size, _logger=_logger)

        mm_state.distributed_sampling_state = DistributedSamplingState(
            mode='fixed',
            dataset_num_replicas={key: num_replicas[i] for i, key in enumerate(dataset_keys)},
            dataset_rank={dataset_keys[key_index]: p_state.dp_rank - rank_bias},
            cur_key=dataset_keys[key_index],
            cur_key_group=list(range(cum_ranks[key_index], cum_ranks[key_index + 1])),
            cur_rank=p_state.dp_rank - rank_bias,
            cur_size=num_replicas[key_index],
        )

    else:
        raise ValueError(f"Invalid combined_iterator_sampling_mode: {args.combined_iterator_sampling_mode}. "
                         f"Valid values are ['random', 'fixed'].")
    return mm_state
