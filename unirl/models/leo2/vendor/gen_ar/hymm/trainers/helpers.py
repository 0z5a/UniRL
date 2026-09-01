import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.distributed as dist
import numpy as np
import shutil

from hymm.utils.states import DataClassMixin
from hymm.utils.file_utils import safe_dir
from hymm.models.ema.ema import EMA, DistributedEMA


def dynamic_values_wrapper(anchor_sizes, values):
    """
    Dynamically mapping the size (w, h) to one of the list batch sizes.

    Args:
        anchor_sizes (tuple): list of anchor sizes.
        values (tuple): list of values for different image scale.
    """
    if not isinstance(anchor_sizes, (list, tuple)):
        raise ValueError(f"anchor_sizes should be a list or tuple, but got {type(anchor_sizes)}")
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"batch_sizes should be a list or tuple, but got {type(values)}")
    if len(anchor_sizes) != len(values):
        raise ValueError(
            f"anchor_sizes and values should have the same length, but got {len(anchor_sizes)} and " f"{len(values)}."
        )
    anchor_sizes = np.array(anchor_sizes)

    def get_value(size):
        sqrt_area = (size[0] * size[1]) ** 0.5
        val = values[np.argmin(np.abs(anchor_sizes - sqrt_area))]
        return val

    return get_value


@dataclass
class BaseStates(DataClassMixin):
    epoch: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset_epoch_based_states(self):
        pass

    def inc_epoch(self):
        """ Increase (image) epoch by 1. """
        self.epoch += 1
        self.reset_epoch_based_states()
        return self.epoch

    @classmethod
    def from_pretrained(cls, scalar_state, rank=None, world_size=None, default_rank0_ss=None, default=None):
        _scalar_state = default or {}

        def deserialize_and_update(state, val):
            if isinstance(val, dict):
                state.update(val)
            elif isinstance(val, str):
                state.update(json.loads(val))

        if isinstance(scalar_state, (dict, str)):
            deserialize_and_update(_scalar_state, scalar_state)
        elif isinstance(scalar_state, list):
            assert rank is not None and world_size is not None and default_rank0_ss is not None, (
                f"rank({rank}), world_size({world_size}), and default_rank0_ss({default_rank0_ss}) should be provided "
                f"when scalar_state is a list."
            )
            if len(scalar_state) != world_size:
                # Check if scalar_state all the same
                if all([scalar_state[0] == x for x in scalar_state]):
                    default_rank0_ss = True

                if default_rank0_ss:
                    from loguru import logger
                    logger.info(f" {len(scalar_state)=} != {world_size=}, Set default rank0 scalar state.")
                    deserialize_and_update(_scalar_state, scalar_state[0])
                else:
                    raise ValueError(
                        f"Scalar state length {len(scalar_state)} is not equal to "
                        f"world size {world_size}."
                    )
            else:
                deserialize_and_update(_scalar_state, scalar_state[rank])
        else:
            raise ValueError(f"Unknown scalar state type: {type(scalar_state)}")
    
        return cls(**_scalar_state)


@dataclass
class ScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """
    # ==== Image modal ====
    # - Epoch level: Should be reset at the end of every epoch (except the `epoch` field) ====
    epoch: int = 0  # Accumulated training epochs
    epoch_train_steps: int = 0  # Accumulated training steps in current epoch
    epoch_update_steps: int = 0  # Accumulated update steps in current epoch
    epoch_consumed_samples_total: int = 0  # Accumulated consumed samples in current epoch
    epoch_consumed_samples_per_dp: int = 0  # Accumulated consumed samples per data-parallel group in current epoch

    # - Global level
    train_steps: int = 0  # Accumulated training steps
    update_steps: int = 0  # Accumulated update steps
    current_run_update_steps: int = 0  # Update steps in current run (Reset for every resume)
    consumed_samples_total: int = 0  # Accumulated consumed samples
    consumed_tokens_total: int = 0  # Accumulated consumed tokens
    consumed_computations_attn: int = 0  # Accumulated consumed computations of attention + mlp
    consumed_computations_total: int = 0  # Accumulated consumed computations of total

    lr: float = 0.0  # Current learning rate

    def reset_epoch_based_states(self):
        self.epoch_train_steps = 0
        self.epoch_update_steps = 0
        self.epoch_consumed_samples_total = 0
        self.epoch_consumed_samples_per_dp = 0


@dataclass
class TextImageScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    """
    epoch: int = 0

    # - Epoch level
    epoch_train_steps: int = 0
    epoch_update_steps: int = 0
    # - Cumulative level
    train_steps: int = 0
    update_steps: int = 0
    current_run_update_steps: int = 0
    consumed_computations_attn: int = 0
    consumed_computations_total: int = 0

    # ==== Image modal ====
    # - Epoch level: Should be reset at the end of every epoch (except the `epoch` field) ====
    consumed_image_samples_per_dp: int = 0
    epoch_consumed_image_samples: int = 0
    # - Cumulative level
    consumed_image_epoch: int = 0
    consumed_image_samples_total: int = 0
    consumed_image_tokens_total: int = 0

    # ==== Text modal ====
    # - Epoch level
    consumed_text_samples_per_dp: int = 0
    epoch_consumed_text_samples: int = 0
    # - Cumulative level
    consumed_text_epoch: int = 0
    consumed_text_samples_total: int = 0
    consumed_text_tokens_total: int = 0

    lr: float = 0.0  # Current learning rate

    def reset_epoch_based_states(self):
        self.epoch_train_steps = 0
        self.epoch_update_steps = 0
        self.consumed_image_samples_per_dp = 0
        self.epoch_consumed_image_samples = 0
        self.consumed_text_samples_per_dp = 0
        self.epoch_consumed_text_samples = 0


@dataclass
class MultiModalScalarStates(BaseStates):
    """
    Training states for the whole training lifecycle. Should be saved/loaded along with the model checkpoint.
    Notice: Should only use serializable data structures (e.g., dict, list) in this class.
    """
    epoch: int = 0

    # - Epoch level
    epoch_train_steps: int = 0
    epoch_update_steps: int = 0
    # - Cumulative level
    train_steps: int = 0
    update_steps: int = 0
    current_run_update_steps: int = 0  # only used for saving training data, deprecated
    current_forward_times: int = 0  # only used for saving training data
    consumed_computations_attn: int = 0
    consumed_computations_total: int = 0

    # - Epoch level: Should be reset at the end of every epoch (except the `epoch` field) ====
    consumed_samples_per_dp: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consumed_epoch_per_dp: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # The `epoch_consumed_samples` represents the consumed number of samples by the model for each
    # dataset. When using IndexBatchSampler (with indices buffer), it is generally less than the
    # `epoch_required_samples`.
    epoch_consumed_samples: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # The `epoch_required_samples` represents the required number of samples for each dataset
    # from the DistributedSampler in IndexBatchSampler. The required samples should be the max-all-reduced
    # value across all data-parallel ranks within the same dataset key. When resuming training from a
    # checkpoint, if the `epoch_required_samples` is available, it will have higher priority than the
    # `epoch_consumed_samples` to restore the state of the DistributedSampler. If IndexBatchSampler
    # is not used, this field theoretically equals to `epoch_consumed_samples`, so -1 is assigned for
    # simplicity.
    epoch_required_samples: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # A dict[str, [{"state_dict": {...}, ...]} to save the IndexBatchSampler state_dict per dataset.
    # Note that the state_dict is already synchronized across data-parallel ranks with the same
    # dataset key. The `synchronized` means that all data-parallel ranks with the same dataset key
    # have the same `required samples` from the upstream DistributedSampler.
    grouped_batch_sampler_state_dict: Dict[str, Any] = field(default_factory=lambda: {})
    # - Cumulative level
    consumed_epoch: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consumed_samples_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consumed_tokens_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # - Running metrics tracking
    running_loss_dict: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    running_step_dict: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    running_consumed_samples_dict: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    running_consumed_tokens_dict: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    lr: float = 0.0  # Current learning rate

    def __post_init__(self):
        # from_pretrained() will not convert dict to defaultdict. We need to do it manually.
        for key, dtype in [
            ("consumed_samples_per_dp", int),
            ("consumed_epoch_per_dp", int),
            ("epoch_consumed_samples", int),
            ("epoch_required_samples", int),
            ("consumed_epoch", int),
            ("consumed_samples_total", int),
            ("consumed_tokens_total", int),
            ("running_loss_dict", float),
            ("running_step_dict", int),
            ("running_consumed_samples_dict", int),
            ("running_consumed_tokens_dict", int),
        ]:
            if not isinstance(getattr(self, key), defaultdict):
                setattr(self, key, defaultdict(dtype, getattr(self, key)))

        # Sync global epoch to local
        self.assign_global_epoch_states()

    def update(self, loss_dict, consumed_metrics, grad_norm=None, is_update_step=None, lr=None):
        self.train_steps += 1
        if grad_norm is not None:
            self.running_loss_dict["grad_norm"] += grad_norm
            self.running_step_dict["grad_norm"] += 1
        for k, v in loss_dict.items():
            if v is None: # skip None value when use_mot=False
                continue
            if isinstance(v, tuple) and len(v) == 2:
                # (loss_sum, sample_count) for sample-weighted global averaging.
                loss_sum, sample_count = v
                self.running_loss_dict[k] += float(loss_sum)
                self.running_step_dict[k] += float(sample_count)
            else:
                self.running_loss_dict[k] += v.clone().detach().mean().item()
                self.running_step_dict[k] += 1
        for dataset_tag, consumed in consumed_metrics.items():
            self.running_consumed_samples_dict[dataset_tag] += consumed["samples"]
            self.running_consumed_tokens_dict[dataset_tag] += consumed["tokens"]
        
        # We enable `is_update_step` if the current step is the gradient accumulation boundary.
        if is_update_step is not None and is_update_step:
            # A ([forward-backward] x grad_accu)-update step is counted as one update step.
            self.update_steps += 1
            self.current_run_update_steps += 1
            if lr is not None:
                self.lr = lr

    def all_reduce(self, loss_names, data_keys, group):
        # All reduce losses, consumed samples, and consumed tokens
        loss_values = []            # reduce sum
        loss_counts = []            # reduce sum
        consumed_samples = []       # reduce sum
        consumed_tokens = []        # reduce sum
        consumed_epoch = []         # reduce max
        epoch_consumed_samples = [] # reduce max (negate for max reduce)
        for name in loss_names:
            if name in self.running_loss_dict:
                loss_values.append(self.running_loss_dict[name])
                loss_counts.append(float(self.running_step_dict[name]))
            else:
                loss_values.append(0.0)
                loss_counts.append(0.0)
        for name in data_keys:
            consumed_samples.append(self.running_consumed_samples_dict.get(name, 0))
            consumed_tokens.append(self.running_consumed_tokens_dict.get(name, 0))
            consumed_epoch.append(self.consumed_epoch_per_dp.get(name, 0))
            epoch_consumed_samples.append(-self.epoch_consumed_samples.get(name, 0))   # negate for max reduce

        loss_reduced = {}
        if len(loss_names) == 0 and len(data_keys) == 0:
            return loss_reduced

        # -- reduce sum
        sum_reduced = torch.tensor(
            loss_values + loss_counts + consumed_samples + consumed_tokens, device='cuda', dtype=torch.float64)
        torch.distributed.all_reduce(sum_reduced, group=group)
        sum_losses, sum_counts, sum_samples, sum_tokens = torch.split(
            sum_reduced, [len(loss_names), len(loss_names), len(data_keys), len(data_keys)]
        )
        sum_counts = sum_counts.clamp(min=1.0)  # avoid division by zero
        for name, log_loss, count in zip(loss_names, sum_losses, sum_counts):
            if log_loss != 0:
                loss_reduced[name] = (log_loss / count).item()

        if len(data_keys) == 0:
            return loss_reduced

        # -- reduce max and assign consumed epoch and epoch consumed samples
        max_reduced = torch.tensor(consumed_epoch + epoch_consumed_samples, device='cuda', dtype=torch.int64)
        torch.distributed.all_reduce(max_reduced, op=torch.distributed.ReduceOp.MAX, group=group)
        max_epochs, max_neg_samples = torch.split(max_reduced, [len(data_keys), len(data_keys)])
        min_samples = -max_neg_samples
        for name, max_epoch, min_sample in zip(data_keys, max_epochs, min_samples):
            self.consumed_epoch[name] = max_epoch.item()
            self.epoch_consumed_samples[name] = min_sample.item()

        # Assign consumed samples and tokens
        for name, sample_count, token_count in zip(data_keys, sum_samples.tolist(), sum_tokens.tolist()):
            self.epoch_consumed_samples[name] += int(sample_count)
            self.consumed_samples_total[name] += int(sample_count)
            self.consumed_tokens_total[name] += int(token_count)

        return loss_reduced

    def reset_epoch_based_states(self):
        self.epoch_train_steps = 0
        self.epoch_update_steps = 0
        self.consumed_samples_per_dp = defaultdict(int)
        self.epoch_required_samples = defaultdict(int)
        self.epoch_consumed_samples = defaultdict(int)

    def reset_running_states(self):
        self.running_loss_dict = defaultdict(float)
        self.running_step_dict = defaultdict(int)
        self.running_consumed_samples_dict = defaultdict(int)
        self.running_consumed_tokens_dict = defaultdict(int)

    def assign_global_epoch_states(self):
        # Call this function just after resuming from a checkpoint
        for key in self.consumed_epoch.keys():
            self.consumed_epoch_per_dp[key] = self.consumed_epoch[key]


@dataclass
class MultiModalGRPOScalarStates(MultiModalScalarStates):
    """
    MultiModalScalarStates with GRPO training states.
    Inherits from MultiModalScalarStates and adds GRPO-specific fields for saving/loading.
    """
    # GRPO training states
    grpo_cur_timestep: int = 0
    grpo_cur_iter_in_group: int = 0


@dataclass
class CycleStates:
    """
    Training states for every logging interval. Should be reset at the end of every logging interval.
    """
    log_steps: int = 0
    running_loss: float = 0
    running_sub_loss_dict: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    running_sub_step_dict: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    running_tokens: int = 0
    running_samples: int = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, getattr(self, k) + v)

    def reset(self):
        self.log_steps = 0
        self.running_loss = 0
        self.running_sub_loss_dict = defaultdict(float)
        self.running_sub_step_dict = defaultdict(int)
        self.running_tokens = 0
        self.running_samples = 0


@dataclass
class TextImageCycleStates(CycleStates):
    # ==== Image modal ====
    running_image_tokens: int = 0
    running_image_samples: int = 0

    # ==== Text modal ====
    running_text_tokens: int = 0
    running_text_samples: int = 0

    def reset(self):
        super().reset()
        self.running_image_tokens = 0
        self.running_image_samples = 0
        # ==== Text modal ====
        self.running_text_tokens = 0
        self.running_text_samples = 0


@dataclass
class MultiModalCycleStates(CycleStates):
    running_tokens: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    running_samples: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def reset(self):
        super().reset()
        self.running_tokens = defaultdict(int)
        self.running_samples = defaultdict(int)


def save_checkpoint(
        config,
        rank: int,
        logger,
        model_engine,
        ema,
        scalar_state: ScalarStates,
        ckpt_dir: Path,
        serialize_scalar_state: bool = False,
):
    _ = rank  # Currently not used.
    if serialize_scalar_state:
        scalar_state_dict = scalar_state.serialize()
    else:
        scalar_state_dict = scalar_state.to_dict()
    # sync scalar_state
    if dist.is_available() and dist.is_initialized():
        gather_results_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(gather_results_list, scalar_state_dict)
        scalar_state_dict = gather_results_list

    client_state = {
        "config": config,
        "scalar_state": scalar_state_dict,
    }

    def try_save(_save_name):
        checkpoint_path = ckpt_dir / _save_name
        try:
            model_engine.save_checkpoint(
                str(ckpt_dir),
                client_state=client_state,
                tag=_save_name,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            if config.launcher == 'pure_torch':
                raise # 提前暴露问题
            return None

    def try_save_torch(_save_name):
        checkpoint_path = ckpt_dir / _save_name
        try:
            if hasattr(model_engine, 'full_state_dict'):
                sd = model_engine.full_state_dict()
            else:
                sd = model_engine.state_dict()
            checkpoint = {
                "module": sd,
            }
            checkpoint.update(client_state)
            if rank == 0:
                torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()
            return checkpoint_path
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            return None
    
    def try_save_dist_ema(_save_path, _rank):
        try:
            safe_dir(_save_path)
            torch.save({
                "ema": ema.state_dict(),
                "ema_config": ema.config,
            }, _save_path / f"{_rank}.pt")
            logger.info(f"Saved Distributed EMA to {_save_path}")
            return _save_path
        except Exception as e:
            logger.error(f"Failed to save Distributed EMA to {_save_path}. {type(e)}: {e}")
            return None

    update_steps = scalar_state.update_steps

    if ema is not None:
        # if distributed EMA, save it distributedly
        if isinstance(ema, DistributedEMA):
            dist_ema_save_path = ckpt_dir / f"{update_steps:07d}" / "dist_ema"
            try_save_dist_ema(dist_ema_save_path, rank)
        # if regular EMA, save it in client_state
        elif isinstance(ema, EMA):
            client_state["ema"] = ema.state_dict()
            client_state["ema_config"] = ema.config
        else:
            raise ValueError(f"Unsupported EMA type: {type(ema)}")

    if config.launcher == 'pure_torch':
        save_name = f"{update_steps:07d}"
        save_path = try_save(save_name)
    elif config.launcher == "deepspeed":
        save_name = f"{update_steps:07d}"
        save_path = try_save(save_name)
    elif config.launcher == "torch":
        save_name = f"ckpt_{update_steps:07d}.pt"
        save_path = try_save_torch(save_name)
    else:
        raise ValueError(f"Unsupported launcher: {config.launcher}")

    return [save_path]


def check_disk_space(path, required_gb=1):
    """Check if there's enough disk space in the given path."""
    
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (2**30)  # Convert bytes to GB
        # print(f"Disk space at {path}: total={total}, used={used}, free={free} ({free_gb} GB)")
        return free_gb >= required_gb
    except Exception as e:
        print(f"Failed to check disk space for {path}. {type(e)}: {e}")
        return False

def wait_for_disk_space(path, required_gb=1, max_attempts=600000, sleep_time=60):
    """Wait until there's enough disk space available."""
    import time
    
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
    for attempt in range(max_attempts):
        if check_disk_space(path, required_gb):
            return True
            
        if attempt < max_attempts - 1:
            print(f"Waiting for disk space to be available at {path}. Attempt {attempt + 1} of {max_attempts}")
            time.sleep(sleep_time)
            
    return False

def get_trainable_params(model, training_parts):
    if training_parts is None:
        params = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    elif hasattr(model, 'get_training_parts'):
        params = model.get_training_parts(training_parts)
    else:
        raise NotImplementedError()
    return params


class WarmupCosineHelper(object):
    """
    Support resume with changed total_num_steps.
    """
    def __init__(self, lr, warmup_num_steps, cos_min_ratio, start_lr, start_step, total_num_steps):
        self.lr = lr
        self.warmup_num_steps = warmup_num_steps
        self.cos_min_ratio = cos_min_ratio
        self.start_lr = start_lr
        self.start_step = start_step
        self.total_num_steps = total_num_steps

        self.xi = self.get_xi()

    def warmup_cos_lr(self, x):
        real_last_step = x - self.warmup_num_steps + 1
        real_total_steps = self.total_num_steps - self.warmup_num_steps
        ratio_delta = 1. - self.cos_min_ratio
        ratio = (1 + np.cos(np.pi * real_last_step / real_total_steps)) / 2
        ratio = np.maximum(0.0, self.cos_min_ratio + ratio_delta * ratio)
        return ratio * self.lr

    def get_xi(self):
        xs = np.arange(self.total_num_steps).astype(float)
        ys = self.warmup_cos_lr(xs[self.warmup_num_steps:])
        xi = np.argmin((self.start_lr - ys) ** 2) + self.warmup_num_steps
        return xi

    def __call__(self, step):
        if step < self.warmup_num_steps:
            return step
        n = self.total_num_steps
        return n - (n - step) * (n - self.xi) / (n - self.start_step)


@dataclass
class GRPOTrainingStates:
    """Parameters for grpo training strategy.
    
    This class manages the parameters and state for grpo training, where
    training is done in groups of timesteps.
    
    Attributes:
        iters_per_group (int): Number of iterations to train on each group of timesteps
        group_size (int): Number of timesteps in each group
        max_timesteps (int): Maximum number of timesteps to train on
        cur_timestep (int): Current timestep being trained on
        cur_iter_in_group (int): Current iteration within the current group
        sample_strategy (str): Strategy for sampling timesteps ("progressive", "random", "decay", or "dynamic")
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        max_iters_per_group (int): Maximum number of iterations per group for decay strategy
        min_iters_per_group (int): Minimum number of iterations per group for decay strategy
        dynamic_t1 (int): Threshold timestep for dynamic strategy, default=12
        dynamic_k (float): Decay rate for dynamic strategy, default=0.5
        dynamic_y0 (int): Initial group size for dynamic strategy, default=5
    """
    iters_per_group: int
    group_size: int
    max_timesteps: int
    cur_timestep: int = 0
    cur_iter_in_group: int = 0
    sample_strategy: str = "progressive"
    stride: int = 1
    overlap: bool = False
    max_iters_per_group: int = None
    min_iters_per_group: int = None
    dynamic_t1: int = 12
    dynamic_k: float = 0.5
    dynamic_y0: int = 5

    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
    
    def __post_init__(self):
        if self.sample_strategy == "decay":
            if self.max_iters_per_group is None:
                self.max_iters_per_group = self.iters_per_group
            if self.min_iters_per_group is None:
                self.min_iters_per_group = max(1, self.iters_per_group // 4)
    
    def get_dynamic_iters_per_group(self) -> int:
        """Calculate the dynamic number of iterations per group based on current timestep.
        
        Returns:
            int: Number of iterations for the current group.
        """
        if self.sample_strategy == "decay":
            # Linear interpolation between max_iters_per_group and min_iters_per_group
            progress = self.cur_timestep / self.max_timesteps
            current_iters = int(self.max_iters_per_group * (1 - progress) + self.min_iters_per_group * progress)
            return max(self.min_iters_per_group, current_iters)
        elif self.sample_strategy == "dynamic":
            # Calculate group size using exponential decay formula: y(t) = y_0 * exp(-k * ReLU(t-t_1))
            t = self.cur_timestep
            relu_term = max(0, t - self.dynamic_t1)
            group_size = int(self.dynamic_y0 * np.exp(-self.dynamic_k * relu_term))
            return max(1, group_size)  # Ensure group size is at least 1
        else:
            return self.iters_per_group
    
    def _update_progressive(self) -> None:
        """Update iteration for progressive strategy."""
        self.cur_iter_in_group += 1
        if self.cur_iter_in_group >= self.iters_per_group:
            self.cur_iter_in_group = 0
            if self.overlap:
                self.cur_timestep += self.stride
            else:
                self.cur_timestep += self.group_size
        self._clip_cur_timestep()

    def _update_random(self, seed: int = None) -> None:
        """Update iteration for random strategy."""
        rng = np.random.default_rng(seed)
        self.cur_timestep = rng.integers(0, self.max_timesteps - self.group_size + 1)

    def _update_dynamic_strategies(self) -> None:
        """Update iteration for decay and dynamic strategies."""
        self.cur_iter_in_group += 1
        current_iters = self.get_dynamic_iters_per_group()
        if self.cur_iter_in_group >= current_iters:
            self.cur_iter_in_group = 0
            if self.overlap:
                self.cur_timestep += self.stride
            else:
                self.cur_timestep += self.group_size
        self._clip_cur_timestep()

    def _clip_cur_timestep(self) -> None:
        """Clip cur_timestep to max_timesteps, if it exceeds (max_timesteps - group_size), reset to 0."""
        if self.cur_timestep >= self.max_timesteps - self.group_size + 1:
            self.cur_timestep = 0

    def update_iteration(self, seed: int = None) -> None:
        """Update the current iteration counter and timestep if needed.
        
        Args:
            seed (int, optional): Random seed for random strategy. Defaults to None.
        """
        strategy_handlers = {
            "progressive": self._update_progressive,
            "random": lambda: self._update_random(seed),
            "decay": self._update_dynamic_strategies,
            "dynamic": self._update_dynamic_strategies,
        }

        handler = strategy_handlers.get(self.sample_strategy)
        if handler is None:
            raise ValueError(f"Invalid sample strategy: {self.sample_strategy}")
        
        handler()

    def get_current_timesteps(self) -> List[int]:
        """Get the list of timesteps to train on in the current group.
        
        Returns:
            List[int]: List of timesteps to train on. 
            For example, if cur_timestep=5 and group_size=2, returns [5, 6].
        """
        return list(range(self.cur_timestep, min(self.cur_timestep + self.group_size, self.max_timesteps)))

    def restore_from_scalar_states(self, scalar_states):
        """Restore GRPO states from MultiModalGRPOScalarStates.
        
        Args:
            scalar_states: MultiModalGRPOScalarStates instance containing GRPO state
        """
        if hasattr(scalar_states, 'grpo_cur_timestep'):
            self.cur_timestep = scalar_states.grpo_cur_timestep
        if hasattr(scalar_states, 'grpo_cur_iter_in_group'):
            self.cur_iter_in_group = scalar_states.grpo_cur_iter_in_group

    def update_scalar_states(self, scalar_states):
        """Update MultiModalGRPOScalarStates with current GRPO state.
        
        Args:
            scalar_states: MultiModalGRPOScalarStates instance to update
        """
        scalar_states.grpo_cur_timestep = self.cur_timestep
        scalar_states.grpo_cur_iter_in_group = self.cur_iter_in_group


if __name__ == "__main__":
    grpo_states = GRPOTrainingStates(
        iters_per_group=5,
        group_size=2,
        max_timesteps=40,
        sample_strategy="dynamic",
        dynamic_t1=15,
        dynamic_k=0.5,
        dynamic_y0=25,
    )

    for i in range(10, 40):
        grpo_states.set_params({
            "cur_timestep": i, 
        })
        print(f"cur_timestep: {i}, group_size: {grpo_states.get_dynamic_iters_per_group()}")
