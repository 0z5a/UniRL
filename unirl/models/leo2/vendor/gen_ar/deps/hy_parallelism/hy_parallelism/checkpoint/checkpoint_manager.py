# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import enum
import gc
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from typing import Type

import loguru
from loguru import logger
from packaging import version

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor, distribute_tensor
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._state_dict_utils import (
    _copy_state_dict,
    _create_cpu_state_dict,
)
from torch.distributed.checkpoint.format_utils import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hy_parallelism.checkpoint.stateful import (
    OptimizersContainer,
    LRSchedulersContainer,
    ModelWrapper,
    EPModelWrapper,
    wrap_model_to_stateful,
)


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

T = TypeVar("T", bound=Optimizer)

import copy
import functools
from typing import Any, Callable, Iterator

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        gc.disable()
        self.collect("Initial GC collection.")

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Peforming periodical GC collection.")

    @staticmethod
    def collect(reason: str):
        # begin = time.monotonic()
        gc.collect(1)
        # logger.info("[GC] %s %.2f seconds." % (reason, time.monotonic() - begin))


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    pass


@torch.no_grad()
def save_with_gc(state, checkpoint_id, dcp_save_kwargs: dict | None = None):
    if dcp_save_kwargs is None:
        dcp_save_kwargs = {}
    torch.cuda.empty_cache()
    dcp.save(state, checkpoint_id=checkpoint_id, **dcp_save_kwargs)
    torch.cuda.empty_cache()
    GarbageCollection.collect("GC collection invoked by checkpointer.")


def checkpoint_mp(recv: mp.Queue, send: mp.Queue):
    """Process to save the checkpoint in the background.

    This is only used when async_checkpoint_with_pinned_memory is enabled.

    Args:
        recv (mp.Queue): The queue to receive the state_dict and Terminate signal.
        send (mp.Queue): The queue to send the SaveDone signal.
    """
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id, dcp_save_kwargs = obj
            save_with_gc(state, checkpoint_id=checkpoint_id, dcp_save_kwargs=dcp_save_kwargs)
            logger.info(
                "Finish saving the checkpoint in the background process in %.2f seconds.",
                time.monotonic() - begin,
                )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info(f"Checkpointer is deleting {path}.")
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Checkpointer {path} deleted in {time.monotonic() - begin:.2f} seconds.", )
    finally:
        # logger.info("Destroying the purge thread.")
        pass

@dataclass
class Checkpoint:
    enable_checkpoint: bool = False
    """Whether to enable checkpoint"""
    dump_folder: str = 'dump_folder'
    folder: str = "checkpoint"
    """
    The folder to store the checkpoints.
    When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.
    """

    interval: int = 500
    """Checkpointing interval in steps."""

    model_weights_only: bool = False
    """
    When model_weights_only=True, only model weights will be saved at the end of training.
    With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion.
    When model_weights_only=False, the full checkpoint will be saved.
    A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
    The default value is false.
    """

    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    """
    Converts to the specified precision when training completes and model_weights_only=true.
    """

    create_seed_checkpoint: bool = False
    """
    Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
    Could be implemented as a separate script, but this way shares more code.
    """

    async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
    """
    Which async checkpoint mode to use. Currently there are 3 different modes.
    - "disabled": synchronized checkpointing will be used.
    - "async": torch.distributed.checkpoint.async_save will be used.
    - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
      separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
      The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
      may degrade due to memory paging. For most users, "async" should suffice as the performance
      overhead is typically small (on the order of tens of seconds) compared to checkpointing
      frequency. This mode can be employed to pursue near-zero checkpointing times
      (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

    "disabled" is the default mode.
    """

    keep_latest_k: int = 0
    """
    Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
    K cannot be 1 as the last one may be in the process of being saved. As a result,
    the metadata of the last one may not be ready yet. The default value is 10 to avoid
    filling up the disk.
    """

    load_step: int = -1
    """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

    exclude_from_loading: list[str] = field(default_factory=list)
    """
    Exclude specific keys from being loaded from the checkpoint.
    Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
    This will load the model only, excluding the specified keys.
    """

class CheckpointManager:
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is gauranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Note: TorchFT checkpointing flow

    There are two types of checkpoints: when TorchFT is enabled: 1) the full perisistent
    checkpoint, 2) the per-replica checkpoint.

    The full perisistent checkpoint is saved by the replica with
    ``ft_manager.participating_rank() == 0``. It contains everything including the model,
    optimizer, lr_scheduler, dataloader, and train_state. Right now the full perisistent
    checkpoint is loaded by all replicas. However, we can optimize it to only load if
    there are no other alive replicas.

    The per-replica checkpoint contains only the dataloader and is saved/loaded by all
    replicas to/from the its own folder. The folder name is prefixed with the ft_replica_id.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        job_config (JobConfig): The job config used to configure the checkpointing.
        ft_manager (Optional[ft.Manager]): The FTManager from TorchFT.
    """

    def __init__(
            self,
            # dataloader: DataLoader,
            states: dict[str, Any],
            ckpt_config: Checkpoint,
            model_parts: list[nn.Module]=None,
            optimizers: OptimizersContainer=None,
            lr_schedulers: LRSchedulersContainer=None,
            stateful_class: Type[Stateful]=None,
    ) -> None:
        self.ckpt_config = ckpt_config
        self.enable_checkpoint = ckpt_config.enable_checkpoint

        async_mode = ckpt_config.async_mode.lower()
        self.enable_staging = (
                self.enable_checkpoint and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        if not self.enable_checkpoint:
            return

        self.states = states

        if model_parts is not None:
            self.states.update(
                {
                    MODEL: wrap_model_to_stateful(model_parts, stateful_class),
                    # DATALOADER: dataloader,
                    # LR_SCHEDULER: lr_schedulers,
                }
            )
        if optimizers:
            assert isinstance(optimizers, OptimizersContainer)
            self.states.update({OPTIMIZER: optimizers,})

        if lr_schedulers:
            assert isinstance(lr_schedulers, LRSchedulersContainer)
            self.states.update({LR_SCHEDULER: lr_schedulers,})

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.staging_stream = torch.cuda.Stream() if self.enable_staging else None

        # self.folder = os.path.join(ckpt_config.dump_folder, ckpt_config.folder)
        self.folder = ckpt_config.dump_folder
        self.interval = ckpt_config.interval
        async_mode = ckpt_config.async_mode.lower()
        if async_mode == AsyncMode.ASYNC:
            self.pg = dist.new_group(backend="gloo")

        self.keep_latest_k = ckpt_config.keep_latest_k
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()
        else:
            self.purge_thread = None

        self.model_weights_only = ckpt_config.model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[ckpt_config.export_dtype]
        self.exclude_from_loading = ckpt_config.exclude_from_loading

        self.mp = None
        self.async_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
            ctx = mp.get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
        else:
            raise ValueError(f"Unkown checkpoint async_mode {ckpt_config.async_mode}")

        # logger.info(
        #     f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        # )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable_checkpoint") and self.enable_checkpoint:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                    hasattr(self, "purge_thread")
                    and self.purge_thread
                    and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

    @torch.no_grad()
    def save(self, step_or_tag: Union[int, str], force: bool = True, dcp_save_kwargs: dict | None = None) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``force`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            force (bool, optional): Whether to force save the checkpoint. Defaults to False.

        Returns:
            None
        """

        # if not self._should_save(curr_step, force):
        #     return

        if dcp_save_kwargs is None:
            dcp_save_kwargs = {}

        begin = time.monotonic()
        # logger.info("Saving the checkpoint (or staging if async is enabled).")
        checkpoint_id = self._create_checkpoint_id(step_or_tag)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if force:
            self._save_last_step(step_or_tag, dcp_save_kwargs=dcp_save_kwargs)
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self._async_with_pinned_memory(checkpoint_id, dcp_save_kwargs=dcp_save_kwargs)
        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.async_future = dcp.async_save(self.states, checkpoint_id=checkpoint_id, process_group=self.pg, **dcp_save_kwargs)
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        else:
            save_with_gc(self.states, checkpoint_id=checkpoint_id, dcp_save_kwargs=dcp_save_kwargs)
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled) "
            f"in {time.monotonic() - begin:.2f} seconds. "
            f"Checkpoint is saved to {checkpoint_id}"
        )

    def load_full_sd(self, state_dict, strict=True, is_non_standard_sharded_sd: bool = False):
        r"""
        将完整 ``state_dict`` 原地加载到当前模型参数中。

        该接口默认要求输入为完整权重（full state dict）。若输入为非完整切分权重，
        仅在 ``is_non_standard_sharded_sd=True`` 且模型本身也是对应并行度的
        non-standard sharded model（例如旧 EP 手工切分实现）时支持。

        Args:
            state_dict (dict): 待加载的参数字典。默认应为完整 ``state_dict``。
            strict (bool, optional): 是否严格检查缺失键和多余键。Default: ``True``。
            is_non_standard_sharded_sd (bool, optional): 输入 ``state_dict`` 是否为
              non-standard sharded 形式。Default: ``False``。

        .. note::
            当 ``is_non_standard_sharded_sd=False`` 时，会按当前模型并行方式（包含旧 EP 场景）
            对 checkpoint 做切分后加载。

            当 ``is_non_standard_sharded_sd=True`` 时，仅按当前 FSDP/标准 DTensor 逻辑处理，
            不再对非标准并行分片（如旧 EP）做二次切分，因为输入已是本地分片。

        .. warning::
            non-standard sharding 指“手工切分并以 ``Tensor`` 保存”，而非 ``DTensor``。
            该模式只在模型分片方式与输入 ``state_dict`` 完全一致时可用。
        """

        stateful = self.states[MODEL]

        if not is_non_standard_sharded_sd:
            assert not isinstance(stateful, EPModelWrapper), "EPModelWrapper is not tested. EPModelWrapper 返回的 sd 不是 share 内存的，虽然后面调用 load 了，但是仍需要测试一下再使用比较保险"
            model_state_dict = stateful.state_dict()
        else:
            assert len(self.states[MODEL].model) == 1 # TODO: Interleaved PP
            model_state_dict = get_model_state_dict(self.states[MODEL].model[0])

        missing, unexpected, used = [], [], set()
        for k, v in model_state_dict.items():
            v: nn.Parameter
            if k not in state_dict:
                missing.append(k)
            else:
                target_weight = state_dict[k]
                # assert not isinstance(target_weight, DTensor), "target_weight is a DTensor, which is not supported"


                if isinstance(v, DTensor):
                    if isinstance(target_weight, DTensor):
                        if target_weight.device_mesh == v.device_mesh:
                            target_weight = target_weight.redistribute(placements=v.placements)
                        else:
                            target_weight = target_weight.full_tensor()
                        
                    if not isinstance(target_weight, DTensor): # Full Tensor from a different device mesh
                        if version.parse(torch.__version__) >= version.parse("2.7.0"):
                            # src_data_rank=None means use the local data instead of preserving the single-device semantic via scatter/broadcast.
                            # This is faster but requires the user to ensure the data consistency across ranks.
                            target_weight = distribute_tensor(target_weight.data, v.device_mesh, placements=v.placements, src_data_rank=None)
                        else:
                            target_weight = distribute_tensor(target_weight.data, v.device_mesh, placements=v.placements)


                    try:
                        v.to_local().data.copy_(target_weight.to_local())
                    except Exception as e:
                        raise RuntimeError(
                            f"Error copying {k} from {target_weight.to_local().shape}(saved sd) to {v.to_local().shape}(current model) "
                            f"{v.device_mesh=} {v.placements=} {type(stateful)=}: {e}"
                        )
                else:
                    v.data.copy_(target_weight)
                used.add(k)

        for k in state_dict:
            if k not in used:
                unexpected.append(k)
        
        if not is_non_standard_sharded_sd:
            stateful.load_state_dict(model_state_dict)

        if len(missing) != 0 or len(unexpected) != 0:
            from hy_parallelism.utils import get_missing_unexpected_str
            missing_unexpected_str = get_missing_unexpected_str(missing, unexpected, used)
            if strict:
                if dist.get_rank() == 0:
                    logger.warning(f"With {strict=}:\n{missing_unexpected_str}")
                raise ValueError(f'Loading checkpoint with {strict=}, but ({len(missing)} missing and {len(unexpected)} unexpected) params found. ({len(used)} loaded)')
            else:
                if dist.get_rank() == 0:
                    logger.warning(f"With {strict=}:\n{missing_unexpected_str}")
        else:
            logger.info(f"Perfect match. Parameters loaded")

        return missing, unexpected


    @torch.no_grad()
    def load_from_path(self, path, strict=True) -> bool:
        checkpoint_id = path
        if os.path.isdir(checkpoint_id):

            logger.info(f"Loading checkpoint from {path}.")
            begin = time.monotonic()
            states = self._states_to_load()
            torch.cuda.empty_cache()
            dcp.load(states, checkpoint_id=checkpoint_id, planner=DefaultLoadPlanner(allow_partial_load=True) if not strict else None)
            torch.cuda.empty_cache()
            GarbageCollection.collect("GC collection for checkpoint loading.")
            logger.info(f"Finished loading checkpoint in {time.monotonic() - begin:.2f} seconds.")
            return True
        else:
            self.load_full_sd(load_pt_or_safetensors(path), strict=strict)
            return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            if not self.staging_stream.query():
                begin = time.monotonic()
                self.staging_stream.synchronize()
                logger.info(
                    "Checkpointer waited staging %.2f seconds.",
                    time.monotonic() - begin,
                    )
            self.staging = False

            if self.sending_to_checkpoint_mp:
                # Copy the sync staging result to another process.
                def sync_func():
                    self.mp_queue_send.put_nowait(
                        (self.cpu_offload_state_dict, self.staging_id, self.staging_dcp_save_kwargs)
                    )

                # This may be a faster way to do zero-overhead checkpointing staging
                # checkpointing but we need more thorough investigation before
                # swithing to this method.
                # self.my_thread = threading.Thread(target=func).start()
                begin = time.monotonic()
                sync_func()
                logger.info(
                    "Checkpointer sent staged state_dict to another process %.2f seconds",
                    time.monotonic() - begin,
                    )
                self.sending_to_checkpoint_mp = False

    def _find_load_step(self, folder: str = "") -> int:
        raise NotImplementedError
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            metadata_probe = os.path.join(folder, filename, ".metadata")
            if match and os.path.isfile(metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step_or_tag: Union[int, str], folder: str = "") -> str:
        folder = folder if folder else self.folder
        if isinstance(step_or_tag, str):
            return os.path.join(folder, f"{step_or_tag}/{self.ckpt_config.folder}")
        elif isinstance(step_or_tag, int):
            return os.path.join(folder, f"{step_or_tag}/{self.ckpt_config.folder}")
        else:
            raise ValueError(f"step_or_tag must be int or str, got {type(step_or_tag)}")

    def _states_to_load(self, step: int = None) -> dict[str, Any]: # NOW, step is just a placeholder for compatibility
        """Determines which states to load for the given step.

        When checkpointer determines which step of the checkpoint to load, this API is
        used to determine which states to load based on the step.

        Args:
            step (int): The step to load the checkpoint for.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model weights.
        # states = {MODEL: self.states[MODEL]} if step == 0 else self.states
        states = self.states
        states_to_load = {
            k: v for k, v in states.items() if k not in self.exclude_from_loading
        }
        for exclude_key in self.exclude_from_loading:
            if exclude_key not in states:
                raise ValueError(f"{exclude_key} not found in state_dict.")
        return states_to_load

    def _save_last_step(self, step_or_tag: Union[int, str], dcp_save_kwargs: dict | None = None) -> None:
        # We only consider saving weights only at the end of the training. So
        # this won't affect preemption and training resume. We also only allow
        # dtype conversion when we are checkpoint model weights only and the
        # current dtype is not the same as the export dtype at the end of the training.

        if self.model_weights_only:
            # We update self.states to keep the model only.
            # After this update, self.states = {
            #      'tok_embeddings.weight':...,
            #      'layers.0.attention.wq.weight': ...
            # }.
            raise NotImplementedError('这里会改变 self.states, 多存几次，就容易出问题')
            self.states = self.states[MODEL].state_dict()

            # For now, we will manually pop the freqs_cis buffer, as we made this permanent
            # temporarily and we don't want to include it in the exported state_dict.
            # Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L348
            if 'freqs_cis' in self.states:
                self.states.pop("freqs_cis")

            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }
            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} "
                f"at last step, step {step_or_tag}."
            )
        else:
            pass
            # logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        save_with_gc(self.states, checkpoint_id=self._create_checkpoint_id(step_or_tag), dcp_save_kwargs=dcp_save_kwargs)

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        # Force saving a checkpoint at step 1 to fail fast if checkpointer is not
        # compatible with the cluster.
        if curr_step == 1:
            return True

        if force:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the background process to finish, {time.monotonic()=}.:.2f"
            )
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()
                self.async_future = None
        elif self.async_future is not None:
            raise RuntimeError(
                "self.async_future is not None, but self.async_mode is not enabled "
                "and fault tolerance is not active."
            )

    def _async_with_pinned_memory(self, checkpoint_id: str, dcp_save_kwargs: dict | None = None) -> None:
        self._cpu_staging(checkpoint_id, dcp_save_kwargs=dcp_save_kwargs)
        self.sending_to_checkpoint_mp = True

    def _cpu_staging(self, checkpoint_id: str | None, dcp_save_kwargs: dict | None = None) -> None:
        """Offload state_dict to CPU memory"""
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_id = checkpoint_id
            self.staging_dcp_save_kwargs = dcp_save_kwargs or {}

    def _purge_stale_checkpoints(self):
        if (
                self.keep_latest_k > 0
                and dist.get_rank() == 0
                and os.path.isdir(self.folder)
        ):
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):

                # match = re.search(r"step-(\d+)", filename)
                # path = os.path.join(self.folder, filename)
                # discovered_checkpoints.append((int(match.group(1)), path))
                if os.path.isdir(os.path.join(self.folder, filename)):
                    # get last modification time
                    discovered_checkpoints.append((os.path.getmtime(os.path.join(self.folder, filename)), os.path.join(self.folder, filename)))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)


def has_meta(module):
    for param in module.parameters():
        if param.device == torch.device('meta'):
            return True
    return False

def load_ckpt(model, ckpt_path, strict=True, post_process_sd_fn=None):
    assert Path(ckpt_path).exists()
    if ckpt_path.suffix == '.safetensors':
        from safetensors import safe_open
        state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        if post_process_sd_fn is not None:
            state_dict = post_process_sd_fn(state_dict)
        model.load_state_dict(state_dict, strict=strict, assign=has_meta(model))
    elif os.path.isdir(ckpt_path):
        assert next(Path(ckpt_path).glob('*')).suffix == '.distcp'
        # if sum(p.numel() for p in model.parameters()) > 20e9:
        #     raise NotImplementedError(
        #         'Loading a large model from a dcp checkpoint in a naive way. This may lead to OOM.\n'
        #         'Recommended solution: Slice the model first, then load the checkpoint with `dcp.load`.'
        #     )
        # state_dict = dcp_to_torch_state_dict(ckpt_path)
        if has_meta(model):
            model = model.to_empty(device='cpu')
        loguru.logger.warning(f'When loading dcp model, post_process_sd_fn will be ignored')
        CheckpointManager(
            {},
            Checkpoint(
                enable_checkpoint=True,
            ),
            model_parts=[model],
        ).load_from_path(ckpt_path)
    else:
        state_dict = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        if post_process_sd_fn is not None:
            state_dict = post_process_sd_fn(state_dict)
        model.load_state_dict(state_dict, strict=strict, assign=has_meta(model))

    assert not has_meta(model)



def dcp_to_torch_state_dict(model_dir):
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    if os.path.exists(model_dir):
        save_checkpoint_path = model_dir
    else:
        raise FileNotFoundError(f"Checkpoint directory {model_dir} does not exist.")

    # Load the state_dict from the DCP checkpoint
    state_dict: STATE_DICT_TYPE = {}

    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    if "state" in state_dict:
        state_dict = state_dict["state"]

    return state_dict
    # return state_dict["model"]


def load_pt_or_safetensors(sd_path):
    sd_path = Path(sd_path)
    if sd_path.suffix == '.safetensors':
        import safetensors
        state_dict = safetensors.torch.load_file(sd_path)
    else:
        try:
            state_dict = torch.load(sd_path, weights_only=True, map_location='cpu')
        except Exception as e:
            if '`weights_only` argument in `torch.load` from `False` to ' in str(e):
                state_dict = torch.load(sd_path, weights_only=False, map_location='cpu')
            else:
                raise e
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'module' in state_dict:
        state_dict = state_dict['module']
    return state_dict

def torch_state_dict_to_dcp(dcp_save_path, sd_path='', sd_input=None):
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
    if sd_input:
        sd = sd_input
    else:
        sd = load_pt_or_safetensors(sd_path)
    while 'model' in sd:
        sd = sd['model']
    while 'module' in sd:
        sd = sd['module']
    # adapt dcp_to_torch_state_dict
    sd = {'model':sd}
    # dcp.save(sd, checkpoint_id=dcp_save_path)
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.

    Path(dcp_save_path).mkdir(parents=True, exist_ok=True)
    _save_state_dict(
        sd, storage_writer=FileSystemWriter(dcp_save_path), no_dist=True
    )
