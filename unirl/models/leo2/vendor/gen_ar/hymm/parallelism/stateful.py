from packaging import version
from dataclasses import dataclass
from itertools import chain
from typing import Any, Generic, Iterator, TypeVar, Union, Dict
import torch
import loguru
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor.placement_types import _StridedShard
from torch.distributed.tensor import distribute_tensor
from torch.optim import Optimizer
from .parallel_states import get_parallel_state

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

T = TypeVar("T", bound=Optimizer)
class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[list[T]]
    model_parts: list[nn.Module]

    def check(self):
        _PARAMS = "params"
        from itertools import chain
        optim = self.optimizers[0]
        model = self.model_parts[0]

        params = set(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
        for p in model.parameters():
            assert p in params
        import loguru
        # sharding issue, not resharding after forward lead to inconsistent param type
        loguru.logger.info('Optimizer check 通过！！！！！！')

    def get_parameter_groups(self, model):
        from hymm.parallelism import utils
        meshs = list({p.device_mesh for p in model.parameters()})
        meshs.sort(key=lambda x: x.mesh_dim_names)
        meshs = tuple(meshs)

        mesh_names_of_all_ranks = utils.gather_obj([k.mesh_dim_names for k in meshs])
        for mesh_names_i in mesh_names_of_all_ranks:
            assert mesh_names_i == mesh_names_of_all_ranks[0], f'{mesh_names_of_all_ranks=}'

        if len(meshs) > 1:
            loguru.logger.info(f'{len(meshs)} different device meshes found. {meshs}')
        for mesh in meshs:
            params = []
            for p in model.parameters():
                if p.requires_grad and p.device_mesh == mesh:
                    params.append(p)
            yield params



    def __init__(
            self,
            model_parts: list[nn.Module],
            optimizer_cls: type[T],
            optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = [] # [pp, n_diff_mesh]
        self.model_parts = model_parts
        for model in self.model_parts:
            # params = [p for p in model.parameters() if p.requires_grad]
            optimizer_n_diff_mesh = []
            for params in self.get_parameter_groups(model):
                optimizer_n_diff_mesh.append(optimizer_cls(params, **optimizer_kwargs))
                all_params.extend(params)
            self.optimizers.append(optimizer_n_diff_mesh)

        # self._validate_length(len(self.model_parts))

        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer_pp in self.optimizers:
            for optimizer in optimizer_pp:
                optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer_pp in self.optimizers:
            for optimizer in optimizer_pp:
                optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        ret = {
            k: v
            for model_part, optimizer_pp in zip(self.model_parts, self.optimizers)
            for optimizer in optimizer_pp
            for k, v in func(model_part, optimizer).items()
        }


        # TODO: FIX THIS BUG
        # from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate
        # parallel_dims = get_parallel_state()
        # for fqn in list(ret.keys()):
        #     param  = ret[fqn]
        # #
        #     if 'experts' in fqn and fqn.startswith('state.') and isinstance(param, DTensor):
        #         expert_fsdp_mesh = parallel_dims.expert_fsdp_mesh
        #         if expert_fsdp_mesh.ndim == 2:
        #             placements = [Replicate(), Shard(0)]
        #         else:
        #             placements = [Shard(0)]
        #         shape = list(param._spec.shape)
        #         assert shape[0] % parallel_dims.ep == 0
        #         shape[0] = shape[0] // parallel_dims.ep
        #         shape = torch.Size(shape) # hashable
        #         dtensor = DTensor.from_local(
        #             local_tensor=param.to_local(), device_mesh=expert_fsdp_mesh, placements=placements,
        #             shape=shape, stride=param._spec.stride,
        #         )
        #         ret[fqn] = dtensor

        return ret

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # if get_parallel_state().ep_enabled:
        #     msg = ('Resuming optimizer in ep mode is not implemented yet. '
        #            'Because ep is implemented in pure `torch.Tensor`, size mismatch will occur when ep size changes.'
        #            'The optimizer resuming process will be skipped.')
        #     loguru.logger.critical(msg)
        #     return
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for optimizer_pp in self.optimizers:
            list(map(func, self.model_parts, optimizer_pp))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
            self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


import copy
import functools
from typing import Any, Callable, Iterator

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler

class LRSchedulersContainer(Stateful):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Note**
    Users who want to customize the lr_scheduler behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same
    signature as ``torch.optim.lr_scheduler.LRScheduler`` class: ``step()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes all the lr schedulers are the same. There is no easy way to support
    resharding for multiple different LRSchedulers because LRScheduler.state_dict() is not
    resharding friendly. Therefore, the limitation is used to allow TorchTitan to support
    lr scheduler resharding.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    schedulers: list[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer,
                 # lr_lambda: Callable
                 get_lr_sheculer: Callable
                 ) -> None:
        assert (
                len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [get_lr_sheculer(optimizer) for optimizers_pp in optimizers for optimizer in optimizers_pp]

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self, *args, **kwargs) -> None:
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because
        # the state_dict is the same for all. See the limitations section in the
        # docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``training.steps`` and ``lr_scheduler.warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))

class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        # self.cache_state_dict = self.get_state_dict()

    def get_state_dict(self):
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        # self.cache_state_dict = self.get_state_dict()


class EPModelWrapper(ModelWrapper):
    def __init__(self, *args, parallel_dims, **kwargs):
        self.parallel_dims = parallel_dims
        super().__init__(*args, **kwargs)
        assert not self.parallel_dims.tp_enabled

    def state_dict(self) -> dict[str, Any]:
        # can not use cache anymore
        return self.get_state_dict()

    def get_state_dict(self):
        sd = super().get_state_dict()
        import time
        import torch
        from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate


        for fqn in list(sd.keys()):
            param  = sd[fqn]

            if self.model[0].is_expert(fqn):
                # ('dp_replicate', 'dp_shard_mod_ep', 'ep')
                ep_related_mesh = self.parallel_dims.ep_related_mesh
                ep_related_mesh:DeviceMesh


                # ======================= old implementation =========================
                # kevinkhwu:
                #   The old implementation will raise an error when fsdp is sharded unevenly.
                #   This is a bug of PyTorch's _StridedShard. _StridedShard is not considered
                #   in most cases of PyTorch implementation. PyTorch simply uses
                #   `isinstance(..., Shard)` for placement check and there is no _StridedShard
                #   case.
                #
                #   Error logs:
                #   ```
                #   key:model.p invalid fill tensor-volume:
                #   10 chunks-volume: 9
                #   ```
                #   ```
                #   AssertionError: req MetadataIndex(fqn='model.language_model.transformer.h.0.mlp.experts.gate_proj',
                #   offset=torch.Size([27, 0, 0]), index=54) mismatch sizes torch.Size([0, 3072, 4096]) vs torch.Size([1, 3072, 4096])
                #   ```
                #
                #   To ensure correct checkpoint saving and loading (before this bug is fixed),
                #   we have to rearrange the tensor to fully shard(0) case. Even the tensor
                #   is not in a semantically correct placement after rearrangement, full parameters
                #   are safely saved. When loading, we will rearrange the tensor back.
                # ---------------------------------------------------------------------
                #
                # if ep_related_mesh.ndim == 3:
                #     placements = [Replicate(), _StridedShard(0, split_factor=self.parallel_dims.ep), Shard(0)]
                # else:
                #     placements = [_StridedShard(0, split_factor=self.parallel_dims.ep), Shard(0)]
                #
                # shape = list(param._spec.shape)
                # shape[0] = shape[0] * self.parallel_dims.ep
                # shape = torch.Size(shape) # hashable
                # dtensor = DTensor.from_local(
                #     local_tensor=param.to_local(), device_mesh=ep_related_mesh, placements=placements,
                #     shape=shape, stride=param._spec.stride
                # )
                # ================================================================

                param = param.full_tensor()
                if ep_related_mesh.ndim == 3:
                    placements = [Replicate(), Replicate(), Shard(0)]
                    target_placements = [Replicate(), Shard(0), Shard(0)]
                else:
                    placements = [Replicate(), Shard(0)]
                    target_placements = [Shard(0), Shard(0)]

                dtensor = DTensor.from_local(
                    local_tensor=param,
                    device_mesh=ep_related_mesh,
                    placements=placements,
                )
                # HACK(kevinkhwu):
                # This is a workaround for a PyTorch bug (<2.8.0).
                if version.parse(torch.__version__) >= version.parse("2.8.0"):
                    loguru.logger.warning(
                        "The workaround for uneven sharding is only needed for PyTorch < 2.8.0. "
                        f"You are running torch=={torch.__version__}, please consider removing this workaround if the bug is fixed."
                    )
                if param.shape[0] % ep_related_mesh.size(mesh_dim=-2) == 0:
                    dtensor = dtensor.redistribute(placements=target_placements)


                sd[fqn] = dtensor

        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:

        from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate

        for fqn in list(state_dict.keys()):
            param  = state_dict[fqn]
            if self.model[0].is_expert(fqn):
                expert_fsdp_mesh = self.parallel_dims.expert_fsdp_mesh
                ep_related_mesh = self.parallel_dims.ep_related_mesh

                # ======================= old implementation =========================
                # if expert_fsdp_mesh.ndim == 2:
                #     placements = [Replicate(), Shard(0)]
                # else:
                #     placements = [Shard(0)]
                # shape = list(param._spec.shape)
                # assert shape[0] % self.parallel_dims.ep == 0
                # shape[0] = shape[0] // self.parallel_dims.ep
                # shape = torch.Size(shape) # hashable
                # dtensor = DTensor.from_local(
                #     local_tensor=param.to_local(), device_mesh=expert_fsdp_mesh, placements=placements,
                #     shape=shape, stride=param._spec.stride,
                # )
                # ================================================================

                # Wrong implementation, redistribute(_StridedShard) is dangerous uneven sharding case.
                # if ep_related_mesh.ndim == 3:
                #     placements = [Replicate(), _StridedShard(0, split_factor=self.parallel_dims.ep), Shard(0)]
                # else:
                #     placements = [_StridedShard(0, split_factor=self.parallel_dims.ep), Shard(0)]
                # param.redistribute(placements=placements)


                # as _StridedShard is dangerous in uneven sharding case, we use a two-stage sharding paradigm.
                if ep_related_mesh.ndim == 3:
                    placements = [Replicate(), Replicate(), Shard(0)]
                else:
                    placements = [Replicate(), Shard(0)]

                param = param.full_tensor()
                param = DTensor.from_local(
                    local_tensor=param, device_mesh=ep_related_mesh,
                ).redistribute(placements=placements)

                if expert_fsdp_mesh.ndim == 2:
                    placements = [Replicate(), Shard(0)]
                else:
                    placements = [Shard(0)]

                # The uneven sharding behavior is experimental and subject to change.
                # dtensor = distribute_tensor(param.to_local(), device_mesh=expert_fsdp_mesh, placements=placements)
                dtensor = DTensor.from_local(param.to_local(), device_mesh=expert_fsdp_mesh).redistribute(placements=placements)


                state_dict[fqn] = dtensor

        return super().load_state_dict(state_dict)

class TrainingState(Stateful, dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        super(Stateful, self).__init__()

    @staticmethod
    def from_dict(dic):
        ret = TrainingState()
        for k, v in dic:
            ret[k] = v
        return ret

    def state_dict(self) -> Dict[str, Any]:
        return self.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.clear()
        self.update(state_dict)



def wrap_model_to_stateful(models):
    parallel_dims = get_parallel_state()
    if parallel_dims.ep_enabled:
        loguru.logger.debug('Use EPModelWrapper')
        return EPModelWrapper(models, parallel_dims=parallel_dims)
    else:
        loguru.logger.debug('Use Naive ModelWrapper')
        return ModelWrapper(models)
