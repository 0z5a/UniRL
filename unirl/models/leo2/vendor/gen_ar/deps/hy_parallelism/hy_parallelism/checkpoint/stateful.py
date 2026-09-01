# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os
import copy
import functools
from itertools import chain
from typing import Any, Callable, Dict, Generic, Iterator, Optional, Type, TypeVar

import loguru
import torch
import torch.nn as nn
from packaging import version
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.distributed.checkpoint.state_dict import _get_fqns
from hy_parallelism.globabl_states import get_global_states

from hy_parallelism.parallel_states import get_parallel_state

T = TypeVar("T", bound=Optimizer)


def maybe_scatter_ep_param(param):
    # ('dp_replicate', 'dp_shard_mod_ep', 'ep')
    ep_related_mesh = get_parallel_state().ep_related_mesh
    ep_related_mesh: DeviceMesh

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

    is_dtensor = isinstance(param, DTensor)
    if is_dtensor:
        param = param.full_tensor()

    if get_parallel_state().tp_enabled:
        raise NotImplementedError('Old EP implementation with TP is not implemented yet.')

    if ep_related_mesh.ndim == 4: # ep fsdp rep, ep fsdp shard, ep, etp
        placements = [Replicate(), Replicate(), Shard(0), Replicate()]
        target_placements = [Replicate(), Shard(0), Shard(0), Replicate()]
    elif ep_related_mesh.ndim == 3: # ep fsdp rep, ep, etp  | ep fsdp rep, ep fsdp shard, ep
        placements = [Replicate(), Replicate(), Shard(0)]
        target_placements = [Replicate(), Shard(0), Shard(0)]
    elif ep_related_mesh.ndim == 2: # ep fsdp shard, ep
        placements = [Replicate(), Shard(0)]
        target_placements = [Shard(0), Shard(0)]

    dtensor = DTensor.from_local(
        local_tensor=param,
        device_mesh=ep_related_mesh,
        placements=placements,
    )

    # HACK(kevinkhwu):
    # Only shard when the tensor can be evenly sharded.
    # This is a workaround for a PyTorch bug (<2.8.0).
    # See also: https://github.com/pytorch/pytorch/commit/c3bc6b354239d78a15e2bcd43f6567c71db1ba71
    if param.shape[0] % ep_related_mesh.size(mesh_dim=-2) == 0:
        # Reduce peak memory
        dtensor = dtensor.redistribute(placements=target_placements)
    else:
        # Uneven case, save without manual sharding
        import warnings
        if version.parse(torch.__version__) >= version.parse("2.8.0"):
            if os.environ.get('RANK', '0') == '0' and os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
                loguru.logger.debug(
                    f'PyTorch >= 2.8.0 ({torch.__version__}), uneven sharding works fine now. '
                    'Consider refactoring the code for lower memory usage.'
                )

    return dtensor


def maybe_recover_ep_param(param, fsdp_applied_for_experts):

    expert_fsdp_mesh = get_parallel_state().expert_fsdp_mesh
    ep_related_mesh = get_parallel_state().ep_related_mesh

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
    if ep_related_mesh.ndim == 4:
        placements = [Replicate(), Replicate(), Shard(0), Replicate()] # 用于切 ep
    elif ep_related_mesh.ndim == 3:
        placements = [Replicate(), Replicate(), Shard(0)]
    elif ep_related_mesh.ndim == 2:
        placements = [Replicate(), Shard(0)]
    else:
        raise ValueError(f'Unexpected EP related mesh dimension: {ep_related_mesh.ndim}')

    param = param.full_tensor()

    # ep sharding
    param = DTensor.from_local(
        local_tensor=param,
        device_mesh=ep_related_mesh,
    ).redistribute(placements=placements)

    # TODO(kevinkhwu):
    # This implmentation assumes that the expert fsdp mesh is always sharded.
    # This is not the case when expert fsdp is not enabled.
    # We need to add a check for this.
    if expert_fsdp_mesh.ndim == 2:
        placements = [Replicate(), Shard(0)]
    elif expert_fsdp_mesh.ndim == 1:
        placements = [Shard(0)]
    else:
        raise ValueError(f'Unexpected expert FSDP mesh dimension: {expert_fsdp_mesh.ndim}')

    # fsdp sharding
    dtensor = DTensor.from_local(param.to_local(), device_mesh=expert_fsdp_mesh).redistribute(placements=placements)

    if not fsdp_applied_for_experts:
        dtensor = dtensor.full_tensor()
        loguru.logger.debug('Convert DTensor to Tensor for expert')

    return dtensor


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

        optim = self.optimizers[0]
        model = self.model_parts[0]

        params = set(chain.from_iterable(g[_PARAMS] for g in optim.param_groups))
        for p in model.parameters():
            assert p in params
        import loguru

        # sharding issue, not resharding after forward lead to inconsistent param type
        loguru.logger.info('Optimizer check 通过！！！！！！')

    def _group_parameters_by_device_mesh(
        self, parameters: list[nn.Parameter]
    ) -> Iterator[list[nn.Parameter]]:
        if len(parameters) == 0:
            return

        from hy_parallelism import utils

        is_dtensor = [isinstance(p, DTensor) for p in parameters]
        assert all(is_dtensor) or not any(is_dtensor), (
            "Expected all parameters to be DTensor or all parameters to be Tensor"
        )

        if not any(is_dtensor):
            # Keep behavior: do not filter by requires_grad for plain Tensor parameters.
            yield parameters
            return

        meshs = list({p.device_mesh for p in parameters})
        meshs.sort(key=lambda x: x.mesh_dim_names)
        meshs = tuple(meshs)

        mesh_names_of_all_ranks = utils.gather_obj([k.mesh_dim_names for k in meshs])
        for mesh_names_i in mesh_names_of_all_ranks:
            assert mesh_names_i == mesh_names_of_all_ranks[0], f"{mesh_names_of_all_ranks=}"

        if len(meshs) > 1:
            loguru.logger.info(f"{len(meshs)} different device meshes found. {meshs}")

        for mesh in meshs:
            grouped_params: list[nn.Parameter] = []
            for p in parameters:
                if p.requires_grad and p.device_mesh == mesh:
                    grouped_params.append(p)

            # Skip if there is no trainable parameters on this mesh.
            if len(grouped_params) > 0:
                yield grouped_params

    def get_parameter_groups(self, model, skip_special=False):
        """
        This function groups parameters by their mesh and creates separate optimizers
        for each group in order to avoid this cross-mesh communication issue.

        Explanation:
            Model parameters may reside on different meshes, e.g., dense layers on an FSDP mesh,
            experts on an E-FSDP mesh, and tensor-parallel (TP) layers on yet another mesh.
            If we do not distinguish between these meshes and simply create a single optimizer
            for all DTensors across different meshes, it will cause cross-mesh communication
            during optimizer steps, which is inefficient and potentially incorrect.
        """
        self._special_optimizer_group_configs = []

        from hy_parallelism.distributed.fsdp_util import get_fsdp_named_parameters
        named_params = list(get_fsdp_named_parameters(model))


        is_dtensor = [isinstance(p, DTensor) for _, p in named_params]
        assert all(is_dtensor) or not any(is_dtensor), f'Expected all parameters to be DTensor or all parameters to be Tensor'

        special_params_by_optimizer_cfg: list[tuple[list[nn.Parameter], type[T], dict[str, Any]]] = []

        trainable_params = []
        for name, param in named_params:
            if skip_special:
                if self.filter_param_func is not None and not self.filter_param_func(name, param):
                    continue

                if self.optimizer_factory_for_special_param is not None:
                    special_optim = self.optimizer_factory_for_special_param(name, param)
                    if special_optim is not None:
                        special_optimizer_cls, special_optimizer_kwargs = special_optim
                        if param.requires_grad:
                            grouped = False
                            # 因为 dict 不是 hashable 的，所以这里暂时用 for 循环检查
                            for (
                                grouped_params,
                                grouped_optimizer_cls,
                                grouped_optimizer_kwargs,
                            ) in special_params_by_optimizer_cfg:
                                # TODO: 其实也可以在这里判断 param 是不是在同一个device_mesh
                                if (
                                    grouped_optimizer_cls is special_optimizer_cls
                                    and grouped_optimizer_kwargs == special_optimizer_kwargs
                                ):
                                    grouped_params.append(param)
                                    grouped = True
                                    break
                            if not grouped:
                                special_params_by_optimizer_cfg.append(
                                    ([param], special_optimizer_cls, special_optimizer_kwargs)
                                )
                        continue

            if param.requires_grad:
                trainable_params.append(param)

        # 因为用户传入的 factory 规则可能并不能完美保证同一个 optimizer 配置都是同一个 device_mesh 的参数
        # 所以需要再次按 device_mesh 分组
        for params, special_optimizer_cls, special_optimizer_kwargs in special_params_by_optimizer_cfg:
            for mesh_params in self._group_parameters_by_device_mesh(params):
                self._special_optimizer_group_configs.append(
                    (mesh_params, special_optimizer_cls, special_optimizer_kwargs)
                )


        if not any(is_dtensor):
            assert len(named_params) > 0
            if len(trainable_params) > 0:
                yield trainable_params
            return

        yield from self._group_parameters_by_device_mesh(trainable_params)

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T]=None,
        optimizer_kwargs: dict[str, Any]=None,
        optimizer_factory_for_special_param: Callable[[str, nn.Parameter], tuple[type[T], dict[str, Any]]] | None = None,
        pre_optimizer_hook=None,
        filter_param_func=None,
    ) -> None:
        r"""Initialize a distributed optimizer wrapper over model partitions.

        Args:
            model_parts (list[nn.Module]): Model partitions whose parameters are grouped and
              optimized, potentially across different device meshes.
            optimizer_cls (type[T]): Optimizer class used to build optimizers for regular
              parameter groups.
            optimizer_kwargs (dict[str, Any]): Keyword arguments forwarded to
              ``optimizer_cls``.
            optimizer_factory_for_special_param (callable, optional): Factory that receives a
              parameter name and parameter itself, and may return a pair of
              ``(special_optimizer_cls, special_optimizer_kwargs)`` for that parameter.
              Default: ``None``.
            filter_param_func (callable, optional): Predicate used to decide whether a
              parameter should be included. Default: ``None``.
        """
        if optimizer_cls is None and optimizer_kwargs is None:
            optimizer_kwargs = {}
            assert optimizer_factory_for_special_param is not None, "optimizer_cls and optimizer_kwargs are required if optimizer_factory_for_special_param is not provided"
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.filter_param_func = filter_param_func
        self.optimizer_factory_for_special_param = optimizer_factory_for_special_param


        all_params = []
        self.optimizers = []  # [pp, n_diff_mesh]
        self.model_parts = model_parts
        self.param_groups_by_mesh = []
        for model in self.model_parts:
            if pre_optimizer_hook is not None:
                pre_optimizer_hook(model)

            # For faster clip_grad_norm. All params in the same mesh (no matter special or not) are grouped together.
            for params in self.get_parameter_groups(model, skip_special=False):
                self.param_groups_by_mesh.append(params)

            optimizer_n_diff_mesh = []
            # Special params are not included, ensuring no duplicate optimizers
            for params in self.get_parameter_groups(model, skip_special=True):
                optimizer_n_diff_mesh.append(optimizer_cls(params, **optimizer_kwargs))
                all_params.extend(params)
            
            # Only special params are included, ensuring no duplicate optimizers
            for params, special_optimizer_cls, special_optimizer_kwargs in self._special_optimizer_group_configs:
                optimizer_n_diff_mesh.append(special_optimizer_cls(params, **special_optimizer_kwargs))
                all_params.extend(params)
            self.optimizers.append(optimizer_n_diff_mesh)

        # self._validate_length(len(self.model_parts))

        self.all_params = all_params
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[list[T]]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs):
        for optimizer_pp in self.optimizers:
            for optimizer in optimizer_pp:
                optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer_pp in self.optimizers:
            for optimizer in optimizer_pp:
                optimizer.zero_grad(*args, **kwargs)

    def is_ep_states(self, fqn: str, param: Any) -> bool:
        return self.model_parts[0].is_expert(fqn) and fqn.startswith('state.') and isinstance(param, DTensor)

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

        global_states = get_global_states()
        if get_parallel_state().ep_enabled and not global_states.use_titan_moe: 
            for fqn in list(ret.keys()):
                param  = ret[fqn]
                if self.is_ep_states(fqn, param):
                    dtensor = maybe_scatter_ep_param(param)
                    ret[fqn] = dtensor

        return ret

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        global_states = get_global_states()
        if get_parallel_state().ep_enabled and not global_states.use_titan_moe: 
            for fqn in list(state_dict.keys()):
                if self.is_ep_states(fqn, state_dict[fqn]):
                    state_dict[fqn] = maybe_recover_ep_param(state_dict[fqn], True)

        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        for optimizer_pp in self.optimizers:
            list(map(func, self.model_parts, optimizer_pp))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if " "using OptimizersInBackwardContainer."
        )

    def _post_init(self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


    def register_moe_balancing_hook(self):
        parallel_dims = get_parallel_state()
        from hy_parallelism.optimizers.moe_bias_optimizer_hook import _update_expert_bias
        self.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                self.model_parts, parallel_dims=parallel_dims
            )
        )


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

    def __init__(
        self,
        optimizers: OptimizersContainer,
        # lr_lambda: Callable
        get_lr_sheculer: Callable,
    ) -> None:
        assert len(optimizers) > 0, "Must have at least one optimizer to create LRScheduler"

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

    def get_last_lr(self):
        assert len(self.schedulers) > 0, 'No schedulers found'
        lrs = [s.get_last_lr() for s in self.schedulers]
        assert all(lr == lrs[0] for lr in lrs), f'Inconsistent lr for different schedulers: {lrs}'
        return lrs[0]


class ModelWrapper(Stateful):
    """
    Note:
        When using `dcp.load`, model tensors might be updated in-place after
        `_load_state_dict` and before `elem.load_state_dict(stateful_sd[key])`.
        However, depending only on in-place updates is risky: if `get_state_dict`
        changes the state dict, those changes only update a copied statedict inplace, 
        rather than the original state dict.
        To ensure parameters are properly restored after checkpoint loading,
        `dcp.load` explicitly calls `elem.load_state_dict(stateful_sd[key])`.

    Best Practice:
        Always implement `load_state_dict` with the loaded state dict,
        even if tensors were possibly modified in-place, to ensure correct and
        reproducible checkpoint restoration.
    """


    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        # self.cache_state_dict = self.get_state_dict()

    def get_state_dict(self):
        """
        Handles interleaved pp and fully qualified names (FQN) in the state_dict.

        Potential exception:
            ValueError: Raised if the state_dict contains a tensor whose metadata type is not BytesStorageMetadata.
            This error can appear with a message such as:
                "Invalid checkpoint metadata for {fqn}, expected BytesStorageMetadata but found {type(md)}"
            For example:
                expected BytesStorageMetadata but found <class 'torch.distributed.checkpoint.metadata.TensorStorageMetadata'>

            Explanation:
                Regular Tensor state_dicts use BytesStorageMetadata, whereas DTensor state_dicts use TensorStorageMetadata.
                This exception can occur if you try to load a DCP checkpoint saved from a DTensor model into a plain Tensor model (without FSDP).

            For more details, see `create_default_local_load_plan` in default_planner.py.
        """

        return {
            k: v
            for sd in map(get_model_state_dict, self.model)
            for k, v in sd.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return self.get_state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))

        """
        NOTE: We comment out the caching code below for memory saving.
        """
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        # self.cache_state_dict = self.get_state_dict()


class EPModelWrapper(ModelWrapper):
    """
    Only torch moe v1 and ptm moe should use this wrapper.

    只支持 expert 的参数是 fuse 起来的实现，并且务必保证 fsdp 在 expert 上正确切分，
    因为内部逻辑 assume 了 fsdp 的切分情况
    """

    def __init__(self, *args, parallel_dims, **kwargs):
        self.parallel_dims = parallel_dims
        super().__init__(*args, **kwargs)
        assert not self.parallel_dims.tp_enabled

        self.fsdp_applied_for_experts = False
        for k, v in super().get_state_dict().items():
            if self.model[0].is_expert(k) and isinstance(v, DTensor):
                self.fsdp_applied_for_experts = True
                break

    def state_dict(self) -> dict[str, Any]:
        # can not use cache anymore
        return self.get_state_dict()

    def get_state_dict(self):
        sd = super().get_state_dict()
        for fqn in list(sd.keys()):
            param = sd[fqn]
            if self.model[0].is_expert(fqn):
                dtensor = maybe_scatter_ep_param(param)
                # print(f'{fqn} is expert, scatter to {dtensor.placements} {dtensor.device_mesh} {dtensor.shape}')
                sd[fqn] = dtensor
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for fqn in list(state_dict.keys()):
            param = state_dict[fqn]
            if self.model[0].is_expert(fqn):
                dtensor = maybe_recover_ep_param(param, self.fsdp_applied_for_experts)
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

def get_reverse_mapped_stateful(stateful, reverse_mapping_fn: Callable):
    ...
    class MappedStateful(Stateful):
        def __init__(self, stateful, mapping_fn: Callable):
            self.stateful = stateful
            self.mapping_fn = mapping_fn
            import torch.distributed.checkpoint as dcp
            dcp_path = '.'
            reader = dcp.FileSystemReader(dcp_path)
            md = reader.read_metadata()
            state_dict_metadata = md.state_dict_metadata
            for k, v in state_dict_metadata.items():
                param_name = k
                try:
                    shape = v.size
                    print(param_name, shape)
                except:
                    print(param_name, 'error')


        def state_dict(self) -> Dict[str, Any]:
            ...
        
        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            ...


def wrap_model_to_stateful(models, stateful_class: Optional[Type[Stateful]]=None):
    if stateful_class is None:
        parallel_dims = get_parallel_state()
        global_states = get_global_states()
        if parallel_dims.ep_enabled and not global_states.use_titan_moe:
            msg = (
                "EPModelWrapper is only used for old MOE implementation and is deprecated. "
                "Implicitly wrapping model to EPModelWrapper is dangerous and forbidden since 1.0.0. "
                "If you are using old MOE implementation, please set `model_stateful_class` to the Engine instead."
            )
            loguru.logger.warning(msg)
            return EPModelWrapper(models, parallel_dims=parallel_dims)
        else:
            return ModelWrapper(models)
    else:
        return stateful_class(models)
        
