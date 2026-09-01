# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

import os
import datetime
from typing import Optional
from dataclasses import field
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from packaging import version

import loguru
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


from loguru import logger

def set_timeout_for_mesh(device_mesh: DeviceMesh, timeout: datetime.timedelta | float | None = None):
    assert timeout is not None

    if not isinstance(timeout, datetime.timedelta):
        timeout = datetime.timedelta(seconds=timeout)
    
    if dist.get_backend() == "gloo":
        loguru.logger.error(f'Gloo backend may not work well with timeout configuration. See https://github.com/pytorch/pytorch/issues/165422')

    for dim in range(device_mesh.ndim):
        dist.distributed_c10d._set_pg_timeout(timeout, device_mesh.get_group(dim))

def init_device_mesh(
    device_type: str,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim_names: Optional[tuple[str, ...]] = None,
    timeout: datetime.timedelta | float | None = None, # flaot timeout or time delta
    **kwargs,
) -> DeviceMesh:
    from torch.distributed.device_mesh import init_device_mesh as init_device_mesh_impl
    device_mesh = init_device_mesh_impl(device_type, mesh_shape, mesh_dim_names=mesh_dim_names, **kwargs)
    if timeout is not None:
        set_timeout_for_mesh(device_mesh, timeout)
    return device_mesh


device_type = 'cuda'

def flatten_mesh(mesh, keys, target_name, force=False):
    valid_keys = []
    for k in keys:
        if k in mesh.mesh_dim_names and (mesh[k].size() > 1 or k in ['dp_shard', 'dp_replicate']):
            valid_keys.append(k)
        else:
            # loguru.logger.debug(f'Skipping key {k} during flattening. Ignore this message if you don\'t understand.')
            pass
    if valid_keys:
        mesh[tuple(valid_keys)]._flatten(mesh_dim_name=target_name)
    else:
        loguru.logger.info(f'No valid keys found from {keys}')
        if force:
            raise RuntimeError(f'Fail to flatten {mesh} with {keys=}')


@dataclass
class HYParallelDims:
    world_size: int

    dp_replicate: int  # setting dp_replicate to -1 will enable HSDP if possible
    dp_shard: int
    tp: int = 1
    etp: int = 1 # Degree of tensor model parallelism of expert layer. Default is same to --tensor-model-parallel-size.
    cp: int = 1
    ep: int = 1
    pp: int = 1
    world_splits: int = 1
    timeout: datetime.timedelta | float | None = None

    # ep sharding will produce zero shape weights, which is incompatible with ptm MOE
    # (maybe we need _experts_shard_placement_fn as in https://github.com/pytorch/torchtitan/blob/fbafd44da2baef0afac58989f07d799c4251bdef/torchtitan/experiments/transformers_modeling_backend/infra/parallelize.py#L350)
    # In addition, sharding ep makes it hard to broadcast parameters during model initialization in dp_replicate mode
    enable_expert_fsdp_sharding = False 

    _is_built = False

    # Type hints for dynamically set attributes (set in _build_mesh_with_ep)
    # These attributes are set for each key in mesh_mapping: 'etp', 'ep', 'cp', 'sp', 'pp', 'tp', 'dp'
    # Note: dp_enabled is a @property method, not a dynamically set attribute
    if TYPE_CHECKING:
        from torch.distributed import ProcessGroup
        
        # etp attributes
        etp_enabled: bool
        etp_group: ProcessGroup | None
        etp_mesh: DeviceMesh | None
        etp_rank: int
        etp_size: int

        # ep_etp attributes
        ep_etp_enabled: bool
        ep_etp_group: ProcessGroup | None
        ep_etp_mesh: DeviceMesh | None
        ep_etp_rank: int
        ep_etp_size: int
        
        # ep attributes
        ep_enabled: bool
        ep_group: ProcessGroup | None
        ep_mesh: DeviceMesh | None
        ep_rank: int
        ep_size: int
        
        # cp attributes
        cp_enabled: bool
        cp_group: ProcessGroup | None
        cp_mesh: DeviceMesh | None
        cp_rank: int
        cp_size: int
        
        # sp attributes (compatibility with old code, same as cp)
        sp_enabled: bool
        sp_group: ProcessGroup | None
        sp_mesh: DeviceMesh | None
        sp_rank: int
        sp_size: int
        
        # pp attributes
        pp_enabled: bool
        pp_group: ProcessGroup | None
        pp_mesh: DeviceMesh | None
        pp_rank: int
        pp_size: int
        
        # tp attributes
        tp_enabled: bool
        tp_group: ProcessGroup | None
        tp_mesh: DeviceMesh | None
        tp_rank: int
        tp_size: int
        
        # dp attributes (dp_enabled is a @property, so not included here)
        dp_group: ProcessGroup | None
        dp_mesh: DeviceMesh | None
        dp_rank: int
        dp_size: int

    def __post_init__(self):
        self._validate()
        self.build_mesh(device_type=device_type)

    def _validate(self):
        if self.ep > 1 and self.tp > 1 and self.etp == 1:
            self.etp = self.tp # Fix the wrong etp value provided by user
            # assert self.tp == self.ep, 'We only support ETp with equal tp and ep size'

        # compatibility with old code
        self.sp = self.cp

        dp_replicate, dp_shard, cp, tp, pp, ep, world_splits = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.world_splits,
        )
        for d in (cp, tp, pp, ep, self.etp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard and dp_replicate"

        # if ep > 1:
        #     # assert cp == 1, "Current EP implementation can not work with CP."
        #     assert tp == 1, "Current EP implementation can not work with TP."

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must be -1 or >=1."

        original_dp_shard = dp_shard
        original_dp_replicate = dp_replicate

        if dp_replicate == -1:

            if self.dp_shard == -1:
                assert self.world_size % (tp * pp * world_splits) == 0
                if self.world_size // (tp * pp * world_splits) >= 8:
                    self.dp_shard = dp_shard = min(8, self.world_size // (tp * pp * world_splits))
                    assert self.world_size // (tp * pp * world_splits) % self.dp_shard == 0, f'{self.world_size=}, {tp=}, {pp=}, {world_splits=}, {self.dp_shard=}'

                    self.dp_replicate = self.world_size // (tp * pp * world_splits * self.dp_shard)
                    assert self.world_size % (tp * pp * world_splits * self.dp_shard) == 0
                else:
                    self.dp_shard = self.world_size // (tp * pp * world_splits)
                    self.dp_replicate = 1
            else:
                self.dp_replicate = dp_replicate = self.world_size // (self.dp_shard * tp * pp * world_splits)
                assert self.dp_replicate == self.world_size // (self.dp_shard * tp * pp * world_splits), f'{self.dp_replicate=}, {self.world_size=}, {self.dp_shard=}, {tp=}, {pp=}, {world_splits=}'

        else:
            if dp_shard < 0:
                self.dp_shard = dp_shard = self.world_size // (self.dp_replicate * tp * pp * world_splits)
            else:
                self.dp_shard = dp_shard

        assert self.dp_shard >= 1

        assert self.dp_replicate * self.dp_shard * tp * pp * world_splits == self.world_size, (
            f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) * world_splits({world_splits}) != WORLD_SIZE({self.world_size})"
        )

        model_size = self.tp * self.pp * self.cp
        if self.world_size % model_size != 0:
            raise RuntimeError(f"world_size ({self.world_size}) is not divisible by {model_size}")

            # Build expert rank generator

        expert_tensor_parallel_size = self.etp
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size = self.tp
        expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size * self.ep * self.pp
        )
        self.expert_data_parallel_size = self.world_size // expert_tensor_model_pipeline_parallel_size
        if self.world_size % expert_tensor_model_pipeline_parallel_size != 0:
            raise RuntimeError(
                f"world_size ({self.world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
            )


        # if ep > 1:
            # EP would borrow all cp and some dp_shard degree
            # assert ep % cp == 0, f'{ep=}, {cp=}, {dp_shard=}'

    def safe_div(self, a, b):
        assert a % b == 0, f'{a=}, {b=}'
        return a // b

    def __build_mesh_with_ep(self, device_type: str):
        """
        self.pp,
        self.dp_replicate,                                                                      # dense dp-allreduce    # moe dp-allreduce

        # multiply -> self.dp_shard
        dp_shard_mod_ep, # where local expert fsdp-shard        # where dense fsdp-shard        # dense dp-allreduce    # moe dp-allreduce
        dp_shard_in_ep,  # ep (when cp==1)                      # where dense fsdp-shard        # dense dp-allreduce

        self.cp,         # ep                                   # where dense fsdp-shard        # dense dp-allreduce
        self.tp,
        """
        # TODO: 判断 common replicate

        self.gpu_wise_mesh = init_device_mesh(device_type, (self.world_size,), mesh_dim_names=('gpu',), timeout=self.timeout)

        self.device_mesh_for_default_fsdp = init_device_mesh(
            device_type, (self.world_splits, self.pp, self.dp_replicate, self.dp_shard, self.tp), mesh_dim_names=('world_splits', 'pp', 'dp_replicate', 'dp_shard', 'tp'), timeout=self.timeout
        )
        if self.dp_replicate > 1:
            self.default_fsdp_mesh = self.device_mesh_for_default_fsdp['dp_replicate', 'dp_shard']
        else:
            # self.device_mesh_for_default_fsdp = init_device_mesh(device_type, (self.world_splits, self.pp, self.dp_shard, self.tp), mesh_dim_names=('world_splits', 'pp', 'dp_shard', 'tp'))
            self.default_fsdp_mesh = self.device_mesh_for_default_fsdp['dp_shard']

        if self.ep > 1:
            ep_shardable_space = self.world_size // (self.world_splits * self.pp * self.ep * self.etp)
            assert ep_shardable_space == self.expert_data_parallel_size, f'{ep_shardable_space=}, {self.expert_data_parallel_size=} ({self.ep=} {self.tp=} {self.etp=})'
            # expert_data_parallel_size = self.world_size // expert_tensor_model_pipeline_parallel_size
            assert self.world_size % (self.world_splits * self.pp * self.ep * self.etp) == 0, f'{self.world_size=}, {self.world_splits=}, {self.pp=}, {self.ep=}, {self.etp=}'
            if self.enable_expert_fsdp_sharding:
                raise NotImplementedError(
                    'EP sharding is not implemented yet. '
                    'ep sharding will produce zero shape weights, which is incompatible with ptm MOE. '
                    'In addition, sharding ep makes it hard to broadcast parameters during model initialization in dp_replicate mode. '
                )
                self.device_mesh_for_ep = init_device_mesh(
                    device_type, (self.world_splits, self.pp, ep_shardable_space, self.ep, self.etp), mesh_dim_names=('world_splits', 'pp', 'ep_shardable', 'ep', 'etp'), timeout=self.timeout
                )
                self.expert_fsdp_mesh = self.device_mesh_for_ep['ep_shardable']
                self.ep_related_mesh = self.device_mesh_for_ep[('ep_shardable', 'ep')]
            else:
                # In this implementation, even if we don't want to apply FSDP sharding on experts,
                # we still create a dummy FSDP mesh (replicate=n, shard=1).
                # Note: This dummy mesh may not be used if we skip the apply_fsdp function.

                self.device_mesh_for_ep = init_device_mesh(
                    device_type,
                    (self.world_splits, self.pp, ep_shardable_space, 1, self.ep, self.etp),
                    mesh_dim_names=('world_splits', 'pp', 'ep_fsdp_replicate', 'ep_fsdp_shard', 'ep', 'etp'),
                    timeout=self.timeout
                )
                self.expert_shard_mesh = self.device_mesh_for_ep['ep_fsdp_shard'] # helps determine whether to use `shard_placement_fn`
                self.expert_fsdp_mesh = self.device_mesh_for_ep['ep_fsdp_replicate', 'ep_fsdp_shard']
                # self.ep_related_mesh = self.device_mesh_for_ep['ep_fsdp_replicate', 'ep_fsdp_shard', 'ep']
                self.ep_related_mesh = self.device_mesh_for_ep['ep_fsdp_replicate', 'ep_fsdp_shard', 'ep', 'etp']
        else:
            self.expert_fsdp_mesh = self.default_fsdp_mesh
            # if self.dp_shard > 1:
                # assert self.enable_expert_fsdp_sharding

        dp = self.world_size // (self.world_splits * self.pp * self.cp * self.tp)
        assert self.world_size % (self.world_splits * self.pp * self.cp * self.tp) == 0
        self.device_mesh_for_pp_dp_cp_tp = init_device_mesh(device_type, (self.world_splits, self.pp, dp, self.cp, self.tp), mesh_dim_names=('world_splits', 'pp', 'dp', 'cp', 'tp'), timeout=self.timeout)
        self.device_mesh_for_pp_dp_cp_tp['pp', 'cp', 'tp']._flatten(mesh_dim_name='non_dp')


        dataloading_mesh = self.device_mesh_for_pp_dp_cp_tp
        loss_mesh = self.device_mesh_for_pp_dp_cp_tp["dp", "cp"]._flatten("loss_mesh")
        dense_mesh = self.device_mesh_for_default_fsdp

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
        }

        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["dp"],
            "dp": dataloading_mesh["dp"],
            "loss": loss_mesh,
            "fsdp": dense_mesh["dp_shard"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
        }
        if self.ep > 1:
            # TODO: create dummy mesh with size 1 when ep==1 and remove this `if` branch
            sparse_mesh = self.device_mesh_for_ep
            self._global_meshes["sparse"] = sparse_mesh
            self._meshes.update({
                "ep": sparse_mesh["ep"],
                "efsdp": sparse_mesh["ep_fsdp_shard"],
                "etp": sparse_mesh["etp"],
            })
        if self.dp_replicate > 1:
            self._meshes["dp_replicate"] = dense_mesh["dp_replicate"]


        mesh_mapping = {
            'ep_etp': ('device_mesh_for_ep', ('ep', 'etp')),

            'etp': ('device_mesh_for_ep', 'etp'),
            'ep': ('device_mesh_for_ep', 'ep'),
            'cp': ('device_mesh_for_pp_dp_cp_tp', 'cp'),
            'sp': ('device_mesh_for_pp_dp_cp_tp', 'cp'), # compatibility with old code
            'pp': ('device_mesh_for_pp_dp_cp_tp', 'pp'),

            # 'tp': ('device_mesh_for_pp_dp_cp_tp', 'tp'),
            'tp': ('device_mesh_for_default_fsdp', 'tp'), # AssertionError: FSDP requires the DP and model parallel TP/EP mesh to have the same parent mesh

            'dp': ('device_mesh_for_pp_dp_cp_tp', 'dp'),


            # 'dp_replicate': ('device_mesh_for_default_fsdp', 'dp_replicate'),
            # 'dp_shard': ('device_mesh_for_default_fsdp', 'dp_shard'),
        }

        # HACK(kevinkhwu): 
        # DeviceMesh.get_group() calls Tensor.tolist()
        # We pre-cache the group to avoid calling tolist() in jvp.
        # See also: tolist issue in jvp: https://github.com/pytorch/pytorch/issues/161943 (Seems to be fixed)
        for key, (mesh_attr, mesh_dim) in mesh_mapping.items():

            if not hasattr(self, mesh_attr): # fall back and set default values
                setattr(self, f'{key}_enabled', False)
                setattr(self, f'{key}_group', None)
                setattr(self, f'{key}_mesh', None)
                setattr(self, f'{key}_rank', 0)
                setattr(self, f'{key}_size', 1)
                continue

            mesh = getattr(self, mesh_attr)[mesh_dim]
            if (isinstance(mesh_dim, tuple) or isinstance(mesh_dim, list)) and len(mesh_dim) > 1:
                ...
            else:
                setattr(self, f'{key}_group', mesh.get_group())
                setattr(self, f'{key}_rank', mesh.get_local_rank())
            setattr(self, f'{key}_mesh', mesh)
            setattr(self, f'{key}_size', mesh.size())
            if hasattr(self, key):
                setattr(self, f'{key}_enabled', getattr(self, key) > 1)


    @property
    def world_mesh(self):
        raise DeprecationWarning('world_mesh is deprecated')

    def build_mesh(self, device_type='cuda'):
        if self._is_built:
            return
        self._is_built = True

        self.__build_mesh_with_ep(device_type)

        import torch.distributed.tensor._random as random
        from torch.distributed.tensor._random import (
            is_rng_supported_mesh,
            OffsetBasedRNGTracker,
        )

        device_mesh = self.device_mesh_for_pp_dp_cp_tp
        device_type = device_mesh.device_type
        # https://github.com/pytorch/pytorch/issues/157662
        if not random._rng_tracker and is_rng_supported_mesh(device_mesh):
            from packaging import version
            if version.parse(torch.__version__) >= version.parse("2.7.1"):
                random._rng_tracker = OffsetBasedRNGTracker(device_mesh)
            else:
                random._rng_tracker = OffsetBasedRNGTracker(device_type)

        if dist.get_rank() == 0:
            loguru.logger.info(f'n_replicate: {self.dp_replicate}')
            if self.ep_enabled:
                loguru.logger.info(f'EP mesh: {self.ep_mesh}')
                loguru.logger.info(f'Expert shard size: {self.expert_fsdp_mesh.size(-1)}')
                loguru.logger.info(f'Expert fsdp mesh: {self.expert_fsdp_mesh}')

    def is_replica_leader_rank(self):
        ret = True
        if self.pp_enabled:
            ret = ret and self.pp_mesh.get_local_rank() == self.pp - 1
        if self.tp_enabled:
            ret = ret and self.tp_mesh.get_local_rank() == 0
        if self.cp_enabled:
            ret = ret and self.cp_mesh.get_local_rank() == 0
        return ret


    def is_log_rank(self, dp_rank=0):
        if dp_rank == -1:
            ret = True
        else:
            ret = self.dp_mesh.get_local_rank() == dp_rank
        if self.pp_enabled:
            ret = ret and self.pp_mesh.get_local_rank() == self.pp - 1
        if self.tp_enabled:
            ret = ret and self.tp_mesh.get_local_rank() == 0
        if self.cp_enabled:
            ret = ret and self.cp_mesh.get_local_rank() == 0
        return ret

    @property
    def dp_enabled(self):
        return self.device_mesh_for_pp_dp_cp_tp['dp'].size() > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_replicate_mesh(self):
        return self.device_mesh_for_default_fsdp['dp_replicate']

    @property
    def dp_replicate_group(self):
        return self.device_mesh_for_default_fsdp['dp_replicate'].get_group()

    @property
    def dp_replicate_rank(self):
        return self.device_mesh_for_default_fsdp['dp_replicate'].get_local_rank()

    @property
    def dp_replicate_size(self):
        return self.device_mesh_for_default_fsdp['dp_replicate'].size()

    @property
    def world_splits_group(self):
        return self.device_mesh_for_pp_dp_cp_tp['world_splits'].get_group()

    @property
    def world_splits_rank(self):
        return self.device_mesh_for_pp_dp_cp_tp['world_splits'].get_local_rank()

    @property
    def world_splits_size(self):
        return self.world_splits

    def get_local_mesh(self):
        device_count = torch.cuda.device_count()
        assert device_count == 8
        n_node = dist.get_world_size() // device_count
        node_mesh = init_device_mesh('cuda', (n_node, device_count), mesh_dim_names=('nodes', 'gpus'), timeout=self.timeout)
        return node_mesh['gpus']

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        # return self.dp_replicate * self.dp_shard * self.sp

        # Our dp_shard is different from torchtitan's dp_shard.
        # our dp_shard = torchtitan's dp_shard * cp
        return self.dp_replicate * self.dp_shard

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def non_dp_mesh(self):
        return self.device_mesh_for_pp_dp_cp_tp['non_dp']

    @property
    def non_dp_group(self):
        return self.non_dp_mesh.get_group()

    @property
    def non_dp_rank(self):
        return self.non_dp_mesh.get_local_rank()

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)

    def __str__(self):
        return f'ParallelDims(pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, ep={self.ep}, cp={self.cp}, tp={self.tp}, world_size={self.world_size})'

    def get_mesh(self, name: str) -> DeviceMesh:
        if hasattr(self, f'{name}_mesh'):
            return getattr(self, f'{name}_mesh')
        else:   
            raise ValueError(f'Mesh {name} not found')

@dataclass
class TitanParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    etp: int
    world_size: int
    timeout: datetime.timedelta | float | None = None

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep, etp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.etp,
        )
        for d in (dp_replicate, cp, tp, pp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"

    def _mesh_exist(self, name: str, degree: int) -> bool:
        if name == "efsdp":
            # We always keep the efsdp if EP is larger than 1 because we need
            # FSDP wrapping to help the MoE layers do mixed precision training.
            return True if self.ep > 1 else False
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """
        Build the device mesh with the required mesh dimensions.

        The following mesh dimensions will be created:

            pp:      Pipeline Parallelism (PP).
            batch:   Used by data loading to determine the global batch size and which
                     part of the data each rank should read. This dimension includes both
                     ``dp_replicate`` and ``dp_shard``. The backend is set to ``fake`` for
                     this dimension to avoid unnecessary process group creation.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                     ``dp_shard``, and ``cp`` degrees, as all of them parallelize the data,
                     essentially require the weight gradients reduction.
            dp_replicate: For DDP or HSDP replicate dimension.
            fsdp:    For FSDP dimension. This includes ``dp_shard`` and ``cp``. Note that
                     we always assume that when ``cp`` is used, FSDP is also applied to
                     utilize its weight all-gather and gradients reduce_scatter even if
                     there may be no data parallelism (e.g., global batch size is 1).
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            efsdp:   FSDP in the EP region.
            etp:     TP in the EP region.

        Note: Most dimensions above are created by unflattening the world mesh, except for loss,
        which is created by flattening the batch and cp dimensions.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "batch", "cp", "tp"]  # dataloading_mesh
            ["pp", "dp_replicate", "fsdp", "tp"]  # dense_mesh
            ["pp", "dp_replicate", "efsdp", "ep", "etp"]  # sparse_mesh

        Note: DeviceMesh currently recreates the process group for each dimension.
        It should share the process group for the same dim group to avoid unnecessary
        process group creation. We can also use Fake to achieve a similar goal.
        However, using Fake to avoid redundancy messing up the code. We only use Fake
        when it is necessary. For now, we just let DeviceMesh create redundant process
        group and wait for DeviceMesh to fix the issue.
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            """Unflatten the world mesh to create the required mesh dimensions.

            Uses fake backend for dimensions with degree 1 or for 'batch' dimension
            to avoid unnecessary process group creation.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if (not self._mesh_exist(name, degree)) or name == "batch":
                    backend_override[name] = "fake"
            if hasattr(world_mesh, "_unflatten"):
                return world_mesh._unflatten(
                    0, dim_degrees, dim_names, backend_override=backend_override
                )
            else:
                raise RuntimeError(f'world_mesh {world_mesh} does not have _unflatten method. Your Pytorch version is {torch.__version__}.')
                # HACK(kevinkhwu): Support for older versions of PyTorch.
                return init_device_mesh(device_type, dim_degrees, mesh_dim_names=dim_names, timeout=self.timeout)

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}, etp={self.etp}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // (self.etp * self.ep)

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",), timeout=self.timeout
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        dense_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "fsdp", "tp"),
            (self.pp, self.dp_replicate, fsdp, self.tp),
        )
        sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
            "sparse": sparse_mesh,
        }

        self._meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": sparse_mesh["ep"],
            "efsdp": sparse_mesh["efsdp"],
            "etp": sparse_mesh["etp"],
        }

        # Validate mesh sizes
        self._validate_meshes()

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes."""
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // (self.etp * self.ep),
            "etp": self.etp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._meshes[mesh_name].size()
            assert actual_size == expected_size, (
                f"Mesh '{mesh_name}' has unexpected size: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist (except efsdp which can exist even if size is 1 when ep > 1)

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._meshes.keys())}"
                )

        if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
            return None

        if len(dims) == 1:
            return self._meshes[dims[0]]
        else:
            for global_mesh in self._global_meshes.values():
                assert global_mesh.mesh_dim_names is not None
                if not set(dims).issubset(set(global_mesh.mesh_dim_names)):
                    continue
                return global_mesh[tuple(dims)]
            raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'etp', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s).

        Raises:
            ValueError: If the mesh is not available (dimension size = 1 or not enabled),
                or if the requested dimension name(s) is not valid.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary of enabled one-dimensional device meshes, allowing you to
        access their process groups.

        Note:
            Device meshes created with the Fake backend are still included in the results.

        Returns:
            dict[str, DeviceMesh]: A dictionary mapping mesh dimension names to their
                corresponding DeviceMesh objects. Only includes meshes where:
                - ndim == 1 (one-dimensional)
                - parallelism is enabled (size > 1)

        Example:
            >>> parallel_dims = ParallelDims(
            ...     dp_replicate=2, dp_shard=2, cp=1, tp=2, pp=1, ep=1, etp=1, world_size=8
            ... )
            >>> meshes = parallel_dims.get_all_one_dimensional_meshes()
            >>> print(meshes.keys())
            dict_keys(['dp_replicate', 'fsdp', 'tp', 'batch', 'loss', 'efsdp'])
        """
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def etp_enabled(self):
        return self.etp > 1

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self.dp_replicate * self.dp_shard * self.cp

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp * (self.cp * 2)

class HyTitanParallelDims(TitanParallelDims):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dp_replicate_enabled:
            self.default_fsdp_mesh = self.get_mesh(("dp_replicate", "fsdp"))
        else:
            self.default_fsdp_mesh = self.get_mesh(("fsdp"))

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if self.dp_replicate_enabled
            else ["efsdp"]
        )
        self.expert_fsdp_mesh = self.get_optional_mesh(edp_mesh_names)

if version.parse(torch.__version__) >= version.parse("2.20.1"):
    ParallelDims = HyTitanParallelDims
else:
    ParallelDims = HYParallelDims

_PARALLEL_STATE_DICT = {}
_DEFAULT_PARALLEL_STATE_KEY = 'default'
_PARALLEL_STATE_KEY = _DEFAULT_PARALLEL_STATE_KEY
_UNDER_DEVICE_MESH_CONTEXT = False

def get_or_init_parallel_state(
    dp_replicate: int = 1,
    dp_shard: int = -1,
    sp: int = 1, # legacy
    cp: int = 1,
    etp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    world_size: int = None,
    mesh_tag='default',
):
    if mesh_tag in _PARALLEL_STATE_DICT:
        return _PARALLEL_STATE_DICT[mesh_tag]
    return init_parallel_state(dp_replicate, dp_shard, sp, cp, etp, tp, pp, ep, world_size, mesh_tag)

def init_parallel_state(
    dp_replicate: int = 1,
    dp_shard: int = -1,
    sp: int = 1, # legacy
    cp: int = 1,
    etp: int = 1,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    world_size: int = None,
    mesh_tag=_DEFAULT_PARALLEL_STATE_KEY,
    timeout: datetime.timedelta | float | None = None,
):
    """
    Initializes global parallel state.
    """
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
    global _PARALLEL_STATE_DICT
    if mesh_tag in _PARALLEL_STATE_DICT:
        raise RuntimeError(f'Parallel state already initialized for mesh {mesh_tag}')

    if sp > 1 and cp > 1:
        raise ValueError("sp is kept only for legacy compatibility, but cp is preferred.")

    # HACK(kevinkhwu): 
    #     This is a workaround for https://github.com/pytorch/pytorch/issues/173789
    #     Temporarily set _PARALLEL_STATE_KEY, which is used to compute the hash of DeviceMesh,
    #     to avoid the hash collision (in monkey_patches/torch.py).
    global _PARALLEL_STATE_KEY
    original_parallel_state_key = _PARALLEL_STATE_KEY
    _PARALLEL_STATE_KEY = mesh_tag


    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,  # world_size//8 在单个Node下进行切片
        dp_shard=dp_shard,
        cp=max(sp, cp),
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        world_size=world_size,
        timeout=timeout,
    )
    parallel_dims.mesh_tag = mesh_tag
    _PARALLEL_STATE_DICT[mesh_tag] = parallel_dims
    parallel_dims.build_mesh()
    _PARALLEL_STATE_KEY = original_parallel_state_key
    return parallel_dims


@contextmanager
def device_mesh_context(mesh_tag=None, reentrant=True):

    global _PARALLEL_STATE_DICT, _PARALLEL_STATE_KEY, _UNDER_DEVICE_MESH_CONTEXT
    if _UNDER_DEVICE_MESH_CONTEXT and not reentrant and mesh_tag != _PARALLEL_STATE_KEY:
        raise RuntimeError(f'Root context sets reentrant=False with mesh {_PARALLEL_STATE_KEY}. Trying to enter another mesh {mesh_tag} is not allowed.')

    _UNDER_DEVICE_MESH_CONTEXT = True
    if mesh_tag is None:
        mesh_tag = _PARALLEL_STATE_KEY
    if mesh_tag not in _PARALLEL_STATE_DICT:
        raise RuntimeError(f'Parallel state not initialized for mesh {mesh_tag}')
    old_tag = _PARALLEL_STATE_KEY
    _PARALLEL_STATE_KEY = mesh_tag
    yield
    _PARALLEL_STATE_KEY = old_tag
    _UNDER_DEVICE_MESH_CONTEXT = False

def is_parallel_state_initialized(key=None) -> bool:
    if key is None:
        key = _PARALLEL_STATE_KEY
    return key in _PARALLEL_STATE_DICT

def get_parallel_state() -> HYParallelDims | HyTitanParallelDims:
    """
    Returns global parallel state.
    """
    parallel_dims = _PARALLEL_STATE_DICT.get(_PARALLEL_STATE_KEY, None)
    if parallel_dims is None:
        if os.environ.get('SUPRESS_HY_PARALLELISM_LOG', '0') == '0':
            loguru.logger.opt(depth=1).warning("Parallel state has not been initialized. returning default Single-process state.")
        # raise ValueError(f'_PARALLEL_STATE is not inited yet')
        return ParallelDims(
            dp_replicate=1,  # world_size//8 在单个Node下进行切片
            dp_shard=-1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=int(os.environ.get('WORLD_SIZE', '1')),
        )

    return parallel_dims
