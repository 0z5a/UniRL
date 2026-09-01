import loguru
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard


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
class ParallelDims:
    dp_replicate: int # setting dp_replicate to -1 will enable HSDP if possible
    dp_shard: int
    sp: int
    tp: int
    pp: int
    ep: int
    world_size: int

    enable_ep_sharding = False

    ep_fsdp_replicate: int = -1 # TODO:
    ep_fsdp_shard: int = -1

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, sp, tp, pp, ep = (
            self.dp_replicate,
            self.dp_shard,
            self.sp,
            self.tp,
            self.pp,
            self.ep,
        )
        for d in (sp, tp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard and dp_replicate"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        original_dp_shard = dp_shard
        original_dp_replicate = dp_replicate

        if dp_replicate == -1:

            if self.dp_shard == -1:
                assert self.world_size % (tp * pp) == 0
                if self.world_size // (tp * pp) >= 8:
                    self.dp_shard = dp_shard = 8
                    self.dp_replicate = self.world_size // (tp * pp * self.dp_shard)
                    assert self.world_size % (tp * pp * self.dp_shard) == 0
                else:
                    self.dp_shard = self.world_size // (tp * pp)
                    self.dp_replicate = 1
            else:
                # 明明可以自动设，非要手动设，那就检查一下吧
                assert self.dp_replicate == self.world_size // (self.dp_shard * tp * pp)


        else:
            if dp_shard < 0:
                self.dp_shard = dp_shard = self.world_size // (self.dp_replicate * tp * pp)
            else:
                self.dp_shard = dp_shard


        assert self.dp_shard >= 1

        assert self.dp_replicate * self.dp_shard * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) * "
            f"sp({sp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            # EP would borrow all sp and some dp_shard degree
            assert (ep % sp == 0), f'{ep=}, {sp=}, {dp_shard=}'


    def safe_div(self, a, b):
        assert a % b == 0, f'{a=}, {b=}'
        return a // b


    def _build_mesh_with_ep(self, device_type: str):
        """
                self.pp,
                self.dp_replicate,

                # multiply -> self.dp_shard
                dp_shard_mod_ep, # where local expert fsdp-shard        # where dense fsdp-shard
                dp_shard_in_ep,  # ep (when sp==1)                      # where dense fsdp-shard

                self.sp,         # ep                                   # where dense fsdp-shard
                self.tp,
        """
        # TODO: 判断 common replicate

        if self.dp_replicate > 1:
            self.device_mesh_for_default_fsdp = init_device_mesh(device_type, (self.pp, self.dp_replicate, self.dp_shard, self.tp), mesh_dim_names=('pp', 'dp_replicate', 'dp_shard', 'tp'))
            self.default_fsdp_mesh = self.device_mesh_for_default_fsdp['dp_replicate', 'dp_shard']
        else:
            self.device_mesh_for_default_fsdp = init_device_mesh(device_type, (self.pp, self.dp_shard, self.tp), mesh_dim_names=('pp', 'dp_shard', 'tp'))
            self.default_fsdp_mesh = self.device_mesh_for_default_fsdp['dp_shard']

        if self.ep > 1:
            ep_shardable_space = self.world_size // (self.pp  * self.ep * self.tp)
            assert self.world_size % (self.pp * self.ep * self.tp) == 0, f'{self.world_size=}, {self.pp=}, {self.ep=}, {self.tp=}'
            if self.enable_ep_sharding:
                self.device_mesh_for_ep = init_device_mesh(device_type, (self.pp, ep_shardable_space, self.ep, self.tp), mesh_dim_names=('pp', 'ep_shardable', 'ep', 'tp'))
                self.expert_fsdp_mesh = self.device_mesh_for_ep['ep_shardable']
                self.ep_related_mesh = self.device_mesh_for_ep[('ep_shardable', 'ep')]
            else:
                self.device_mesh_for_ep = init_device_mesh(device_type, (self.pp, ep_shardable_space, 1, self.ep, self.tp), mesh_dim_names=('pp', 'ep_fsdp_replicate', 'ep_fsdp_shard', 'ep', 'tp'))
                self.expert_fsdp_mesh = self.device_mesh_for_ep['ep_fsdp_replicate', 'ep_fsdp_shard']
                self.ep_related_mesh = self.device_mesh_for_ep['ep_fsdp_replicate', 'ep_fsdp_shard', 'ep']
        else:
            self.expert_fsdp_mesh = self.default_fsdp_mesh
            if self.dp_shard > 1:
                assert self.enable_ep_sharding

        dp = self.world_size // (self.pp * self.sp * self.tp)
        assert self.world_size % (self.pp * self.sp * self.tp) == 0
        self.device_mesh_for_pp_dp_sp_tp = init_device_mesh(device_type, (self.pp, dp, self.sp, self.tp), mesh_dim_names=('pp', 'dp', 'sp', 'tp'))

    @property
    def world_mesh(self):
        raise DeprecationWarning('world_mesh is deprecated')


    def build_mesh(self, device_type):
        self._build_mesh_with_ep(device_type)

        import torch.distributed.tensor._random as random
        from torch.distributed.tensor._random import (
            is_rng_supported_mesh,
            OffsetBasedRNGTracker,
        )

        device_mesh = self.device_mesh_for_pp_dp_sp_tp
        device_type = device_mesh.device_type
        if not random._rng_tracker and is_rng_supported_mesh(device_mesh):
            random._rng_tracker = OffsetBasedRNGTracker(device_type)


        if dist.get_rank() == 0:
            loguru.logger.info(f'n_replicate: {self.dp_replicate}')
            if self.ep_enabled:
                loguru.logger.info(f'EP mesh: {self.ep_mesh}')
                loguru.logger.info(f'Expert shard size: {self.expert_fsdp_mesh.size(-1)}')
                loguru.logger.info(f'Expert fsdp mesh: {self.expert_fsdp_mesh}')



    @property
    def dp_replicate_mesh(self):
        return self.device_mesh_for_default_fsdp[('dp_replicate')]

    @property
    def dp_enabled(self):
        return self.device_mesh_for_pp_dp_sp_tp['dp'].size() > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.device_mesh_for_default_fsdp['dp_shard'].size() > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1


    @property
    def dp_mesh(self):
        return self.device_mesh_for_pp_dp_sp_tp['dp']

    @property
    def ep_mesh(self):
        if self.ep_enabled:
            return self.device_mesh_for_ep['ep']

    @property
    def tp_mesh(self):
        return self.device_mesh_for_pp_dp_sp_tp['tp']

    @property
    def sp_mesh(self):
        return self.device_mesh_for_pp_dp_sp_tp['sp']

    @property
    def pp_mesh(self):
        return self.device_mesh_for_pp_dp_sp_tp['pp']

    @property
    def ep_group(self):
        if self.ep_enabled:
            return self.ep_mesh.get_group()

    @property
    def tp_group(self):
        return self.tp_mesh.get_group()

    @property
    def sp_group(self):
        return self.sp_mesh.get_group()

    @property
    def pp_group(self):
        return self.pp_mesh.get_group()

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    # @cached_property
    # def non_data_parallel_size(self):
    #     return self.sp * self.tp * self.pp

    def __str__(self):
        return f'ParallelDims(pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, ep={self.ep}, sp={self.sp}, tp={self.tp}, world_size={self.world_size})'


_PARALLEL_STATE_DICT = {}
_PARALLEL_STATE_KEY = 'default'
def init_parallel_state(
    dp_replicate: int=1,
    dp_shard: int=-1,
    sp: int=1,
    tp: int=1,
    pp: int=1,
    ep: int=1,
    world_size: int=None,
    mesh_tag='default',
):
    """
    Initializes global parallel state.
    """
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
    global _PARALLEL_STATE_DICT
    if mesh_tag in _PARALLEL_STATE_DICT:
        raise RuntimeError(f'Parallel state already initialized for mesh {mesh_tag}')

    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate, # world_size//8 在单个Node下进行切片
        dp_shard=dp_shard,
        sp=sp,
        tp=tp,
        pp=pp,
        ep=ep,
        world_size=world_size,
    )
    _PARALLEL_STATE_DICT[mesh_tag] = parallel_dims
    return parallel_dims



from contextlib import contextmanager
@contextmanager
def device_mesh_context(mesh_tag=None):
    global _PARALLEL_STATE_DICT, _PARALLEL_STATE_KEY
    if mesh_tag is None:
        mesh_tag = _PARALLEL_STATE_KEY
    if mesh_tag not in _PARALLEL_STATE_DICT:
        raise RuntimeError(f'Parallel state not initialized for mesh {mesh_tag}')
    old_tag = _PARALLEL_STATE_KEY
    _PARALLEL_STATE_KEY = mesh_tag
    yield
    _PARALLEL_STATE_KEY = old_tag


def get_parallel_state() -> ParallelDims:
    """
    Returns global parallel state.
    """
    parallel_dims = _PARALLEL_STATE_DICT.get(_PARALLEL_STATE_KEY, None)
    if parallel_dims is None:
        loguru.logger.warning("Parallel state has not been initialized. returning default Single-process state.")
        # raise ValueError(f'_PARALLEL_STATE is not inited yet')
        return ParallelDims(
            dp_replicate=1, # world_size//8 在单个Node下进行切片
            dp_shard=-1,
            sp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=int(os.environ.get('WORLD_SIZE', '1')),
        )

    return parallel_dims


class COMM_INFO:
    def __init__(self):
        self.sp_group = None
        self.tp_group = None
        self.sp_size = 1
        self.tp_size = 1
        self.global_rank = 0
        self.rank_within_spgroup = 0
        self.rank_within_tpgroup = 0
        self.parallel_dims = None
        self.device_mesh = None
        self.use_dynamic_ring_attention = False
        self.sp_stream = None
        self.sp_rank_list = None
        

nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_TEACHER_STUDENT_PARALLEL_STATE = False


def initialize_sequence_parallel_state_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    global _SEQUENCE_PARALLEL_STATE
    if parallel_dims.sp_enabled or parallel_dims.tp_enabled:
        initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention)
    else:
        nccl_info.sp_size = 1
        nccl_info.tp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_spgroup = 0
        nccl_info.rank_within_tpgroup = 0
        nccl_info.sp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.tp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.parallel_dims = parallel_dims
        nccl_info.device_mesh = device_mesh


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def set_tensor_parallel_state(state):
    global _TENSOR_PARALLEL_STATE
    _TENSOR_PARALLEL_STATE = state

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def get_tensor_parallel_state():
    return _TENSOR_PARALLEL_STATE

def initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    nccl_info.sp_size = parallel_dims.sp
    nccl_info.tp_size = parallel_dims.tp
    nccl_info.global_rank = rank
    nccl_info.parallel_dims = parallel_dims
    nccl_info.device_mesh = device_mesh
    nccl_info.use_dynamic_ring_attention = use_dynamic_ring_attention
    if use_dynamic_ring_attention:
        nccl_info.sp_stream = torch.cuda.Stream()
        nccl_info.sp_rank_list = device_mesh["sp"].mesh.tolist()
    if nccl_info.sp_size > 1:
        set_sequence_parallel_state(True)
        nccl_info.sp_group = device_mesh.get_group(mesh_dim="sp")
        nccl_info.rank_within_spgroup = device_mesh.get_local_rank(mesh_dim="sp")
    if nccl_info.tp_size > 1:
        set_tensor_parallel_state(True)
        nccl_info.tp_group = device_mesh.get_group(mesh_dim="tp")
        nccl_info.rank_within_tpgroup = device_mesh.get_local_rank(mesh_dim="tp")

def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i


def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()
