from dataclasses import dataclass
from typing import Any

from torch.distributed import ProcessGroup

from .global_vars import set_parallel_state


@dataclass
class ParallelState:
    # Data parallel
    dp_rank: int
    dp_size: int
    dp_group: ProcessGroup = None
    # Tensor parallel
    tp_rank: int = 0
    tp_size: int = 1
    tp_group: ProcessGroup = None
    # Pipeline parallel
    pp_rank: int = 0
    pp_size: int = 1
    pp_group: ProcessGroup = None
    # Expert parallel
    ep_rank: int = 0
    ep_size: int = 1
    ep_group: ProcessGroup = None
    # Context parallel
    cp_rank: int = 0
    cp_size: int = 1
    cp_group: ProcessGroup = None

    backend: str = ''
    backend_state: Any = None

    def __post_init__(self):
        set_parallel_state(self)

    def __repr__(self):
        res = []
        res.append(f"dp: {self.dp_rank}/{self.dp_size}")
        res.append(f"tp: {self.tp_rank}/{self.tp_size}")
        res.append(f"pp: {self.pp_rank}/{self.pp_size}")
        res.append(f"cp: {self.cp_rank}/{self.cp_size}")
        return ", ".join(res)

    @staticmethod
    def from_megatron():
        from megatron.core import parallel_state
        p_state = ParallelState(
            dp_rank=parallel_state.get_data_parallel_rank(),
            dp_size=parallel_state.get_data_parallel_world_size(),
            dp_group=parallel_state.get_data_parallel_group(),
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            tp_size=parallel_state.get_tensor_model_parallel_world_size(),
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
            pp_size=parallel_state.get_pipeline_model_parallel_world_size(),
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
            ep_rank=parallel_state.get_expert_data_parallel_rank(),
            ep_size=parallel_state.get_expert_data_parallel_world_size(),
            ep_group=parallel_state.get_expert_data_parallel_group(),
            cp_rank=parallel_state.get_context_parallel_rank(),
            cp_size=parallel_state.get_context_parallel_world_size(),
            cp_group=parallel_state.get_context_parallel_group(),
            backend='megatron',
            backend_state=parallel_state,
        )
        print(f"ParallelState from megatron: {p_state}")
        return p_state

    @staticmethod
    def from_pure_torch():
        from hy_parallelism.parallel_states import get_parallel_state
        parallel_dims = get_parallel_state()
        if parallel_dims is None:
            raise RuntimeError("Parallel state is not initialized in pure torch environment.")
        p_state = ParallelState(
            dp_rank=parallel_dims.dp_mesh.get_local_rank(),
            dp_size=parallel_dims.dp_mesh.size(),
            dp_group=parallel_dims.dp_mesh.get_group(),
            tp_rank=parallel_dims.tp_group.rank() if parallel_dims.tp_enabled else 0,
            tp_size=parallel_dims.tp,
            tp_group=parallel_dims.tp_group if parallel_dims.tp_enabled else None,
            pp_rank=parallel_dims.pp_group.rank() if parallel_dims.pp_enabled else 0,
            pp_size=parallel_dims.pp,
            pp_group=parallel_dims.pp_group if parallel_dims.pp_enabled else None,
            ep_rank=parallel_dims.ep_group.rank() if parallel_dims.ep_enabled else 0,
            ep_size=parallel_dims.ep,
            ep_group=parallel_dims.ep_group if parallel_dims.ep_enabled else None,
            cp_rank=parallel_dims.cp_group.rank() if parallel_dims.cp_enabled else 0,
            cp_size=parallel_dims.cp,
            cp_group=parallel_dims.cp_group if parallel_dims.cp_enabled else None,
            backend='pure_torch',
            backend_state=parallel_dims,
        )
        return p_state

    def get_tensor_and_data_parallel_group(self, with_context_parallel: bool = False):
        if self.backend == 'megatron':
            from megatron.core import parallel_state
            return parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=with_context_parallel)
        elif self.backend == 'pure_torch':
            from hy_parallelism.parallel_states import get_parallel_state
            parallel_dims = get_parallel_state()
            return parallel_dims.device_mesh_for_pp_dp_cp_tp["loss_mesh"].get_group()
        else:
            raise NotImplementedError(f"get_tensor_and_data_parallel_group is not implemented for backend {self.backend}")
