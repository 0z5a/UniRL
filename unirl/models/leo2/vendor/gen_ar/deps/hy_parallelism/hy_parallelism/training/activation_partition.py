# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from ast import main
import contextlib
from typing import Union
from warnings import warn

import psutil
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from loguru import logger
from torch.distributed.tensor import DTensor, Shard
# import torchao
# from torchao.dtypes.nf4tensor import NF4Tensor

# from torchtune.modules import TiedLinear
# from torchtune.utils import get_logger

# log = get_logger("DEBUG")


class PartitionActivations(saved_tensors_hooks):
    def __init__(self, mesh1d):

        self.mesh1d = mesh1d
        self.tracker = {}
        self.tensor_id = 0
        raise NotImplementedError("PartitionActivations is not tested yet")

        def pack_hook(tensor: torch.Tensor) -> int:
            placements = [Shard(0)]
            dtensor = DTensor.from_local(tensor, device_mesh=self.mesh1d).redistribute(placements=placements)
            tensor_id = self.get_tensor_id()
            self.tracker[tensor_id] = dtensor
            return tensor_id

        def unpack_hook(tensor_id: int) -> torch.Tensor:
            return self.tracker[tensor_id].full_tensor()

        super().__init__(pack_hook, unpack_hook)

    def get_tensor_id(self) -> int:
        # create a unique id for each tensor we are managing
        self.tensor_id += 1
        return self.tensor_id


if __name__ == '__main__':
    from hy_parallelism.training.activation_partition import PartitionActivations
    from hy_parallelism.parallel_states import get_parallel_state
    with PartitionActivations(get_parallel_state().pp_mesh):
        ...