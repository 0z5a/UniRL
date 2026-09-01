import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.tensor.placement_types import Replicate

from hy_parallelism.utils import gather_obj
from hy_parallelism.parallel_states import get_parallel_state, ParallelDims


def register_object(self, name: str, object):
    if not hasattr(self, '_returned_objects'):
        setattr(self, '_returned_objects', {})
    self._returned_objects[name] = object

def get_object(self, name: str, gather=False):
    if not hasattr(self, '_returned_objects'):
        setattr(self, '_returned_objects', {})
    if name not in self._returned_objects:
        raise ValueError(f"Object {name} not found")
    ret = self._returned_objects[name]
    if gather:
        ret = gather_obj(ret)
    return ret

def make_registerable(model):
    for m in model.modules():
        m.register_object = register_object.__get__(model)
        m.get_object = get_object.__get__(model)
