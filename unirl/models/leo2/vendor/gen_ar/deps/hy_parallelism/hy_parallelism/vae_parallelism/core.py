import math
import os
from functools import cache

import torch
from torch import distributed as dist
from hy_parallelism.utils import gather_obj


COMMUNICATION_DTYPE = torch.float16

CODE_EXAMPLE = """

"""

def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD

def get_rank():
    try:
        return int(os.environ['RANK'])
    except:
        return mpi_comm().Get_rank()

def get_world_size():
    try:
        return int(os.environ['WORLD_SIZE'])
    except:
        return mpi_comm().Get_size()


class TorchIGather:

    def __init__(self, backend='nccl', igather_group=None):

        self.handles = []
        self.buffers = []


        if igather_group:
            ranks = dist.get_process_group_ranks(igather_group)
            self.world_size = igather_group.size()
            self.local_rank = igather_group.rank()
        else:
            self.world_size = dist.get_world_size()
            self.local_rank = dist.get_rank()
            ranks = list(range(self.world_size))
        assert len(ranks) == self.world_size
        self.groups_ids = []
        self.group = {} # local rank 到 rank 的映射

        for i in range(self.world_size):
            self.groups_ids.append(tuple(range(i + 1)))

        ranks_all_gather = [None] * dist.get_world_size()
        dist.all_gather_object(ranks_all_gather, ranks)
        ranks_all_gather = [tuple(k) for k in ranks_all_gather]
        # make unique
        ranks_all_gather = list(set(ranks_all_gather))

        self.debug_group = {}
        for group in self.groups_ids:
            for i in range(len(ranks_all_gather)):
                new_group_ranks = [ranks_all_gather[i][k] for k in group]
                new_group = dist.new_group(new_group_ranks)
                if dist.get_rank() in new_group_ranks:
                    self.group[group[-1]] = new_group


    def gather(self, tensor, n_rank=None, tensor_shapes=None):
        if n_rank is not None:
            group = self.group[n_rank - 1]
        else:
            group = None
        rank = self.local_rank
        tensor = tensor.to(COMMUNICATION_DTYPE)

        if tensor_shapes is None:
            tensor_shapes = gather_obj(tensor.shape, group=group) # TODO: 会让异步gather变成同步gather, 但考虑到本身这个的增益可能只有几秒，对于几十秒的任务提升不明显
        else:
            assert len(tensor_shapes) == dist.get_world_size(group), f'tensor_shapes length {len(tensor_shapes)} does not match world size {dist.get_world_size(group)}'

            # Skipping the check can further boost the performance.
            # check = gather_obj(tensor.shape, group=group)
            # assert len(check) == len(tensor_shapes), f'check length {len(check)} does not match tensor_shapes length {len(tensor_shapes)}'
            # for i in range(len(check)):
            #     assert list(check[i]) == list(tensor_shapes[i]), f'check[{i}] {check[i]} does not match tensor_shapes[{i}] {tensor_shapes[i]}'

        consistent_shapes = all(x == tensor_shapes[0] for x in tensor_shapes)
        
        if not consistent_shapes: # dist.gather does not support different shapes among ranks, applying neccesary padding
            max_shape = [max(shape[i] for shape in tensor_shapes) for i in range(len(tensor_shapes[0]))]

            # pad tensor to max shape
            new_tensor = torch.empty(max_shape, dtype=tensor.dtype, device=tensor.device)
            # Create slicing indices: [slice(0, tensor.shape[0]), slice(0, tensor.shape[1]), ..., slice(0, tensor.shape[n])]
            slices = tuple(slice(0, dim) for dim in tensor.shape)
            new_tensor[slices] = tensor
            tensor = new_tensor
        else:
            max_shape = tensor_shapes[0]

        if rank == 0:
            buffer = [torch.empty(max_shape, dtype=tensor.dtype, device=tensor.device) for i in range(n_rank)]
        else:
            buffer = None

        self.buffers.append(buffer)
        handle = torch.distributed.gather(tensor, buffer, async_op=True, group=group, dst=dist.get_process_group_ranks(group)[0])
        self.handles.append((handle, buffer, tensor_shapes))

    def wait(self):
        for handle, buffer, shapes in self.handles:
            handle.wait()
            for i, tensor in enumerate(buffer):
                slices = tuple(slice(0, dim) for dim in shapes[i])
                buffer[i] = tensor[slices]

    def clear(self):
        self.buffers = []
        self.handles = []


# _igather_obj_dict = {}

@cache
def _instantiate_igather_obj(backend='nccl', igather_group=None):
    return TorchIGather(backend, igather_group)

def get_igather_obj(backend='nccl', igather_group=None):
    ret = _instantiate_igather_obj(backend, igather_group)
    ret.clear()
    return ret
    # key = backend, id(igather_group)
    # if (backend, id(igather_group)) in _igather_obj_dict:
    #     return _igather_obj_dict[key]
    # else:
    #     ret = TorchIGather(backend, igather_group)
    #     _igather_obj_dict[key] = ret
    #     return ret

def get_ratio(total: int, world_size: int):
    return math.ceil(total / world_size)

class VAEParallelismTask:
    def __init__(self, parallel_dims):
        self.backend = 'nccl'
        self.gather_to_rank0 = True

        if parallel_dims is None:
            from hy_parallelism.parallel_states import get_parallel_state
            parallel_dims = get_parallel_state()

        self.parallel_dims = parallel_dims


        # self.gather_group = dist.new_group(range(self.vae_parallel_world_size))
        assert self.parallel_dims.sp > 1, 'VAE Parallel requires sp > 1.'
        self.gather_group = self.parallel_dims.sp_mesh.get_group()
        self.gather_local_rank = self.parallel_dims.sp_mesh.get_local_rank()
        self.gather_world_size = self.parallel_dims.sp_mesh.size()
        self.vae_parallel_world_size = self.gather_world_size

class TileAssigner(VAEParallelismTask):

    def assign_tiles(self, tiles: torch.Tensor):
        total = len(tiles)
        ratio = get_ratio(total, self.vae_parallel_world_size)

        def get_tiles(rank, world_size, ratio):
            return tiles[rank * ratio: None if rank == world_size - 1 else (rank + 1) * ratio]


        tiles_curr_rank = get_tiles(self.gather_local_rank, self.gather_world_size, ratio)

        # rank 0 分配 tile0, tile1, ..., tile(ratio-1)
        # rank 1 分配 tile(ratio), tile(ratio+1), ..., tile(2*ratio-1)
        # ...
        # rank (world_size-1) 分配 tile((world_size-1)*ratio), tile((world_size-1)*ratio+1), ..., tile(total-1)

        # tile_shapes_across_ranks_given_tile_idx[0] = [tile0.shape, tile_ratio.shape, ...]
        # tile_shapes_across_ranks_given_tile_idx[1] = [tile1.shape, tile_ratio+1.shape, ...]

        tile_shapes_across_ranks_given_tile_idx = []
        for tile_idx in range(ratio):
            lst = []
            for rank in range(self.gather_world_size):
                curr_tiles = get_tiles(rank, self.gather_world_size, ratio)
                if tile_idx < len(curr_tiles):
                    lst.append(curr_tiles[tile_idx].shape)
            tile_shapes_across_ranks_given_tile_idx.append(lst)

        return tiles_curr_rank, tile_shapes_across_ranks_given_tile_idx


class TileGatherer(VAEParallelismTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results = []
        
        self.igather = get_igather_obj(backend=self.backend, igather_group=self.gather_group)
        self.dummy_result = [torch.zeros((1, 1, 1)).cuda()]  # ndim = 0, as a placeholder for empty list

    def handle_tiles(self, tensor: torch.Tensor, i: int, total, tensor_shapes=None): 
        # total: the number of total tasks (before parallelization)
        # tensor_shapes: 这一次 gather, 不同rank的tensor的shape, 如果提供了，就不用重新通信获取

        self.results.append(tensor)

        # is only used for nccl backend with async gather (gather_to_rank0)
        ratio = get_ratio(total, self.vae_parallel_world_size)
        n_task = ([ratio] * (total // ratio) + ([total % ratio] if total % ratio else []))
        n_task = n_task + [0] * (self.vae_parallel_world_size - len(n_task))

        def find(n):
            return next((i for i, task_n in enumerate(n_task) if task_n < n), len(n_task))

        if self.backend in ['nccl', 'gloo'] and self.gather_to_rank0:
            self.igather.gather(tensor, n_rank=find(i + 1), tensor_shapes=tensor_shapes) # TODO:

    def gather_tiles(self):

        if self.backend == 'mpi':
            if self.gather_to_rank0:
                ret = mpi_comm().gather(self.results, root=0)
            else:
                ret = mpi_comm().allgather(self.results)

            ret = sum(ret, [])

        elif self.backend in ['nccl', 'gloo']:
            # [Kevin]:
            # We expect all tiles obtained from the same rank have the same shape.
            # Shapes among ranks can differ due to the imbalance of task assignment.
            if self.gather_to_rank0:
                if self.gather_local_rank == 0:
                    self.igather.wait()
                    gather_results = self.igather.buffers
                self.igather.clear()
                if self.gather_local_rank != 0:
                    return None

                ret = [col[i] for i in range(max([len(k) for k in gather_results])) for col in gather_results if i < len(col)]
            else:
                # TODO:
                if len(self.results) == 0:
                    self.results = [torch.zeros((1, 1, 1)).cuda()]  # ndim = 0, as a placeholder for empty list
                ret = torch.stack(self.results)
                def gather_obj(obj):
                    lst = [None for _ in range(self.gather_world_size)]
                    dist.all_gather_object(lst, obj, group=self.gather_group)
                    return lst

                def torch_dist_allgather(tensor):
                    # shapes among ranks can differ
                    tensor_shape = gather_obj(tensor.shape) # gloo backend 有点不一样
                    recv_buffer = [torch.zeros(tensor_shape[i]).cuda() for i in range(self.gather_world_size)]
                    torch.distributed.all_gather(recv_buffer, tensor.float(), group=self.gather_group)
                    return recv_buffer
                gather_results = torch_dist_allgather(ret)
                gather_results = [k for k in gather_results if k.numel() > 1] # filter out place holder
                ret = list(torch.concatenate([k.float() for k in gather_results]))
        else:
            raise ValueError(f'Unsupported backend {self.backend}.')
        return ret


def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    if blend_extent == 0:
        return b

    a_region = a[..., -blend_extent:, :]
    b_region = b[..., :blend_extent, :]

    weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
    weights = weights.view(1, 1, 1, blend_extent, 1)

    blended = a_region * (1 - weights) + b_region * weights

    b[..., :blend_extent, :] = blended
    return b

def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    if blend_extent == 0:
        return b

    a_region = a[..., -blend_extent:]
    b_region = b[..., :blend_extent]

    weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
    weights = weights.view(1, 1, 1, 1, blend_extent)

    blended = a_region * (1 - weights) + b_region * weights

    b[..., :blend_extent] = blended
    return b


def blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    if blend_extent == 0:
        return b

    a_region = a[..., -blend_extent:, :, :]
    b_region = b[..., :blend_extent, :, :]

    weights = torch.arange(blend_extent, device=a.device, dtype=a.dtype) / blend_extent
    weights = weights.view(1, 1, blend_extent, 1, 1)

    blended = a_region * (1 - weights) + b_region * weights

    b[..., :blend_extent, :, :] = blended
    return b

# def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
#     blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
#     for y in range(blend_extent):
#         b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
#     return b
#
# def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
#     blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
#     for x in range(blend_extent):
#         b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
#     return b

# def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
#     blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
#     for x in range(blend_extent):
#         b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
#     return b