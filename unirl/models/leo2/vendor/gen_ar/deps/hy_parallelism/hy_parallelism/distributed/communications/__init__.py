import loguru
import torch
import torch.distributed as dist
from hy_parallelism.parallel_states import get_parallel_state

def gather_obj(obj, group=None):
    if group is None:
        ws = dist.get_world_size()
    else:
        ws = dist.get_world_size(group)
    lst = [None for _ in range(ws)]
    dist.all_gather_object(lst, obj, group=group)
    return lst

def tree_map(fn, obj, _visited=None):
    """
    Recursively apply ``fn`` to every tensor leaf in a python object tree.
    """
    if _visited is None:
        _visited = set()

    if isinstance(obj, torch.Tensor):
        return fn(obj)

    if isinstance(obj, (int, str, bool, float, bytes, type(None))):
        return obj

    oid = id(obj)
    if oid in _visited:
        return obj

    if isinstance(obj, dict):
        _visited.add(oid)
        return {k: tree_map(fn, v, _visited) for k, v in obj.items()}

    if isinstance(obj, list):
        _visited.add(oid)
        return [tree_map(fn, x, _visited) for x in obj]

    if isinstance(obj, tuple):
        _visited.add(oid)
        return tuple(tree_map(fn, x, _visited) for x in obj)

    if isinstance(obj, set):
        _visited.add(oid)
        return {tree_map(fn, x, _visited) for x in obj}

    for k in dir(obj):
        if k.startswith('__') and k.endswith('__'):
            continue
        v = getattr(obj, k)
        print(f'setting attr {k}')
        setattr(obj, k, tree_map(fn, v, _visited))

    return obj


def sync_object_for_parallel_training(object, inplace=False, force_object=False, debug_with_check=False, skip_nonbasic_types=False, trace='arg0', parallel_dims=None):
    """
    这个和 auto_broadcast 类似，但区别在于：
    当 debug_with_check=False 时，就是 auto_broadcast, 强制执行，一切以 rank0 为准，并且要求内部所有东西都是可 pickle 的
    当 debug_with_check=True 时，会做检查，如果长度，key，类型不一样，会抛异常，并非以 rank0 为准
        例如 Rank 0: [tensor, int, str], Rank 1: [int, int, int], Rank 2: [str, str] 
        长度不一样, 类型不一样, 该函数会做检查并抛出异常
    """
    if parallel_dims is None:
        parallel_dims = get_parallel_state()
    assert parallel_dims is not None
    sync_groups = []
    group_names = []
    if parallel_dims.tp_enabled:
        group_names.append('tp')
        sync_groups.append(parallel_dims.tp_group)
    if parallel_dims.pp_enabled:
        group_names.append('pp')
        sync_groups.append(parallel_dims.pp_group)
    if parallel_dims.sp_enabled:
        group_names.append('sp')
        sync_groups.append(parallel_dims.sp_group)

    if not debug_with_check:
        for group in sync_groups:
            if force_object:
                object = broadcast_object(object, src=dist.get_process_group_ranks(group)[0], group=group)
            else:
                object = auto_broadcast(object, src=dist.get_process_group_ranks(group)[0], group=group, skip_nonbasic_types=skip_nonbasic_types)
        return object


    obj_type = type(object)
    for i, group in enumerate(sync_groups):
        buffer = gather_obj(obj_type, group=group)
        assert all(x == buffer[0] for x in buffer), f'checking group {group_names[i]}, {trace} has different types ({buffer})'

    if force_object:
        if debug_with_check:
            for group in sync_groups:
                buffer = [None] * group.size()
                dist.all_gather_object(buffer, object, group=group)
                assert all(x == buffer[0] for x in buffer)

        buffer = [object]
        for group in sync_groups:
            src = dist.get_process_group_ranks(group)[0]
            # dist.broadcast_object_list(buffer, group_src=0, group=group)
            dist.broadcast_object_list(buffer, src=src, group=group)
        object = buffer[0]
        return object

    if isinstance(object, torch.Tensor):
        # assert object.device.type == 'cuda'
        old_device = object.device
        object = object.cuda()
        if inplace:
            # To guarantee the tensor shape is consistent within groups, we deprecate the inplace broadcasting implementation.
            raise NotImplementedError(f"Syncing inplace is not implemented yet.")

        if inplace:
            original_object_id = id(object)
            assert object.is_contiguous()
        else:
            object = object.contiguous()

        for i, group in enumerate(sync_groups):
            shapes = gather_obj(object.shape, group=group)
            if not all(x == shapes[0] for x in shapes):
                msg = f'checking group {i} ({group_names[i]} group), {trace} has different shapes ({shapes})'
                loguru.logger.opt(depth=1).error(msg)
                # raise ValueError(msg)
                #  f'checking group {i} ({group_names[i]} group), {trace} has different shapes ({shapes})'

        if debug_with_check:
            for group_id, group in enumerate(sync_groups):
                shapes = gather_obj(object.shape, group=group)
                dtypes = gather_obj(object.dtype, group=group)
                assert all(x == shapes[0] for x in shapes), f'checking group {group_id} ({group_names[group_id]} group), {trace} has different shapes ({shapes})'
                assert all(y == dtypes[0] for y in dtypes), f'checking group {group_id} ({group_names[group_id]} group), {trace} has different dtypes ({dtypes})'

                buffer = [torch.empty_like(object) for _ in range(group.size())]
                dist.all_gather(buffer, object, group=group)
                if not all(torch.allclose(x, buffer[0]) for x in buffer):
                    loguru.logger.opt(depth=1).error(f'checking group {group_id} ({group_names[group_id]} group), {trace} not equal, {buffer} ')
                # assert all(torch.allclose(x, buffer[0]) for x in buffer), f'checking group {group_id} ({group_names[group_id]} group), {trace} not equal'


        for group in sync_groups:
            src = dist.get_process_group_ranks(group)[0]
            # dist.broadcast(object, group_src=0, group=group)
            dist.broadcast(object, src=src, group=group)
        # for group in sync_groups:
        #     src = dist.get_process_group_ranks(group)[0]
        #     object = auto_broadcast(object, src=src, group=group)

        object = object.to(old_device)

        if inplace:
            assert id(object) == original_object_id

    elif isinstance(object, list) or isinstance(object, tuple):
        list_length = len(object)
        for i, group in enumerate(sync_groups):
            lens = gather_obj(list_length, group=group)
            # loguru.logger.info(f'List len: {lens}')
            assert all(x == lens[0] for x in lens), f'checking group {i} ({group_names[i]} group), {trace} has different lengths ({lens})'
        object = list(object)
        for i, obj in enumerate(object):
            object[i] = sync_object_for_parallel_training(obj, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'.list[{i}]', parallel_dims=parallel_dims)
    elif isinstance(object, dict):
        dict_length = len(object)
        for i, group in enumerate(sync_groups):
            lens = gather_obj(dict_length, group=group)
            assert all(x == lens[0] for x in lens), f'checking group {i} ({group_names[i]} group), {trace} has different lengths ({lens})'
        for key, value in object.items():
            object[key] = sync_object_for_parallel_training(value, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'.dict[{key}]', parallel_dims=parallel_dims)
    elif isinstance(object, set):
        set_length = len(object)
        for i, group in enumerate(sync_groups):
            lens = gather_obj(set_length, group=group)
            assert all(x == lens[0] for x in lens), f'checking group {i} ({group_names[i]} group), {trace} has different lengths ({lens})'
        new_set = set()
        for i, item in enumerate(object):
            new_set.add(sync_object_for_parallel_training(item, inplace=inplace, debug_with_check=debug_with_check, trace=trace + f'.set[{i}]', parallel_dims=parallel_dims))
        object = new_set
    elif object is None:
        return None
    else: # int float str and others
        if debug_with_check:
            if isinstance(object, (int, float, str, bool, slice)):
                for group in sync_groups:
                    buffer = [None] * group.size()
                    dist.all_gather_object(buffer, object, group=group)
                    assert all(x == buffer[0] for x in buffer), f'checking group {group_names[i]} ({group_names[i]} group), {trace} not equal, {buffer} '
            else:
                loguru.logger.opt(depth=1).warning(f'Skipping checking for non-basic types {trace} ({type(object)})')

        # 走到这里了，应该是只 for debug 的，就不要强行 broadcast object 了，因为有可能里面包含tensor，会导致device不对
        # 比如 partial, 所以这里先注释了下面部分
        # buffer = [object]
        # for group in sync_groups:
        #     src = dist.get_process_group_ranks(group)[0]
        #     # dist.broadcast_object_list(buffer, group_src=0, group=group)
        #     dist.broadcast_object_list(buffer, src=src, group=group)
        # object = buffer[0]


        # raise NotImplementedError(f"Unsupported type {type(object)}")
    return object


def map_tensor(obj, func):
    if isinstance(obj, (int, str, bool, float)):
        return obj
    elif isinstance(obj, torch.Tensor):
        # return obj.to(torch.cuda.current_device())
        return func(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = map_tensor(v, func=func)
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj, tuple):
            obj = list(obj)
        for i in range(len(obj)):
            obj[i] = map_tensor(obj[i], func=func)
    elif isinstance(obj, set):
        new_set = set()
        for item in obj:
            new_set.add(map_tensor(item, func=func))
        obj = new_set
    elif obj is None:
        obj = None
    else:
        raise ValueError(f"Unsupported type {type(obj)}")
        # assert_no_tensor(obj, f'Syncing {type(obj)} could lead to potential device mismatch. e.g. getting stuck when trying to conduct gathering or reducing')
    return obj

def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise NotImplementedError

def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        device=device,
        # async_op=async_op,
    )
    buffer = [obj] if is_src(src, group_src, group) else [None]
    if group_src is None:
        del kwargs['group_src']

    # loguru.logger.debug(f'broadcast_object: {buffer=}, {kwargs}')
    dist.broadcast_object_list(buffer, **kwargs)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if group_src is None:
        del kwargs['group_src']
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, **kwargs)
    return buffer

def auto_broadcast(
        obj,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
        skip_nonbasic_types = False,
):
    # Maybe not necessary
    if group is None and src is None:
        assert group_src is not None
        src = group_src
        group_src = None

    # WARNING: This does extra checks for tensor dtype and shape, and the broadcast does not happen inplace
    kwargs = dict(
        src=src,
        group_src=group_src,
        group=group,
        async_op=async_op,
    )
    if group_src is None:
        del kwargs['group_src']

    # obj: None or list/dict/tensor or basic type
    obj_type = type(obj)
    obj_type = broadcast_object(obj_type, src=src, group_src=group_src, group=group, )

    if obj_type == torch.Tensor:
        return broadcast_tensor(obj, **kwargs)
    elif obj_type == list or obj_type == tuple:
        if is_src(src, group_src, group):
            length = len(obj)
        else:
            length = None
        length = broadcast_object(length, src=src, group_src=group_src, group=group, )
        if not is_src(src, group_src, group):
            obj = [None] * length
        return [auto_broadcast(x, **kwargs) for x in obj]
    elif obj_type == dict:
        if is_src(src, group_src, group):
            keys = list(obj.keys())
        else:
            keys = None
        def get_value(key):
            if is_src(src, group_src, group):
                return obj[key]
            else:
                return None
        keys = broadcast_object(keys, src=src, group_src=group_src, group=group, )
        return {k: auto_broadcast(get_value(k), **kwargs) for k in keys}
    elif obj_type in (int, float, str, bool, slice):
        return broadcast_object(obj, src=src, group_src=group_src, group=group, )
    elif obj_type == type(None):
        return None
    else:
        if skip_nonbasic_types:
            loguru.logger.warning(f'Skip {obj_type} broadcasting.')
            return obj
        return broadcast_object(obj, src=src, group_src=group_src, group=group, )

class Test(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return torch.split(input_, 2, dim=0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # grad_outputs is a tuple of gradient tensors
        return torch.cat(grad_outputs, dim=0)


class AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Different from context_parallel _AllGather:
    1. this implementation supports None.
    2. this implementation does not concatenate the gathered tensors.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(input_, group=None, async_op=False):
        if input_ is None:
            shape = [0]
            dtype = torch.float32
            input_ = torch.empty(shape, dtype=dtype, device='cuda')
        else:
            shape = input_.shape
            dtype = input_.dtype
        
        shapes = gather_obj(shape, group=group)
        dtypes = gather_obj(dtype, group=group)
        tensor_list = [torch.empty(shape, dtype=dtype, device='cuda') for dtype, shape in zip(dtypes, shapes)]
        dist.all_gather(tensor_list, input_, group=group, async_op=async_op)
        tensor_list = [t.cuda() for t in tensor_list if t.numel() > 0]
        return tuple(tensor_list)

    @staticmethod
    def setup_context(ctx, inputs, output):
        group = inputs[1]
        ctx.group = group

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError('支持 None 会让返回值数量不确定， grad_outputs 的数量也不确定，这里不确定是否有bug，不建议使用 backward')
        rank = dist.get_rank(ctx.group)

        grad = grad_outputs[rank]
        grad = grad.contiguous()
        grad_outputs = [t.contiguous() for t in grad_outputs]

        dist.reduce_scatter(grad, grad_outputs, group=ctx.group, op=dist.ReduceOp.SUM)
        return grad




def all_gather_tensor(
    tensor, group=None, async_op=False
) -> list[torch.Tensor]:
    kwargs = dict(
        group=group,
        async_op=async_op,
    )
    # shape = tensor.shape
    # shapes = gather_obj(shape, group=group)
    # # all gather supports different shapes among ranks
    # tensor_list = [torch.empty(shape, device=tensor.device, dtype=tensor.dtype) for shape in shapes]
    # dist.all_gather(tensor_list, tensor, **kwargs)
    # return tensor_list
    return AllGather.apply(tensor, group)

def auto_all_gather(
    obj, group=None, async_op=False
):
    # obj: None or list/dict/tensor or basic type
    kwargs = dict(
        group=group,
        async_op=async_op,
    )
    obj_type = type(obj)
    obj_types = gather_obj(obj_type, group=group)
    if not all(x == obj_types[0] for x in obj_types):
        return gather_obj(obj, group=group)

    if obj_type == torch.Tensor:
        return all_gather_tensor(obj, **kwargs)
    else:
        return gather_obj(obj, group=group)