# ================================================
# MagiAttention context-parallel backend.
# https://sandai-org.github.io/MagiAttention/docs/v1.1.1/user_guide/quickstart.html
# ================================================

import os
from functools import wraps
from typing import List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.distributed.device_mesh import DeviceMesh

from hy_parallelism.distributed.communications import gather_obj
from hy_parallelism.parallel_states import get_parallel_state
from hy_parallelism.utils import auto_broadcast

from .cp_info import CPInfo, is_cp_ops_disabled


def _get_attn_mask_type_enum():
    from magi_attention.common.enum import AttnMaskType

    return AttnMaskType


def attn_mask_type_to_attn_type_map(
    attn_mask_type: Sequence,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Convert ``AttnMaskType`` values to FFA ``attn_type_map`` int32 tensor.

    Each enum is encoded as: FULL=0, CAUSAL=1, INVCAUSAL=2, BICAUSAL=3.
    """
    int_values = [mask_type.to_int_type() for mask_type in attn_mask_type]
    tensor = torch.tensor(int_values, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def attn_type_map_to_attn_mask_type(
    attn_type_map: Union[torch.Tensor, Sequence[int]],
) -> List:
    """Convert FFA ``attn_type_map`` back to ``AttnMaskType`` values."""
    AttnMaskType = _get_attn_mask_type_enum()
    if isinstance(attn_type_map, torch.Tensor):
        int_values = attn_type_map.detach().cpu().tolist()
    else:
        int_values = list(attn_type_map)
    return [AttnMaskType.from_int_type(int(value)) for value in int_values]



def no_sp_runnable(fn):
    @wraps(fn)
    def wrapper(x, *args, **kwargs):
        if get_parallel_state().cp > 1:
            return fn(x, *args, **kwargs)
        else:
            return x

    return wrapper


def _require_magi_key(cp_info: Optional[CPInfo]):
    if cp_info is None or cp_info.magi_key is None:
        raise ValueError(
            "magi backend requires CPInfo with magi_key; "
            "register it via register_cp_info() or pass cp_info to maybe_scatter_seq/maybe_gather_seq"
        )
    return cp_info.magi_key


def _flatten_batch_seq(x: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, int, int, Tuple[int, ...]]:
    b = x.shape[0]
    if seq_dim != 1:
        raise NotImplementedError(f"magi backend only supports seq_dim=1, got {seq_dim}")
    rest_shape = x.shape[2:]
    s = x.shape[1]
    flat = x.reshape(b * s, *rest_shape)
    return flat, b, s, rest_shape


def _unflatten_batch_seq(
    flat: torch.Tensor,
    batch_size: int,
    rest_shape: Tuple[int, ...],
) -> torch.Tensor:
    local_len = flat.shape[0]
    if batch_size > 1:
        assert local_len % batch_size == 0, (
            f"magi local seqlen ({local_len}) must divide evenly by batch size ({batch_size})"
        )
        local_s = local_len // batch_size
        return flat.reshape(batch_size, local_s, *rest_shape)
    return flat.reshape(1, local_len, *rest_shape)


def _magi_dispatch_scatter(x: torch.Tensor, cp_info: CPInfo) -> torch.Tensor:
    from magi_attention.api import dispatch

    magi_key = _require_magi_key(cp_info)
    flat, batch_size, _, rest_shape = _flatten_batch_seq(x)
    local_flat = dispatch(flat, magi_key)
    return _unflatten_batch_seq(local_flat, batch_size, rest_shape)


def _magi_undispatch_gather(x: torch.Tensor, cp_info: CPInfo) -> torch.Tensor:
    from magi_attention.api import undispatch

    magi_key = _require_magi_key(cp_info)
    flat, batch_size, _, rest_shape = _flatten_batch_seq(x)
    global_flat = undispatch(flat, magi_key)
    return _unflatten_batch_seq(global_flat, batch_size, rest_shape)


@no_sp_runnable
def maybe_to_split_head(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    # MagiAttention keeps full head count on each rank; no all-to-all is needed.
    return x


@no_sp_runnable
def maybe_to_split_seq(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    return x


def maybe_scatter_seq(x, return_split_meta=False, *, cp_info=None):
    if get_parallel_state().cp <= 1:
        return (x, None) if return_split_meta else x
    if is_cp_ops_disabled():
        return (x, None) if return_split_meta else x
    if os.environ.get('HY_PARALLELISM_DEBUG', '0') == '1':
        shape = x.shape
        shapes = gather_obj(shape, group=get_parallel_state().cp_group)
        assert all(item == shapes[0] for item in shapes), f'Shape mismatch in CP group. {shapes}'

    _require_magi_key(cp_info)
    out = _magi_dispatch_scatter(x, cp_info)
    meta = cp_info if return_split_meta else None
    return (out, meta) if return_split_meta else out


@no_sp_runnable
def maybe_gather_seq(x, cp_info: Optional[CPInfo] = None):
    if is_cp_ops_disabled():
        return auto_broadcast(x, group_src=0, group=get_parallel_state().cp_group)
    return _magi_undispatch_gather(x, cp_info)


@no_sp_runnable
def maybe_scatter_head(x):
    return x


@no_sp_runnable
def maybe_gather_head(x):
    return x


def wrap_list(t):
    if not isinstance(t, list):
        t = [t]
    return t


all_to_all_sp2hp = maybe_to_split_head
all_to_all_hp2sp = maybe_to_split_seq


def maybe_to_cp_region_num_head(num_head):
    # MagiAttention uses full num_heads on each rank after dispatch.
    return num_head


def maybe_to_normal_region_num_head(cp_num_head):
    return cp_num_head


maybe_to_normal_reigion_num_head = maybe_to_normal_region_num_head


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
):
    raise NotImplementedError(
        "MagiAttention context parallel does not use PyTorch experimental context_parallel."
    )
