# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from typing import Optional

from hy_parallelism.context_parallel.communications import (
    all_gather,
    all_to_all_4D,
    set_enable_sp_padding,
)
from hy_parallelism.context_parallel.cp_info import (
    CPInfo,
    clear_cp_info,
    cp_info_scope,
    get_cp_backend,
    get_cp_info,
    get_split_seq_info,
    is_cp_ops_disabled,
    register_cp_info,
    set_cp_backend,
    set_disable_cp_ops,
)


def _get_backend_module():
    if get_cp_backend() == "magi":
        from . import magi as backend
    else:
        from . import ulysses as backend
    return backend


def maybe_scatter_seq(x, return_split_meta=False, *, cp_info=None):
    return _get_backend_module().maybe_scatter_seq(
        x, return_split_meta=return_split_meta, cp_info=cp_info
    )


def maybe_gather_seq(x, cp_info: Optional[CPInfo] = None):
    return _get_backend_module().maybe_gather_seq(x, cp_info=cp_info)


def maybe_to_split_head(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    return _get_backend_module().maybe_to_split_head(x, cp_info=cp_info, async_op=async_op)


def maybe_to_split_seq(x, cp_info: Optional[CPInfo] = None, async_op: bool = False):
    return _get_backend_module().maybe_to_split_seq(x, cp_info=cp_info, async_op=async_op)


def maybe_to_cp_region_num_head(num_head):
    return _get_backend_module().maybe_to_cp_region_num_head(num_head)


def maybe_to_normal_region_num_head(cp_num_head):
    return _get_backend_module().maybe_to_normal_region_num_head(cp_num_head)


def maybe_scatter_head(x):
    return _get_backend_module().maybe_scatter_head(x)


def maybe_gather_head(x):
    return _get_backend_module().maybe_gather_head(x)


def create_context_parallel_ctx(*args, **kwargs):
    return _get_backend_module().create_context_parallel_ctx(*args, **kwargs)


from .ulysses import (  # noqa: E402
    AllReduceGradientsForSequenceParallel,
    SequentialSharder,
    _sp_scatter,
    all_to_all_hp2sp,
    all_to_all_sp2hp,
    maybe_to_normal_reigion_num_head,
    wrap_list,
)
