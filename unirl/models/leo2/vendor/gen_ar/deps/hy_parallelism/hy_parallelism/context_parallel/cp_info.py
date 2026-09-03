import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Tuple

_DISABLE_CP_OPS = False
_CP_BACKEND = os.environ.get("HY_PARALLELISM_CP_BACKEND", "ulysses")
_SUPPORTED_CP_BACKENDS = ("ulysses", "magi")


def set_disable_cp_ops(disable: bool):
    global _DISABLE_CP_OPS
    _DISABLE_CP_OPS = disable


def is_cp_ops_disabled() -> bool:
    return _DISABLE_CP_OPS


def set_cp_backend(backend: str):
    global _CP_BACKEND
    if backend not in _SUPPORTED_CP_BACKENDS:
        raise ValueError(
            f"Unsupported CP backend {backend!r}, expected one of {_SUPPORTED_CP_BACKENDS}"
        )
    _CP_BACKEND = backend


def get_cp_backend() -> str:
    return _CP_BACKEND


def _resolve_cp_rank_info(cp_group_or_mesh) -> Tuple[int, int]:
    from torch.distributed.device_mesh import DeviceMesh
    import torch.distributed as dist

    if isinstance(cp_group_or_mesh, DeviceMesh):
        return cp_group_or_mesh.size(), cp_group_or_mesh.get_local_rank()
    return dist.get_world_size(cp_group_or_mesh), dist.get_rank(cp_group_or_mesh)


def _magi_flex_key_supports_head_dims() -> bool:
    """MagiAttention >= 1.1 passes num_heads/head_dim into magi_attn_flex_key."""
    import inspect

    from magi_attention.api import magi_attn_flex_key

    return "num_heads_q" in inspect.signature(magi_attn_flex_key).parameters


def _call_magi_attn_flex_key(
    *,
    q_ranges,
    k_ranges,
    attn_mask_type,
    total_seqlen_q: int,
    total_seqlen_k: int,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    dist_attn_config=None,
):
    from magi_attention.api import magi_attn_flex_key

    key_kwargs = dict(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
    )
    if _magi_flex_key_supports_head_dims():
        key_kwargs.update(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
        )
    if dist_attn_config is not None:
        key_kwargs["dist_attn_config"] = dist_attn_config
    return magi_attn_flex_key(**key_kwargs)

class CPInfo:
    __slots__ = ("seq_lens", "sp_size", "sp_rank", "magi_key")

    def __init__(
        self,
        seq_lens: Tuple[int, ...],
        sp_size: int,
        sp_rank: int,
        magi_key=None,
    ):
        self.seq_lens = seq_lens
        self.sp_size = sp_size
        self.sp_rank = sp_rank
        self.magi_key = magi_key

    @classmethod
    def from_seq_len(cls, seq_len: int, sp_size: int, sp_rank: int) -> "CPInfo":
        seq_lens = tuple(
            get_split_seq_info(seq_len, sp_size, rank)["length"]
            for rank in range(sp_size)
        )
        return cls(seq_lens, sp_size, sp_rank)

    @classmethod
    def from_magi_key(
        cls,
        magi_key,
        seq_len: int,
        sp_size: int,
        sp_rank: int,
    ) -> "CPInfo":
        cp_info = cls.from_seq_len(seq_len, sp_size, sp_rank)
        cp_info.magi_key = magi_key
        return cp_info

    @classmethod
    def from_magi_metas(
        cls,
        q_ranges,
        k_ranges,
        attn_mask_type: List[Any],
        total_seqlen_q: int,
        total_seqlen_k: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        cp_group_or_mesh,
        chunk_size: int = 512,
        dist_attn_config=None,
    ) -> "CPInfo":
        """
        Wraps `magi_attn_flex_key`, and stores the resulting runtime key in
        `CPInfo.magi_key` for `maybe_scatter_seq` / `maybe_gather_seq`.
        """
        from magi_attention.api import compute_pad_size

        sp_size, sp_rank = _resolve_cp_rank_info(cp_group_or_mesh)
        pad_size = compute_pad_size(
            total_seqlen_q=total_seqlen_q,
            cp_size=sp_size,
            chunk_size=chunk_size,
        )

        magi_key = _call_magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=cp_group_or_mesh,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            dist_attn_config=dist_attn_config,
        )
        return cls.from_magi_key(magi_key, total_seqlen_q, sp_size, sp_rank)

    @property
    def local_seq_len(self) -> int:
        return self.seq_lens[self.sp_rank]


_CP_INFO_REGISTRY: Dict[str, Optional[CPInfo]] = {}
_cp_info_key_prefix: ContextVar[str] = ContextVar("_cp_info_key_prefix", default="")


def _resolve_cp_info_key(name: str) -> str:
    return f"{_cp_info_key_prefix.get()}{name}"


@contextmanager
def cp_info_scope(prefix: str) -> Iterator[None]:
    """Scope register_cp_info / get_cp_info keys with *prefix*.

    Nested scopes concatenate prefixes. Outside any scope the prefix is empty.
    """
    assert isinstance(prefix, str), f"prefix must be a string, got {type(prefix)}"
    token = _cp_info_key_prefix.set(f"{_cp_info_key_prefix.get()}{prefix}")
    try:
        yield
    finally:
        _cp_info_key_prefix.reset(token)


def register_cp_info(name: str, cp_info: Optional[CPInfo]):
    assert isinstance(name, str), f"name must be a string, got {type(name)}"
    _CP_INFO_REGISTRY[_resolve_cp_info_key(name)] = cp_info


def get_cp_info(name: str) -> Optional[CPInfo]:
    assert isinstance(name, str), f"name must be a string, got {type(name)}"
    resolved_name = _resolve_cp_info_key(name)
    if resolved_name not in _CP_INFO_REGISTRY:
        from hy_parallelism.bing_utils import log_less
        # log_less(f"CPInfo {resolved_name!r} is not registered", level="warning")
    return _CP_INFO_REGISTRY.get(resolved_name)


def clear_cp_info():
    _CP_INFO_REGISTRY.clear()


def get_split_seq_info(seq_len, sp_size, sp_rank):
    """
    splitting logic is like:
        x[chunk_len * sp_rank : chunk_len * (sp_rank + 1)]

    This function is used to get the size of the chunk for the given sp_rank.
    """
    assert sp_size > 0, f"sp_size must be positive, got {sp_size}"
    assert 0 <= sp_rank < sp_size, f"sp_rank ({sp_rank}) must be in [0, {sp_size})"
    assert seq_len >= 0, f"seq_len must be non-negative, got {seq_len}"

    chunk_len = math.ceil(seq_len / sp_size)
    start = chunk_len * sp_rank
    end = min(chunk_len * (sp_rank + 1), seq_len)

    length = max(0, end - start)
    return {'start': start, 'end': end, 'length': length}
