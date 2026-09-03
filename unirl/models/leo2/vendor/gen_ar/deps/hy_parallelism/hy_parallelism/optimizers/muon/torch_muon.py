# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import os
import math
from packaging import version
from dataclasses import dataclass
from collections.abc import Callable, Iterator, MutableMapping
from typing import Protocol
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch import Tensor

from torch.optim.optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    Optimizer,
    ParamsT,
)
from hy_parallelism.tools.profiling import profile_class, profile_func, profile_range
from hy_parallelism.parallel_states import get_parallel_state

def _to_scalar(x):
    r"""This function converts a hyperparameter to a 0-dimension (scalar) tensor
    if it is a nonzero-dimensions 1-element tensor. If it is not a tensor, it is
    kept as is.

    Args:
        x (float or Tensor): A hyperparameter of the optimizer.
            If it is Tensor, it is needed to be 1-element.

    Returns:
        float or Tensor:
            a scalar tensor if x is Tensor otherwise Python scalar (float) value.
    """
    if isinstance(x, torch.Tensor) and x.dim() != 0:
        return x.squeeze()
    else:
        return x


__all__ = ["Muon"]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
DEFAULT_ADJUST_LR_FN = 'match_rms_adamw'
NS_SEQUENTIAL_BROADCAST_THRESHOLD_BYTES = 8 * 1024 ** 3


class _MuonSpec(Protocol):
    def split(self, tensor: Tensor) -> list[Tensor]: ...
    def merge(self, tensors: list[Tensor]) -> Tensor: ...


@dataclass(frozen=True)
class _MuonSplitSpec:
    kind: str
    original_shape: tuple[int, ...] | None = None
    split_sizes: tuple[int, ...] | None = None

    def split(self, tensor: Tensor):
        if self.kind == "split_dim0":
            return list(tensor)
        if self.kind == "flatten_conv":
            return [tensor.view(tensor.size(0), -1)]
        if self.kind == "qkv":
            assert self.split_sizes is not None
            return list(torch.split(tensor, self.split_sizes, dim=0))
        raise ValueError(f"Unknown Muon split spec: {self.kind}")

    def merge(self, tensors: list[Tensor]):
        if self.kind == "split_dim0":
            return torch.stack(tensors)
        if self.kind == "flatten_conv":
            assert self.original_shape is not None
            return tensors[0].view(self.original_shape)
        if self.kind == "qkv":
            return torch.cat(tensors, dim=0)
        raise ValueError(f"Unknown Muon split spec: {self.kind}")


@dataclass(frozen=True)
class _MuonFnSpec:
    split_fn: Callable[[Tensor], list[Tensor]]
    merge_fn: Callable[[list[Tensor]], Tensor]

    def split(self, tensor: Tensor) -> list[Tensor]:
        return self.split_fn(tensor)

    def merge(self, tensors: list[Tensor]) -> Tensor:
        return self.merge_fn(tensors)


def make_spec(split_fn, merge_fn):
    return _MuonFnSpec(split_fn, merge_fn)


@profile_func
def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if grad.ndim < 2:
        raise ValueError("Input tensor gradient must have at least 2 dimensions")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        ortho_grad = ortho_grad.mT
    # Ensure spectral norm is at most 1
    ortho_grad = ortho_grad / (ortho_grad.norm(dim=(-2, -1), keepdim=True) + eps)
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.mT
        gram_update = b * gram_matrix + c * gram_matrix @ gram_matrix
        ortho_grad = a * ortho_grad + gram_update @ ortho_grad

    if grad.size(-2) > grad.size(-1):
        ortho_grad = ortho_grad.mT
    return ortho_grad


@profile_func
def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    if len(param_shape) < 2:
        raise ValueError("Parameter shape must have at least 2 dimensions")
    A, B = param_shape[-2], param_shape[-1]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        # pyrefly: ignore [no-matching-overload]
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


@dataclass
class MuonNsWorkItem:
    param: Tensor
    split_grads: list[Tensor]
    merge_fn: Callable[[list[Tensor]], Tensor]
    dtensor_kwargs: dict
    merged_shape: torch.Size
    device: torch.device


@profile_func
def ns_and_merge_split_grads(
    split_grads: list[Tensor],
    merge_fn: Callable[[list[Tensor]], Tensor],
    *,
    lr: float,
    adjust_lr_fn: str | None,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    split_results = []
    for sub_grad in split_grads:
        ns_result = _zeropower_via_newtonschulz(
            sub_grad, ns_coefficients, ns_steps, eps
        )
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, sub_grad.shape)
        split_results.append(ns_result * -adjusted_lr)
    return merge_fn(split_results)


def _balanced_param_order(costs: list[int], ns_dist_size: int) -> list[int]:
    n = len(costs)
    sorted_indices = sorted(range(n), key=lambda i: (-costs[i], i))
    order = [0] * n
    for chunk_start in range(0, n, ns_dist_size):
        batch = sorted_indices[chunk_start : chunk_start + ns_dist_size]
        slot_loads = [0] * len(batch)
        chunk_perm = [-1] * len(batch)
        for idx in sorted(batch, key=lambda i: (-costs[i], i)):
            slot = min(range(len(batch)), key=lambda s: (slot_loads[s], s))
            chunk_perm[slot] = idx
            slot_loads[slot] += costs[idx]
        for slot, idx in enumerate(chunk_perm):
            order[chunk_start + slot] = idx
    return order


def _chunk_comm_bytes(items: list[MuonNsWorkItem], chunk_len: int) -> int:
    # dist_ns_run gathers merged grads in bfloat16.
    return sum(items[i].merged_shape.numel() * 2 for i in range(chunk_len))


def dist_ns_run(
    items: list[MuonNsWorkItem],
    chunk_len: int,
    ns_dist_size: int,
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    lr: float,
    adjust_lr_fn: str | None,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    ns_sequential_broadcast: bool = False,
) -> Iterator[Tensor]:
    device = items[0].device
    comm_dtype = torch.bfloat16
    ns_kwargs = dict(
        lr=lr,
        adjust_lr_fn=adjust_lr_fn,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )

    if local_rank < chunk_len:
        my_contrib = ns_and_merge_split_grads(
            items[local_rank].split_grads,
            items[local_rank].merge_fn,
            **ns_kwargs,
        ).to(comm_dtype)
    else:
        my_contrib = None

    if ns_sequential_broadcast:
        for slot in range(chunk_len):
            if slot == local_rank:
                tensor = my_contrib
            else:
                tensor = torch.empty(
                    items[slot].merged_shape, device=device, dtype=comm_dtype
                )
            dist.broadcast(
                tensor, src=dist.get_global_rank(group, slot), group=group
            )
            yield tensor
            del tensor
            if slot == local_rank:
                my_contrib = None
        return

    gather_buf = []
    for slot in range(ns_dist_size):
        if slot < chunk_len:
            gather_buf.append(
                torch.empty(items[slot].merged_shape, device=device, dtype=comm_dtype)
            )
        else:
            gather_buf.append(torch.zeros(1, device=device, dtype=comm_dtype))

    dist.all_gather(
        gather_buf,
        my_contrib if my_contrib is not None else gather_buf[local_rank],
        group=group,
    )
    for i in range(chunk_len):
        yield gather_buf[i]


@profile_class
class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = DEFAULT_ADJUST_LR_FN,
        sum_decay_momentum: bool = False, # with sum_decay_momentum, the momentum will be (1 - momentum) larger
        ns_dist_size: int | None = None,
        ns_sequential_broadcast: bool = False,
        ns_balance_load: bool = True,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "sum_decay_momentum": sum_decay_momentum,
        }
        super().__init__(params, defaults)

        self._muon_split_specs: list[list[_MuonSpec | None]] = []
        for group in self.param_groups:
            specs: list[_MuonSpec | None] = []
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon only supports parameters with at least 2 dimensions whereas we found a parameter with size: {p.size()}"
                    )
                split_spec = getattr(p, "_muon_split_spec", None)
                if split_spec is None and hasattr(p, "_muon_split_fn") and hasattr(p, "_muon_merge_fn"):
                    split_spec = make_spec(p._muon_split_fn, p._muon_merge_fn)
                if p.ndim > 3 and split_spec is None:
                    raise ValueError(
                        f"Please register `_muon_split_spec` or `_muon_split_fn` and `_muon_merge_fn` "
                        f"for parameters with "
                        f"more than 3 dimensions before building Muon. Found {p.shape}"
                        f"{f' ({param_name})' if (param_name := getattr(p, '_param_name', None)) else ''}"
                    )
                specs.append(split_spec)
            self._muon_split_specs.append(specs)

        if ns_dist_size is None:
            if dist.is_available() and dist.is_initialized() and get_parallel_state().ep == 1 and get_parallel_state().tp == 1:
                w = dist.get_world_size()
                self.ns_dist_size = 8 if w % 8 == 0 else 1
            else:
                self.ns_dist_size = 1
        else:
            self.ns_dist_size = ns_dist_size

        if get_parallel_state().ep > 1 or get_parallel_state().tp > 1:
            assert self.ns_dist_size == 1, "ns_dist_size must be 1 when ep > 1 or tp > 1"

        if self.ns_dist_size < 1:
            raise ValueError(f"ns_dist_size must be >= 1 but is: {self.ns_dist_size}")


        with profile_range('empty_cache'):
            if self.ns_dist_size > 1:
                from hy_parallelism import set_non_torch_allocator_buffer
                torch.cuda.empty_cache()
                set_non_torch_allocator_buffer(n_gb=NS_SEQUENTIAL_BROADCAST_THRESHOLD_BYTES / 1024**3)

        self.ns_sequential_broadcast = ns_sequential_broadcast
        self.ns_balance_load = ns_balance_load
        self.ns_pg_groups: list[dist.ProcessGroup] | None = None

    @profile_func
    def _get_ns_process_group(self) -> dist.ProcessGroup | None:
        if self.ns_dist_size <= 1:
            return None
        if not dist.is_initialized():
            raise RuntimeError("ns_dist_size > 1 requires torch.distributed to be initialized")
        if self.ns_pg_groups is None:
            world_size = dist.get_world_size()
            if world_size % self.ns_dist_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by ns_dist_size ({self.ns_dist_size})"
                )
            self.ns_pg_groups = []
            for i in range(world_size // self.ns_dist_size):
                ranks = list(range(i * self.ns_dist_size, (i + 1) * self.ns_dist_size))
                self.ns_pg_groups.append(dist.new_group(ranks))
        return self.ns_pg_groups[dist.get_rank() // self.ns_dist_size]

    @profile_func
    def _init_group(
        self,
        group: MutableMapping,
        group_index: int,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
        muon_split_specs: list[_MuonSpec | None],
    ) -> bool:
        group_specs = self._muon_split_specs[group_index]
        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            with profile_range('init_momentum_buffer'):
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
            muon_momentum_bufs.append(state["momentum_buffer"])
            muon_split_specs.append(group_specs[i])

        return False  # has_complex


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []
            muon_split_specs: list[_MuonSpec | None] = []

            has_complex = self._init_group(
                group,
                group_index,
                params_with_grad,
                grads,
                muon_momentum_bufs,
                muon_split_specs,
            )

            ns_process_group = (
                self._get_ns_process_group() if self.ns_dist_size > 1 else None
            )
            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                muon_split_specs=muon_split_specs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
                sum_decay_momentum=group["sum_decay_momentum"],
                ns_dist_size=self.ns_dist_size,
                ns_process_group=ns_process_group,
                ns_sequential_broadcast=self.ns_sequential_broadcast,
                ns_balance_load=self.ns_balance_load,
            )
        return loss


Muon.__doc__ = (
    r"""Implements Muon algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input}      : \gamma \text{ (lr)},\ \lambda \text{ (weight decay)},\
               \mu \text{ (momentum)},\ \textit{nesterov}\in\{True,False\},\\
            &\hspace{13mm}(a,b,c)\ \text{ (NS coefficients)},\
               \varepsilon \text{ (epsilon)},\ k \text{ (NS steps)},\
               \theta_0 \text{ (params)},\ f(\theta) \text{ (objective)} \\
            &\textbf{initialize} : B_0 \leftarrow 0 \text{ (momentum buffer)} \\[-1.ex]
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\[0.25ex]
            &\hspace{5mm} g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1}) \\[0.25ex]
            &\hspace{5mm} B_t \leftarrow \mu B_{t-1} + g_t \\[0.25ex]
            &\hspace{5mm} \widetilde{B}_t \leftarrow
                \begin{cases}
                   g_t + \mu B_t, & \text{if nesterov}=True \\
                   B_t,           & \text{if nesterov}=False
                \end{cases} \\[1.0ex]
            &\hspace{5mm} O_t \leftarrow \mathrm{NS}^{(a,b,c)}_{k}\!\big(\widetilde{B}_t;\ \varepsilon\big) \\[0.5ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma\,\lambda\,\theta_{t-1}
               \quad\text{(decoupled weight decay)} \\[0.25ex]

            &\hspace{5mm} \gamma \leftarrow \mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big) \\[0.25ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_t - \gamma\, O_t \\
            &\rule{110mm}{0.4pt} \\[-1.ex]
            &\mathbf{return}\ \theta_t \\[-1.ex]
            &\rule{110mm}{0.4pt}s
       \end{aligned}

    Here, :math:`\mathrm{NS}^{(a,b,c)}_{k}(\cdot;\varepsilon)` denotes :math:`k` iterations of the
    Newton–Schulz orthogonalization operator parameterized by coefficients :math:`(a,b,c)`
    with numerical stabilization :math:`\varepsilon`.

    The purpose for :math:`\mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big)`
    is to make the orthogonalized update have a consistent :math:`RMS` across rectangular matrices.

    Keller's original implementation scales the update by :math:`\sqrt{\max\!\left(1, \frac{A}{B}\right)}`,
    where :math:`A` and :math:`B` are dimension of the matrix being optimized.

    Moonshot's implementation also focuses on matching :math:`RMS` of AdamW. The adjustment is computed as:
    :math:`\gamma \leftarrow {0.2}\gamma\,\sqrt{\max\!\left({A}, {B}\right)}`
    The method is adopted from `Muon is Scalable for LLM Training`_. Research
    results show that with this adjustment Muon can directly reuse the learning rate
    and weight decay tuned for AdamW.

    We provide two options for the learning rate adjustment: "original", which follows Keller's
    implementation, and "match_rms_adamw", which refers to Moonshot's implementation. This gives users the
    flexibility to choose between the two. If `adjust_lr_fn` is not specified, the default is "original".

    For further details regarding the algorithm we refer to `Muon: An optimizer for hidden layers in neural networks`_
    and `Muon is Scalable for LLM Training`_.
    """
    + rf"""
    Args:
        {_params_doc}. Note that Muon is an optimizer for parameters with at least 2 dimensions of neural network hidden layers. Other
            parameters, such as bias, and embedding, should be optimized by a standard method such as AdamW.
        lr (float, Tensor, optional): learning rate (default: 1e-3).
        weight_decay (float, optional): weight decay (L2 penalty). (default: 0.1)
        momentum (float, optional): momentum factor (default: 0.95)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero
        ns_coefficients (tuple of three floats, optional): coefficients \(a,b,c\) for the
            Newton–Schulz orthogonalization polynomial (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C}))
        eps (float, optional): term added to the denominator for numerical stability. (default: {EPS})
        ns_steps (int, optional): number of Newton–Schulz iteration steps. (default: {DEFAULT_NS_STEPS})
        adjust_lr_fn (str, optional): function to adjust learning rate. One of "original" and "match_rms_adamw".
            If not specified, we will default to use "original". (default: None)

    .. _Muon\: An optimizer for hidden layers in neural networks:
        https://kellerjordan.github.io/posts/muon/
    .. _Muon is Scalable for LLM Training:
        https://arxiv.org/pdf/2502.16982

    """
)

def full_tensor(dtensor: DTensor, grad_placements = None, async_op: bool = False) -> Tensor:
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor._api import _ToTorchTensor
    if not async_op:
        return dtensor.full_tensor()
    redist_res = dtensor.redistribute(
        placements=[Replicate()] * dtensor.device_mesh.ndim, async_op=async_op
    )
    return _ToTorchTensor.apply(redist_res, grad_placements)


@profile_func
def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    muon_split_specs: list[_MuonSpec | None],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    sum_decay_momentum: bool = False,
    ns_dist_size: int = 1,
    ns_process_group: dist.ProcessGroup | None = None,
    ns_sequential_broadcast: bool = False,
    ns_balance_load: bool = True,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    ns_kwargs = dict(
        lr=lr,
        adjust_lr_fn=adjust_lr_fn,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )

    def prepare_work_item(i: int, *, keep_split_grads: bool = True, async_op: bool = False, stream_to_record=None) -> MuonNsWorkItem:
        param = params[i]
        grad = grads[i]
        if grad.ndim < 2:
            raise ValueError("Param gradient must have at least 2 dimensions")

        buf = muon_momentum_bufs[i]
        # https://github.com/KellerJordan/Muon/issues/52
        if sum_decay_momentum:
            buf.mul_(momentum).add_(grad) # buf = momentum * buf + grad
            update = grad.add(buf, alpha=momentum) if nesterov else buf
        else:
            buf.lerp_(grad, 1 - momentum) # buf = momentum * buf + (1 - momentum) * grad
            update = grad.lerp(buf, momentum) if nesterov else buf

        is_dtensor = isinstance(update, DTensor)
        assert is_dtensor, 'Non-DTensor case is not implemented yet'
        dtensor_kwargs = dict(
            placements=update.placements,
            device_mesh=update.device_mesh,
        )
        if version.parse(torch.__version__) >= version.parse('2.7.0'):
            dtensor_kwargs['src_data_rank'] = None

        split_spec = muon_split_specs[i]
        if split_spec is not None:
            full_update = full_tensor(update, async_op=async_op)
            merge_fn = split_spec.merge
            if keep_split_grads:
                split_grads = split_spec.split(full_update)
                for split_grad in split_grads:
                    assert split_grad.ndim >= 2, f"Split gradients must have at least 2 dimensions, but got {split_grad.ndim}. Please check the _muon_split_spec implementation."
                    if stream_to_record is not None:
                        split_grad.record_stream(stream_to_record)
            else:
                split_grads = []
        else:
            full_update = full_tensor(update, async_op=async_op)
            if stream_to_record is not None:
                full_update.record_stream(stream_to_record)
            split_grads = [full_update] if keep_split_grads else []
            merge_fn = lambda x: x[0]

        return MuonNsWorkItem(
            param=param,
            split_grads=split_grads,
            merge_fn=merge_fn,
            dtensor_kwargs=dtensor_kwargs,
            merged_shape=full_update.shape,
            device=full_update.device,
        )

    def apply_work_item(item: MuonNsWorkItem, merged_grad: Tensor) -> None:
        grad = merged_grad
        if item.dtensor_kwargs:
            from torch.distributed.tensor import distribute_tensor
            grad = distribute_tensor(grad, **item.dtensor_kwargs)

        item.param.mul_(1 - lr * weight_decay)
        try:
            item.param.add_(grad)
        except:
            print("param.shape", item.param.shape, "param.placements", item.param.placements)
            print("param._local_tensor.shape", item.param._local_tensor.shape)
            print("grad.shape", grad.shape, "type", type(grad))
            if isinstance(grad, DTensor):
                print("grad.placements", grad.placements)
            raise

    if ns_dist_size <= 1:
        for i in range(len(params)):
            item = prepare_work_item(i)
            merged_grad = ns_and_merge_split_grads(
                item.split_grads, item.merge_fn, **ns_kwargs
            )
            apply_work_item(item, merged_grad)
        return

    assert ns_process_group is not None
    local_rank = dist.get_rank(ns_process_group)
    if ns_balance_load:
        costs = [params[i].shape.numel() for i in range(len(params))]
        param_order = _balanced_param_order(costs, ns_dist_size)
    else:
        param_order = list(range(len(params)))

    prefetch_items = None
    for base in range(0, len(params), ns_dist_size):
        chunk_indices = param_order[base : base + ns_dist_size]
        chunk_len = len(chunk_indices)
        if prefetch_items is not None:
            prefetch_items = prefetch_items.wait()
            items = prefetch_items
            prefetch_items = None
        else:
            items = [
                prepare_work_item(i, keep_split_grads=(slot == local_rank))
                for slot, i in enumerate(chunk_indices)
            ]
        use_sequential_broadcast = ns_sequential_broadcast or (
            _chunk_comm_bytes(items, chunk_len)
            > NS_SEQUENTIAL_BROADCAST_THRESHOLD_BYTES
        )
        # prefetch
        if base + ns_dist_size < len(params):
            from hy_parallelism.distributed.communications.utils import run_on_async_stream
            from functools import partial
            base_next = base + ns_dist_size
            default_stream = torch.cuda.current_stream()
            def prefetch_fn(param_order, base_next, stream_to_record):
                return [
                    prepare_work_item(i, keep_split_grads=(slot == local_rank), async_op=False, stream_to_record=stream_to_record)
                    for slot, i in enumerate(param_order[base_next : base_next + ns_dist_size])
                ]
            prefetch_items = run_on_async_stream(partial(prefetch_fn, param_order, base_next, default_stream), device=torch.device('cuda'))
        for item, merged_grad in zip(
            items,
            dist_ns_run(
                items,
                chunk_len,
                ns_dist_size,
                local_rank,
                ns_process_group,
                ns_sequential_broadcast=use_sequential_broadcast,
                **ns_kwargs,
            ),
        ):
            apply_work_item(item, merged_grad)
        # torch.cuda.synchronize()



@profile_func
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = None,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    muon_split_specs: list[_MuonSpec | None],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    sum_decay_momentum: bool = False,
    ns_dist_size: int = 1,
    ns_process_group: dist.ProcessGroup | None = None,
    ns_sequential_broadcast: bool = False,
    ns_balance_load: bool = True,
) -> None:
    r"""Functional API that performs Muon algorithm computation.

    See :class:`~torch.optim.Muon` for details.
    """
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    func = _single_tensor_muon

    func(
        params,
        grads,
        muon_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        muon_split_specs=muon_split_specs,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
        sum_decay_momentum=sum_decay_momentum,
        ns_dist_size=ns_dist_size,
        ns_process_group=ns_process_group,
        ns_sequential_broadcast=ns_sequential_broadcast,
        ns_balance_load=ns_balance_load,
    )

def default_pre_optimizer_hook(model):
    """
    为无法直接做 Newton-Schulz 的参数注册 split/merge spec。

    3D 参数（如 MoE expert weight ``[E, H, H]``）已由 batched NS 原生支持，
    无需 split。仅 4D（conv）及 qkv 等需要特殊处理的参数才注册 spec。

    这个 object 需要实现 ``split`` 和 ``merge``：``split`` 把一个 tensor
    转成一系列 2D tensor 的 list，``merge`` 把 Muon update 的 list 恢复成
    一个 tensor。
    """
    # Should be call after fsdp
    from hy_parallelism.distributed.fsdp_util import get_fsdp_named_parameters
    for param_name, param in get_fsdp_named_parameters(model):
        if param.ndim == 4:
            param._muon_split_spec = _MuonSplitSpec(
                "flatten_conv",
                original_shape=tuple(param.shape),
            )
        elif param_name.endswith("qkv_proj.weight"):
            # TODO: HANDLE QKV
            # Muon works better for optimizing transformers if it is applied to their Q, K, V parameters separately,
            # rather than together as would be the default for transformer implementations that parametrize QKV as
            # a single linear layer whose outputs are split.
            param._muon_split_spec = _MuonSplitSpec(
                "qkv",
                split_sizes=(
                    model.config.num_attention_heads,
                    model.config.num_kv_heads,
                    model.config.num_kv_heads,
                ),
            )


def default_pre_optimizer_hook_old(model):
    # Should be call after fsdp
    from hy_parallelism.distributed.fsdp_util import get_fsdp_named_parameters
    for param_name, param in get_fsdp_named_parameters(model):
        if param.ndim == 3:
            param._muon_split_fn = lambda x: list(x)
            param._muon_merge_fn = lambda tensors: torch.stack(tensors)
        elif param.ndim == 4:
            from functools import partial
            def merge_fn(x, shape):
                return x[0].view(shape)
            original_shape = param.shape
            partial_merge_fn = partial(merge_fn, shape=original_shape)
            param._muon_split_fn = lambda x: [x.view(x.size(0), -1)]
            param._muon_merge_fn = partial_merge_fn
        elif param_name.endswith("qkv_proj.weight"):
            # TODO: HANDLE QKV
            # Muon works better for optimizing transformers if it is applied to their Q, K, V parameters separately,
            # rather than together as would be the default for transformer implementations that parametrize QKV as
            # a single linear layer whose outputs are split.
            param._muon_split_fn = lambda x: torch.split(x, [model.config.num_attention_heads, model.config.num_kv_heads, model.config.num_kv_heads], dim=0)

def default_muon_adam_optimizer_factory(name, param, default_muon_kwargs, default_adam_kwargs):
    from torch.optim.adamw import AdamW
    # 4D param (may be conv) will be handled by AdamW for now
    # TODO: add support for 4D params
    if param.ndim >= 2:
        selected_optimizer_cls = Muon
        selected_optimizer_kwargs = default_muon_kwargs
    else:
        selected_optimizer_cls = AdamW
        selected_optimizer_kwargs = default_adam_kwargs
    return selected_optimizer_cls, selected_optimizer_kwargs
