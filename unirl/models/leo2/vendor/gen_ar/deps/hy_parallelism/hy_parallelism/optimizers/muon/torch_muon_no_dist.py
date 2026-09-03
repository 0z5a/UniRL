# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from packaging import version
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import Callable, Protocol
import torch
from torch.distributed.tensor import DTensor
from torch import Tensor

from torch.optim.optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    Optimizer,
    ParamsT,
)
from hy_parallelism.tools.profiling import profile_class, profile_func

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
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
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
        if is_dtensor:
            dtensor_kwargs = dict(
                placements=update.placements,
                device_mesh=update.device_mesh
            )
            if version.parse(torch.__version__) >= version.parse('2.7.0'):
                dtensor_kwargs['src_data_rank'] = None
        split_spec = muon_split_specs[i]
        if split_spec is not None:
            if is_dtensor:
                split_grads = split_spec.split(update.full_tensor())
            else:
                split_grads = split_spec.split(update)
            merge_fn = split_spec.merge

            for split_grad in split_grads:
                assert split_grad.ndim >= 2, f"Split gradients must have at least 2 dimensions, but got {split_grad.ndim}. Please check the _muon_split_spec implementation."
        else:
            if is_dtensor:
                split_grads = [update.full_tensor()]
            else:
                split_grads = [update]
            merge_fn = lambda x: x[0]

        # 加了 can_batch, 即便外面的 pre_optimizer_hook 对 param 做了 split (如 MOE), 这里也会把它们 batch 起来
        # can_batch = (
        #     len(split_grads) > 1
        #     and all(g.shape == split_grads[0].shape for g in split_grads)
        # )
        if False: # can_batch
            # 这个功能可能出错，例如对于 chunk + cat 的 split 和 merge.  grad 会是组 batch 的，和原本的 param 的形状不同
            # 即便 view 也不行，因为不能假设给定的 merge_fn 不会做内存不连续的操作
            batched_grad = torch.stack(split_grads)
            ns_result = _zeropower_via_newtonschulz(
                batched_grad, ns_coefficients, ns_steps, eps
            )
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, split_grads[0].shape)
            grad = ns_result * -adjusted_lr
        else:
            split_results = []
            for sub_grad in split_grads:
                ns_result = _zeropower_via_newtonschulz(
                    sub_grad, ns_coefficients, ns_steps, eps
                )
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, sub_grad.shape)
                split_results.append(ns_result * -adjusted_lr)
            grad = merge_fn(split_results)
        if is_dtensor:
            from torch.distributed.tensor import distribute_tensor
            grad = distribute_tensor(grad, **dtensor_kwargs)

        param.mul_(1 - lr * weight_decay)
        try:
            param.add_(grad)
        except:
            print("param.shape", param.shape, "param.placements", param.placements)
            print("param._local_tensor.shape", param._local_tensor.shape)
            print("update.shape", update.shape, "update.placements", update.placements)
            print("grad.shape", grad.shape, "type", type(grad))
            if isinstance(grad, DTensor):
                print("grad.placements", grad.placements)
            raise



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
