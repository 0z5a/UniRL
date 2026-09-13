"""Native Leo2 Muon/AdamW optimizer composition."""

from __future__ import annotations

import fnmatch
from typing import Iterable

import torch


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _native_optimizer_spec(name: str, parameter: torch.nn.Parameter, config, muon_cls):
    """Return the native optimizer class and kwargs for one Leo2 parameter."""
    special_adamw = tuple(getattr(config, "special_adamw_params", ()) or ())
    if not special_adamw:
        special_adamw = (
            "*embed_tokens*",
            "*lm_head*",
            "*final_layer*",
            "*time_embed*",
            "*timestep_emb*",
            "*patch_embed*",
            "*embedding*",
        )
    weight_decay = float(config.weight_decay)
    if parameter.ndim >= 2 and not _matches(name, special_adamw):
        return muon_cls, {
            "lr": float(config.learning_rate),
            "weight_decay": weight_decay,
            "momentum": float(config.momentum),
        }
    special_weight_decay = tuple(getattr(config, "special_weight_decay_params", ()) or ())
    no_decay = not _matches(name, special_weight_decay) and (
        name.endswith(".bias")
        or parameter.ndim == 1
        or "embedding" in name
        or "embed_tokens" in name
    )
    return torch.optim.AdamW, {
        "lr": float(config.learning_rate),
        "betas": (float(config.adam_beta1), float(config.adam_beta2)),
        "eps": float(config.adam_epsilon),
        "weight_decay": 0.0 if no_decay else weight_decay,
    }


def build_native_muon_optimizer(*, model: torch.nn.Module, config):
    """Build the native multi-optimizer container over FSDP parameters."""
    from hy_parallelism.checkpoint.stateful import OptimizersContainer
    from hy_parallelism.optimizers.muon.torch_muon import Muon
    from hymm.utils.optimzer_utils import pre_optimizer_hook

    optimizer_kwargs = {
        "lr": float(config.learning_rate),
        "betas": (float(config.adam_beta1), float(config.adam_beta2)),
        "weight_decay": float(config.weight_decay),
        "momentum": float(config.momentum),
        "adamw_eps": float(config.adam_epsilon),
    }

    class NativeOptimizersContainer(OptimizersContainer):
        def materialize_missing_grads(self) -> None:
            """Match native ParallelEngine's first-step zero-gradient fill."""
            if getattr(self, "_missing_grads_initialized", False):
                return
            for parameter in self.all_params:
                if parameter.requires_grad and parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            self._missing_grads_initialized = True

        def load_state_dict(self, state_dict) -> None:
            super().load_state_dict(state_dict)
            self._missing_grads_initialized = True

    optimizer = NativeOptimizersContainer(
        [model],
        optimizer_cls=None,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_factory_for_special_param=lambda name, parameter: _native_optimizer_spec(
            name, parameter, config, Muon
        ),
        pre_optimizer_hook=pre_optimizer_hook,
    )
    optimizer._unirl_native_optimizer_container = True
    return optimizer


def build_native_lr_scheduler(*, config, optimizer):
    """Build the native scheduler container over Muon and AdamW children."""
    from hy_parallelism.checkpoint.stateful import LRSchedulersContainer

    scheduler_type = str(config.type)
    if scheduler_type != "constant":
        raise ValueError(f"Leo2 native Muon currently requires a constant scheduler, got {scheduler_type!r}")
    return LRSchedulersContainer(
        optimizer,
        lambda child: torch.optim.lr_scheduler.LambdaLR(child, lambda _: 1.0),
    )


__all__ = ["build_native_lr_scheduler", "build_native_muon_optimizer"]
