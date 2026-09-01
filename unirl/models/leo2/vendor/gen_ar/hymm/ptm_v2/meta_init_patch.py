"""Meta-device streaming construction for LeoLayer backbone (unit modules).

Goal (PROGRESS §9.D): the FSDP iter-1 build-model GPU peak is dominated by
materializing the full *unsharded* 48-layer backbone on every rank before
MegatronFSDP shards it. We cut that peak by building only the memory-dominant
leaf modules of LeoLayer on ``torch.device('meta')`` and letting MegatronFSDP
materialize them per-shard (``init_model_with_meta_device``).

Scope (user-confirmed: "只对 unit module 做"):
  * ColumnParallelLinear / RowParallelLinear  -> q/k/v/o_proj + dense MLP fc1/fc2
  * GroupedMLP                                -> MoE experts (weight1/weight2)
  These three classes are the ~51GB backbone bulk. Everything else
  (ModulateDiT, qk/layer norms, projectors, embeds, excluded heads) stays on
  CUDA -- they are small and not the peak source.

Why this is bitwise-safe for *random* init (the user requires bitwise-identical
loss/grad-norm vs the non-meta path):
  * The local-impl TP leaves hardcode ``device=torch.cuda.current_device()`` and
    run ``_initialize_affine_weight_gpu`` inside ``__init__``. That init forks the
    model-parallel CUDA RNG. If we let it run on CUDA at build time *and* reset
    again at materialization, RNG is consumed twice -> divergence. So we must
    prevent the build-time CUDA init entirely.
  * Empirically verified: ``get_cuda_rng_tracker().fork()`` with a body that
    inits a *meta* tensor advances the RNG by ZERO (a meta init is a no-op),
    whereas a real CUDA init advances it. Therefore, if weight/bias allocate on
    meta during build, the original ``__init__`` init becomes a no-op consuming
    no RNG, and RNG is consumed exactly once -- in ``reset_parameters`` at
    materialization, in ``group.params`` order. (That order matching the
    monolithic build is the residual risk verified end-to-end by the bitwise
    run; see PROGRESS meta-init verification.)

Mechanism:
  * A module-level flag ``_META_BUILD_ACTIVE`` is turned on only while
    model_provider constructs ``LeoLayerBlock`` (see meta_build_active()).
  * ``__init__`` of the three classes is wrapped: when the flag is on, a scoped
    ``torch.empty`` redirect forces the weight/bias allocations onto meta. The
    only ``torch.empty`` calls inside those ``__init__``s are weight/bias, so the
    redirect is safe. We also stash the args needed to replay init.
  * ``reset_parameters`` is injected onto each class, replaying exactly the
    ``__init__`` init (``_initialize_affine_weight_gpu`` with the same
    init_method/partition_dim/stride/is_expert, then bias zero_).
"""

import contextlib
import threading

import torch

from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.moe.experts import GroupedMLP


# ---------------------------------------------------------------------------
# meta-build flag (set only around LeoLayerBlock construction in model_provider)
# ---------------------------------------------------------------------------
_META_BUILD = threading.local()

# Global enable switch, set once from args in model_provider. When False the
# LeoLayerBlock.__init__ wrapper does NOT activate meta build (full back-compat).
_META_INIT_ENABLED = False


def set_meta_init_enabled(enabled: bool):
    """Enable/disable LeoLayer meta-device construction. Driven by the yaml flag
    / args.init_model_with_meta_device, set in model_provider before the model is
    built."""
    global _META_INIT_ENABLED
    _META_INIT_ENABLED = bool(enabled)


def is_meta_init_enabled() -> bool:
    return _META_INIT_ENABLED


def _is_meta_build_active() -> bool:
    return getattr(_META_BUILD, "active", False)


@contextlib.contextmanager
def meta_build_active():
    """Context manager: while active, the patched leaf ``__init__``s allocate
    their weight/bias on meta (and skip CUDA init RNG)."""
    prev = getattr(_META_BUILD, "active", False)
    _META_BUILD.active = True
    try:
        yield
    finally:
        _META_BUILD.active = prev


@contextlib.contextmanager
def _force_meta_empty():
    """Scope-redirect ``torch.empty`` so any CUDA-targeted allocation lands on
    meta instead. Used only for the duration of one leaf ``__init__`` call, whose
    only ``torch.empty`` uses are the weight/bias Parameters."""
    orig_empty = torch.empty

    def _empty(*args, **kwargs):
        dev = kwargs.get("device", None)
        if dev is not None:
            dev_t = torch.device(dev) if not isinstance(dev, torch.device) else dev
            if dev_t.type == "cuda":
                kwargs["device"] = "meta"
        return orig_empty(*args, **kwargs)

    torch.empty = _empty
    try:
        yield
    finally:
        torch.empty = orig_empty


# ---------------------------------------------------------------------------
# reset_parameters injected onto the three leaf classes
# ---------------------------------------------------------------------------
def _column_reset_parameters(self):
    if getattr(self, "weight", None) is not None and self.weight.device.type != "meta":
        a = self._leo_reset_args
        _initialize_affine_weight_gpu(
            self.weight,
            a["init_method"],
            partition_dim=0,
            stride=a["stride"],
            is_expert=a["is_expert"],
        )
    bias = getattr(self, "bias", None)
    if bias is not None:
        set_tensor_model_parallel_attributes(bias, True, 0, self._leo_reset_args["stride"])
        with torch.no_grad():
            bias.zero_()


def _row_reset_parameters(self):
    if getattr(self, "weight", None) is not None and self.weight.device.type != "meta":
        a = self._leo_reset_args
        _initialize_affine_weight_gpu(
            self.weight,
            a["init_method"],
            partition_dim=1,
            stride=a["stride"],
            is_expert=a["is_expert"],
        )
    bias = getattr(self, "bias", None)
    if bias is not None:
        with torch.no_grad():
            bias.zero_()


def _grouped_reset_parameters(self):
    cfg = self.config
    if getattr(self, "weight1", None) is not None and self.weight1.device.type != "meta":
        _initialize_affine_weight_gpu(
            self.weight1, cfg.init_method, partition_dim=1, is_expert=True
        )
    if getattr(self, "weight2", None) is not None and self.weight2.device.type != "meta":
        _initialize_affine_weight_gpu(
            self.weight2, cfg.output_layer_init_method, partition_dim=0, is_expert=True
        )


# ---------------------------------------------------------------------------
# __init__ wrappers (registered via run_patches as *_wrapper -> decorator)
# ---------------------------------------------------------------------------
def column_parallel_linear_init_wrapper(orig_init):
    def __init__(self, *args, **kwargs):
        if _is_meta_build_active():
            with _force_meta_empty():
                orig_init(self, *args, **kwargs)
        else:
            orig_init(self, *args, **kwargs)
        # Stash args needed to replay init at materialization. init_method may be
        # positional or kw; ColumnParallelLinear has it as a kw-only arg.
        self._leo_reset_args = {
            "init_method": kwargs.get("init_method"),
            "stride": kwargs.get("stride", 1),
            "is_expert": kwargs.get("is_expert", False),
        }

    return __init__


def row_parallel_linear_init_wrapper(orig_init):
    def __init__(self, *args, **kwargs):
        if _is_meta_build_active():
            with _force_meta_empty():
                orig_init(self, *args, **kwargs)
        else:
            orig_init(self, *args, **kwargs)
        self._leo_reset_args = {
            "init_method": kwargs.get("init_method"),
            "stride": kwargs.get("stride", 1),
            "is_expert": kwargs.get("is_expert", False),
        }

    return __init__


def grouped_mlp_init_wrapper(orig_init):
    def __init__(self, *args, **kwargs):
        if _is_meta_build_active():
            with _force_meta_empty():
                orig_init(self, *args, **kwargs)
        else:
            orig_init(self, *args, **kwargs)

    return __init__


def leo_layer_block_init_wrapper(orig_init):
    """Activate meta build only for the duration of LeoLayerBlock construction
    (all LeoLayer unit modules). Everything built outside this block -- embeds,
    projectors, FSDP-excluded heads -- stays on CUDA. Gated by the global enable
    switch so the non-meta path is byte-for-byte unchanged when off."""

    def __init__(self, *args, **kwargs):
        if is_meta_init_enabled():
            with meta_build_active():
                orig_init(self, *args, **kwargs)
        else:
            orig_init(self, *args, **kwargs)

    return __init__


def install_meta_init_methods():
    """Attach reset_parameters to the three leaf classes. Idempotent. Called from
    run_patches() so it happens once, before any model construction."""
    ColumnParallelLinear.reset_parameters = _column_reset_parameters
    RowParallelLinear.reset_parameters = _row_reset_parameters
    GroupedMLP.reset_parameters = _grouped_reset_parameters
