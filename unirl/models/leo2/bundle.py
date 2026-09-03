"""Leo2 bundle bootstrap and weight holder."""

from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Any

import torch
from torch import nn

from unirl.models.transformers_compat import install_transformers_flash_attention_compat
from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import Leo2PipelineConfig

_HYMM_BOOTSTRAPPED = False


def ensure_hy_parallel_state() -> bool:
    """Initialize hy_parallelism's singleton process state before Leo2 forwards."""
    import torch.distributed as dist
    from hy_parallelism import parallel_states as hy_ps

    if hy_ps.is_parallel_state_initialized():
        return True
    if not dist.is_initialized():
        return False
    world = dist.get_world_size()
    dp_shard = min(8, world)
    hy_ps.init_parallel_state(
        dp_replicate=world // dp_shard,
        dp_shard=dp_shard,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=world,
    )
    return True


def _bootstrap_hymm(config: Leo2PipelineConfig):
    """Import hymm, parse args from the yaml, set globals. Idempotent."""
    global _HYMM_BOOTSTRAPPED

    repo = os.path.abspath(config.hymm_repo_path)
    config_yaml = os.path.abspath(config.config_yaml)
    required_paths = {
        "hymm runtime": os.path.join(repo, "hymm"),
        "processors runtime": os.path.join(repo, "processors"),
        "hy_parallelism dependency": os.path.join(repo, "deps", "hy_parallelism", "hy_parallelism"),
        "IndexKits dependency": os.path.join(repo, "deps", "IndexKits", "index_kits"),
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(
            "Leo2 hymm runtime is incomplete; set hymm_repo_path to a complete runtime. Missing: " + "; ".join(missing)
        )
    if not os.path.isfile(config_yaml):
        raise FileNotFoundError(f"Leo2 config_yaml does not exist: {config_yaml}")

    if repo not in sys.path:
        sys.path.insert(0, repo)
        for dep in ("deps/hy_parallelism", "deps/IndexKits"):
            p = os.path.join(repo, dep)
            if p not in sys.path:
                sys.path.insert(0, p)
    if not isinstance(config.assets_base, str) or not config.assets_base.strip():
        raise ValueError("Leo2 assets_base must be a non-empty path")
    os.environ["ASSETS_BASE"] = os.path.abspath(os.path.expanduser(config.assets_base))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import argparse

    from hymm.config import add_core_args, validate_args
    from hymm.core import global_vars
    from hymm.core.arguments import parse_argv_from_yaml

    if _HYMM_BOOTSTRAPPED:
        return global_vars.get_args()

    argv = [
        "--config-path",
        config_yaml,
        "--ckpt",
        config.ckpt_path,
        "--task-id",
        "unirl-leo2",
        "--framework",
        "fsdp",
        *config.extra_hymm_args,
    ]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-path", type=str, required=True)
    pre.add_argument("--framework", type=str, default="fsdp")
    known, remaining = pre.parse_known_args(argv)

    config_argv, frozen = parse_argv_from_yaml(known.config_path, allow_frozen=True, argv_overrides=remaining)
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + config_argv + remaining
        parser = argparse.ArgumentParser(description="Leo2 UniRL bundle")
        parser = add_core_args(parser)
        args = parser.parse_args()
    finally:
        sys.argv = saved_argv
    args = validate_args(args, frozen)
    if not args.model_structure.endswith("HF"):
        args.model_structure += "HF"

    # per-process globals (Ray worker == fresh process, but stay idempotent)
    global_vars._GLOBAL_ARGS = None
    global_vars.set_args(args)

    from loguru import logger as _loguru_logger

    global_vars._GLOBAL_LOGGER = None
    global_vars.set_logger(_loguru_logger)

    from hymm.core.parallel_states import ParallelState

    ParallelState(dp_rank=0, dp_size=1)

    # The bundle may be built before torch.distributed is up (then this is a
    # no-op); predict_noise() re-checks right before the first forward.
    ensure_hy_parallel_state()

    _HYMM_BOOTSTRAPPED = True
    return args


def _dcp_load_into(model: nn.Module, weights_dir: str, *, model_dtype: torch.dtype) -> None:
    """Fill the (empty) model from the torch-dcp checkpoint, dtype-exact."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    weights_dir = os.path.abspath(os.path.expanduser(weights_dir))
    metadata_path = os.path.join(weights_dir, ".metadata")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Leo2 native DCP metadata does not exist: {metadata_path}")

    reader = FileSystemReader(weights_dir)
    try:
        saved = reader.read_metadata().state_dict_metadata
    except Exception as exc:
        raise RuntimeError(f"Leo2 native DCP metadata is unreadable: {metadata_path}") from exc

    dest = model.state_dict()
    expected = {f"model.{key}" for key in dest}
    actual = set(saved)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Leo2 native DCP keys do not exactly match the model: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)."
        )

    invalid = []
    load_state = {}
    for key, tensor in dest.items():
        checkpoint_key = f"model.{key}"
        metadata = saved[checkpoint_key]
        properties = getattr(metadata, "properties", None)
        saved_dtype = getattr(properties, "dtype", None)
        saved_size = getattr(metadata, "size", None)
        if not isinstance(tensor, torch.Tensor):
            invalid.append(f"{checkpoint_key}: destination is {type(tensor).__name__}, not Tensor")
        elif tensor.is_meta:
            invalid.append(f"{checkpoint_key}: destination is still on meta")
        elif saved_size is None or tuple(saved_size) != tuple(tensor.shape):
            invalid.append(f"{checkpoint_key}: shape checkpoint={saved_size}, model={tuple(tensor.shape)}")
        elif saved_dtype is None:
            invalid.append(f"{checkpoint_key}: checkpoint carries no dtype")
        elif saved_dtype != tensor.dtype and not (
            saved_dtype == model_dtype
            and tensor.dtype == torch.float32
            and key.startswith("layers.")
            and key.endswith(".mlp.gate.wg.weight")
        ):
            invalid.append(f"{checkpoint_key}: dtype checkpoint={saved_dtype}, model={tensor.dtype}")
        else:
            load_state[key] = (
                tensor
                if saved_dtype == tensor.dtype
                else torch.empty(tuple(tensor.shape), dtype=saved_dtype, device=tensor.device)
            )
    if invalid:
        raise RuntimeError(
            "Leo2 native DCP tensor metadata does not match the model: "
            + "; ".join(invalid[:8])
            + (f"; {len(invalid)} mismatches total" if len(invalid) > 8 else "")
        )

    dcp.load({"model": load_state}, storage_reader=reader)
    result = model.load_state_dict(load_state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Leo2 native DCP load was incomplete: "
            f"missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )
    loaded = model.state_dict()
    incomplete = [
        key
        for key, tensor in loaded.items()
        if tensor.is_meta
        or tuple(tensor.shape) != tuple(saved[f"model.{key}"].size)
        or tensor.dtype != saved[f"model.{key}"].properties.dtype
    ]
    if incomplete:
        raise RuntimeError(f"Leo2 native DCP post-load validation failed for: {incomplete[:8]}")
    print(f"[leo2 bundle] loaded and validated {len(dest)} tensors from {weights_dir}", flush=True)


_BLOCK_CLASSES = ("LeoLayer", "LeoDualLayer", "LeoTripleLayer")


def _move_non_block_to_device(model: nn.Module, device) -> None:
    """Move every root-level child that does not contain a DiT block to device."""
    block_roots = set()
    for name, mod in model.named_modules():
        if type(mod).__name__ in _BLOCK_CLASSES:
            block_roots.add(name.split(".")[0])
    moved = 0
    for name, child in model.named_children():
        if name in block_roots:
            continue
        child.to(device)
        moved += sum(p.numel() for p in child.parameters())
    for pname, p in list(model._parameters.items()):
        if p is not None:
            model._parameters[pname] = nn.Parameter(p.to(device), requires_grad=p.requires_grad)
    for bname, b in list(model._buffers.items()):
        if b is not None:
            model._buffers[bname] = b.to(device)
    print(
        f"[leo2 bundle] moved non-block root modules to {device}: {moved / 1e9:.3f}B params; "
        f"block roots kept for FSDP: {sorted(block_roots)}",
        flush=True,
    )


def _patch_router_dtype(model: nn.Module) -> None:
    import torch.nn.functional as F

    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".gate.wg") and isinstance(mod, nn.Linear):

            def _fwd(x, _m=mod):
                w = _m.weight
                b = _m.bias
                return F.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))

            mod.forward = _fwd
            n += 1
    print(f"[leo2 bundle] router dtype-follow patch applied to {n} gate.wg modules", flush=True)


def _make_inference_cache_config(config: Leo2PipelineConfig) -> Any | None:
    """Validate Leo2 inference-cache options and build the Diffusers config."""
    if not isinstance(config.inference_cache_method, str):
        raise TypeError("Leo2 inference_cache_method must be a string.")
    method = config.inference_cache_method.strip().lower()
    if method == "none":
        return None
    if method not in {"first_block", "taylor"}:
        raise ValueError(
            "Leo2 inference_cache_method must be 'none', 'first_block', or 'taylor', "
            f"got {config.inference_cache_method!r}."
        )
    if not isinstance(config.inference_cache_threshold, (int, float)):
        raise TypeError("Leo2 inference_cache_threshold must be numeric.")
    threshold = float(config.inference_cache_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Leo2 inference_cache_threshold must be finite and non-negative.")

    if method == "taylor":
        max_extrapolation = config.inference_cache_taylor_max_extrapolation
        if not isinstance(max_extrapolation, (int, float)):
            raise TypeError("Leo2 inference_cache_taylor_max_extrapolation must be numeric.")
        max_extrapolation = float(max_extrapolation)
        if not math.isfinite(max_extrapolation) or max_extrapolation < 0:
            raise ValueError("Leo2 inference_cache_taylor_max_extrapolation must be finite and non-negative.")
        from hymm.models.diffusion.leo_cache import LeoTaylorCacheConfig

        return LeoTaylorCacheConfig(threshold=threshold, max_extrapolation=max_extrapolation)

    from diffusers import FirstBlockCacheConfig

    return FirstBlockCacheConfig(threshold=threshold)


class Leo2Bundle(Bundle):
    """Loaded Leo2 components."""

    def __init__(
        self,
        *,
        model: nn.Module,
        hymm_args: Any,
        dtype: torch.dtype,
        device: torch.device,
        config: Leo2PipelineConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.hymm_args = hymm_args
        self.dtype = dtype
        self.device = device
        self.config = config
        self.pretrained_path = config.ckpt_path
        self.use_system_prompt = "li-dit-encode-visual-qwen-3.5"
        if "--use-system-prompt" in config.extra_hymm_args:
            i = config.extra_hymm_args.index("--use-system-prompt")
            self.use_system_prompt = config.extra_hymm_args[i + 1]

    @classmethod
    def from_config(cls, config: Leo2PipelineConfig) -> "Leo2Bundle":
        if config.skip_load_ckpt:
            raise ValueError(
                "Leo2Bundle does not support skip_load_ckpt: build_model uses initialize_weights=False, "
                "so skipping the native DCP load would leave uninitialized parameters."
            )
        install_transformers_flash_attention_compat()
        args = _bootstrap_hymm(config)
        inference_cache_config = _make_inference_cache_config(config)

        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", 0)))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
        device = (
            torch.device(config.device)
            if config.device
            else (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        )

        from hymm.models import build_model

        dtype = parse_torch_dtype(config.model_precision, field_name="Leo2Bundle.model_precision")
        if dtype != torch.bfloat16:
            raise ValueError(f"Leo2's pinned DCP checkpoint requires model_precision='bf16', got {dtype}.")
        model, _model_config = build_model(args, dtype=dtype, device="cpu", initialize_weights=False)

        _dcp_load_into(model, config.ckpt_path, model_dtype=dtype)
        model.requires_grad_(False)
        model.eval()
        if config.uniform_bf16:
            # FSDP2 asserts a uniform original dtype per shard group; the MoE
            # router (gate.wg) is stored fp32 inside every block. Cast the whole
            # DiT to bf16 for the smoke recipe (known trade-off: fp32 router).
            model.to(dtype=torch.bfloat16)
            # hymm computes the MoE router in an autocast-disabled fp32 region
            # (`hidden.float() @ gate.wg`); with wg now bf16 that matmul raises
            # a dtype mismatch. Patch wg.forward to follow the input dtype at
            # call time (works under FSDP: weight is unsharded by then).
            _patch_router_dtype(model)
        # UniRL's FSDPBackend shards + moves only the block classes; everything
        # else (patch/time embeds, projectors, final layers, RoPE buffers) must
        # already sit on the device or the first forward fails with a
        # cpu/cuda mismatch (H3 sidesteps this by eager .to(device) of the
        # whole transformer, impossible for a 150GB model). Move the non-block
        # root children now -- they are small.
        _move_non_block_to_device(model, device)

        # tokenizer + frozen aux models + the hymm pipeline object
        from hymm.core.extra_model_provider import build_text_encoder, build_tkwrapper, build_vae

        model.tokenizer = build_tkwrapper()
        if getattr(args, "use_vae", False):
            vae = build_vae(dp_rank=0, only_encoder=False)
            model.model_dict["vae"] = vae
        text_encoder = build_text_encoder()
        model.model_dict["text_encoder"] = text_encoder
        # Transient mode parks the 18GB conditioner on CPU and shuttles it to
        # GPU per encode; resident mode keeps it on the device (the shuttle
        # costs ~10-20 s per rollout and the peak is unchanged: the encoder is
        # on GPU during conditioning either way, never during the DiT forward).
        try:
            text_encoder.to("cpu" if config.text_encoder_gpu_transient else device)
        except Exception:
            pass

        model.load_generation_config(config.generation_config_path)
        model.build_diffusion_pipeline()
        if inference_cache_config is not None:
            model.enable_cache(inference_cache_config)

        return cls(model=model, hymm_args=args, dtype=dtype, device=device, config=config)

    def trainable_module(self) -> nn.Module:
        return self.model

    @contextlib.contextmanager
    def text_encoder_ctx(self):
        """Transiently host the text encoder on GPU for an encode pass."""
        te = self.model.model_dict.get("text_encoder")
        moved = False
        if te is not None and self.config.text_encoder_gpu_transient:
            try:
                te.to(self.device)
                moved = True
            except Exception:
                moved = False
        try:
            yield
        finally:
            if moved:
                te.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


__all__ = ["Leo2Bundle"]
