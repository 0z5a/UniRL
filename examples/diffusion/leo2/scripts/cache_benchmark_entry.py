"""Instrument the native Leo2 sampler for paired cache benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUEST_MARKER = "LEO2_CACHE_BENCH_REQUEST_JSON="
CONFIG_MARKER = "LEO2_CACHE_BENCH_CONFIG_JSON="
TAIL_METHODS = {"first_block", "taylor", "magcache", "magcache_calibrate"}
COUNTER_FIELDS = (
    "full_steps",
    "skipped_steps",
    "tail_compute_steps",
    "tail_reuse_steps",
    "predicted_steps",
    "static_fallback_steps",
    "attention_compute_calls",
    "attention_reuse_calls",
    "cfg_compute_calls",
    "cfg_reuse_calls",
    "cache_bytes",
)


@dataclass(frozen=True)
class BenchmarkOptions:
    """Hold benchmark-only options removed before hymm CLI parsing."""

    method: str
    cache_threshold: float | None
    latent_dir: Path
    prompts_by_seed: dict[int, tuple[int, str]]
    require_cache_hit: bool
    baseline_case: str | None
    reference_root: Path | None
    taylor_max_extrapolation: float
    magcache_profile: Path | None
    magcache_threshold: float | None
    magcache_max_skip_steps: int
    magcache_retention_ratio: float
    dfr_start_step: int
    dfr_end_step: int
    dfr_interval: int
    dfr_layers: tuple[int, ...] | None


def _bootstrap_vendor() -> None:
    """Expose only this checkout and its vendored Leo2 dependencies."""
    repo_root = Path(__file__).resolve().parents[4]
    vendor_root = repo_root / "unirl/models/leo2/vendor/gen_ar"
    for path in (vendor_root, vendor_root / "deps/hy_parallelism", vendor_root / "deps/IndexKits"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    from unirl.models.transformers_compat import install_transformers_flash_attention_compat

    install_transformers_flash_attention_compat()


def _non_negative_float(raw: str, *, field: str) -> float:
    """Parse a finite non-negative benchmark option."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative.")
    return value


def _parse_threshold(raw: str | None) -> float | None:
    """Parse a threshold while retaining the original cache-off aliases."""
    if raw is None or raw.strip().lower() in {"off", "none", "disabled"}:
        return None
    return _non_negative_float(raw, field="Leo2 cache threshold")


def _parse_layers(raw: str | None) -> tuple[int, ...] | None:
    """Parse semicolon-separated layer indices and inclusive ranges."""
    if raw is None or raw.strip().lower() in {"", "all"}:
        return None
    layers = set()
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            raise ValueError(f"Invalid empty Leo2 DFR layer token in {raw!r}.")
        if "-" in token:
            left, separator, right = token.partition("-")
            if not separator or not left.isdigit() or not right.isdigit():
                raise ValueError(f"Invalid Leo2 DFR layer range: {token!r}.")
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f"Leo2 DFR layer range is reversed: {token!r}.")
            layers.update(range(start, end + 1))
        elif token.isdigit():
            layers.add(int(token))
        else:
            raise ValueError(f"Invalid Leo2 DFR layer index: {token!r}.")
    return tuple(sorted(layers))


def _parse_benchmark_options() -> BenchmarkOptions:
    """Remove benchmark-only options before hymm parses its CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--leo2-cache-method",
        choices=("off", "first_block", "taylor", "magcache", "magcache_calibrate", "fastercache_dfr"),
    )
    parser.add_argument("--leo2-cache-threshold")
    parser.add_argument("--leo2-latent-dir", type=Path, required=True)
    parser.add_argument("--leo2-prompt-csv", type=Path, required=True)
    parser.add_argument("--leo2-require-cache-hit", action="store_true")
    parser.add_argument("--leo2-baseline-case")
    parser.add_argument("--leo2-reference-root", type=Path)
    parser.add_argument("--leo2-taylor-max-extrapolation", default="1.0")
    parser.add_argument("--leo2-magcache-profile", type=Path)
    parser.add_argument("--leo2-magcache-threshold")
    parser.add_argument("--leo2-magcache-max-skip-steps", type=int, default=4)
    parser.add_argument("--leo2-magcache-retention-ratio", default="0.2")
    parser.add_argument("--leo2-dfr-start-step", type=int, default=4)
    parser.add_argument("--leo2-dfr-end-step", type=int, default=46)
    parser.add_argument("--leo2-dfr-interval", type=int, default=2)
    parser.add_argument("--leo2-dfr-layers")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    threshold = _parse_threshold(args.leo2_cache_threshold)
    method = args.leo2_cache_method
    if method is None:
        if args.leo2_cache_threshold is None:
            raise ValueError("Set --leo2-cache-method, or use the legacy --leo2-cache-threshold option.")
        method = "off" if threshold is None else "first_block"
    if method == "off" and threshold is not None:
        raise ValueError("Leo2 cache method 'off' cannot have a cache threshold.")
    if method in {"first_block", "taylor"} and threshold is None:
        raise ValueError(f"Leo2 cache method {method!r} requires --leo2-cache-threshold.")
    if method not in {"first_block", "taylor", "off"} and args.leo2_cache_threshold is not None:
        raise ValueError(f"Leo2 cache method {method!r} does not use --leo2-cache-threshold.")
    magcache_threshold = (
        None
        if args.leo2_magcache_threshold is None
        else _non_negative_float(args.leo2_magcache_threshold, field="Leo2 MagCache threshold")
    )
    if method in {"magcache", "magcache_calibrate"} and magcache_threshold is None:
        raise ValueError(f"Leo2 cache method {method!r} requires --leo2-magcache-threshold.")
    if method == "magcache" and args.leo2_magcache_profile is None:
        raise ValueError("Leo2 cache method 'magcache' requires --leo2-magcache-profile.")
    if args.leo2_magcache_max_skip_steps < 0:
        raise ValueError("Leo2 MagCache max skip steps must be non-negative.")
    retention_ratio = _non_negative_float(args.leo2_magcache_retention_ratio, field="Leo2 MagCache retention ratio")
    if retention_ratio > 1:
        raise ValueError("Leo2 MagCache retention ratio must not exceed one.")
    if args.leo2_dfr_start_step < 0 or args.leo2_dfr_end_step < 0:
        raise ValueError("Leo2 DFR start/end steps must be non-negative.")
    if args.leo2_dfr_start_step >= args.leo2_dfr_end_step:
        raise ValueError("Leo2 DFR start step must be less than its end step.")
    if args.leo2_dfr_interval <= 0:
        raise ValueError("Leo2 DFR interval must be positive.")

    prompts_by_seed: dict[int, tuple[int, str]] = {}
    prompt_indices = set()
    prompt_hashes = set()
    with args.leo2_prompt_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            prompt_index = int(row["index"])
            prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
            if seed in prompts_by_seed:
                raise ValueError(f"Duplicate benchmark seed: {seed}")
            if prompt_index in prompt_indices:
                raise ValueError(f"Duplicate benchmark prompt index: {prompt_index}")
            if prompt_hash in prompt_hashes:
                raise ValueError(f"Duplicate benchmark prompt content at index {prompt_index}")
            prompts_by_seed[seed] = (prompt_index, prompt_hash)
            prompt_indices.add(prompt_index)
            prompt_hashes.add(prompt_hash)
    if not prompts_by_seed:
        raise ValueError(f"Benchmark prompt CSV is empty: {args.leo2_prompt_csv}")
    return BenchmarkOptions(
        method=method,
        cache_threshold=threshold,
        latent_dir=args.leo2_latent_dir.resolve(),
        prompts_by_seed=prompts_by_seed,
        require_cache_hit=args.leo2_require_cache_hit,
        baseline_case=args.leo2_baseline_case,
        reference_root=args.leo2_reference_root.resolve() if args.leo2_reference_root else None,
        taylor_max_extrapolation=_non_negative_float(
            args.leo2_taylor_max_extrapolation, field="Leo2 Taylor max extrapolation"
        ),
        magcache_profile=args.leo2_magcache_profile.resolve() if args.leo2_magcache_profile else None,
        magcache_threshold=magcache_threshold,
        magcache_max_skip_steps=args.leo2_magcache_max_skip_steps,
        magcache_retention_ratio=retention_ratio,
        dfr_start_step=args.leo2_dfr_start_step,
        dfr_end_step=args.leo2_dfr_end_step,
        dfr_interval=args.leo2_dfr_interval,
        dfr_layers=_parse_layers(args.leo2_dfr_layers),
    )


def _distributed_extrema(values: list[float]) -> tuple[list[float], list[float]]:
    """Return elementwise minima and maxima from all ranks."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return values, values
    device = torch.device("cuda", torch.cuda.current_device())
    minimum = torch.tensor(values, dtype=torch.float64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return (
        [float(value) for value in minimum.cpu().tolist()],
        [float(value) for value in maximum.cpu().tolist()],
    )


def _json_seed(seed: Any) -> Any:
    """Convert sampler seed containers to JSON-compatible values."""
    if hasattr(seed, "detach"):
        seed = seed.detach().cpu()
    if hasattr(seed, "tolist"):
        return seed.tolist()
    if isinstance(seed, tuple):
        return list(seed)
    return seed


def _scalar_seed(seed: Any) -> int:
    """Resolve the batch-one sampler seed to an integer."""
    value = _json_seed(seed)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"Expected one integer seed, got {value!r}.")
    return int(value)


def _package_versions() -> dict[str, str | None]:
    """Record relevant installed distribution versions without importing extensions."""
    versions = {}
    for name in ("torch", "diffusers", "transformers", "flash-attn", "deep-ep", "nvidia-nvshmem-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _load_magcache_profile(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], str]:
    """Load a finite MagCache profile and return its content digest."""
    if not path.is_file():
        raise FileNotFoundError(f"Leo2 MagCache profile does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Leo2 MagCache profile must be a JSON object: {path}")
    ratios = payload.get("ratios")
    timesteps = payload.get("expected_timesteps")
    if not isinstance(ratios, list) or not ratios:
        raise ValueError(f"Leo2 MagCache profile has no non-empty 'ratios' list: {path}")
    if not isinstance(timesteps, list) or not timesteps:
        raise ValueError(f"Leo2 MagCache profile has no non-empty 'expected_timesteps' list: {path}")
    parsed_ratios = tuple(_non_negative_float(str(value), field="MagCache profile ratio") for value in ratios)
    parsed_timesteps = tuple(_non_negative_float(str(value), field="MagCache profile timestep") for value in timesteps)
    return parsed_ratios, parsed_timesteps, hashlib.sha256(path.read_bytes()).hexdigest()


def _build_cache_config(options: BenchmarkOptions) -> tuple[object | None, dict[str, Any]]:
    """Build the internal cache config selected by the benchmark method."""
    from hymm.models.diffusion import leo_cache

    if options.method == "off":
        return None, {}
    if options.method == "first_block":
        config = leo_cache.LeoFirstBlockCacheConfig(threshold=options.cache_threshold)
        return config, {"threshold": options.cache_threshold}
    if options.method == "taylor":
        config = leo_cache.LeoTaylorCacheConfig(
            threshold=options.cache_threshold,
            max_extrapolation=options.taylor_max_extrapolation,
        )
        return config, {
            "threshold": options.cache_threshold,
            "max_extrapolation": options.taylor_max_extrapolation,
        }
    if options.method in {"magcache", "magcache_calibrate"}:
        ratios: tuple[float, ...] = ()
        timesteps: tuple[float, ...] = ()
        profile_sha256 = None
        if options.magcache_profile is not None:
            ratios, timesteps, profile_sha256 = _load_magcache_profile(options.magcache_profile)
        config = leo_cache.LeoMagCacheConfig(
            threshold=options.magcache_threshold,
            max_skip_steps=options.magcache_max_skip_steps,
            retention_ratio=options.magcache_retention_ratio,
            ratios=ratios,
            expected_timesteps=timesteps,
            calibrate=options.method == "magcache_calibrate",
        )
        return config, {
            "threshold": options.magcache_threshold,
            "max_skip_steps": options.magcache_max_skip_steps,
            "retention_ratio": options.magcache_retention_ratio,
            "profile": str(options.magcache_profile) if options.magcache_profile else None,
            "profile_sha256": profile_sha256,
            "profile_ratio_count": len(ratios),
            "profile_timestep_count": len(timesteps),
            "calibrate": options.method == "magcache_calibrate",
        }
    config = leo_cache.LeoFasterCacheConfig(
        start_step=options.dfr_start_step,
        end_step=options.dfr_end_step,
        interval=options.dfr_interval,
        layers=options.dfr_layers,
    )
    return config, {
        "start_step": options.dfr_start_step,
        "end_step": options.dfr_end_step,
        "interval": options.dfr_interval,
        "layers": list(options.dfr_layers) if options.dfr_layers is not None else None,
    }


def _counter(value: Any, *, field: str) -> int:
    """Validate one non-negative integral cache statistic."""
    if not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or int(value) != value or value < 0:
        raise ValueError(f"Leo2 cache statistic {field!r} is not a non-negative integer: {value!r}")
    return int(value)


def _normalize_cache_stats(stats: Any, method: str) -> dict[str, int]:
    """Normalize method-specific counters into stable benchmark fields."""
    if not isinstance(stats, dict):
        raise TypeError(f"Leo2 cache_stats() must return a dict, got {type(stats).__name__}.")
    full_steps = _counter(stats.get("full_steps", 0), field="full_steps")
    skipped_steps = _counter(stats.get("skipped_steps", 0), field="skipped_steps")
    normalized = {
        "full_steps": full_steps,
        "skipped_steps": skipped_steps,
        "tail_compute_steps": _counter(
            stats.get("tail_compute_steps", full_steps if method in TAIL_METHODS else 0),
            field="tail_compute_steps",
        ),
        "tail_reuse_steps": _counter(
            stats.get("tail_reuse_steps", skipped_steps if method in TAIL_METHODS else 0),
            field="tail_reuse_steps",
        ),
        "predicted_steps": _counter(stats.get("predicted_steps", 0), field="predicted_steps"),
        "static_fallback_steps": _counter(stats.get("static_fallback_steps", 0), field="static_fallback_steps"),
        "attention_compute_calls": _counter(stats.get("attention_compute_calls", 0), field="attention_compute_calls"),
        "attention_reuse_calls": _counter(stats.get("attention_reuse_calls", 0), field="attention_reuse_calls"),
        "cfg_compute_calls": _counter(stats.get("cfg_compute_calls", 0), field="cfg_compute_calls"),
        "cfg_reuse_calls": _counter(stats.get("cfg_reuse_calls", 0), field="cfg_reuse_calls"),
        "cache_bytes": _counter(stats.get("cache_bytes", 0), field="cache_bytes"),
    }
    return normalized


def _validate_cache_stats(stats: dict[str, int], *, method: str, expected_steps: int) -> None:
    """Apply accounting invariants appropriate to each cache family."""
    if method == "off":
        active = {field: value for field, value in stats.items() if field != "cache_bytes" and value}
        if active or stats["cache_bytes"]:
            raise RuntimeError(f"Cache-off control unexpectedly reported cache activity: {stats}")
        return
    if method in TAIL_METHODS:
        accounted = stats["tail_compute_steps"] + stats["tail_reuse_steps"]
        if accounted != expected_steps:
            raise RuntimeError(f"Tail cache accounted for {accounted}/{expected_steps} denoising steps.")
        if stats["predicted_steps"] + stats["static_fallback_steps"] > stats["tail_reuse_steps"]:
            raise RuntimeError("Taylor prediction/fallback counters exceed tail reuse steps.")
    elif method == "fastercache_dfr":
        if stats["tail_compute_steps"] or stats["tail_reuse_steps"]:
            raise RuntimeError("FasterCache DFR unexpectedly reported whole-tail cache steps.")


def _save_latent(
    latent: Any,
    *,
    options: BenchmarkOptions,
    prompt_index: int,
    prompt_hash: str,
    seed: int,
) -> tuple[str, str | None]:
    """Persist one rank-zero final VAE-input latent and its identity."""
    import torch
    import torch.distributed as dist

    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"Expected final video latent tensor, got {type(latent).__name__}.")
    filename = f"{prompt_index:02d}_{seed}.pt"
    digest = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        options.latent_dir.mkdir(parents=True, exist_ok=True)
        path = options.latent_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "latent": latent.detach().to(device="cpu").contiguous(),
                "latent_space": "denormalized_vae_input",
                "prompt_hash": prompt_hash,
                "prompt_index": prompt_index,
                "seed": seed,
            },
            temporary,
        )
        os.replace(temporary, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if dist.is_initialized():
        dist.barrier()
    return filename, digest


def _install_instrumentation(options: BenchmarkOptions) -> None:
    """Enable cache after loading and wrap each video request with metrics."""
    import torch
    import torch.distributed as dist
    from hymm.core.global_vars import get_args
    from hymm.models.diffusion.leo_cache import LeoFirstBlockCacheController
    from hymm.models.diffusion.leo_hf import LeoModelHF
    from hymm.samplers.hunyuan_multimodal_sampler import HunyuanMultimodalSampler

    cache_config, method_options = _build_cache_config(options)

    original_init = HunyuanMultimodalSampler.__init__

    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if self.model.is_cache_enabled:
            self.model.disable_cache()
        if cache_config is not None:
            self.model.enable_cache(cache_config)
        sync_plan = None
        if options.method in TAIL_METHODS:
            sync_plan = LeoFirstBlockCacheController._synchronization_plan(self.model.layers[0])
        if options.method in TAIL_METHODS and sync_plan is None:
            raise RuntimeError("Leo2 cache benchmark topology is unsupported; cache decisions would always fall back.")
        if self.rank == 0:
            generation_config = self.model.generation_config
            parallel_state = self.parallel_state.backend_state
            payload = {
                "baseline_case": options.baseline_case,
                "benchmark_schema_version": 2,
                "cache_decision_supported": sync_plan is not None if options.method in TAIL_METHODS else None,
                "cache_enabled": bool(self.model.is_cache_enabled),
                "cache_method": options.method,
                "cache_threshold": options.cache_threshold,
                "cache_method_options": method_options,
                "require_cache_hit": options.require_cache_hit,
                "communication_env": {
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                    "MASTER_PORT": os.environ.get("MASTER_PORT"),
                    "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
                    "NCCL_IB_GID_INDEX": os.environ.get("NCCL_IB_GID_INDEX"),
                    "NCCL_NET": os.environ.get("NCCL_NET"),
                    "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE"),
                    "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
                    "NVSHMEM_HOME": os.environ.get("NVSHMEM_HOME"),
                },
                "diff_infer_steps": int(generation_config.diff_infer_steps),
                "effective_args": {
                    name: getattr(get_args(), name, None)
                    for name in (
                        "attn_impl",
                        "compile_engine",
                        "deepep_moe_shared_expert_overlap",
                        "fsdp_impl",
                        "gate_impl",
                        "moe_enable_deepep",
                        "moe_grouped_gemm",
                        "moe_impl",
                        "overlap_grad_reduce",
                        "overlap_param_gather",
                    )
                },
                "flow_shift_video": float(generation_config.flow_shift_video),
                "guidance_scale": float(generation_config.diff_guidance_scale),
                "gpu": {
                    "capability": list(torch.cuda.get_device_capability()),
                    "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(),
                    "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                },
                "image_size": get_args().image_size,
                "inter_request_barrier": True,
                "num_frames": int(get_args().num_frames),
                "parallelism": {
                    name: int(getattr(parallel_state, name))
                    for name in ("dp_replicate", "dp_shard", "tp", "etp", "pp", "ep", "cp")
                },
                "runtime": {
                    "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "distributions": _package_versions(),
                    "executable": sys.executable,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                },
                "reference_root": str(options.reference_root) if options.reference_root else None,
                "timing_scope": "prepare_model_inputs+text_conditioning+denoise+vae_decode",
                "world_size": int(self.world_size),
            }
            print(CONFIG_MARKER + json.dumps(payload, sort_keys=True), flush=True)

    HunyuanMultimodalSampler.__init__ = instrumented_init

    original_generate_video = LeoModelHF.generate_video
    request_number = 0

    def measured_generate_video(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal request_number
        request_number += 1
        seed = _scalar_seed(kwargs.get("seed"))
        try:
            prompt_index, prompt_hash = options.prompts_by_seed[seed]
        except KeyError as exc:
            raise KeyError(f"Sampler seed {seed} is absent from the benchmark prompt CSV.") from exc

        def fail(reason: str, error_type: str = "RuntimeError") -> None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                payload = {
                    "error": reason,
                    "error_type": error_type,
                    "prompt_hash": prompt_hash,
                    "prompt_index": prompt_index,
                    "request": request_number,
                    "seed": seed,
                    "status": "error",
                }
                print(REQUEST_MARKER + json.dumps(payload, sort_keys=True), flush=True)

        kwargs["return_latents"] = True
        # Rank zero writes the previous request's MP4 after ``generate_video``
        # returns, while the other ranks can reach the next request immediately.
        # Synchronize before starting the timer so that previous-request output
        # encoding is never charged to the following request.
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            generated = original_generate_video(self, *args, **kwargs)
        except BaseException as exc:
            fail(str(exc), type(exc).__name__)
            raise
        if not isinstance(generated, tuple) or len(generated) != 2:
            reason = "Leo2 return_latents=True did not return (outputs, latent_outputs)."
            fail(reason, "TypeError")
            raise TypeError(reason)
        output, latent_outputs = generated
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = float(torch.cuda.max_memory_allocated())
            reserved = float(torch.cuda.max_memory_reserved())
        else:
            allocated = 0.0
            reserved = 0.0
        elapsed = time.perf_counter() - started
        stats = _normalize_cache_stats(self.cache_stats(), options.method)
        minima, maxima = _distributed_extrema(
            [elapsed, allocated, reserved, *(float(stats[field]) for field in COUNTER_FIELDS)]
        )
        if minima[3:] != maxima[3:]:
            reason = f"Leo2 cache counters diverged across ranks: min={minima[3:]}, max={maxima[3:]}"
            fail(reason)
            raise RuntimeError(reason)
        elapsed, allocated, reserved = maxima[:3]
        stats = {field: int(value) for field, value in zip(COUNTER_FIELDS, maxima[3:])}
        expected_steps = int(self.generation_config.diff_infer_steps)
        try:
            _validate_cache_stats(stats, method=options.method, expected_steps=expected_steps)
        except RuntimeError as exc:
            fail(str(exc))
            raise
        if options.method in {"first_block", "taylor"} and options.cache_threshold == 0 and stats["tail_reuse_steps"]:
            reason = "Threshold-zero correctness control unexpectedly skipped denoising steps."
            fail(reason)
            raise RuntimeError(reason)
        latent_file, latent_sha256 = _save_latent(
            latent_outputs.videos,
            options=options,
            prompt_index=prompt_index,
            prompt_hash=prompt_hash,
            seed=seed,
        )
        reuse_units = stats["tail_reuse_steps"] + stats["attention_reuse_calls"] + stats["cfg_reuse_calls"]
        if options.require_cache_hit and reuse_units == 0:
            reason = f"Cache method {options.method!r} reused no work for prompt {prompt_index}."
            fail(reason)
            raise RuntimeError(reason)
        if not dist.is_initialized() or dist.get_rank() == 0:
            payload = {
                "baseline_case": options.baseline_case,
                "benchmark_schema_version": 2,
                "cache_method": options.method,
                "cache_threshold": options.cache_threshold,
                "elapsed_seconds": elapsed,
                "flow_shift_video": float(self.generation_config.flow_shift_video),
                "guidance_scale": float(self.generation_config.diff_guidance_scale),
                "latent_dtype": str(latent_outputs.videos.dtype),
                "latent_file": latent_file,
                "latent_sha256": latent_sha256,
                "latent_shape": list(latent_outputs.videos.shape),
                "max_memory_allocated_bytes": int(allocated),
                "max_memory_reserved_bytes": int(reserved),
                "prompt_hash": prompt_hash,
                "prompt_index": prompt_index,
                "request": request_number,
                "reference_root": str(options.reference_root) if options.reference_root else None,
                "seed": seed,
                "status": "ok",
                **stats,
            }
            print(REQUEST_MARKER + json.dumps(payload, sort_keys=True), flush=True)
        return output

    LeoModelHF.generate_video = measured_generate_video


def main() -> None:
    """Run the native sampler with benchmark-only cache instrumentation."""
    options = _parse_benchmark_options()
    _bootstrap_vendor()
    _install_instrumentation(options)

    from hymm.samplers.entry import run

    run()


if __name__ == "__main__":
    main()
