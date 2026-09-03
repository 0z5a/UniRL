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


@dataclass(frozen=True)
class BenchmarkOptions:
    """Hold benchmark-only options removed before hymm CLI parsing."""

    cache_threshold: float | None
    latent_dir: Path
    prompts_by_seed: dict[int, tuple[int, str]]
    require_cache_hit: bool


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


def _parse_benchmark_options() -> BenchmarkOptions:
    """Remove benchmark-only options before hymm parses its CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--leo2-cache-threshold", required=True)
    parser.add_argument("--leo2-latent-dir", type=Path, required=True)
    parser.add_argument("--leo2-prompt-csv", type=Path, required=True)
    parser.add_argument("--leo2-require-cache-hit", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    raw = args.leo2_cache_threshold.strip().lower()
    if raw in {"off", "none", "disabled"}:
        threshold = None
    else:
        try:
            threshold = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid Leo2 cache threshold: {args.leo2_cache_threshold!r}") from exc
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Leo2 cache threshold must be finite and non-negative.")

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
        cache_threshold=threshold,
        latent_dir=args.leo2_latent_dir.resolve(),
        prompts_by_seed=prompts_by_seed,
        require_cache_hit=args.leo2_require_cache_hit,
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

    cache_config = None
    if options.cache_threshold is not None:
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError as exc:
            raise RuntimeError("This cache benchmark requires diffusers.FirstBlockCacheConfig.") from exc
        cache_config = FirstBlockCacheConfig(threshold=options.cache_threshold)

    original_init = HunyuanMultimodalSampler.__init__

    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if self.model.is_cache_enabled:
            self.model.disable_cache()
        if cache_config is not None:
            self.model.enable_cache(cache_config)
        sync_plan = LeoFirstBlockCacheController._synchronization_plan(self.model.layers[0])
        if sync_plan is None:
            raise RuntimeError("Leo2 cache benchmark topology is unsupported; cache decisions would always fall back.")
        if self.rank == 0:
            generation_config = self.model.generation_config
            parallel_state = self.parallel_state.backend_state
            payload = {
                "cache_decision_supported": True,
                "cache_enabled": bool(self.model.is_cache_enabled),
                "cache_threshold": options.cache_threshold,
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
        stats = self.cache_stats()
        minima, maxima = _distributed_extrema(
            [elapsed, allocated, reserved, float(stats["full_steps"]), float(stats["skipped_steps"])]
        )
        if minima[3:] != maxima[3:]:
            reason = f"Leo2 cache full/skipped step counters diverged across ranks: min={minima[3:]}, max={maxima[3:]}"
            fail(reason)
            raise RuntimeError(reason)
        elapsed, allocated, reserved, full_steps, skipped_steps = maxima
        expected_steps = int(self.generation_config.diff_infer_steps)
        if options.cache_threshold is None:
            if full_steps or skipped_steps:
                reason = "Cache-off control unexpectedly reported cache steps."
                fail(reason)
                raise RuntimeError(reason)
        elif int(full_steps + skipped_steps) != expected_steps:
            reason = f"Cache accounted for {int(full_steps + skipped_steps)}/{expected_steps} denoising steps."
            fail(reason)
            raise RuntimeError(reason)
        elif options.cache_threshold == 0 and skipped_steps:
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
        if options.require_cache_hit and skipped_steps == 0:
            reason = f"Cache threshold {options.cache_threshold} skipped no steps for prompt {prompt_index}."
            fail(reason)
            raise RuntimeError(reason)
        if not dist.is_initialized() or dist.get_rank() == 0:
            payload = {
                "cache_threshold": options.cache_threshold,
                "elapsed_seconds": elapsed,
                "flow_shift_video": float(self.generation_config.flow_shift_video),
                "full_steps": int(full_steps),
                "latent_dtype": str(latent_outputs.videos.dtype),
                "latent_file": latent_file,
                "latent_sha256": latent_sha256,
                "latent_shape": list(latent_outputs.videos.shape),
                "max_memory_allocated_bytes": int(allocated),
                "max_memory_reserved_bytes": int(reserved),
                "prompt_hash": prompt_hash,
                "prompt_index": prompt_index,
                "request": request_number,
                "seed": seed,
                "skipped_steps": int(skipped_steps),
                "status": "ok",
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
