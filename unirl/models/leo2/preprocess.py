"""Offline Leo2 preprocessing without materializing DiT weights."""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
from pathlib import Path

import torch

from .config import Leo2PipelineConfig
from .preprocessing_cache import PREPROCESSING_SEED, Leo2PreprocessingCache


def build_preprocessing_bundle(config: Leo2PipelineConfig, *, with_text: bool):
    """Construct native input processors on a meta DiT shell and load only frozen codec components."""
    from unirl.models.transformers_compat import install_transformers_flash_attention_compat

    from .bundle import Leo2Bundle, _bootstrap_hymm

    install_transformers_flash_attention_compat()
    args = _bootstrap_hymm(config)
    args.local_rank = torch.cuda.current_device()
    from hymm.core.extra_model_provider import build_text_encoder, build_tkwrapper, build_vae
    from hymm.models.diffusion.leo_config import LeoConfig, core_model_config_from_args
    from hymm.models.diffusion.leo_hf import LeoModelHF

    device = torch.device(config.device)

    class PreprocessingModel(LeoModelHF):
        """Reuse native input methods while forbidding a forward on unmaterialized weights."""

        @property
        def device(self):
            return device

        def forward(self, *args, **kwargs):
            raise RuntimeError("The Leo2 preprocessing shell cannot run DiT forwards.")

    def branch(prefix: str = ""):
        name = getattr(args, f"{prefix}model_name")
        return (
            None
            if name is None
            else LeoConfig.from_name(
                name.split(".")[-1],
                **core_model_config_from_args(args, prefix=prefix),
            )
        )

    model = PreprocessingModel(
        args,
        branch(),
        txt_config=branch("text_branch_"),
        audio_config=branch("audio_branch_"),
        dtype=torch.bfloat16,
        device="meta",
        initialize_weights=False,
    )
    if any(not parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("Leo2 preprocessing unexpectedly materialized DiT parameters.")
    model.requires_grad_(False).eval()
    model.model_dict["vae"] = build_vae(dp_rank=0, only_encoder=False).requires_grad_(False).eval().to("cpu")
    model.load_generation_config(config.generation_config_path)
    if with_text:
        model.tokenizer = build_tkwrapper()
        model.model_dict["text_encoder"] = build_text_encoder().requires_grad_(False).eval()
        model.build_diffusion_pipeline()
    return Leo2Bundle(model=model, hymm_args=args, dtype=torch.bfloat16, device=device, config=config)


def _records(paths: list[str], *, encode_targets: bool) -> list[dict]:
    """Read prompt text files or the existing raw supervised manifest format."""
    from unirl.data.sft import SupervisedDataset

    records = []
    for path in paths:
        if Path(path).suffix == ".txt":
            if encode_targets:
                raise ValueError("Target encoding requires a JSON/JSONL manifest with target video media refs.")
            records.extend({"prompt": line.strip()} for line in Path(path).read_text().splitlines() if line.strip())
        else:
            records.extend(SupervisedDataset(path).records)
    if not records:
        raise ValueError("No Leo2 preprocessing records found.")
    for record in records:
        if "messages" in record or not isinstance(record.get("prompt"), str):
            raise ValueError("Leo2 preprocessing supports prompt:str T2V records only.")
        if any(ref.role != "target" for ref in record.get("media_refs", [])):
            raise ValueError("Leo2 preprocessing does not support image/video/audio prompt conditioning.")
    return records


def preprocess(
    config: Leo2PipelineConfig,
    records: list[dict],
    *,
    geometry: dict,
    encode_targets: bool,
    max_decode_frames: int = 256,
) -> dict:
    """Resume immutable condition and target caches, releasing text GPU storage before video encoding."""
    from unirl.types.primitives import Texts

    from .pipeline import Leo2Pipeline
    from .sft import target_video_uri, validate_target
    from .text_embed import Leo2CondStage
    from .vae import encode_target_video

    cache = Leo2PreprocessingCache(config, writable=True)
    if encode_targets:
        for record in records:
            target_video_uri(record)
    bundle = None
    stage = None
    counts = {"conditions_written": 0, "targets_written": 0, "records": len(records)}
    for index, record in enumerate(records):
        key = cache.condition_key(record["prompt"], **geometry)
        if not cache.contains(key):
            if bundle is None:
                bundle = build_preprocessing_bundle(config, with_text=True)
                stage = Leo2CondStage(bundle)
            conditions = stage.build(Texts(texts=[record["prompt"]]), seeds=[PREPROCESSING_SEED], **geometry)
            blob = {key: value for key, value in conditions.hymm[0].items() if key not in ("captured_call", "_device")}
            cache.write(key, blob)
            counts["conditions_written"] += 1
            del conditions, blob
        print(f"[leo2 preprocess] conditions {index + 1}/{len(records)}", flush=True)
    if bundle is not None:
        bundle.model.model_dict["text_encoder"].to("cpu")
        del stage
        gc.collect()
        torch.cuda.empty_cache()
    if encode_targets:
        for index, record in enumerate(records):
            conditions = cache.read_condition(cache.condition_key(record["prompt"], **geometry))
            blob = conditions.hymm[0]
            uri = target_video_uri(record)
            key = cache.target_key(uri, blob, max_decode_frames=max_decode_frames)
            if not cache.contains(key):
                if bundle is None:
                    bundle = build_preprocessing_bundle(config, with_text=False)
                latent = encode_target_video(bundle, uri, blob, max_decode_frames=max_decode_frames)
                validate_target(latent, Leo2Pipeline._latent_shape_from_conditions(conditions))
                cache.write(key, latent)
                counts["targets_written"] += 1
            print(f"[leo2 preprocess] targets {index + 1}/{len(records)}", flush=True)
    print(f"[leo2 preprocess] {counts}; cache={cache.root}", flush=True)
    return counts


def main() -> None:
    """Resolve a training recipe and preprocess one independently scheduled dataset shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="Recipe under examples/, e.g. sft/leo2_sft_cached")
    parser.add_argument("--manifest", nargs="+", required=True, help="Prompt .txt or raw .json/.jsonl manifests")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--encode-targets", action="store_true")
    parser.add_argument("--max-decode-frames", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--override", action="append", default=[], help="Hydra override, repeatable")
    cli = parser.parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if not 0 <= cli.shard_index < cli.num_shards:
        parser.error("Require 0 <= shard-index < num-shards.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run each preprocessing shard as a separate single-GPU process (WORLD_SIZE=1).")
    examples = Path(__file__).resolve().parents[3] / "examples"
    with initialize_config_dir(version_base=None, config_dir=str(examples)):
        recipe = compose(config_name=cli.recipe, overrides=cli.override)
    raw = OmegaConf.to_container(recipe.bundle.config, resolve=True)
    if raw.pop("_target_", None) != "unirl.models.leo2.config.Leo2PipelineConfig":
        parser.error("The recipe must use Leo2PipelineConfig.")
    raw.update(
        preprocessing_cache_dir=cli.cache_dir,
        preprocessing_cache_mode="off",
        load_video_vae=True,
        vae_on_gpu=False,
        text_encoder_gpu_transient=False,
        condition_cache_size=0,
        context_parallel_size=1,
        expert_parallel_size=1,
        enable_deepep=False,
    )
    records = _records(cli.manifest, encode_targets=cli.encode_targets)[cli.shard_index :: cli.num_shards]
    if not records:
        print("[leo2 preprocess] empty shard", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Native Leo2 preprocessing requires CUDA and the pinned frozen encoder assets.")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    raw["device"] = f"cuda:{torch.cuda.current_device()}"
    config = Leo2PipelineConfig(**raw)
    geometry = {name: int(recipe.sampling[name]) for name in ("height", "width", "num_frames")}
    with tempfile.TemporaryDirectory(prefix="leo2-preprocess-") as temp_dir:
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"file://{temp_dir}/rendezvous",
            rank=0,
            world_size=1,
        )
        try:
            preprocess(
                config,
                records,
                geometry=geometry,
                encode_targets=cli.encode_targets,
                max_decode_frames=cli.max_decode_frames,
            )
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
