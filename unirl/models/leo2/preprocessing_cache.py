"""Persist frozen Leo2 conditions and supervised targets as CPU tensors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import _QWEN_ASSET_FILES, _VAE_ASSET_FILES

if TYPE_CHECKING:
    from .conditions import Leo2Conditions

_SCHEMA = 1
PREPROCESSING_SEED = 0


def _digest(value: Any) -> str:
    """Hash a canonical JSON cache identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _file_identity(path: Path) -> dict:
    """Fingerprint small files by content and large weights by size and modification time."""
    stat = path.stat()
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size < 1024 * 1024:
        identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def map_tensors(value: Any, device: torch.device | str) -> Any:
    """Copy containers and move every nested tensor without changing its dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: map_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(item, device) for item in value)
    return value


def _pack(value: Any) -> Any:
    """Encode container types and slices using only weights-only-safe values."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value.detach().cpu().contiguous().clone())
    if isinstance(value, slice):
        return ("slice", (value.start, value.stop, value.step))
    if isinstance(value, dict):
        return ("dict", {key: _pack(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_pack(item) for item in value])
    if value is None or type(value) in (str, int, float, bool):
        return ("scalar", value)
    raise TypeError(f"Unsupported Leo2 cache value: {type(value).__name__}")


def _unpack(node: Any) -> Any:
    """Restore native transport containers from the cache representation."""
    kind, value = node
    if kind in ("tensor", "scalar"):
        return value
    if kind == "slice":
        return slice(*value)
    if kind == "dict":
        return {key: _unpack(item) for key, item in value.items()}
    if kind in ("list", "tuple"):
        items = [_unpack(item) for item in value]
        return tuple(items) if kind == "tuple" else items
    raise ValueError(f"Unknown Leo2 cache node: {kind!r}")


class Leo2PreprocessingCache:
    """Read immutable per-example tensor files under a preprocessing fingerprint."""

    def __init__(self, config, *, writable: bool = False) -> None:
        if not config.preprocessing_cache_dir:
            raise ValueError("Set preprocessing_cache_dir before using the Leo2 preprocessing cache.")
        repo = Path(config.hymm_repo_path)
        assets = Path(config.assets_base)
        asset_files = [assets / "text_encoder/Qwen3.5-9B" / name for name in _QWEN_ASSET_FILES]
        asset_files += [assets / "image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2" / n for n in _VAE_ASSET_FILES]
        source_files = sorted((repo / "hymm").rglob("*.py")) + sorted((repo / "processors").rglob("*.py"))
        identity = {
            "schema": _SCHEMA,
            "seed": PREPROCESSING_SEED,
            "config": hashlib.sha256(Path(config.config_yaml).read_bytes()).hexdigest(),
            "generation": hashlib.sha256(Path(config.generation_config_path).read_bytes()).hexdigest(),
            "args": list(config.extra_hymm_args),
            "assets": {str(p.relative_to(assets)): _file_identity(p) for p in asset_files},
            "native_code": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "adapter_code": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("text_embed.py", "vae.py", "preprocess.py")
            },
        }
        self.fingerprint = _digest(identity)
        self.root = Path(config.preprocessing_cache_dir).expanduser().resolve() / self.fingerprint
        self.writable = writable
        if writable:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise FileNotFoundError(
                f"Leo2 preprocessing cache missing or stale: {self.root}. Run leo2.preprocess first."
            )

    def condition_key(self, prompt: str, *, height: int, width: int, num_frames: int) -> str:
        """Identify a prompt and requested geometry independently of rollout noise seeds."""
        return _digest(["condition", prompt, int(height), int(width), int(num_frames)])

    def target_key(self, uri: str, blob: dict, *, max_decode_frames: int) -> str:
        """Identify deterministic target encoding including source revision and effective geometry."""
        path = Path(uri).expanduser().resolve()
        return _digest(
            [
                "target",
                str(path),
                _file_identity(path),
                blob["image_size"],
                blob["video_duration"],
                max_decode_frames,
            ]
        )

    def contains(self, key: str) -> bool:
        """Check whether an atomically published entry exists."""
        return (self.root / f"{key}.pt").is_file()

    def read(self, key: str) -> Any:
        """Load an entry on CPU and fail explicitly when preprocessing is incomplete."""
        path = self.root / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Leo2 preprocessing entry missing: {path}. Preprocess this prompt/target and geometry."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != _SCHEMA or payload["fingerprint"] != self.fingerprint or payload["key"] != key:
            raise ValueError(f"Leo2 preprocessing identity mismatch: {path}")
        return _unpack(payload["data"])

    def write(self, key: str, value: Any) -> None:
        """Publish an entry atomically so interrupted or concurrent writers cannot expose partial tensors."""
        if not self.writable:
            raise RuntimeError("Leo2 training caches are readonly; use the preprocessing command to write entries.")
        payload = {"schema": _SCHEMA, "fingerprint": self.fingerprint, "key": key, "data": _pack(value)}
        fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
            os.replace(tmp, self.root / f"{key}.pt")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def read_condition(self, key: str) -> Leo2Conditions:
        """Validate the complete native conditioning blob without a redundant text embedding copy."""
        from .conditions import Leo2Conditions

        return Leo2Conditions.from_dict({"hymm": [self.read(key)]})
