"""Validate Leo2 external artifacts against the packaged manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path() -> Path:
    """Return the installed Leo2 artifact-manifest path."""
    return Path(__file__).resolve().parent / "resources/artifacts.yaml"


def main() -> None:
    """Validate configured artifact roots, sizes, and checksums."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=manifest_path())
    parser.add_argument("--require-checksums", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported Leo2 artifact manifest: {args.manifest}")

    failures: list[str] = []
    checked = 0
    hashed = 0
    for artifact_name, artifact in manifest["artifacts"].items():
        env_name = artifact["root_env"]
        root_value = os.environ.get(env_name, "")
        if not root_value:
            failures.append(f"{artifact_name}: required environment variable {env_name} is unset")
            continue
        root = Path(root_value).expanduser().resolve()
        for spec in artifact["files"]:
            path = root / spec["path"]
            if not path.is_file():
                failures.append(f"{artifact_name}: missing {path}")
                continue
            checked += 1
            if path.stat().st_size != spec["size"]:
                failures.append(f"{artifact_name}: size mismatch for {path}")
                continue
            expected_hash = spec.get("sha256")
            if expected_hash:
                hashed += 1
                if _sha256(path) != expected_hash:
                    failures.append(f"{artifact_name}: SHA-256 mismatch for {path}")
            elif args.require_checksums:
                failures.append(f"{artifact_name}: manifest has no SHA-256 for {path}")

    if failures:
        raise SystemExit("Leo2 artifact validation failed:\n  " + "\n  ".join(failures))
    print(f"Leo2 artifact validation passed: files={checked}, hashed={hashed}, manifest={args.manifest}")


if __name__ == "__main__":
    main()
