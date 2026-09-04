#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERIFY_PYTHON=${LEO2_VERIFY_PYTHON:-python3}
"${VERIFY_PYTHON}" - "${HERE}" <<'PY'
import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
checksum_path = root / "SHA256SUMS"
if not checksum_path.is_file():
    raise SystemExit(f"missing release checksum manifest: {checksum_path}")

links = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
if links:
    raise SystemExit("release directory contains symbolic links:\n  " + "\n  ".join(links))

record_pattern = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")
records: dict[str, str] = {}
for line_number, line in enumerate(checksum_path.read_text().splitlines(), 1):
    match = record_pattern.fullmatch(line)
    if match is None:
        raise SystemExit(f"invalid SHA256SUMS record at line {line_number}")
    digest, _, name = match.groups()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or name in {"", ".", "SHA256SUMS"}:
        raise SystemExit(f"unsafe SHA256SUMS target at line {line_number}: {name!r}")
    if name in records:
        raise SystemExit(f"duplicate SHA256SUMS target: {name}")
    records[name] = digest

actual_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != checksum_path
}
manifest_files = set(records)
if actual_files != manifest_files:
    missing = sorted(actual_files - manifest_files)
    stale = sorted(manifest_files - actual_files)
    raise SystemExit(f"SHA256SUMS coverage mismatch: unlisted={missing}, missing={stale}")

required = {
    "ENVIRONMENT.md",
    "MIGRATION_AUDIT.md",
    "README.md",
    "artifacts.yaml",
    "build_unirl_overlay.sh",
    "compute_release_contract.py",
    "deepep_8rank_smoke.py",
    "doctor_leo2.sh",
    "finalize_release.sh",
    "heavy-assets.env.example",
    "init_leo2_assets.sh",
    "install_leo2_runtime.sh",
    "leo2-runtime-py312-torch271-cu129-20260903.tar.gz",
    "leo2-runtime-py312-torch271-cu129-20260903.tar.gz.sha256",
    "pack_leo2_runtime.sh",
    "release-manifest-20260904.json",
    "requirements-20260903.txt",
    "runtime-manifest-20260903.json",
    "skills/README.md",
    "skills/heavy-assets/SKILL.md",
    "skills/inference/SKILL.md",
    "skills/release-build/SKILL.md",
    "skills/runtime-install/SKILL.md",
    "skills/troubleshooting/SKILL.md",
    "unirl-0.1.0-py3-none-any.whl",
    "unirl-0.1.0-py3-none-any.whl.sha256",
    "verify_leo2_runtime.py",
    "verify_release.sh",
}
if not required <= actual_files:
    raise SystemExit(f"release is missing required files: {sorted(required - actual_files)}")


def file_sha256(path: Path) -> str:
    """Hash one release file without loading large archives into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for name, expected in sorted(records.items()):
    digest = file_sha256(root / name)
    if digest != expected:
        raise SystemExit(f"SHA-256 mismatch for {name}: expected {expected}, got {digest}")

release = json.loads((root / "release-manifest-20260904.json").read_text())
if release.get("schema_version") != 1:
    raise SystemExit("unsupported release manifest schema")
archive = release["dependency_archive"]
overlay = release["code_overlay"]
if overlay.get("git_dirty_at_build") is not False or re.fullmatch(r"[0-9a-f]{40}", overlay.get("git_head", "")) is None:
    raise SystemExit("code overlay has invalid Git provenance")
for key in (
    "installed_unirl_tree_sha256",
    "checkout_unirl_tree_sha256",
    "leo2_tree_sha256",
    "checkout_leo2_tree_sha256",
    "native_launcher_sha256",
    "native_smoke_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", overlay.get(key, "")) is None:
        raise SystemExit(f"invalid code-overlay digest: {key}")
for key in (
    "installed_unirl_source_files",
    "checkout_unirl_source_files",
    "leo2_source_files",
    "checkout_leo2_source_files",
):
    if not isinstance(overlay.get(key), int) or overlay[key] <= 0:
        raise SystemExit(f"invalid code-overlay file count: {key}")
for section in (archive, overlay):
    payload = root / section["name"]
    if payload.stat().st_size != section["bytes"] or records[section["name"]] != section["sha256"]:
        raise SystemExit(f"release manifest does not match {section['name']}")
    sidecar = root / f"{section['name']}.sha256"
    sidecar_match = record_pattern.fullmatch(sidecar.read_text().rstrip("\n"))
    if sidecar_match is None or sidecar_match.group(1) != section["sha256"] or sidecar_match.group(3) != section["name"]:
        raise SystemExit(f"sidecar is not bound to {section['name']}")

artifact_relative = Path(release["artifact_profile"]["manifest"])
if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
    raise SystemExit("artifact manifest path is unsafe")
artifact_path = root / artifact_relative
artifact_digest = file_sha256(artifact_path)
if artifact_digest != release["artifact_profile"]["sha256"]:
    raise SystemExit("artifact manifest digest is stale")
artifact_text = artifact_path.read_text()
profile_match = re.search(r"^profile: (\S+)$", artifact_text, re.MULTILINE)
if profile_match is None or profile_match.group(1) != release["artifact_profile"]["name"]:
    raise SystemExit("artifact profile name is stale")
yaml_records = re.findall(
    r"^      - path: (.+)\n        size: ([0-9]+)\n        sha256: ([0-9a-f]{64})$",
    artifact_text,
    re.MULTILINE,
)
initializer = (root / "init_leo2_assets.sh").read_text()
initializer_match = re.search(r"^ARTIFACT_MANIFEST_SHA256=([0-9a-f]{64})$", initializer, re.MULTILINE)
if initializer_match is None or initializer_match.group(1) != artifact_digest:
    raise SystemExit("artifact initializer is not synchronized with artifacts.yaml")
initializer_records = re.findall(r"^  '([^'|]+)\|([0-9]+)\|([0-9a-f]{64})'$", initializer, re.MULTILINE)
if not yaml_records or initializer_records != yaml_records:
    raise SystemExit("artifact initializer file table does not match artifacts.yaml")

historical_relative = Path(archive["historical_manifest"])
if historical_relative.is_absolute() or ".." in historical_relative.parts:
    raise SystemExit("historical runtime manifest path is unsafe")
historical_path = root / historical_relative
historical = json.loads(historical_path.read_text())
for key in ("name", "bytes", "sha256"):
    if historical["archive"][key] != archive[key]:
        raise SystemExit(f"historical runtime manifest disagrees on archive {key}")
if historical["compatibility"] != release["compatibility"]:
    raise SystemExit("historical runtime and release compatibility fields disagree")

verifier_path = root / "verify_leo2_runtime.py"
verifier_tree = ast.parse(verifier_path.read_text(), filename=str(verifier_path))
verifier_constants = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in verifier_tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
}
expected_versions = verifier_constants["EXPECTED"]
compatibility_contract = {
    "python": expected_versions["python"],
    "torch": expected_versions["torch"],
    "cuda_runtime": expected_versions["cuda"],
}
for key, value in compatibility_contract.items():
    if release["compatibility"].get(key) != value:
        raise SystemExit(f"release compatibility does not match verifier: {key}")
if verifier_constants["EXPECTED_UNIRL_TREE_SHA256"] != overlay["installed_unirl_tree_sha256"]:
    raise SystemExit("verifier and release manifest disagree on the UniRL source digest")
if verifier_constants["EXPECTED_LEO2_TREE_SHA256"] != overlay["leo2_tree_sha256"]:
    raise SystemExit("verifier and release manifest disagree on the Leo2 source digest")


def wheel_tree_digest(wheel: zipfile.ZipFile, prefix: str) -> tuple[int, str]:
    """Hash one wheel source subtree with the runtime verifier's algorithm."""
    names = [
        name
        for name in wheel.namelist()
        if name.startswith(prefix)
        and not name.endswith("/")
        and "/__pycache__/" not in name
        and not name.endswith(".pyc")
    ]
    if len(names) != len(set(names)):
        raise SystemExit(f"wheel contains duplicate entries below {prefix}")
    digest = hashlib.sha256()
    for name in sorted(names):
        relative = name.removeprefix(prefix)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(wheel.read(name)).digest())
        digest.update(b"\0")
    return len(names), digest.hexdigest()


with zipfile.ZipFile(root / overlay["name"]) as wheel:
    wheel_contract = {
        "installed_unirl_source_files": wheel_tree_digest(wheel, "unirl/")[0],
        "installed_unirl_tree_sha256": wheel_tree_digest(wheel, "unirl/")[1],
        "leo2_source_files": wheel_tree_digest(wheel, "unirl/models/leo2/")[0],
        "leo2_tree_sha256": wheel_tree_digest(wheel, "unirl/models/leo2/")[1],
    }
for key, value in wheel_contract.items():
    if value != overlay[key]:
        raise SystemExit(f"wheel {key} mismatch: expected {overlay[key]!r}, got {value!r}")

for key in ("native_launcher_path", "native_smoke_path"):
    checkout_path = Path(overlay[key])
    if checkout_path.is_absolute() or ".." in checkout_path.parts:
        raise SystemExit(f"checkout path in release manifest is unsafe: {key}")
print(f"Leo2 release contract passed: files={len(actual_files)}, wheel={overlay['name']}")
PY

gzip -t "${HERE}/leo2-runtime-py312-torch271-cu129-20260903.tar.gz"
echo "Leo2 release files, manifests, wheel and runtime archive passed"
