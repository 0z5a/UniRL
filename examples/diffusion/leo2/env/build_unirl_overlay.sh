#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${LEO2_REPO_DIR:-$(cd "${HERE}/../../../.." && pwd)}
BUILD_PYTHON=${LEO2_BUILD_PYTHON:-python}
OUTPUT_DIR=${LEO2_PACK_OUTPUT:-${HERE}}
WHEEL_NAME=unirl-0.1.0-py3-none-any.whl

die() {
  echo "error: $*" >&2
  exit 2
}

git -C "${REPO_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "not a Git checkout: ${REPO_DIR}"
repo_status=$(git -C "${REPO_DIR}" status --porcelain) \
  || die "cannot inspect repository state: ${REPO_DIR}"
[[ -z "${repo_status}" ]] || die "repository must be clean before building the code overlay: ${REPO_DIR}"

mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
command -v flock >/dev/null 2>&1 || die "overlay build requires flock"
exec {LEO2_BUILD_LOCK_FD}<"${OUTPUT_DIR}"
flock -n "${LEO2_BUILD_LOCK_FD}" || die "another release build owns ${OUTPUT_DIR}"
for target in "${OUTPUT_DIR}/${WHEEL_NAME}" "${OUTPUT_DIR}/${WHEEL_NAME}.sha256"; do
  [[ ! -e "${target}" && ! -L "${target}" ]] || die "refusing to overwrite release file: ${target}"
done

temporary_dir=$(mktemp -d "${OUTPUT_DIR}/.leo2-wheel-build.XXXXXX")
trap 'rm -rf -- "${temporary_dir}"' EXIT
"${BUILD_PYTHON}" -m pip wheel --disable-pip-version-check --no-deps \
  --no-build-isolation --wheel-dir "${temporary_dir}" "${REPO_DIR}"
test -f "${temporary_dir}/${WHEEL_NAME}"
"${BUILD_PYTHON}" "${HERE}/compute_release_contract.py" \
  --repo "${REPO_DIR}" --wheel "${temporary_dir}/${WHEEL_NAME}"
"${BUILD_PYTHON}" - "${temporary_dir}/${WHEEL_NAME}" "${HERE}/verify_leo2_runtime.py" <<'PY'
import ast
import hashlib
import sys
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])
verifier_path = Path(sys.argv[2])
tree = ast.parse(verifier_path.read_text(), filename=str(verifier_path))
constants = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
}
def wheel_digest(wheel: zipfile.ZipFile, prefix: str) -> tuple[int, str]:
    names = sorted(
        name
        for name in wheel.namelist()
        if name.startswith(prefix)
        and not name.endswith("/")
        and "/__pycache__/" not in name
        and not name.endswith(".pyc")
    )
    if len(names) != len(set(names)):
        raise SystemExit(f"wheel contains duplicate source entries below {prefix}")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.removeprefix(prefix).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(wheel.read(name)).digest())
        digest.update(b"\0")
    return len(names), digest.hexdigest()


with zipfile.ZipFile(wheel_path) as wheel:
    checks = (
        (*wheel_digest(wheel, "unirl/"), constants["EXPECTED_UNIRL_TREE_SHA256"], "UniRL"),
        (*wheel_digest(wheel, "unirl/models/leo2/"), constants["EXPECTED_LEO2_TREE_SHA256"], "Leo2"),
    )
for count, actual, expected, label in checks:
    if actual != expected:
        raise SystemExit(
            f"wheel {label} digest is not synchronized with the verifier: files={count}, sha256={actual}"
        )
    print(f"wheel {label} contract passed: files={count}, sha256={actual}")
PY
(cd "${temporary_dir}" && sha256sum "${WHEEL_NAME}" > "${WHEEL_NAME}.sha256")
mv -nT -- "${temporary_dir}/${WHEEL_NAME}" "${OUTPUT_DIR}/${WHEEL_NAME}"
mv -nT -- "${temporary_dir}/${WHEEL_NAME}.sha256" "${OUTPUT_DIR}/${WHEEL_NAME}.sha256"
[[ ! -e "${temporary_dir}/${WHEEL_NAME}" && ! -e "${temporary_dir}/${WHEEL_NAME}.sha256" ]] \
  || die "release files appeared concurrently in ${OUTPUT_DIR}"
echo "built ${OUTPUT_DIR}/${WHEEL_NAME} from $(git -C "${REPO_DIR}" rev-parse HEAD)"
