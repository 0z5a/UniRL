#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCHIVE_NAME=leo2-runtime-py312-torch271-cu129-20260903.tar.gz
WHEEL_NAME=unirl-0.1.0-py3-none-any.whl
SOURCE_TAR=${LEO2_RUNTIME_TAR:-${HERE}/${ARCHIVE_NAME}}
SOURCE_WHEEL=${LEO2_UNIRL_WHEEL:-${HERE}/${WHEEL_NAME}}
ENV_DIR=${LEO2_ENV_DIR:-${HOME}/.local/leo2-runtime-py312-torch271-cu129}
FORCE=${FORCE:-0}
VERIFY_GPU=${LEO2_VERIFY_GPU:-1}

verify_sidecar() {
  local payload=$1 sidecar=${1}.sha256 expected referenced extra actual
  [[ -f "${sidecar}" ]] || { echo "checksum not found: ${sidecar}" >&2; return 2; }
  [[ "$(wc -l < "${sidecar}")" == 1 ]] \
    || { echo "checksum must contain exactly one record: ${sidecar}" >&2; return 2; }
  read -r expected referenced extra < "${sidecar}"
  [[ "${expected}" =~ ^[0-9a-f]{64}$ && -z "${extra:-}" ]] \
    || { echo "invalid checksum record: ${sidecar}" >&2; return 2; }
  referenced=${referenced#\*}
  [[ "${referenced}" == "$(basename -- "${payload}")" ]] \
    || { echo "checksum does not name $(basename -- "${payload}"): ${sidecar}" >&2; return 2; }
  actual=$(sha256sum -- "${payload}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] \
    || { echo "checksum mismatch for ${payload}" >&2; return 2; }
  echo "$(basename -- "${payload}"): OK"
}

if [[ ! -f "${SOURCE_TAR}" ]]; then
  echo "archive not found: ${SOURCE_TAR}" >&2
  exit 2
fi
verify_sidecar "${SOURCE_TAR}"
if [[ ! -f "${SOURCE_WHEEL}" || ! -f "${SOURCE_WHEEL}.sha256" ]]; then
  echo "UniRL wheel or checksum not found: ${SOURCE_WHEEL}" >&2
  exit 2
fi
verify_sidecar "${SOURCE_WHEEL}"

parent=$(dirname "${ENV_DIR}")
mkdir -p "${parent}"
command -v flock >/dev/null 2>&1 || { echo "runtime installation requires flock" >&2; exit 2; }
exec {LEO2_INSTALL_LOCK_FD}<"${parent}"
flock -n "${LEO2_INSTALL_LOCK_FD}" || { echo "another runtime installation owns ${parent}" >&2; exit 2; }
if [[ ( -e "${ENV_DIR}" || -L "${ENV_DIR}" ) && "${FORCE}" != 1 ]]; then
  echo "target exists: ${ENV_DIR}; choose another LEO2_ENV_DIR or set FORCE=1" >&2
  exit 3
fi

stage=${ENV_DIR}.inprogress.$$
backup=${ENV_DIR}.backup.$(date +%Y%m%d_%H%M%S).$$
cleanup() {
  [[ ! -d "${stage}" ]] || rm -rf -- "${stage}"
}
trap cleanup EXIT
mkdir "${stage}"
tar -xzf "${SOURCE_TAR}" -C "${stage}"

if [[ -e "${ENV_DIR}" || -L "${ENV_DIR}" ]]; then
  mv -T -- "${ENV_DIR}" "${backup}"
  echo "previous runtime preserved at ${backup}"
fi
mv -T -- "${stage}" "${ENV_DIR}"
rollback() {
  rc=$?
  if (( rc != 0 )); then
    failed=${ENV_DIR}.failed.$(date +%Y%m%d_%H%M%S).$$
    [[ ! -e "${ENV_DIR}" && ! -L "${ENV_DIR}" ]] || mv -T -- "${ENV_DIR}" "${failed}"
    [[ ! -e "${backup}" && ! -L "${backup}" ]] || mv -T -- "${backup}" "${ENV_DIR}"
  fi
  exit "${rc}"
}
trap rollback EXIT
"${ENV_DIR}/bin/conda-unpack"
"${ENV_DIR}/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --no-deps --no-index --force-reinstall "${SOURCE_WHEEL}"
for direct_url in "${ENV_DIR}"/lib/python*/site-packages/unirl-*.dist-info/direct_url.json; do
  [[ ! -f "${direct_url}" ]] || rm -- "${direct_url}"
done
verify_args=()
[[ "${VERIFY_GPU}" == 0 ]] || verify_args+=(--gpu)
"${ENV_DIR}/bin/python" "${HERE}/verify_leo2_runtime.py" "${verify_args[@]}"
trap - EXIT
echo "Leo2 runtime installed at ${ENV_DIR}"
