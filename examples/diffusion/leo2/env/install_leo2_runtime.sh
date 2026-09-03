#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARCHIVE_NAME=leo2-runtime-py312-torch271-cu129-20260903.tar.gz
SOURCE_TAR=${LEO2_RUNTIME_TAR:-${HERE}/${ARCHIVE_NAME}}
ENV_DIR=${LEO2_ENV_DIR:-${HOME}/.local/leo2-runtime-py312-torch271-cu129}
FORCE=${FORCE:-0}

if [[ ! -f "${SOURCE_TAR}" ]]; then
  echo "archive not found: ${SOURCE_TAR}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_TAR}.sha256" ]]; then
  echo "checksum not found: ${SOURCE_TAR}.sha256" >&2
  exit 2
fi
(cd "$(dirname "${SOURCE_TAR}")" && sha256sum -c "$(basename "${SOURCE_TAR}.sha256")")

if [[ -e "${ENV_DIR}" && "${FORCE}" != 1 ]]; then
  echo "target exists: ${ENV_DIR}; choose another LEO2_ENV_DIR or set FORCE=1" >&2
  exit 3
fi

parent=$(dirname "${ENV_DIR}")
mkdir -p "${parent}"
stage=${ENV_DIR}.inprogress.$$
backup=${ENV_DIR}.backup.$(date +%Y%m%d_%H%M%S).$$
cleanup() {
  [[ ! -d "${stage}" ]] || rm -rf -- "${stage}"
}
trap cleanup EXIT
mkdir "${stage}"
tar -xzf "${SOURCE_TAR}" -C "${stage}"

if [[ -e "${ENV_DIR}" ]]; then
  mv "${ENV_DIR}" "${backup}"
  echo "previous runtime preserved at ${backup}"
fi
mv "${stage}" "${ENV_DIR}"
rollback() {
  rc=$?
  if (( rc != 0 )); then
    failed=${ENV_DIR}.failed.$(date +%Y%m%d_%H%M%S).$$
    [[ ! -e "${ENV_DIR}" ]] || mv "${ENV_DIR}" "${failed}"
    [[ ! -e "${backup}" ]] || mv "${backup}" "${ENV_DIR}"
  fi
  exit "${rc}"
}
trap rollback EXIT
"${ENV_DIR}/bin/conda-unpack"
"${ENV_DIR}/bin/python" "${HERE}/verify_leo2_runtime.py" --gpu
trap - EXIT
echo "Leo2 runtime installed at ${ENV_DIR}"
