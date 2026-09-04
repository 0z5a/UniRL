#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${LEO2_SOURCE_ENV:?set LEO2_SOURCE_ENV to the verified conda prefix}"
OUTPUT_DIR=${LEO2_PACK_OUTPUT:-${HERE}}
ARCHIVE_NAME=leo2-runtime-py312-torch271-cu129-20260903.tar.gz
SOURCE_ENV=$(cd "${LEO2_SOURCE_ENV}" && pwd)
REQUIREMENTS_NAME=requirements-20260903.txt
mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
command -v flock >/dev/null 2>&1 || { echo "packing requires flock" >&2; exit 2; }
exec {LEO2_PACK_LOCK_FD}<"${OUTPUT_DIR}"
flock -n "${LEO2_PACK_LOCK_FD}" || { echo "another release build owns ${OUTPUT_DIR}" >&2; exit 2; }
for target in \
  "${OUTPUT_DIR}/${ARCHIVE_NAME}" \
  "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256" \
  "${OUTPUT_DIR}/${REQUIREMENTS_NAME}"; do
  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "refusing to overwrite release file: ${target}" >&2
    exit 2
  fi
done

"${SOURCE_ENV}/bin/python" "${HERE}/verify_leo2_runtime.py" --gpu
temporary_dir=$(mktemp -d "${OUTPUT_DIR}/.leo2-runtime-pack.XXXXXX")
temporary=${temporary_dir}/${ARCHIVE_NAME}
trap 'rm -rf -- "${temporary_dir}"' EXIT
"${SOURCE_ENV}/bin/conda-pack" -p "${SOURCE_ENV}" -o "${temporary}"
gzip -t "${temporary}"
(
  cd "${temporary_dir}"
  sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
)
"${SOURCE_ENV}/bin/python" -m pip list --format=freeze \
  | LC_ALL=C sort > "${temporary_dir}/${REQUIREMENTS_NAME}"
mv -nT -- "${temporary}" "${OUTPUT_DIR}/${ARCHIVE_NAME}"
mv -nT -- "${temporary}.sha256" "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
mv -nT -- "${temporary_dir}/${REQUIREMENTS_NAME}" "${OUTPUT_DIR}/${REQUIREMENTS_NAME}"
for temporary_target in \
  "${temporary}" \
  "${temporary}.sha256" \
  "${temporary_dir}/${REQUIREMENTS_NAME}"; do
  [[ ! -e "${temporary_target}" ]] || { echo "release files appeared concurrently" >&2; exit 2; }
done
trap - EXIT
rm -rf -- "${temporary_dir}"
echo "packed ${OUTPUT_DIR}/${ARCHIVE_NAME}"
