#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${LEO2_SOURCE_ENV:?set LEO2_SOURCE_ENV to the verified conda prefix}"
OUTPUT_DIR=${LEO2_PACK_OUTPUT:-${HERE}}
ARCHIVE_NAME=leo2-runtime-py312-torch271-cu129-20260903.tar.gz
SOURCE_ENV=$(cd "${LEO2_SOURCE_ENV}" && pwd)
mkdir -p "${OUTPUT_DIR}"

"${SOURCE_ENV}/bin/python" "${HERE}/verify_leo2_runtime.py" --gpu
temporary_dir=$(mktemp -d /tmp/leo2-runtime-pack.XXXXXX)
temporary=${temporary_dir}/${ARCHIVE_NAME}
trap 'rm -rf -- "${temporary_dir}"' EXIT
"${SOURCE_ENV}/bin/conda-pack" -p "${SOURCE_ENV}" -o "${temporary}"
gzip -t "${temporary}"
mv "${temporary}" "${OUTPUT_DIR}/${ARCHIVE_NAME}"
(cd "${OUTPUT_DIR}" && sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256")
"${SOURCE_ENV}/bin/python" -m pip list --format=freeze \
  | LC_ALL=C sort > "${OUTPUT_DIR}/requirements-20260903.txt"
trap - EXIT
rm -rf -- "${temporary_dir}"
echo "packed ${OUTPUT_DIR}/${ARCHIVE_NAME}"
