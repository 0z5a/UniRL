#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${LEO2_PACK_OUTPUT:-${HERE}}
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
CHECKSUM_PATH=${OUTPUT_DIR}/SHA256SUMS

[[ ! -e "${CHECKSUM_PATH}" && ! -L "${CHECKSUM_PATH}" ]] \
  || { echo "refusing to overwrite release checksum manifest: ${CHECKSUM_PATH}" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "release finalization requires flock" >&2; exit 2; }
exec {LEO2_FINALIZE_LOCK_FD}<"${OUTPUT_DIR}"
flock -n "${LEO2_FINALIZE_LOCK_FD}" || { echo "another release build owns ${OUTPUT_DIR}" >&2; exit 2; }

checksum_temp=$(mktemp "${OUTPUT_DIR}.SHA256SUMS.XXXXXX")
trap 'rm -f -- "${checksum_temp}"' EXIT
(
  cd "${OUTPUT_DIR}"
  find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${checksum_temp}"
mv -nT -- "${checksum_temp}" "${CHECKSUM_PATH}"
[[ ! -e "${checksum_temp}" ]] || { echo "checksum manifest appeared concurrently" >&2; exit 2; }
trap - EXIT
echo "wrote ${CHECKSUM_PATH}"
