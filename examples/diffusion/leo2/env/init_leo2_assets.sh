#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARTIFACT_MANIFEST=${HERE}/artifacts.yaml
ARTIFACT_MANIFEST_SHA256=c0463cd1e0108b08fceb8e4ed2b669c80d2ff9905b431c453d123b5d64faef06

CHECKPOINT_FILES=(
  '.metadata|583131|23868c4466dffe5ff0eca5cfa3ea5697f110bfe5be0369dae56069bca672a4c5'
  '__0_0.distcp|150508043106|ac5570875348a9e3619f0a53650f88d3a829c400a38e5fe47b7a4e3907ae0c71'
)

ASSET_FILES=(
  'text_encoder/Qwen3.5-9B/config.json|3126|d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05'
  'text_encoder/Qwen3.5-9B/model.safetensors.index.json|79657|26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'
  'text_encoder/Qwen3.5-9B/tokenizer.json|12807982|5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
  'text_encoder/Qwen3.5-9B/tokenizer_config.json|16710|316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'
  'text_encoder/Qwen3.5-9B/vocab.json|6722759|ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'
  'text_encoder/Qwen3.5-9B/merges.txt|3353259|a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d'
  'text_encoder/Qwen3.5-9B/chat_template.jinja|7756|a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
  'text_encoder/Qwen3.5-9B/preprocessor_config.json|390|27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516'
  'text_encoder/Qwen3.5-9B/video_preprocessor_config.json|385|7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13'
  'text_encoder/Qwen3.5-9B/model.safetensors-00001-of-00004.safetensors|5276436216|db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e'
  'text_encoder/Qwen3.5-9B/model.safetensors-00002-of-00004.safetensors|5335161512|31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330'
  'text_encoder/Qwen3.5-9B/model.safetensors-00003-of-00004.safetensors|5368717440|7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec'
  'text_encoder/Qwen3.5-9B/model.safetensors-00004-of-00004.safetensors|3325995712|b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6'
  'image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2/config.json|464|68e8f1140ee22cbc0af4e3d45f307d01e31b264624aef19be697d63fa2b4f12b'
  'image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2/latent_norm_stats.pt|3767|5a768eb65f85f108ccd481e9d2266555faae7b9c199ef192972065e18ce5b4cc'
  'image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2/pytorch_model.pt|7706430615|6b3158d045e661c81d829542333f78a109d00547eda9c810e48ebca665af6b66'
)

die() {
  echo "error: $*" >&2
  exit 2
}

require_absolute() {
  local name=$1 value=$2
  [[ -n "${value}" ]] || die "set ${name} explicitly"
  [[ "${value}" == /* ]] || die "${name} must be an absolute path: ${value}"
}

is_nested() {
  local path=$1 base=$2
  [[ "${path}" == "${base}" || "${path}" == "${base}/"* ]]
}

verify_file() {
  local path=$1 expected_size=$2 expected_sha=$3 checksums=$4
  [[ -f "${path}" && ! -L "${path}" ]] || die "required regular file is missing or symbolic: ${path}"
  local actual_size
  actual_size=$(stat -c '%s' -- "${path}")
  [[ "${actual_size}" == "${expected_size}" ]] \
    || die "size mismatch for ${path}: expected ${expected_size}, got ${actual_size}"
  if [[ "${checksums}" == 1 ]]; then
    local actual_sha
    actual_sha=$(sha256sum -- "${path}" | awk '{print $1}')
    [[ "${actual_sha}" == "${expected_sha}" ]] \
      || die "SHA-256 mismatch for ${path}: expected ${expected_sha}, got ${actual_sha}"
  fi
}

verify_records() {
  local root=$1 checksums=$2 array_name=$3 record relative size sha resolved
  local -n records=${array_name}
  for record in "${records[@]}"; do
    IFS='|' read -r relative size sha <<<"${record}"
    resolved=$(realpath -e -- "${root}/${relative}") \
      || die "required artifact cannot be resolved: ${root}/${relative}"
    is_nested "${resolved}" "${root}" \
      || die "artifact escapes its declared source root: ${root}/${relative} -> ${resolved}"
    verify_file "${root}/${relative}" "${size}" "${sha}" "${checksums}"
  done
}

preflight_link() {
  local target=$1 source=$2 resolved
  if [[ -L "${target}" ]]; then
    resolved=$(realpath -e -- "${target}") \
      || die "existing link is broken: ${target}"
    [[ "${resolved}" == "${source}" ]] \
      || die "existing link has a different target: ${target} -> ${resolved}"
  elif [[ -e "${target}" ]]; then
    die "link target already exists and is not the requested link: ${target}"
  fi
}

preflight_copy_tree() {
  local target=$1 first_link
  [[ ! -L "${target}" ]] || die "copy target is symbolic: ${target}"
  [[ ! -e "${target}" || -d "${target}" ]] || die "copy target is not a directory: ${target}"
  if [[ -d "${target}" ]]; then
    first_link=$(find "${target}" -type l -print -quit)
    [[ -z "${first_link}" ]] || die "copy target contains a symbolic link: ${first_link}"
  fi
}

preflight_copy_records() {
  local target_root=$1 checksums=$2 array_name=$3 record relative size sha target
  local -n records=${array_name}
  for record in "${records[@]}"; do
    IFS='|' read -r relative size sha <<<"${record}"
    target=${target_root}/${relative}
    if [[ -e "${target}" || -L "${target}" ]]; then
      verify_file "${target}" "${size}" "${sha}" "${checksums}"
    fi
  done
}

copy_records() {
  local source_root=$1 target_root=$2 checksums=$3 array_name=$4
  local -n records=${array_name}
  local record relative size sha source target partial
  for record in "${records[@]}"; do
    IFS='|' read -r relative size sha <<<"${record}"
    source=${source_root}/${relative}
    target=${target_root}/${relative}
    partial=${target}.leo2-partial
    if [[ -e "${target}" || -L "${target}" ]]; then
      verify_file "${target}" "${size}" "${sha}" "${checksums}"
      continue
    fi
    if [[ -e "${partial}" || -L "${partial}" ]]; then
      [[ -f "${partial}" && ! -L "${partial}" ]] \
        || die "partial target is not a regular file: ${partial}"
    fi
    mkdir -p -- "$(dirname "${target}")"
    echo "copying ${relative}"
    rsync --archive --partial --append-verify -- "${source}" "${partial}"
    verify_file "${partial}" "${size}" "${sha}" "${checksums}"
    if [[ -e "${target}" || -L "${target}" ]]; then
      verify_file "${target}" "${size}" "${sha}" "${checksums}"
      rm -f -- "${partial}"
    else
      mv -nT -- "${partial}" "${target}"
      verify_file "${target}" "${size}" "${sha}" "${checksums}"
      if [[ -e "${partial}" || -L "${partial}" ]]; then
        rm -f -- "${partial}"
      fi
    fi
  done
}

for variable in LEO2_HEAVY_ROOT LEO2_CKPT_SOURCE LEO2_ASSETS_SOURCE; do
  require_absolute "${variable}" "${!variable:-}"
done

MODE=${LEO2_ASSET_MODE:-link}
VERIFY_CHECKSUMS=${LEO2_VERIFY_CHECKSUMS:-1}
[[ "${MODE}" == link || "${MODE}" == copy ]] || die "LEO2_ASSET_MODE must be link or copy"
[[ "${VERIFY_CHECKSUMS}" == 0 || "${VERIFY_CHECKSUMS}" == 1 ]] \
  || die "LEO2_VERIFY_CHECKSUMS must be 0 or 1"
if [[ "${VERIFY_CHECKSUMS}" == 0 ]]; then
  echo "warning: SHA-256 verification is disabled; do not publish this initialization" >&2
fi

[[ ! -L "${LEO2_HEAVY_ROOT}" ]] || die "LEO2_HEAVY_ROOT itself must not be a symbolic link"
[[ ! -e "${LEO2_HEAVY_ROOT}" || -d "${LEO2_HEAVY_ROOT}" ]] \
  || die "LEO2_HEAVY_ROOT is not a directory: ${LEO2_HEAVY_ROOT}"
[[ -d "${LEO2_CKPT_SOURCE}" ]] || die "checkpoint source is not a directory: ${LEO2_CKPT_SOURCE}"
[[ -d "${LEO2_ASSETS_SOURCE}" ]] || die "asset source is not a directory: ${LEO2_ASSETS_SOURCE}"

HEAVY_ROOT=$(realpath -m -- "${LEO2_HEAVY_ROOT}")
CKPT_SOURCE=$(realpath -e -- "${LEO2_CKPT_SOURCE}")
ASSETS_SOURCE=$(realpath -e -- "${LEO2_ASSETS_SOURCE}")
[[ "${HEAVY_ROOT}" != / ]] || die "LEO2_HEAVY_ROOT must not be /"
for source in "${CKPT_SOURCE}" "${ASSETS_SOURCE}"; do
  if is_nested "${HEAVY_ROOT}" "${source}" || is_nested "${source}" "${HEAVY_ROOT}"; then
    die "LEO2_HEAVY_ROOT and source trees must not contain one another: ${HEAVY_ROOT}, ${source}"
  fi
done

[[ -f "${CKPT_SOURCE}/.metadata" && ! -L "${CKPT_SOURCE}/.metadata" ]] \
  || die "checkpoint source lacks a regular .metadata file"
shopt -s nullglob
DISTCP_FILES=("${CKPT_SOURCE}"/*.distcp)
shopt -u nullglob
(( ${#DISTCP_FILES[@]} > 0 )) || die "checkpoint source has no .distcp shard"
(( ${#DISTCP_FILES[@]} == 1 )) \
  || die "packaged artifact profile requires exactly one .distcp shard"
[[ "$(basename -- "${DISTCP_FILES[0]}")" == __0_0.distcp ]] \
  || die "packaged artifact profile requires __0_0.distcp"
for required_dir in \
  "${ASSETS_SOURCE}/text_encoder/Qwen3.5-9B" \
  "${ASSETS_SOURCE}/image_encoder/vae_3d/hyvae_vid_leo2.0_v2.5.1_release2"; do
  [[ -d "${required_dir}" && ! -L "${required_dir}" ]] \
    || die "required asset directory is missing or symbolic: ${required_dir}"
done

[[ -f "${ARTIFACT_MANIFEST}" ]] || die "artifact manifest is missing: ${ARTIFACT_MANIFEST}"
actual_manifest_sha=$(sha256sum -- "${ARTIFACT_MANIFEST}" | awk '{print $1}')
[[ "${actual_manifest_sha}" == "${ARTIFACT_MANIFEST_SHA256}" ]] \
  || die "artifacts.yaml changed; synchronize the embedded file table before running this script"

source_checksums=${VERIFY_CHECKSUMS}
[[ "${MODE}" == link ]] || source_checksums=0
verify_records "${CKPT_SOURCE}" "${source_checksums}" CHECKPOINT_FILES
verify_records "${ASSETS_SOURCE}" "${source_checksums}" ASSET_FILES

command -v flock >/dev/null 2>&1 || die "asset initialization requires flock"
mkdir -p -- "${HEAVY_ROOT}"
exec {LEO2_LOCK_FD}<"${HEAVY_ROOT}"
flock -n "${LEO2_LOCK_FD}" || die "another asset initialization owns ${HEAVY_ROOT}"

CKPT_TARGET=${HEAVY_ROOT}/checkpoint
ASSETS_TARGET=${HEAVY_ROOT}/assets
ENV_FILE=${HEAVY_ROOT}/leo2-assets.env
ENV_TEMP=$(mktemp /tmp/leo2-assets-env.XXXXXX)
trap 'rm -f -- "${ENV_TEMP}"' EXIT
{
  echo '# Generated by init_leo2_assets.sh; source this file from Bash.'
  printf 'export LEO2_HEAVY_ROOT=%q\n' "${HEAVY_ROOT}"
  printf 'export LEO2_CKPT_DIR=%q\n' "${CKPT_TARGET}"
  printf 'export LEO2_ASSETS_BASE=%q\n' "${ASSETS_TARGET}"
} >"${ENV_TEMP}"

if [[ -L "${ENV_FILE}" || ( -e "${ENV_FILE}" && ! -f "${ENV_FILE}" ) ]]; then
  die "generated environment target is not a regular file: ${ENV_FILE}"
fi
if [[ -f "${ENV_FILE}" ]] && ! cmp -s -- "${ENV_TEMP}" "${ENV_FILE}"; then
  die "existing environment file has different content: ${ENV_FILE}"
fi

if [[ "${MODE}" == link ]]; then
  preflight_link "${CKPT_TARGET}" "${CKPT_SOURCE}"
  preflight_link "${ASSETS_TARGET}" "${ASSETS_SOURCE}"
  [[ -L "${CKPT_TARGET}" ]] || ln -sT -- "${CKPT_SOURCE}" "${CKPT_TARGET}"
  [[ -L "${ASSETS_TARGET}" ]] || ln -sT -- "${ASSETS_SOURCE}" "${ASSETS_TARGET}"
  preflight_link "${CKPT_TARGET}" "${CKPT_SOURCE}"
  preflight_link "${ASSETS_TARGET}" "${ASSETS_SOURCE}"
else
  command -v rsync >/dev/null 2>&1 || die "copy mode requires rsync"
  preflight_copy_tree "${CKPT_TARGET}"
  preflight_copy_tree "${ASSETS_TARGET}"
  preflight_copy_records "${CKPT_TARGET}" "${VERIFY_CHECKSUMS}" CHECKPOINT_FILES
  preflight_copy_records "${ASSETS_TARGET}" "${VERIFY_CHECKSUMS}" ASSET_FILES
  mkdir -p -- "${CKPT_TARGET}" "${ASSETS_TARGET}"
  copy_records "${CKPT_SOURCE}" "${CKPT_TARGET}" "${VERIFY_CHECKSUMS}" CHECKPOINT_FILES
  copy_records "${ASSETS_SOURCE}" "${ASSETS_TARGET}" "${VERIFY_CHECKSUMS}" ASSET_FILES
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  chmod 0644 "${ENV_TEMP}"
  mv -nT -- "${ENV_TEMP}" "${ENV_FILE}"
  [[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" ]] \
    || die "generated environment target changed concurrently: ${ENV_FILE}"
  if [[ -e "${ENV_TEMP}" || -L "${ENV_TEMP}" ]]; then
    cmp -s -- "${ENV_TEMP}" "${ENV_FILE}" \
      || die "generated environment target changed concurrently: ${ENV_FILE}"
  fi
fi
trap - EXIT
rm -f -- "${ENV_TEMP}"

echo "Leo2 assets initialized in ${MODE} mode at ${HEAVY_ROOT}"
echo "source ${ENV_FILE}"
