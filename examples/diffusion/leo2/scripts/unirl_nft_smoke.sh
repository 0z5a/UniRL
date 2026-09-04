#!/usr/bin/env bash
# Leo2 x UniRL trainside DiffusionNFT — portable single-node launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
VENDOR_ROOT="${REPO_ROOT}/unirl/models/leo2/vendor/gen_ar"

: "${LEO2_CKPT_DIR:?Set LEO2_CKPT_DIR to a native Torch DCP weights directory}"
: "${LEO2_ASSETS_BASE:?Set LEO2_ASSETS_BASE to the hymm_ar_assets directory}"

if [[ -n "${LEO2_RUNTIME_PYTHON:-}" ]]; then
    [[ -x "${LEO2_RUNTIME_PYTHON}" ]] || {
        echo "LEO2_RUNTIME_PYTHON is not executable: ${LEO2_RUNTIME_PYTHON}" >&2
        exit 2
    }
    export PATH="$(dirname "${LEO2_RUNTIME_PYTHON}"):${PATH}"
fi

# The Taiji login shell prepends /usr/local/nvshmem, whose ABI masks the
# runtime-packaged NVSHMEM 3.7 used by DeepEP R03C03.
unset LD_LIBRARY_PATH
export ASSETS_BASE="${LEO2_ASSETS_BASE}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${VENDOR_ROOT}:${VENDOR_ROOT}/deps/hy_parallelism:${VENDOR_ROOT}/deps/IndexKits"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

exec bash "${REPO_ROOT}/examples/run_experiment_single_node.sh" \
    diffusion/leo2/leo2_t2v_nft "$@"
