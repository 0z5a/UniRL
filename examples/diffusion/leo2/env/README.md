# Leo2 relocatable runtime

This directory publishes a migration-safe snapshot of the Leo2 runtime that was
actually exercised on eight H20 GPUs. Its ABI is Python 3.12, Torch 2.7.1 and
CUDA 12.9. It includes FA2 2.7.4.post1, FA3 3.0.0b1, Tencent DeepEP
1.2.1+R03C03, NVSHMEM 3.7.0 and a non-editable UniRL wheel.

This is not the Python 3.13/Torch 2.10 environment specified by the v1.9 image.
Compiled extensions cannot be moved between those ABIs. Use this pack to
reproduce the runtime validated on the current host; rebuild all four compiled
kernel components inside v1.9 for a strict v1.9-compatible release.

## Install

Keep the archive, its `.sha256`, the installer and verifier in one directory:

```bash
LEO2_ENV_DIR="$HOME/runtime/leo2" bash install_leo2_runtime.sh
```

The installer verifies the checksum, extracts to the requested prefix, runs
`conda-unpack`, rejects external absolute links/RPATHs/provenance metadata and
runs FA2/FA3 kernels. It does not alter shell startup files. Activate it with:

```bash
export PATH="$HOME/runtime/leo2/bin:$PATH"
```

Weights and model assets are intentionally not part of this Python runtime.
Set `LEO2_CKPT_DIR` and `LEO2_ASSETS_BASE` as described in `../ENVIRONMENT.md`.
The host must provide an NVIDIA driver compatible with CUDA 12.9; DeepEP also
requires the node's IB/GPUDirect fabric configuration when used across nodes.

## DeepEP collective check

The package/import check is insufficient for DeepEP. On one eight-GPU node run:

```bash
TRMT_LOG_ENABLE=false NCCL_IB_DISABLE=1 \
  "$LEO2_ENV_DIR/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 deepep_8rank_smoke.py
```

The release test requires `DEEPEP_8RANK_ROUNDTRIP_OK` and zero maximum absolute
error for both first-dispatch and cached-handle dispatch/combine paths.

## Repack

After installing a non-editable UniRL wheel and removing all `direct_url.json`
files from the golden prefix:

```bash
LEO2_SOURCE_ENV=/path/to/golden-prefix \
LEO2_PACK_OUTPUT=/path/to/release-dir bash pack_leo2_runtime.sh
```
