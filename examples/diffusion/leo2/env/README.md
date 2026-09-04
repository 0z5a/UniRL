# Leo2 relocatable runtime

This directory publishes a migration-safe snapshot of the Leo2 runtime that was
actually exercised on eight H20 GPUs. Its dependency ABI is Python 3.12, Torch
2.7.1 and CUDA 12.9. It includes FA2 2.7.4.post1, FA3 3.0.0b1, Tencent DeepEP
1.2.1+R03C03 and NVSHMEM 3.7.0. The installer then applies the bundled current
UniRL wheel offline and verifies the exact Leo2 source-tree digest, so the old
wheel embedded in the dependency archive cannot silently pass validation. The
complete installed UniRL tree and the checkout used by the native launcher are
also bound to the release manifest.

This is not the Python 3.13/Torch 2.10 environment specified by the v1.9 image.
Compiled extensions cannot be moved between those ABIs. Use this pack to
reproduce the runtime validated on the current host; rebuild all compiled
extensions inside v1.9 for a strict v1.9-compatible release.

## Install

Keep the archive, current UniRL wheel, both `.sha256` files, installer and
verifier in one directory:

```bash
LEO2_ENV_DIR="$HOME/runtime/leo2" bash install_leo2_runtime.sh
```

The installer verifies the checksum, extracts to the requested prefix, runs
`conda-unpack`, installs the wheel with `--no-index --no-deps`, rejects external
absolute links/RPATHs and direct-install provenance metadata, then runs FA2/FA3
kernels. It does not alter shell startup files. Activate it with:

```bash
export PATH="$HOME/runtime/leo2/bin:$PATH"
```

Weights and model assets are intentionally not part of this Python runtime.
Set `LEO2_CKPT_DIR` and `LEO2_ASSETS_BASE` as described in
[`ENVIRONMENT.md`](ENVIRONMENT.md).
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

## Migration playbooks

Reusable task guides live in [`skills/`](skills/README.md). Start with runtime
installation, then initialize the external heavyweight assets, run the doctor,
and finally launch the one-sample native inference smoke. The asset helper can
reuse an existing payload through links or copy only the manifest-required
checkpoint, Qwen3.5-9B and video-VAE files.

After assembling a new release, `bash finalize_release.sh` writes its complete
checksum inventory without overwriting an existing one. Before installation,
`bash verify_release.sh` validates the complete release directory against
`SHA256SUMS`, cross-checks the archive/wheel/manifests and checks the compressed
archive stream.

## Repack

After installing a non-editable UniRL wheel and removing all `direct_url.json`
files from the golden prefix:

```bash
LEO2_SOURCE_ENV=/path/to/golden-prefix \
LEO2_PACK_OUTPUT=/path/to/release-dir bash pack_leo2_runtime.sh
```

Build the current clean checkout as the offline code overlay separately:

```bash
LEO2_REPO_DIR=/path/to/Unirl-leo2 \
LEO2_BUILD_PYTHON=/path/to/leo2/bin/python \
LEO2_PACK_OUTPUT=/path/to/release-dir bash build_unirl_overlay.sh
```
