---
name: runtime-install
description: Install and validate the relocatable Leo2 Python, CUDA, FA2, FA3, DeepEP and NVSHMEM runtime on a target host.
---

# Install the Leo2 runtime

The release directory must contain the dependency archive and checksum, current
UniRL wheel and checksum, installer, verifier, release manifest and
`SHA256SUMS`. Validate the complete directory before installing:

```bash
export LEO2_RELEASE_DIR=/path/to/packed/leo2
export LEO2_ENV_DIR=/opt/conda/envs/leo2
bash "$LEO2_RELEASE_DIR/verify_release.sh"
bash "$LEO2_RELEASE_DIR/install_leo2_runtime.sh"
export PATH="$LEO2_ENV_DIR/bin:$PATH"
```

The installer validates the runtime layer. Run the doctor only after the
heavy-assets skill has produced and sourced `leo2-assets.env`; the doctor also
binds the actual checkout and launcher used for inference to the release.

On an eight-GPU Hopper node, also validate the DeepEP collective:

```bash
TRMT_LOG_ENABLE=false NCCL_IB_DISABLE=1 \
  "$LEO2_ENV_DIR/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 "$LEO2_RELEASE_DIR/deepep_8rank_smoke.py"
```

Require `DEEPEP_8RANK_ROUNDTRIP_OK` with zero first and cached round-trip
errors. The published Python 3.12/Torch 2.7.1 pack is not ABI-compatible with a
Python 3.13/Torch 2.10 process; rebuild all compiled kernels for that ABI rather
than mixing the two environments.
