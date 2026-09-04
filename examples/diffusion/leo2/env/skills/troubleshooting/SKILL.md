---
name: troubleshooting
description: Diagnose Leo2 runtime relocation, checkpoint, model asset, FlashAttention, DeepEP and NVSHMEM failures without replacing data.
---

# Troubleshoot Leo2 migration

Run checks in this order and stop at the first failing layer:

Set `LEO2_RELEASE_DIR=/path/to/packed/leo2` before using the packaged helpers.

1. Confirm the generated paths are active:

   ```bash
   source /path/to/leo2-heavy/leo2-assets.env
   printf '%s\n' "$LEO2_CKPT_DIR" "$LEO2_ASSETS_BASE"
   ```

2. Verify every pinned artifact:

   ```bash
   python -m unirl.models.leo2.verify_artifacts --require-checksums
   ```

3. Verify the relocated runtime and GPU kernels:

   ```bash
   /path/to/leo2-runtime/bin/python \
     "$LEO2_RELEASE_DIR/verify_leo2_runtime.py" --gpu
   ```

4. With both asset variables set, bind the runtime, complete checkout, native
   launcher and heavyweight payload to the release contract:

   ```bash
   LEO2_ENV_DIR=/path/to/leo2-runtime \
   LEO2_REPO_DIR=/path/to/Unirl-leo2 \
     bash "$LEO2_RELEASE_DIR/doctor_leo2.sh"
   ```

5. If DeepEP import passes but inference hangs, run the eight-rank smoke from
   [`runtime-install`](../runtime-install/SKILL.md). Import success alone does
   not test NVSHMEM dispatch/combine or the node's IB/GPUDirect setup.

Undefined Torch symbols indicate a Python/Torch/CUDA ABI mismatch. Missing
`libnvshmem_host.so.3` indicates an incomplete runtime extraction or invalid
RPATH. If asset initialization reports a conflict, inspect the named target;
the initializer deliberately does not overwrite or repair it automatically.
