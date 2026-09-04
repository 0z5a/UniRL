# Leo2 inference runtime contract

## Supported release

The packaged runtime is the environment that passed native Leo2 inference and
kernel checks on eight H20 GPUs:

| Component | Pinned value |
|---|---|
| Python | 3.12.12 |
| Torch / CUDA runtime | 2.7.1 / 12.9 |
| Transformers / Diffusers | 5.6.0 / 0.38.0 |
| Flash Linear Attention / causal-conv1d | 0.5.1 / 1.7.0 |
| FlashAttention 2 / 3 | 2.7.4.post1 / 3.0.0b1 |
| DeepEP | 1.2.1+R03C03 |
| NVSHMEM | 3.7.0 |
| GPU architecture | Hopper, compute capability 9.x |

This is an inference runtime, not the complete `.[leo2,train]` environment.
Some UniRL training extras are intentionally absent, so a global `pip check`
is not its acceptance criterion. The release verifier instead checks the
versions and imports used by native Leo2 inference, exact installed source
digest, relocation safety, FA2/FA3 kernel parity and optionally the eight-rank
DeepEP collective. FLA and causal-conv1d are import/version checked here; their
actual model path was covered by the recorded end-to-end inference runs.

The release is not ABI-compatible with the documented v1.9 image's Python 3.13
and Torch 2.10 stack. A strict v1.9 runtime requires rebuilding every compiled
extension, including FLA, causal-conv1d, FA2, FA3, DeepEP and NVSHMEM, against
that image. Do not mix extensions between the two stacks even when imports
appear to succeed.

## Release layers

1. `leo2-runtime-py312-torch271-cu129-20260903.tar.gz` contains the relocatable
   dependency prefix and compiled kernels.
2. `unirl-0.1.0-py3-none-any.whl` overlays the current, pinned Leo2 source.
3. Checkpoint, Qwen3.5-9B and video-VAE payloads stay external and are selected
   explicitly with `LEO2_CKPT_DIR` and `LEO2_ASSETS_BASE`.

The installer verifies both layer checksums, applies the wheel without network
access, removes local installation provenance and verifies the complete
installed UniRL tree. `runtime-manifest-20260903.json` remains the historical
dependency-prefix record; `release-manifest-20260904.json` binds it to the
current code overlay and the native launcher used from the checkout.

## Host requirements

- Linux x86_64 with a driver capable of running CUDA 12.9 user-space libraries.
- Eight Hopper GPUs for the released topology and collective smoke.
- Sufficient local space for the unpacked runtime (about 6.2 GiB) and either
  links to, or local copies of, the heavyweight artifacts.
- Standard GNU userland with `flock`; resumable asset copy mode also requires
  `rsync`.
- NCCL/IB/GPUDirect configuration appropriate to the target cluster. The
  one-node acceptance command uses `NCCL_IB_DISABLE=1` to isolate NVLink/PCIe
  dispatch from site-specific IB configuration.

Follow [`skills/README.md`](skills/README.md) in order for installation, asset
initialization, diagnostics and inference.
