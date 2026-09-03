# Leo2 portable environment contract

The repository contains the Leo2 Python runtime, 480p stage-3 model YAML,
generation configuration and prompt sets. Model weights and model assets are
external artifacts because the validated set is about 178 GB.

## Base runtime

Use this image as the ABI boundary:

```text
mirrors.tencent.com/multimodal_gen_ar/tlinux3.2-cuda12.9.1-cudnn9-py3.13-torch2.10.0:v1.9
```

Install the Python dependencies from the repository's single authoritative
metadata source:

```bash
python -m pip install -e '.[leo2,train]'
```

The `leo2` extra pins Torch 2.10.0, torchvision 0.25.0, torchaudio 2.10.0,
Transformers 5.6.0 and Flash Linear Attention 0.5.1. `setup.py` is only a
compatibility shim over `pyproject.toml`; it does not define a second dependency
set.

## Compiled kernel overlays

The following components are required for the 8-GPU Hopper path but are not
installed by the `leo2` extra. They must be built against the exact Python,
Torch, CUDA and CXX11 ABI of the image above:

| Module | Validated source version | Requirement |
|---|---|---|
| `flash_attn` | FlashAttention 2.7.4.post1 | FA2 Python API and padding helpers |
| `flash_attn_interface` | FlashAttention 3.0.0b1 | Hopper `sm90a` packed attention |
| `deep_ep` | Tencent DeepEP 1.2.1+R03C03 | MoE dispatch; includes the R03C03 NVL-combine patch |
| NVSHMEM | 3.7.0 CUDA 12 | DeepEP runtime |

Do not reuse the existing Torch-2.7/Python-3.12 binary pack in a Torch-2.10
process. Rebuild its FA2, FA3, DeepEP and FLA kernels for this image; otherwise
imports can succeed and later fail with undefined Torch symbols. FA3 is commonly
installed by placing `flash_attn_interface.py` and its extension beside FA2;
ensure valid distribution metadata remains visible to Transformers 5.6.0.

At minimum, verify the overlay before loading the 75B model:

```bash
python - <<'PY'
import torch
import transformers
import deep_ep
import flash_attn
import flash_attn_interface
import fla
from transformers.utils import import_utils

from unirl.models.transformers_compat import install_transformers_flash_attention_compat

assert torch.__version__.split('+', 1)[0] == '2.10.0'
assert transformers.__version__ == '5.6.0'
install_transformers_flash_attention_compat()
assert 'flash-attn-3' in import_utils.PACKAGE_DISTRIBUTION_MAPPING['flash_attn_interface']
assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
print('Leo2 runtime imports OK')
PY
```

An import check does not validate DeepEP collectives. Run the repository's
DeepEP multi-rank smoke on the target node before an EP job.

## External artifact contract

Set both variables explicitly; there are no user-specific fallback paths:

```bash
export LEO2_CKPT_DIR=/path/to/iter_XXXXXXX_torch/weights
export LEO2_ASSETS_BASE=/path/to/hymm_ar_assets
export LEO2_RUNTIME_PYTHON=/path/to/torch-2.10/bin/python  # optional when python is already correct
"${LEO2_RUNTIME_PYTHON:-python}" -m unirl.models.leo2.verify_artifacts --require-checksums
```

`LEO2_CKPT_DIR` must be a native Torch DCP directory containing `.metadata`
and one or more `*.distcp` shards. `LEO2_ASSETS_BASE` must contain the local
Qwen3.5-9B and release-2 video VAE trees listed in the packaged
`resources/artifacts.yaml`. The checked-in manifest describes the validated
iter-0063300 snapshot, including byte sizes and SHA-256 values. For another
compatible checkpoint, copy the manifest, update its profile/files/checksums
and pass it with `--manifest`.

The bundled manifest pins every required payload by both exact size and SHA-256.
Use `--require-checksums` for a complete integrity check before inference. If a
different checkpoint or asset publication is used, copy the manifest, update
its profile/files/checksums, and pass that file with `--manifest`.

## Launch

From the repository root:

```bash
bash examples/diffusion/leo2/scripts/unirl_smoke.sh \
  sampling.height=464 sampling.width=848 sampling.num_frames=121
```

The recipe defaults to the checked-in train/test prompt files. Override them
with `LEO2_PROMPT_FILE` and `LEO2_EVAL_PROMPT_FILE` only when needed.
