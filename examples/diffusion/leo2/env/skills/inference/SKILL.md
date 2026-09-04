---
name: inference
description: Run a verified Leo2 native video inference smoke or full 848x464, 121-frame generation after migration.
---

# Run Leo2 inference

Activate the runtime, load the generated asset environment, then change to the
UniRL repository root before model construction:

```bash
export LEO2_ENV_DIR=/path/to/leo2-runtime
export LEO2_RELEASE_DIR=/path/to/packed/leo2
export PATH="$LEO2_ENV_DIR/bin:$PATH"
export LEO2_RUNTIME_PYTHON="$LEO2_ENV_DIR/bin/python"
source /path/to/leo2-heavy/leo2-assets.env
cd /path/to/Unirl-leo2
python -m unirl.models.leo2.verify_artifacts --require-checksums
LEO2_REPO_DIR="$PWD" LEO2_VERIFY_ASSET_CHECKSUMS=0 \
  bash "$LEO2_RELEASE_DIR/doctor_leo2.sh"
```

The doctor uses the layout-only override above because the immediately
preceding command already hashed every artifact; without that override it
performs the complete artifact checksum pass itself.

First run a one-step load, forward and decode smoke:

```bash
LEO2_INFER_STEPS=1 \
LEO2_OUTPUT_DIR=outputs/leo2/migration-smoke \
  bash examples/diffusion/leo2/scripts/native_t2v.sh
```

Then run the validated 50-step 848x464, 121-frame path:

```bash
LEO2_INFER_STEPS=50 \
LEO2_OUTPUT_DIR=outputs/leo2/migration-full \
  bash examples/diffusion/leo2/scripts/native_t2v.sh
```

Do not skip the artifact checksum pass: a successful import does not validate
the 150 GB DCP shard, Qwen weights or VAE payload.
