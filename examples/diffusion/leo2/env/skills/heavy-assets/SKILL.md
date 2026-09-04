---
name: heavy-assets
description: Initialize the external Leo2 checkpoint, Qwen3.5-9B and video VAE assets safely by link or resumable manifest-limited copy.
---

# Initialize Leo2 heavy assets

Set all three paths explicitly. Start from the checked-in historical-source
example, then choose a managed target outside both source trees.

```bash
export LEO2_RELEASE_DIR=/path/to/packed/leo2
source "$LEO2_RELEASE_DIR/heavy-assets.env.example"
export LEO2_HEAVY_ROOT=/path/to/managed/leo2-heavy
bash "$LEO2_RELEASE_DIR/init_leo2_assets.sh"
source "$LEO2_HEAVY_ROOT/leo2-assets.env"
/path/to/leo2-runtime/bin/python -m unirl.models.leo2.verify_artifacts --require-checksums
```

The default `link` mode is fast and reuses the sources. For an independent
copy, set `LEO2_ASSET_MODE=copy`; only files pinned by
`unirl/models/leo2/resources/artifacts.yaml` are transferred and interrupted
files resume from `.leo2-partial` files.

The initializer never replaces a different link or file. Do not set
`LEO2_VERIFY_CHECKSUMS=0` for a release initialization; that override exists
only for structural tests with synthetic files. It uses `flock` to serialize
initialization of one target root; copy mode additionally requires `rsync`.
The helper is deliberately locked to the adjacent `iter0063300`
`artifacts.yaml`; publish a synchronized manifest and initializer for another
checkpoint profile instead of overriding only one of them.
