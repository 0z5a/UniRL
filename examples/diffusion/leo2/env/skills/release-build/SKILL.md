---
name: release-build
description: Rebuild and publish the Leo2 dependency archive, current offline UniRL overlay, manifests and mirrored release directories.
---

# Build a Leo2 environment release

Treat the compiled dependency archive and UniRL source wheel as separate,
versioned layers. Never overwrite the last validated archive in place. Both
build helpers require an empty destination for their outputs and fail on a
concurrent builder.

```bash
export LEO2_REPO_DIR=/path/to/clean/Unirl-leo2
export LEO2_TOOL_DIR="$LEO2_REPO_DIR/examples/diffusion/leo2/env"
export LEO2_BUILD_PYTHON=/path/to/verified-golden-prefix/bin/python
LEO2_PACK_OUTPUT=/path/to/new-release \
  bash "$LEO2_TOOL_DIR/build_unirl_overlay.sh"
```

Always use helpers from the checkout being published, not from an older
release. The builder prints the complete checkout/wheel contract before
checking the verifier constants. On a code update its first run may therefore
print the new values and intentionally fail; synchronize both verifier digest
constants, discard the empty destination and rerun. Then capture the final
manifest fields explicitly:

```bash
"$LEO2_BUILD_PYTHON" "$LEO2_TOOL_DIR/compute_release_contract.py" \
  --repo "$LEO2_REPO_DIR" \
  --wheel /path/to/new-release/unirl-0.1.0-py3-none-any.whl
git -C "$LEO2_REPO_DIR" rev-parse HEAD
stat -c '%s' /path/to/new-release/unirl-0.1.0-py3-none-any.whl
sha256sum /path/to/new-release/unirl-0.1.0-py3-none-any.whl
```

Synchronize every printed field, wheel provenance and hash into the release
manifest. If the compiled dependency prefix did not change, reuse the exact
validated archive, sidecar, dependency runtime manifest and requirements
snapshot. If it did change, install the final wheel into the verified golden
prefix first, then pack it into an empty destination:

```bash
LEO2_SOURCE_ENV=/path/to/verified-golden-prefix \
LEO2_PACK_OUTPUT=/path/to/new-release \
  bash "$LEO2_TOOL_DIR/pack_leo2_runtime.sh"
```

Copy both build helpers, the contract calculator, release finalizer, installer,
runtime verifier, doctor, DeepEP smoke, asset initializer, asset example,
artifact manifest, release verifier, README, environment/audit guides and the
complete `skills/` directory into the release. Preserve the archive/wheel
sidecars and dependency runtime manifest. Update one release
manifest with archive/wheel bytes and hashes, complete UniRL and Leo2 tree
digests, native-launcher digest and artifact-manifest digest. The overlay
builder rejects source digests that are not synchronized with the runtime
verifier, while `verify_release.sh` rejects stale release-manifest values or an
incomplete `SHA256SUMS`.

Install into a fresh prefix, run the doctor, run the eight-rank DeepEP smoke,
and only then mirror the exact release bytes to the second storage site. Verify
every mirrored hash; do not use symlinks across storage sites.

With no pre-existing checksum manifest, finalize and verify the candidate:

```bash
LEO2_PACK_OUTPUT=/path/to/new-release bash "$LEO2_TOOL_DIR/finalize_release.sh"
bash /path/to/new-release/verify_release.sh
```

`SHA256SUMS` must contain all release files except itself. Generate it only as
the final publishing step, then run `verify_release.sh` again at each mirror.
