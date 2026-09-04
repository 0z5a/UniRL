# Leo2 migration readiness audit — 2026-09-04

## Verdict

| Unit | Status | Boundary |
|---|---|---|
| Leo2 inference source in UniRL | PASS | Native entry, active resources and dependency source are vendored; active paths are package-resolved. |
| Entire UniRL repository | PARTIAL | The unrelated optional `verl_omni` speed benchmark remains an uninitialized Git submodule. Leo2 does not import it. |
| Py3.12/Torch2.7 inference runtime | PASS | Fresh install, second relocation, FA2/FA3 and 8-rank DeepEP passed on 8×H20. |
| Current UniRL code in runtime | PASS | Offline wheel from `bbd4f84` is installed and its 616-file Leo2 digest is checked. |
| Heavy-artifact workflow | PASS for iter0063300 | Link/copy initializer is fail-closed and pinned to `artifacts.yaml`. |
| Py3.13/Torch2.10 v1.9 ABI | NOT BUILT | Compiled extensions from this release must not be loaded into that image ABI. |

The environment is ready for the validated native Leo2 inference path when the
runtime, current repository and `iter0063300` artifact profile are used
together. It is not a full UniRL training environment: several training-only
declared packages are absent, so an unqualified `pip check` is expected to
report missing extras even though the inference acceptance suite passes.

Relocatability here means there are no functional absolute symlinks, RPATHs,
foreign shebangs or install-provenance links. Precompiled `.pyc` debug metadata
inside the dependency archive can still display the historical build prefix in
tracebacks; it is not used for import or dynamic linking and did not prevent a
second-prefix relocation.

## What was repaired

- The dependency archive embedded an older 2026-09-01 UniRL wheel with only the
  First-block cache. The installer now verifies and applies a current offline
  wheel containing First-block, Taylor, MagCache and FasterCache DFR.
- Runtime verification now pins the installed Leo2 source-tree digest and
  rejects absolute, broken or prefix-escaping links in addition to absolute
  RPATHs and direct-install metadata.
- `init_leo2_assets.sh` provides fail-closed `link` and resumable `copy` modes,
  transfers only the checkpoint/Qwen/VAE files in `artifacts.yaml`, and emits a
  sourceable `leo2-assets.env`.
- `doctor_leo2.sh`, a complete checksum manifest and reusable skills cover the
  install → assets → diagnostics → inference → release sequence.

## Artifact-version warning

This release is pinned to the benchmarked `iter0063300` native DCP payload:

```text
/root/zuhao/HYV2.0/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights
/root/zuhao/HYV2.0/assets/hymm_ar_assets
```

The earlier `iter0068800` path is a different checkpoint and is not covered by
the packaged sizes or SHA-256 values. Do not point the initializer at it until
a separate artifact profile and synchronized initializer have been generated
and validated.

## Remaining work for stricter definitions

1. For a repository-wide, rather than Leo2-only, source closure, vendor or
   remove `benchmarks/speed_benchmarks/verl_omni/upstream` and `.gitmodules`.
2. Build a separate Python 3.13/Torch 2.10 release inside the v1.9 image; FA2,
   FA3, FLA, causal-conv1d, DeepEP and NVSHMEM must all be rebuilt and re-run
   through the same release acceptance suite.
3. Generate a distinct manifest for `iter0068800` if that checkpoint is the
   desired production target.
4. Multi-node IB/GPUDirect behavior remains site-specific; the published
   DeepEP acceptance is single-node with `NCCL_IB_DISABLE=1`.
5. Vendored upstream history still contains unused training defaults with old
   absolute paths. They are not opened by the native inference entry, but a
   repository-wide textual-path policy would require a separate prune pass.
6. Add upstream LICENSE/NOTICE records before distributing the vendored source
   outside the current internal deployment boundary.
