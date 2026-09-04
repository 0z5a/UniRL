# Leo2 migration skills

Load only the skill needed for the current migration step:

Set `LEO2_RELEASE_DIR` to the directory containing the packed archive and these
guides. Commands that launch inference separately state when they require the
UniRL repository root.

- [`runtime-install`](runtime-install/SKILL.md): install and validate the packed Python/CUDA runtime.
- [`heavy-assets`](heavy-assets/SKILL.md): initialize checkpoint and model assets by link or manifest-limited copy.
- [`inference`](inference/SKILL.md): validate artifacts and run the native 480p, 121-frame smoke or full inference.
- [`troubleshooting`](troubleshooting/SKILL.md): diagnose ABI, artifact, FA3, DeepEP and relocation failures.
- [`release-build`](release-build/SKILL.md): rebuild, verify and mirror a versioned environment release.

The runtime, repository and heavy assets are separate migration units. Do not
claim a host ready until all three units pass their own verification.
