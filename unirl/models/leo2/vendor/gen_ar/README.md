# Vendored Leo2 runtime

This directory is a source snapshot for the UniRL Leo2 video pipeline. It is
part of the `unirl` wheel and must not be replaced with Git submodules or an
external `hunyuan_multimodal_gen_ar` checkout at runtime.

Included source:

- the `hymm` Leo2 model, conditioning, diffusion, VAE and native T2V sampler
  closure;
- `processors` media utilities;
- the required `hy_parallelism` and `IndexKits` sources under `deps/`;
- the required base defaults and validated A12B 480p stage-3 YAML.

Training entrypoints, unrelated model samplers, reward/metric services, tests,
cluster tooling and unused model YAMLs are deliberately omitted. Native metric
evaluation is therefore unsupported; write generated media here and evaluate
it in a separate environment.

Weights, the Qwen3.5 text encoder and the video VAE are data artifacts rather
than source dependencies. Configure those with `LEO2_CKPT_DIR` and
`LEO2_ASSETS_BASE`; see `examples/diffusion/leo2/ENVIRONMENT.md`.

Exact source revisions, exclusions and local compatibility patches are recorded
in `VENDOR_COMMIT.txt`.
