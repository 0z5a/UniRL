# Leo2 Acceleration Lab

Manifest-driven, read-only Lab Hub subpage for paired Leo2 acceleration results.
It serves metrics, release metadata, prompt/case records, figures and manifest-
approved media. Video responses support HTTP Range for synchronized same-seed
playback.

```bash
python3 server.py \
  --bind 127.0.0.1 \
  --port 16023 \
  --manifest /apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/leo2_acceleration_benchmark/releases/current.json
```

The reverse proxy should preserve the existing Lab Hub Basic Auth and route
`/leo2-acceleration/` to `http://127.0.0.1:16023/`. The frontend uses only
relative URLs, so it works under that prefix when the proxy strips the location
prefix.

The release manifest has `schema_version: 1` and these top-level fields:

- `release_id`, `artifact_root`, and the immutable experiment `contract`
- `filters` for steps, guidance and methods
- flat `metrics` rows for overall, English and Chinese aggregates
- `prompts`, each containing exact and accelerated same-seed `videos`
- published `figures`
- `files`, the complete allowlist of media paths and SHA-256 digests

Run the backend tests with:

```bash
python3 -m unittest discover -s tests -q
```
