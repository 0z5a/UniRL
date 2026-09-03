# Acceleration quality-evaluation provenance

Evaluation completed on 2026-09-04 for Taylor, MagCache, and FasterCache DFR.
Each candidate and the shared exact/static references contain 16 distinct
prompt/seed pairs at 848x464, 121 frames, and 24 FPS. VBench and VideoScore2
were run in separate environments because their Torch and Transformers
requirements differ.

| Component | Pinned source or runtime |
|---|---|
| VBench source | `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` |
| VBench runtime | Python 3.10, Torch 2.5.1+cu121, Transformers 4.33.2 |
| VBench dimensions | subject/background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality |
| VideoScore2 source | `a88168af50b1dd98f0c3d973620ac495daab2de4` |
| VideoScore2 model snapshot | `09a2732cb64fa566a1f332f978368292ce5c295c` |
| VideoScore2 runtime | Python 3.10, Torch 2.6.0+cu124, Transformers 4.53.2 |
| VideoScore2 inference | FPS 2, temperature 0.7, evaluator seed equal to generation seed |

Runner fingerprints from the evaluated repository state:

| Script | SHA-256 |
|---|---|
| `evaluate_videoscore2.py` | `beaafdcbb087bf6daa0be59a14ebb6167adaa44011860a8eb28e953d8613df2e` |
| `run_quality_evaluation.sh` | `776480130bd3c2a901a277857c44b5bda572145fc40c3bc524efdcbe44f6373b` |
| `summarize_quality_evaluation.py` | `9873a8915e37093e5c28b86038a614e67857e38983915d02e382665b750dfd35` |
| `summarize_acceleration_results.py` | `bcca24983c9cf5f2ea1ff63df05b3714a6ac33605578c0d03e3f21e8a1d00058` |

The evaluation covered 48 newly evaluated candidate videos (16 per method).
The consolidated report also reads the previously completed exact and static
quality results from
`../cache_benchmark_20260903/quality_eval/summary/quality_metrics_cases.*`.
All raw candidate scores and VBench per-item records are retained under each
full-suite `quality_eval` directory. The JSON report records source artifact
hashes, while `SHA256SUMS` covers every committed snapshot file.
