# Leo2 first-block cache 8×H20 实验报告（2026-09-03）

## 1. 结论摘要

本轮正式实验已完成并通过完整性校验：8 个 case 全部正常退出，共生成并验证
128/128 个 MP4 和 128/128 个最终 latent。每个 case 使用相同的 16 组
prompt/seed，视频规格为 848×464、121 帧、24 FPS，采样 50 步。

主要结论如下：

- `threshold=0.02, shift=9` 没有命中 cache，输出与 cache-off 在 latent 和解码
  pixel 上逐值一致，速度也没有可测收益。
- `threshold=0.05` 是有效的中间档：跳过 30.1%～32.1% 的 denoise step，成对
  加速 1.365×～1.403×。其中 `shift=3` 的平均 latent/pixel 偏差最小，适合作为
  后续质量验收的首选候选；`shift=9` 稍快，但偏差更大。
- `threshold=0.10, shift=9` 跳过 55.5% step，成对加速 1.980×，但 latent
  relative L2 达到 0.222，解码 pixel aggregate RMSE 达到 0.0666；VBench
  dynamic degree 从 0.8125 降到 0.6875、imaging quality 从 0.5729 降到
  0.5465，不能作为默认生产设置。
- cache 带来的额外峰值 allocated memory 约 0.27 GiB；峰值 reserved memory
  基本不变，不是本轮方案的主要约束。
- `flow_shift_video` 本身没有造成可见的基线速度差：三个 cache-off case 的
  steady mean 均约 197.1 秒。它主要改变生成轨迹以及 cache 误差/命中行为。

VBench custom-input 与 VideoScore2 已完成 128/128 条评估。三个 `0.05` setting
都没有被自动指标一致判定为最优：shift 3 的逐值漂移最小且 VBench 基本不变，但
VideoScore2 三维均下降；shift 9 更快且部分 VBench 一致性略升，但 dynamic degree、
imaging quality 和 VideoScore2 physical consistency 下降。因此仍建议从
`threshold=0.05, shift=3` 开始 blind A/B，同时保留 `0.05@shift9` 作为性能候选。

## 2. 实验配置

| 项目 | 实际配置 |
|---|---|
| 节点/GPU | 单机 8× NVIDIA H20（compute capability 9.0） |
| 并行拓扑 | CP=8，DP shard=8，DP replicate=1，EP=1，ETP=1，TP=1，PP=1 |
| FSDP | FSDP2，`fsdp_impl=new` |
| Attention | `flash3_packed` |
| MoE | `ep_moe`，DeepEP disabled，grouped GEMM disabled |
| 输入 | 16 个不同 prompt，seed 42000～42015；各 case 严格复用 |
| 输出 | 848×464，121 frames，24 FPS，H.264 MP4 |
| 采样 | 50 denoise steps，guidance=1.0 |
| 对比矩阵 | off@shift 3/7/9；0.02@9；0.05@3/7/9；0.10@9 |
| 模型权重 | `/root/zuhao/HYV2.0/ckpts/leo2_moe_a12b_480p/iter_0063300_torch/weights` |
| 模型资产 | `/root/zuhao/HYV2.0/assets/hymm_ar_assets` |
| 仓库 HEAD | `471969969b4a9ab3e7f504e2468e19ff1abfb600` |

正式运行记录的关键指纹为：working-tree diff
`f433ba1832a96a50c8c54ecd53ea6e884e75314d1a03f2b832b60b4980e41e7f`，benchmark
harness `086eced6dafc8383e02777476e3a32f8ef23a3484f87764c93f71f440411a810`，prompt
CSV `9f7a596d1e7dd37a2f1a9ddffe566591e04f232be9fb4c4258d1231d5d18bfcc`，case CSV
`8ae66e620f0a239013bbd0000915df7cbf9dec7719fed08b7e91a5af6ec850d9`。完整 artifact、
model config 和 generation config 指纹保存在结果目录的 `benchmark.env` 中。

有效运行环境是 Python 3.12.12、Torch 2.7.1、CUDA 12.9、cuDNN 9.10.2、
Transformers 5.6.0、Diffusers 0.38.0、FA2 2.7.4.post1、FA3 3.0.0b1、
DeepEP 1.2.1+R03C03、NVSHMEM 3.7.0。它是本机实测的可迁移环境，不是目标镜像
声明的 Python 3.13/Torch 2.10 ABI；若必须严格匹配 v1.9 镜像，需要在该 ABI 下
重新编译 FA2、FA3、DeepEP 和 FLA kernel。

本次 cache benchmark 使用 EP=1，因此 DeepEP 不参与主实验；DeepEP 的 8-rank
dispatch/combine correctness 已在迁移后的独立环境中另行验证。

## 3. 测速结果

单条 latency 是 8 个 rank 的 CUDA-synchronized `MAX`，范围包括
`prepare_model_inputs + text conditioning + denoise + VAE decode`，不包括模型加载、
MP4 编码和文件写入。每个请求计时前都有跨 rank barrier，避免 rank 0 编码上一条
MP4 的时间污染下一条请求。`steady` 排除每个 case 的第一条 warm-up 样本。

| Case | Mean (s) | Steady (s) | Paired speedup（95% CI） | Skip steps | Skip ratio | 核心吞吐（video/h） |
|---|---:|---:|---:|---:|---:|---:|
| off, shift 3 | 198.077 | 197.070 | 1.000× | 0/800 | 0.000% | 18.17 |
| off, shift 7 | 198.103 | 197.072 | 1.000× | 0/800 | 0.000% | 18.17 |
| off, shift 9 | 198.118 | 197.090 | 1.000× | 0/800 | 0.000% | 18.17 |
| 0.02, shift 9 | 198.042 | 197.117 | 1.000× [0.999, 1.001] | 0/800 | 0.000% | 18.18 |
| 0.05, shift 3 | 144.877 | 143.843 | 1.368× [1.359, 1.377] | 241/800 | 30.125% | 24.85 |
| 0.05, shift 7 | 145.181 | 143.834 | 1.365× [1.353, 1.378] | 241/800 | 30.125% | 24.80 |
| 0.05, shift 9 | 141.272 | 140.345 | 1.403× [1.394, 1.411] | 257/800 | 32.125% | 25.48 |
| 0.10, shift 9 | 100.179 | 99.006 | 1.980× [1.954, 2.007] | 444/800 | 55.500% | 35.94 |

这里的核心吞吐是 `3600 / mean latency`，表示整台 8-GPU 节点在上述计时范围内
顺序生成视频的理论吞吐；它不包括模型 reload、编码和落盘。每个 case 的完整进程
wall time（包括加载和媒体输出）为 1796～3365 秒。

逐 prompt 的成对加速范围为：

- `0.05, shift 3`：1.329×～1.402×。
- `0.05, shift 7`：1.291×～1.403×。
- `0.05, shift 9`：1.368×～1.439×。
- `0.10, shift 9`：1.813×～2.013×。

## 4. Latent-space 误差

以同一 shift、prompt、seed 的 cache-off 最终 latent 为 reference。表中 L1/L2
均为相对范数，cosine 越接近 1 越好，max abs 是 16 对样本中的最大绝对误差。

| Case | Relative L1 mean | Relative L2 mean | Cosine mean | Max abs max |
|---|---:|---:|---:|---:|
| 0.02, shift 9 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.05, shift 3 | 0.0683 | 0.0755 | 0.9969 | 0.7709 |
| 0.05, shift 7 | 0.1153 | 0.1252 | 0.9908 | 0.9586 |
| 0.05, shift 9 | 0.1334 | 0.1436 | 0.9874 | 0.8700 |
| 0.10, shift 9 | 0.2122 | 0.2220 | 0.9729 | 1.3805 |

`threshold=0.02` 的 cache 路径虽然被启用，但 800 个 step 均执行 full forward，
最终 latent 与基线完全相同。这同时验证了“启用但零命中”不会改变数值结果。

## 5. Pixel-space 误差

像素指标对每对 MP4 使用 RGB24 解码并按 `[0, 1]` 归一化，逐帧锁步流式计算。
所有 80 对 candidate/reference 都验证了宽、高、帧数和 FPS。MAE/RMSE 是绝对
像素误差；relative L1/L2 使用 cache-off 像素范数作分母。

| Case | MAE mean / max | RMSE mean / max | Relative L1 mean / max | Relative L2 mean / max | Max abs mean / max |
|---|---:|---:|---:|---:|---:|
| 0.02, shift 9 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 0.05, shift 3 | 0.0167 / 0.0240 | 0.0267 / 0.0393 | 0.0439 / 0.0868 | 0.0632 / 0.1342 | 0.7728 / 0.9804 |
| 0.05, shift 7 | 0.0220 / 0.0425 | 0.0384 / 0.0837 | 0.0578 / 0.0944 | 0.0885 / 0.1489 | 0.8373 / 1.0000 |
| 0.05, shift 9 | 0.0242 / 0.0379 | 0.0427 / 0.0751 | 0.0664 / 0.1495 | 0.1023 / 0.2327 | 0.8593 / 1.0000 |
| 0.10, shift 9 | 0.0380 / 0.0578 | 0.0633 / 0.1004 | 0.1019 / 0.1791 | 0.1491 / 0.2743 | 0.9002 / 1.0000 |

高运动或高频反射场景更容易成为 worst case：skateboarder（index 10）、neon-lit
convertible（index 14）和 snowy tram（index 8）多次取得最大误差。因此后续视觉
验收应优先检查这些样本，而不是只看均值。单点 `max_abs` 容易被少量局部变化
主导，解释质量时应优先看 MAE/RMSE、感知指标和实际视频。

上述 L1/L2 只度量逐值偏离，不等价于语义一致性、运动质量或感知质量。下一节已
补充自动质量指标，但生产决策仍需 blind human A/B；本报告不声称
`threshold=0.05` 已通过最终质量门槛。

## 6. 自动质量评估与逐 shift 汇总

使用 [VBench](https://github.com/Vchitect/VBench) custom-input 官方支持的六个维度，
并用 [VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2) 补充视觉质量、文本
对齐和物理/常识一致性。VBench 六维和 VideoScore2 三维均已覆盖 8×16=128 条
视频；每个单元格是相应 setting 的 16 条均值。安装版本、benchmark 调研、协议和
复现命令见 [质量评估方案](QUALITY_EVALUATION_PLAN.md)。

表中 latent/pixel 误差以同一 shift 的 cache-off 为 reference，因此 baseline 为
零；它们表示 cache 漂移，不是绝对视频质量。`pixel RMSE` 是
`sqrt(mean(per-video MSE))`，与第 5 节的 `mean(per-video RMSE)` 定义不同。VBench
分数均按官方输出归一化到 `[0,1]`；dynamic degree 是 16 条中被判为动态的视频比例，
是描述性指标，不应与其余五维平均。VideoScore2 是 1～5 的离散 hard score，均值
每变化 0.0625 就对应一条视频变化 1 分，本轮不把小差异解释为统计显著。

### flow_shift_video=3

| 指标 | disable_cache | threshold=0.05 |
|---|---:|---:|
| Generation mean (s) ↓ | 198.077 | 144.877 |
| Paired speedup (×) ↑ | 1.000 | 1.368 |
| Skipped denoise steps (%) | 0.000% | 30.125% |
| Latent MSE vs off ↓ | 0.000000e+00 | 8.727567e-05 |
| Latent RMSE vs off ↓ | 0.000000 | 0.009342 |
| Latent relative L1 ↓ | 0.0000 | 0.0683 |
| Latent relative L2 ↓ | 0.0000 | 0.0755 |
| Latent cosine ↑ | 1.0000 | 0.9969 |
| Pixel MSE `[0,1]` vs off ↓ | 0.000000e+00 | 7.658778e-04 |
| Pixel MAE `[0,1]` vs off ↓ | 0.0000 | 0.0167 |
| Pixel RMSE `[0,1]` vs off ↓ | 0.0000 | 0.0277 |
| Pixel relative L1 ↓ | 0.0000 | 0.0439 |
| Pixel relative L2 ↓ | 0.0000 | 0.0632 |
| VBench subject consistency ↑ | 0.8926 | 0.8922 |
| VBench background consistency ↑ | 0.9272 | 0.9273 |
| VBench motion smoothness ↑ | 0.9760 | 0.9769 |
| VBench dynamic degree（描述性） | 0.8125 | 0.8125 |
| VBench aesthetic quality ↑ | 0.4842 | 0.4854 |
| VBench imaging quality ↑ | 0.6011 | 0.5985 |
| VideoScore2 visual `[1,5]` ↑ | 3.0000 | 2.8125 |
| VideoScore2 text alignment `[1,5]` ↑ | 2.9375 | 2.8125 |
| VideoScore2 physical `[1,5]` ↑ | 3.0625 | 2.7500 |

### flow_shift_video=7

| 指标 | disable_cache | threshold=0.05 |
|---|---:|---:|
| Generation mean (s) ↓ | 198.103 | 145.181 |
| Paired speedup (×) ↑ | 1.000 | 1.365 |
| Skipped denoise steps (%) | 0.000% | 30.125% |
| Latent MSE vs off ↓ | 0.000000e+00 | 3.102445e-04 |
| Latent RMSE vs off ↓ | 0.000000 | 0.017614 |
| Latent relative L1 ↓ | 0.0000 | 0.1153 |
| Latent relative L2 ↓ | 0.0000 | 0.1252 |
| Latent cosine ↑ | 1.0000 | 0.9908 |
| Pixel MSE `[0,1]` vs off ↓ | 0.000000e+00 | 1.743850e-03 |
| Pixel MAE `[0,1]` vs off ↓ | 0.0000 | 0.0220 |
| Pixel RMSE `[0,1]` vs off ↓ | 0.0000 | 0.0418 |
| Pixel relative L1 ↓ | 0.0000 | 0.0578 |
| Pixel relative L2 ↓ | 0.0000 | 0.0885 |
| VBench subject consistency ↑ | 0.8837 | 0.8852 |
| VBench background consistency ↑ | 0.9267 | 0.9249 |
| VBench motion smoothness ↑ | 0.9782 | 0.9793 |
| VBench dynamic degree（描述性） | 0.6875 | 0.6875 |
| VBench aesthetic quality ↑ | 0.5011 | 0.5038 |
| VBench imaging quality ↑ | 0.5764 | 0.5748 |
| VideoScore2 visual `[1,5]` ↑ | 3.4375 | 3.3750 |
| VideoScore2 text alignment `[1,5]` ↑ | 3.5000 | 3.3125 |
| VideoScore2 physical `[1,5]` ↑ | 3.4375 | 3.2500 |

### flow_shift_video=9

| 指标 | disable_cache | threshold=0.02 | threshold=0.05 | threshold=0.10 |
|---|---:|---:|---:|---:|
| Generation mean (s) ↓ | 198.118 | 198.042 | 141.272 | 100.179 |
| Paired speedup (×) ↑ | 1.000 | 1.000 | 1.403 | 1.980 |
| Skipped denoise steps (%) | 0.000% | 0.000% | 32.125% | 55.500% |
| Latent MSE vs off ↓ | 0.000000e+00 | 0.000000e+00 | 4.681244e-04 | 9.620577e-04 |
| Latent RMSE vs off ↓ | 0.000000 | 0.000000 | 0.021636 | 0.031017 |
| Latent relative L1 ↓ | 0.0000 | 0.0000 | 0.1334 | 0.2122 |
| Latent relative L2 ↓ | 0.0000 | 0.0000 | 0.1436 | 0.2220 |
| Latent cosine ↑ | 1.0000 | 1.0000 | 0.9874 | 0.9729 |
| Pixel MSE `[0,1]` vs off ↓ | 0.000000e+00 | 0.000000e+00 | 2.084417e-03 | 4.439328e-03 |
| Pixel MAE `[0,1]` vs off ↓ | 0.0000 | 0.0000 | 0.0242 | 0.0380 |
| Pixel RMSE `[0,1]` vs off ↓ | 0.0000 | 0.0000 | 0.0457 | 0.0666 |
| Pixel relative L1 ↓ | 0.0000 | 0.0000 | 0.0664 | 0.1019 |
| Pixel relative L2 ↓ | 0.0000 | 0.0000 | 0.1023 | 0.1491 |
| VBench subject consistency ↑ | 0.8938 | 0.8938 | 0.8951 | 0.8970 |
| VBench background consistency ↑ | 0.9279 | 0.9279 | 0.9305 | 0.9302 |
| VBench motion smoothness ↑ | 0.9797 | 0.9797 | 0.9807 | 0.9825 |
| VBench dynamic degree（描述性） | 0.8125 | 0.8125 | 0.7500 | 0.6875 |
| VBench aesthetic quality ↑ | 0.5123 | 0.5123 | 0.5147 | 0.5073 |
| VBench imaging quality ↑ | 0.5729 | 0.5729 | 0.5655 | 0.5465 |
| VideoScore2 visual `[1,5]` ↑ | 3.1250 | 3.1250 | 3.0625 | 3.1875 |
| VideoScore2 text alignment `[1,5]` ↑ | 3.3125 | 3.3125 | 3.3750 | 3.1875 |
| VideoScore2 physical `[1,5]` ↑ | 3.3750 | 3.3750 | 3.1875 | 3.3125 |

`threshold=0.02@shift9` 在视频内容、所有成对误差、VBench 和 VideoScore2 上均与
cache-off 完全一致，符合零 cache hit 的预期。`0.05@shift3` 的 VBench 六维变化均
小于 0.003，但 VideoScore2 physical mean 下降 0.3125；`0.05@shift7` 的三项
VideoScore2 分别下降 0.0625/0.1875/0.1875。`0.05@shift9` 在 1.403× 加速下
subject/background/motion/aesthetic 略升，dynamic degree 下降 0.0625、imaging
下降 0.0074、VideoScore2 physical 下降 0.1875。`0.10@shift9` 的 1.980× 加速伴随
最大的逐值漂移和明显的动态/成像下降，自动指标不支持把它设为默认值。

## 7. 显存与系统行为

cache-off 的最大 allocated memory 为 64.65 GiB；cache-on 为 64.92 GiB，增量约
0.27 GiB。最大 reserved memory 在 78.15～78.47 GiB 之间，没有随 threshold
单调增长。GPU 总显存约 95 GiB/卡，本轮没有 OOM。

实验显式设置 `NCCL_IB_DISABLE=0`、`NCCL_SOCKET_IFNAME=bond1` 和
`NCCL_IB_GID_INDEX=3`。cache decision 在 CP=8 统计后，再在实际 FSDP shard group
内同步。当前实现对于 EP/ETP/TP/PP 大于 1 的拓扑会保守退化为 full forward，
所以本报告不能外推到 DeepEP/EP>1 的 cache 性能。

## 8. 结果完整性与被舍弃的预运行

最终结果目录：

```text
/root/leo2-output/cache-full-final-848x464x121-20260903-1039
```

目录占用约 1.3 GiB。`final_results.sha256` 覆盖最终 summary、成对 latent 指标和
成对/聚合 pixel 指标；2026-09-03 验收时 7/7 文件均通过 `sha256sum -c`。每个
case 均满足：exit code 0、16 requests、16 valid MP4、16 valid latents、world size 8。

正式运行前有两次主动舍弃的预运行，不应与最终结果合并：

- `...-1008`：运行 5 条后发现请求前缺少跨 rank barrier，上一条 rank-0 MP4
  编码约 2.85 秒会污染下一条计时，已标记 `INVALID_TIMING_BARRIER_MISSING.md`。
- `...-1034`：第一条视频前发现继承了 `NCCL_IB_DISABLE=1`，step latency 从约
  3.62 秒升至约 9 秒，已标记 `INVALID_NCCL_IB_DISABLED.md`。

这两个问题在最终 harness 中分别通过 request 前 barrier 和显式
`NCCL_IB_DISABLE=0` 修复。

## 9. 结果文件索引与复算

最终目录中的主要文件：

| 文件 | 内容 |
|---|---|
| `summary.csv` / `summary.json` | case 级性能、cache 命中、显存、完整性 |
| `paired_metrics.csv` | 逐 prompt 成对加速及 latent L1/L2/cosine/max abs |
| `pixel_metrics_pairs.csv` | 80 对视频的逐 prompt pixel 指标 |
| `pixel_metrics_cases.csv` | pixel 指标的 case 级 mean/max |
| `quality_eval/vbench/` | 8 个 case 的 VBench 六维逐视频与聚合结果 |
| `quality_eval/videoscore2/` | 128 条 VideoScore2 三维分数与原始回答 |
| `quality_eval/summary/quality_metrics_cases.*` | 本报告逐 shift 表格的统一机器可读来源 |
| `quality_eval/evaluation.env` / `quality_results.sha256` | 评估版本、参数、runner 指纹和完整结果校验和 |
| `benchmark.env` | commit、代码/输入/artifact hash、模型和运行参数 |
| `<case>/summary.json` | 单 case 配置、逐请求时延、cache step 和媒体校验 |
| `<case>/samples/.../videos/*.mp4` | 实际视频 |
| `<case>/latents/*.pt` | VAE decode 前的最终 latent |
| `final_results.sha256` | 最终聚合结果校验和 |

上述聚合 CSV/JSON 和运行指纹另有一份仓库内快照：
[`docs/results/cache_benchmark_20260903/`](results/cache_benchmark_20260903/README.md)。
它不包含大体积视频和 latent，但带有独立 `SHA256SUMS`，即使完整输出目录日后被
清理，报告中的数值仍可在仓库内追溯。

可从仓库根目录复算 pixel 指标（默认拒绝覆盖已有结果）：

```bash
/root/leo2-runtime/bin/python \
  examples/diffusion/leo2/scripts/compute_pixel_metrics.py \
  --root /root/leo2-output/cache-full-final-848x464x121-20260903-1039 \
  --output-dir /tmp/leo2-pixel-metrics-recomputed
```

视频目录的一般形式为：

```text
<benchmark-root>/<case>/samples/cache_benchmark_16__16/videos/
```

迁移环境发布目录为：

```text
/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/envs/leo2
```

归档 `leo2-runtime-py312-torch271-cu129-20260903.tar.gz` 约 2.4 GiB，SHA-256 为
`ddc1fd7f587a05799a1f5d81bb530b35c1591f380c540b69409a5adf8a193cf7`。它已通过
干净环境迁移安装、绝对 symlink/RPATH/provenance 检查、FA2/FA3 kernel smoke，
以及 8-rank DeepEP 首次和 cached-handle dispatch/combine round-trip（最大误差 0）。

## 10. 后续建议

1. 对 `0.05@shift3`、`0.05@shift9`、`off@对应 shift` 做 blind human A/B；优先
   覆盖 index 8、10、14，并增加人物手部、文字、快速镜头与细小刚体运动场景。
2. 用 blind human A/B 和业务指标建立可接受阈值，再决定默认采用
   `0.05@shift3` 还是 `0.05@shift9`；VBench/VideoScore2 已完成，但 16 条样本不足以
   替代人工验收，也不要只按加速比选型。
3. 若要测试 EP>1/DeepEP 拓扑，先扩展 cache decision 的同步域并增加硬失败检查，
   然后重新跑完整 paired benchmark；当前结果仅适用于本报告中的 CP8/FSDP8/EP1。
4. 若部署基线必须是 v1.9 的 Python 3.13/Torch 2.10，应在目标镜像内重建并重新
   验证 compiled kernels，不能直接复用本报告的 Py3.12/Torch2.7 二进制包。
