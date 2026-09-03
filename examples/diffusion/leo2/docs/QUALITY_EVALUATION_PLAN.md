# Leo2 视频质量评估方案（2026-09-03）

## 目标与结论

本轮目标是比较相同 prompt/seed/flow-shift 下 cache-off 与不同 cache threshold 的
质量，而不是生成可与公开 leaderboard 横比的分数。当前 16 个自定义 prompt 每个
setting 只有一个视频，因此主评估采用：

1. VBench custom-input 官方支持的六维无参考指标；
2. VideoScore2 的视觉质量、文本对齐、物理/常识一致性三个 1～5 分维度；
3. 同 prompt/seed 的 latent 与解码 pixel 成对误差；
4. 人工 blind A/B，作为自动指标之外的最终上线门槛。

VBench 标准协议有固定 prompt suite，且通常要求每个 prompt 多次生成；本轮 16 个
自定义 prompt 的结果只能称为 `VBench custom-input`，不得作为 VBench leaderboard
分数。VBench dynamic degree 表示视频是否有足够动态，并非单调的画质分数，因此
不与另外五维强行平均成“总分”。

## Benchmark 调研

| Benchmark | 主要覆盖 | 本轮采用方式 |
|---|---|---|
| [VBench](https://github.com/Vchitect/VBench) / [论文](https://arxiv.org/abs/2311.17982) | T2V 的 16 个质量与语义维度 | 采用 custom-input 支持的主体一致性、背景一致性、运动平滑、动态程度、美学、成像质量六维 |
| [VBench++](https://arxiv.org/abs/2411.13503) | 扩展到 I2V、变长宽比和可信性 | 后续增加 I2V 或标准 prompt suite 时采用；本轮 T2V 核心仍按 VBench |
| [VBench-2.0](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md) / [论文](https://arxiv.org/abs/2503.21755) | 人体、创造性、物理、语义和偏见的 18 维评估 | 当前 custom-input 覆盖有限且部分维度要求每 prompt 20 个视频，暂不作为主指标 |
| [VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2) | 人类偏好对齐的视觉、文本对齐和物理一致性 | 采用官方 query 与采样参数，记录可审计的 1～5 hard score 和原始回答 |
| [EvalCrafter](https://github.com/evalcrafter/EvalCrafter) | 700 prompts、17 个客观和主观指标 | 与 VBench 重合较多且环境较旧，本轮不重复安装 |
| [T2V-CompBench](https://t2v-compbench.github.io/) | 属性绑定、空间、动作等组合性 | 后续专门补生成其 700-prompt suite 时使用 |
| [PhyGenBench](https://github.com/PhyGenBench/PhyGenBench) | 160 prompts、27 条物理规律 | 后续物理专项；当前由 VideoScore2 先做低成本筛查 |
| [FVD](https://arxiv.org/abs/1812.01717) | 生成集与参考集的分布距离 | 当前没有匹配参考分布且每 setting 仅 16 条，不报告容易误导的 FVD |
| [DOVER](https://github.com/VQAssessment/DOVER) | 无参考 UGC 技术/美学质量 | 可作补充，但不衡量 prompt alignment，本轮已有 VBench 与 VideoScore2 覆盖 |

## 已安装环境

为避免 VBench 与 VideoScore2 对 Torch/Transformers 的冲突，使用两个独立环境：

| 项目 | 路径/版本 |
|---|---|
| VBench source | `/root/leo2-eval/sources/VBench`，commit `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490` |
| VBench env | `/root/leo2-eval/envs/vbench`，Python 3.10、Torch 2.5.1+cu121、Transformers 4.33.2 |
| VBench weights | `/root/leo2-eval/cache/vbench`，另设 `TORCH_HOME=/root/leo2-eval/cache/torch` |
| VideoScore2 source | `/root/leo2-eval/sources/VideoScore2`，commit `a88168af50b1dd98f0c3d973620ac495daab2de4` |
| VideoScore2 env | `/root/leo2-eval/envs/videoscore2`，Python 3.10、Torch 2.6.0+cu124、Transformers 4.53.2 |
| VideoScore2 weights | `TIGER-Lab/VideoScore2`，snapshot `09a2732cb64fa566a1f332f978368292ce5c295c` |

两个环境均已完成单视频端到端 GPU smoke。VideoScore2 上游正则表达式不能解析模型
实际输出中带编号和破折号描述的 label；本仓库 runner 只增强解析，不改变官方
query、`fps=2`、`temperature=0.7` 或模型权重，并保留每条原始回答供复核。

## 执行和统计协议

完整输入为 8 settings × 16 prompts = 128 个 848×464、121 帧视频。VBench 和
VideoScore2 均以每个 setting 的 16 条算术平均作为表中结果；VideoScore2 对同一
prompt 在所有 settings 重置为相同 evaluator seed，降低其采样噪声。比较时必须在
同一 flow-shift 内进行：shift 3 和 7 各比较 off/0.05，shift 9 比较
off/0.02/0.05/0.10。

复现命令：

```bash
examples/diffusion/leo2/scripts/run_quality_evaluation.sh \
  /root/leo2-output/cache-full-final-848x464x121-20260903-1039

/root/leo2-runtime/bin/python \
  examples/diffusion/leo2/scripts/summarize_quality_evaluation.py \
  --benchmark-root /root/leo2-output/cache-full-final-848x464x121-20260903-1039 \
  --cases-csv examples/diffusion/leo2/data/cache_benchmark_cases.csv \
  --output-dir /root/leo2-output/cache-full-final-848x464x121-20260903-1039/quality_eval/summary
```

人工 A/B 应隐藏 setting，随机左右顺序，并按 prompt 配对打分。优先复核 pixel
误差最大的 index 8、10、14，以及自动指标差异与视觉观感矛盾的样本。最终选择应
同时满足速度收益、相对 cache-off 的成对保真度和独立质量指标，而不是按任一单项
排序。
