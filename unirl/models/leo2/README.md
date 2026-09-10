# Leo2：条件预处理、缓存与组件 offload

## Prompt 到 latent 的准确边界

生成路径是 `prompt → 原生 tokenizer / prompt template → 冻结 Qwen3.5-9B → cond_text_states`，
再将文本条件、attention masks、RoPE 媒体位置和有效视频尺寸交给 Leo2。
Leo2 接收的是 **当前 noisy latent + sigma + 完整条件**，每步输出 velocity；
UniRL 的采样器迭代更新 latent，最后得到 `clean_latent`。因此并不是 prompt embeddings 经一次 DiT forward 就得到 clean latent。

SFT 的 clean target 来自另一条路径：`目标视频 → 冻结 VAE encoder → latent normalization → x0`。
训练时每次重新采样 `ε, σ`，构造 `xσ = (1−σ)x0 + σε`，以 `ε−x0` 为 velocity 监督目标。
冻结文本编码器的结果和目标 x0 都可以缓存；噪声、sigma、DiT 激活与预测仍在训练时计算。
DiT 内的文本投影、文本分支等可训练层不能提前缓存，它们仍属于训练模型。

## 文本如何注入

当前 pinned Leo2 是 `LeoTripleLayer × 48`：视觉、文本、音频有独立的归一化、Q/K/V/O 投影和 FFN。
T2V 使用视觉与文本 token；音频分支参数仍在 checkpoint 中。
注意力内部把各模态 Q/K/V 按统一 token 位置组合，施加对应 RoPE 和 attention mask，执行 joint attention，
然后按模态拆回各自的输出投影和 FFN。

这不是额外的 `Q_visual × K_text` cross-attention 子层，也不是所有 token 共用整套 block 参数的纯 single-stream DiT。
更准确地说，是 **多流参数结构 + 联合序列注意力**；“统一 attention sequence”与“整层共享参数”是不同概念。
具体参数量和逐层结构见[架构报告](../../../examples/diffusion/leo2/docs/ARCHITECTURE_SFT_INFERENCE_REPORT.md)。

## 实现与 Flow-Factory 的对应关系

参考 `/root/Flow-Factory/src/flow_factory/trainers/abc.py::_init_dataloader` 的
“加载 preprocessing modules → 构建缓存数据 → offload”顺序，以及
`data_utils/dataset.py`、`data_utils/loader.py` 的 CPU tensor、配置指纹和分片缓存设计。
UniRL 使用独立预处理进程：它结束后再启动训练，可避免预处理组件与训练模型的加载峰值重叠。

| 阶段 | 文本编码器 | 视频 VAE | Leo2 DiT |
|---|---|---|---|
| 离线文本条件编码 | GPU，冻结 | CPU | meta 配置壳，无权重存储、不运行 forward |
| 离线目标视频编码 | CPU | 每次编码时 GPU，之后 CPU | 同上 |
| 缓存 SFT | 不加载 | 不加载 | 训练所需的分片权重、激活和优化器状态 |
| 缓存在线 RL | 不加载 | 仅解码生成视频时 GPU | 采样和训练时运行 |

RL 生成的视频随策略变化，不能把固定缓存的 x0 替代在线 rollout。像 PickScore 这类像素奖励仍需 VAE 解码，
奖励模型本身的显存也仍需计入。`vae_on_gpu: false` 在 codec 的 `finally` 中清理 VAE 内部缓存并移回 CPU。

## 配置

| 字段 | 默认值 | 含义 |
|---|---|---|
| `preprocessing_cache_mode` | `off` | `off` 使用现有在线条件编码；`readonly` 只读取离线条件 |
| `preprocessing_cache_dir` | `null` | 所有训练 worker 可见的本地/共享目录 |
| `load_video_vae` | `true` | 缓存 SFT 可设 `false`；生成像素视频必须为 `true` |
| `vae_on_gpu` | `true` | `false` 时 VAE 常驻 CPU，codec 调用期间进入 GPU |
| `condition_cache_size` | `1` | 原有在线条件 LRU，缓存 tensor 放 CPU；不控制磁盘缓存 |

只读模式从 bundle 构建阶段就跳过 Qwen 和 tokenizer 的加载，不创建依赖冻结组件的原生 diffusion pipeline。
DiT 前向改为独立生成 T2V 的零 channel conditions，并递归搬运缓存中的所有 tensor。
缓存 SFT 配方使用 `Leo2CachedSupervisedTrackBuilder`，走现有 `FlowMatchSFT → predict_noise_at_step` 链路。
当前配方是 DiT **LoRA** 训练，不是默认全参数训练；rollout 和训练 micro-batch 均可使用 packed batch。

## Packed sequence forward

每个样本仍由原生预处理产生完整的单行 token 序列。前向时将多个序列拼成一个物理行，并通过
segment-length attention mask 和 `sample_offsets` 保留样本边界；FA3 因而只在各样本内部计算 attention。
视频和音频 latent 保留逻辑 batch 维；共享 timestep 保持 scalar，不同 timestep 保留 B 个值。
patch projection 后再沿 token 轴拼接。
B=1 和 B>1 使用同一条路径，避免两套 timestep 与输出拆分逻辑产生偏差。

Leo2 pipeline 为每次真实 generate call 记录该 pack 的完整有序 sample ids；CountPlanner 将该 pack
作为不可拆分的 replay micro。pack 数不能整除 optimizer update 数时会在训练前报错。

`und/gen/audio_token_indices` 会按样本 offset 平移。使用 CP 时，每个分支独立补齐到 CP size 的倍数，
padding 通过 mask 从 attention 和输出 projection 中排除。该路径要求 `flash_packed` 或
`flash3_packed`，且当前 B>1 仅支持 `linear` text projection；dense attention backend 无法表达拼接
序列的隔离边界。B=1 的 timestep modulation
保持 `[1,1,H]` 并由逐元素运算广播，避免每层重复物化完整 token 长度的 modulation tensor。

模型边界已经接受同一任务内不同 H/W/F、不同音频长度的 ragged latent，并在一次 packed forward 后
按原始顺序和 shape 拆回预测。当前在线 rollout、trajectory、SDE 和 decoder 仍使用矩形 batch，正式
Flow-GRPO 配方因此先支持同分辨率的 T2V/T2VA。端到端异构训练还需逐样本生成 noise/trajectory、逐行
执行 SDE transition，并按 shape 分桶解码；不能用 `Videos.from_list` 把输出缩放到共同分辨率。

原生 Leo2 还定义了 T2I、I2VA、FL2VA 和 T2A。I2VA/FL2VA 需要逐样本 channel condition，T2I/T2A
使用不同 scheduler，缺失模态则由原生 dummy branch 补齐。跨任务 packed batch 因而还需要逐样本 task
spec、modality presence 和 scheduler，并在 decoder/reward 边界按任务拆分后恢复原始顺序。原生训练器本身
也只在同一 dataset/task 的 buffer 内组 pack；多任务是在不同 step 间抽样，并未直接提供跨任务 pack。

## 使用方法

使用现有 Leo2 runtime Python 和已配置的外部权重，设置：

```bash
export LEO2_CKPT_DIR=/path/to/leo2_dcp
export LEO2_ASSETS_BASE=/path/to/assets
export LEO2_PREPROCESS_CACHE=/shared/leo2_preprocessed
export SFT_DATA=/shared/train.jsonl
export SFT_EVAL_DATA=/shared/val.jsonl
```

SFT manifest 保持原始数据格式；相对视频路径按 manifest 所在目录解析：

```json
{"sample_id":"video-001","prompt":"A dog runs across a field.","media_refs":[{"modality":"video","role":"target","uri":"videos/dog.mp4"}]}
```

先离线预处理 **train 和 eval**：

```bash
python -m unirl.models.leo2.preprocess \
  --recipe sft/leo2_sft_cached \
  --manifest "$SFT_DATA" "$SFT_EVAL_DATA" \
  --cache-dir "$LEO2_PREPROCESS_CACHE" --encode-targets

ENTRY=train_sft bash examples/run_experiment_single_node.sh sft/leo2_sft_cached
```

在线 RL 只预计算文本条件，仍使用原始 prompt 数据集。以现有训练/评估 prompt 文件为输入：

```bash
python -m unirl.models.leo2.preprocess \
  --recipe diffusion/leo2/leo2_t2v_cached \
  --manifest /path/to/train_prompts.txt /path/to/eval_prompts.txt \
  --cache-dir "$LEO2_PREPROCESS_CACHE"

bash examples/run_experiment_single_node.sh diffusion/leo2/leo2_t2v_cached
```

`--override sampling.height=464` 等 Hydra overrides 可重复传入；预处理与训练必须使用相同的尺寸、帧数和原生配置。
命令读取配方，但不启动 trainer/Ray worker。它自行初始化一个单进程 NCCL group，预处理原生编码器需要 CUDA。
并行预处理使用独立单 GPU 进程，例如分别设置 `CUDA_VISIBLE_DEVICES=0/1`，传
`--num-shards 2 --shard-index 0/1`；不能以多 rank `torchrun` 启动同一预处理任务。
每条记录按索引分片，重复 prompt 共用同一条缓存；文件原子发布，可重跑补全中断的工作。

## 缓存契约与限制

- 每项 `.pt` 保存 CPU tensor，保留 bf16/fp32、整数及 bool dtype；通过 `weights_only=True` 加载。
  RoPE 中的 `slice` 显式编码为安全的内置类型，恢复时不依赖自定义 pickle 类。
- 缓存的是完整 `input_ids/model_kwargs/image_size/video_duration`，包括 masks、文本状态、RoPE 和 token indices。
  读取时使用实际的原生 bucket 尺寸，而非假设请求尺寸没有被调整。
- 指纹覆盖 schema、固定预处理 seed、模型/生成配置、原生参数、编码器资产和原生/适配层预处理代码。
  小资产按内容校验，大权重按文件大小和纳秒 mtime 校验；不为每次启动扫描数十 GB 权重内容。
  不能检测人为保持大小和 mtime 不变的权重替换。复制资产时应保留时间戳，或重新预处理。
- prompt key 包含文本与请求几何，不含 rollout seed。当前支持的纯 T2V 条件以固定 seed=0 编码，
  rollout 的随机初始噪声与 SFT 的随机噪声仍独立变化。暂不支持依赖随机媒体增强的条件预处理。
- target key 额外包含原视频绝对路径、文件修订信息、有效几何和 `max_decode_frames`。
  训练仍需能 stat 原视频，且资产路径仍通过现有配置校验；“不加载组件”不等于可以删除原始资产文件。
- 目标视频先经现有 `load_video` 限制解码帧数，再均匀抽取/重复到有效帧数，bicubic resize 到有效尺寸，
  将像素映射到 `[-1,1]`，使用 VAE posterior **mode** 并应用原生 latent normalization。
  这是确定性的缓存训练策略，不复刻上游所有随机裁剪或 posterior sampling 增强。
- 当前仅支持纯文本 T2V（guidance=1）。媒体条件会拒绝；不同有效尺寸不能拼入同一个 builder batch。
- 缓存目录指纹不匹配或条目缺失直接报错，不在训练中回退到编码器加载。
  显存节省来自移除冻结组件；75B DiT 本身仍需现有 FSDP/LoRA 方案，本改动不改变训练权重加载方式。

## 验证范围

本次以临时 CPU harness 验证缓存 tensor/slice 往返、缓存失效、读取不调用编码器、SFT 数据和梯度链路、
组件跳过加载及 codec 异常时的 offload；检查新配方和 Python 语法。
2026-09-06 已在 8 节点 64 张 H20 上完成 50,000 条 prompt 的原生 GPU 预处理，
并完成缓存 RL 的视频 rollout；日志确认文本编码器未加载、rollout 前 VAE 位于 CPU。
完整拓扑测试和 64 GPU 训练结果见[运行报告](../../../examples/diffusion/leo2/docs/results/preprocessing_topology_20260906/BENCHMARK_REPORT.md)。
在线/离线条件已用一个 prompt 在 8 个 rank 上比较：80 次 tensor 检查均逐值一致，max abs diff=0。缓存 SFT 的原生 GPU step 尚未验证。

## 独立 rollout

`Leo2RolloutEngine` 在独立 GPU slab 中构建原生 Leo2 pipeline，支持与训练侧不同的 CP/EP。
配方见 `examples/diffusion/leo2/leo2_t2v_flowgrpo_separate.yaml`；默认将 GPU 对半分配。
Rollout 侧复用 FSDP 权重分片和 LoRA 注入，关闭 activation checkpointing，不执行 optimizer step。
`RemoteLoraWeightSync` 发布完整 adapter，写入现有参数并校验回读；训练与 rollout 的 LoRA rank/alpha 必须一致。
当前只接受 readonly 条件缓存、单个无 bias/dropout 的 default adapter；单次 forward 可包含多个 packed 视频。
CPU 配置与 adapter 往返已验证；16 GPU 同卡独立进程模式也已通过两次更新、参数同步回读与跨拓扑 replay 检查，性能结果见运行报告。

`leo2_t2v_flowgrpo_colocated.yaml` 使用同一批物理 GPU 上的两个 worker slot，分别构建训练和 rollout 进程组。
`offload_on_sleep=true` 使 sampler 在训练前将权重移至 CPU；训练权重在 rollout 前同样 offload。
它复用 `gpu_store` 的跨进程 tensor 传输，仍按 rollout → train 的顺序执行，适合比较不同拓扑与权重搬运的总成本。
独立 slab 配方默认不在 sleep 时搬运 sampler 权重，避免不必要的 CPU/GPU 传输。

## Gotchas

- 原生 `hymm` / `hy_parallelism` 的 CP/EP 是进程级全局状态。不同拓扑的训练和 rollout 必须使用独立进程组；不能在同一 worker 中切换两个模型的全局拓扑。
- 原生 VAE、tokenizer 和文本编码器也使用全局单例。预处理 builder 应在独立进程初始化；已加载完整 bundle 的进程应复用组件，不能再次构建同一个单例。
- 独立 rollout 的 `fsdp_cfg.sp_size` 同时决定 UniRL prompt scatter 的 CP 分组，必须与 `bundle_config.context_parallel_size` 一致。两侧 DP 必须满足 prompt / generated sample 的整除约束。
- Leo2 rollout 直接使用现有 PEFT 参数，没有另一个推理框架的 adapter buffer；回读的 B 矩阵乘以 alpha/r，以匹配共享同步校验协议。
- 多节点 `hybrid8` 需要先初始化默认进程组再构造二维 FSDP mesh。冷启动时跳过 mesh 会意外变成全局分片，改变通信与显存，不能把这种运行记作节点内 HSDP。
