# Leo2 预处理与 64 GPU FlowGRPO 拓扑测试（2026-09-06）

**拓扑筛选已完成（共 52 条历史及本轮记录）；选择共享 DiT、CP2/EP1、检查点开，正在进入完整 64 GPU、32×16、121 帧验证。正式长训尚未启动。**

## 最终选择

选择 `CP2 / EP1 / TP1 / HSDP8 / activation_checkpointing=true`，训练与采样共享 DiT。
CP2/EP1 第二轮 1365.6 秒、峰值 62.48 GiB；同卡训推分离为 1362.9 秒、89.09 GiB，0.2% 差异不足以证明吞吐改善。
在吞吐差异不超过 1% 的候选中优先选择较低峰值显存，以留出完整 512 视频批量的容量；这不是对所有批量或所有拓扑的全局最优声明。
所有后续运行均使用 464×848×121、group_size=16、num_groups=32；原生 CP/EP/FSDP 分片限制在节点内，但数据并行梯度同步仍跨节点。
完整批量验证完成后，从实际保存的 checkpoint 恢复正式训练，并延续同一个 W&B run。
证据：`selected_config.json`。训推分离功能及其可复现实验配方保留。

## 生产几何与预处理

实际 bucket 为 **464×848×121**（480p），latent `[1,48,31,29,53]`，47,647 visual tokens。
训练集 48,976 条、评估集 1,024 条，共 50,000 条。8 节点 × 8 GPU 的 64 个预处理分片全部退出成功，
全量 50,000 缓存条目存在，抽查 64 条均为上述实际几何。证据：`production121_cache_verified.json`。
旧 49 帧测试另见 [完整历史基线](SHORT_VIDEO_BASELINE.md)，不能据此选择生产长度最优配置。

缓存位于 `/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/preprocess_topology_20260906/cache`。
指纹 `5b5ed000c547f8048659c0fcdbb3090d8cdde3d6b6d26f638c30ad8007f2c913`；几何参与条目 key，49 与 121 帧缓存隔离。
RL 使用 `preprocessing_cache_mode=readonly`，不加载 tokenizer / 冻结 Qwen 编码器。
VAE 仅解码时移入 GPU，随后返回 CPU；在线 RL 仍运行视频解码和 PickScore，不能将其表述为全流程 GPU 只运行 DiT。

## FlowGRPO 逐 timestep 反传修正

对照 `/root/Flow-Factory/src/flow_factory/trainers/rl/grpo.py` 和 `nft.py`：
UniRL DiffusionNFT 已逐步 backward；FlowGRPO 原实现收集 K 个独立计算图后一次 backward，已修正为
`replay([t]) → loss_t → backward(loss_t * loss_scale / K)`。
梯度为 `loss_scale / K × Σ_t ∇loss_t`，保持跨 timestep 平均与原 optimizer step 边界，期间不清梯度。
五步来自 `[0,1,2,3,4]` 五个 SDE timestep；不对生成轨迹做跨步链式反传。
Replay anchor 也逐步计算；仅保留 detached log-prob 以汇总诊断指标。
48 组 CPU 等价验证通过，覆盖梯度、loss、指标、KL、不同 K / batch / loss_scale，实际 forward/backward 严格交替。
已完成的修正后 GPU 结果见历史基线；121 帧全部使用修正后的实现。

## 测量方法与边界

- 8 节点、64 张 H20；所有节点参与预处理与拓扑测试，最终联合 smoke / 训练使用完整 64 GPU。
- 生产采样：30 步，eta=0.5，前五步 SDE、其余 ODE、shift=3，同组共享初始噪声；PickScore 均匀四帧。
- LoRA r64 / alpha256，lr=2.5e-5，microbatch=1，每批一次 optimizer step。
- 拓扑筛选关闭 W&B、媒体上传和 checkpoint 保存；最终联合 smoke 开启 W&B 指标记录，保存真实更新后的 checkpoint。表中完整迭代是 `step_time_s`，不包含启动加载及迭代后 checkpoint 写盘。
- 每个候选完成两次真实更新；第二轮完整迭代衡量吞吐，包含采样、奖励、训练等阶段；启动加载不计入稳定吞吐，完整作业 wall time 保留在 JSON。
- 单节点 CP≥2 时 `groups=8/CP, group_size=CP`，每轮 8 视频；最终 `num_groups=32, group_size=16`，每轮 512 视频。
- rollout/replay log-prob 差异阈值 1e-2，超阈值报错；不静默换用 replay anchor 掩盖不一致。
- 表中 Train 是 `stack.train_track` 的 driver 计时，包含远程调用、轨迹本地化/传输和训练；不是纯 backward kernel 时间。完整迭代还包含参数同步、模型搬运、奖励等阶段。
- nvidia-smi 每两秒采样，报告全程最大显存；两个更新只能做配置筛选，小幅差距不视为统计显著。
- CP/EP/HSDP 节点内分组并不消除跨节点数据并行梯度同步。
- 当前 Leo2/FSDP 不支持有效 TP>1；历史 TP8 请求退化 TP1，不能记为有效 TP benchmark。
- rollout prompt scatter 要求 groups 被 rollout DP 整除；全 64 GPU colocated 下 CP1 不满足 32 groups 的约束。

## 121 帧实测结果（节点内及跨节点）

| Case | GPU | CP / EP | 检查点 | 更新数 | Rollout / Train / 完整迭代 (s) | 视频/s/8 GPU | 峰值 GiB | 状态 |
|---|---:|---|---|---:|---|---:|---:|---|
| `cp16_ep16_ac0_w16_cross_production121` | 16 | 16 / 16 | 关 | 2 | 1335.0 / 477.0 / 1848.9 | 0.00433 | 82.84 | 通过 |
| `cp16_ep16_ac0_w16_isolated_r_cp2_ep1_production121` | 16 | 16 / 16 | 关 | 0 | — / — / — | — | 16.26 | 退出 1 |
| `cp16_ep16_ac0_w16_isolated_r_cp2_ep1_production121_v2` | 16 | 16 / 16 | 关 | 0 | — / — / — | — | 49.56 | 退出 1 |
| `cp16_ep16_ac0_w16_isolated_r_cp2_ep1_production121_v3` | 16 | 16 / 16 | 关 | 2 | 844.5 / 481.7 / 1362.9 | 0.00587 | 89.09 | 通过 |
| `cp16_ep16_ac1_w16_cross_production121` | 16 | 16 / 16 | 开 | 2 | 1331.6 / 653.7 / 2022.6 | 0.00396 | 46.86 | 通过 |
| `cp16_ep8_ac0_w16_cross_production121` | 16 | 16 / 8 | 关 | 2 | 1268.1 / 459.5 / 1764.5 | 0.00453 | 87.96 | 通过 |
| `cp2_ep1_ac1_production121` | 8 | 2 / 1 | 开 | 2 | 844.3 / 518.1 / 1365.6 | 0.00586 | 62.48 | 通过 |
| `cp2_ep4_ac1_production121` | 8 | 2 / 4 | 开 | 2 | 861.2 / 585.3 / 1449.7 | 0.00552 | 95.00 | 通过 |
| `cp2_ep8_ac1_production121` | 8 | 2 / 8 | 开 | 2 | 888.6 / 586.8 / 1478.6 | 0.00541 | 95.00 | 通过 |
| `cp4_ep1_ac1_production121` | 8 | 4 / 1 | 开 | 2 | 908.7 / 525.7 / 1441.4 | 0.00555 | 50.62 | 通过 |
| `cp4_ep4_ac1_production121` | 8 | 4 / 4 | 开 | 2 | 913.2 / 527.3 / 1447.5 | 0.00553 | 84.76 | 通过 |
| `cp4_ep8_ac1_production121` | 8 | 4 / 8 | 开 | 2 | 935.1 / 535.8 / 1477.9 | 0.00541 | 72.96 | 通过 |
| `cp8_ep1_ac0_production121` | 8 | 8 / 1 | 关 | 0 | — / — / — | — | 95.00 | OOM |
| `cp8_ep1_ac1_production121` | 8 | 8 / 1 | 开 | 2 | 1029.1 / 558.5 / 1602.7 | 0.00499 | 51.32 | 通过 |
| `cp8_ep8_ac0_production121` | 8 | 8 / 8 | 关 | 0 | — / — / — | — | 95.00 | OOM |
| `cp8_ep8_ac1_production121` | 8 | 8 / 8 | 开 | 2 | 1047.8 / 564.2 / 1627.2 | 0.00492 | 55.62 | 通过 |

CP8/EP8/检查点开为延长监督时间而替换了监督进程，训练 driver 未重启。该案例 GPU 峰值的采样窗口从替换时刻开始；退出成功由完整两次更新日志、有限梯度及 driver 退出联合验证，不冒称获得原父进程的 OS wait 返回码。

首轮八个节点内案例已完成；追加 CP8/EP1 开关检查点、跨节点 CP16/EP16 开关检查点，以及 rollout CP2/EP1 + training CP16/EP16 的同卡独立进程对照。随后利用空闲两节点补充 CP16/EP8/无检查点，检验 EP 留在节点内的通信取舍。初始化失败也保留记录，不计为性能结果。

## 跨节点与训推分离

独立 overlay `deep_ep 1.2.1+R03C03` / NVSHMEM 已在八节点验证，16 GPU 真实 dispatch/combine 往返数值检查通过，max error=0；该微基准往返约 0.733 ms，不能外推完整模型通信成本。
通信微基准与完整模型测试分开判断；CP16/EP16 无检查点现已通过两次真实 121 帧更新，第二轮完整迭代 1848.9 秒、峰值 82.84 GiB。其按每 8 GPU 归一化的完整吞吐仍低于当前 CP2/EP1 开检查点基线。
节点使用唯一 hostname，避免 NVSHMEM 错认物理节点。

CP16/EP16 的 121 帧开关检查点对照已完成两轮，下面取第二轮；每轮 16 视频、使用 16 GPU。

| 梯度检查点 | Rollout (s) | Train 阶段 (s) | 完整迭代 (s) | 峰值 GiB |
|---|---:|---:|---:|---:|
| 开 | 1331.6 | 653.7 | 2022.6 | 46.86 |
| 关 | 1335.0 | 477.0 | 1848.9 | 82.84 |

关闭检查点将训练阶段耗时降低约 27%，完整迭代耗时降低约 8.6%，代价是约 36 GiB 额外峰值显存。


FSDP `hybrid8` 冷启动原先在进程组未初始化时返回空 mesh，可能意外采用全局 mesh。
已改为先初始化进程组，再创建显式二维 mesh；CPU 验证通过，新的 16 GPU 实测已确认实际 mesh 为 2×8，且每个 8 卡分片组均在单节点内。

新增原生 `Leo2RolloutEngine` 和 [独立 rollout 配方](../../../leo2_t2v_flowgrpo_separate.yaml)，
在不同 GPU 进程组中分别设置 rollout / training 的 CP、EP，并通过完整 LoRA 发布及校验同步参数。
这是原生 Leo2 sampler；没有声称已有 vLLM-Omni Leo2 adapter。
CPU adapter 写入/回读和配置测试通过；同卡独立进程模式现已完成真实 16 GPU 两次更新，包含参数发布、回读校验和跨拓扑 replay 检查。
独立 slab 会使同步训练流程的部分 GPU 在某阶段空闲，最终比较必须计入参数同步、轨迹传输和完整 wall time。
不能仅用 rollout 最快 + backward 最快拼出未经测量的总吞吐。

## 同卡独立进程的训推分离

新增 [同卡独立进程配方](../../../leo2_t2v_flowgrpo_colocated.yaml)：两组 worker 使用相同物理 GPU，分别保留自己的 CP/EP 进程组。
`workers_per_device=2`、`gpu_store` 传输、训练 offload 和 sampler sleep 配合使用；仍同步执行，不引入策略滞后。
Sampler 构建前先将训练权重移到 CPU；运行中先提取 LoRA、offload 训练，再唤醒 sampler 并发布 LoRA；生成后先让 sampler 休眠，再恢复训练。
CPU 已验证相同设备/不同 slot、采样成功和异常时的恢复顺序、重复 sleep/wake 及 onload 失败恢复。
实际 16 GPU 对照为 rollout CP2/EP1 + training CP16/EP16/无检查点。首版因 EP1/DeepEP 配置冲突退出；v2 已生成视频，但返回轨迹时遇到空 CUDA tensor 的 IPC handle 缺失。
已修复 GPU Store 空 tensor 的写入/读取边界，真实两 GPU 的 IPC、NCCL、重复引用和释放测试全部通过（`gpu_store_empty_verification.json`）。
v3 已通过两次真实更新，并在第一次采样前额外执行 LoRA 发布校验；第二轮同步后的最大 rollout/replay log-prob 差异为 2.304e-5，clip fraction=0，梯度有限且非零。

| 模式 | GPU / 视频数 | Rollout CP/EP | Train CP/EP | Rollout / Train (s) | 其他阶段 (s) | 完整迭代 (s) | 峰值 GiB |
|---|---:|---|---|---|---:|---:|---:|
| 共享模型，检查点开 | 8 / 8 | 2/1 | 2/1 | 844.3 / 518.1 | 3.2 | 1365.6 | 62.48 |
| 共享模型，检查点关 | 16 / 16 | 16/16 | 16/16 | 1335.0 / 477.0 | 36.9 | 1848.9 | 82.84 |
| 同卡独立进程，检查点关 | 16 / 16 | 2/1 | 16/16 | 844.5 / 481.7 | 36.7 | 1362.9 | 89.09 |

各行都是每 GPU 一条视频的基准工作量；“其他阶段”是完整迭代减去两个已计时阶段，包含奖励、参数同步、模型搬运等，不能全部归为通信。
组合拓扑明显改善了共享 CP16/EP16 的整体效率，但与共享 CP2/EP1 只差约 0.2%，不足以证明吞吐提升，且峰值显存高约 26.6 GiB。
最终 512 视频批量可能摊薄固定搬运成本；当前小批量拓扑筛选并未实测这个摊薄幅度，不能直接声称任一拓扑在所有批量上最优。

## 条件一致性与采样抽查

已在实际加载 DiT 的 8 GPU 进程中执行原生在线文本编码，并与离线缓存比较。
抽查 prompt 为 `white glowing smoke on a blue background`；每个 rank 的 10 个条件 tensor（文本状态、IDs、mask、索引），合计 80 次比较全部逐值一致，max abs diff=0，元数据也一致。此结论覆盖一个 prompt、8 个 rank。
抽帧最初看起来较抽象；核对提示词后，不能把这种外观直接认定为生成错误。
30 步 FlowSDE、30/50 步 ODE 和 30 步 CPS 四个对照均已完成，所有 rank 的结果均为有限值。FlowSDE 首次运行含冷启动开销，因此不将这些诊断耗时用于 sampler 性能排名；生产测试继续使用原定 30 步 FlowSDE。

| 单 prompt 采样诊断 | 最大 rank 耗时 (s) | 有限值检查 |
|---|---:|---|
| FlowSDE 30 步（首次运行） | 155.721 | 8/8 通过 |
| ODE 30 步 | 128.602 | 8/8 通过 |
| ODE 50 步 | 201.906 | 8/8 通过 |
| CPS 30 步 | 128.910 | 8/8 通过 |

诊断的 121 帧几何、同一 prompt 和初始噪声固定；单样本不能构成生成质量评估集。详见 `quality121_verification.json`、`quality_collected/`。

探针前两次因 device 参数未解析、重复构建原生 VAE 单例而失败；第三版复用实际 DiT/VAE，保留原始诊断日志。

基准使用固定 128 条 prompt 文件 `smoke64.txt`（文件名是历史命名，实际为 128 条）；所有几何均为 121 帧。

## 证据与交付

全部运行原始日志和脚本在仓库下 `outputs/leo2/topology_preprocess_20260906/`（gitignored）。
`benchmark_results.json` 包含逐轮时间和算法诊断；`collected/<node>/` 保留日志与 GPU 遥测。
`benchmark121_launch.json` 记录八节点启动前代码 checksum drift=0 和相同 git HEAD。
`stepwise_flowgrpo_verification.json`、`rollout_mesh_verification.json` 是 CPU 验证结果。
`stopped_run.json` 记录已停止的旧 NFT 作业；`runtime_verification.json` 是八节点 runtime 验证。

[架构 / SFT / 推理报告](../../ARCHITECTURE_SFT_INFERENCE_REPORT.md) · [生产训练配方](../../../leo2_t2v_flowgrpo_cached.yaml)。

## 完整 64 GPU 运行状态

已启动 `leo2_flowgrpo_cached_64gpu_smoke_0906_1613`，实际 64 rank / 64 物理 GPU、8×8 mesh、原生 CP2/EP1 均通过日志核验。
完整批量为 32×16=512 视频、464×848×121；优化器更新与 checkpoint 保存仍在等待。
[W&B 运行](https://wandb.ai/315229706-xi-an-jiaotong-university-/leo2-rl/runs/81febwid)。
初次显存采样在 Ray 启动期间退出，已于 16:27 恢复；后续峰值不含初始化阶段。
逐 rank mesh 核验使用 worker 原始日志；附加进度环境变量已在后续训练启动器中显式传入 Ray job。
证据：`smoke64_init_verified.json`、`gpu64_monitor_restart.json`。

完整批量 rollout 已于 18:12 返回，耗时 6819.610 秒；奖励在 18:14 完成，32 个 group 的奖励标准差均非零。
30 GB/节点的 Ray 对象存储在交接时累计溢写约 498 GiB；八节点可用主存均超过 1.2 TiB，`/dev/shm` 均超过 462 GiB。
正式训练启动器已配置 128 GiB/节点对象存储，仅在 smoke checkpoint 验证后重启 Ray 生效；尚未测得这项调整的耗时改善。
证据：`object_store_capacity_verified.json`。
