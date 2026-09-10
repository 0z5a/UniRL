# Leo2 packed sequence：Flow-GRPO forward benchmark

日期：2026-09-10

状态：GPU correctness、parity、B1 同节点回归对照和 B1/B2/B4 吞吐测试均已完成。

## 结论

- Leo2 的统一 B=1/B>1 packed forward 已在正式 Flow-GRPO T2VA 路径跑通。R1 中 packed B2/B4 相对 packed B1 的 rollout 模型吞吐分别提高 6.0%/8.9%，replay 模型吞吐分别提高 6.8%/12.1%。
- 在同一节点、同一 checkpoint/assets/cache 上背靠背比较旧 dense HEAD B1 与统一 packed B1。合并两轮 raw wave 后，packed B1 的 rollout/replay 模型吞吐分别高 0.83%/1.31%，lifecycle 完成率高 0.91%。这个幅度接近短测噪声，不作为额外加速主张；它支持“统一 timestep/packed B1 路径没有复现此前性能回退”的结论。
- 双 rollout 的 R2 排除首次初始化影响后，packed B2 相对 packed B1 的第二轮 rollout 模型吞吐提高 5.7%--6.0%，replay 提高 6.4%--6.7%；B4/U1 和合法 local8 B4/U2 的 rollout 均提高约 7.8%，replay 提高 12.5%--12.8%。
- B1/U1、B2/U1 和 B4/U1 的 on-policy `ratio/min/max` 均为 `1.0000/1.0000/1.0000`，`|Δlogp|` 为精确 0。U2 的第一个 optimizer update 也通过 threshold=0 的逐 micro 硬 gate；第二个 update 已在参数更新后执行，属于预期的 off-policy replay，允许出现微小偏差。
- R2 的 B4/U2 发现了真实的数据编排缺陷：rollout 使用一个 B4 pack，旧 CountPlanner 却把它拆成两个 B2 replay micro，严格 gate 立即检测到 `max |Δlogp|=2.38419e-7`。provenance 修复记录每次 rollout forward 的精确 pack，并要求 replay 保持 pack 内内容、顺序和边界；整 pack 仍可重排。修复后的 B4/U1 和每个 DP shard 含两个完整 pack 的 B4/U2 均完成两轮正式训练并通过 strict on-policy gate；单 pack B4/U2 会 fail fast。
- packed width 不改变统计样本数。等规模 case 每轮的 16 条 trajectory、local8 B4/U2 每轮的 32 条 trajectory 都只生成和消费一次，每条包含 12 个 rollout step 和 2 个训练 SDE transition。当前结果证明的是更高的计算吞吐和样本使用完整性；两轮 smoke 不足以判断达到同一 reward 所需样本数，因此不能宣称统计意义上的 reward/sample efficiency 已提高。
- 正式 Flow-GRPO 本轮使用同构的 H/W/F 和 audio length。CP2 tiny-model GPU 测试已经验证一个 pack 内两个不同 video shape 的输出与各自 B1 精确一致，且反向梯度有限；完整 trajectory、SDE、decoder 和 condition pipeline 仍有矩形 batch 假设，异构分辨率、帧数和任务混训尚未完成端到端验证。

B2/B4 的主表加速以 packed B1 为基线，衡量增大 forward pack 的收益。dense HEAD 与 packed B1 的同节点对照单独衡量统一 B=1/B>1 路径是否引入 B1 回退，不能混用两个基线。

## 测试配置

- 单 case 使用一个节点的 8× NVIDIA H20（每卡 97,871 MiB）；PyTorch 2.7.1、Ray 2.56.0。
- Leo2 MoE-A12B，bf16，LoRA r128/alpha256，FSDP `hybrid8`，activation checkpointing；CP=2、EP=1，因此每个作业有 4 个有效 DP group。
- checkpoint 为节点系统盘上的 `/root/leo2-iter0063300-local/weights`。DCP 主文件大小 150,508,043,106 bytes，SHA-256 为 `ac5570875348a9e3619f0a53650f88d3a829c400a38e5fe47b7a4e3907ae0c71`。视频和音频 VAE 资产也复制到系统盘，避免训练期间密集读取 Ceph。
- Flow-GRPO T2VA：`enable_audio=true`、`audio_joint_sde=true`，464×848、9 帧、12 个 rollout denoise step、2 个训练 SDE step，`old_logp_source=rollout`。
- 配置以 W&B run [`dvjwe80y`](https://wandb.ai/315229706-xi-an-jiaotong-university-/leo2-rl/runs/dvjwe80y) 为正式训练参考，并缩小为 smoke：每轮 4 个 prompt group × 每组 4 个样本，共 16 条 trajectory。DP scatter 后每个 DP group 连续取得 4 条；B1/B2/B4 分别将这 4 条切成 4/2/1 个 pack。
- parity 配置为 `max_rollout_replay_logp_absdiff=0` 和 `rollout_replay_parity_action=raise`；本轮使用 guidance=1、`inference_cache_method=none`，rollout/replay 使用相同的 `flash3_packed` attention backend。R1/R2 run 的 `metadata.txt` 保存实际工作树和关键源文件 SHA。

## 指标口径

### Raw 8-worker 同步模型指标

每个模型 rank 在 `predict_joint_noise` 中输出：

```text
[leo2 perf] fwd#N grad=<bool> bsz=<logical rows> packed=True tokens=<pre-CP tokens> \
  dt=<seconds> before=<GiB> after=<GiB> peak=<GiB> gap_peak=<GiB>
```

`dt` 在 `model(**model_inputs)` 前后执行 CUDA synchronize，只测 Leo2 模型 forward；不含 Python packing、condition/input 准备、SDE、VAE decode、backward 或 optimizer。同一 DP group 的两个 CP rank 会重复报告该次 pack。

本文直接解析 8 个原始 `worker-*.out`，不使用 driver stdout 中可能被 Ray 合并为 `[repeated Nx]` 的展示行。每个 `fwd#` 必须收齐 8 个 rank，以 rank 中最大的 `dt` 作为该同步 wave 的 wall time。逻辑 sample-step 数为 `sum(rank bsz) / CP`，再除以所有 wave wall time 之和得到 sample-step/s。rollout 0 的 steady 数字丢弃前两个 warm-up wave；rollout 1 已经过整轮 warm-up，因此使用全部 wave。replay 使用全部 wave。所有有效 R1/R2 case 均无缺失 wave。

这个 sample-step/s 是一次 denoise/replay model call 所推进的逻辑样本数，不等于完整 trajectory/s。rollout 每条 trajectory 有 12 个 step，replay 有 2 个训练 step。

### Driver lifecycle 指标

`rollout.generate` 和 `stack.train_track` 是 driver 记录的完整阶段 wall time，分别包含 model forward 之外的 rollout/SDE/decode 和 replay/backward/optimizer 开销。报告中的 lifecycle stage sum 是 `generate + reward.score_and_attach + score finalize + train_track`；由本轮 trajectory 数除以 stage sum 得到每秒完成的 trajectory 数。除 local8 B4/U2 每轮使用 32 条以外，其余 case 均为 16 条。它是所列训练阶段的计算完成率，不包含进程启动、checkpoint 加载和未记录的调度间隙，也不是统计 reward/sample efficiency。

### 显存指标

`peak` 是该 model forward 内的 PyTorch allocated peak。`gap_peak` 是上次 reset 以后、当前 forward 开始前的峰值；训练阶段它可能覆盖中间的 backward/optimizer，因此只能作为这些操作峰值的下界。二者均不是 `nvidia-smi` 的整卡 used/reserved 值。

## R1：单 rollout 正式 smoke

run 前缀：`20260910_110042_*`。四个 case 都退出 0，且所有 forward 行均为 `packed=True`。

| case | pack / updates | rollout waves | steady p50 (s/wave) | rollout sample-step/s | vs B1 | rollout token/s | replay waves | replay p50 (s/wave) | replay sample-step/s | vs B1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1/U1 | 1 / 1 | 48 | 4.440 | 0.8748 | 1.000× | 18.17k | 8 | 4.660 | 0.8349 | 1.000× |
| B2/U1 | 2 / 1 | 24 | 8.600 | 0.9274 | 1.060× | 19.26k | 4 | 8.855 | 0.8916 | 1.068× |
| B2/U2 | 2 / 2 | 24 | 8.570 | 0.9197 | 1.051× | 19.10k | 4 | 8.665 | 0.9132 | 1.094× |
| B4/U1 | 4 / 1 | 12 | 16.800 | 0.9526 | 1.089× | 19.79k | 2 | 17.100 | 0.9357 | 1.121× |

| case | `rollout.generate` (s) | reward + finalize (s) | `stack.train_track` (s) | lifecycle stage sum (s) | 16 trajectories/s | vs B1 |
|---|---:|---:|---:|---:|---:|---:|
| B1/U1 | 268.689 | 4.377 | 139.3 | 412.366 | 0.03880 | 1.000× |
| B2/U1 | 257.782 | 4.370 | 134.5 | 396.652 | 0.04034 | 1.040× |
| B2/U2 | 258.052 | 4.443 | 132.7 | 395.195 | 0.04049 | 1.044× |
| B4/U1 | 249.956 | 4.201 | 132.6 | 386.757 | 0.04137 | 1.066× |

R1 是单次冷启动测试，适合验证路径并估计量级。B4/U1 的模型 forward 增益高于 lifecycle 增益，因为 packing 之外的 condition、SDE、decode、backward 和 optimizer 仍需付费。

### R1 parity

| case | on-policy gate | driver 汇总 |
|---|---|---|
| B1/U1 | threshold=0，逐 micro 通过 | ratio/min/max = 1.0000/1.0000/1.0000；mean/max \|Δlogp\| = 0/0 |
| B2/U1 | threshold=0，逐 micro 通过 | ratio/min/max = 1.0000/1.0000/1.0000；mean/max \|Δlogp\| = 0/0 |
| B4/U1 | threshold=0，逐 micro 通过 | ratio/min/max = 1.0000/1.0000/1.0000；mean/max \|Δlogp\| = 0/0 |
| B2/U2 | update 0 threshold=0，逐 micro 通过 | 两个 update 的汇总 mean/max \|Δlogp\| = 1.51e-6/4.14e-6 |

U2 的 update 0 是 on-policy，硬 gate 只在该 update 激活；作业退出 0 证明所有受检 micro 的最大误差为精确 0。update 1 在一次 optimizer step 后执行，属于 off-policy。当前 TrainStack 会对多个 micro/update 的 numeric metrics 求均值，因此 U2 stdout 中以四位小数显示的 ratio/min/max 以及汇总 `max` 不是跨 update 的真实极值；逐 micro 的 threshold=0 gate 才是 on-policy parity 的判据。

### R1 显存

| case | pre-rollout allocated (GiB) | rollout forward peak (GiB) | replay forward peak (GiB) | run 内最大 `gap_peak` (GiB) | final replay `after` (GiB) |
|---|---:|---:|---:|---:|---:|
| B1/U1 | 23.7 | 33.9 | 38.2 | 39.6 | 29.2 |
| B2/U1 | 23.7 | 36.2 | 44.6 | 48.9 | 33.8 |
| B2/U2 | 23.7 | 36.2 | 44.9 | 49.0 | 34.1 |
| B4/U1 | 23.7 | 41.6 | 58.6 | 67.1 | 43.0 |

视频 VAE 在 CPU；日志中的音频 VAE GPU 参数约占 2.3 GiB。四组均无 OOM。B4 的吞吐收益伴随明显的 replay/backward 显存增长，需要在大分辨率、长帧和异构 pack 上继续测量。

## R2：双 rollout 稳态复测

R2 使用相同的 8-GPU/CP2 配置，每个 case 连续运行两个 rollout。B1/U2、B2/U1、B2/U2、修复后的 B4/U1 和合法 local8 B4/U2 均退出 0。

### Raw worker 模型吞吐

| case | rollout 0 steady p50 (s) | rollout 0 sample-step/s | rollout 0 replay sample-step/s | rollout 1 p50 (s) | rollout 1 sample-step/s | rollout 1 replay p50 (s) | rollout 1 replay sample-step/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1/U2 | 4.430 | 0.8761 | 0.8165 | 4.410 | 0.8843 | 4.505 | 0.8340 |
| B2/U1 | 8.570 | 0.9317 | 0.8837 | 8.520 | 0.9348 | 8.830 | 0.8874 |
| B2/U2 | 8.600 | 0.9223 | 0.9127 | 8.510 | 0.9372 | 8.735 | 0.8901 |
| B4/U1 fixed | 16.780 | 0.9532 | 0.9201 | 16.770 | 0.9538 | 17.060 | 0.9379 |
| B4/U2 fixed, local8 | 16.790 | 0.9478 | 0.9438 | 16.710 | 0.9534 | 16.940 | 0.9403 |

以 rollout 1 的 B1/U2 为基线，B2/U1 的 rollout/replay sample-step/s 分别提高 5.71%/6.41%，B2/U2 分别提高 5.99%/6.73%；B4/U1 分别提高 7.86%/12.46%，B4/U2 分别提高 7.81%/12.75%。合法 B4/U2 的 raw replay 行全部为 `bsz=4`，证明 CountPlanner 没有再拆分 rollout pack。B4/U1 与 B4/U2 的第二轮 raw 吞吐相近，后者主要用于验证两个完整 pack 在两个 update 间的正确编排。

### Driver lifecycle

| case | rollout | trajectories | generate (s) | reward + finalize (s) | train (s) | stage sum (s) | trajectories/s | vs 同轮 B1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1/U2 | 0 | 16 | 267.815 | 4.400 | 138.4 | 410.615 | 0.03897 | 1.000× |
| B2/U1 | 0 | 16 | 255.804 | 4.358 | 132.8 | 392.962 | 0.04072 | 1.045× |
| B2/U2 | 0 | 16 | 258.823 | 4.406 | 133.0 | 396.229 | 0.04038 | 1.036× |
| B4/U1 fixed | 0 | 16 | 249.480 | 4.147 | 129.9 | 383.527 | 0.04172 | 1.071× |
| B4/U2 fixed, local8 | 0 | 32 | 477.906 | 12.120 | 247.9 | 737.926 | 0.04336 | 1.113× |
| B1/U2 | 1 | 16 | 247.069 | 3.616 | 127.4 | 378.085 | 0.04232 | 1.000× |
| B2/U1 | 1 | 16 | 236.231 | 3.603 | 122.3 | 362.134 | 0.04418 | 1.044× |
| B2/U2 | 1 | 16 | 236.213 | 3.458 | 122.7 | 362.371 | 0.04415 | 1.043× |
| B4/U1 fixed | 1 | 16 | 229.056 | 3.486 | 119.3 | 351.842 | 0.04548 | 1.075× |
| B4/U2 fixed, local8 | 1 | 32 | 458.462 | 10.605 | 237.8 | 706.867 | 0.04527 | 1.070× |

在 rollout 1，B2/U1 相对 B1 的 generate/train wall time 分别缩短 4.39%/4.00%；等价的阶段吞吐提高 4.59%/4.17%。B2/U2 的 generate/train 阶段吞吐提高 4.60%/3.83%。包含 reward 阶段的 lifecycle 完成率分别提高 4.40% 和 4.34%。B4/U1 和 B4/U2 的第二轮 lifecycle 完成率分别提高 7.46% 和 6.97%；B4/U2 将样本数翻倍后仍保持与 B4/U1 接近的归一化完成率。

### R2 parity 与显存

| case | on-policy gate | driver 汇总中的 off-policy 信号 | rollout forward peak (GiB) | replay forward peak (GiB) | max `gap_peak` (GiB) |
|---|---|---|---:|---:|---:|
| B1/U2 | 两轮 update 0 均以 threshold=0 通过 | 两轮汇总 \|Δlogp\| = 1.05e-6、5.96e-7 | 34.4 | 38.5 | 39.9 |
| B2/U1 | 两轮所有 replay 均精确为 0 | 两轮汇总 \|Δlogp\| = 0、0 | 36.7 | 44.8 | 49.0 |
| B2/U2 | 两轮 update 0 均以 threshold=0 通过 | 两轮汇总 \|Δlogp\| = 1.29e-6、1.27e-5 | 36.7 | 44.9 | 49.2 |
| B4/U1 fixed | 两轮所有 replay 均精确为 0 | 两轮汇总 \|Δlogp\| = 0、0 | 42.5 | 58.7 | 67.3 |
| B4/U2 fixed, local8 | 两轮 update 0 均以 threshold=0 通过 | 两轮汇总 \|Δlogp\| = 6.82e-7、6.82e-7 | 42.6 | 59.1 | 67.8 |

所有 driver 汇总的 ratio 都显示为 `1.0000±0.0000`。对 U2 case，这个四位小数显示包含 off-policy update，不能替代 update 0 的精确 gate 结果。

## CP2 异构 shape 模型边界验证

在两张 H20 上用 CP=2 和 `flash3_packed` 运行 tiny Leo2 T2VA 模型，将 video latent shape
`[1,8,13,29,53]` 与 `[1,8,9,20,32]` 放入同一个物理 packed row，并给两个样本分别传入
timestep 250 和 750。随后分别以 B1 执行相同输入作为参照，并对 packed 输出执行一次反向：

| 项目 | 结果 |
|---|---:|
| video max abs diff vs corresponding B1 | 0.0 |
| audio max abs diff vs corresponding B1 | 0.0 |
| video gradients finite | true |
| audio gradients finite | true |
| PyTorch allocated peak | 0.1751 GiB |

这项测试覆盖 ragged video 拆分、不同 per-sample timestep、CP scatter/gather、FA3 segment 隔离和
video/audio 联合反向。audio length 在这个 GPU case 中相同；不同 audio length 的 ragged contract 由
CPU 单测覆盖。tiny model 只验证 model-forward 边界，不代表当前 pipeline 已能构造和训练完整的异构 trajectory。

## B4/U2 pack split：发现、原因和修复

R2 case `20260910_112714_steady_b4_u2_r2_node253` 在第一个 replay micro 因 strict gate 退出 1：

```text
FlowGRPO rollout/replay parity failed: mean/max |Δlogp|=
1.19209e-07/2.38419e-07; expected finite values and max <= 0
```

rollout 的 raw worker 显示 `bsz=4`，但旧 CountPlanner 为两个 optimizer update 将 DP-local `[0,1,2,3]` 拆成两个 replay B2 micro。由于 packed attention 的浮点执行不保证跨 pack composition 的 bitwise batch invariance，虽然每个 sample 的内容和相对顺序没变，B4 rollout 与 B2 replay 仍产生了约 1e-7 的差异。这个失败与 attention 隔离或 timestep repeat 的计算错误无关；它定位到 rollout/replay pack composition 不一致。

当前工作树中的修复：

1. Leo2 pipeline 在每次实际 forward 后，将该 pack 的 sample-id tuple 写入每个输出 row 的 `forward_pack_sample_ids`。
2. CountPlanner 发现该 provenance 后，以记录的 forward pack 作为不可拆分的 replay micro；它验证 pack 内 sample 内容、顺序和边界，并拒绝被截断、篡改或超过 micro-batch 上限的 pack。
3. 完整 pack 可以连同 provenance 在 batch 之间重排。planner 按重排后的 pack 顺序生成 plan，不要求全局 replay 顺序与 rollout 相同。
4. 一个 B4 pack 无法分给两个非空 update，因此 B4/local-batch=4/U2 现在会明确报错。合法 B4/U2 需要每个 DP shard 至少两个完整 B4 pack，例如 local batch=8，让每个 update 消费一个 pack。
5. 没有 provenance 的其他模型继续使用原 CountPlanner 行为。当前 Leo2 recipes 固定使用 CountPlanner。

相关 focused CPU tests 已覆盖 B2/U2、单 B4/U2 fail-fast、两个 B4 pack/U2、完整 pack 重排、marker 篡改/截断、无 marker 的兼容路径和 Leo2 pipeline marker 写入。修复后又完成两项正式 GPU 验证：B4/U1 每个 DP shard 一个完整 pack，两轮 rollout/replay 的 `|Δlogp|` 均为精确 0；B4/U2 每个 shard 两个完整 B4 pack、每个 update 一个，两轮 on-policy update 均通过 threshold=0 gate，且所有 replay raw 行保持 `bsz=4`。两个作业都退出 0。

## 样本使用与统计效率

R1/R2 的 B1/B2/B4 有相同的样本语义：等规模 case 每轮生成 16 条唯一 trajectory；local8 B4/U2 为了给每个 DP shard 提供两个完整 B4 pack，每轮生成 32 条。每条 trajectory 只出现在一个 DP shard 中，并只在一个 update 中消费一次。`num_updates_per_batch=2` 不表示整批 replay 两次；它把 DP-local 样本划分到两个连续 update，所以第二组在第一次 optimizer step 后成为 off-policy。每条 trajectory 的两个 SDE transition 在所属 update 内各执行一次。

因此可以报告两类已验证结果：

- **计算吞吐**：在相同 trajectory 数、denoise step 和 SDE step 下，B2/B4 每秒推进更多 sample-step，也缩短了 lifecycle wall time。
- **样本使用完整性**：没有因为 pack width 复制、丢弃或重复消费 trajectory；provenance 修复进一步保证 replay 不拆开 rollout pack。

当前不能报告“同等 reward 使用更少训练样本”。R2 只有两个 rollout，且 U2 第二个 update 本身是 off-policy。统计 reward/sample efficiency 需要固定数据和随机种子、足够长的多次训练曲线，并比较达到预设 reward/quality 阈值所需的唯一 trajectory 数。

## 异构训练支持边界

本次 GPU benchmark 使用同一 H/W/F 和同构 audio length。当前 direct Leo2 model-forward boundary 可以接收 per-sample ragged video/audio list：输入按 sample 顺序拼接，video/audio branch 分别执行 CP padding 和 block-diagonal FA3 mask，再按原顺序拆分输出。

端到端异构训练仍受以下矩形假设限制：pipeline condition stage 使用一组共享 H/W/F；initial noise、trajectory、`LatentSegment`、SDE replay 和 VAE decoder 仍依赖 Tensor stack；AV rollout 还要求 pack 内 audio token length 一致。现阶段可审查的承诺是 ragged model forward 和同构正式 Flow-GRPO packed training，而不是不同分辨率、帧数、audio length 或任务的完整混训。

动态 token-budget packing 还必须保证各 train-DP rank 在每个 optimizer update 中执行相同数量的
replay micro-forward。当前固定 FBS 的正式配方在各 rank 上具有相同 pack 数，不触发这个问题；未来若
不同 rank 得到不同数量的完整 pack，需要在进入 FSDP forward 前做一次训练组级的一致性检查并统一失败，
不能通过拆包或合包来补齐 collective 次数。

Leo2 原生任务集合包括 T2I、T2V、T2VA、I2VA、FL2VA 和 T2A。要实现跨任务混训，还需把 task-specific condition/media、noise/trajectory 表示、SDE 和 decode 边界改为 per-sample/ragged，并在 pack mask 和输出拆分后分别测试每种 branch 组合。

## 已验证前提与剩余限制

- 正式 GPU 测试使用当前 `leo-2-moe-v1-1-pack-noTR` checkpoint，其 text refiner 是 linear text projection。`single_refiner` 的 packed B>1 路径尚未支持，因此本报告不覆盖该变体。
- bitwise on-policy parity 的已验证条件是 guidance=1、inference cache 关闭，并且 rollout 与 replay 使用相同的 packed-attention backend。其他 guidance/cache 组合或在训推之间更换 attention backend 需要单独验证，不能从本报告直接外推 exact-zero 结论。
- final layer 仍会对 CUDA 上的 `token_lengths` 执行一次 `.tolist()`，引入低量级 host synchronization。它已包含在本轮实测路径中，不影响 correctness/parity 结论；可以作为后续小优化移除或提前物化，但预计不是当前 B1/B2/B4 差异的主要来源。

## Dense HEAD 与统一 packed B1：同节点对照

为隔离节点差异，在 `node-28-7-193-171` 上依次运行 dense HEAD 和最终统一 packed B1。两侧都使用同一份系统盘 checkpoint、VAE assets、readonly preprocessing cache、8×H20、CP2、FBS=1、MBS=1、U2、4 groups × 4 samples、两个 rollout 和 strict-zero gate。dense 侧为未包含 packed 修改的 commit `8d524b7d396ba0c39dcf916e32fe6195c4469341`；packed 侧 metadata 保存最终源文件 SHA，测试源码总 digest 为 `14426844f3cd7d6276d8a05194d9083096a090d5188a9990f90cd767081bd6f6`。

对应 case 为：

- dense：`20260910_1208_dense_head_b1_u2_r2_localassets_node171`
- packed：`20260910_1228_packed_b1_u2_r2_samehost_node171`

两侧均退出 0，parser 均收齐 8 个 worker 的 896 行，没有 incomplete wave 或其他 warning。

### Raw model forward

| path | rollout | rollout p50 (s/wave) | rollout sample-step/s | replay p50 (s/wave) | replay sample-step/s | packed vs dense |
|---|---:|---:|---:|---:|---:|---:|
| dense HEAD B1 | 0 steady | 4.470 | 0.8885 | 4.915 | 0.8044 | baseline |
| unified packed B1 | 0 steady | 4.440 | 0.8952 | 4.665 | 0.8205 | rollout +0.75%；replay +2.00% |
| dense HEAD B1 | 1 | 4.450 | 0.8900 | 4.530 | 0.8226 | baseline |
| unified packed B1 | 1 | 4.420 | 0.8980 | 4.510 | 0.8277 | rollout +0.91%；replay +0.62% |

将 rollout 0 的 46 个 steady wave 与 rollout 1 的 48 个 wave 按总 sample-step/总 wall time 合并，dense/packed rollout 吞吐为 0.88927/0.89665 sample-step/s，packed 高 0.83%。将两轮各 8 个 replay wave 同样合并，dense/packed 为 0.81342/0.82411，packed 高 1.31%。两边的 token 数基本相同；结果不是通过降低计算量获得。

### Driver lifecycle

| path | rollout | generate (s) | reward + finalize (s) | train (s) | stage sum (s) | trajectories/s | packed vs dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense HEAD B1 | 0 | 271.660 | 4.619 | 141.3 | 417.579 | 0.038316 | baseline |
| unified packed B1 | 0 | 268.354 | 4.482 | 140.1 | 412.936 | 0.038747 | +1.12% |
| dense HEAD B1 | 1 | 250.321 | 3.532 | 129.0 | 382.853 | 0.041791 | baseline |
| unified packed B1 | 1 | 248.749 | 3.710 | 127.8 | 380.259 | 0.042077 | +0.68% |

合并两轮后，dense/packed 的 lifecycle 完成率为 0.039978/0.040343 trajectory/s，packed 高 0.91%。packed 的 generate 吞吐在两轮分别高 1.23%/0.63%，train 吞吐分别高 0.86%/0.94%。两侧 rollout/replay forward peak 都是 34.4/38.5 GiB，最大 `gap_peak` 都是 39.9 GiB。

两侧 U2 的两个 rollout 均通过 update 0 的 threshold=0 on-policy gate。dense driver 的两轮 off-policy 汇总 `|Δlogp|` 为 7.00e-7/5.32e-6，packed 为 1.03e-6/1.06e-6，均符合一次 optimizer step 后允许漂移的语义。

这是一次背靠背而非交错重复 A/B，约 1% 的差异可能包含时序噪声。可复现的结论是统一 packed B1 没有出现相对旧 dense B1 的性能或显存回退；不把 0.6%--2.0% 的单项正差异解释为稳定加速。

## 证据位置

- R1 driver logs 与 metadata：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/flowgrpo/20260910_110042_*`
- R2 driver logs 与 metadata：同一根目录下的 `20260910_112714_steady_*`
- provenance 修复后的 B4/U1 driver logs 与 metadata：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/flowgrpo/20260910_1158_fixed_b4_u1_r2_node253`
- provenance 修复后的 local8 B4/U2 driver logs 与 metadata：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/flowgrpo/20260910_1202_fixed_b4_u2_local8_r2_node250`
- 同节点 dense B1 driver logs 与 metadata：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/flowgrpo/20260910_1208_dense_head_b1_u2_r2_localassets_node171`
- 同节点 packed B1 driver logs 与 metadata：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/flowgrpo/20260910_1228_packed_b1_u2_r2_samehost_node171`
- 持久化 raw worker 归档、机器可读聚合、解析脚本和 `SHA256SUMS`：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/evidence/flowgrpo_forward_bench`
- R1 raw worker 摘录：`/tmp/leo2_r1_raw_20260910/{b1_u1,b2_u1,b4_u1,b2_u2}`
- R1 机器可读聚合：`/tmp/leo2_r1_{b1_u1,b2_u1,b4_u1,b2_u2}.json`
- R2 机器可读聚合：`/tmp/leo2_r2_{b1_u2,b2_u1,b2_u2,b4_u2_invalid}.json`
- provenance 修复后的 B4 聚合：`/tmp/leo2_r2_fixed_b4_u1.{json,txt}`、`/tmp/leo2_r2_fixed_b4_u2_local8.{json,txt}`
- 同节点 B1 A/B 聚合：`/tmp/leo2_r2_dense_head_b1_u2_samehost.{json,txt}`、`/tmp/leo2_r2_packed_b1_u2_samehost.{json,txt}`
- raw-worker 聚合脚本：`/tmp/parse_leo2_r2.py`
- CP2 ragged tiny-model 脚本、日志和源文件摘要：`/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/UniRL/leo2_runs/packed_recovery_20260910/evidence/cp2_ragged_tiny`

`/tmp` 下的派生文件用于交叉检查；共享 evidence 目录持久保存了每个 case 中实际含 perf 数据的 8 个 raw worker log、解析结果和校验清单，原始 driver log 与 metadata 保存在对应 run 目录。所有列入有效结果的 parser 输出都收齐预期 8 个 worker，且 `warnings=[]`。B4/U2 invalid case 保留为严格 parity gate 定位 pack split 的负例，不纳入有效性能比较。
