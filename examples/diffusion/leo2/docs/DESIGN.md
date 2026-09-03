# Leo2 × UniRL FlowGRPO 接入设计（2026-08-27 夜）

目标：Leo2-MoE-A12B 接入开源 UniRL，trainside 模式（同进程 rollout，无 vllm-omni、无权重同步），
FlowGRPO 跑通 t2v，最初验证于 32×H20 节点。

## 蓝本

- 框架副本：`code/UniRL-leo2`（源 = H3 实战版 UniRL-repro，含 σ-band/C2 归一化/共享 x_T 全部修复）
- 模型包模板：`unirl/models/minimax_h3/`（同为 packed 多模态视频 DiT）
- 配置模板：`examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml`（t2v 纯视频 trainside 全菜谱）
- H3 成品配方超参参考：LoRA r64/α256、lr 2.5e-5、6 σ-band transitions、2 updates/batch

## 已核实的关键事实

1. **UniRL 模型合同**：bundle（纯权重容器，`trainable_module()`）/ pipeline（`generate(sample)->sample`）/
   DiffusionStage（`generate()` 存轨迹+SDE logp → LatentSegment；`replay()` 重算 logp → ReplayResult）/
   Conditions（typed）。rollout 与训练共享同一 `bundle.transformer`（FSDP 原地改造）。
2. **符号约定**：FlowSDEStrategy 在 eta=0 时 `x_next = x + noise_pred·(σ_next−σ)`，与 Leo2
   FlowMatchDiscreteScheduler 的 Euler 完全一致 → **Leo2 的 `diffusion_prediction` 直接作 noise_pred，无需变号**
   （H3 的负号是它自己的数据向约定，勿抄）。
3. **Leo2 侧对接点**（全部在 leo_hf.py / pipeline_leo.py，已验证 6 模式推理可用）：
   - prompt → inputs：`model.prepare_model_inputs(prompt/message_list, use_system_prompt='li-dit-encode-visual-qwen-3.5', mode='gen_video', ...)`（leo_hf:946）
   - 文本编码：`pipeline.encode_prompt(model_kwargs)`（pipeline_leo:711，跑 Qwen3.5 得 cond_text_states）+ pop input_ids + attention_mask（:810-815）
   - 每步前向：`model.prepare_inputs_for_generation(input_ids, latents=…, timesteps=σ*1000, audio_latents=None, **model_kwargs)` → `model(**inputs)['diffusion_prediction']`（pipeline_leo:845-860）
   - t2v：无 audio latent、无 channel cond；CFG 用 guidance=1.0（单分支，logp 干净，抄 HV1 trainside）
4. **latent 几何**：unpacked (48, (F−1)/4+1, H/16, W/16)；packing 由 prepare_inputs_for_generation 内部处理，
   UniRL 侧 latent_shape 返回 unpacked 形状即可（与 H3 的显式 packing 不同，更简单）。
5. **权重**：`LEO2_CKPT_DIR` 指向 iter-0063300 native Torch DCP（约 150GB 单分片）。
   加载策略：meta-init 构模 → UniRL FSDPBackend 分片 → DCP 按需读取各 rank 分片（DCP ranged reads，单文件也高效）。
   具体挂钩方式等 FSDPBackend 侦察结论。
6. **LoRA**：只打视频/文本/音频分支的 attention 投影 + dense MLP（q/k/v/o_proj{,_txt,_audio}、gate_and_up/down_proj 非专家部分）；
   **不碰 64 个 MoE 专家**。r64/α256（HV1 验证 α=4r 才涨）。
7. **奖励**：先 VideoPickScoreScorer（PickScore 首帧，unirl/reward/local 现成）；ImageBind 已部署 4 pod 可后换。
8. **采样超参（smoke）**：256p 系（192×336 或 256×448）、帧数少（≤49）、10 步、eta 0.7、
   AllSDEScheduler timestep_fraction [0,0.5] num_sde_steps 4、samples_per_prompt 8、静态 shift 3.0（弃 flux 动态 shift）。

## 计划中的产出

- `unirl/models/leo2/`：bundle.py / config.py / conditions.py / text_embed.py / diffusion.py / pipeline.py / vae.py，
  并在 `vendor/gen_ar` 内携带 hymm、hy_parallelism、IndexKits 与 processors 的运行闭包。
- `examples/diffusion/leo2/leo2_t2v_trainside.yaml`
- 启动脚本（4 节点 × 8 卡，经 ceph 队列 qx.sh 派发）+ 环境补装（UniRL 依赖入 leo2-venv）

## 风险清单

- FSDP 包 hymm 模型：block 类名待定（hymm leo.py 的层类）；MoE 专家在 FSDP full-shard 下的显存/通信
- prepare_inputs_for_generation 是否有 per-step 状态副作用（需无状态才能 replay）
- timestep 数值约定（σ×1000?）需与 retrieve_timesteps 对齐做 parity
- 文本编码器 Qwen3.5-9B 18GB：CPU 放置（抄 H3 aux_components_on_cpu）或每 rank GPU 常驻的取舍
- parity 验证：eta=0 的 UniRL rollout vs 原生 pipeline 输出应接近一致

## 参照系（用户指认的已跑通 RL：hymm/trainers/rl/leo2_grpo_puretorch_dit.py）

1. **同款捕获哲学**：他们的 model_kwargs 也来自捕获（`_capture_grpo_sample_inputs`），与我们的 recorder 方案同构——路线合法性得到实证背书。
2. **KeyError 陷阱（1443-1458 行文档）**：`prepare_inputs_for_generation` 对 `visual_mask/text_mask/timesteps_index/audio_mask/und|gen|audio_token_indices` 用 `kwargs[...]` 硬解引用——键必须存在（None 可以，缺失必炸）。我们整包透传原生 model_kwargs、从不丢 None → 天然安全；**将来任何"清理字典"的重构都是禁区**。
3. **denoise_step 备胎**：他们逐步调用走 `pipeline.denoise_step(latents, timesteps, idx, model_kwargs, guidance_scale=…)`（内部处理 CFG）。若我们的 prepare+forward 直调有问题，这是已验证的单调用替代。注意他们 rollout 用 guidance 6.0（带 CFG 的 rollout、logp 记在 guided pred 上）——与我们 guidance 1.0 的正统 flow-grpo 是两种流派；若日后 reward 不涨，guided rollout 是备用杠杆。

## 冒烟迭代记录（8/28）

| 轮 | 卡点 | 修复 |
|---|---|---|
| R1 | `rollout dp_size=8 must divide batch_size` | prompts_per_rollout 2→8（DP_SCATTER 整树分发） |
| R2 | `prepare_channel_cond_latents` 返回三元组 | 解包 `(cond, mask, task_type)` |
| R3 | 条件张量 CPU vs 权重 cuda | 前向时张量 `.to(device)`、None 透传（照参照 RL） |
| R4 | 根级模块（time_embed 等）仍在 CPU | bundle 加载后把非 block 根子模块搬上卡（FSDP 只搬 block） |
| R5 | `FSDP expects uniform original parameter dtype {fp32,bf16}` | MoE router gate.wg 在 block 内为 fp32 → 整模统一 bf16（`uniform_bf16`，已知取舍：fp32 router 精度；正式配方可改为把 router 排除出 FSDP 组或走 fp32 主参数方案） |

加载优化：通过 `LEO2_CKPT_DIR` 指向节点本地 stage-in 目录，加载阶段 35min → 分钟级。
| R6 | 同上断言仍在（整模 bf16 后） | 真凶是 **LoRA 主参数**：wrap.py 把可训练参数预转 `master_dtype`(fp32)，与 block 内 bf16 冻结参数同组 → torch 2.7.1 FSDP2 不支持组内混 dtype（HV1/H3 配方在官方 torch2.10 镜像上无此限制）。冒烟改 `master_dtype: bf16`；正式配方两条路：换 v1.9 镜像(torch2.10) 或把 LoRA 模块单独成 FSDP 组保 fp32 master |
| R7 | 前向深入到 MoE router：`gate(hidden)` 报 `float != BFloat16` | hymm 在 autocast 关闭区用 fp32 输入乘 router 权重，而 uniform_bf16 把 wg 转成了 bf16。运行时 monkeypatch `gate.wg.forward` 让权重跟随输入 dtype（不改共享 clone 的文件，符合隔离协议） |
| R8 | 49f 前向 OOM（92.9GB 已分配）；21f 探针报 `image_seq [1,1512] vs index [1,3276]` | ① 21f 探针无效：hymm 把时长 snap 到训练 bucket 下限 **49 帧**（duration_range [49,361] step 4），捕获按 49 算 token 而 UniRL latent 按 21 算 → 几何必须 ≥49 且 4k+1；② OOM 根因待定：已加 pre-rollout 显存/分片报告（dtensor vs plain 参数、本地 GPU 参数字节）与 forward 前打点 |
| R9 | 打点定位 OOM：pre-rollout 基线 24.5GB 健康（75.49B 全部 DTensor 分片、本地 17.7GB；TE 在 CPU、VAE fp32 3.1GB 在 GPU），首个前向从 24.5GB 冲到 92.9GB | **根因：yaml 抄自 H3 的 `root_wrap: false`** → 48 个 block 各自是 FSDP2 root，而 FSDP2 对 root 单元前向后不 reshard → 逐层 unshard 累积（~3.1GB/层 × 22 层 ≈ 68GB 后爆）。H3/HV1 模型小未暴露。改 `root_wrap: true` |
| R10 | **OOM 消失**：forward peak 33.5GB（基线 24.5GB，前向瞬时 ~9GB），10 步去噪全部跑完；死在 VAE decode 后打包 `Videos.from_list` 收到裸 Tensor（需 Video 对象 `.frames`） | 修 vae.py 的 Videos 构造 |
| R11 | vae.py 改为 `Videos.from_list([Video(frames=...)])`（同 H3）；diffusion.py 每次前向打 `[leo2 perf] fwd#N grad dt before/after/peak` 供性能/显存表 | 待 decode→PickScore→replay→GRPO 首步 |
| R12 | R11 走完 rollout（4 样本×10 步，fwd 1.3–8s，peak 33.5GB）后在 MoE 前向内死于 `new_group: Resource temporarily unavailable`。根因：**hy_parallelism 全局并行状态从未初始化**，`get_parallel_state()` 每次调用都 new 一个 ParallelDims→init_device_mesh→新 NCCL 进程组+watchdog 线程，每层调用多次，累计几千个进程组后 pthread_create EAGAIN；也是前向异常慢的原因 | bundle `_bootstrap_hymm` 里按 `hymm/samplers/entry.py` 的 fsdp 分支一次性 `init_parallel_state(dp_shard=min(8,world), 其余 1)`；另：ceph 上 devcloud `tail -f` 收不到远端追加（隔壁 agent 同样结论），监控改轮询 |
| R13 | R12 仍死于同一 EAGAIN（PG ID 已到 13767，fwd#0 就挂）：bootstrap 时 `dist.is_initialized()` 还是 False（bundle 在进程组建立前构建，DCP 加载是单进程模式），守卫直接跳过 | 抽成 `ensure_hy_parallel_state()`：bootstrap 尝试一次 + `predict_noise` 首次前向前再确保（此时 FSDP 已起、dist 必已初始化；所有 DP rank 同步到达，满足集合语义） |
| R14 | **R13 进程组泄漏彻底消失**（EAGAIN 0、无 not-init 警告），rollout 4 样本×10 步全部跑完：no_grad 前向稳定 **1.29–1.30 s**（3334 token，192×336×49），峰值 33.5GB；PickScore 加载成功。随后 driver 在 `ray.get` 反序列化 rollout 结果时 `No module named 'hymm'`：`Leo2Conditions.hymm` blob 的 pickle 引用 hymm 类，而 driver/reward actor 进程 sys.path 没有 hymm | 短期：smoke 脚本 PYTHONPATH 加 hymm 仓 + deps；同时在 build() 打印 blob 内非 tensor/builtin 类型，后续把 blob 收敛为纯 tensor/builtin（多机与 reward actor 更干净） |
| R15 | **R14 首次打通 rollout→decode→PickScore→replay(old_logp)→进入 GRPO 更新**；blob 内非 builtin 类型只有 `tokenizer_output`(TokenizerEncodeOutput) 与 `batch_gen_video_info`(list[VideoInfo])。更新阶段带梯度 replay 死于 `leo.py:2553 diffusion_loss_fn(...)`=None：TrainStack 在 update 前 `model.train()`，而 LeoModel.forward 在 training 模式走 SFT 损失路径（且文本 de-padding 分支不同） | `predict_noise` 内强制 `model.eval()`（梯度照常流动；rollout/replay 走同一张图，保证 parity）。隔壁原生 trainer 反而 `transformer.train()`，因其自建 pipeline 绕过了损失路径——不照搬 |
| R16 | **🎉 t2v FlowGRPO 首个完整训练步跑通（2026-08-29 22:26）**：`rollout 1/100 reward=0.7310 loss=-0.0000 gn=0.0026 ratio=1.0000±0.0000 |Δlogp|=0`（首步 old=new 策略，ratio 恒 1 属预期）。带梯度前向 1.31 s、前向后常驻 25.9GB、峰值 34.8GB（含 AC 保留激活；反向峰值未单独打点）。单步 = rollout.generate 98 s + train_track 95 s ≈ 193 s（8 卡，32 样本，10 步 SDE，训练时间步 4/10）。奖励水位与隔壁原生框架一致（0.71 级） | 后续：反向峰值打点、rollout 非前向开销（~46 s：文本编码器 GPU 往返 + VAE decode + 条件捕获）优化、16 卡扩展 |
| R15 稳态 | 8 步稳态：195–203 s/步（rollout.generate 97–99 s + train_track 95–103 s），reward 0.727–0.733，gn 0.0012–0.0026，ratio 恒 1（单更新 + replay 旧 logp，属预期）。训练中 nvidia-smi 每卡占用峰值 51.1–52.4 GB、平均利用率 73–77%。实测报告 `docs/R15_perf_report.html`（artifact https://claude.ai/code/artifact/1b60e2d0-2c0a-4f17-b026-87bce9a8fae3 ） | 22:49 经 node1 队列 ssh 停掉冒烟、恢复 node0 占卡（`remote/n0_kill_unirl.sh`；node0 自身队列被冒烟串行占住、PTY 通道丢字节） |
| 长跑 L1 | 2026-08-29 23:2x 起 node0 长跑：`remote/unirl_longrun.sh`（R15 配方不变；num_rollouts 300、save_interval 25 存 LoRA adapter 到 `ckpts/<run>`、wandb 上线）。节点无 W&B 凭据 → 先 offline（`WANDB_DIR=$X/wandb`），凭据放 `$H/env/wandb.env` 后可 `wandb sync` 或重启在线；trap 退出自动恢复占卡 | 监控轮询 `logs/longrun_latest.path` |
| 长跑 L2 | 03:25 在线重启（W&B run i2iowk7e）。gap_peak 打点：反向+优化器阶段分配器峰值 38.9 GB（≈前向峰 39.1）；每个 rollout 首前向前 gap_peak 40.3–40.4 GB = 常驻 24.7 + Qwen3.5-9B 临时上卡（≈18 GB）——文本编码器常驻 GPU 不会抬高全程峰值（前向峰 ≈51、更新峰 ≈57，仍远低于 87），只会省掉每步往返 | 下一轮优化首选项 |
| 长跑 L1 结论 | v1（lr 1e-4、独立 x_T、单更新）55 步：reward 0.731 → 0.705 单调缓降（前 15 步均值 0.729，第 50/55 步 0.707/0.705），gn 尖峰 0.019；ratio 恒 1、无 clip、无 KL → 无信任区间的漂移。checkpoint-25/50 已存（各 1.8 GB adapter+optim）。W&B run i2iowk7e | 06:5x 停 v1，切 v2 = `unirl_longrun_v2.sh`：`init_same_noise=true`（组内共享 x_T，H3 教训）+ `num_updates_per_batch=2` + lr 2.5e-5 |
| 长跑 L2 @25 | v2（W&B gmdid9wv）25 步：5 步均值 0.729/0.730/0.729/0.729/0.729 **持平**；同期 v1 已从 0.730 降到 0.724（55 步降到 0.709）。ratio 0.999±0.001、clip 31–38%、|Δlogp| 5e-4～2e-3（远低于 1e-2 告警线）、gn 多数 <0.01（孤立尖峰 0.42/0.04），checkpoint-25 1.8 GB | 漂移已止住，尚无上行；继续观察至 100 步 |
| R17 **根因** | 抽帧目检：v1/v2 长跑的 rollout 视频全是无结构色块、帧间不连续（W&B 媒体 gif）——PickScore 0.73 是对彩色噪声的打分，reward 曲线无意义。二分探针（node1，`unirl_probe.sh`，eta 0.02 近 ODE）：**同一份权重、同一 stage，近 ODE 采样出的是连贯可辨的场景** → 权重/包裹/条件 blob/步进全部正确；**坏在 FlowSDE 核 eta 0.7 @ [0,3,6,9] 的噪声注入**：σ=1 处 std_dev_t=√(0.99/0.01)·0.7≈7，首步均值系数只剩 0.12、注入 std 1.3 的噪声（离散化方差 1.76 vs 真实边际 0.93）；末步 (σ≈0.25→0) 又往最终 latent 注入 std 0.2 的噪声且不再去噪。SD3 图像扛得住，Leo2 视频扛不住 | 换 **CPSSDEStrategy**（系数保持：边际方差逐步守恒、σ_next=0 时 std=0）+ eta 0.6 + sde_indices [0,2,4,6]（H3 成品配方即 CPS + c2 归一化 logp + σ 带）。探针 `unirl_probe_cps.sh` 验证中；W&B v2 run 数据作废 |
| R18 | 探针 1（FlowSDE eta 0.02，近 ODE）：**采样正常**（连贯场景、人物可辨）→ 确认坏在 SDE 噪声注入而非权重/条件/步进。探针 2（CPS eta 0.6 @[0,2,4,6]）首跑因 `LEO2_DEBUG_HYMM_SAMPLE` 的 hymm A/B 路径在主机侧 CPU OOM（8 rank 同时走 generate_video）拖死 worker；已禁用 A/B 重跑 | 待帧图 |
| R19 | CPS 探针重跑：`latents_at(10) not in stored [0..7]`——`compute_trajectory_positions` 只存到 max(sde)+1，sde 带不含末段时终态 latent 没存 | generate() 的 needed 集合并入 `num_steps`。v2 长跑已停、node0 占卡已恢复 |
| R20 | CPS 探针（10 步，eta 0.6）两种带 [0,2,4,6]/[2,4,6] 的 rollout 均为连贯可辨场景（个别样本过饱和，10 步固有）→ CPS 是可用备选。按用户指示改为**镜像原生 pure-torch 配方**：FlowSDE、30 步、eta 0.5、只在前 5 个转移加噪并训练（原生 progressive/timesteps_group_size 5：其余 25 步纯 ODE）、k=8 共享 x_T、组内 std、PickScore 4 帧均值、clip 1e-4；LoRA lr 2.5e-5、无 KL 为固有差异 | `unirl_longrun_v3.sh` 15:4x 于 node0 拉起（64 样本/步，预计 ~10 min/步），首轮 rollout 抽帧落盘目检 |
| R21 | **v3（镜像原生配方）首轮 rollout 目检通过**：30 步 FlowSDE eta 0.5 只在前 5 步加噪，8 卡首/中/末帧拼图全是真实视频内容（餐厅人物、舞台表演、多镜头切换与提示词相符）。`rollout 1/200 reward=0.7211 gn=0.0003 ratio=1.0000 |Δlogp|=8.6e-6`——高噪声步的 logp 对 θ 极不敏感（dt≈0.011、std≈0.5），ratio/clip 基本不动，属该配方特性（原生同样）。单步 ≈9.5 min（64 样本 × 30 步 + 5 步×8 样本训练）。W&B run qyjuqgxk | 观察 30–50 步看 reward 是否上行；同步准备 16 卡扩展 |
| v3 基线 | 单步 ≈630 s = rollout.generate ≈400 s（240 次前向 312 s + ≈88 s 条件/文本编码器往返/decode/打分）+ train_track ≈225 s（replay 40 次 52 s + 2 轮更新）。前 4 步 reward 0.721/0.716/0.719/0.727，第 4 步 clip 首次非零 (0.05) | node1 perf 探针：`old_logp_source=rollout` + 文本编码器常驻 GPU（bundle 新增 resident 分支），预期 −50~70 s/步 |
| perf 探针 | node1，v3 配方 + `old_logp_source=rollout` + 文本编码器常驻 GPU（15.6 GB）：rollout.generate 329 s（v3 400 s，−71 s）、train_track 165–171 s（v3 225 s，−55 s），单步 ≈495 s（−21%）；首步 reward 0.7211 与 v3 逐位一致，rollout-vs-replay |Δlogp| 2.1e-5（告警线 1e-2）→ 两项提速均安全 | 用于后续所有运行；16 卡因 DP_SCATTER 按 prompt 整除只能翻倍样本数、不能缩短单步，暂缓 |
| **R22 首次学习信号** | 配对对照（同 seed、同提示词顺序）20 步：**v3b（lr 1e-4）5 步均值 0.7225→0.7247→0.7296→0.7328 单调上行，前 10 步 0.7236 → 后 10 步 0.7312（+0.0076）；v3（lr 2.5e-5）0.7210→0.7216 持平**。步间噪声 std≈0.005 → 10 步均值差 0.0096 ≈ 4σ，显著。LoRA 需要比 H3 更高的 lr 才能在这一时间尺度动起来；ratio/clip 仍 ≈1/0（高噪声步 logp 不敏感），与原生同 | v3b（W&B x8grheg7）继续；v3 作对照已完成使命，计划停掉 node0 改跑 lr 2e-4 括号试验。运维缺口：两台机的队列 runner 执行长任务期间不接单，没有控制通道——给长跑脚本加 STOP 文件轮询 |
| 运维 | PTY 通道（含放慢击键版 tx_slow.sh）仍不可用；两台机队列被长跑占住 → v3（node0）与 v3b（node1）都只能跑完（v3b 约 8/31 21:30、v3 约 9/1 03:00）。新版 `unirl_longrun.sh` 已加 STOP 文件轮询（`touch logs/STOP_<run>` 或 `STOP_ALL`），对后续运行生效。实测报告 artifact 已加勘误 | 期间只做离线分析 |
| R23 | 配对 10 步均值：v3b（lr 1e-4）0.7236 → 0.7312 → 0.7363 → 0.7409（1–36 步，+0.017，稳定上行，斜率与原生 kl30_rb8 前 40 步相当）；v3（lr 2.5e-5）0.7210 → 0.7216 → 0.7218 → 0.7250（31–40 步开始微升）。两条都在学，lr 决定速度；checkpoint-40 已存 | 继续观察至 100 步；下一步用 STOP 文件机制换 lr 2e-4 括号 + 16 卡样本翻倍 |
| R24 @60 | 配对 10 步均值：v3b 0.7236/0.7312/0.7363/0.7413/0.7428/0.7472（60 步 +0.024，第 60 步单步 0.7595 新高）；v3 0.7210/0.7216/0.7218/0.7250/0.7261/0.7278（+0.007）。原生 kl30_rb8 同期约 +0.03（0.71→0.74）——UniRL LoRA 线的学习斜率已与原生全参线同量级。checkpoint-60 两条都已存 | 继续到 100 步 |
| R25 @100 | v3b 过半程：20 步均值 0.7274 → 0.7388 → 0.7450 → 0.7573 → 0.7527（100 步 +0.025~0.030，第 75 步单步峰 0.7599），与原生 kl30_rb8 的曲线形态一致（0.71→~0.75@100）；v3（lr 2.5e-5）同期 0.7213 → 0.7334 也在稳升。checkpoint-100 已存 | UniRL trainside 的 t2v FlowGRPO 全链路修复完成：能采样、能打分、能学习 |
| **R26 质量-奖励背离** | W&B 媒体目检：**v3（lr 2.5e-5）在 rollout ~45（坦克，清晰连贯）与 ~90（玻璃走廊人物，锐利）画质保持**，reward 缓升 +0.012；**v3b（lr 1e-4）在 rollout ~50 起收敛为竖条纹高频纹理**（~50 与 ~99 两个采样点均如此），其 reward +0.03 大部分是 reward hacking——PickScore 4 帧均值在 192×336 下被高频纹理骗过。原生线能在 0.77 不崩靠的是 kl_weight 0.001 参考模型 + grpo_guard，此前被我当作"固有差异"跳过是错误判断 | 结论：无 KL 时 LoRA lr 上限 ≈2.5e-5；要用 1e-4 必须补 KL。v3 继续（健康且在学）；v3b 跑完后 node1 转向实现 ref-model KL |
| R27 | UniRL FlowGRPO 原生支持 KL：`algorithm.beta>0` 对 **LoRA 关闭态的基模型**做逐步高斯 KL（`adapters_disabled`，零额外显存，每次更新多一趟 no_grad replay；Leo2 stage 的 replay 已返回 prev_sample_means ✓）。v4 = v3b + beta 0.001，待 node1 释放后拉起 | 注意：两条在跑的 wrapper 曾在运行中被编辑（STOP 轮询），bash 按字节偏移续读可能在 python 退出后小概率报错，trap 的占卡恢复不受影响 |
| 运维补记 | `wandb.Api().run(...).stop()` 发送成功但训练不退出（stop 标志只对 wandb agent/sweep 启动的进程生效，train_diffusion 不查询）→ 该通道无效，记忆已更正。v3b 只能跑完（预计 8/31 ~21:30），已挂守望：其队列 .rc 一出现即自动派发 v4 | — |
