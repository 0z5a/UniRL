# 视音频联合 GAN 蒸馏 trainer（Diffusion-GAN，判别 r 时刻 noisy latent）。
#
# 全新、自包含；不修改任何现有文件。通过 --trainer distill.gan_trainer.GANTrainer 选用。
#
# ============================ 设计总览 ============================
# 迁移自 HunyuanVideo_pureTorch 的 T2VA GAN，并融合本仓库 cfgdistill / opd 的工程范式：
#   * 判别器 backbone = 冻结 teacher（复用 cfgdistill 的 build_teacher_model）；
#     teacher 中间特征通过 forward hook 非侵入抽取（gan_helpers.TeacherFeatureExtractor）。
#   * 判别器 head = 视音频【联合】打分（gan_helpers.JointDiscriminatorHead），
#     DDP 复制 + 独立 AdamW；判别 r 时刻的 noisy latent（Diffusion-GAN）。
#   * 时间：离散 64 步、K=8 跳步；video / audio 各用自己的 flow_shift
#     （gan_helpers.gan_time_utils，与推理 FlowMatchDiscreteScheduler 完全一致），
#     绝不硬编码 shift=5。
#   * 一步训练 = 1 次 D 更新（手动 backward/step）+ 1 次 G 更新（经引擎闭包，
#     base train() 负责 backward/step student）。即 TTUR=1。
#
# ============================ fake / real 构造 ============================
# 从标准数据管道拿到 (x_t, t, ut)，反推干净/噪声端点（assumption 1，已确认）：
#       x1 = x_t - t*ut ,  x0 = x1 + ut
# 在 GAN 自定义时刻重建：
#       x_t_gan = (1-t_gan)*x1 + t_gan*x0
#       fake_x_r = x_t_gan - (t_gan-r_gan)*student_pred      （student 一步到 r）
#       real_x_r = (1-r_gan)*x1 + r_gan*x0                    （同噪声插值到 r）
# video / audio 各自做一套，judge 时联合喂判别器。
#
# ============================ 梯度路径（assumption 2，已确认）============================
#   * G 步：fake_x_r(带梯度) -> 冻结 teacher 抽特征(grad on) -> 联合 head -> hinge G，
#     梯度经冻结 teacher 回流到 student。
#   * D 步：student no_grad 出 fake；real/fake 特征 detach -> 联合 head -> hinge D。
#
# =================================================================================
# 【给 infra 同事的关键说明 / READ ME FIRST】
# ---------------------------------------------------------------------------------
# 1. 这是【纯 GAN】训练：训练生成器 G（=student，LeoModelMeanFlow）的 loss【只有】
#    GAN 对抗 loss（hinge G），【没有】任何 meanflow / flow-matching / anchor loss 参与
#    G 的 backward。证据见下方 train_step 的 G 步：引擎只 backward `loss_closure` 返回的
#    `gan_g_weight * loss_g`（get_gan_loss(..., "G")）。
#      - 注意：student 是 MeanFlow 模型，其 forward 在 train 模式下【必然】会走内部 loss
#        分支（leo_meanflow.py:484+）。该内部 loss 会被引擎【丢弃】、【不参与】backward
#        （引擎用 loss_closure 的返回值，见 parallel_engine.py:1013）。我们用 `_drop_anchor_kw`
#        包一层让它不崩。
#      - 【anchor 已关闭】student 的 anchor 分支（重算最后 1/4 层 tail+head 得 u(t,t)）是
#        iMF 一致性项才需要的，纯 GAN 用不到。我们已通过 leo_meanflow.py:404 的 opt-in 开关
#        `_gan_disable_anchor` 关掉它（见本类 _disable_student_anchor()），need_anchor 恒 False，
#        省掉这部分前向算力+显存，且不影响 diff_pred=u(t,r)。该开关默认 False，对 iMF 零影响。
#        注意"算 loss"本身（FM 的 MSE，逐元素运算）算力/显存可忽略；真正省下的是 anchor 的
#        tail+head 重算。
#      - 【残留可优化点】anchor 关闭后，student forward 仍会执行内部 diffusion_loss_fn 得到一个
#        被丢弃的 FM MSE loss（cheap，但仍建了一点计算图）。infra 若想彻底短路，可让引擎在注册了
#        loss_closure 时跳过模型内部 loss 分支（return prediction only）。收益已不大。
#
# 2. 一个 train_step 内做两件事（TTUR=1）：
#      D 步：student(no_grad) 出 fake -> teacher(no_grad) 抽 fake/real 特征 -> 判别器
#            打分 -> hinge D -> 判别器【独立优化器】手动 backward/step（绕开引擎）。
#      G 步：student(grad) 出 fake -> teacher(grad, 保留计算图) 抽特征 -> 判别器(冻结) 打分
#            -> hinge G -> 经引擎 loss_closure 更新 student。
#
# 3. 显存/性能现状（当前为"先跑通"的轻量化配置）：
#      - teacher engine 建时 enable_gradient_checkpointing=False（继承自 cfgdistill），
#        G 步 teacher grad-forward 会存全部已运行层的 activation。
#        【可优化点 B】给 teacher 开 gradient checkpointing 可大幅省显存（本文件未改，
#        因受"只新增文件"约束；infra 可直接在 teacher engine 构建处打开）。
#      - feature_extractor 用"抓完最后一个特征层就抛异常提前中止 teacher forward"来省显存
#        （_EarlyStopExtraction）。【可优化点 C】更干净的做法是让 teacher 支持"只跑到第 L 层"
#        的原生截断 forward，避免异常控制流。
#      - 判别器 CrossAttention 对超长 context 序列先自适应池化到 disc_max_ctx_tokens 个 token
#        （见 joint_discriminator.py）。
#      - 默认只抽 1 层浅特征（disc_extract_layers 默认 [num_layers//4]），可用
#        --disc-extract-layers "a,b,c" 加深/加多层。
#
# 4. 并行：判别器 head 是【DDP 复制】(dp_size>1 时)，非 FSDP 分片；teacher/student 走
#    本仓库的 CP（context parallel）+ FSDP。特征在 hook 内已做 CP all-gather（见
#    feature_extractor.py 的 maybe_gather_seq）。
#
# 5. 可通过命令行覆盖的 GAN 超参（均有默认值，见 run_pure_torch_leo_gan.sh）：
#      --gan-num-steps 64 --gan-step-jump 8 --disc-lr --gan-g-weight --gan-d-weight
#      --disc-num-heads --disc-grad-clip --disc-extract-layers --disc-max-ctx-tokens
#      --gan-loss-type {hinge,non_saturating,softplus}
# =================================================================================

from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from ..distill.cfgdistill_trainer_v1 import MultimodalTrainer as CfgDistillTrainer
# 关键：import 以下两个模块会触发其 install_*_dispatch() 副作用，为 build_model 注册
#   - LeoModelMeanFlow（student，--model-structure LeoModelMeanFlow）
#   - leo-*-noTR（teacher，无 timestep_r 变体）
# 否则 build_model 会落到 legacy 分支并抛 "Model ... not implemented"。
# 与 imf_trainer.py 第 80-81 行做法一致。
from ...models.diffusion import leo_dummy_tr as _leo_dummy_tr  # noqa: F401
from ...models.diffusion import leo_meanflow as _leo_meanflow  # noqa: F401
from .gan_helpers.gan_time_utils import GanTimeConfig, build_modality_axis
from .gan_helpers.joint_discriminator import JointDiscriminatorHead, get_gan_loss
from .gan_helpers.feature_extractor import TeacherFeatureExtractor, _EarlyStopExtraction
from .gan_helpers import gan_ragged as R


def _move_tensors_to_device(obj, device):
    """递归把结构中的 torch.Tensor 叶子挪到 device；非张量（partial/None/标量/字符串）原样保留。

    用于 teacher 抽特征前统一 disc_kwargs 的设备，避免 full-seqlen 输入分支里
    input_ids / visual_mask 等元数据张量设备不一致导致的 masked_select 报错。
    只挪 CPU->device 的张量；已在目标设备的为 no-op。不触碰 functools.partial 的内部 keywords。
    """
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_tensors_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in obj)
    return obj


def _drop_anchor_kw(fn):
    """包装 loss_fn，吞掉 MeanFlow 内部 forward 传入的 model_output_anchor 关键字。

    student 是 LeoModelMeanFlow，G 步（train + timestep_r + grad_enabled）会置
    need_anchor=True，从而以 diffusion_loss_fn(model_output=..., model_output_anchor=...)
    调用；而数据管道给的普通 Transport.training_losses_fn 不接受 model_output_anchor。
    该内部 loss 会被引擎丢弃（真正 backward 的是 GAN loss_closure），这里只需让它不崩。
    """
    if fn is None:
        return None

    def wrapped(*args, model_output_anchor=None, **kwargs):  # noqa: ARG001
        return fn(*args, **kwargs)

    return wrapped


class GANTrainer(CfgDistillTrainer):
    """视音频联合 GAN 蒸馏 trainer。"""

    def __init__(self, args: Namespace):
        # CfgDistillTrainer.__init__ -> 构建 model_engine / 各 denoiser / text_encoder / teacher
        super().__init__(args)
        device = torch.device("cuda", args.local_rank)

        # ---- GAN 时间配置（N=64, K=8；shift 取 config 的 video/audio 各自值）----
        num_train_ts = int(getattr(args, "num_train_timesteps", 1000))
        reverse = bool(getattr(args, "flow_reverse", True))
        self.gan_time_cfg = GanTimeConfig(
            num_steps=int(getattr(args, "gan_num_steps", 64)),
            step_jump=int(getattr(args, "gan_step_jump", 8)),
            num_train_timesteps=num_train_ts,
            reverse=reverse,
            flow_shift_video=float(getattr(args, "flow_shift_video", 7.0)),
            flow_shift_audio=float(getattr(args, "flow_shift_audio", 1.0)),
        )
        self.video_axis = build_modality_axis(self.gan_time_cfg.flow_shift_video, self.gan_time_cfg, device)
        self.audio_axis = build_modality_axis(self.gan_time_cfg.flow_shift_audio, self.gan_time_cfg, device)

        # ---- 关闭 student(MeanFlow) 的 anchor 分支（纯 GAN 用不到 u(t,t)）----
        # 见 leo_meanflow.py:404 的 `_gan_disable_anchor` opt-in 开关。G 步会因
        # (training & timestep_r & grad) 触发 need_anchor=True，从而额外重跑最后 1/4 层 tail
        # + head 去算 diff_pred_anchor=u(t,t)——那是 iMF anchor 一致性项才需要的，纯 GAN 用不到。
        # 打开此开关后 need_anchor 恒 False，直接省掉这部分前向算力+显存，且不影响 diff_pred=u(t,r)。
        # 只作用于本 GAN trainer 持有的 student，对 iMF/其它训练零影响（默认 False）。
        self._disable_student_anchor()

        # ---- 抽取层 index（teacher 层；视音频共用同一组层）----
        num_layers = len(self.teacher_model.layers)
        cfg_layers = getattr(args, "disc_extract_layers", None)
        if cfg_layers:
            if isinstance(cfg_layers, str):
                self.disc_layer_indices = [int(x) for x in cfg_layers.replace(",", " ").split()]
            else:
                self.disc_layer_indices = [int(x) for x in cfg_layers]
        else:
            # 轻量化默认（以跑通为目标）：只抽【1 层浅特征】(≈1/4 深度)。
            # teacher engine 未开 gradient checkpointing，G 步带梯度 forward 会存下
            # 0..max(index) 层的全部 activation；配合 feature_extractor 的早停，最深层设在
            # 1/4 可让 teacher 只跑 1/4 的层，且判别器只持有 1(video)+1(audio) 个全序列特征
            # （而非 3+3），显存峰值大幅下降，避免部分节点 cuBLAS/CUDA OOM。
            # 显存有余时可用 --disc-extract-layers "a,b,c" 加深/加多层。
            self.disc_layer_indices = [num_layers // 4]
        self.logger.info(f"[GAN] teacher num_layers={num_layers}, disc extract layers={self.disc_layer_indices}")

        # ---- teacher 特征抽取器（forward hook，非侵入）----
        # 【重要 / 集合通信安全】early_stop 必须为 False！
        # teacher 的 MoE forward 跑在 torch.compile + EP（专家并行 all-to-all）下，若用异常从
        # forward 中途抛出提前中止，会破坏各 rank 的 all-to-all 对称性，导致 mesh_ep 集合通信
        # desync / NCCL 超时卡死（已复现：各 rank 卡在不同 SeqNum）。因此这里【禁用早停】，
        # teacher 跑完整 forward（所有 rank 对称，集合通信安全）。省显存改用其它对称手段：
        #   - 关闭 anchor（_disable_student_anchor）
        #   - 只抽 1 层浅特征 + 判别器 context 池化
        #   - 若仍紧张，交给 infra 给 teacher engine 开 gradient checkpointing（对称、安全）。
        self.feat_extractor = TeacherFeatureExtractor(
            self.teacher_model,
            layer_indices=self.disc_layer_indices,
            cp_gather=True,
            capture_audio=True,
            early_stop=False,
        )

        # ---- GAN 超参 ----
        self.gan_loss_type = str(getattr(args, "gan_loss_type", "hinge"))
        self.gan_g_weight = float(getattr(args, "gan_g_weight", 1.0))
        self.gan_d_weight = float(getattr(args, "gan_d_weight", 1.0))
        self.disc_lr = float(getattr(args, "disc_lr", 1e-5))
        self.disc_num_heads = int(getattr(args, "disc_num_heads", 16))
        self.disc_grad_clip = float(getattr(args, "disc_grad_clip", 1.0))
        # 轻量化：判别器 cross-attention 前把 context 序列池化到该 token 预算（<=0 不池化）。
        self.disc_max_ctx_tokens = int(getattr(args, "disc_max_ctx_tokens", 512))

        # 判别器懒构建（首个 D 步拿到特征维度后再建，避免依赖 config 中 hidden_size 字段名）
        self.discriminator = None
        self.discriminator_optimizer = None
        self._gan_iter = 0

    def _disable_student_anchor(self):
        """给底层 student(LeoModelMeanFlow) 打上 _gan_disable_anchor=True，关闭 anchor 重算。

        纯 GAN 只需 diff_pred=u(t,r)，不需要 anchor 分支的 u(t,t)。见 leo_meanflow.py:404
        的 opt-in 开关；对 iMF/其它训练零影响（默认 False）。

        注意：FSDP2 / torch.compile 包装后 type(m).__name__ 可能不再是 "LeoModelMeanFlow"，
        因此这里用 meanflow 专属方法 `_mf_run_tail` 作为鲁棒判据，并同时扫描 engine.model /
        fsdp_models 两处。关闭 anchor 会让所有 rank 一致地少跑最后 1/4 层 tail+head（含 MoE
        all-to-all），是【集合通信对称】的省显存/算力手段。
        """
        n = 0
        seen = set()
        candidates = []
        m0 = getattr(self.model_engine, "model", None)
        if m0 is not None:
            candidates.append(m0)
        candidates.extend(list(getattr(self.model_engine, "fsdp_models", [])))
        for root in candidates:
            if root is None:
                continue
            for m in root.modules():
                if id(m) in seen:
                    continue
                seen.add(id(m))
                if hasattr(m, "_mf_run_tail"):  # meanflow 专属方法 -> 就是 LeoModelMeanFlow(HF)
                    m._gan_disable_anchor = True
                    n += 1
        self.logger.info(f"[GAN] disabled MeanFlow anchor recompute on {n} student module(s).")

    # ================================================================= #
    # 判别器：懒构建（首次 D 步用捕获特征的维度构建 head + optimizer）
    # ================================================================= #
    def _ensure_discriminator(self, video_feats, audio_feats):
        if self.discriminator is not None:
            return
        device = torch.device("cuda", self.args.local_rank)
        video_dim = video_feats[0].shape[-1]
        audio_dim = audio_feats[0].shape[-1]
        head = JointDiscriminatorHead(
            video_dim=video_dim,
            audio_dim=audio_dim,
            video_num_layers=len(video_feats),
            audio_num_layers=len(audio_feats),
            num_heads=self.disc_num_heads,
            max_ctx_tokens=self.disc_max_ctx_tokens,
        ).to(device=device, dtype=torch.bfloat16)
        self.logger.info(
            f"[GAN] built JointDiscriminatorHead: video_dim={video_dim}, audio_dim={audio_dim}, "
            f"layers={len(video_feats)}/{len(audio_feats)}, "
            f"params={sum(p.numel() for p in head.parameters())/1e6:.2f}M"
        )
        if self.dp_size > 1:
            from torch.nn.parallel import DistributedDataParallel
            head = DistributedDataParallel(
                head,
                device_ids=[device.index],
                output_device=device.index,
                process_group=self.p_state.dp_group,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        self.discriminator = head
        self.discriminator_optimizer = torch.optim.AdamW(
            [p for p in self.discriminator.parameters() if p.requires_grad],
            lr=self.disc_lr, betas=(0.0, 0.999), weight_decay=0.01, eps=1e-8,
        )
        self._resume_discriminator()

    def _disc_module(self):
        return self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator

    def _resume_discriminator(self):
        if not getattr(self.args, "resume", False):
            return
        ckpt_dir = self.checkpoint_dir
        if not ckpt_dir.exists():
            return
        import re
        pat = re.compile(r"^iter_\d+$")
        iters = [d for d in ckpt_dir.iterdir() if d.is_dir() and pat.match(d.name)]
        if not iters:
            return
        latest = max(iters, key=lambda d: d.name)
        disc_ckpt = latest / "discriminator.pt"
        if not disc_ckpt.exists():
            self.logger.warning(f"[GAN] no discriminator.pt under {latest}; discriminator starts fresh.")
            return
        state = torch.load(disc_ckpt, map_location="cpu", weights_only=False)
        self._disc_module().load_state_dict(state["discriminator"])
        self._disc_module().to(torch.device("cuda", self.args.local_rank), dtype=torch.bfloat16)
        if "discriminator_optimizer" in state and self.discriminator_optimizer is not None:
            self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        self.logger.info(f"[GAN] discriminator resumed from {disc_ckpt}.")

    # ================================================================= #
    # teacher 特征抽取（forward hook）
    # ================================================================= #
    def _extract_joint_features(self, video_latents, audio_latents, video_model_t, audio_model_t, base_kwargs, grad):
        """用冻结 teacher 在 r 时刻抽取视频+音频中间特征。grad=True 时保留计算图（G 步）。"""
        disc_kwargs = dict(base_kwargs)
        # 只 pop 掉 meanflow 第二时间（强制 teacher 走单时间/无 anchor 路径，即"在 r 时刻抽特征"）
        # 与 CFG guidance；【务必保留】diffusion_loss_fn / audio_diffusion_loss_fn：
        # teacher 底层 fsdp_model 处于 train 模式（self.training=True，FSDP collective 需要），
        # forward 会走 loss 分支并调用 loss_fn；若 pop 掉则 None(...) 崩溃。
        # loss 值我们不用（只取 hook 抓到的中间特征），但 loss_fn 必须可调用。
        # 与 cfgdistill_trainer_v1.prepare_cfgdistill_target 的 teacher 调用范式一致。
        for k in (
            "timestep_r", "audio_timestep_r", "timesteps_r", "audio_timesteps_r",
            "guidance", "guidance_index",
        ):
            disc_kwargs.pop(k, None)
        disc_kwargs["latents"] = video_latents
        disc_kwargs["audio_latents"] = audio_latents
        disc_kwargs["timesteps"] = video_model_t
        disc_kwargs["audio_timesteps"] = audio_model_t

        # teacher 走 full-seqlen 输入分支（leo_meanflow.py:305-313）时，
        # hidden_states_fl = zeros(device=input_ids.device)，其 arange index 需与 visual_mask
        # 同设备做 masked_select。这里把所有 tensor 元数据统一挪到计算设备，避免
        # "input_ids 在 cpu、visual_mask 在 cuda" 之类的设备不一致（已在 cuda 则为 no-op）。
        dev = torch.device("cuda", self.args.local_rank)
        disc_kwargs = _move_tensors_to_device(disc_kwargs, dev)

        self.feat_extractor.clear()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            try:
                self.teacher_engine(**disc_kwargs)
            except _EarlyStopExtraction:
                # 抓完最后一个特征层后主动中止 teacher forward（省显存）；特征已在 hook 中捕获。
                pass
        return self.feat_extractor.collect()

    # ================================================================= #
    # student 无梯度预测（D 步造 fake 用；eval recipe 同 imf_trainer）
    # ================================================================= #
    def _student_predict_nograd(self, kwargs):
        was_training = bool(getattr(self.model_engine, "training", False))
        self.model_engine.eval()
        for m in self.model_engine.fsdp_models:
            m.train()
        try:
            with torch.no_grad():
                out = self.model_engine(**kwargs)
        finally:
            if was_training:
                self.model_engine.train()
        v = out["diffusion_prediction"] if isinstance(out, dict) else out
        a = out.get("audio_diffusion_prediction") if isinstance(out, dict) else None
        return v, a

    # ================================================================= #
    # 离散时间：从数据 t 结构采 idx，转各模态 (flow-time, model-t)
    # ================================================================= #
    def _sample_modality_times(self, t_struct, axis, device):
        idx_t, idx_r = R.sample_idx_like(
            t_struct, self.gan_time_cfg.num_steps, self.gan_time_cfg.step_jump, device
        )
        # CP 组内对齐 idx，避免 collective 错位
        if self.p_state.cp_group is not None and self.p_state.cp_size > 1:
            src = dist.get_global_rank(self.p_state.cp_group, 0)
            R.broadcast_idx_(idx_t, src, self.p_state.cp_group)
            R.broadcast_idx_(idx_r, src, self.p_state.cp_group)
        t_flow = R.idx_to_flow(idx_t, axis)
        r_flow = R.idx_to_flow(idx_r, axis)
        t_model = R.idx_to_model_t(idx_t, axis)
        r_model = R.idx_to_model_t(idx_r, axis)
        return t_flow, r_flow, t_model, r_model

    # ================================================================= #
    # 主训练步：D 步（手动）+ G 步（引擎闭包）
    # ================================================================= #
    def train_step(self, batch):
        args = self.args
        device = torch.device("cuda", args.local_rank)

        model_input_kwargs, _uncond_kwargs, bsz, seqlen = self.prepare_model_inputs(batch, device)

        v_loss_fn = model_input_kwargs.get("diffusion_loss_fn", None)
        a_loss_fn = model_input_kwargs.get("audio_diffusion_loss_fn", None)
        has_video = ("latents" in model_input_kwargs) and isinstance(v_loss_fn, partial)
        has_audio = (
            ("audio_latents" in model_input_kwargs)
            and isinstance(a_loss_fn, partial)
            and a_loss_fn.keywords.get("ut", None) is not None
        )
        assert has_video and has_audio, (
            "[GAN] joint video+audio GAN requires both branches present in the batch."
        )

        # ---- 反推 x0/x1，构建 GAN 时刻的 x_t / real_x_r（video & audio）----
        # 注意：video 的 latents 是 latent_channel_extend 布局（2C+1 通道）：
        #   [x_t(C) | extended_latents(C) | extended_mask(1)]，而 ut 只有 C 通道。
        #   所有 flow 数学只在 base(C) 通道上做；喂 student/teacher 前把 conditioning
        #   tail 原样拼回（audio 无扩展，tail=None 自动透传）。
        v_xt = model_input_kwargs["latents"]
        v_t_data, v_ut = v_loss_fn.keywords["t"], v_loss_fn.keywords["ut"]
        v_dtype = R.leaf_dtype(v_xt)
        v_base_xt, v_tail = R.split_cond(v_xt, v_ut)
        v_x1 = R.recover_clean(v_base_xt, v_t_data, v_ut)
        v_x0 = R.recover_noise(v_x1, v_ut)

        a_xt = model_input_kwargs["audio_latents"]
        a_t_data, a_ut = a_loss_fn.keywords["t"], a_loss_fn.keywords["ut"]
        a_dtype = R.leaf_dtype(a_xt)
        a_base_xt, a_tail = R.split_cond(a_xt, a_ut)
        a_x1 = R.recover_clean(a_base_xt, a_t_data, a_ut)
        a_x0 = R.recover_noise(a_x1, a_ut)

        v_t_flow, v_r_flow, v_t_model, v_r_model = self._sample_modality_times(v_t_data, self.video_axis, device)
        a_t_flow, a_r_flow, a_t_model, a_r_model = self._sample_modality_times(a_t_data, self.audio_axis, device)

        # base 通道上构造 GAN 时刻的 x_t / real_x_r
        v_xt_gan_base = R.reconstruct_xt(v_x1, v_x0, v_t_flow)
        a_xt_gan_base = R.reconstruct_xt(a_x1, a_x0, a_t_flow)
        v_td = R.time_diff(v_t_flow, v_r_flow)
        a_td = R.time_diff(a_t_flow, a_r_flow)

        v_real_r_base = R.reconstruct_xt(v_x1, v_x0, v_r_flow)
        a_real_r_base = R.reconstruct_xt(a_x1, a_x0, a_r_flow)

        # 喂模型前拼回 conditioning（student 输入需完整通道）
        v_xt_gan = R.attach_cond(v_xt_gan_base, v_tail, v_dtype)
        a_xt_gan = R.attach_cond(a_xt_gan_base, a_tail, a_dtype)
        v_real_r = R.attach_cond(v_real_r_base, v_tail, v_dtype)
        a_real_r = R.attach_cond(a_real_r_base, a_tail, a_dtype)

        # student / teacher 公共 kwargs（含文本条件等）
        g_kwargs = dict(model_input_kwargs)
        g_kwargs["latents"] = v_xt_gan
        g_kwargs["timesteps"] = v_t_model
        g_kwargs["timestep_r"] = v_r_model
        g_kwargs["audio_latents"] = a_xt_gan
        g_kwargs["audio_timesteps"] = a_t_model
        g_kwargs["audio_timestep_r"] = a_r_model
        # student 是 MeanFlow：G 步（train + timestep_r + grad）会触发 need_anchor=True，
        # 模型内部 forward 会用 model_output_anchor 调 diffusion_loss_fn（line 494）。
        # 数据管道给的是普通 Transport.training_losses_fn（不接受 model_output_anchor）。
        # 该内部 loss 会被引擎丢弃（真正 backward 的是下方 loss_closure 的 GAN loss），
        # 只需让它不崩：套一层 wrapper 吞掉 model_output_anchor。anchor 图在此后即释放。
        g_kwargs["diffusion_loss_fn"] = _drop_anchor_kw(model_input_kwargs.get("diffusion_loss_fn"))
        g_kwargs["audio_diffusion_loss_fn"] = _drop_anchor_kw(model_input_kwargs.get("audio_diffusion_loss_fn"))

        # ============================ D 步（训练判别器）============================
        # 判别器用【独立优化器】手动 backward/step，完全绕开引擎（引擎只管 student）。
        # student 不更新（no_grad），只用来造 fake 样本。
        # student no_grad 出 fake_x_r（base 通道），拼回 conditioning 后喂 teacher
        v_pred_ng, a_pred_ng = self._student_predict_nograd(g_kwargs)
        v_fake_r = R.attach_cond(R.fake_from_pred(v_xt_gan_base, v_pred_ng, v_td), v_tail, v_dtype)
        a_fake_r = R.attach_cond(R.fake_from_pred(a_xt_gan_base, a_pred_ng, a_td), a_tail, a_dtype)

        # teacher 抽特征（no_grad）
        fake_vfeat, fake_afeat = self._extract_joint_features(
            v_fake_r, a_fake_r, v_r_model, a_r_model, model_input_kwargs, grad=False
        )
        real_vfeat, real_afeat = self._extract_joint_features(
            v_real_r, a_real_r, v_r_model, a_r_model, model_input_kwargs, grad=False
        )

        self._ensure_discriminator(fake_vfeat, fake_afeat)
        disc = self.discriminator
        for p in disc.parameters():
            p.requires_grad_(True)
        self.discriminator_optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fake_logits = disc([f.detach() for f in fake_vfeat], [f.detach() for f in fake_afeat])
            real_logits = disc([f.detach() for f in real_vfeat], [f.detach() for f in real_afeat])
            _, loss_d = get_gan_loss(real_logits, fake_logits, self.gan_loss_type, "D")
            loss_d = self.gan_d_weight * torch.mean(loss_d)
        loss_d.backward()
        d_params = [p for g in self.discriminator_optimizer.param_groups for p in g["params"]]
        d_grad_norm = torch.nn.utils.clip_grad_norm_(d_params, self.disc_grad_clip, foreach=True)
        if not (torch.isnan(d_grad_norm) or torch.isinf(d_grad_norm)):
            self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad()
        loss_d_val = loss_d.detach()

        # 释放 D 步的大块中间显存（teacher 抽出的视音频特征、logits、student no_grad 预测、
        # 拼好的 fake/real latents），避免与 G 步的 student+teacher 双图峰值叠加导致 OOM。
        del fake_vfeat, fake_afeat, real_vfeat, real_afeat
        del fake_logits, real_logits, loss_d
        del v_pred_ng, a_pred_ng, v_fake_r, a_fake_r, v_real_r, a_real_r
        torch.cuda.empty_cache()

        # ============================ G 步（训练生成器 = student）============================
        # 判别器冻结（只训 student）。
        for p in disc.parameters():
            p.requires_grad_(False)

        captured = {}

        # ---------------------------------------------------------------------------------
        # 【纯 GAN 关键点】loss_closure 是引擎在 train 模式下真正用于 backward 的 loss 来源
        # （parallel_engine.py:1013：pp_loss, out = self.loss_closure(model_forward_ret, ...)）。
        # 这里返回的 loss【只有】GAN 对抗 loss（hinge G），因此更新 student 的梯度【只来自】
        # GAN loss，绝无 meanflow / flow-matching / anchor loss 参与。
        # 说明：model_output 是 student(MeanFlow) 的完整 forward 返回；其中虽含被丢弃的内部
        # FM loss（我们不取用），我们只取 diffusion_prediction / audio_diffusion_prediction。
        # ---------------------------------------------------------------------------------
        def loss_closure(model_output, _input_args_kwargs):  # noqa
            v_pred = model_output["diffusion_prediction"]         # student 预测的平均速度 u(t,r)
            a_pred = model_output["audio_diffusion_prediction"]
            # base 通道构造 fake_x_r = x_t_gan - (t-r)*pred（保留梯度），拼回 conditioning 后喂 teacher
            v_fake = R.attach_cond(R.fake_from_pred(v_xt_gan_base, v_pred, v_td), v_tail, v_dtype)
            a_fake = R.attach_cond(R.fake_from_pred(a_xt_gan_base, a_pred, a_td), a_tail, a_dtype)
            # 冻结 teacher 抽中间特征（保留计算图，梯度经冻结 teacher 回流到 student）
            vfeat, afeat = self._extract_joint_features(
                v_fake, a_fake, v_r_model, a_r_model, model_input_kwargs, grad=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake_logits_g = disc(vfeat, afeat)                 # 视音频联合打分
                loss_g_raw, _ = get_gan_loss(None, fake_logits_g, self.gan_loss_type, "G")  # hinge G
                loss_g_raw = torch.mean(loss_g_raw)
            loss_g = self.gan_g_weight * loss_g_raw
            captured["gan_g_loss"] = loss_g_raw.detach()
            # 返回值的第一项就是引擎要 backward 的 loss —— 【只有 GAN loss_g】
            return loss_g, {"loss": loss_g}

        self.model_engine.register_loss_closure(loss_closure)
        # 引擎跑 student forward 后调用上面的 loss_closure，并对其返回的 loss_g 做 backward + step。
        loss = self.model_engine(**g_kwargs)

        if torch.isnan(loss).any():
            self.nan_grad_count += 1
            self.logger.warning(f"NaN G-loss on rank {self.rank}, total NaN count: {self.nan_grad_count}")

        loss_dict = self.model_engine.get_cached_result("loss_dict")
        loss_dict["loss"] = loss
        loss_dict["gan_g_loss"] = captured.get("gan_g_loss", loss.detach())
        loss_dict["gan_d_loss"] = loss_d_val
        loss_dict["gan_d_grad_norm"] = d_grad_norm.detach()

        self._gan_iter += 1
        consumed_metrics = {
            batch["dataset_tag"][0]: {
                "samples": batch["n_samples"].sum().item(),
                "tokens": bsz * seqlen,
            }
        }
        return loss_dict, consumed_metrics

    # ================================================================= #
    # 日志 loss 名 + 判别器 ckpt
    # ================================================================= #
    def after_initialize(self):
        super().after_initialize()
        for name in ("gan_g_loss", "gan_d_loss", "gan_d_grad_norm"):
            if name not in self.loss_names:
                self.loss_names.append(name)

    def save_checkpoint(self):
        super().save_checkpoint()
        # 判别器 DDP 复制，rank0 存一份即可
        if self.discriminator is None:
            return
        tag = f"iter_{self.ss.update_steps:07d}"
        out_dir = self.checkpoint_dir / tag
        if self.rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "discriminator": self._disc_module().state_dict(),
                    "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                },
                out_dir / "discriminator.pt",
            )
            self.logger.info(f"[GAN] saved discriminator to {out_dir / 'discriminator.pt'}.")
        if dist.is_initialized():
            dist.barrier()
