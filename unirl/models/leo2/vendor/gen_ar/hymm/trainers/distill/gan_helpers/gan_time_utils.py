# 离散时间步 + 分模态 shift 工具（GAN 蒸馏）。
#
# 设计动机（见与用户的讨论）：
#   * 源仓库 HunyuanVideo_pureTorch 的 GAN 在 sample_discrete / discrete_time_to_continuous
#     里把 self.shift 硬编码成 5（写死 /64），既不对应 config，也不区分 video / audio。
#     这是一个隐患，迁移时必须修正。
#   * 本仓库推理（hymm/models/diffusion/leo_hf.py:149-151）为 video / audio 各建一个
#     FlowMatchDiscreteScheduler：
#         video_scheduler = FlowMatchDiscreteScheduler(shift=flow_shift_video)
#         audio_scheduler = FlowMatchDiscreteScheduler(shift=flow_shift_audio)
#     去噪循环 `for i, (t, at) in enumerate(zip(timesteps, audio_timesteps))` 按同一步
#     index i 同步推进，但 video 用 t、audio 用 at（连续时间因 shift 不同而不同）。
#
# 本模块完全复刻这一推理范式：
#   * 用 FlowMatchDiscreteScheduler.set_timesteps(N) 构造长度 N+1 的离散轴
#     （sigmas 是连续 flow-time，timesteps_full = sigmas * num_train_timesteps 是 model-t）。
#   * 采样一个【共享】的离散步 index idx_t ∈ [0, N]，idx_r = clamp(idx_t - K, 0)。
#   * video / audio 各自用自己的 scheduler，把同一个 idx 转成各自的 (flow-time, model-t)。
#
# 约定（来自 leo2 t2va，reverse=True）：
#   x_t = (1 - t) * x_clean + t * x_noise ，t=sigma；t=0 干净，t=1 纯噪声。
#   sigmas[0] = 1.0（噪声端），sigmas[N] = 0.0（干净端）。

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from ....diffusion.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler


# 用户确认的两个常量。
DEFAULT_NUM_STEPS = 64   # N：离散步数（与源 GAN 的 64 步轴一致）
DEFAULT_STEP_JUMP = 8    # K：student 大跳步距（idx_r = idx_t - K）


@dataclass
class GanTimeConfig:
    """GAN 离散时间配置。shift 取自 config 的 flow_shift_video / flow_shift_audio。"""
    num_steps: int = DEFAULT_NUM_STEPS                 # N
    step_jump: int = DEFAULT_STEP_JUMP                 # K
    num_train_timesteps: int = 1000
    reverse: bool = True
    flow_shift_video: float = 7.0
    flow_shift_audio: float = 1.0
    # 与推理保持一致的 flux-shift 选项（默认关闭；如推理开启需同步打开）。
    use_flux_shift: bool = False
    use_flux2_shift: bool = False


class ModalityAxis:
    """单个模态（video 或 audio）的离散轴。封装一个 FlowMatchDiscreteScheduler。

    sigmas        : (N+1,) 连续 flow-time（噪声端 1.0 -> 干净端 0.0）
    model_t_full  : (N+1,) model-t = sigmas * num_train_timesteps（喂给模型的 timesteps）
    """

    def __init__(self, shift: float, cfg: GanTimeConfig, device: torch.device):
        self.cfg = cfg
        self.scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            shift=shift,
            reverse=cfg.reverse,
            use_flux_shift=cfg.use_flux_shift,
            use_flux2_shift=cfg.use_flux2_shift,
        )
        # 注意：flux-shift 依赖 n_tokens；若推理用 flux-shift，需在 build 时传 n_tokens。
        # 这里默认 sd3 shift（与 leo2 t2va 视音频一致），不需要 n_tokens。
        self.scheduler.set_timesteps(cfg.num_steps, device=device)
        # set_timesteps 写入 self.sigmas (N+1) 与 self.timesteps_full (N+1)
        self.sigmas = self.scheduler.sigmas.to(device=device, dtype=torch.float32)            # (N+1,)
        self.model_t_full = self.scheduler.timesteps_full.to(device=device, dtype=torch.float32)  # (N+1,)
        assert self.sigmas.numel() == cfg.num_steps + 1, (
            f"expected {cfg.num_steps + 1} sigmas, got {self.sigmas.numel()}"
        )

    def flow_time(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: 整数张量 -> 对应的连续 flow-time（用于 path_sampler 构造 x_t/x_r）。"""
        return self.sigmas[idx]

    def model_t(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: 整数张量 -> 对应的 model-t（喂给模型 timesteps / audio_timesteps）。"""
        return self.model_t_full[idx]


def build_modality_axis(shift: float, cfg: GanTimeConfig, device: torch.device) -> ModalityAxis:
    """构造单模态离散轴。video 传 cfg.flow_shift_video，audio 传 cfg.flow_shift_audio。"""
    return ModalityAxis(shift=shift, cfg=cfg, device=device)


def sample_shared_step_index(
    bsz: int,
    cfg: GanTimeConfig,
    device: torch.device,
    broadcast_group: Optional[dist.ProcessGroup] = None,
    broadcast_src: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """采样【视音频共享】的离散步 index。

    返回 (idx_t, idx_r)，均为 (bsz,) 的 long 张量，且：
        idx_t ~ randint(0, N+1)
        idx_r = clamp(idx_t - K, min=0)

    与源 GAN 一致：video / audio 共用同一离散步 index（在同一采样步），
    连续时间在 build_*_axis 里因各自 shift 而不同。

    broadcast_group / broadcast_src：若提供，则在该组内广播 idx，保证
    同一个样本的所有并行 rank（如 CP 组）拿到完全一致的 idx，避免 collective 错位。
    若要求全局一致（所有 rank 同一组 idx），传入全局组与 src=0。
    """
    idx_t = torch.randint(
        low=0, high=cfg.num_steps + 1, size=(bsz,),
        device=device, dtype=torch.long, generator=generator,
    )
    idx_r = torch.clamp(idx_t - cfg.step_jump, min=0)

    if broadcast_group is not None and dist.is_initialized():
        src = broadcast_src if broadcast_src is not None else dist.get_global_rank(broadcast_group, 0)
        pair = torch.stack([idx_t, idx_r], dim=0).contiguous()
        dist.broadcast(pair, src=src, group=broadcast_group)
        idx_t, idx_r = pair[0], pair[1]
    return idx_t, idx_r


def gather_times_for_index(axis: ModalityAxis, idx_t: torch.Tensor, idx_r: torch.Tensor):
    """把一对离散 index 转成某模态的 (flow-time t, flow-time r, model-t t, model-t r)。

    返回的四个张量形状均与 idx 相同（通常 (bsz,)）：
        t_flow, r_flow   —— 用于 path_sampler.plan 构造 x_t / real_x_r
        t_model, r_model —— 用于模型 timesteps 输入（teacher 抽特征 / student 预测）
    """
    t_flow = axis.flow_time(idx_t)
    r_flow = axis.flow_time(idx_r)
    t_model = axis.model_t(idx_t)
    r_model = axis.model_t(idx_r)
    return t_flow, r_flow, t_model, r_model
