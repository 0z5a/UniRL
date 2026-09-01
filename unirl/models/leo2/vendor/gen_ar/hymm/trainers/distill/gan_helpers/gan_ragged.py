# GAN 蒸馏的 ragged（packed）数学辅助。
#
# 本仓库训练是 packed/ragged：每个字段是 list[B]，每项是 [N_i, C, ...] 的 stacked
# 张量（或 list[N_i]）；时间字段 t/r 是 list[B] 的 1-D [N_i]；模型 prediction 比
# 数据字段多一层嵌套（list[B] of list[N_i] of [1, C, ...]）。
#
# 这里直接复用 imf_loss 已经验证过的 ragged walker 原语（_walk_fields /
# _walk_fields_time / _match_ndim / _broadcast_time / _get / _index_time），
# 保证与现有 iMF 训练完全一致的 ragged 语义。
#
# 路径约定（leo2 t2va，reverse 线性 flow matching）：
#     x_t = (1 - t) * x1 + t * x0 ，  ut = d/dt x_t = x0 - x1
#   => x1(clean) = x_t - t * ut
#      x0(noise) = x1 + ut
#   在 GAN 自定义时刻 r：x_r = (1 - r) * x1 + r * x0
#
# 【给 infra 同事】
#   * 本仓库训练数据是 packed/ragged：每个字段是 list[B]，每项是 [N_i, C, ...] 的 stacked
#     张量；时间字段 t/r 是 list[B] 的 1-D [N_i]。这些辅助全部沿用 imf_loss 的 walker 原语
#     （_walk_fields / _walk_fields_time），逐叶子做四则运算，语义与现有 iMF 训练一致。
#   * 这些函数都是【纯张量代数】（无通信、无 autograd 特殊处理），是 GAN 里
#     "反推 x0/x1、构造 x_t/fake_x_r/real_x_r" 的数学工具，计算量相对模型 forward 很小。
#   * split_cond / attach_cond 处理 video 的 latent-channel-extend 布局（latents=2C+1 通道，
#     其中前 C 是 base latent、后 C+1 是与时间无关的 conditioning）；flow 数学只在 base 通道
#     上做，喂模型前把 conditioning 拼回。audio 无扩展（tail=None）。

from typing import Optional

import torch
import torch.distributed as dist

from ....diffusion.flow.imf_loss import (
    _walk_fields,
    _walk_fields_time,
    _match_ndim,
    _is_seq,
)


# --------------------------------------------------------------------------- #
# x0 / x1 反推（assumption 1：从数据管道的 (x_t, t, ut) 恢复干净/噪声端点）
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# conditioning 通道拆分 / 拼回
#   本仓库 video 走 latent_channel_extend（fl2v/i2v/t2v）：
#       latents = cat([x_t(C), extended_latents(C), extended_mask(1)]) -> 2C+1 通道，
#       而 ut（目标速度）仍为 C 通道（只对 base latent 有意义）。
#   audio 无扩展：latents 通道 == ut 通道，tail=None。
#   leaf 形状 [N_i, C, ...]，通道在 dim1。
# 因此：flow 数学（反推 x0/x1、构造 x_t_gan / fake_x_r / real_x_r）只在 base 通道上做；
#      喂 student/teacher 前，把与时间无关的 conditioning tail 原样拼回。
# --------------------------------------------------------------------------- #
def leaf_dtype(x):
    """取 ragged 结构第一个叶子的 dtype（用于 attach 时对齐模型输入 dtype）。"""
    if _is_seq(x):
        return leaf_dtype(x[0])
    return x.dtype


def split_cond(latents, ut):
    """按 ut 的通道数把 latents 切成 (base, tail)。ragged，结构随 latents。

    base = latents[:, :C]（C = ut 通道数）；
    tail = latents[:, C:]（无扩展时为 None）。
    """
    if _is_seq(latents):
        bases, tails = [], []
        for lat, u in zip(latents, ut):
            b, tl = split_cond(lat, u)
            bases.append(b)
            tails.append(tl)
        return bases, tails
    c = ut.shape[1]
    if latents.shape[1] == c:
        return latents, None
    return latents[:, :c].contiguous(), latents[:, c:].contiguous()


def attach_cond(base, tail, out_dtype=None):
    """把 conditioning tail 沿 dim1 拼回 base（喂 student/teacher）。ragged，结构随 base。

    tail 叶子为 None（audio）时原样返回 base；out_dtype 指定则统一转到该 dtype
    （保持与模型原始输入 latents 相同的 dtype，避免 cat/前向 dtype 冲突）。
    梯度：base 若带梯度（G 步），cast/cat 后仍保留；tail 为常量条件。
    """
    if _is_seq(base):
        return [attach_cond(b, tl, out_dtype) for b, tl in zip(base, tail)]
    if tail is None:
        return base if out_dtype is None else base.to(out_dtype)
    dt = out_dtype if out_dtype is not None else tail.dtype
    return torch.cat([base.to(dt), tail.to(dt)], dim=1)


def recover_clean(x_t, t, ut):
    """x1(clean) = x_t - t * ut。driver = x_t（保持 latents 结构）。"""
    def leaf(xt, u, t_b):
        u = _match_ndim(u.float(), xt)
        return xt.float() - t_b * u
    return _walk_fields_time((x_t, ut), t, leaf)


def recover_noise(x1, ut):
    """x0(noise) = x1 + ut。"""
    def leaf(c, u):
        u = _match_ndim(u.float(), c)
        return c.float() + u
    return _walk_fields((x1, ut), leaf)


def reconstruct_xt(x1, x0, t):
    """x_t = (1 - t) * x1 + t * x0。driver = x1（保持 latents 结构）。"""
    def leaf(c, n, t_b):
        n = _match_ndim(n.float(), c)
        return (1.0 - t_b) * c.float() + t_b * n.float()
    return _walk_fields_time((x1, x0), t, leaf)


# --------------------------------------------------------------------------- #
# fake_x_r = x_t_gan - (t - r) * pred
#   driver = x_t_gan（latents 结构），pred 比它多一层（list[N_i] / 多一个前导 1），
#   _walk_fields_time 以 x_t_gan 为 driver、把 pred 沿 dim0 对齐，输出仍是 latents 结构。
# --------------------------------------------------------------------------- #
def fake_from_pred(x_t_gan, pred, time_diff):
    """time_diff = t_gan - r_gan（ragged 时间）。返回与 x_t_gan 同结构的 fake_x_r。"""
    def leaf(xt, pr, td_b):
        pr = _match_ndim(pr.float(), xt)
        return xt.float() - td_b * pr
    return _walk_fields_time((x_t_gan, pred), time_diff, leaf)


# --------------------------------------------------------------------------- #
# 离散步 index 采样（ragged，匹配时间结构）+ 转连续 flow-time / model-t
# --------------------------------------------------------------------------- #
def sample_idx_like(
    t_struct,
    num_steps: int,
    step_jump: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
):
    """按 t 的 ragged 结构采样【共享】离散步 index。

    返回 (idx_t, idx_r)，结构与 t_struct 一致，每个叶子是 long 张量：
        idx_t ~ randint(0, num_steps+1)
        idx_r = clamp(idx_t - step_jump, 0)
    video / audio 各自调用一次（共享 index 的含义见 gan_time_utils）。
    """
    if _is_seq(t_struct):
        pairs = [sample_idx_like(x, num_steps, step_jump, device, generator) for x in t_struct]
        idx_t = [p[0] for p in pairs]
        idx_r = [p[1] for p in pairs]
        return idx_t, idx_r
    shape = t_struct.shape
    idx_t = torch.randint(
        low=0, high=num_steps + 1, size=shape, device=device, dtype=torch.long, generator=generator
    )
    idx_r = torch.clamp(idx_t - step_jump, min=0)
    return idx_t, idx_r


def broadcast_idx_(idx, src, group):
    """ragged in-place broadcast（CP 组内对齐 index，避免 collective 错位）。"""
    if _is_seq(idx):
        for x in idx:
            broadcast_idx_(x, src, group)
    else:
        dist.broadcast(idx, src=src, group=group)


def idx_to_flow(idx, axis):
    """ragged idx -> flow-time（用 axis.sigmas 查表，保持结构）。"""
    if _is_seq(idx):
        return [idx_to_flow(x, axis) for x in idx]
    return axis.sigmas[idx]


def idx_to_model_t(idx, axis):
    """ragged idx -> model-t（用 axis.model_t_full 查表，保持结构）。"""
    if _is_seq(idx):
        return [idx_to_model_t(x, axis) for x in idx]
    return axis.model_t_full[idx]


def time_diff(a, b):
    """a - b（ragged，逐叶子）。"""
    if _is_seq(a):
        return [time_diff(x, y) for x, y in zip(a, b)]
    return a.float() - b.float()
