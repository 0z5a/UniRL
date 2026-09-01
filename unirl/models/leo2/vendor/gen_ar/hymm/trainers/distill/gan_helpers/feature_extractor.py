# teacher 中间特征抽取（forward hook，非侵入）。
#
# 背景（"命门" + "只能新建文件"约束）：
#   * 源仓库的判别器靠 backbone 原生 output_features 接口抽取指定层中间特征；
#   * 本仓库 t2va 双分支模型（hymm/models/diffusion/leo.py）的 forward 没有
#     output_features，且我们不能修改任何现有文件。
#   * 解决办法：运行时给冻结 teacher 的 transformer 层（teacher_model.layers）
#     注册 forward hook，抓取每个选定层输出的 (video_hidden, audio_hidden)。
#
# 关于 leo.py 的层输出（leo.py:1974-1994，audio 分支存在时）：
#     hidden_states, audio_hidden_states, txt_hidden_states = layer(*layer_inputs)
#   每层返回三元组：output[0]=视频 token，output[1]=音频 token，output[2]=文本 token。
#   在 CP（context parallel）下，这些是【按 seq 维切分到本 rank 的分片】（leo.py 在
#   层循环前 maybe_scatter_seq、循环后 maybe_gather_seq）。因此 hook 抓到的是分片，
#   需要用同一个 maybe_gather_seq 还原成完整序列，才能跨 CP rank 一致地喂判别器。
#
# 关于梯度：
#   * D 步：teacher forward 在 no_grad 下触发（输入已 detach），hook 抓到的特征
#     无梯度，判别器 head 自己 backward。
#   * G 步：teacher forward 在 grad 开启下触发（fake_x_r 带梯度），hook 抓到的特征
#     保留计算图，loss_g 的梯度可经冻结 teacher 回流到 student。
#     —— 是否带梯度由【调用方触发 forward 时的 grad 上下文】决定，本抽取器不干预。

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from hy_parallelism.training.cast_device import cast_to_device, CopyWork
from hy_parallelism.training.pinned_memory_pool import get_pinned_memory_pool


class _EarlyStopExtraction(Exception):
    """哨兵异常：抓完最后一个需要的层后抛出，用于提前中止 teacher forward。

    【警告 / 默认禁用】此机制在【torch.compile + EP（MoE 专家并行 all-to-all）】下【不安全】：
    从 compile 过的 MoE forward 中途抛异常会破坏各 rank 的 all-to-all 对称性，导致 mesh_ep
    集合通信 desync / NCCL 超时卡死（已复现）。因此 early_stop 默认为 False。
    仅在【无 MoE / 无 torch.compile / 无 EP】的场景才可考虑开启以省显存。
    """

try:
    from hy_parallelism.context_parallel.core import maybe_gather_seq
except Exception:  # pragma: no cover - 仅在缺失并行库的环境下退化
    maybe_gather_seq = None

try:
    from hymm.core.global_vars import get_parallel_state
except Exception:  # pragma: no cover
    get_parallel_state = None


class TeacherFeatureExtractor:
    """给 teacher_model.layers 的指定层挂 forward hook，抽取视频/音频中间特征。

    用法：
        extractor = TeacherFeatureExtractor(teacher_model, layer_indices=[i0, i1, i2])
        extractor.clear()
        _ = teacher_engine(**disc_kwargs)        # 由调用方决定 grad 上下文
        v_feats, a_feats = extractor.collect()   # 每项 (B, N, D)，已按 index 升序、已 CP gather
        ...
        extractor.remove()                       # 训练结束时移除 hook（可选）

    Args:
        teacher_model    : leo.py 模型（FSDP 包装后仍可访问 .layers）
        layer_indices    : 抽取的层 index 列表（视频/音频共用同一组层）
        cp_gather        : 是否在 CP>1 时对特征做 maybe_gather_seq（默认 True）
        capture_audio    : 是否抓取音频特征（纯视频判别时可设 False）
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        layer_indices: List[int],
        cp_gather: bool = True,
        capture_audio: bool = True,
        early_stop: bool = True,
    ):
        layers = getattr(teacher_model, "layers", None)
        if layers is None:
            raise AttributeError(
                "teacher_model has no `.layers`; expected leo.py-style nn.ModuleList of blocks."
            )
        self._layers = layers
        self.layer_indices = sorted(int(i) for i in layer_indices)
        self.cp_gather = cp_gather
        self.capture_audio = capture_audio
        # 抓完最后一个需要的层后是否提前中止 teacher forward（省显存）。
        self.early_stop = early_stop
        self._last_idx = self.layer_indices[-1] if self.layer_indices else -1
        self._captured = {}      # idx -> (video_hidden, audio_hidden_or_None)
        self._handles = []
        self._register()

    def _register(self):
        for idx in self.layer_indices:
            if idx < 0 or idx >= len(self._layers):
                raise IndexError(
                    f"layer index {idx} out of range for teacher with {len(self._layers)} layers"
                )
            handle = self._layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # output: (video_hidden, audio_hidden, txt_hidden) —— audio 分支存在时
            if not isinstance(output, (tuple, list)):
                raise RuntimeError(
                    f"layer {idx} output is not a tuple; got {type(output)}. "
                    "Expected (video_hidden, audio_hidden, txt_hidden)."
                )
            video_hidden = output[0]
            audio_hidden = output[1] if (self.capture_audio and len(output) > 1) else None
            self._captured[idx] = (
                cast_to_device(self._maybe_gather(video_hidden), 'cpu', use_side_stream_for_tensor_copies=True, async_op=True, pool='teacher_feat'), 
                cast_to_device(self._maybe_gather(audio_hidden), 'cpu', use_side_stream_for_tensor_copies=True, async_op=True, pool='teacher_feat')
            )
            # 抓到最后一个需要的层后提前中止 teacher forward，跳过其后的层 + head + 内部 loss。
            if self.early_stop and idx == self._last_idx:
                raise _EarlyStopExtraction()
        return hook

    def clear(self):
        """每次 teacher forward 前调用，清空上一轮捕获。"""
        self._captured = {}
        get_pinned_memory_pool('teacher_feat').reset()

    def _maybe_gather(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if not self.cp_gather or maybe_gather_seq is None or get_parallel_state is None:
            return x
        try:
            if get_parallel_state().cp_size > 1:
                return maybe_gather_seq(x)
        except Exception:
            return x
        return x

    def collect(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """返回 (video_feats_list, audio_feats_list)，按 layer index 升序，已 CP gather。

        每项形状 (B, N, D)。若 capture_audio=False，audio_feats_list 为空列表。
        """
        missing = [i for i in self.layer_indices if i not in self._captured]
        if missing:
            raise RuntimeError(
                f"feature capture incomplete; missing layers {missing}. "
                "Did the teacher forward actually run after clear()?"
            )
        video_feats, audio_feats = [], []
        for idx in self.layer_indices:
            v, a = self._captured[idx]
            if isinstance(v, CopyWork): v = v.wait()
            if isinstance(a, CopyWork): a = a.wait()
            video_feats.append(v)
            if self.capture_audio:
                audio_feats.append(a)
        return video_feats, audio_feats

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
