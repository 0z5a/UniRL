# 视音频联合判别器 head + GAN hinge loss。
#
# 迁移自 HunyuanVideo_pureTorch/hymm/discriminator/model.py 的 JointDiscriminatorHead
# 与 gan_loss.py 的 get_gan_loss，保持"同时判别视频和音频"的设计：
#   * 对每层视频特征 / 音频特征各用一个带可学习 query token 的 CrossAttentionBlock
#     聚合成单 token；
#   * 所有 video / audio token 拼接成 joint context；
#   * LayerNorm + spectral_norm(Linear) 输出【一个】联合标量 logit。
# 这样判别器必须同时感知视频和音频才能打分，天然建模音画对齐。
#
# 与源实现的唯一区别：这里只有 head（可训练），backbone 特征由外部
# TeacherFeatureExtractor 通过 forward hook 从冻结 teacher 抽取后传入，
# 因此本文件不持有 backbone，也不涉及 SP/CP 同步（特征在传入前已 gather）。
#
# 【给 infra 同事】
#   * 判别器 head 是唯一可训练部件（冻结 teacher 只做特征提取器），参数量小（约百 M 级，
#     取决于抽取层数/维度），在 gan_trainer 里用 DDP【复制】（非 FSDP 分片）。
#   * get_gan_loss 是纯逐元素/reduce 运算（hinge/non_saturating/softplus），
#     G 步只用 fake_logit，D 步用 real+fake_logit。这就是全部的对抗 loss，无其它项。
#   * CrossAttentionBlock 里对超长 context 序列先做 adaptive_avg_pool1d 池化到
#     max_ctx_tokens（默认 512），是为降低显存/算力的轻量化手段；关掉设 <=0 即可。
#   * spectral_norm 会引入 power-iteration 的 buffer（_u/_v），DDP 下由 all-reduce/broadcast
#     保持各 rank 一致；若改并行方式需注意这些 buffer 的同步。

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


class CrossAttentionBlock(nn.Module):
    """用一个 query token 对一串 visual/audio token 做 cross-attention 聚合（带 spectral_norm）。"""

    def __init__(self, dim, num_heads=8, qk_norm=True, max_ctx_tokens=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm
        # 轻量化：context 序列过长时先自适应平均池化到该预算，显著降低 K/V + 注意力显存。
        # <=0 表示不池化。
        self.max_ctx_tokens = int(max_ctx_tokens)

        self.proj_q = spectral_norm(nn.Linear(dim, dim, bias=False))
        self.norm_context = RMSNorm(dim)
        self.proj_k = spectral_norm(nn.Linear(dim, dim, bias=False))
        self.proj_v = spectral_norm(nn.Linear(dim, dim, bias=False))
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj_o = spectral_norm(nn.Linear(dim, dim, bias=False))

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(dim, dim * 4)),
            nn.GELU(),
            spectral_norm(nn.Linear(dim * 4, dim)),
        )

    def forward(self, query_token, visual_tokens):
        """query_token: (B, 1, D); visual_tokens: (B, N, D) -> (B, 1, D)"""
        # 轻量化：序列过长时先池化到 max_ctx_tokens，降低 K/V 与注意力显存（对判别足够）。
        if self.max_ctx_tokens > 0 and visual_tokens.shape[1] > self.max_ctx_tokens:
            vt = visual_tokens.transpose(1, 2)                       # (B, D, N)
            # kevinkhwu: pytorch 的bug，需要 contigous
            #     RuntimeError: false INTERNAL ASSERT FAILED at "/root/source/pytorch/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu":708, please report a bug to PyTorch. Couldn't reduce launch bounds to accommodate sharedMemPerBlock limit
            vt = F.adaptive_avg_pool1d(vt.float().contiguous(), self.max_ctx_tokens).to(visual_tokens.dtype)
            visual_tokens = vt.transpose(1, 2).contiguous()         # (B, max_ctx_tokens, D)
        B, N, C = visual_tokens.shape
        residual = query_token
        q = self.proj_q(query_token)
        ctx_normed = self.norm_context(visual_tokens)
        k = self.proj_k(ctx_normed)
        v = self.proj_v(ctx_normed)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, 1, C)
        x = self.proj_o(x)
        x = x + residual

        residual_mlp = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + residual_mlp
        return x


class JointDiscriminatorHead(nn.Module):
    """联合判别 head：对若干层的视频特征 + 音频特征联合打分，输出一个标量 logit。

    Args:
        video_dim / audio_dim     : 视频 / 音频特征维度（= teacher 视频 / 音频 hidden_size）
        video_num_layers / audio_num_layers : 抽取的层数（= 抽取层 index 的个数）
        num_heads                 : cross-attention 头数
    """

    def __init__(self, video_dim, audio_dim, video_num_layers, audio_num_layers, num_heads=16,
                 max_ctx_tokens=512):
        super().__init__()
        self.video_num_layers = video_num_layers
        self.audio_num_layers = audio_num_layers
        # joint_dim 与源实现一致：每层 (video_dim + audio_dim)，按视频层数计。
        self.joint_dim = video_num_layers * video_dim + audio_num_layers * audio_dim

        self.video_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=video_dim, num_heads=num_heads, max_ctx_tokens=max_ctx_tokens)
            for _ in range(video_num_layers)
        ])
        self.video_query_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, video_dim) * 0.02) for _ in range(video_num_layers)
        ])

        self.audio_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=audio_dim, num_heads=num_heads, max_ctx_tokens=max_ctx_tokens)
            for _ in range(audio_num_layers)
        ])
        self.audio_query_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, audio_dim) * 0.02) for _ in range(audio_num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.joint_dim)
        self.final_proj = spectral_norm(nn.Linear(self.joint_dim, 1))

    def forward(
        self,
        video_features_list: List[torch.Tensor],
        audio_features_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """video_features_list / audio_features_list: 每项 (B, N, D)。返回 (B, 1) 联合 logit。"""
        assert len(video_features_list) == self.video_num_layers, (
            f"expected {self.video_num_layers} video feats, got {len(video_features_list)}"
        )
        assert len(audio_features_list) == self.audio_num_layers, (
            f"expected {self.audio_num_layers} audio feats, got {len(audio_features_list)}"
        )
        B = video_features_list[0].shape[0]
        outputs = []
        for i, feat in enumerate(video_features_list):
            q = self.video_query_tokens[i].expand(B, -1, -1).to(feat.dtype)
            outputs.append(self.video_blocks[i](q, feat.cuda()))
        for i, feat in enumerate(audio_features_list):
            q = self.audio_query_tokens[i].expand(B, -1, -1).to(feat.dtype)
            outputs.append(self.audio_blocks[i](q, feat.cuda()))

        joint_out = torch.cat(outputs, dim=-1)          # (B, 1, joint_dim)
        normed = self.final_norm(joint_out)
        logits = self.final_proj(normed)                # (B, 1, 1)
        return logits.squeeze(-1)                        # (B, 1)


def get_gan_loss(real_logit, fake_logit, loss_type: str, adv_type: str):
    """GAN 损失（迁移自源 gan_loss.py）。

    adv_type == "G": 只用 fake；adv_type == "D": 用 real + fake。
    支持 hinge / non_saturating / softplus。返回 (loss_g, loss_d)，其中一个为零张量。
    """
    if loss_type == "hinge":
        if adv_type == "G":
            loss_g = torch.mean(torch.relu(1.0 - fake_logit.float()))
            loss_d = torch.zeros_like(loss_g)
        elif adv_type == "D":
            loss_d = (
                torch.mean(torch.relu(fake_logit.float() + 1.0))
                + torch.mean(torch.relu(1.0 - real_logit.float()))
            )
            loss_g = torch.zeros_like(loss_d)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")
    elif loss_type == "non_saturating":
        if adv_type == "G":
            loss_g = -torch.mean(torch.log(torch.sigmoid(fake_logit.float()) + 1e-8))
            loss_d = torch.zeros_like(loss_g)
        elif adv_type == "D":
            loss_d = -torch.mean(
                torch.log(torch.sigmoid(real_logit.float()) + 1e-8)
                + torch.log(1.0 - torch.sigmoid(fake_logit.float()) + 1e-8)
            )
            loss_g = torch.zeros_like(loss_d)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")
    elif loss_type == "softplus":
        if adv_type == "G":
            loss_g = torch.mean(F.softplus(-fake_logit.float()))
            loss_d = torch.zeros_like(loss_g)
        elif adv_type == "D":
            loss_d = torch.mean(F.softplus(real_logit.float()) + F.softplus(-fake_logit.float()))
            loss_g = torch.zeros_like(loss_d)
        else:
            raise ValueError(f"Unknown adv_type: {adv_type}")
    else:
        raise NotImplementedError(f"Unsupported GAN loss type: {loss_type}")
    return loss_g, loss_d
