import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange
from transformers.utils.generic import ModelOutput

from .activation_layers import get_activation_layer
from .attn_layers import apply_rotary_emb, attention
from .embed_layers import TimestepEmbedder, PatchEmbed, ConditionProjection
from .mlp_layers import MLP, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate, ckpt_wrapper
from .norm_layers import get_norm_layer
from .posemb_layers import get_nd_rotary_pos_embed
from ..multimodal.hunyuan_multimodal_state import HunyuanMultimodalState
from ...utils.helpers import default
from ...utils.states import DataClassMixin

Messages = list[dict[str, Any]]


@dataclass
class AriesOutput(ModelOutput):
    """
    Base class for hunyuan multimodal model (generalized autoregressive) outputs.

    Args:
        losses (dict[str, torch.Tensor], optional): A dictionary of loss components.
            Language modeling loss and diffusion loss are usually included.
        diffusion_prediction (torch.Tensor, optional): The predicted noise or denoised images
            from the diffusion modeling.
    """

    losses: Optional[dict[str, torch.Tensor]] = None
    diffusion_prediction: Optional[torch.Tensor] = None


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qkv_bias: bool = False,
        attn_mode: str = 'torch',
        reverse: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.deterministic = False
        self.reverse = reverse
        self.attn_mode = "self_flash" if attn_mode == "flash" else attn_mode
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

        self.cond_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.cond_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.cond_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.cond_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.cond_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.cond_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.cond_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.cond_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs
        )

    def set_attn_mode(self, new_mode):
        if new_mode != "torch":
            raise NotImplementedError(f"Only support 'torch' mode, got {new_mode}.")
        self.attn_mode = new_mode

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = (
            self.img_mod(vec).chunk(6, dim=-1)
        )
        cond_mod1_shift, cond_mod1_scale, cond_mod1_gate, cond_mod2_shift, cond_mod2_scale, cond_mod2_gate = (
            self.cond_mod(vec).chunk(6, dim=-1)
        )

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
            img_q, img_k = img_qq, img_kk

        # Prepare cond for attention.
        cond_modulated = self.cond_norm1(cond)
        cond_modulated = modulate(cond_modulated, shift=cond_mod1_shift, scale=cond_mod1_scale)
        cond_qkv = self.cond_attn_qkv(cond_modulated)
        cond_q, cond_k, cond_v = rearrange(cond_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        # Apply QK-Norm if needed.
        cond_q = self.cond_attn_q_norm(cond_q).to(cond_v)
        cond_k = self.cond_attn_k_norm(cond_k).to(cond_v)

        if self.reverse:
            # Run actual attention.
            q = torch.cat((img_q, cond_q), dim=1)
            k = torch.cat((img_k, cond_k), dim=1)
            v = torch.cat((img_v, cond_v), dim=1)

            attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask, deterministic=self.deterministic)
            img_attn, cond_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]
        else:
            # Run actual attention.
            q = torch.cat((cond_q, img_q), dim=1)
            k = torch.cat((cond_k, img_k), dim=1)
            v = torch.cat((cond_v, img_v), dim=1)

            assert "flash" not in self.attn_mode, "Flash attention not supported in non-reverse mode yet."
            attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask, deterministic=self.deterministic)
            cond_attn, img_attn = attn[:, :cond.shape[1]], attn[:, cond.shape[1]:]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), gate=img_mod2_gate)

        # Calculate the cond bloks.
        cond = cond + apply_gate(self.cond_attn_proj(cond_attn), gate=cond_mod1_gate)
        cond = cond + apply_gate(self.cond_mlp(modulate(self.cond_norm2(cond), shift=cond_mod2_shift, scale=cond_mod2_scale)), gate=cond_mod2_gate)

        return img, cond


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = 'gelu_tanh',
        qk_norm: bool = True,
        qk_norm_type: str = 'rms',
        qk_scale: float = None,
        attn_mode: str = 'torch',
        reverse: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.deterministic = False
        self.reverse = reverse
        self.attn_mode = "self_flash" if attn_mode == "flash" else attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)

    def set_attn_mode(self, new_mode):
        if new_mode != "torch":
            raise NotImplementedError(f"Only support 'torch' mode, got {new_mode}.")
        self.attn_mode = new_mode

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        vec: torch.Tensor,
        cond_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = (
            self.modulation(vec).chunk(3, dim=-1)
        )
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            if self.reverse:
                img_q, cond_q = q[:, :-cond_len, :, :], q[:, -cond_len:, :, :]
                img_k, cond_k = k[:, :-cond_len, :, :], k[:, -cond_len:, :, :]
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                    f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
                img_q, img_k = img_qq, img_kk
                q = torch.cat((img_q, cond_q), dim=1)
                k = torch.cat((img_k, cond_k), dim=1)
            else:
                cond_q, img_q = q[:, :cond_len, :, :], q[:, cond_len:, :, :]
                cond_k, img_k = k[:, :cond_len, :, :], k[:, cond_len:, :, :]
                img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
                assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, \
                    f'img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}'
                img_q, img_k = img_qq, img_kk
                q = torch.cat((cond_q, img_q), dim=1)
                k = torch.cat((cond_k, img_k), dim=1)

        # Compute attention.
        attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask, deterministic=self.deterministic)
        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class Aries(HunyuanMultimodalState, ModelMixin, ConfigMixin):
    """
    Transformer model for flow matching on sequences.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206,
               https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """
    @register_to_config
    def __init__(
        self,
        args,  # used to pass some other model-irrelevant configs, such as gradient_checkpoint
        model_config,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.depth_double_blocks = model_config.get("depth_double_blocks", 19)
        self.depth_single_blocks = model_config.get("depth_single_blocks", 38)
        # Gradient checkpoint.
        self.gradient_checkpoint = getattr(args, "gradient_checkpoint", False)
        self.gradient_checkpoint_layers = getattr(args, "gradient_checkpoint_layers", -1)
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= self.depth_double_blocks + self.depth_single_blocks, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={self.depth_double_blocks + self.depth_single_blocks}."


        # Condition projection. Default to linear projection.
        self.condition_projection = model_config.get("condition_projection", "linear")
        self.condition_dim = model_config.get("condition_dim", None)
        self.use_attention_mask = model_config.get("use_attention_mask", False)

        self.patch_size = model_config.get("patch_size", 1)
        self.in_channels = model_config.get("in_channels", 4)
        self.out_channels = model_config.get("out_channels", self.in_channels)
        self.unpatchify_channels = self.out_channels
        self.reverse = model_config.get("reverse", False)

        self.num_heads = model_config.get("num_heads", 24)
        self.hidden_size = model_config.get("hidden_size", 3072)
        self.rope_dim_list = model_config.get("rope_dim_list", None)
        self.mlp_ratio = model_config.get("mlp_ratio", 4.0)
        self.mlp_act_type = model_config.get("mlp_act_type", "gelu_tanh")

        self.qkv_bias = model_config.get("qkv_bias", True)
        self.qk_norm = model_config.get("qk_norm", True)
        self.qk_norm_type = model_config.get("qk_norm_type", "rms")
        self.attn_mode = model_config.get("attn_mode", "torch")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )

        # image
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            **factory_kwargs
        )

        # condition
        if self.condition_projection == "linear":
            self.cond_in = ConditionProjection(
                self.condition_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported condition_projection: {self.condition_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size,
            get_activation_layer("silu"),
            **factory_kwargs
        )

        # blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    qkv_bias=self.qkv_bias,
                    attn_mode=self.attn_mode,
                    reverse=self.reverse,
                    **factory_kwargs
                )
                for _ in range(self.depth_double_blocks)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    attn_mode=self.attn_mode,
                    reverse=self.reverse,
                    **factory_kwargs
                )
                for _ in range(self.depth_single_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs
        )

    def get_printable_layers(self):
        return [self.double_blocks[0], self.single_blocks[0]]

    def set_attn_mode(self, new_mode):
        for block in self.double_blocks:
            block.set_attn_mode(new_mode)
        for block in self.single_blocks:
            block.set_attn_mode(new_mode)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor, # Should be in range(0, 1000).
        cond: torch.Tensor = None,
        cond_mask: torch.Tensor = None, # Now we don't use it.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        diffusion_loss_fn: Optional[Callable] = None,
    ) -> AriesOutput | tuple:
        img = x
        _, _, oh, ow = img.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # Prepare modulation vectors.
        # time modulation
        vec = self.time_in(t)

        # Embed image and text.

        img = self.img_in(img)
        if self.condition_projection == "linear":
            cond = self.cond_in(cond)
        else:
            raise NotImplementedError(f"Unsupported condition_projection: {self.condition_projection}")

        img_seq_len = img.shape[1]
        cond_seq_len = cond.shape[1]
        # Compute 'self-attention mask'.
        # Added by ckczzjzhang
        attn_mask = None
        if self.use_attention_mask:
            batch_size = img.shape[0]

            seq_len = cond_seq_len+img_seq_len

            img_mask = torch.ones(batch_size, img_seq_len, dtype=torch.bool, device=img.device)
            # batch_size x seq_len
            if self.reverse:
                concat_mask = torch.cat([img_mask, cond_mask], dim=1)
            else:
                concat_mask = torch.cat([cond_mask, img_mask], dim=1)
            if self.attn_mode == "flash":
                assert self.reverse, "Flash attention mode only supports reverse=True for now."
                attn_mask = concat_mask
            else:
                # batch_size x 1 x seq_len x seq_len
                attn_mask_1 = concat_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
                # batch_size x 1 x seq_len x seq_len
                attn_mask_2 = attn_mask_1.transpose(2, 3)
                # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
                attn_mask = (attn_mask_1 & attn_mask_2).bool()
                # avoids self-attention weight being NaN for text padding tokens
                attn_mask[:, :, :, 0] = True

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for layer_num, block in enumerate(self.double_blocks):
            double_block_args = [img, cond, attn_mask, vec, freqs_cis]
            if self.training and self.gradient_checkpoint and \
                    (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                img, cond = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *double_block_args, use_reentrant=False)
            else:
                img, cond = block(*double_block_args)

        # Merge cond and img to pass through single stream blocks.
        if self.reverse:
            x = torch.cat((img, cond), 1)
        else:
            x = torch.cat((cond, img), 1)

        # Compatible with MMDiT.
        if len(self.single_blocks) > 0:
            for layer_num, block in enumerate(self.single_blocks):
                single_block_args = [x, attn_mask, vec, cond_seq_len, (freqs_cos, freqs_sin)]
                if self.training and self.gradient_checkpoint and \
                        (self.gradient_checkpoint_layers == -1 or layer_num + len(self.double_blocks) < self.gradient_checkpoint_layers):
                    x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *single_block_args, use_reentrant=False)
                else:
                    x = block(*single_block_args)

        if self.reverse:
            img = x[:, :img_seq_len, ...]
        else:
            img = x[:, cond_seq_len:, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        img = self.unpatchify(img, th, tw)

        if return_dict:
            if diffusion_loss_fn is not None:
                loss = diffusion_loss_fn(model_output=img)["loss"].mean()
                return AriesOutput(losses={"loss": loss}, diffusion_prediction=img)
            else:
                return AriesOutput(diffusion_prediction=img)
        else:
            if diffusion_loss_fn is not None:
                return diffusion_loss_fn(model_output=img)["loss"], img
            return img

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def params_count(self):
        counts = {
            "double": sum([
                sum(p.numel() for p in block.img_attn_qkv.parameters()) +
                sum(p.numel() for p in block.img_attn_proj.parameters()) +
                sum(p.numel() for p in block.img_mlp.parameters()) +
                sum(p.numel() for p in block.cond_attn_qkv.parameters()) +
                sum(p.numel() for p in block.cond_attn_proj.parameters()) +
                sum(p.numel() for p in block.cond_mlp.parameters())
                for block in self.double_blocks
            ]),
            "single": sum([
                sum(p.numel() for p in block.linear1.parameters()) +
                sum(p.numel() for p in block.linear2.parameters())
                for block in self.single_blocks
            ]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


@dataclass
class AriesGenerationConfig(DataClassMixin):
    infer_steps: int
    guidance_scale: float
    flow_shift: float
    bot_task: str = "image"


class AriesHF(Aries):
    def __init__(self, *args, **kwargs):
        from hymm.core.global_vars import get_args
        from hymm.data_kits.utils.image_utils import ImageProcessor

        super().__init__(*args, **kwargs)
        self.args = get_args()

        # Initialize image processor
        self.image_processor = ImageProcessor(self.args)

        self._diffusion_pipeline = None
        self.vae = None
        self.text_encoder = None

    def load_generation_config(self, generation_config_path=None):

        _ = generation_config_path
        args = self.args

        self.generation_config = AriesGenerationConfig(
            infer_steps=args.diff_infer_steps,
            guidance_scale=args.diff_guidance_scale,
            flow_shift=default(args.sample_flow_shift, args.flow_shift),
        )

    @property
    def diffusion_pipeline(self):
        return self._diffusion_pipeline

    def build_diffusion_pipeline(self):
        if self._diffusion_pipeline is None:
            from ...diffusion import load_scheduler
            from ...diffusion import load_pipeline
            scheduler = load_scheduler(self.args)
            assert self.vae is not None, "VAE must be initialized before building diffusion pipeline."
            self._diffusion_pipeline = load_pipeline("text2image")(
                args=self.args,
                diffusion_model=self,
                scheduler=scheduler,
                text_encoder=self.text_encoder,
                vae=self.vae,
            )

    @staticmethod
    def check_inputs(prompt=None, message_list=None):
        if prompt is None and message_list is None:
            raise ValueError("Either `prompt` or `message_list` should be provided.")
        if prompt is not None and message_list is not None:
            raise ValueError("`prompt` and `message_list` cannot be provided at the same time.")
        if message_list is not None:
            if not isinstance(message_list, list):
                raise ValueError(f"`message_list` should be a list of messages, but got {type(message_list)}.")
            assert len(message_list) > 0, "`message_list` should be a non-empty list."
            for message in message_list:
                assert isinstance(message, list) or isinstance(message, dict), \
                    f"Each message should be a list of dicts or a dict, but got {type(message)}."

    @staticmethod
    def _validate_and_batchify_text(text, name, check_batch_size=None):
        if text is None:
            return text
        assert isinstance(text, str) or isinstance(text, list), \
            f"Input `{name}` should be a string or a list of strings, but got {type(text)}."
        if isinstance(text, str):
            text = [text]
        assert len(text) > 0 and all(isinstance(p, str) and len(p) > 0 for p in text), \
            f"Input `{name}` should be a non-empty list of non-empty strings, got {text}."
        if check_batch_size is not None:
            assert len(text) == check_batch_size, \
                f"Input `{name}` should have the same batch size as other inputs({check_batch_size}), got {len(text)}."
        return text

    def get_rope(self, image_info):
        if self.args.rope_type_extended == "2d":
            latents_size = [image_info.token_height, image_info.token_width]
            rope_dim_list = self.args.rope_dim_list
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list=rope_dim_list,
                start=latents_size,
                theta=self.args.rope_theta,
                use_real=True,
            )
        else:
            raise NotImplementedError(f"Unsupported rope_type_extended: {self.args.rope_type_extended}")
        return freqs_cos, freqs_sin

    @staticmethod
    def prepare_seed(seed, batch_size):
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [random.randint(0, 10_000_000) for _ in range(batch_size)]
        elif isinstance(seed, int):
            seeds = [seed for _ in range(batch_size)]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [int(seed[i]) for i in range(batch_size)]
            else:
                raise ValueError(f"Length of seed must be equal to the batch_size({batch_size}), got {seed}.")
        else:
            raise ValueError(f"Seed must be an integer, a list of integers, or None, got {seed}.")
        return seeds

    def generate_image(
            self,
            prompt: Optional[str | list[str]] = None,
            message_list: Optional[Messages | list[Messages]] = None,
            image_size=None,
            image_output_type="pil",
            generation_config: Optional[AriesGenerationConfig] = None,
            verbose=0,
            **kwargs,
    ):
        # 1. Sanity check
        self.check_inputs(prompt=prompt, message_list=message_list)
        gen_config = default(generation_config, self.generation_config)

        # 2. Format inputs
        batch_message_list = message_list
        batch_prompt = prompt

        #   -- 2.1 message_list
        if batch_message_list is not None:
            if isinstance(batch_message_list[0], dict):
                batch_message_list = [batch_message_list]
            batch_prompt = [message_list[0]["content"] for message_list in batch_message_list]

        batch_prompt = self._validate_and_batchify_text(batch_prompt, 'prompt')
        batch_size = len(batch_prompt)
        image_info = self.image_processor.build_gen_image_info(image_size)

        #   -- 2.3 seed
        seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

        # 3. Calculate RoPE
        cos, sin = self.get_rope(image_info=image_info)
        model_input_extra_kwargs = dict(
            freqs_cos=cos,
            freqs_sin=sin,
        )

        # 4. Log info
        if verbose >= 1:
            info_list = [
                ("batch_size", batch_size),
                ("prompt", batch_prompt)
            ]
            if generator is not None:
                info_list.extend([
                    ("seed", [g.initial_seed() for g in generator]),
                ])
            info_list.extend([
                ("image_size", f"{image_info.image_height}x{image_info.image_width}"),
                ("infer_steps", gen_config.infer_steps),
                ("guidance_scale", gen_config.guidance_scale),
                ("flow_shift", gen_config.flow_shift),
            ])
            max_key_len = max(len(k) for k, _ in info_list)
            info_str = "=" * 50 + \
                       "\nModel input info:\n" + \
                       "\n".join([f"    {k.rjust(max_key_len)}: {v}" for k, v in info_list]) + \
                       "\n--------------------------------------------------"
            print(info_str, flush=True)
            start_time = time.time()

        # 5. Generate images
        self.build_diffusion_pipeline()
        results = self.diffusion_pipeline(
            prompt=batch_prompt,
            height=image_info.image_height,
            width=image_info.image_width,
            num_inference_steps=gen_config.infer_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            output_type=image_output_type,
            model_input_extra_kwargs=model_input_extra_kwargs,
        )
        samples = results[0]

        if verbose >= 1:
            end_time = time.time()
            print(f"Generation completed in {end_time - start_time:.2f} seconds.", flush=True)

        return None, samples


#################################################################################
#                                   Aries Configs                                 #
#################################################################################

# DiT_CONFIG = {                                                                          # Attn+MLP / Total
#     'DiT-S/2': {'depth': 12, 'hidden_size': 576, 'num_heads': 6, 'mlp_ratio': 4},       #    64M   /   92M
#     'DiT-SP/2': {'depth': 12, 'hidden_size': 864, 'num_heads': 9, 'mlp_ratio': 4},      #   143M   /  204M
#     'DiT-B/2': {'depth': 12, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   255M   /  361M
#     'DiT-L/2': {'depth': 24, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   510M   /  712M
#     'DiT-XL/2': {'depth': 24, 'hidden_size': 1680, 'num_heads': 14, 'mlp_ratio': 4},    #   1.1B   /  1.5B
#     'DiT-XXL/2': {'depth': 36, 'hidden_size': 1920, 'num_heads': 16, 'mlp_ratio': 4},   #   2.1B   /  2.9B
#     'DiT-G/2': {'depth': 36, 'hidden_size': 2880, 'num_heads': 24, 'mlp_ratio': 4},     #   4.8B   /  6.6B
#     'DiT-T/2': {'depth': 40, 'hidden_size': 3840, 'num_heads': 32, 'mlp_ratio': 4},     #   9.4B   / 13.0B
# }

Aries_CONFIG = {                                                                         # Attn+MLP / Total
    'Aries-S/2': {                                                                       #    66M   /   94M
        'depth_double_blocks': 6,
        'depth_single_blocks': 12,
        'hidden_size': 480,
        'num_heads': 5,
        'mlp_ratio': 4
    },
    'Aries-SP/2': {                                                                      #    130M   /  184M
        'depth_double_blocks': 6,
        'depth_single_blocks': 12,
        'hidden_size': 672,
        'num_heads': 7,
        'mlp_ratio': 4,
    },
    'Aries-B/2': {                                                                       #   265M   /  373M
        'depth_double_blocks': 6,
        'depth_single_blocks': 12,
        'hidden_size': 960,
        'num_heads': 10,
        'mlp_ratio': 4,
    },
    'Aries-L/2': {                                                                       #   531M   /  738M
        'depth_double_blocks': 12,
        'depth_single_blocks': 24,
        'hidden_size': 960,
        'num_heads': 10,
        'mlp_ratio': 4,
    },
    'Aries-L/2-R': {                                                                       #   531M   /  738M
        'depth_double_blocks': 12,
        'depth_single_blocks': 24,
        'hidden_size': 960,
        'num_heads': 10,
        'mlp_ratio': 4,
        'reverse': True,
    },
    'Aries-XL/2': {                                                                      #   1.14B   /  1.58B
        'depth_double_blocks': 12,
        'depth_single_blocks': 24,
        'hidden_size': 1408,
        'num_heads': 11,
        'mlp_ratio': 4,
    },
    'Aries-XXL/2': {                                                                     #   2.04B   /  2.82B
        'depth_double_blocks': 18,
        'depth_single_blocks': 36,
        'hidden_size': 1536,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'Aries-G/2': {                                                                       #   4.6B   /  6.3B
        'depth_double_blocks': 18,
        'depth_single_blocks': 36,
        'hidden_size': 2304,
        'num_heads': 18,
        'mlp_ratio': 4,
    },
    'Aries-T/2': {                                                                       #   9.0B   / 12.5B
        'depth_double_blocks': 20,
        'depth_single_blocks': 40,
        'hidden_size': 3072,
        'num_heads': 24,
        'mlp_ratio': 4,
    },
    'AriesIT-T/2': {                                                                       #   9.0B   / 12.5B
        'depth_double_blocks': 20,
        'depth_single_blocks': 40,
        'hidden_size': 3072,
        'num_heads': 24,
        'mlp_ratio': 4,
        'reverse': True,
    },
}

MMDiT_CONFIG = {                                                                        # Attn+MLP / Total
    'MMDiT-S/2': {                                                                      #    61M   /   94M
        'depth_double_blocks': 11,
        'depth_single_blocks': 0,
        'hidden_size': 480,
        'num_heads': 5,
        'mlp_ratio': 4,
    },
    'MMDiT-SP/2': {                                                                     #    119M   /  184M
        'depth_double_blocks': 11,
        'depth_single_blocks': 0,
        'hidden_size': 672,
        'num_heads': 7,
        'mlp_ratio': 4,
    },
    'MMDiT-B/2': {                                                                      #   243M   /  373M
        'depth_double_blocks': 11,
        'depth_single_blocks': 0,
        'rope_dim_list':[12, 42, 42],
        'hidden_size': 960,
        'num_heads': 10,
        'mlp_ratio': 4,
    },
    'MMDiT-L/2': {                                                                      #   487M   /  738M
        'depth_double_blocks': 22,
        'depth_single_blocks': 0,
        'rope_dim_list':[12, 42, 42],
        'hidden_size': 960,
        'num_heads': 10,
        'mlp_ratio': 4,
    },
    'MMDiT-XL/2': {                                                                     #   1.05B   /  1.58B
        'depth_double_blocks': 22,
        'depth_single_blocks': 0,
        'hidden_size': 1408,
        'num_heads': 11,
        'mlp_ratio': 4,
    },
    'MMDiT-XXL/2': {                                                                    #   1.87B   /  2.82B
        'depth_double_blocks': 33,
        'depth_single_blocks': 0,
        'hidden_size': 1536,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'MMDiT-G/2': {                                                                      #   4.2B   /  6.3B
        'depth_double_blocks': 33,
        'depth_single_blocks': 0,
        'hidden_size': 2304,
        'num_heads': 18,
        'mlp_ratio': 4,
    },
    'MMDiT-T/2': {                                                                      #   8.37B   / 12.6B
        'depth_double_blocks': 44,
        'depth_single_blocks': 0,
        'hidden_size': 2816,
        'num_heads': 22,
        'mlp_ratio': 4,
    },
}
