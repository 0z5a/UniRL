from typing import Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import PatchEmbed, TimestepEmbedder, ConditionProjection
from .attn_layers import SelfAttentionLayer, CrossAttentionLayer
from .mlp_layers import MLP, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate, ckpt_wrapper


def xavier_initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def zero_initialize(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 mlp_drop_rate=0,
                 qk_norm=True,
                 act_type="silu",
                 norm_type="layer",
                 attn_mode="flash",
                 modulate_type="dit",
                 deterministic=False,
                 dtype=None,
                 device=None,
                 layer=None,
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.layer = layer

        norm_layer = get_norm_layer(norm_type)

        self.norm1 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.attn1 = SelfAttentionLayer(dim=hidden_size,
                                        num_heads=num_heads,
                                        qkv_bias=True,
                                        qk_norm=qk_norm,
                                        attn_drop=attn_drop_rate,
                                        proj_drop=proj_drop_rate,
                                        norm_type=norm_type,
                                        attn_mode="self_flash" if attn_mode == "flash" else attn_mode,
                                        deterministic=deterministic,
                                        **factory_kwargs,
                                        )

        self.norm2 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.attn2 = CrossAttentionLayer(qdim=hidden_size,
                                         kdim=hidden_size,
                                         num_heads=num_heads,
                                         qkv_bias=True,
                                         qk_norm=qk_norm,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=proj_drop_rate,
                                         norm_type=norm_type,
                                         attn_mode="cross_flash" if attn_mode == "flash" else attn_mode,
                                         deterministic=deterministic,
                                         **factory_kwargs,
                                         )

        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(in_channels=hidden_size,
                       hidden_channels=int(hidden_size * mlp_ratio),
                       act_layer=act_layer,
                       drop=mlp_drop_rate,
                       **factory_kwargs,
                       )

        # Initialize the weights
        self.attn1.apply(xavier_initialize)
        self.attn2.apply(xavier_initialize)
        self.mlp.apply(xavier_initialize)

        # Modulation
        self.modulate_type = modulate_type
        if modulate_type == "dit":
            # self.adaLN_modulation = ModulateDiT(hidden_size, 6, act_layer, **factory_kwargs)
            self.adaLN_modulation = nn.Sequential(
                act_layer(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True, **factory_kwargs)
            )
            # Zero-initialize the modulation
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

        elif modulate_type == "pixart":
            self.adaLN_table = nn.Parameter(torch.randn(6, hidden_size, **factory_kwargs) / hidden_size ** 0.5)

        elif modulate_type == "lumina":
            self.adaLN_modulation = ModulateDiT(hidden_size, 4, act_layer, **factory_kwargs)

        else:
            raise ValueError(f"Unknown modulate_type: {modulate_type}")

    def set_attn_mode(self, new_mode):
        if new_mode == "flash":
            self.attn1.set_attn_mode("self_flash")
            self.attn2.set_attn_mode("cross_flash")
        else:
            self.attn1.set_attn_mode(new_mode)
            self.attn2.set_attn_mode(new_mode)

    def enable_deterministic(self):
        self.attn1.enable_deterministic()
        self.attn2.enable_deterministic()

    def disable_deterministic(self):
        self.attn1.disable_deterministic()
        self.attn2.disable_deterministic()

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor = None,
                cond: torch.Tensor = None,
                attn_mask: torch.Tensor = None,
                freqs_cis: tuple = None,
                ):
        if self.modulate_type == "dit":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
            gate_tanh = False
        elif self.modulate_type == "pixart":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_table[None] + c.reshape(c.size(0), 6, -1)
            ).unbind(dim=1)
            gate_tanh = False
        elif self.modulate_type == "lumina":
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa, shift_mlp = None, None
            gate_tanh = True
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = None, None, None, None, None, None
            gate_tanh = False

        # Self-Attention
        x = x + apply_gate(
            self.attn1(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis),
            gate_msa, tanh=gate_tanh
        )

        # Cross-Attention
        x = x + self.attn2(self.norm2(x), cond, attn_mask)

        # FFN Layer
        x = x + apply_gate(
            self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp)),
            gate_mlp, tanh=gate_tanh
        )

        return x


class DiT(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone.
    Reference:
    [1] DiT: http://arxiv.org/abs/2212.09748, https://github.com/facebookresearch/DiT
    [2] HunyuanDiT: https://arxiv.org/abs/2405.08748, https://github.com/Tencent/HunyuanDiT
    [3] Lumina-T2X: http://arxiv.org/abs/2405.05945, https://github.com/Alpha-VLLM/Lumina-T2X
    [4] Lumina-Next: http://arxiv.org/abs/2406.18583, https://github.com/Alpha-VLLM/Lumina-T2X
    [5] ADM: https://arxiv.org/pdf/2105.05233, https://github.com/openai/guided-diffusion
    [6] LI-DiT: http://arxiv.org/abs/2406.11831

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """
    @register_to_config
    def __init__(self,
                 args, # used to pass some other model-irrelevant configs, such as gradient_checkpoint
                 model_config,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 ):
        """
        Initialize the DiT model.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.depth = model_config.get("depth", 40)
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= self.depth, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={self.depth}."

        # Condition projection. Default to linear projection.
        self.condition_projection = model_config.get("condition_projection", "linear")
        self.condition_dim = model_config.get("condition_dim", None)
        self.use_attention_mask = model_config.get("use_attention_mask", False)
        
        self.patch_size = model_config.get("patch_size", 2)
        self.in_channels = model_config.get("in_channels", 4)
        self.out_channels = model_config.get("out_channels", self.in_channels)
        self.unpatchify_channels = self.out_channels
        
        self.num_heads = model_config.get("num_heads", 32)
        self.hidden_size = model_config.get("hidden_size", 3840)
        self.rope_dim_list = model_config.get("rope_dim_list", None)
        self.mlp_ratio = model_config.get("mlp_ratio", 4.0)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )

        # Modulation type.
        # See more details (DiT): https://github.com/facebookresearch/DiT
        self.modulate_type = model_config.get("modulate_type", "dit")

        # Common hyperparameters
        self.attn_drop_rate = model_config.get("attn_drop_rate", 0.0)
        self.proj_drop_rate = model_config.get("proj_drop_rate", 0.0)
        self.mlp_drop_rate = model_config.get("mlp_drop_rate", 0.0)
        self.norm_type = model_config.get("norm_type", "layer")
        self.act_type = model_config.get("act_type", "silu")

        # QK-Norm. See more details (ViT-22B): http://arxiv.org/abs/2302.05442
        self.qk_norm = model_config.get("qk_norm", True)
        self.attn_mode = model_config.get("attn_mode", "torch")

        act_layer = get_activation_layer(self.act_type)

        # Build image/video patchify layer
        self.x_embedder = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)

        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(self.hidden_size, act_layer, **factory_kwargs)

        # Build condition embedding layer
        if self.condition_projection == "linear":
            self.cond_embedder = ConditionProjection(
                self.condition_dim,
                self.hidden_size,
                act_layer,
                **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported condition_projection: {self.condition_projection}")

        # Build modulation layer
        if self.modulate_type == "pixart":
            self.adaLN_modulation_single = ModulateDiT(self.hidden_size, 6, act_layer, **factory_kwargs)

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size=self.hidden_size,
                             num_heads=self.num_heads,
                             mlp_ratio=self.mlp_ratio,
                             attn_drop_rate=self.attn_drop_rate,
                             proj_drop_rate=self.proj_drop_rate,
                             mlp_drop_rate=self.mlp_drop_rate,
                             qk_norm=self.qk_norm,
                             act_type=self.act_type,
                             norm_type=self.norm_type,
                             attn_mode=self.attn_mode,
                             modulate_type=self.modulate_type,
                             layer=layer,
                             **factory_kwargs,
                             )
            for layer in range(self.depth)
        ])

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels, act_layer, **factory_kwargs)
        self.unpatchify_channels = self.out_channels

    def set_attn_mode(self, new_mode):
        for block in self.blocks:
            block.set_attn_mode(new_mode)

    def enable_deterministic(self):
        for block in self.blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.blocks:
            block.disable_deterministic()

    def forward(self,
                x: torch.Tensor,
                t: torch.LongTensor,
                cond: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None,
                freqs_cos: Optional[torch.Tensor] = None,
                freqs_sin: Optional[torch.Tensor] = None,
                return_dict: bool = True,
                ):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor with shape [b, 3, h, w].
            t (torch.LongTensor): Time steps of the diffusion/flow matching schedule. Shape [b].
            freqs_cos (torch.Tensor, optional): Real part of the image RoPE.
            freqs_sin (torch.Tensor, optional): Imaginary part of the image RoPE.
            return_dict (bool):

        .. note: We use separated :attr:`cos_cis` and :attr:`sin_cis` for RoPE computation in real space,
        because TensorRT does not support complex number computation.
        """
        out = {}

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # --------------------- Get timestep embedding -----------------------
        input_t = t
        t = self.t_embedder(input_t)

        # Maybe apply pre-processing to the `t`
        if self.modulate_type in ["dit", "lumina"]:
            c = t
        elif self.modulate_type == "pixart":
            c = self.adaLN_modulation_single(t)
        else:
            raise ValueError(f"Unknown modulate_type: {self.modulate_type}")

        # ------------------------ Get condition embedding ------------------------
        if self.condition_projection == "linear":
            cond = self.cond_embedder(cond)
        else:
            raise NotImplementedError(f"Unsupported condition_projection: {self.condition_projection}")

        # -------------------- Get image/video embedding ---------------------
        x = self.x_embedder(x)

        # Compute 'cross attention mask'.
        if self.use_attention_mask:
            assert cond_mask is not None, "Condition mask must be provided when use_attention_mask is True."
            if self.attn_mode == "flash":
                cross_attn_mask = cond_mask
            elif self.attn_mode in ["torch", "vanilla"]:
                seqlen = x.size(1)
                bs, seqlen1 = cond_mask.shape
                cross_attn_mask = cond_mask.view(bs, 1, 1, seqlen1).repeat(1, 1, seqlen, 1)  # [b, 1, s, s1]
                cross_attn_mask = cross_attn_mask.bool()
            else:
                raise NotImplementedError(f'Unsupported attention mode: {self.attn_mode}')
        else:
            cross_attn_mask = None

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for layer_num, block in enumerate(self.blocks):
            block_args = [x, c, cond, cross_attn_mask, freqs_cis]

            if self.training and self.gradient_checkpoint and \
                    (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *block_args, use_reentrant=False)
            else:
                x = block(*block_args)

        # ---------------------------- Final layer ------------------------------
        x = self.final_layer(x, t)
        x = self.unpatchify(x, th, tw)

        if return_dict:
            out['x'] = x
            return out
        return x

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def params_count(self):
        counts = {
            "attn+mlp": sum([
                sum(p.numel() for p in block.attn1.parameters()) +
                sum(p.numel() for p in block.attn2.parameters()) +
                sum(p.numel() for p in block.mlp.parameters())
                for block in self.blocks
            ]),
            "total": sum(p.numel() for p in self.parameters()),
        }
        return counts


#################################################################################
#                                   DiT Configs                                 #
#################################################################################

DiT_CONFIG = {                                                                          # Attn+MLP / Total
    'DiT-S/2': {'depth': 12, 'hidden_size': 576, 'num_heads': 6, 'mlp_ratio': 4},       #    64M   /   92M
    'DiT-SP/2': {'depth': 12, 'hidden_size': 864, 'num_heads': 9, 'mlp_ratio': 4},      #   143M   /  204M
    'DiT-B/2': {'depth': 12, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   255M   /  361M
    'DiT-L/2': {'depth': 24, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   510M   /  712M
    'DiT-XL/2': {'depth': 24, 'hidden_size': 1680, 'num_heads': 14, 'mlp_ratio': 4},    #   1.1B   /  1.5B
    'DiT-XXL/2': {'depth': 36, 'hidden_size': 1920, 'num_heads': 16, 'mlp_ratio': 4},   #   2.1B   /  2.9B
    'DiT-G/2': {'depth': 36, 'hidden_size': 2880, 'num_heads': 24, 'mlp_ratio': 4},     #   4.8B   /  6.6B
    'DiT-T/2': {'depth': 40, 'hidden_size': 3840, 'num_heads': 32, 'mlp_ratio': 4},     #   9.4B   / 13.0B

    # When using PixArt modulation, we adjust hyperparameters to match the number of
    # total parameters.                                                                    # Attn+MLP / Total
    'PixArt-S/2': {'depth': 16, 'hidden_size': 576, 'num_heads': 6, 'mlp_ratio': 4},       #    85M   /   91M
    'PixArt-SP/2': {'depth': 16, 'hidden_size': 864, 'num_heads': 9, 'mlp_ratio': 4},      #   191M   /  203M
    'PixArt-B/2': {'depth': 16, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   340M   /  359M
    'PixArt-L/2': {'depth': 32, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   680M   /  700M
    'PixArt-XL/2': {'depth': 32, 'hidden_size': 1680, 'num_heads': 14, 'mlp_ratio': 4},    #  1.44B   / 1.48B
    'PixArt-XXL/2': {'depth': 48, 'hidden_size': 1920, 'num_heads': 16, 'mlp_ratio': 4},   #  2.83B   / 2.88B
    'PixArt-G/2': {'depth': 48, 'hidden_size': 2880, 'num_heads': 24, 'mlp_ratio': 4},     #  6.37B   / 6.47B

    # When using Lumina modulation, we adjust hyperparameters to match the number of
    # total parameters.                                                                    # Attn+MLP / Total
    'Lumina-S/2': {'depth': 13, 'hidden_size': 576, 'num_heads': 6, 'mlp_ratio': 4},       #    69M   /   90M
    'Lumina-SP/2': {'depth': 13, 'hidden_size': 864, 'num_heads': 9, 'mlp_ratio': 4},      #   155M   /  201M
    'Lumina-B/2': {'depth': 13, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   276M   /  355M
    'Lumina-L/2': {'depth': 26, 'hidden_size': 1152, 'num_heads': 12, 'mlp_ratio': 4},     #   552M   /  701M
    'Lumina-XL/2': {'depth': 26, 'hidden_size': 1680, 'num_heads': 14, 'mlp_ratio': 4},    #  1.17B   / 1.49B
    'Lumina-XXL/2': {'depth': 39, 'hidden_size': 1920, 'num_heads': 16, 'mlp_ratio': 4},   #  2.30B   / 2.90B
    'Lumina-G/2': {'depth': 39, 'hidden_size': 2880, 'num_heads': 24, 'mlp_ratio': 4},     #  5.18B   / 6.52B
    'Lumina-T/2': {'depth': 43, 'hidden_size': 3840, 'num_heads': 32, 'mlp_ratio': 4},     # 10.15B   / 12.8B
}
