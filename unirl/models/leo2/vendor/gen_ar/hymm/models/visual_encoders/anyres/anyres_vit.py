import random
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput

from .anyres_vit_config import AnyResViTConfig


class AnyResCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: AnyResViTConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.max_image_size
        self.patch_size = config.patch_size
        self.max_seq_len = config.max_vit_seq_len
        self.adaptor_patch_size = config.adaptor_patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.add_patch_emb_bias,
        )

        self.anyres_vit_max_image_size = config.anyres_vit_max_image_size
        self.num_patches = (self.anyres_vit_max_image_size // config.patch_size) ** 2

        self.skip_cls_token = True
        self.interpolate_mode = config.interpolate_mode
        self.num_positions = self.num_patches + 1
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)

        if config.remove_prenorm:
            self.pre_layernorm = None
        else:
            self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.rand_crop_rng = random.Random(1234)

    def reset_parameters(self):
        # NOTE: 必须显式 device，否则vision embed取 index 时 cpu vs cuda device mismatch
        self.position_ids.copy_(
            torch.arange(self.num_positions, device=self.position_ids.device).expand((1, -1))
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1]
        position_embeddings = self.position_embedding(self.position_ids)

        patch_pos_embed = position_embeddings[:, 1:]
        num_positions = position_embeddings.shape[1] - 1

        if num_patches == num_positions and height == width:
            return patch_pos_embed

        # class_pos_embed = position_embeddings[:, 0]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_size
        w0 = width // self.patch_size

        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        raw_type = patch_pos_embed.dtype

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.float(),
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode=self.interpolate_mode,
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.to(raw_type)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_single(self, pixel_values: torch.FloatTensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]
        batch_size, num_channels, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        b, c, h, w = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (bsz, num_patches, hidden_size)
        embeddings = patch_embeds + self.interpolate_pos_encoding(patch_embeds, height, width)
        return embeddings, (h, w)

    def forward(self, images):
        emb_0, hw_0 = self.forward_single(images)
        s0 = emb_0.shape[1]

        if self.pre_layernorm is not None:
            embeddings = self.pre_layernorm(emb_0)
        else:
            embeddings = emb_0

        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SelfAttention(nn.Module):
    def __init__(self, config: AnyResViTConfig, layer_idx: int):
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        self.head_size = self._config.hidden_size // self._config.num_attention_heads

        if not config.split_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3)
        else:
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.size()
        n_head = self._config.num_attention_heads
        head_size = self.head_size

        if not self._config.split_qkv:
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.view(bsz, seqlen, n_head, 3, head_size)
            # split batched computation into three
            q, k, v = qkv.unbind(dim=3)
        else:
            new_q = self.q_proj(hidden_states).view(bsz, seqlen, n_head, head_size)
            new_k = self.k_proj(hidden_states).view(bsz, seqlen, n_head, head_size)
            new_v = self.v_proj(hidden_states).view(bsz, seqlen, n_head, head_size)
            q, k, v = map(lambda x: x.transpose(1, 2), [new_q, new_k, new_v])

        if self._config.use_flash_attention:
            raise NotImplementedError("Flash attention is not implemented yet.")
        else:
            y = self.scaled_dot_product_attention(q, k, v, attention_mask)

        y = y.reshape(bsz, seqlen, head_size * n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.o_proj(y)

    def scaled_dot_product_attention(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # q, k, v: (bsz, n_head, seqlen, head_size)
        scale = 1.0 / math.sqrt(self.head_size)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale,
        )
        return y.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, config: AnyResViTConfig):
        super().__init__()
        self._config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class NavitDecoderLayer(nn.Module):
    def __init__(self, config: AnyResViTConfig, layer_idx: int):
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + residual

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class NavitVisionTransformer(nn.Module):
    def __init__(self, config: AnyResViTConfig):
        super().__init__()
        self.config = config

        self.embeddings = AnyResCLIPVisionEmbeddings(config)
        self.layers = nn.ModuleList([
            NavitDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

    def forward(self, pixel_values: Optional[torch.FloatTensor] = None):
        # embeddings
        hidden_states = self.embeddings(pixel_values)

        # decoder layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)

        return hidden_states


class SimpleConvMLP(nn.Module):
    def __init__(self, config: AnyResViTConfig, output_channels: int):
        super().__init__()
        self._config = config
        self.out_channels = output_channels
        self.embed_std = 1 / math.sqrt(output_channels)

        self.proj = nn.Sequential(
            nn.Conv2d(
                config.hidden_size,
                config.hidden_size * 2,
                kernel_size=config.adaptor_patch_size,
                stride=config.adaptor_patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(config.hidden_size * 2, config.hidden_size * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(config.hidden_size * 4, output_channels)

        self.image_newline = nn.Parameter(torch.randn(config.hidden_size * 4) * self.embed_std)
        if config.cat_extra_token:
            self.image_begin = nn.Parameter(torch.randn(output_channels) * self.embed_std)
            self.image_end = nn.Parameter(torch.randn(output_channels) * self.embed_std)
        # self.image_sep = nn.Parameter(torch.randn(output_channels) * self.embed_std)

        self.before_rms = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_after_rms:
            self.after_rms = RMSNorm(output_channels, eps=config.layer_norm_eps)
        else:
            self.after_rms = None

    def reset_parameters(self):
        self.image_newline.data.copy_(torch.randn(self._config.hidden_size * 4) * self.embed_std)
        if self._config.cat_extra_token:
            self.image_begin.data.copy_(torch.randn(self.out_channels) * self.embed_std)
            self.image_end.data.copy_(torch.randn(self.out_channels) * self.embed_std)
        # self.image_sep.data.copy_(torch.randn(self.out_channels) * self.embed_std)

    def forward(self, x, size, data_mode=0):
        """
        Args:
            x (torch.Tensor): vit model output, (bsz, seqlen, hidden_size)
            size (tuple[int, int]): (height, width) for patches
            data_mode (int): 0 for normal, 1 for down_sample in video, 2 for temporal compress in video
        """
        x = self.before_rms(x)

        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)

        x = self.proj(x)  # (bsz, hidden_size * 4, h, w)

        # video_temporal_compress
        if data_mode in [2, 3]:
            compress_size = 4
            b = x.shape[0]
            b_up = (b + compress_size - 1) // compress_size * compress_size
            if b_up > b:
                x_pad = x[-1:].expand(b_up - b, -1, -1, -1)
                x = torch.cat([x, x_pad], dim=0)
            x = x.permute(1, 0, 2, 3)
            x = F.avg_pool3d(x, kernel_size=(compress_size, 1, 1), stride=(compress_size, 1, 1))
            x = x.permute(1, 0, 2, 3)

        # video_spatial_compress
        if data_mode in [1, 3]:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Concat the <newline> token and project out
        b, c, h, w = x.shape
        x = torch.cat([
            x,
            self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype)
        ], dim=-1)
        x = x.reshape(b, c, -1).permute(0, 2, 1)    # (bsz, seqlen, hidden_size * 4)
        x = self.mlp(x)     # (bsz, seqlen, out_channels)

        if self._config.cat_extra_token:
            # Concat the <begin> and <end> tokens
            begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
            end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype)
            x = torch.cat([begin, x, end], dim=1)

        if self.after_rms:
            # Normed output
            x = self.after_rms(x)

        return x


class AnyResViT(nn.Module):
    def __init__(self, config: AnyResViTConfig, output_channels: int):
        super().__init__()
        self.config = config
        self.perceive = SimpleConvMLP(config, output_channels)
        self.vit = NavitVisionTransformer(config)

    def forward(self, images: torch.FloatTensor, data_mode=0):
        feat = self.vit(images)
        patch_size = self.vit.config.patch_size
        patch_height_num = images.shape[-2] // patch_size
        patch_width_num = images.shape[-1] // patch_size
        hidden_states = self.perceive(
            feat,
            (patch_height_num, patch_width_num),
            data_mode=data_mode
        )
        return BaseModelOutput(
            last_hidden_state=hidden_states
        )


class AnyResViTImageProcessor(BaseImageProcessor):
    def __init__(self, config: AnyResViTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        if config.use_imagenet_norm:
            image_mean = (0.48145466, 0.4578275, 0.40821073)
            image_std = (0.26862954, 0.26130258, 0.27577711)
        else:
            image_mean = (0.5, 0.5, 0.5)
            image_std = (0.5, 0.5, 0.5)

        self.pil_to_image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
        self.patch_size = config.patch_size * config.adaptor_patch_size

    def preprocess(self, image: ImageInput, **kwargs) -> BatchFeature:
        # image = ImageOps.exif_transpose(image)
        width, height = image.size

        width_new, height_new = self.get_aligned_image_size(width, height)
        image = image.resize((width_new, height_new))
        img_out_tensor = self.pil_to_image_tensor(image)

        return BatchFeature(
            data={
                "pixel_values": img_out_tensor[None],
                "image_height": height,
                "image_width": width,
                "resized_image_height": height_new,
                "resized_image_width": width_new,
            }
        )

    def get_aligned_image_size(self, width, height):

        def grid(width, height, max_size, min_size, grid_size, grid_type='floor'):
            if grid_type == 'rounding':
                width_grid = max(1, round(width / grid_size)) * grid_size
                height_grid = max(1, round(height / grid_size)) * grid_size
            else:
                width_grid = max(1, math.floor(width / grid_size)) * grid_size
                height_grid = max(1, math.floor(height / grid_size)) * grid_size

            if width_grid * height_grid > max_size * max_size:
                scale = (max_size * max_size / (width * height)) ** 0.5
                width_new, height_new = int(width * scale), int(height * scale)
                width_grid = max(1, math.floor(width_new / grid_size)) * grid_size
                height_grid = max(1, math.floor(height_new / grid_size)) * grid_size
            elif width_grid * height_grid < min_size * min_size:
                scale = (min_size * min_size / (width * height)) ** 0.5
                width_new, height_new = width * scale, height * scale
                if grid_type == 'rounding':
                    width_grid = max(1, math.ceil(width_new / grid_size)) * grid_size
                    height_grid = max(1, math.ceil(height_new / grid_size)) * grid_size
                else:
                    width_grid = max(1, math.floor(width_new / grid_size)) * grid_size
                    height_grid = max(1, math.floor(height_new / grid_size)) * grid_size

            return width_grid, height_grid

        max_width_height = max(width, height)
        max_size = int(self.config.max_image_size * 4)
        # single size (resize --> smaller/same)
        if max_width_height > max_size:
            scale = 1.0 * max_size / max_width_height
            width = int(width * scale)
            height = int(height * scale)

        return grid(
                width, height,
                self.config.max_image_size, self.patch_size,
                self.patch_size,
                grid_type='rounding'
        )
