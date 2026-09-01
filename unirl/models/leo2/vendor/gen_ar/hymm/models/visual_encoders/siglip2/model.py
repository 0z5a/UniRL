from typing import Optional, Tuple, Union
import json
from easydict import EasyDict
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from hymm.models.visual_encoders.siglip2.loss import SigLipLoss

from loguru import logger


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward

class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`List[Tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """
        if pixel_values.ndim == 2:
            pixel_values = pixel_values[None]
        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class Siglip2SdpaAttention(Siglip2Attention):
    """
    Siglip2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Siglip2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    is_causal = False

    # Adapted from Siglip2Attention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Siglip2Model is using Siglip2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if self.is_causal and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2Attention(config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2Config
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer_index, encoder_layer in enumerate(self.layers): # len(self.layers): 27
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.training and self.gradient_checkpointing:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(encoder_layer),
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False
                )
                # layer_outputs = self._gradient_checkpointing_func(
                #     encoder_layer.__call__,
                #     hidden_states,
                #     attention_mask,
                #     output_attentions,
                # )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._load_parameters_fn = None

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state, attention_mask) if self.use_head else None
        if not return_dict:
            return (last_hidden_state, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, path, pretrained=True, **kwargs):
        with (Path(path) / "vision_config.json").open() as f:
            config = EasyDict(json.load(f))
        config.update(kwargs)   # Add extra config parameters: vision_use_head
        model = cls(config)

        def load_parameters(loc="cuda"):
            return dict(
                state_dict=torch.load(Path(path) / "vision_model.pt", map_location=loc, weights_only=True),
                strict=config.get("vision_use_head", True),
            )

        if pretrained:
            model.load_state_dict(**load_parameters(loc="cpu"))
        else:
            model._load_parameters_fn = load_parameters
        return model

    def get_pretrained_state_dict(self):
        assert self._load_parameters_fn is not None, "_load_parameters_fn is not set."
        return self._load_parameters_fn()["state_dict"]


class Siglip2TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class Siglip2TextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = Siglip2TextEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, config.projection_size)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: Siglip2's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, path):
        config = EasyDict(json.load(open(Path(path) / "text_config.json")))
        model = cls(config)
        model.load_state_dict(torch.load(Path(path) / "text_model.pt", map_location="cpu", weights_only=True), strict=True)
        return model


class Siglip2(nn.Module):
    def __init__(self, config, rank, world_size):
        super().__init__()
        self.vision_transformer = Siglip2VisionTransformer(config.vision_config)
        self.text_transformer = Siglip2TextTransformer(config.text_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(10.))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)

        self.loss = SigLipLoss(
            rank=rank,
            world_size=world_size,
            bidir=config.siglip_loss_bidir,
        )

    def forward(self, vision_inputs, text_inputs):
        vision_outputs = self.vision_transformer(
            pixel_values=vision_inputs["pixel_values"],
            attention_mask=vision_inputs["pixel_attention_mask"],
            spatial_shapes=vision_inputs["spatial_shapes"]
        )
        text_outputs = self.text_transformer(**text_inputs)

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        loss = self.loss(image_embeds, text_embeds, self.logit_scale.exp(), self.logit_bias)
        return {
            "loss": loss,
        }


SIGLIP2_CONFIG = {
    "siglip2-so400m-patch16-naflex": {
        "vision_config": {
            "_attn_implementation": "sdpa",
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-06,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "num_patches": 256,
            "patch_size": 16,
            "torch_dtype": "float32",
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True
        },
        "text_config": {
            "_attn_implementation": "sdpa",
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "vocab_size": 256000,
            "max_position_embeddings": 64,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "layer_norm_eps": 1e-06,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "projection_size": 1152,
            "torch_dtype": "float32",
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True
        }
    }
}

if __name__ == "__main__":
    # config = {
    #     "_attn_implementation": "sdpa",
    #     "attention_dropout": 0.0,
    #     "hidden_act": "gelu_pytorch_tanh",
    #     "hidden_size": 1152,
    #     "intermediate_size": 4304,
    #     "layer_norm_eps": 1e-06,
    #     "num_attention_heads": 16,
    #     "num_channels": 3,
    #     "num_hidden_layers": 27,
    #     "num_patches": 256,
    #     "patch_size": 16,
    #     "torch_dtype": "float32",
    #     "output_attentions": False,
    #     "output_hidden_states": False,
    #     "use_return_dict": True
    # }
    # from easydict import EasyDict
    # config = EasyDict(config)
    # vision_transformer = Siglip2VisionTransformer(config)
    # vision_transformer.load_state_dict(torch.load("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/vision_model.pt"))
    # vision_transformer.eval()
    # vision_transformer.cuda()

    # from transformers import Siglip2ImageProcessorFast
    # image_processor = Siglip2ImageProcessorFast.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex")

    # from PIL import Image
    # image = Image.open("./__my/image/vase.jpg")
    # inputs = image_processor.preprocess(image).to("cuda")

    # with torch.no_grad():   
    #     outputs = vision_transformer(
    #         pixel_values=inputs["pixel_values"],
    #         attention_mask=inputs["pixel_attention_mask"],
    #         spatial_shapes=inputs["spatial_shapes"]
    #     )

    # print(outputs.last_hidden_state.mean(), outputs.last_hidden_state.std(), outputs.last_hidden_state.pow(3).std(), outputs.last_hidden_state.pow(7).mean())

    from transformers import Siglip2ImageProcessorFast
    image_processor = Siglip2ImageProcessorFast.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/official")
    from PIL import Image
    image = Image.open("./__my/image/vase.jpg")
    inputs = image_processor.preprocess(image, max_num_patches=1024).to("cuda")
    print(inputs.keys())

    from hymm.models.visual_encoders import load_vision_model
    model = load_vision_model("siglip2-so400m-patch16-naflex", device="cuda")

    print(f"{image.height=}, {image.width=}, ratio={image.height / image.width}")
    print(inputs["pixel_values"].shape)
    print(inputs["pixel_attention_mask"].shape, inputs["pixel_attention_mask"].dtype, inputs["pixel_attention_mask"].reshape(32, 32))
    print(inputs["spatial_shapes"], inputs["spatial_shapes"].dtype, inputs["spatial_shapes"][0][0] / inputs["spatial_shapes"][0][1])

    with torch.no_grad():
        outputs = model(
            pixel_values=torch.zeros_like(inputs["pixel_values"]),
            attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"]
        )
    print(outputs.last_hidden_state)
    print(outputs.last_hidden_state.shape)
    print(outputs.last_hidden_state.mean(), outputs.last_hidden_state.std(), outputs.last_hidden_state.pow(3).std(), outputs.last_hidden_state.pow(7).mean())


    exit()

    text_transformer = Siglip2TextTransformer.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex")
    text_transformer.eval()
    text_transformer.cuda()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/official")

    # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
    input_ids = tokenizer([
        "This is a photo of vase.",
        "This is a photo of cat.",
        "This is a photo of dog."
    ], padding="max_length", max_length=64, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        text_outputs = text_transformer(input_ids.cuda())

    image_embeds = outputs.pooler_output
    text_embeds = text_outputs.pooler_output

    # normalized features
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

    logits = torch.load("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/logits.pt")
    logit_scale = logits["logit_scale"]
    logit_bias = logits["logit_bias"]

    logit_scale, logit_bias = logit_scale.to(text_embeds.device), logit_bias.to(text_embeds.device)
    logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

    print(torch.sigmoid(logits_per_text))




    from transformers import AutoModel
    model = AutoModel.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/official")
    model.eval()
    model.cuda()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("/apdcephfs_nj10/share_301739632/1_public_models/hymm_ar_assets/vision_encoder/siglip2-so400m-patch16-naflex/official")

    texts = [
        "This is a photo of vase.",
        "This is a photo of cat.",
        "This is a photo of dog."
    ]

    # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
    inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    print(torch.sigmoid(outputs.logits_per_text))

