# ================================================
# Author: kevinkhwu
# Email: kevinkhwu@tencent.com
# ================================================

from functools import partial
from typing import Any, Dict, Optional, Union

import torch

from ..communications import all_gather


####### add timestep_r to support MeanFlow #######
def apply_sp(model, sp_group, sp_rank, sp_size):
    from hyvideo.models.flash_attn_no_pad import get_cu_seqlens
    from hyvideo.models.hunyuan.modules.attenion import parallel_attention

    def sp_forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            text_states: torch.Tensor,
            text_states_2: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            output_features=False,
            output_features_stride=8,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            return_dict: bool = False,
            guidance=None,
            extra_kwargs=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        # txt = encoder_hidden_states[:, 1:]
        # text_states_2 = encoder_hidden_states[:, 0, : self.config.text_states_dim_2]
        txt = text_states
        input_shape = x.shape
        if len(input_shape) == 5:
            _, _, ot, oh, ow = x.shape
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )
            if freqs_cos is None or freqs_sin is None:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))
        elif len(input_shape) == 4:
            _, ot, oh, ow = x.shape
            th, tw = (
                oh // self.patch_size[0],
                ow // self.patch_size[1],
            )
            if freqs_cos is None and freqs_sin is None:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((th, tw))
        else:
            raise ValueError(f"Unsupported hidden_states shape: {x.shape}")

        img = self.img_in(img)

        assert img.shape[1] % sp_size == 0, f"Cannot split video sequence into ulysses_degree  ({sp_size}) parts evenly"
        img = torch.chunk(img, sp_size, dim=1)[sp_rank]
        freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
        freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        # Prepare modulation vectors.
        vec = self.time_in(t)
        # # text modulation
        # vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")


        if self.glyph_byT5_v2:
            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]
            byt5_txt = self.byt5_in(byt5_text_states)

            txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        cu_seqlens, max_s = get_cu_seqlens(text_mask, img_seq_len)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for index, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis, text_mask, cu_seqlens, max_s]

            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if output_features:
            features_list = []
        if len(self.single_blocks) > 0:
            for index, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    cu_seqlens,
                    max_s,
                ]
                x = block(*single_block_args)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = all_gather(img, dim=1, group=sp_group, rank=sp_rank)

        if len(input_shape) == 5:
            img = self.unpatchify(img, tt, th, tw)
            shape = (tt, th, tw)
        elif len(input_shape) == 4:
            img = self.unpatchify_2d(img, th, tw)
            shape = (th, tw)
        else:
            raise ValueError(f"Unsupported input_shape: {input_shape}")
        assert not return_dict, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
            features_list = all_gather(features_list, dim=2, group=sp_group, rank=sp_rank)
        else:
            features_list = None
        return (img, features_list, shape)

    for block in model.double_blocks + model.single_blocks:
        block.core_attn = partial(parallel_attention, sp_group=sp_group, sp_rank=sp_rank, sp_size=sp_size)

    model.__class__.forward = sp_forward
