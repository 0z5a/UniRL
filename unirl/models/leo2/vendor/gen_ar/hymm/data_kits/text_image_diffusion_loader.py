import random

import torch

from .mm_loader import MultimodalIndexDataset
from ..models.diffusion.posemb_layers import get_nd_rotary_pos_embed
from ..utils.image_base import ImageInfo


class TextImageArrowStream(MultimodalIndexDataset):
    def __init__(self, *args, **kwargs):
        kwargs["use_tokenizer"] = False
        super().__init__(*args, **kwargs)

        self.sequence_template = "pretrain"

    def get_rope_from_image_info(self, image_info: ImageInfo):
        if self.args.rope_type_extended == "2d":
            latents_size = [image_info.token_height, image_info.token_width]
            assert all(s % self.args.patch_size == 0 for s in latents_size), \
                f"Latent size(last 2 dimensions) should be divisible by patch size({self.args.patch_size}), " \
                f"but got {latents_size}."
            rope_sizes = [s // self.args.patch_size for s in latents_size]
            rope_dim_list = self.args.rope_dim_list
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list=rope_dim_list,
                start=rope_sizes,
                theta=self.args.rope_theta,
                use_real=True,
            )
        else:
            raise NotImplementedError(f"Rope type extended {self.args.rope_type_extended} not implemented.")
        return freqs_cos, freqs_sin

    def __getitem__(self, index):
        index = int(index)

        data = self.get_t2i_data(index)

        # uncondition
        do_uncond = (self.uncond_p > 0) and (random.random() < self.uncond_p)
        prompt = "" if do_uncond else data.prompt

        image_tensor = data.images[0]

        # 2d rope
        freqs_cos, freqs_sin = self.get_rope_from_image_info(image_tensor.i)

        ret = {
            "dataset_tag": self.dataset_tag,
            "n_samples": 1,
            "index": index,
            "prompt": prompt,
            "image_tensor": image_tensor,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
        }

        return ret

    def collate_fn(self, batch):
        ret = {
            # === required ===
            "dataset_tag": [item["dataset_tag"] for item in batch],
            "n_samples": torch.tensor([item["n_samples"] for item in batch]),
            "index": torch.tensor([item["index"] for item in batch]),
            "prompt": [item["prompt"] for item in batch],
            "image_tensor": torch.stack([item["image_tensor"] for item in batch]),
            "freqs_cos": torch.stack([item["freqs_cos"] for item in batch]),
            "freqs_sin": torch.stack([item["freqs_sin"] for item in batch]),
        }
        return ret