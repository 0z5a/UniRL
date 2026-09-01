import torch
from torch.utils.data import get_worker_info

from ..models.basic.rope import get_3d_rope
from hymm.data_kits.text_image_loader import TextImageArrowStream
from hymm.constants import VAE_META_INFO


# following args of TextImageArrowStream is deprecated here
# - image_token_offset
# - use_pre_extracted_token
class TransfusionTextImageArrowStream(TextImageArrowStream):
    def __post_init__(
            self,
            return_attention_mask=False,
            dummy_number=0,
    ):
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = self.args.patch_size
        self.use_3d_rope = self.args.get('rope_type', 'default') in ['3d', '3d-interleave']
        self.add_timestep_token = self.args.add_timestep_token
        self.use_recaption_template = self.args.get('use_recaption_template', False)
        self.pred_text_boi_eos = self.args.get('pred_text_boi_eos', [True, True, True])

        # Prepare attention mask
        self.return_attention_mask = return_attention_mask
        self.dummy_number = dummy_number

    def __getitem__(self, index):
        """
        Get image and text from a given index

        Args:
            index (int): Index of the dataset

        Returns:
            image (torch.FloatTensor): Image tensor with shape (3, H, W)
            text (str): Original text
            kwargs (dict): Additional information
                origin_size (torch.LongTensor): Original size of the image (W, H)
                target_size (torch.LongTensor): Target size of the image (W, H)
                crop_coords_xy (torch.LongTensor): Crop coordinates (x, y)
                index (torch.LongTensor): Index of the dataset
        """

        # Get text
        text = self.get_text(index)

        # Get image
        image, kwargs, image_flag = self.get_image_with_size(index, image_key=self.image_key, shadow=self.image_key_shadow)
        if image_flag == "gray":
            text = "A gray image"

        h, w = image.shape[1], image.shape[2]

        assert h % (self.downsample_factor[0] * self.patch_size) == 0 and w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({h} x {w}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"

        tk_height = h // (self.downsample_factor[0] * self.patch_size)
        tk_width = w // (self.downsample_factor[1] * self.patch_size)
        actual_image_token_length = tk_height * tk_width

        # Build template and encode tokens
        if isinstance(text, str):
            text = [text]
        uncond_enabled = None
        if self.use_recaption_template and len(text) == 4:
            if hasattr(self.tokenizer.tokenizer, 'custom_extra_tokens'):
                text = [
                    text[0],
                    self.tokenizer.tokenizer.custom_extra_tokens[text[1]],
                    text[2],
                    self.tokenizer.tokenizer.custom_extra_tokens[text[3]],
                ]
            uncond_enabled = [True, False, True, False]     # [short] <recaption> [long] </recaption>
            #                                                 [<cfg>] <recaption> [<cfg>] </recaption>
        tokens, iw_ih_scatter_index, timestep_scatter_index, text_mask, image_mask = self.tokenizer.encode_transfusion(
            *text,
            uncond_enabled=uncond_enabled,
            image_token_length=actual_image_token_length,
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_p=self.uncond_p,
            add_iw_ih_token=self.add_iw_ih_token if not self.use_3d_rope else False,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
        )
        target_token = tokens.clone()
        target_token[text_mask == 0.0] = -100
        # target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        # target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100
        # target_token[target_token == self.tokenizer.special_token_map["<img>"]] = -100
        # # Predicting <eoi> token has no benefit for t2i. So, mask it.
        # target_token[target_token == self.tokenizer.special_token_map["<eoi>"]] = -100
        # if not self.use_3d_rope:
        #     target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
        #     target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
        # if self.add_timestep_token:
        #     target_token[target_token == self.tokenizer.special_token_map["<timestep>"]] = -100

        # here use resized image size as scatter_src of iw and ih
        image_token_shape_wh = torch.tensor([w, h], dtype=torch.long)

        if self.use_3d_rope:
            pad_length = self.image_token_length - tk_height * tk_width
            img_pos = torch.where(tokens == self.tokenizer.special_token_map["<img>"])[0][0].item()
            freqs_cos, freqs_sin = get_3d_rope(
                self.args.rope_dim_list,
                tk_height, tk_width,
                max_len=self.text_token_length + 1 + pad_length,
                img_pos=img_pos,
                device=None,
                theta=self.args.get('rope_theta', 10000),
                use_real=True,
                interleave=self.args.rope_type == "3d-interleave",
            )
        else:
            freqs_cos, freqs_sin = None, None

        if self.return_attention_mask:
            # Prepare attention mask
            n_tokens = tokens.shape[0] - 1 + self.dummy_number
            causal_mask = torch.ones(n_tokens, n_tokens, dtype=torch.bool).tril(diagonal=0)
            if self.dummy_number - 1 > 0:
                image_mask_expand = torch.cat([image_mask.bool(), torch.zeros(self.dummy_number - 1, dtype=torch.bool)])
            elif self.dummy_number - 1 == -1:
                image_mask_expand = image_mask.bool()[:-1]
            elif self.dummy_number - 1 == 0:
                image_mask_expand = image_mask.bool()
            else:
                raise ValueError(f"Invalid dummy_number: {self.dummy_number}")

            image_mask_1 = image_mask_expand.view(1, n_tokens).repeat(n_tokens, 1)
            image_mask_2 = image_mask_1.transpose(0, 1)
            attention_mask = causal_mask | (image_mask_1 & image_mask_2)
            attention_mask = attention_mask.unsqueeze(0)    # head dim

        # Prepare kwargs
        kwargs["index"] = index
        kwargs["text"] = text if isinstance(text, str) else "".join(text)

        ret = {
            "dtype": "t2i",
            "image": image,                     # (3, H~, W~)
            "n_samples": 1,                     # ()
            "tokens": tokens,                   # (L), L = text_token_length + 1 + image_token_length
            "target_tokens": target_token,      # (L)
            "text_mask": text_mask,             # (L)
            "image_mask": image_mask,           # (L)
            "kwargs": {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        }

        if iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": iw_ih_scatter_index,     # (2)
                "iw_ih_scatter_src": image_token_shape_wh,      # (2)
            })
        if timestep_scatter_index is not None:
            ret.update({
                "timestep_scatter_index": timestep_scatter_index,   # (1)
            })
        if freqs_cos is not None:
            ret.update({
                "freqs_cos": freqs_cos,     # (L, d), d = sum(rope_dim_list) = (80 for phi-2) or (128 for hy-dense-7b)
                "freqs_sin": freqs_sin,     # (L, d)
            })
        if self.return_attention_mask:
            ret.update({
                "attention_mask": attention_mask,
            })

        return ret
