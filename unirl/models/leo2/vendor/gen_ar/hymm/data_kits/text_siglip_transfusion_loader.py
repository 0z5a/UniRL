import torch

from hymm.data_kits.text_image_loader import TextImageArrowStream
from hymm.constants import VISION_ENCODER_META_INFO

from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast

class TransfusionTextSiglipArrowStream(TextImageArrowStream):
    def __post_init__(
            self,
            return_attention_mask=False,
            dummy_number=0,
    ):
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_pretrained(VISION_ENCODER_META_INFO[self.args.vision_encoder_type]["path"])
        self.vision_encoder_max_num_patches = self.args.vision_encoder_max_num_patches

        self.patch_size = self.args.patch_size
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
        pil_image, kwargs, image_flag = self.get_pil_image_with_size(index, image_key=self.image_key, shadow=self.image_key_shadow)
        if image_flag == "gray":
            text = "A gray image"
        
        vision_encoder_inputs = self.vision_encoder_processor(pil_image, max_num_patches=self.vision_encoder_max_num_patches)
        # 1 x 256 x 768
        pixel_values=vision_encoder_inputs["pixel_values"]
        pixel_values=pixel_values.squeeze(0)
        # 1 x 256
        pixel_attention_mask=vision_encoder_inputs["pixel_attention_mask"]
        pixel_attention_mask=pixel_attention_mask.squeeze(0)
        # 1 x 2 (h, w) 
        spatial_shapes=vision_encoder_inputs["spatial_shapes"]
        spatial_shapes=spatial_shapes.squeeze(0)
        # h x w 是pixel_attention_mask中1的个数
        h, w = spatial_shapes[0], spatial_shapes[1]

        # always be 256
        actual_image_token_length = pixel_values.shape[0]

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
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=self.pred_text_boi_eos,
        )
        target_token = tokens.clone()
        target_token[text_mask == 0.0] = -100

        # here use resized image size as scatter_src of iw and ih
        image_token_shape_wh = torch.tensor([w, h], dtype=torch.long)

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
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
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
