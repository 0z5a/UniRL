import torch

from .text_image_loader import TextImageArrowStream
from ..constants import VAE_META_INFO


class TextMaskImageArrowStream(TextImageArrowStream):
    def __post_init__(
            self,
            multitask=False,
            task="t2i",
            patch_size=1,
            use_ar_template=True,
            text_encoder=None,
    ):
        self.use_ar_template = use_ar_template
        self.text_encoder = text_encoder

        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = patch_size

        self.bos_id = self.tokenizer.bos_token
        self.boi_id = self.tokenizer.special_token_map["<boi>"]
        self.pad_id = self.tokenizer.special_token_map["<pad>"]
        self.uncond_id = self.tokenizer.special_token_map["<cfg>"]
        self.ignore_id = self.args.ignore_index

        self.multitask = multitask
        self.task = task

    def get_whole_tokens_and_labels(self, text, tk_height, tk_width, dtype=torch.long):
        if not self.multitask and self.task == "t2i":
            whole_tokens = self.tokenizer.encode_mlm_t2i(
                text,
                image_token_len=tk_height * tk_width,
                max_text_token_length=self.text_token_length,
                text_uncond_p=self.uncond_p,
                dtype=dtype,
            )
        elif self.multitask and self.task == "t2i":
            whole_tokens = self.tokenizer.encode_mlm_multitask_t2i(
                text,
                image_token_len=tk_height * tk_width,
                max_text_token_length=self.text_token_length,
                text_uncond_p=self.uncond_p,
            )
        elif self.multitask and self.task == "mmu":
            whole_tokens = self.tokenizer.encode_mlm_multitask_mmu(
                text,
                image_token_len=tk_height * tk_width,
                max_text_token_length=self.text_token_length,
            )
        else:
            raise NotImplementedError(f"Not implemented task for `get_with_pixel`: {self.task}")

        if self.args.loss_predict.startswith("text__image"):
            # Predict text tokens(next token prediction) and image tokens(mask prediction).
            whole_labels = whole_tokens.clone()
            whole_labels[
                (whole_labels == self.pad_id) | (whole_labels == self.uncond_id) |
                (whole_labels == self.bos_id) | (whole_labels == self.boi_id)
            ] = self.ignore_id
        elif self.args.loss_predict.startswith("image"):
            # Predict all image tokens
            whole_labels = torch.full_like(whole_tokens, self.ignore_id)
        else:
            raise ValueError(f"Unknown loss_predict: {self.args.loss_predict}")
        return whole_tokens, whole_labels

    def __getitem__(self, index):
        kwargs = {
            "index": index,
        }

        # Get image_tensor, image_tokens_shape_wh(tk_width, tk_height), kwargs['target_size']
        if self.use_pre_extracted_token:
            image_tensor = ""
            shifted_image_tokens, image_tokens_shape_wh, image_flag = self.get_token(index, dtype=torch.int)
            tk_width, tk_height = image_tokens_shape_wh.tolist()
            image_tokens = shifted_image_tokens - self.image_token_offset
            kwargs["target_size"] = (tk_width * self.downsample_factor[0], tk_height * self.downsample_factor[1])

        else:
            image_tokens = ""
            image_tensor, pixel_kwargs, image_flag = self.get_image_with_size(index)
            kwargs.update(pixel_kwargs)
            image_height, image_width = image_tensor.shape[-2:]
            tk_height = image_height // self.downsample_factor[0] // self.patch_size
            tk_width = image_width // self.downsample_factor[1] // self.patch_size
            image_tokens_shape_wh = torch.tensor([tk_width, tk_height], dtype=torch.long)

        # Get text
        text = self.get_text(index) if image_flag != "gray" else "A gray image"
        kwargs["text"] = text

        if self.use_ar_template:
            # Prepare whole tokens sequence and labels sequence
            whole_tokens, whole_labels = self.get_whole_tokens_and_labels(text, tk_height, tk_width, dtype=torch.int)
            sequence = {
                "whole_tokens": whole_tokens,
                "whole_labels": whole_labels,
            }
        else:
            text_inputs = self.text_encoder.text2tokens(text)
            text_ids = text_inputs['input_ids'].squeeze(0)
            text_mask = text_inputs['attention_mask'].squeeze(0)
            sequence = {
                "text_ids": text_ids,
                "text_mask": text_mask,
            }

        ret = dict(
            image=image_tensor,
            **sequence,
            image_tokens=image_tokens,
            image_tokens_shape_wh=image_tokens_shape_wh,
            kwargs={k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        )

        return ret
