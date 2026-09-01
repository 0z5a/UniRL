import torch

from ..ar.mask_schedulers import get_mask_code, build_mask_ratio_generator, create_attention_mask_general
from .instruction_tuning_ar_loader import InstructionTuningARArrowStream
from .irregular_mask_dataset import IrregularMaskDataset
from ..constants import VAE_META_INFO
from ..models.basic.rope import get_mlm_rope


class InstructionTuningMaskArrowStream(InstructionTuningARArrowStream):
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
        self.mask_dataset = None

        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = patch_size

        self.bos_id = self.tokenizer.bos_token
        self.boi_id = self.tokenizer.special_token_map["<boi>"]
        self.pad_id = self.tokenizer.special_token_map["<pad>"]
        self.uncond_id = self.tokenizer.special_token_map["<cfg>"]
        self.mask_id = self.tokenizer.special_token_map["<mask>"]

        self.multitask = multitask
        self.task = task

        self.mask_ratio_generator = build_mask_ratio_generator(
            schedule_type=self.args.get('train_schedule_type', 'arccos'),
            logger=self.logger,
        )

    def get_editing_data_item(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        editing_src_img_token_col = self.index_kwargs.get("eiditing_src_img_token_col", None)
        if editing_src_img_token_col is None:
            raise ValueError("Missing `eiditing_src_img_token_col` in `index_kwargs`")
        editing_tgt_img_token_col = self.index_kwargs.get("eiditing_tgt_img_token_col", None)
        if editing_tgt_img_token_col is None:
            raise ValueError("Missing `eiditing_tgt_img_token_col` in `index_kwargs`")
        instruction_uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)

        src_image_tokens, src_image_tokens_shape_wh, _ = self.get_img_token(index, column=editing_src_img_token_col, shift=False)
        src_tk_width, src_tk_height = src_image_tokens_shape_wh.tolist()
        tgt_image_tokens, tgt_image_tokens_shape_wh, _ = self.get_img_token(index, column=editing_tgt_img_token_col, shift=False)
        tgt_tk_width, tgt_tk_height = tgt_image_tokens_shape_wh.tolist()
        src_shifted_image_tokens = src_image_tokens + self.image_token_offset
        tgt_shifted_image_tokens = tgt_image_tokens + self.image_token_offset

        # Whether also randomly mask the source image token
        is_mask_src = self.args.loss_predict.startswith("text__image")
        # Apply mask to image token
        if is_mask_src:
            src_masked_shifted_image_tokens, src_mask = get_mask_code(
                generator=self.mask_ratio_generator,
                code=src_shifted_image_tokens[None],
                mask_id=self.mask_id,
            )   # [1, n]
        else:
            src_masked_shifted_image_tokens = src_shifted_image_tokens[None]
        tgt_masked_shifted_image_tokens, tgt_mask = get_mask_code(
            generator=self.mask_ratio_generator,
            code=tgt_shifted_image_tokens[None],
            mask_id=self.mask_id,
        )   # [1, n]

        whole_tokens, whole_labels, src_token_slice, tgt_token_slice = self.tokenizer.encode_mlm_editing(
            instruction,
            src_masked_shifted_image_tokens=src_masked_shifted_image_tokens[0],
            tgt_masked_shifted_image_tokens=tgt_masked_shifted_image_tokens[0],
            src_image_token_len=src_tk_width * src_tk_height,
            tgt_image_token_len=tgt_tk_width * tgt_tk_height,
            max_text_token_length=self.text_token_length,
            max_total_token_length=self.text_token_length + 2 * self.image_token_length,
            instruction_uncond_p=instruction_uncond_p,
            loss_predict=self.args.loss_predict,
        )

        # Prepare whole labels
        #   shift 1 for image token labels
        src_label_slice = slice(src_token_slice.start + 1, src_token_slice.stop + 1)
        tgt_label_slice = slice(tgt_token_slice.start + 1, tgt_token_slice.stop + 1)
        if "_masked" in self.args.loss_predict:
            if is_mask_src:
                whole_labels[src_label_slice] = torch.where(src_mask[0], src_shifted_image_tokens, -100)
            whole_labels[tgt_label_slice] = torch.where(tgt_mask[0], tgt_shifted_image_tokens, -100)
        else:
            if is_mask_src:
                whole_labels[src_label_slice] = src_masked_shifted_image_tokens[0]
            whole_labels[tgt_label_slice] = tgt_masked_shifted_image_tokens[0]
        whole_labels = whole_labels.to(torch.long)

        tgt_image_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32)
        tgt_image_loss_mask[tgt_label_slice] = tgt_mask.float()     # loss mask uses label slice
        if self.args.loss_predict.startswith("text__image"):
            text_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32)
            text_loss_mask[:self.text_token_length] = 1.0
            text_loss_mask[whole_labels == -100] = 0.0
        elif self.args.loss_predict.startswith("target_only"):
            text_loss_mask = None
        else:
            raise ValueError(f"Unknown loss_predict: {self.args.loss_predict}")

        attention_mask = create_attention_mask_general(
            whole_tokens[None, :-1],
            self.pad_id,
            (src_token_slice, tgt_token_slice),
            mask_pad=True,
            return_inverse_mask=True,
            dtype=torch.float32,
            pad_endpoint=self.text_token_length,    # Only consider the text pad.
        )[0]

        # ===================================== Build RoPE ==================================
        pad_length = 2 * self.image_token_length - src_tk_height * src_tk_width - tgt_tk_height * tgt_tk_width
        rope_kwargs = dict(theta=self.args.get('rope_theta', 10000), use_real=True,
                           interleave=self.args.rope_type == "3d-interleave")
        freqs_cos, freqs_sin = get_mlm_rope(
            self.args.rope_dim_list,
            [src_tk_height, tgt_tk_height],
            [src_tk_width, tgt_tk_width],
            self.text_token_length + 1 + pad_length,
            [self.text_token_length - 4, self.text_token_length - 1],
            None,
            **rope_kwargs
        )

        return {
            "whole_tokens": whole_tokens,
            "whole_labels": whole_labels,
            "src_image_tokens": src_image_tokens,
            "tgt_image_tokens": tgt_image_tokens,
            "src_image_tokens_shape_wh": src_image_tokens_shape_wh,
            "tgt_image_tokens_shape_wh": tgt_image_tokens_shape_wh,
            "mask": None,
            "tgt_image_loss_mask": tgt_image_loss_mask,
            "text_loss_mask": text_loss_mask,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
            "tgt_mask": tgt_mask,
        }

    def get_inpainting_data_item(self, index):
        if self.mask_dataset is None:
            inpainting_mask_dataset_path = self.index_kwargs.get("inpainting_mask_dataset_path", None)
            inpainting_mask_dataset_set = self.index_kwargs.get("inpainting_mask_dataset_set", "train")
            if inpainting_mask_dataset_path is None:
                raise ValueError("Missing `inpainting_mask_dataset_path` in `index_kwargs`")
            self.mask_dataset = IrregularMaskDataset(path=inpainting_mask_dataset_path, set=inpainting_mask_dataset_set)

        # Get instruction
        instruction = self.get_instruction(index)
        prompt = self.get_prompt(index)

        inpainting_img_token_col = self.index_kwargs.get("inpainting_img_token_col", None)
        if inpainting_img_token_col is None:
            raise ValueError("Missing `inpainting_img_token_col` in `index_kwargs`")

        instruction_uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)

        image_tokens, image_tokens_shape_wh, img_token_flag = self.get_img_token(index, column=inpainting_img_token_col, shift=False)
        tk_width, tk_height = image_tokens_shape_wh.tolist()
        shifted_image_tokens = image_tokens + self.image_token_offset

        # (h, w)
        mask = self.mask_dataset.get_mask((tk_height, tk_width))

        # Whether also randomly mask the source image token
        is_random_mask_src = self.args.loss_predict.startswith("text__image")
        # Apply mask to image token
        src_masked_shifted_image_tokens, src_mask = get_mask_code(
            generator=self.mask_ratio_generator,
            code=shifted_image_tokens[None],
            mask_id=self.mask_id,
            user_mask=mask.view(-1)[None],      # Use the user-defined mask
            use_generator=is_random_mask_src,   # Close the mask generator
        )   # [1, n]
        tgt_masked_shifted_image_tokens, tgt_mask = get_mask_code(
            generator=self.mask_ratio_generator,
            code=shifted_image_tokens[None],
            mask_id=self.mask_id,
        )   # [1, n]

        whole_tokens, whole_labels, src_token_slice, tgt_token_slice = self.tokenizer.encode_mlm_editing(
            instruction, prompt,
            src_masked_shifted_image_tokens=src_masked_shifted_image_tokens[0],
            tgt_masked_shifted_image_tokens=tgt_masked_shifted_image_tokens[0],
            src_image_token_len=tk_width * tk_height,
            tgt_image_token_len=tk_width * tk_height,
            max_text_token_length=self.text_token_length,
            max_total_token_length=self.text_token_length + 2 * self.image_token_length,
            uncond_enabled=[False, True],
            instruction_uncond_p=instruction_uncond_p,
            loss_predict=self.args.loss_predict,
        )

        # Prepare whole labels
        #   shift 1 for image token labels
        src_label_slice = slice(src_token_slice.start + 1, src_token_slice.stop + 1)
        tgt_label_slice = slice(tgt_token_slice.start + 1, tgt_token_slice.stop + 1)
        if "_masked" in self.args.loss_predict:
            if is_random_mask_src:
                whole_labels[src_label_slice] = torch.where(src_mask[0], shifted_image_tokens, -100)
            whole_labels[tgt_label_slice] = torch.where(tgt_mask[0], shifted_image_tokens, -100)
        else:
            if is_random_mask_src:
                whole_labels[src_label_slice] = src_masked_shifted_image_tokens[0]
            whole_labels[tgt_label_slice] = tgt_masked_shifted_image_tokens[0]
        whole_labels = whole_labels.to(torch.long)

        tgt_image_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32)
        tgt_image_loss_mask[tgt_label_slice] = tgt_mask.float()  # loss mask uses label slice
        if self.args.loss_predict.startswith("text__image"):
            text_loss_mask = torch.zeros_like(whole_tokens, dtype=torch.float32)
            text_loss_mask[:self.text_token_length] = 1.0
            text_loss_mask[whole_labels == -100] = 0.0
        elif self.args.loss_predict.startswith("target_only"):
            text_loss_mask = None
        else:
            raise ValueError(f"Unknown loss_predict: {self.args.loss_predict}")

        attention_mask = create_attention_mask_general(
            whole_tokens[None, :-1],
            self.pad_id,
            (src_token_slice, tgt_token_slice),
            mask_pad=True,
            return_inverse_mask=True,
            dtype=torch.float32,
            pad_endpoint=self.text_token_length,  # Only consider the text pad.
        )[0]

        # ===================================== Build RoPE ==================================
        pad_length = 2 * self.image_token_length - tk_height * tk_width * 2
        rope_kwargs = dict(theta=self.args.get('rope_theta', 10000), use_real=True,
                           interleave=self.args.rope_type == "3d-interleave")
        freqs_cos, freqs_sin = get_mlm_rope(
            self.args.rope_dim_list,
            [tk_height, tk_height],
            [tk_width, tk_width],
            self.text_token_length + 1 + pad_length,
            [self.text_token_length - 4, self.text_token_length - 1],
            None,
            **rope_kwargs
        )

        return {
            "whole_tokens": whole_tokens,
            "whole_labels": whole_labels,
            "src_image_tokens": image_tokens,
            "tgt_image_tokens": image_tokens,
            "src_image_tokens_shape_wh": image_tokens_shape_wh,
            "tgt_image_tokens_shape_wh": image_tokens_shape_wh,
            "mask": mask,
            "tgt_image_loss_mask": tgt_image_loss_mask,
            "text_loss_mask": text_loss_mask,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
            "tgt_mask": tgt_mask,
        }

    @staticmethod
    def collate_fn(batch):
        whole_tokens = torch.stack([item["whole_tokens"] for item in batch])
        whole_labels = torch.stack([item["whole_labels"] for item in batch])
        src_image_tokens = [item["src_image_tokens"] for item in batch]
        tgt_image_tokens = [item["tgt_image_tokens"] for item in batch]
        src_image_tokens_shape_wh = torch.stack([item["src_image_tokens_shape_wh"] for item in batch])
        tgt_image_tokens_shape_wh = torch.stack([item["tgt_image_tokens_shape_wh"] for item in batch])
        mask = [item["mask"] for item in batch]
        tgt_image_loss_mask = torch.stack([item["tgt_image_loss_mask"] for item in batch])
        text_loss_mask = torch.stack([item["text_loss_mask"] for item in batch]) \
            if batch[0]["text_loss_mask"] is not None else None
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        freqs_cos = torch.stack([item["freqs_cos"] for item in batch])
        freqs_sin = torch.stack([item["freqs_sin"] for item in batch])
        tgt_mask = [item["tgt_mask"] for item in batch]

        new_batch = {
            "whole_tokens": whole_tokens,
            "whole_labels": whole_labels,
            "src_image_tokens": src_image_tokens,
            "tgt_image_tokens": tgt_image_tokens,
            "src_image_tokens_shape_wh": src_image_tokens_shape_wh,
            "tgt_image_tokens_shape_wh": tgt_image_tokens_shape_wh,
            "mask": mask,
            "tgt_image_loss_mask": tgt_image_loss_mask,
            "text_loss_mask": text_loss_mask,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
            "tgt_mask": tgt_mask,
        }

        return new_batch
