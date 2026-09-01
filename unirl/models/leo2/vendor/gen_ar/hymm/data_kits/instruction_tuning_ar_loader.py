import ast
from collections import defaultdict
import json
import random

import numpy as np
import torch

from hymm.models import TokenizerWrapper
from hymm.utils.helpers import default
from hymm.data_kits.caption_strategy.caption_process import CaptionAug
from .index_dataset import IndexDataset
from .irregular_mask_dataset import IrregularMaskDataset


class InstructionTuningARArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        image_token_length=1024,
        image_token_offset=0,
        text_token_length=256,
        uncond_p=0.0,
        tokenizer_name=None,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
        tokenizer=None,
    ):
        super().__init__(
            index_file,
            multireso,
            index_kwargs.get("batch_size", 1),
            index_kwargs.get("world_size", 1),
            logger,
            debug
        )
        self.args = args
        self.uncond_p = uncond_p
        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.use_front_boi_token = self.args.use_front_boi_token

        self.index_kwargs = index_kwargs
        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=index_kwargs.get("ceph_base", None),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        self.inpainting_caption_rate = index_kwargs.get("inpainting_caption_rate", 0.0)

        self.inpainting_text_col = index_kwargs.get("inpainting_text_col", None)
        if self.inpainting_caption_rate < 1.0 and self.inpainting_text_col is None:
            raise ValueError("Missing `inpainting_text_col` in `index_kwargs`")

        self.inpainting_caption_col = index_kwargs.get("inpainting_caption_col", None)
        if self.inpainting_caption_rate > 0.0 and self.inpainting_caption_col is None:
            raise ValueError("Missing `inpainting_caption_col` in `index_kwargs`")

        self.inpainting_caption_sample_ratio = index_kwargs.get("inpainting_caption_sample_ratio", None)
        if self.inpainting_caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.inpainting_caption_sample_ratio = json.loads(self.inpainting_caption_sample_ratio)
            self.caption_aug = CaptionAug(self.inpainting_caption_sample_ratio, logger=self.logger)
        else:
            self.use_structural_caption = False

        self.image_token_offset = image_token_offset

        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    Using {self.index_manager}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.text_token_length = text_token_length
        self.image_token_length = image_token_length
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        self.mask_dataset = None

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_instruction(self, ind):
        try:
            instruction = self.index_manager.get_attribute(ind, column="instruction")
        except Exception as e:
            self.handle_exception_message(self.get_instruction, e)
            instruction = ""
        instruction = str(instruction).strip()

        # Remove meaningless characters
        instruction = instruction.replace("\\N", "").strip("，,")

        return instruction

    def get_prompt(self, ind):
        try:
            if self.inpainting_caption_rate > 0.0 and random.random() < self.inpainting_caption_rate:
                prompt = self.index_manager.get_attribute(ind, self.inpainting_caption_col)
                if self.use_structural_caption:
                    prompt = self.caption_aug.caption_aug(prompt)
            else:
                prompt = self.index_manager.get_attribute(ind, self.inpainting_text_col)
        except Exception as e:
            self.handle_exception_message(self.get_text, e)
            prompt = ""
        prompt = str(prompt).strip()

        # Remove meaningless characters
        prompt = prompt.replace("\\N", "").strip("，,")

        return prompt

    def get_img_token(self, ind, column, shift=True):
        """ Get image token from a given index, column and shadow.

        Notes
        -----
        This function handles the exception when the image token is not available. In this case, it will return
        a gray image token tensor with shape [32, 32] and all elements are 8846, which is the token id for gray
        color for `88-magvitv2-hy_241024` tokenizer. For other tokenizers, the gray token id should be adjusted,
        and the function will raise an error to avoid potential bugs.
        """
        try:
            image_token = self.index_manager.get_attribute(ind, column)
            image_token = np.array(ast.literal_eval(image_token))
            image_token_shape = image_token.shape # h, w
            image_token = torch.tensor(image_token, dtype=torch.long).reshape(-1)
            token_flag = "normal"
        except Exception as e:
            if self.args.vae_type != "88-vqgan-hy_241024":
                raise NotImplementedError(f"{self.get_img_token.__name__} | {e.__class__.__name__}: {str(e)}, we now only support handle of exception for vae_type=88-vqgan-hy_241024 when get_token fails.")
            # TODO(ckczzjzhang) find an elegant way to handle this exception
            self.logger.error(f"{self.get_img_token.__name__} | {e.__class__.__name__}: {str(e)}")
            image_token = torch.tensor([8846] * 1024, dtype=torch.long)
            image_token_shape = [32, 32] # h, w
            token_flag = "gray"
        return (
            image_token + self.image_token_offset if shift else image_token,
            torch.tensor([image_token_shape[1], image_token_shape[0]], dtype=torch.long),
            token_flag,
        )

    def get_editing_data_item(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        eiditing_src_img_token_col = self.index_kwargs.get("eiditing_src_img_token_col", None)
        if eiditing_src_img_token_col is None:
            raise ValueError("Missing `eiditing_src_img_token_col` in `index_kwargs`")
        eiditing_tgt_img_token_col = self.index_kwargs.get("eiditing_tgt_img_token_col", None)
        if eiditing_tgt_img_token_col is None:
            raise ValueError("Missing `eiditing_tgt_img_token_col` in `index_kwargs`")
        instruction_uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)

        src_img_token, src_img_token_shape_wh, src_img_token_flag = self.get_img_token(index, column=eiditing_src_img_token_col)
        tgt_img_token, tgt_img_token_shape_wh, tgt_img_token_flag = self.get_img_token(index, column=eiditing_tgt_img_token_col)

        # TODO(ckczzjzhang) handle exceptions when token_flag is gray

        tokens, iw_ih_scatter_index, text_loss_mask, src_image_loss_mask, tgt_image_loss_mask = self.tokenizer.encode_ar(
            instruction=instruction,
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_p=instruction_uncond_p,
            img_token=tgt_img_token,
            src_img_token_lst=[src_img_token],
            add_iw_ih_token=self.add_iw_ih_token,
            use_front_boi_token=self.use_front_boi_token,
        )
        target_token = tokens.clone()
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100

        return {
            "tokens": tokens,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.cat([src_img_token_shape_wh, tgt_img_token_shape_wh], dim=0),
            "target_token": target_token,
            "text_loss_mask": text_loss_mask,
            "src_image_loss_mask": src_image_loss_mask,
            "tgt_image_loss_mask": tgt_image_loss_mask,
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

        img_token, img_token_shape_wh, img_token_flag = self.get_img_token(index, column=inpainting_img_token_col)

        # (h, w)
        mask = self.mask_dataset.get_mask((img_token_shape_wh[1].cpu().item(), img_token_shape_wh[0].cpu().item()))

        # TODO(ckczzjzhang) handle exceptions when token_flag is gray
        src_image_token = torch.where(mask.reshape(-1) == 1.0, self.tokenizer.special_token_map["<mask>"], img_token)
        tokens, iw_ih_scatter_index, text_loss_mask, src_image_loss_mask, tgt_image_loss_mask = self.tokenizer.encode_ar(
            instruction, prompt,
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_enabled=[False, True],   # [instruction=False, prompt=True]
            uncond_p=instruction_uncond_p,
            img_token=img_token,
            src_image_token_lst=[src_image_token],
            add_iw_ih_token=self.add_iw_ih_token,
            use_front_boi_token=self.use_front_boi_token,
        )
        target_token = tokens.clone()
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100

        return {
            "tokens": tokens,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.cat([img_token_shape_wh, img_token_shape_wh], dim=0),
            "target_token": target_token,
            "text_loss_mask": text_loss_mask,
            "src_image_loss_mask": src_image_loss_mask,
            "tgt_image_loss_mask": tgt_image_loss_mask,
        }
    
    def __getitem__(self, index):
        # Get dataset_tag
        dataset_tag = self.index_manager.get_attribute(index, column="dataset_tag")

        if dataset_tag.startswith("editing"):
            return self.get_editing_data_item(index)
        elif dataset_tag.startswith("inpainting"):
            return self.get_inpainting_data_item(index)
        else:
            raise NotImplementedError(f"Dataset tag {dataset_tag} is not supported")
