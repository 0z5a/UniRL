import io
import os
from collections import defaultdict
from functools import partial
import json
import random
from PIL import Image
import cv2

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from index_kits.resolution import ResolutionGroup
from index_kits import arrow_mapper
try:
    import pycocotools.mask as mask_util
except ImportError:
    mask_util = None
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from ..ar.mask_schedulers import create_attention_mask_general
from hymm.constants import VAE_META_INFO
from hymm.models import TokenizerWrapper
from hymm.utils.helpers import to_2tuple, default
from hymm.utils.file_utils import log_in_safe_logger
from hymm.data_kits.caption_strategy.caption_process import CaptionAug
from .index_dataset import IndexDataset
from .irregular_mask_dataset import IrregularMaskDataset
from ..models.basic.rope import get_3d_rope


class InstructionTuningTransfusionArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=256,
        image_token_length=1024,
        text_token_length=256,
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
        self.training_image_size = to_2tuple(training_image_size)
        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.add_timestep_token = self.args.add_timestep_token
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

        # Arguments for id preservation generation
        self.id_caption_sample_ratio = index_kwargs.get("id_caption_sample_ratio", None)
        if self.id_caption_sample_ratio is not None:
            self.id_caption_sample_ratio = json.loads(self.id_caption_sample_ratio)
            self.id_caption_aug = CaptionAug(self.id_caption_sample_ratio, logger=self.logger)

        shadow_file_fn = {}  # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        self.face_analysis_arrow_suffix = index_kwargs.get("face_analysis_arrow_suffix", None)
        if self.face_analysis_arrow_suffix is not None:
            shadow_file_fn.update({"face_analysis": partial(arrow_mapper, suffix=self.face_analysis_arrow_suffix)})
        index_load_kwargs["shadow_file_fn"] = shadow_file_fn
        
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    Using {self.index_manager}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100
        # tokenizer
        self.text_token_length = text_token_length
        self.image_token_length = image_token_length
        self.image_token_length_clip = args.get("image_token_length_clip", None)
        tokenizer = default(tokenizer, tokenizer_name)
        if isinstance(tokenizer, str):
            self.tokenizer = TokenizerWrapper(tokenizer_name, self.logger)
        else:
            self.tokenizer = tokenizer

        # Call __post_init__ to do some post initialization
        post_kwargs = post_kwargs or {}
        self.__post_init__(**post_kwargs)

    def __post_init__(self, **kwargs):
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.trans_type = self.vae_meta_info["trans_type"]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        default_src_condition_type = ["vae"]
        self.src_condition_type = self.args.get("src_condition_type", default_src_condition_type)

        if isinstance(self.src_condition_type, str):
            self.src_condition_type = self.src_condition_type.split("_cat_")
        else: 
            assert isinstance(self.src_condition_type, list), \
                f"src_condition_type should be a list,e.g, ['vae'], ['face_embed', 'clip'], but got {type(self.src_condition_type)}"

        self.patch_size = self.args.patch_size

        if self.trans_type == "-11":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ]
            )
        elif self.trans_type == "01":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                ]
            )
        else:
            raise ValueError("Invalid trans_type: {}".format(self.trans_type))

        self.mask_dataset = None
        self.face_analysis = None
        self.use_3d_rope = self.args.get('rope_type', 'default') in ['3d', '3d-interleave']

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

    def get_id_prompt(self, ind, id_tgt_caption_col):
        try:
            prompt = self.index_manager.get_attribute(ind, id_tgt_caption_col)
            prompt = self.id_caption_aug.caption_aug(prompt)
        except Exception as e:
            self.handle_exception_message(self.get_id_prompt, e)
            prompt = ""
        prompt = str(prompt).strip()

        # Remove meaningless characters
        prompt = prompt.replace("\\N", "").strip("，,")

        return prompt

    def get_src_tgt_index_from_count(self, ind, unique=True, valid_bbox_index=None):
        if valid_bbox_index is None:
            count = self.index_manager.get_attribute(ind, "count")
            valid_bbox_index = range(count)
        if unique:
            # Choose 2 different images from count using random.sample
            indices = random.sample(valid_bbox_index, 2)
        else:
            indices = random.choices(valid_bbox_index, k=2)
        return indices

    def get_raw_image(self, index, column):
        try:
            img_bytes = self.index_manager.get_attribute(index, column)
            image_bytes = io.BytesIO(img_bytes)
            image_bytes.seek(0)
            ret = Image.open(image_bytes).convert("RGB")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"{type(e)}: {e}")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_image_with_size(self, index, column):
        image, image_flag = self.get_raw_image(index, column=column)

        origin_size = image.size  # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs, image_flag

    def get_image_with_size_resize_and_pad(self, index, column, crop_bbox=None):
        image, image_flag = self.get_raw_image(index, column=column)
        if crop_bbox is not None:
            log_in_safe_logger(image.size, self.logger, f'input image.size before crop using {crop_bbox} in get_image_with_size_resize_and_pad')
            image = image.crop(crop_bbox)
            crop_coords_xy = (crop_bbox[0], crop_bbox[1])
            log_in_safe_logger(image.size, self.logger, f'input image.size after crop using {crop_bbox} in get_image_with_size_resize_and_pad')
        else:
            crop_coords_xy = (0,0)
        origin_size = image.size  # (w_ori, h_ori)
        target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can reuse resize_and_pad in index kit
        image_tensor = self.pil_image_to_tensor(image)
        image_tensor = InstructionTuningTransfusionArrowStream.resize_and_pad(
            image_tensor[None, ...], 
            max_side=target_size, 
            pad_to_max_side=True, 
            logger=self.logger
         )[0, ...]
        

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": crop_coords_xy,
        }
        return image_tensor, kwargs, image_flag

    def get_editing_data_item(self, index):
        # Get instruction
        instruction = self.get_instruction(index)
        eiditing_src_img_col = self.index_kwargs.get("eiditing_src_img_col", None)
        if eiditing_src_img_col is None:
            raise ValueError("Missing `eiditing_src_img_col` in `index_kwargs`")
        eiditing_tgt_img_col = self.index_kwargs.get("eiditing_tgt_img_col", None)
        if eiditing_tgt_img_col is None:
            raise ValueError("Missing `eiditing_tgt_img_col` in `index_kwargs`")
        uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)

        src_image, _, src_image_flag = self.get_image_with_size(index, eiditing_src_img_col)
        tgt_image, _, tgt_image_flag = self.get_image_with_size(index, eiditing_tgt_img_col)

        # TODO(ckczzjzhang) handle exceptions when image_flag is gray

        src_h, src_w = src_image.shape[1], src_image.shape[2]
        assert src_h % (self.downsample_factor[0] * self.patch_size) == 0 and src_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({src_h} x {src_h}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_src_image_token_length = (src_h // (self.downsample_factor[0] * self.patch_size)) * (src_w // (self.downsample_factor[1] * self.patch_size))

        tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]
        assert tgt_h % (self.downsample_factor[0] * self.patch_size) == 0 and tgt_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({tgt_h} x {tgt_w}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_tgt_image_token_length = (tgt_h // (self.downsample_factor[0] * self.patch_size)) * (tgt_w // (self.downsample_factor[1] * self.patch_size))

        tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion(
            instruction,
            image_token_length=actual_tgt_image_token_length,
            src_image_token_lengths=[actual_src_image_token_length],
            max_text_token_length=self.text_token_length + 1,
            max_image_token_length=self.image_token_length,
            uncond_p=uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=[False, False, False],
        )
        # the target_token here is useless, because the discrete loss will not be computed in instruction tuning mode
        target_token = tokens.clone()
        target_token[text_mask == 0.0] = -100

        return {
            "src_image": src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.tensor([src_w, src_h, tgt_w, tgt_h], dtype=torch.long),
            "target_token": target_token,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
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

        inpainting_img_col = self.index_kwargs.get("inpainting_img_col", None)
        if inpainting_img_col is None:
            raise ValueError("Missing `inpainting_img_col` in `index_kwargs`")

        uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)

        src_image, _, src_image_flag = self.get_image_with_size(index, inpainting_img_col)
        # TODO(ckczzjzhang) handle exceptions when token_flag is gray

        src_h, src_w = src_image.shape[1], src_image.shape[2]
        assert src_h % (self.downsample_factor[0] * self.patch_size) == 0 and src_w % (self.downsample_factor[1] * self.patch_size) == 0, f"Image size should be divisible by downsample_factor * patch_size, but got ({src_h} x {src_h}) with downsample_factor={self.downsample_factor} and patch_size={self.patch_size}"
        actual_src_image_token_length = (src_h // (self.downsample_factor[0] * self.patch_size)) * (src_w // (self.downsample_factor[1] * self.patch_size))

        # (1, h, w)
        mask = self.mask_dataset.get_mask((src_h, src_w)).unsqueeze(0)
        # mask with gray
        masked_src_image = src_image * (1 - mask)
        tgt_image = src_image

        tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion(
            instruction, prompt,
            image_token_length=actual_src_image_token_length,
            src_image_token_lengths=[actual_src_image_token_length],
            max_text_token_length=self.text_token_length + 1, 
            max_image_token_length=self.image_token_length,
            uncond_enabled=[False, True],   # [instruction=False, prompt=True]
            uncond_p=uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
            pred_text_boi_eos=[False, False, False],
        )
        # the target_token here is useless, because the discrete loss will not be computed in instruction tuning mode
        target_token = tokens.clone()
        target_token[text_mask == 0.0] = -100

        return {
            "src_image": masked_src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": torch.tensor([src_w, src_h, src_w, src_h], dtype=torch.long),
            "target_token": target_token,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
        }
    
    @staticmethod
    def get_face_info(face_analysis, face_image_torch, logger=None):
        """
            get the face info from the face image
        """
        face_image_np = face_image_torch.cpu().numpy()
        face_image_np = einops.rearrange(face_image_np, 'c h w -> h w c')
        face_image_np = (face_image_np + 1) * 127.5
        face_image_np = face_image_np.astype(np.uint8) # h w c uint8 range: 0, 255
        face_analysis_input = cv2.cvtColor(face_image_np, cv2.COLOR_RGB2BGR)
        log_in_safe_logger(face_analysis_input, logger, 'face_analysis_input')
        face_info = face_analysis.get(face_analysis_input)
        return face_info

    @staticmethod
    def crop_face_image_torch(face_analysis, face_image_torch, max_side=[256, 256], logger=None, src_condition_type=["vae"], return_bbox=False):
        """
            crop the face area from image torch tensor, resize to max_side=256, and return the face image torch tensor

        Args:
            face_image_torch (torch tensor): [3, sh, sw] float32 range: -1, 1
            max_side (list of int tw, th, optional): For tuple [ (tw, sw), (th, sh)], resize side with smaller t/s to corresponding axis, pad larger ratio axis to max_side
        Returns:
            torch tensor: [3, tw, th] float32 range: -1, 1
        """
        face_info = InstructionTuningTransfusionArrowStream.get_face_info(face_analysis, face_image_torch, logger)
        

        # Handle multiple faces, small face, and negative face_bbox and no face

        if len(face_info) >= 1:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            log_in_safe_logger(face_info, logger, 'face_info')
            face_embedding = face_info.get('embedding', None)
            if isinstance(face_embedding, np.ndarray):
                face_embedding = torch.from_numpy(face_embedding)
            face_bbox = face_info['bbox']
            # if area of face is too small, skip
            # TODO: 10% probability to get negative face_bbox coordinates; can be improved later
            if face_bbox[2] - face_bbox[0] < 10 or face_bbox[3] - face_bbox[1] < 10:
                face_image_crop_torch = face_image_torch
            else:
                face_image_crop_torch = face_image_torch[:, int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
            
            n, h, w = face_image_crop_torch.shape
            if h < 10 or w < 10:
                logger.info(f"WARNING: face bbox is {face_bbox}")
                logger.info(f"WARNING: face area {face_image_crop_torch.shape} is too small, skip face crop; set as original image {face_image_torch.shape}")
                face_image_crop_torch = face_image_torch
            
        else:
            logger.info(f"WARNING: no face detected, skip face crop")
            face_image_crop_torch = face_image_torch
            face_embedding = None
            face_bbox = None
        if "face_embed" in src_condition_type:
            if face_embedding is None:
                logger.info(f"WARNING: {src_condition_type=}, but face embedding is None, set as zero")
                face_embedding = torch.zeros([512], dtype=torch.float32)
                face_bbox = torch.zeros([4], dtype=torch.float32)
        try:
            face_image_crop_resize_torch = InstructionTuningTransfusionArrowStream.resize_and_pad(face_image_crop_torch[None, ...], max_side=max_side, pad_to_max_side=True, logger=logger)[0, ...]
        except Exception as e:
            logger.info(f"Error: input face_image_torch shape: {face_image_torch.shape}; face_image_crop_torch shape: {face_image_crop_torch.shape}")
            raise e
        if not return_bbox:
            return face_image_crop_resize_torch, face_embedding
        else:
            return face_image_crop_resize_torch, face_embedding, face_bbox
    
    @staticmethod
    def resize_and_pad( input_image, max_side=[256, 256], size=None, 
                pad_to_max_side=True, base_pixel_number=1, logger=None):
        """
        resize torch tensor [n, c, h, w] range(-1, 1), resize long side to max_side, pad short side to max_side
        Args:
            input_image (torch tensor): [n, c, h, w] float number
            max_side (list of int tw, th, optional): For tuple [ (tw, sw), (th, sh)], resize side with smaller t/s to corresponding axis, pad larger ratio axis to max_side
            pad_to_max_side (bool, optional): pad short side to max_side
            base_pixel_number (int, optional): resize to base_pixel_number
        Returns:
            torch tensor: [n, c, max_side, w* (max_side/h)] or [n, c, h * (max_side/w), max_side] -1, 1
            
        """
        tw, th = max_side
        log_in_safe_logger(f"max_side: {max_side}", logger, 'max_side')
        n, c, h, w = input_image.shape
        if size is not None:
            h_resize_new, w_resize_new = size
        else:
            # Calculate the ratio to resize the image so the larger side is max_side
            ratio = min(tw / w, th / h)
            w, h = round(ratio * w), round(ratio * h)
            w_resize_new = (w // base_pixel_number) * base_pixel_number
            h_resize_new = (h // base_pixel_number) * base_pixel_number
        
        input_image = torch.nn.functional.interpolate(input_image, size=(h_resize_new, w_resize_new), mode='bicubic')
        input_image = input_image.clamp(-1, 1)
        log_in_safe_logger(input_image, logger, 'input_image after interpolate in resize_and_pad')

        if pad_to_max_side:
            res = torch.zeros([n, c, th, tw], dtype=torch.float32)
            offset_x = (tw - w_resize_new) // 2
            offset_y = (th - h_resize_new) // 2
            res[:, :, offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = input_image
            input_image = res
        return input_image
    


    def get_id_data_item(self, index):
        
        if self.face_analysis_arrow_suffix is None:
            # Calculate face embedding online
            if self.face_analysis is None and self.args.get("face_crop", False) is True:
                ASSETS_BASE = os.getenv("ASSETS_BASE", "/apdcephfs_gy2/share_302507476/1_public_models/hymm_ar_assets").rstrip('/')
                insightface_path = f"{ASSETS_BASE}/others/insightface"
                name = 'buffalo_l' # From large to small, support antelopev2, buffalo_l, buffalo_sc
                allowed_modules = ['detection']
                if "face_embed" in self.src_condition_type:
                    allowed_modules.append('recognition')
                self.logger.info(f"Loading face analysis, name: {name}, allowed_modules: {allowed_modules}, providers CPUExecutionProvider, from {insightface_path}")
                self.face_analysis = FaceAnalysis(name=name, root=insightface_path, allowed_modules=allowed_modules, providers=['CPUExecutionProvider'])
                self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    
            # Get instruction
            src_index, tgt_index = self.get_src_tgt_index_from_count(index, unique=self.args.get("face_index_unique", True))
            id_src_img_col = f"image_{src_index}"
            id_tgt_img_col = f"image_{tgt_index}"

            log_in_safe_logger(f"src_index: {src_index}, tgt_index: {tgt_index}", self.logger, 'src_index, tgt_index')
            prompt = self.get_id_prompt(index, f"caption_{tgt_index}")
            # TODO (chenyangqi) a better strategy is to first crop the face in original image, then resize to max_side
            src_image, src_kwargs, src_image_flag = self.get_image_with_size(index, id_src_img_col)
            tgt_image, tgt_kwargs, tgt_image_flag = self.get_image_with_size(index, id_tgt_img_col)

            if self.args.get("face_crop", False) is True:
                src_image, src_face_embedding = InstructionTuningTransfusionArrowStream.crop_face_image_torch(
                    self.face_analysis, src_image, 
                    max_side=src_kwargs.get('target_size', [256, 256]), 
                    logger=self.logger, 
                    src_condition_type=self.src_condition_type
                )
        else:
            arrow_data_list_index = self.get_id_src_embedding_tgt_image(index)
            src_image, src_face_embedding, tgt_image, prompt = arrow_data_list_index
            log_in_safe_logger(arrow_data_list_index, self.logger, 'arrow_data_list_index in get_id_data_item')

        uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)
        face_uncond_p = self.index_kwargs.get("face_uncond_p", 0.0)
        
        src_h, src_w = src_image.shape[1], src_image.shape[2]
        tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]
        actual_tgt_image_token_length = self.tokenizer.get_actual_image_token_length(tgt_image, self.vae_meta_info, patch_size=self.patch_size)

        actual_src_image_token_length_vae = self.tokenizer.get_actual_image_token_length(src_image, self.vae_meta_info, patch_size=self.patch_size)
        if "clip" in self.src_condition_type:
            assert self.clip_meta_info is not None, "clip_meta_info is None, but src_condition_type contains clip"
            actual_src_image_token_length_clip = self.tokenizer.get_actual_image_token_length(src_image, self.clip_meta_info, patch_size=self.args.get("patch_size_clip", 1))
        else:
            actual_src_image_token_length_clip = None

        # Loop over self.src_condition_type, and get the actual_src_image_token_length and max_image_token_length
        src_condition_lengths_dict = self.tokenizer.prepare_src_condition_lengths(
            actual_src_image_token_length_vae,
            self.image_token_length,
            actual_src_image_token_length_clip,
            self.image_token_length_clip,
            self.src_condition_type,
            self.args.get("face_bof_eof", False),
            self.args.get("resampler_token_length", None)
        )

        src_condition_lengths_dict["max_image_token_length_list"].append(self.image_token_length)
        log_in_safe_logger(src_condition_lengths_dict, self.logger, 'src_condition_lengths_dict after prepare_src_condition_lengths')
        
        log_in_safe_logger(self.add_iw_ih_token, self.logger, 'self.add_iw_ih_token')
        log_in_safe_logger(actual_tgt_image_token_length, self.logger, 'actual_tgt_image_token_length')

        
        do_uncond_drop_face = (face_uncond_p is not None) and (random.random() < face_uncond_p)
        if do_uncond_drop_face:
            if src_face_embedding is not None:
                src_face_embedding = np.zeros_like(src_face_embedding)
            if src_image is not None:
                src_image = torch.zeros_like(src_image)
        log_in_safe_logger(src_face_embedding, self.logger, f'src_face_embedding after uncond_drop_face with probability: {uncond_p}')
        
        tokens, iw_ih_scatter_index, _, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion_faceid(
            prompt,
            image_token_length=     actual_tgt_image_token_length,
            src_image_token_lengths=    src_condition_lengths_dict["actual_src_image_token_length_list"],
            src_face_token_lengths=     src_condition_lengths_dict["actual_src_face_token_length_list"],
            max_text_token_length=      self.text_token_length + 1, 
            max_image_token_length=     src_condition_lengths_dict["max_image_token_length_list"],
            max_face_token_length=      src_condition_lengths_dict["max_face_token_length_list"],
            uncond_p=                   uncond_p,
            add_iw_ih_token=            self.add_iw_ih_token,
        )

        target_token = tokens.clone()
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<img>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<face>"]] = -100
        

        actual_src_image_token_length_list = src_condition_lengths_dict["actual_src_image_token_length_list"]
        iw_ih_scatter_src = torch.tensor([src_w, src_h]* len(actual_src_image_token_length_list) + [tgt_w, tgt_h], dtype=torch.long)

        log_in_safe_logger(tokens[::10], self.logger, 'tokens[::10]: ')
        log_in_safe_logger(target_token[::10], self.logger, 'target_token[::10]: ')

        return_dict = {
            "src_image": src_image,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "iw_ih_scatter_index": iw_ih_scatter_index,
            "iw_ih_scatter_src": iw_ih_scatter_src,
            "target_token": target_token,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
        }
        if src_face_embedding is not None:
            return_dict["src_face_embedding"] = src_face_embedding
        log_in_safe_logger(return_dict, self.logger, 'return_dict in get_id_data_item')
        return return_dict

    def get_id_face_analysis(self, ind):
        try:
            bbox = self.index_manager.get_attribute(ind, "bbox", shadow="face_analysis")
            embedding = self.index_manager.get_attribute(ind, "embedding", shadow="face_analysis")
            bbox_np = np.frombuffer(bbox, dtype=np.float32)
            embedding_np = np.frombuffer(embedding, dtype=np.float32)
            bbox_np = bbox_np.reshape(-1, 4)
            embedding_np = embedding_np.reshape(-1, 512)
            assert bbox_np.shape[0] == embedding_np.shape[0], f"bbox and embedding shape mismatch: {bbox_np.shape[0]} != {embedding_np.shape[0]}"
        except Exception as e:
            self.handle_exception_message(self.get_id_face_analysis, e)
            bbox_np = np.zeros((2, 4), dtype=np.float32)
            embedding_np = np.zeros((2, 512), dtype=np.float32)

        return bbox_np, embedding_np

    def get_valid_bbox_index(self, index,bbox):
        """
        bbox: [num_bbox, 4], check width_height_positive and left_top_right_bottom_in_image, return the index of the valid bbox
        each row in bbox is left, top, right, bottom
        bbox == 0,0,0,0 means no face or multiple faces detected
        Args:
            bbox (_type_): [num_bbox, 4]
        """
        count = bbox.shape[0]
        assert count == self.index_manager.get_attribute(index, f"count"), f'bbox.shape[0] should be equal with count index_manager, but  {count}!={self.index_manager.get_attribute(index, f"count")}'
        
        image_width_np_array = np.array([self.index_manager.get_attribute(index, f"width_{image_count_i}") for image_count_i in range(count)])
        image_height_np_array = np.array([self.index_manager.get_attribute(index, f"height_{image_count_i}") for image_count_i in range(count)])
        bbox_left_top_right_bottom_in_image_index = (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 2] <= image_width_np_array) & (bbox[:, 3] <= image_height_np_array)

        # Check if there are any invalid bboxes and clip them
        if not bbox_left_top_right_bottom_in_image_index.all():
            invalid_indices = np.where(~bbox_left_top_right_bottom_in_image_index)[0]
            bbox_clip = bbox.copy()
            # Clip the bbox values
            bbox_clip[:, 0] = np.clip(bbox[:, 0], 0, image_width_np_array)  # left
            bbox_clip[:, 1] = np.clip(bbox[:, 1], 0, image_height_np_array)  # top
            bbox_clip[:, 2] = np.clip(bbox[:, 2], 0, image_width_np_array)  # right
            bbox_clip[:, 3] = np.clip(bbox[:, 3], 0, image_height_np_array)  # bottom
            
            self.logger.warning(f"Found invalid bboxes at indices {invalid_indices}. \
                                Original bbox values: {bbox[invalid_indices]}; \
                                Clipped bbox values: {bbox_clip[invalid_indices]}")
            bbox = bbox_clip

        bbox_width_height_positive_index = (bbox[:, 0] < bbox[:, 2]) & (bbox[:, 1] < bbox[:, 3])        
        valid_bbox_index = np.where(bbox_width_height_positive_index)[0]
        # convert valid_bbox_index to list
        valid_bbox_index = valid_bbox_index.tolist()
        # if no valid bbox, loggger.error and return None
        if len(valid_bbox_index) == 0:
            self.logger.error(f"No valid bbox found in {count} images of index {index}, clipped bbox: {bbox}")
        return valid_bbox_index, bbox

    def get_id_src_embedding_tgt_image(self, index):
        """
            (1) get bbox from     self.index_manager.get_attribute(index, column);
            (2) if not all bbox are zero: remove zero bbox in the bbox list, which means no / multiple faces detected;
            (3) sample src_index, tgt_index from the left of the bbox list;
            (4) use src_index face embedding as src_face_embedding; resize and pad face in src_index image using the src_index bbox;
            (5) use tgt_index image as tgt_image;
            (6) return the data_item;
        """

        bbox, embedding = self.get_id_face_analysis(index)
        valid_bbox_index, bbox = self.get_valid_bbox_index(index, bbox)
        log_in_safe_logger(bbox, self.logger, f"bbox of index {index} after get_valid_bbox_index, may be get clipped")
        log_in_safe_logger(valid_bbox_index, self.logger, f"valid_bbox_index obtained from bbox of index {index}")
        # (TODO) filter invalid id using index_manager filter, not in dataloader
        valid_bbox_index_sample = valid_bbox_index if len(valid_bbox_index) >= 2 else range(bbox.shape[0])
        src_index, tgt_index = self.get_src_tgt_index_from_count(index, unique=self.args.get("face_index_unique", True), valid_bbox_index=valid_bbox_index_sample)

        log_in_safe_logger(f"src_index: {src_index}, tgt_index: {tgt_index}", self.logger, 'src_index, tgt_index')
        prompt = self.get_id_prompt(index, f"caption_{tgt_index}")
        src_face_embedding = embedding[src_index]

        src_image, src_kwargs, src_image_flag = self.get_image_with_size_resize_and_pad(
            index, f"image_{src_index}", crop_bbox=bbox[src_index] if len(valid_bbox_index) >= 2 else None)
        tgt_image, tgt_kwargs, tgt_image_flag = self.get_image_with_size(index, f"image_{tgt_index}")

        
        log_in_safe_logger(src_image, self.logger, 'src_image in get_id_data_item')

        return src_image, src_face_embedding, tgt_image, prompt

    def get_target_size(self, height, width, align=16):
        if not hasattr(self, "resolutions"):
            anchor_size = self.training_image_size
            step = anchor_size // 16
            self.resolutions = ResolutionGroup(anchor_size, step, align=align)
        tw, th = self.resolutions.get_target_size(width, height)
        return th, tw

    def __getitem__(self, index):
        # Get dataset_tag
        dataset_tag = self.index_manager.get_attribute(index, column="dataset_tag")

        if dataset_tag.startswith("editing"):
            return self.get_editing_data_item(index)
        elif dataset_tag.startswith("inpainting"):
            return self.get_inpainting_data_item(index)
        elif dataset_tag.startswith("id"):
            return self.get_id_data_item(index)
        else:
            raise NotImplementedError(f"Dataset tag {dataset_tag} is not supported")
            


class InstructionTuningTransfusionArrowStream2(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=256,
        image_token_length=1024,
        text_token_length=256,
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
        self.training_image_size = to_2tuple(training_image_size)
        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.add_timestep_token = self.args.add_timestep_token
        self.use_front_boi_token = self.args.use_front_boi_token

        self.index_kwargs = index_kwargs
        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=index_kwargs.get("ceph_base", None),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )
        
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
        self.vae_meta_info = VAE_META_INFO[self.args.vae_type]
        self.trans_type = self.vae_meta_info["trans_type"]
        self.downsample_factor = self.vae_meta_info["downsample_factor"]
        self.patch_size = self.args.patch_size

        if self.trans_type == "-11":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                    transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
                ]
            )
        elif self.trans_type == "01":
            self.pil_image_to_tensor = transforms.Compose(
                [
                    transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                ]
            )
        else:
            raise ValueError("Invalid trans_type: {}".format(self.trans_type))

        self.use_3d_rope = self.args.get('rope_type', 'default') in ['3d', '3d-interleave']

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_image_with_size(self, index, column):
        image, image_flag = self.get_raw_image(index, column=column)

        origin_size = image.size  # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, kwargs, image_flag

    def parse_annotations(self, index, image_t):
        # image_t means image_tensor with shape [3, H, W] and dynamic range [-1, 1]
        tags = self.index_manager.get_attribute(index, "tags")
        # [{
        #     "class_name": "xxx",
        #     "bbox": [x1, y1, x2, y2],
        #     "segmentation": {
        #         "size": [1024, 1024],
        #         "counts": "xxx"
        #     },
        #     "score": [0.98]
        # }, {...}, ...]
        annotations = json.loads(self.index_manager.get_attribute(index, "annotations"))

        valid_annotations = []
        # Find those annotations with tags. Each tag only match the first annotation.
        for tag in tags:
            for anno in annotations:
                if tag in anno["class_name"]:
                    valid_annotations.append(anno)
                    break
        # If no matched annotation, use the first annotation (already sorted by some rules).
        # See __data/instruct/subject_driven/process_data.py:sort_anno_by_hole_ratio() for details.
        if not valid_annotations:
            # Assume annotations is not empty
            valid_annotations.append(annotations[0])

        # Convert annotations raw data to masked image
        ref_objects = []
        for anno in valid_annotations[:3]:  # No more than 3 reference objects
            # Get mask
            mask = torch.from_numpy(mask_util.decode(anno["segmentation"]))   # [h, w] the same shape with the target image
            mask = F.interpolate(mask[None, None].float(), image_t.shape[-2:], mode="nearest")[0, 0]
            if mask.max() == 0:
                if hasattr(self.index_manager, 'ind_mapper'):
                    in_json_index = self.index_manager.ind_mapper[index]
                else:
                    in_json_index = self.index_manager.indices[index]
                print(f"Warning: mask.max() == 0 in {in_json_index=}")
                continue
            # Find the tight bounding box and crop the image
            xrng = torch.where(mask.max(dim=0)[0] > 0)[0]
            yrng = torch.where(mask.max(dim=1)[0] > 0)[0]
            x1, x2 = xrng[0], xrng[-1] + 1
            y1, y2 = yrng[0], yrng[-1] + 1
            sub_image = image_t[:, y1:y2, x1:x2]
            sub_mask = mask[None, y1:y2, x1:x2]
            ref_obj = torch.clamp(sub_image + (1 - sub_mask) * 2, -1, 1)   # Add 2 to make the background white
            # Extend the sub_image as a square image
            if y2 - y1 > x2 - x1:
                x_pad = (y2 - y1) - (x2 - x1)
                y_pad = 0
            else:
                y_pad = (x2 - x1) - (y2 - y1)
                x_pad = 0
            # Extend the sub_image at least 128 pixels in each direction
            x_pad += max(128 - (x2 - x1) - x_pad, 0)
            y_pad += max(128 - (y2 - y1) - y_pad, 0)
            if x_pad or y_pad:
                pad_counts = (
                    x_pad // 2,
                    x_pad - x_pad // 2,
                    y_pad // 2,
                    y_pad - y_pad // 2,
                )
                ref_obj = F.pad(ref_obj, pad_counts, mode="constant", value=1)  # White background
            # # Resize the ref_obj to align with the multiple of 16 and no more than image_size^2 pixels
            # h, w = ref_obj.shape[-2:]
            # aligned_h = (h + 15) // 16 * 16
            # aligned_w = (w + 15) // 16 * 16
            # if aligned_h * aligned_w > self.training_image_size ** 2:
            #     aligned_h, aligned_w = self.get_target_size(h, w)
            ref_obj = F.interpolate(
                # ref_obj[None], (aligned_h, aligned_w), mode="bilinear", align_corners=False
                ref_obj[None], (256, 256), mode="bilinear", align_corners=False
            )[0]
            ref_objects.append(ref_obj)

        # If no reference object, use a white image
        if len(ref_objects) == 0:
            ref_objects = [torch.zeros(3, 256, 256)]

        return ref_objects

    def __getitem__(self, index):
        uncond_p = self.index_kwargs.get("instruction_uncond_p", 0.0)
        # Get target image
        image, _, image_flag = self.get_image_with_size(index, 'image')
        # Get caption
        caption = self.index_manager.get_attribute(index, 'caption')
        # Get reference objects (already aligned with `downsample_factor * patch_size`)
        ref_objects = self.parse_annotations(index, image)

        h, w = image.shape[-2:]
        ds_factor = self.downsample_factor[0] * self.patch_size
        th = h // ds_factor
        tw = w // ds_factor
        assert h % th == 0 and w % tw == 0, (
            f"Image size should be divisible by downsample_factor * patch_size, but got {h}x{w} with {self.downsample_factor=} and {self.patch_size=}"
        )
        actual_image_token_length = th * tw

        tokens, iw_ih_scatter_index, text_mask, src_image_mask, tgt_image_mask = self.tokenizer.encode_transfusion_editing2(
            caption,
            src_image_token_lengths=[
                ref_obj.size(1) * ref_obj.size(2) // ds_factor ** 2
                for ref_obj in ref_objects
            ],
            tgt_image_token_length=actual_image_token_length,
            max_text_token_length=self.text_token_length + 1,
            max_total_token_length=self.text_token_length + 1 + self.image_token_length * 4,
            uncond_p=uncond_p,
            add_iw_ih_token=self.add_iw_ih_token,
            add_timestep_token=self.add_timestep_token,
            use_front_boi_token=self.use_front_boi_token,
        )
        target_token = tokens.clone()
        target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100
        target_token[target_token == self.tokenizer.special_token_map["<img>"]] = -100

        start_pos = torch.where(tokens == self.tokenizer.special_token_map["<img>"])[0][0].item()
        if self.use_3d_rope:
            img_pos = [start_pos + 3 * i for i in range(len(ref_objects) + 1)]
            freqs_cos, freqs_sin = get_3d_rope(
                self.args.rope_dim_list,
                [ref_obj.size(1) // ds_factor for ref_obj in ref_objects] + [th],
                [ref_obj.size(2) // ds_factor for ref_obj in ref_objects] + [tw],
                max_len=self.text_token_length + 1 + len(ref_objects) + (3 - len(ref_objects)) * self.image_token_length,
                img_pos=img_pos,
                device=None,
                theta=self.args.get('rope_theta', 10000),
                use_real=True,
                interleave=self.args.rope_type == "3d-interleave",
            )
        else:
            freqs_cos, freqs_sin = None, None

        src_token_slices = []
        for ref_obj in ref_objects:
            th_ = ref_obj.size(1) // ds_factor
            tw_ = ref_obj.size(2) // ds_factor
            src_token_slices.append(slice(start_pos, start_pos + th_ * tw_))
            start_pos += th_ * tw_ + 2
        tgt_token_slice = slice(start_pos, start_pos + actual_image_token_length)
        attention_mask = create_attention_mask_general(
            tokens[None, :-1],
            None,
            src_token_slices + [tgt_token_slice],
            mask_pad=False,
            return_inverse_mask=True,
            dtype=torch.float32,
            pad_endpoint=self.text_token_length,  # Only consider the text pad.
        )[0]

        return {
            "src_image": ref_objects,
            "tgt_image": image,
            "tokens": tokens,
            "target_token": target_token,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
        }

    @staticmethod
    def collate_fn(batch):
        src_images = [item["src_image"] for item in batch]
        tgt_image = torch.stack([item["tgt_image"] for item in batch])
        tokens = torch.stack([item["tokens"] for item in batch])
        target_token = torch.stack([item["target_token"] for item in batch])
        text_mask = torch.stack([item["text_mask"] for item in batch])
        src_image_mask = torch.stack([item["src_image_mask"] for item in batch])
        tgt_image_mask = torch.stack([item["tgt_image_mask"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        freqs_cos = torch.stack([item["freqs_cos"] for item in batch]) if batch[0].get("freqs_cos") is not None else None
        freqs_sin = torch.stack([item["freqs_sin"] for item in batch]) if batch[0].get("freqs_sin") is not None else None

        return {
            "src_images": src_images,
            "tgt_image": tgt_image,
            "tokens": tokens,
            "target_token": target_token,
            "text_mask": text_mask,
            "src_image_mask": src_image_mask,
            "tgt_image_mask": tgt_image_mask,
            "attention_mask": attention_mask,
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
        }
