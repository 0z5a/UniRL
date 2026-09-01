import ast
import random
import json
import warnings
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits
from index_kits import arrow_mapper
from processors.arrow_kits import arrow_mapper_adt2ceph

from hymm.utils.helpers import to_2tuple, default
from hymm.models import TokenizerWrapper
from hymm.data_kits.caption_strategy import load_caption_processor
from .index_dataset import IndexDataset


class TextImageArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=256,
        image_token_length=1024,
        image_token_offset=0,
        use_pre_extracted_token=False,
        text_token_length=256,
        uncond_p=0.0,
        tokenizer_name=None,
        raise_text_error=False,
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
            debug,
        )
        self.args = args
        self.training_image_size = to_2tuple(training_image_size)
        self.uncond_p = uncond_p
        self.raise_text_error = raise_text_error

        self.add_iw_ih_token = self.args.add_iw_ih_token
        self.use_front_boi_token = self.args.use_front_boi_token

        if index_kwargs is None:
            index_kwargs = {}

        self.image_key = index_kwargs.get("image_key", args.get("image_key", "image"))
        self.image_key_shadow = None
        self.image_key_arrow_suffix = None
        if '@' in self.image_key:
            self.image_key, self.image_key_arrow_suffix = self.image_key.split('@')
            self.image_key_shadow = self.image_key

        self.image_caption_rate = index_kwargs.get("image_caption_rate", 0.0)

        self.image_text_col = index_kwargs.get("image_text_col", None)
        if self.image_caption_rate < 1.0 and self.image_text_col is None:
            raise ValueError("Missing `image_text_col` in `index_kwargs`")
        if '@' in self.image_text_col:
            self.image_text_col, self.image_text_arrow_suffix = self.image_text_col.split('@')
            self.image_text_shadow = 'text'
        elif (image_text_arrow_suffix := index_kwargs.get("image_text_arrow_suffix")) is not None:
            # for backward compatibility
            self.image_text_arrow_suffix = image_text_arrow_suffix
            self.image_text_shadow = 'text'
            warnings.warn(
                "`image_text_arrow_suffix` is deprecated, please use "
                "`--image-text-col <image_text_col>@<arrow_suffix>` instead.", DeprecationWarning
            )
        else:
            self.image_text_arrow_suffix = None
            self.image_text_shadow = None

        self.image_caption_col = index_kwargs.get("image_caption_col", None)
        if self.image_caption_rate > 0.0 and self.image_caption_col is None:
            raise ValueError("Missing `image_caption_col` in `index_kwargs`")
        if self.image_caption_col is not None and '@' in self.image_caption_col:
            self.image_caption_col, self.image_caption_arrow_suffix = self.image_caption_col.split('@')
            self.image_caption_shadow = 'caption'
        elif (image_caption_arrow_suffix := index_kwargs.get("image_caption_arrow_suffix")) is not None:
            # for backward compatibility
            self.image_caption_arrow_suffix = image_caption_arrow_suffix
            self.image_caption_shadow = 'caption'
            warnings.warn(
                "`image_caption_arrow_suffix` is deprecated, please use "
                "`--image-caption-col <image_caption_col>@<arrow_suffix>` instead.", DeprecationWarning
            )
        else:
            self.image_caption_arrow_suffix = None
            self.image_caption_shadow = None

        self.caption_sample_ratio = index_kwargs.get("caption_sample_ratio", None)
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(
                name=default(args.caption_processor, 'caption_process'),
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
                kwargs=args.get('caption_processor_kwargs'),
            )
        else:
            self.use_structural_caption = False

        # 是否在特定场景下使用 `general_style` 列
        self.use_general_style = args.use_general_style
        # 是否优先使用 source_text (应用 image_caption_ratio 的概率)
        self.try_first_use_source_text = args.try_first_use_source_text

        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        shadow_file_fn = {'clip_score': partial(arrow_mapper, suffix='_clip_score')}  # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        if self.image_caption_rate < 1.0 and self.image_text_arrow_suffix is not None:
            shadow_file_fn.update({self.image_text_shadow: partial(arrow_mapper, suffix=self.image_text_arrow_suffix)})
        if self.image_caption_rate > 0.0 and self.image_caption_arrow_suffix is not None:
            shadow_file_fn.update(
                {self.image_caption_shadow: partial(arrow_mapper, suffix=self.image_caption_arrow_suffix)})
        if self.image_key_shadow:
            shadow_file_fn.update({self.image_key_shadow: partial(arrow_mapper, suffix=self.image_key_arrow_suffix)})
        self.use_pre_extracted_token = use_pre_extracted_token
        self.image_token_offset = image_token_offset
        if use_pre_extracted_token:
            self.image_token_arrow_suffix = index_kwargs["image_token_arrow_suffix"]
            self.image_token_col = index_kwargs["image_token_col"]
            self.image_token_arrow_ceph = index_kwargs.get("image_token_arrow_ceph", None)
            if self.image_token_arrow_ceph is None:
                shadow_file_fn.update({"token": partial(arrow_mapper, suffix=self.image_token_arrow_suffix)})
            else:
                shadow_file_fn.update({"token": partial(arrow_mapper_adt2ceph, suffix=self.image_token_arrow_suffix, ceph=self.image_token_arrow_ceph)})

        index_load_kwargs["shadow_file_fn"] = shadow_file_fn

        self.logger.info(f"    (T2I) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    (T2I) Using {self.index_manager}")
        self.logger.info(f"    Text col: {self.image_text_col} (suffix={self.image_text_arrow_suffix})")
        self.logger.info(f"    Caption col: {self.image_caption_col} (suffix={self.image_caption_arrow_suffix})")
        self.logger.info(f"    Caption rate: {self.image_caption_rate}")
        self.logger.info(f"    Caption sample ratio: {self.caption_sample_ratio}")
        if use_pre_extracted_token:
            self.logger.info(f"    Token col: {self.image_token_col} (suffix={self.image_token_arrow_suffix})")

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.tensor_to_pil_image = transforms.Compose(
            [
                transforms.Normalize([-1], [2]),
                transforms.ToPILImage(),
            ]
        )

        self.logger.info(f"Image transform: {self.pil_image_to_tensor}")

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
        pass

    def get_raw_image(self, index, image_key="image", shadow=None):
        try:
            if image_key == "image":
                ret = self.get_image_from_arrow(index, shadow=shadow)
            elif image_key == "url_cos":
                ret = self.get_image_from_url_cos(index, shadow=shadow)
            elif image_key == "cache_image":
                ret = Image.open(self.index_manager.get_attribute(index, 'cache_image', shadow=shadow)).convert("RGB")
            else:
                raise ValueError(f"Unknown image_key: {image_key}")
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f"({image_key=}, {index=}) {type(e)}: {e}. Fallback to gray image.")
            ret = Image.new("RGB", (self.training_image_size[0], self.training_image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag

    def get_image_with_size(self, index, image_key="image", shadow=None):
        image, image_flag = self.get_raw_image(index, image_key=image_key, shadow=shadow)

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

    def get_pil_image_with_size(self, index, image_key="image", shadow=None):
        image, image_flag = self.get_raw_image(index, image_key=image_key, shadow=shadow)

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

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image, kwargs, image_flag

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")

    def get_text(self, ind):
        try:
            if self.try_first_use_source_text:
                source = self.index_manager.get_attribute(ind, 'source')
                try:
                    clip_score = self.index_manager.get_attribute(ind, 'clip_score', shadow='clip_score')
                except:
                    clip_score = None
                if (
                        (source == 'laion2b_en_sd2.1base' or (clip_score is not None and clip_score >= 0.18))
                        and (self.image_caption_rate == 0 or random.random() >= self.image_caption_rate)
                ):
                    text = self.index_manager.get_attribute(ind, self.image_text_col, shadow=self.image_text_shadow)
                else:
                    text = self.index_manager.get_attribute(ind, self.image_caption_col, shadow=self.image_caption_shadow)
                    if self.use_general_style:
                        style_op = self.index_manager.get_attribute(ind, 'general_style')
                    else:
                        style_op = None
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text, style_op=style_op)

            else:
                if self.image_caption_rate > 0.0 and random.random() < self.image_caption_rate:
                    text = self.index_manager.get_attribute(ind, self.image_caption_col,
                                                            shadow=self.image_caption_shadow)
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text)
                else:
                    text = self.index_manager.get_attribute(ind, self.image_text_col, shadow=self.image_text_shadow)

        except Exception as e:
            if self.raise_text_error:
                print(f"#{torch.distributed.get_rank()}: indices {ind}")
                raise e
            self.handle_exception_message(self.get_text, e)
            text = ""
        if text is None:
            text = ""

        if isinstance(text, str):
            text = str(text).strip()
            # Remove meaningless characters
            text = text.replace("\\N", "").strip("，,")

        return text

    def get_token(self, ind, dtype=torch.long):
        """ Get image token from a given index

        Parameters
        ----------
        ind: int
            Index of the dataset
        dtype: torch.dtype
            Data type of the image token tensor

        Returns
        -------
        image_token: torch.LongTensor
            Image token tensor with shape [H*W]. The token ids are shifted by `image_token_offset`.
        image_token_shape: torch.LongTensor
            Image token shape with shape [2]. The first element is width and the second element is height.
        token_flag: str
            Flag to indicate the token type. It can be "normal" or "gray".

        Notes
        -----
        This function handles the exception when the image token is not available. In this case, it will return
        a gray image token tensor with shape [32, 32] and all elements are 8846, which is the token id for gray
        color for `88-magvitv2-hy_241024` tokenizer. For other tokenizers, the gray token id should be adjusted,
        and the function will raise an error to avoid potential bugs.
        """
        try:
            image_token = self.index_manager.get_attribute(ind, self.image_token_col, shadow="token")
            if isinstance(image_token, str):
                image_token = np.array(ast.literal_eval(image_token))
                image_token_shape = image_token.shape # h, w
            else:
                image_token = np.array(image_token)
                image_token_shape = [32, 32] # h, w
            image_token = torch.tensor(image_token, dtype=dtype).reshape(-1)
            token_flag = "normal"
        except Exception as e:
            if self.args.vae_type != "88-vqgan-hy_241024":
                raise NotImplementedError(f"{self.get_token.__name__} | {e.__class__.__name__}: {str(e)}, we now only support handle of exception for vae_type=88-vqgan-hy_241024 when get_token fails.")
            # TODO(ckczzjzhang) find an elegant way to handle this exception
            self.logger.error(f"{self.get_token.__name__} | {e.__class__.__name__}: {str(e)}")
            image_token = torch.tensor([8846] * 1024, dtype=dtype)
            image_token_shape = [32, 32] # h, w
            token_flag = "gray"
        return (
            image_token + self.image_token_offset,
            torch.tensor([image_token_shape[1], image_token_shape[0]], dtype=dtype),
            token_flag,
        )
    
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
        if self.use_pre_extracted_token:
            image = ""
            kwargs = {}
            image_token, image_token_shape_wh, token_flag = self.get_token(index)
            if token_flag == "gray":
                text = "A gray image"

            # for next token training, we add one token to text,
            # since we will omit last token for input while first token for target
            tokens, iw_ih_scatter_index, text_loss_mask, image_loss_mask = self.tokenizer.encode_ar(
                text,
                max_text_token_length=self.text_token_length + 1,
                max_image_token_length=self.image_token_length,
                uncond_p=self.uncond_p,
                add_iw_ih_token=self.add_iw_ih_token,
                use_front_boi_token=self.use_front_boi_token,
                image_token=image_token,
            )
            target_token = tokens.clone()
            target_token[target_token == self.tokenizer.special_token_map["<pad>"]] = -100
            target_token[target_token == self.tokenizer.special_token_map["<iw>"]] = -100
            target_token[target_token == self.tokenizer.special_token_map["<ih>"]] = -100
            target_token[target_token == self.tokenizer.special_token_map["<cfg>"]] = -100
        else:
            image, kwargs, image_flag = self.get_image_with_size(index, image_key=self.image_key, shadow=self.image_key_shadow)
            if image_flag == "gray":
                text = "A gray image"
            tokens, iw_ih_scatter_index, text_loss_mask, image_loss_mask = self.tokenizer.encode_ar(
                text,
                max_text_token_length=self.text_token_length + 1,
                max_image_token_length=self.image_token_length,
                uncond_p=self.uncond_p,
                add_iw_ih_token=False,   # fixed resolution, no need to add iw_ih_token
                use_front_boi_token=True,   # fixed resolution, use front boi token
            )
            target_token = tokens.clone() # will not be used

        # Prepare kwargs
        kwargs["index"] = index
        kwargs["text"] = text

        ret = {
            "image": image,
            "tokens": tokens,
            "target_token": target_token,
            "text_loss_mask": text_loss_mask,
            "image_loss_mask": image_loss_mask,
            "kwargs": {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        }

        if iw_ih_scatter_index is not None:
            ret.update({
                "iw_ih_scatter_index": iw_ih_scatter_index,
                "iw_ih_scatter_src": image_token_shape_wh,
            })

        return ret
