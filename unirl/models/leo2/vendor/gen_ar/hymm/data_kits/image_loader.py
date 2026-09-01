from collections import defaultdict
from functools import partial

import torch
import torchvision.transforms as transforms
from PIL import Image
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits
from index_kits import arrow_mapper
from processors.arrow_kits import arrow_mapper_adt2ceph

from hymm.utils.helpers import to_2tuple
from .index_dataset import IndexDataset


class ImageArrowStream(IndexDataset):
    def __init__(
        self,
        args,
        index_file=None,
        training_image_size=256,
        multireso=False,
        index_kwargs=None,
        post_kwargs=None,
        debug=False,
        logger=None,
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

        if index_kwargs is None:
            index_kwargs = {}

        self.image_key = index_kwargs.get("image_key", args.get("image_key", "image"))
        self.image_key_shadow = None
        self.image_key_arrow_suffix = None
        if '@' in self.image_key:
            self.image_key, self.image_key_arrow_suffix = self.image_key.split('@')
            self.image_key_shadow = self.image_key

        # Prepare index manager
        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(index_kwargs),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        shadow_file_fn = {}  # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        if self.image_key_shadow:
            shadow_file_fn.update({self.image_key_shadow: partial(arrow_mapper, suffix=self.image_key_arrow_suffix)})

        index_load_kwargs["shadow_file_fn"] = shadow_file_fn

        self.logger.info(f"    (T2I) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    (T2I) Using {self.index_manager}")

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
        pil_image, image_flag = self.get_raw_image(index, image_key=image_key, shadow=shadow)

        origin_size = pil_image.size  # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)
        else:
            target_size = self.training_image_size[1], self.training_image_size[0]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        pil_image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            pil_image, target_size, crop_type="random", resample=Image.Resampling.BICUBIC
        )

        image_tensor = self.pil_image_to_tensor(pil_image)

        kwargs = {
            "origin_size": origin_size,
            "target_size": target_size,
            "crop_coords_xy": (crop_left, crop_top),
        }
        return image_tensor, pil_image, kwargs, image_flag

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f"{func.__name__} | {e.__class__.__name__}: {message}")