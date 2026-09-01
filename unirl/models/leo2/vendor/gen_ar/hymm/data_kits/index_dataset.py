import os
import json
import random
from typing import Union, Optional, Dict
from loguru import logger
from functools import partial, wraps

from torch.utils.data import Dataset
import torch.nn.functional as F
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits
from index_kits import ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2
from index_kits import arrow_mapper
from PIL import Image

from hymm.utils.import_utils import require_version
try:
    from hymm.utils.crypto_utils import aes_decrypt
except (ModuleNotFoundError, ImportError):
    aes_decrypt = None
    print("Crypto module not found.")
from processors.image_kits import read_url_image, read_local_image, read_binary_image
import urllib.parse

# Try to load password from environment variable
FILE_PASSWORD = os.getenv('FILE_PASSWORD')
if FILE_PASSWORD is not None:
    print(f"FILE_PASSWORD loaded: {FILE_PASSWORD[:4]}...")


class IndexColumn(dict):
    def __init__(self, key, dataset: Optional["IndexDataset"] = None, logger=None):
        self._src = key
        self.dataset = dataset
        if dataset is not None and not hasattr(self.dataset, "shadow_file_fn"):
            dataset.shadow_file_fn = {}

        if isinstance(key, str) and '@' in key:
            self.key, self.arrow_suffix = key.split('@')
            self.shadow = self.key
            self.shadow_file_fn = {self.shadow: partial(arrow_mapper, suffix=self.arrow_suffix)}
            if dataset is not None:
                dataset.shadow_file_fn.update(self.shadow_file_fn)
        else:
            self.key = key
            self.shadow = None
            self.arrow_suffix = None
            self.shadow_file_fn = {}

        if logger is not None:
            logger.info(f"{self}")

    def keys(self):
        return 'column', 'shadow'

    def values(self):
        return self.key, self.shadow

    def items(self):
        return [('column', self.key), ('shadow', self.shadow)]

    def __iter__(self):
        return iter(self.items())

    def __getitem__(self, key):
        if key == 'column':
            return self.key
        elif key == 'shadow':
            return self.shadow
        elif key == 'arrow_suffix':
            return self.arrow_suffix
        elif key == 'shadow_file_fn':
            return self.shadow_file_fn
        else:
            raise KeyError(f"Invalid key: {key}")

    def __repr__(self):
        key, shadow, suffix = self.key, self.shadow, self.arrow_suffix
        return f"IndexColumn({key=}, {shadow=}, {suffix=})"


class IndexDataset(Dataset):
    index_manager: Union[ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2]

    def __init__(self,
                 index_file,
                 multireso=False,
                 batch_size=1,
                 world_size=1,
                 logger=None,
                 debug=False,
                 ):
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.index_file = index_file
        if isinstance(self.index_file, str):
            self.index_file = [self.index_file]
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.debug = debug

        if not hasattr(self, "index_kwargs"):
            self.index_kwargs = None
        if not hasattr(self, "task_kwargs"):
            self.task_kwargs = None

    @staticmethod
    def get_ceph_base(index_kwargs, key="ceph_base"):
        ceph_base = index_kwargs.get(key, None)
        if isinstance(ceph_base, str):
            # Assume ceph_base is a json string
            ceph_base = json.loads(ceph_base)
        return ceph_base

    @staticmethod
    def get_ceph_base_inv(index_kwargs, key="ceph_base_inv"):
        ceph_base_inv = index_kwargs.get(key, None)
        if isinstance(ceph_base_inv, str):
            # Assume ceph_base_inv is a json string
            ceph_base_inv = json.loads(ceph_base_inv)
        return ceph_base_inv

    def load_index(
            self,
            shadow_file_fn=None,
            sample_strategy="uniform",
            probability=None,
            ceph_base=None,
            ceph_base_inv=None,
            verbose=0,
        ):
        assert isinstance(self.index_file, (list, tuple)), (
            f"`index_file` should be a str or a list of str, got {type(self.index_file)}"
        )
        extra_kwargs = {}
        if ceph_base_inv is not None:
            require_version("index-kits", "0.5.3", f"{self.__class__.__name__}.load_index with ceph_base_inv")
            extra_kwargs["ceph_base_inv"] = ceph_base_inv
        if verbose > 0:
            require_version("index-kits", "0.5.14", f"{self.__class__.__name__}.load_index with verbose>0")
            extra_kwargs["verbose"] = verbose
        self.logger.info(f"Loading dataset index: {self.index_file}")

        if self.multireso:
            if len(self.index_file) == 1:
                index_manager = MultiResolutionBucketIndexV2(
                    self.index_file[0],
                    self.batch_size,
                    self.world_size,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    **extra_kwargs,
                )
            else:
                index_manager = MultiMultiResolutionBucketIndexV2(
                    self.index_file,
                    self.batch_size,
                    self.world_size,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy,
                    probability=probability,
                    **extra_kwargs,
                )
        else:
            if len(self.index_file) == 1:
                index_manager = ArrowIndexV2(
                    self.index_file[0],
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    **extra_kwargs,
                )
            else:
                index_manager = MultiIndexV2(
                    self.index_file,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy,
                    probability=probability,
                    **extra_kwargs,
                )

        return index_manager

    def setup_index_manager(self, tag):
        if self.index_kwargs is None or self.task_kwargs is None:
            raise ValueError("Please set `self.index_kwargs` and `self.task_kwargs` before calling `setup()`.")

        self.shadow_file_fn = {}
        self.parse_columns_and_register_shadow(self.index_kwargs)

        index_load_kwargs = dict(
            ceph_base=self.get_ceph_base(self.index_kwargs),
            ceph_base_inv=self.get_ceph_base_inv(self.index_kwargs),
            shadow_file_fn=self.shadow_file_fn,
            sample_strategy=self.index_kwargs.get("index_strategy", "uniform"),
            probability=self.index_kwargs.get("index_probability", None),
            verbose=self.index_kwargs.get("verbose", 0),
        )

        self.logger.info(f"    ({tag}) index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"    ({tag}) Using {self.index_manager}")

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 4096)
        return len(self.index_manager)

    @property
    def total_length(self):
        if len(self.index_file) == 1:
            return len(self.index_manager)
        else:
            return sum([len(bucket) for bucket in self.index_manager.buckets])

    def shuffle(self, seed, fast=True, **kwargs):
        self.index_manager.shuffle(seed, fast=fast, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError("Please implement `__getitem__` method in your subclass.")

    # ====================== Help Functions ======================

    @staticmethod
    def read_local_image(file_path, is_encrypted='auto', convert_mode=None, apply_exif=False):
        if is_encrypted == 'auto' and str(file_path).endswith('.enc') or is_encrypted is True:
            assert FILE_PASSWORD is not None, "`FILE_PASSWORD` must be provided when reading an encrypted file"
            file_path = aes_decrypt(file_path, FILE_PASSWORD)
        return read_local_image(file_path, convert_mode=convert_mode, apply_exif=apply_exif)

    def get_image_from_arrow(self, index, column="image", is_list=False, return_first=None, shadow=None,
                             apply_exif=False):
        if is_list:
            if return_first:
                image = read_binary_image(
                    self.index_manager.get_attribute(index, column=column, shadow=shadow)[0],
                    apply_exif=apply_exif,
                )
            else:
                image = [
                    read_binary_image(binary, apply_exif=apply_exif)
                    for binary in self.index_manager.get_attribute(index, column=column, shadow=shadow)
                ]
        else:
            image = self.index_manager.get_image(index, column=column, shadow=shadow, apply_exif=apply_exif)
        return image

    def get_image_from_url_cos(self, index, column="url_cos", cut_param=True, max_retry=3, shadow=None,
                               apply_exif=False):
        if isinstance(index, str):
            url_cos = index
        else:
            url_cos = self.index_manager.get_attribute(index, column=column, shadow=shadow)
        
        if 'prc-videoframe' in url_cos:
            cut_param = True
        else:
            cut_param = False
        
        if cut_param:
            url_cos = url_cos.split('?')[0]
        else:
            url_cos = url_cos

        # Replace external link with internal link. Remove the unexpected single quotes.
        if "'prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn'" in url_cos:
            url_cos = url_cos.replace(
                "'prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn'",
                "prc-videoframe-pub-1258344703.cos-internal.ap-guangzhou.tencentcos.cn",
            )
        url_cos = url_cos.replace("cos.ap-guangzhou.myqcloud.com", "cos-internal.ap-guangzhou.tencentcos.cn")
        url_cos = urllib.parse.unquote(url_cos)

        image = None
        for _ in range(max_retry):
            try:
                image = read_url_image(url_cos, apply_exif=apply_exif)
                break
            except Exception as e:
                logger.warning(f"[PID={os.getpid()}] {e}. Failed to get image ({index=}) from {url_cos}, retrying...")
        if image is None:
            # If failed here, the exception will be catched by get_raw_image and return a gray image
            raise ValueError(f"Failed to get image from {url_cos}")
        return image

    @staticmethod
    def get_from_url_cos(url_cos, content):
        parsed_url = urllib.parse.urlparse(url_cos)
        if content == "bucket":
            ret = parsed_url.netloc.split('.')[0]
        elif content == "key":
            ret = parsed_url.path.lstrip('/')
        elif content == "bucket,key":
            ret = (parsed_url.netloc.split('.')[0], parsed_url.path.lstrip('/'))
        else:
            raise ValueError(f"Invalid content: {content}")
        return ret

    @staticmethod
    def tensor_resize_and_crop(tensor, target_width, target_height, mode, align_corners=None,
                               crop_type='random', crop_coords=None):
        assert tensor.ndim == 3, "tensor should be 3D, but got {tensor.ndim}D"
        tw, th = target_width, target_height
        _, h, w = tensor.shape

        tr = th / tw
        r = h / w

        # resize
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        tensor = F.interpolate(tensor[None], size=(resize_height, resize_width),
                               mode=mode, align_corners=align_corners)[0]

        if crop_type == 'center':
            crop_top = int(round((resize_height - th) / 2.0))
            crop_left = int(round((resize_width - tw) / 2.0))
        elif crop_type == 'random':
            crop_top = random.randint(0, resize_height - th)
            crop_left = random.randint(0, resize_width - tw)
        elif crop_type == 'fixed':
            assert crop_coords is not None, 'crop_coords should be provided when crop_type is fixed.'
            crop_left, crop_top = crop_coords
        else:
            raise ValueError(f'crop_type must be center, random or fixed, but got {crop_type}')

        tensor = tensor[:, crop_top:crop_top + th, crop_left:crop_left + tw]
        return tensor, (crop_left, crop_top)

    # ====================== Extra Helper Functions ======================

    def strip_leading_tag(self, src_dict, required=True, tag=None):
        res_dict = {}
        if tag is None:
            leading_str = f"{self.dataset_tag}_"
        else:
            leading_str = f"{tag}_"
        for key, value in src_dict.items():
            if key.startswith(leading_str):
                key = key[len(leading_str):]
            elif required:
                raise ValueError(f"Key {key} does not start with {leading_str}")
            res_dict[key] = value
        return res_dict

    def _register_caption_cols(self, index_kwargs):
        # 自动注册 caption 的 columns
        for key, value in index_kwargs.items():
            if "col" in key and "image_caption" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    raise ValueError(f"Key {key} is already an attribute of the class.")
                setattr(self, key, IndexColumn(value, self, self.logger))

    def _register_extra_cols(self, index_kwargs):
        # 自动注册额外的 columns
        for key, value in index_kwargs.items():
            if "col" in key and "extra" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    raise ValueError(f"Key {key} is already an attribute of the class.")
                setattr(self, key, IndexColumn(value, self, self.logger))
    
    def _register_instruction_keys(self, index_kwargs):
        # 自动注册额外的 columns
        for key, value in index_kwargs.items():
            if "key" in key and "image_instruction" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    raise ValueError(f"Key {key} is already an attribute of the class.")
                setattr(self, key, IndexColumn(value, None, self.logger))
    
    def _register_extra_keys(self, index_kwargs):
        # 自动注册额外的 columns
        for key, value in index_kwargs.items():
            if "key" in key and "extra" in key:
                # 确保 key 不是 self 的属性
                if hasattr(self, key):
                    raise ValueError(f"Key {key} is already an attribute of the class.")
                setattr(self, key, IndexColumn(value, None, self.logger))

    def register_attributes(self, index_kwargs):
        self._register_caption_cols(index_kwargs)
        self._register_extra_cols(index_kwargs)
        self._register_instruction_keys(index_kwargs)
        self._register_extra_keys(index_kwargs)

    def parse_columns_and_register_shadow(self, index_kwargs):
        self.register_attributes(index_kwargs)


# ============================= Resample on Gray Image =============================
def resample_on_gray(get_func):
    # A decorator to resample data from the index manager when encountering gray image.
    index_manager_methods = [
        "get_arrow_file", "get_data", "get_attribute", "get_image", "get_md5", "get_columns", "get_target_size",
    ]

    @wraps(get_func)
    def wrapper(self, index):
        out = get_func(self, index)
        if not getattr(self, "enable_resample", False):
            return out

        if out.image_flag == "gray":
            # Add use_shuffle=False to index_manager.get_*() methods
            original_get_methods = {}
            for method_name in index_manager_methods:
                original_get_methods[method_name] = getattr(self.index_manager, method_name)
                # Replace the original methods with a version that uses use_shuffle=False
                patch_method = partial(original_get_methods[method_name], use_shuffle=False)
                setattr(self.index_manager, method_name, patch_method)

            # Retry no more than 5 times
            retry_count = 0
            while out.image_flag == "gray":
                # 获取新的随机索引
                new_index = self.index_manager.random_dindex(index, seed=int(index + retry_count * 10000))
                self.logger.info(f"(index={index}) Resample a random index {new_index} and retry.")
                # 使用新索引重试，并设置 use_shuffle=False
                out = get_func(self, new_index)
                retry_count += 1
                if retry_count >= 5:
                    # 重试5次都失败的话, 说明数据集里有效图像不足 1/5, 直接返回.
                    break

            # Restore the original methods
            for method_name, original_method in original_get_methods.items():
                setattr(self.index_manager, method_name, original_method)

        return out

    return wrapper
