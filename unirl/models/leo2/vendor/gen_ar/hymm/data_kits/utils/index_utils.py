import json
from pathlib import Path
from typing import Union, Optional
from functools import partial, wraps

import torch
from torch.utils.data import Dataset
# index_kits is a self-developed library for easily loading and processing large-scale datasets
# More details: https://iwiki.woa.com/p/4010805183
# Source code and doc: https://git.woa.com/CreativeAd/IndexKits
from index_kits import ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2
from index_kits import arrow_mapper

from hymm.utils.import_utils import require_version
from hymm.utils.helpers import default


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
    log_prefix: str
    index_manager: Union[ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2]
    dataset_tag: str

    def __init__(self,
                 index_file=None,
                 multireso=False,
                 batch_size=1,
                 world_size=1,
                 logger=None,
                 ):
        super().__init__()
        if logger is None:
            from loguru import logger
        self.logger = logger
        self.index_file = index_file
        if isinstance(self.index_file, str):
            self.index_file = [self.index_file]
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size

        if not hasattr(self, "index_kwargs"):
            self.index_kwargs = None
        if not hasattr(self, "task_kwargs"):
            self.task_kwargs = None

        self.index_columns: dict[str, IndexColumn] = {}
        self.index_keys: dict[str, str] = {}

        # Blacklist filter, set up by `parse_index_filter` when `index_filter_kwargs` is configured.
        # A `None` column means no filter for this dataset, which is the fast path in `hit_index_filter`.
        self.index_filter_column: Optional[IndexColumn] = None
        self.index_filter_md5s: set[str] = set()

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
            index_file=None,
            multireso=None,
            batch_size=1,
            world_size=1,
            shadow_file_fn=None,
            sample_strategy="uniform",
            probability=None,
            ceph_base=None,
            ceph_base_inv=None,
            verbose=0,
    ):
        index_file = default(index_file, self.index_file)
        multireso = default(multireso, self.multireso)
        batch_size = default(batch_size, self.batch_size)
        world_size = default(world_size, self.world_size)

        assert isinstance(index_file, (list, tuple)), (
            f"`index_file` should be a str or a list of str, got {type(index_file)}"
        )
        extra_kwargs = {}
        if ceph_base_inv is not None:
            require_version("index-kits", "0.5.3", f"{self.__class__.__name__}.load_index with ceph_base_inv")
            extra_kwargs["ceph_base_inv"] = ceph_base_inv
        if verbose > 0:
            require_version("index-kits", "0.5.14", f"{self.__class__.__name__}.load_index with verbose>0")
            extra_kwargs["verbose"] = verbose
        self.logger.info(f"Loading dataset index: {index_file}")

        if multireso:
            if len(index_file) == 1:
                index_manager = MultiResolutionBucketIndexV2(
                    index_file[0],
                    batch_size,
                    world_size,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    **extra_kwargs,
                )
            else:
                index_manager = MultiMultiResolutionBucketIndexV2(
                    index_file,
                    batch_size,
                    world_size,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy,
                    probability=probability,
                    **extra_kwargs,
                )
        else:
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(
                    index_file[0],
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    **extra_kwargs,
                )
            else:
                index_manager = MultiIndexV2(
                    index_file,
                    ceph_base=ceph_base,
                    shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy,
                    probability=probability,
                    **extra_kwargs,
                )

        return index_manager

    def setup_index_manager(self, batch_size=None):
        if self.index_kwargs is None or self.task_kwargs is None:
            raise ValueError("Please set `self.index_kwargs` and `self.task_kwargs` before calling `setup()`.")

        self.shadow_file_fn = {}
        self.parse_columns_and_register_shadow(self.index_kwargs)

        index_load_kwargs = dict(
            index_file=self.index_kwargs["index_file"],
            multireso=self.index_kwargs.get("multireso", False),
            batch_size=batch_size,
            ceph_base=self.get_ceph_base(self.index_kwargs),
            ceph_base_inv=self.get_ceph_base_inv(self.index_kwargs),
            shadow_file_fn=self.shadow_file_fn,
            sample_strategy=self.index_kwargs.get("index_strategy", "uniform"),
            probability=self.index_kwargs.get("index_probability", None),
            verbose=self.index_kwargs.get("verbose", 0),
        )
        if index_load_kwargs["multireso"] and index_load_kwargs["batch_size"] is None:
            raise ValueError("`batch_size` is required when `multireso` is True.")

        self.logger.info(f"{self.log_prefix}index kwargs: {index_load_kwargs}")
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"{self.log_prefix}Using {self.index_manager}")

    def __len__(self):
        return len(self.index_manager)

    @property
    def total_length(self):
        if len(self.index_file) == 1:
            return len(self.index_manager)
        else:
            return sum([len(bucket) for bucket in self.index_manager.buckets])

    def shuffle(self, seed, fast=True, **kwargs):
        if getattr(self, "disable_shuffle", False):
            self.logger.info(f"{self.log_prefix}Shuffle is disabled.")
            return
        self.index_manager.shuffle(seed, fast=fast, **kwargs)

    def __getitem__(self, index):
        raise NotImplementedError("Please implement `__getitem__` method in your subclass.")

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

    def _register_cols_and_keys(self, index_columns, index_keys):
        for key, value in index_columns.items():
            if key.endswith("_col"):
                if key in self.index_columns:
                    raise ValueError(f"Column {key} is already registered.")
                self.index_columns[key] = IndexColumn(value, self, self.logger)
            else:
                raise ValueError(f"Registered column `{key}` should be end with '_col'.")
        for key, value in index_keys.items():
            if key.endswith("_key"):
                if key in self.index_keys:
                    raise ValueError(f"Key {key} is already registered.")
                self.index_keys[key] = value
            else:
                raise ValueError(f"Registered key `{key}` should be end with '_key'.")

    def parse_columns_and_register_shadow(self, index_kwargs):
        self._register_cols_and_keys(
            index_kwargs.get("index_columns", {}),
            index_kwargs.get("index_keys", {})
        )
        # Must run after the call above, which registers the `md5_col` that the filter reads.
        self.parse_index_filter(index_kwargs)

    @staticmethod
    def _load_md5_file(path: Path) -> set[str]:
        # One md5 per line, no header. Blank lines are dropped so that a trailing newline cannot turn
        # into an empty entry that would match samples with an empty/null value in the filter column.
        md5s = {line.strip() for line in path.read_text().splitlines()}
        md5s.discard("")
        return md5s

    def parse_index_filter(self, index_kwargs):
        """
        Parse `index_filter_kwargs` and load the md5 blacklist. Samples whose `md5_col` value hits the
        blacklist are skipped by the loaders through `hit_index_filter`. Example config:

            index_columns:
              md5_col: fine_md5
            index_filter_kwargs:
              md5:
                - /path/to/md5s_1.txt
                - /path/to/md5s_2.txt

        Multiple files are merged and deduplicated. A misconfiguration (missing key or missing file)
        raises at startup instead of degrading to "no filtering", because silently training on data that
        was meant to be dropped is far worse than failing to launch. Datasets without the block are
        left untouched.
        """
        filter_kwargs = index_kwargs.get("index_filter_kwargs") or {}
        if not filter_kwargs:
            return

        md5_files = filter_kwargs.get("md5")
        if not md5_files:
            raise ValueError(
                f"{self.log_prefix}`index_filter_kwargs` requires `md5`, but got {filter_kwargs}."
            )
        if isinstance(md5_files, str):
            md5_files = [md5_files]
        if "md5_col" not in self.index_columns:
            raise ValueError(
                f"{self.log_prefix}`index_filter_kwargs` requires `md5_col` in `index_columns` to tell "
                f"which arrow column holds the md5 to match against."
            )

        # Reuse the registered column, so the filter and any other user of `md5_col` share one definition.
        self.index_filter_column = self.index_columns["md5_col"]
        for md5_file in md5_files:
            path = Path(md5_file)
            if not path.is_file():
                raise FileNotFoundError(f"{self.log_prefix}`index_filter_kwargs.md5` file not found: {path}")
            md5s = self._load_md5_file(path)
            self.index_filter_md5s |= md5s
            self.logger.info(f"{self.log_prefix}index filter: {len(md5s):,} md5s loaded from {path}")
        self.logger.info(
            f"{self.log_prefix}index filter: {len(self.index_filter_md5s):,} unique md5s in total will be "
            f"filtered out by column `{self.index_filter_column.key}`."
        )

    def hit_index_filter(self, index) -> bool:
        """
        Whether the sample at `index` is blacklisted by `index_filter_kwargs` and should be skipped.

        Callers are expected to invoke this before loading any media, so that a blacklisted sample costs
        one arrow attribute read and nothing else. A `KeyError` from a shard missing the filter column is
        intentionally left to propagate: it means config and data disagree, and a crash is preferable to
        keeping the data that was meant to be dropped.
        """
        if self.index_filter_column is None:
            return False

        md5 = self.index_manager.get_attribute(index, **self.index_filter_column)

        return md5 in self.index_filter_md5s


# ============================= Resample on Gray Image =============================
def resample_for_errors(get_func):
    # A decorator to resample data from the index manager when encountering gray image.
    index_manager_methods = [
        "get_arrow_file", "get_data", "get_attribute", "get_image", "get_md5", "get_columns", "get_target_size",
    ]

    @wraps(get_func)
    def wrapper(self, index):
        out = get_func(self, index)
        if not getattr(self, "enable_resample", False):
            return out

        if not out.success:
            # Add use_shuffle=False to index_manager.get_*() methods
            original_get_methods = {}
            for method_name in index_manager_methods:
                original_get_methods[method_name] = getattr(self.index_manager, method_name)
                # Replace the original methods with a version that uses use_shuffle=False
                patch_method = partial(original_get_methods[method_name], use_shuffle=False)
                setattr(self.index_manager, method_name, patch_method)

            # Retry no more than 5 times
            retry_count = 0
            while not out.success:
                # 获取新的随机索引
                new_index = self.index_manager.random_dindex(index, seed=int(index + retry_count * 10000))
                log_prefix = f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 'None'}"
                if hasattr(out, "dataset_tag"):
                    log_prefix = f"{log_prefix}, dataset_tag={out.dataset_tag}"
                log_prefix += f", index={index}, retry_count={retry_count + 1}"
                self.logger.info(f"({log_prefix}) Resample a random index {new_index:,} and retry.")
                # 使用新索引重试，并设置 use_shuffle=False
                out = get_func(self, new_index)
                out.status = "resampled"
                retry_count += 1
                if retry_count >= 20:
                    # 重试20次都失败的话, 说明数据集里有效图像不足 1/10, 直接返回.
                    break

            # Restore the original methods
            for method_name, original_method in original_get_methods.items():
                setattr(self.index_manager, method_name, original_method)

        return out

    return wrapper
