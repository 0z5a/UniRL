import bisect
import hashlib
import json
import random
import platform
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union, Any
from loguru import logger

import numpy as np
from tqdm import tqdm
from PIL import Image
# 尝试导入 torch
try:
    import torch
except:
    torch = None

from .base import IndexBase
from .indexer import ArrowIndexV2
from .resolution import ResolutionGroup
from .utils import EmptyLogger, _arange


class CustomConcatDataset(object):
    dataset_list: List[ArrowIndexV2]


class Bucket(ArrowIndexV2):
    def __init__(self, height, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.height = height
        self.width = width
        self.ratio = height / width
        self.scale_dist = []

    def get_scale_by_index(self, index, size_col='hw', shadow=None):
        """
        Calculate the scale to resize the image to fit the bucket.

        Parameters
        ----------
        index: int
            An in-json index.
        size_col: str
            How to get the size of the image. 'hw' for height and width column,
            while 'image' for decoding image binary and get the PIL Image size.
        shadow: str
            The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        scale: float
        """
        if size_col == 'hw':
            w = int(self.get_attribute_by_index(index, 'width', shadow=shadow))
            h = int(self.get_attribute_by_index(index, 'height', shadow=shadow))
        else:
            w, h = self.get_image_by_index(index, shadow=shadow).size
        tw, th = self.width, self.height

        tr = th / tw
        r = h / w

        scale = th / h if r < tr else tw / w
        return scale

    @staticmethod
    def from_bucket_index(index_file, align=1, ceph_base=None, shadow_file_fn=None,
                          shadows=None, ceph_base_inv=None, verbose=0):
        with open(index_file, 'r') as f:
            res_dict = json.load(f)

        if not isinstance(res_dict['group_length'], dict):
            error_msg = f'Expected group_length type dict, but got {type(res_dict["group_length"])}'
            if isinstance(res_dict['group_length'], list):
                raise ValueError(error_msg)
            else:
                raise TypeError(error_msg)

        assert 'indices_file' in res_dict, f'indices_file not found in {index_file}'
        assert res_dict['indices_file'] != '', f'indices_file is empty in {index_file}'

        indices_file = Path(index_file).parent / res_dict['indices_file']
        assert Path(indices_file).exists(), f'indices_file {indices_file} not found'

        # Loading indices data
        indices_data = np.load(indices_file)

        # Build buckets
        buckets = []
        keys = []
        for k, v in indices_data.items():
            # Skip empty bucket
            if len(v) == 0:
                continue

            data = {
                'data_type': res_dict['data_type'],
                'ceph_base': res_dict['ceph_base'],
                'arrow_files': res_dict['arrow_files'],
                'cum_length': res_dict['cum_length'],
                'config_file': res_dict.get('config_file', ''),
            }
            if ceph_base is not None:
                data['ceph_base'] = ceph_base
            if ceph_base_inv is not None:
                data['ceph_base_inv'] = ceph_base_inv
            data['indices_file'] = ''
            data['indices'] = v
            data['group_length'] = res_dict['group_length'][k]

            height, width = map(int, k.split('x'))
            bucket = Bucket(height, width, res_dict=data, align=align,
                            shadow_file_fn=shadow_file_fn, shadows=shadows, verbose=verbose)

            buckets.append(bucket)
            keys.append(k)

        resolutions = ResolutionGroup.from_list_of_hxw(keys)
        resolutions.attr = [f'{len(bucket):,d}' for bucket in buckets]

        return buckets, resolutions


class MultiIndexV2(IndexBase):
    """
    Multi-bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Args:
    index_files (list): The index files.
    batch_size (int): The batch size of each GPU. Required when using MultiResolutionBucketIndexV2 as base index class.
    world_size (int): The number of GPUs. Required when using MultiResolutionBucketIndexV2 as base index class.
    ceph_base (dict): A dict mapping from ceph path to ceph shortname. If None, the ceph_base inside index_file
        will be used. If not None, the ceph_base inside index_file will be overwritten.
    sample_strategy (str): The sample strategy. Can be 'uniform' or 'probability'. Default to 'uniform'.
        If set to probability, a list of probability must be provided. The length of the list must be the same
        as the number of buckets. Each probability value means the sample rate of the corresponding bucket.
    probability (list): A list of probability. Only used when sample_strategy=='probability'.
    shadow_file_fn (callable or dict): A callable function to map shadow file path to a new path.
        If None, the shadow file path will not be changed.
        If a dict is provided, the keys are the shadow names to call the function, and the values are the callable
            functions to map the shadow file path to a new path.
        If a callable function is provided, the key is 'default'.
    shadows (list of str): A list of shadow names to be loaded. If None, no shadow will be loaded.
    seed (int): Only used when sample_strategy=='probability'. The seed to sample the indices.
    ceph_base_inv (dict): A dict mapping from ceph shortname to ceph path.
    """
    buckets: List[ArrowIndexV2]

    def __init__(self,
                 index_files: List[str],
                 batch_size: Optional[int] = None,
                 world_size: Optional[int] = None,
                 ceph_base: Optional[Dict[str, str]] = None,
                 sample_strategy: str = 'uniform',
                 probability: Optional[List[float]] = None,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 shadows: Optional[List[str]] = None,
                 seed: Optional[int] = None,
                 ceph_base_inv: Optional[Dict[str, str]] = None,
                 verbose=0,
                 use_shard_indices=None,
                 ):
        # 设置打印级别
        self.verbose = verbose
        if self.verbose <= 0:
            self.logger = EmptyLogger()
        else:
            self.logger = logger

        self.index_files = index_files
        self.batch_size = batch_size
        self.world_size = world_size
        self.ceph_base = ceph_base
        self.ceph_base_inv = ceph_base_inv
        self.shadow_file_fn = shadow_file_fn

        self.buckets = self.load_buckets(index_files, ceph_base,
                                         batch_size=batch_size, world_size=world_size,
                                         shadow_file_fn=shadow_file_fn, shadows=shadows,
                                         ceph_base_inv=ceph_base_inv,
                                         use_shard_indices=use_shard_indices)

        self.sample_strategy = sample_strategy
        probability = self.check_sample_strategy(sample_strategy, probability)

        # Set probability to each bucket for shuffling
        for bucket, p in zip(self.buckets, probability):
            bucket.probability = p

        self.cum_length = self.calc_cum_length()

        self.sampler = np.random.RandomState(seed)
        if torch is not None:
            self.g = torch.Generator().manual_seed(seed) if seed is not None else None
        self.logger.info(f"[{self.__class__.__name__}] Creating ind_mapper")
        if sample_strategy == 'uniform':
            self.total_length = sum([len(bucket) for bucket in self.buckets])
            self.ind_mapper = _arange(self.total_length)
        elif sample_strategy == 'probability':
            if all([bucket.probability == 1 for bucket in self.buckets]):
                self.total_length = sum([len(bucket) for bucket in self.buckets])
                self.ind_mapper = _arange(self.total_length)
            else:
                self.ind_mapper = self.sample_indices_with_probability()
                self.total_length = len(self.ind_mapper)
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")
        self.logger.info(f"[{self.__class__.__name__}] Created ind_mapper.")

    def __repr__(self):
        sep = '\n            '
        buckets_reprs = []
        for index_file, bucket in zip(self.index_files, self.buckets):
            buckets_reprs.append(f"({bucket.probability}x) {Path(index_file).absolute()}")
        res_str = f"MultiIndexV2(total_length={self.total_length}, sample_strategy={self.sample_strategy}, \n" \
                  f"index_files={sep.join(buckets_reprs)})"
        return res_str

    def load_buckets(self, index_files, ceph_base, **kwargs):
        buckets = [ArrowIndexV2(index_file, ceph_base=ceph_base, **kwargs) for index_file in index_files]
        return buckets

    def __len__(self):
        return self.total_length

    def check_sample_strategy(self, sample_strategy, probability):
        if sample_strategy == 'uniform':
            probability = [1] * len(self.buckets)
        elif sample_strategy == 'probability':
            if probability is None:
                raise ValueError(f"probability must be provided when sample_strategy is 'probability'.")
            assert isinstance(probability, (list, tuple)), \
                f"probability must be a list, but got {type(probability)}"
            assert len(self.buckets) == len(probability), \
                f"Length of index_files {len(self.buckets)} != Length of probability {len(probability)}"
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")
        return probability

    def sample_indices_with_probability(self, seed=None, info_repr=None):
        if torch is not None:
            return self.sample_indices_with_probability_torch(seed=seed, info_repr=info_repr).numpy()

        if info_repr is None:
            info_repr = ""
        sampler = np.random.RandomState(seed) if seed is not None else self.sampler
        ind_mapper_list = []
        accu = 0
        for i, bucket in enumerate(self.buckets):
            bucket_repr = f"{info_repr}[Bucket {i}, {len(bucket):,}, x{bucket.probability}] "
            self.logger.info(f"{bucket_repr}Start prepare indices")
            p = bucket.probability
            bucket_length = len(bucket)
            if p == 1:
                # Just use all indices
                self.logger.info(f"{bucket_repr}arange")
                indices = _arange(bucket_length)
                if accu > 0:
                    indices += accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                remain_num = int(round(bucket_length * (p - repeat_times)))
                self.logger.info(f"{bucket_repr}arange")
                if repeat_times == 0:
                    indices_part1 = _arange(0)
                else:
                    indices_part1 = _arange(bucket_length)
                if repeat_times > 1:
                    self.logger.info(f"{bucket_repr}repeat")
                    indices_part1 = indices_part1.repeat(repeat_times)
                if remain_num == 0:
                    # 如果是 repeat 整数倍, 则不需要 concat, 也不需要再排序了.
                    if accu == 0:
                        indices = indices_part1
                    else:
                        indices = indices_part1 + accu
                else:
                    self.logger.info(f"{bucket_repr}sampler.choice {remain_num=}")
                    indices_part2 = sampler.choice(bucket_length, remain_num, replace=False)
                    # +accu is to make sure the indices between different buckets are not overlapped
                    self.logger.info(f"{bucket_repr}concat and sort")
                    indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu
            ind_mapper_list.append(indices)
            accu += bucket_length
        self.logger.info(f"{info_repr}[Bucket All] final concat")
        ind_mapper = np.concatenate(ind_mapper_list)
        self.logger.info(f"{info_repr}[Bucket All] final concat done")
        return ind_mapper

    def sample_indices_with_probability_torch(self, seed=None, info_repr=None):
        if info_repr is None:
            info_repr = ""
        g = torch.Generator().manual_seed(seed) if seed is not None else self.g
        ind_mapper_list = []
        accu = 0
        for i, bucket in enumerate(self.buckets):
            bucket_repr = f"{info_repr}[Bucket {i}, {len(bucket):,}, x{bucket.probability}] "
            self.logger.info(f"{bucket_repr}Start prepare indices")
            p = bucket.probability
            bucket_length = len(bucket)
            if p == 1:
                # Just use all indices
                self.logger.info(f"{bucket_repr}arange")
                indices = _arange(bucket_length, as_tensor=True)
                if accu > 0:
                    indices += accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                remain_num = int(round(bucket_length * (p - repeat_times)))
                self.logger.info(f"{bucket_repr}arange")
                if repeat_times == 0:
                    indices_part1 = _arange(0, as_tensor=True)
                else:
                    indices_part1 = _arange(bucket_length, as_tensor=True)
                if repeat_times > 1:
                    self.logger.info(f"{bucket_repr}repeat")
                    indices_part1 = indices_part1.repeat_interleave(repeat_times)
                if remain_num == 0:
                    # 如果是 repeat 整数倍, 则不需要 concat, 也不需要再排序了.
                    if accu == 0:
                        indices = indices_part1
                    else:
                        indices = indices_part1 + accu
                else:
                    self.logger.info(f"{bucket_repr}sampler.choice {remain_num=}")
                    _kwargs = {"generator": g} if g is not None else {}
                    indices_part2 = torch.randperm(bucket_length, **_kwargs)[:remain_num]
                    # +accu is to make sure the indices between different buckets are not overlapped
                    self.logger.info(f"{bucket_repr}concat and sort")
                    indices = torch.sort(torch.cat([indices_part1, indices_part2]))[0] + accu
            ind_mapper_list.append(indices)
            accu += bucket_length
        self.logger.info(f"{info_repr}[Bucket All] final concat")
        ind_mapper = torch.cat(ind_mapper_list)
        self.logger.info(f"{info_repr}[Bucket All] final concat done")
        return ind_mapper

    def calc_cum_length(self):
        cum_length = []
        length = 0
        for bucket in self.buckets:
            length += len(bucket)
            cum_length.append(length)
        return cum_length

    def shuffle(self, seed=None, fast=False, use_cache=False, save_cache=False, seed_multiplier=10000, info=None):
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indices in each index
        # for i, bucket in enumerate(self.buckets):
        #     bucket.shuffle(seed + i * seed_multiplier, fast=fast, use_cache=use_cache, save_cache=save_cache)

        info_repr = f"[{self.__class__.__name__}.shuffle] " if info is None else f"[{info}] [{self.__class__.__name__}.shuffle] "

        # Shuffle ind_mapper
        if self.sample_strategy == 'uniform':
            self.logger.info(f"{info_repr}uniform arange")
            self.ind_mapper = _arange(self.total_length)
        elif self.sample_strategy == 'probability':
            if all([bucket.probability == 1 for bucket in self.buckets]):
                self.logger.info(f"{info_repr}probability arange")
                self.ind_mapper = _arange(self.total_length)
            else:
                self.logger.info(f"{info_repr}probability sample_indices_with_probability")
                self.ind_mapper = self.sample_indices_with_probability(seed=seed, info_repr=info_repr)
            # Make sure the self.total_length is always the same with self.ind_mapper
            self.total_length = len(self.ind_mapper)
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
            self.logger.info(f"{info_repr}sampler.shuffle {seed=}")
            sampler.shuffle(self.ind_mapper)
        else:
            np.random.shuffle(self.ind_mapper)
        self.logger.info(f"{info_repr}shuffle done")

    def get_arrow_file(self, ind, **kwargs):
        """
        Get arrow file by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.
            use_shuffle: bool
                If True, use ind_mapper to map the index. If False, use the original index.

        Returns
        -------
        arrow_file: str
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_arrow_file(ind - bias, **kwargs)

    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, **kwargs):
        """
        Get data by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        columns: str or list
            The columns to be returned. If None, return all columns.
        allow_missing: bool
            If True, omit missing columns. If False, raise an error if the column is missing.
        return_meta: bool
            If True, the resulting dict will contain some meta information:
            in-json index, in-arrow index, and arrow_name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.
            return_table: bool
                If True, the resulting dict will contain the arrow table.
            use_shuffle: bool
                If True, use ind_mapper to map the index. If False, use the original index.

        Returns
        -------
        data: dict
            A dict containing the data.
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_data(ind - bias, columns=columns, allow_missing=allow_missing,
                                        return_meta=return_meta, **kwargs)

    def get_attribute(self, ind, column, **kwargs):
        """
        Get single attribute by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            The column name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.
            use_shuffle: bool
                If True, use ind_mapper to map the index. If False, use the original index.

        Returns
        -------
        attribute: Any
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_attribute(ind - bias, column, **kwargs)

    def get_image(self, ind, column=None, ret_type='pil', max_size=-1, **kwargs):
        """
        Get image by in-dataset index.

        Args:
        ind (int): The in-dataset index.
        column (str): The column name of the image. Default to 'image' or 'binary'.
        ret_type (str): The return type. Can be 'pil' or 'numpy'. Default to 'pil'.
        max_size (int): If not -1, resize the image to max_size. max_size is the size of long edge.
        kwargs (dict):
            shadow (str): The shadow name. If None, return the main arrow file. If not None, return the
                shadow arrow file.
            use_shuffle (bool): If True, use ind_mapper to map the index. If False, use the original index.
            convert_mode (str): The convert mode when loading image. Default to 'RGB'.
            apply_exif (bool): Whether to apply exif orientation. Default to False.

        Returns
        -------
        image: PIL.Image.Image or np.ndarray
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_image(ind - bias, column, ret_type, max_size, **kwargs)

    def get_md5(self, ind, **kwargs):
        """ Get md5 by in-dataset index. """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_md5(ind - bias, **kwargs)

    def get_columns(self, ind, **kwargs):
        """ Get columns by in-dataset index. """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_columns(ind - bias, **kwargs)

    @staticmethod
    def resize_and_crop(image, target_size, resample=Image.Resampling.LANCZOS, crop_type='random', crop_coords=None):
        """
        Resize image without changing aspect ratio, then crop the center/random part.

        Parameters
        ----------
        image: PIL.Image.Image
            The input image to be resized and cropped.
        target_size: tuple
            The target size of the image. A tuple of (width, height).
        resample:
            The resample method. See PIL.Image.Image.resize for details. Default to Image.LANCZOS.
        crop_type: str
            'center' or 'random' or 'fixed'. If 'center', crop the center part of the image. If 'random',
            crop a random part of the image. If 'fixed', crop the part specified by crop_coords. Default to 'random'.
        crop_coords: tuple
            The left top coordinates of the crop. (crop_left, crop_top)
            
        Returns
        -------
        image: PIL.Image.Image
            The resized and cropped image.
        crop_pos: tuple
            The position of the cropped part. (crop_left, crop_top)
        """
        return ArrowIndexV2.resize_and_crop(image, target_size, resample, crop_type, crop_coords)

    @staticmethod
    def resize_and_pad(image, target_size, resample=Image.Resampling.LANCZOS,
                       pad_color=(127, 127, 127)):
        """
        Resize image by longest edge, then pad the image to target size.

        Parameters
        ----------
        image: PIL.Image.Image
            The input image to be resized and padded.
        target_size: tuple
            The target size of the image. A tuple of (width, height).
        resample:
            The resample method. See PIL.Image.Image.resize for details. Default to Image.LANCZOS.
        pad_color: tuple
            The color of the padding. Default to (127, 127, 127).

        Returns
        -------
        image: PIL.Image.Image
            The resized and padded image.
        """
        return ArrowIndexV2.resize_and_pad(image, target_size, resample, pad_color)

    @staticmethod
    def bin_to_image(binary):
        return ArrowIndexV2.bin_to_image(binary)

    @staticmethod
    def _check_bucket_args(base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method):

        def check_equal_length(target, name, default_value):
            if target is not None:
                assert isinstance(target, list) and len(base_size) == len(target), (
                    f'When {name} is specified, it should be a list or tuple with the same length as base_size.')
            else:
                target = [default_value] * len(base_size)
            return target

        if isinstance(base_size, list):
            step = check_equal_length(step, 'step', None)
            align = check_equal_length(align, 'align', 1)
            mode = check_equal_length(mode, 'mode', None)
            preset = check_equal_length(preset, 'preset', None)
            assert isinstance(aspect_ratios, list) and isinstance(aspect_ratios[0], (list, tuple)), (
                'When base_size is a list, aspect_ratios should be a list of list/tuple.'
            )
            aspect_ratios = check_equal_length(aspect_ratios, 'aspect_ratios', None)
            num_buckets = check_equal_length(num_buckets, 'num_buckets', None)
            added_method = check_equal_length(added_method, 'added_method', None)

        elif isinstance(base_size, int):
            base_size = [base_size]
            assert step is None or isinstance(step, int), 'When base_size is an int, step should be an int or None.'
            step = [step]
            assert isinstance(align, int), 'When base_size is an int, step should be an int or None.'
            align = [align]
            assert mode is None or isinstance(mode, str), 'When base_size is an int, mode should be a str or None.'
            mode = [mode]
            assert preset is None or isinstance(preset, str), 'When base_size is an int, preset should be a str or None.'
            preset = [preset]
            assert aspect_ratios is None or \
                (isinstance(aspect_ratios, (list, tuple)) and isinstance(aspect_ratios[0], str)), \
                'When base_size is an int, aspect_ratios should be a list/tuple or None.'
            aspect_ratios = [aspect_ratios]
            assert num_buckets is None or isinstance(num_buckets, int), (
                'When base_size is an int, num_buckets should be an int or None.')
            num_buckets = [num_buckets]
            added_method = [added_method]

        else:
            raise TypeError('base_size should be either int or list/tuple of int.')

        return base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method

    def set_resolution_buckets(self, base_size, step=None, align=1, mode=None, preset=None,
                               aspect_ratios=None, num_buckets=None, added_method="insert"):
        base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method = self._check_bucket_args(
            base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method
        )

        for base_size_i, step_i, align_i, mode_i, preset_i, aspect_ratios_i, num_buckets_i, added_method_i in \
                zip(base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method):
            for bucket in self.buckets:
                bucket.set_resolution_buckets(
                    base_size=base_size_i,
                    step=step_i,
                    align=align_i,
                    mode=mode_i,
                    preset=preset_i,
                    aspect_ratios=aspect_ratios_i,
                    num_buckets=num_buckets_i,
                    added_method=added_method_i,
                )

    def set_duration_and_resolution_buckets(self, duration_range, duration_step, base_size, step=None, align=1,
                                            mode=None, preset=None, aspect_ratios=None, num_buckets=None,
                                            added_method="insert", additional_durations=None):
        if isinstance(duration_range, list):
            assert isinstance(duration_step, list) and len(duration_range) == len(duration_step), (
                'When duration_range is a list, duration_step should also be a list with the same length.')
            assert isinstance(base_size, list) and len(duration_range) == len(base_size), (
                'When duration_range is a list, base_size should also be a list with the same length.')
            if additional_durations is None:
                additional_durations = [additional_durations] * len(duration_step)
            else:
                assert isinstance(additional_durations, list) and len(duration_range) == len(additional_durations), (
                    'When duration_range is a list, additional_durations should also be a list with the same length.')
        else:
            assert isinstance(duration_range, tuple), (
                'When duration_range is not a list, it should be a tuple.')
            assert isinstance(duration_step, int), (
                'When duration_range is not a list, duration_step should be an int.')
            duration_range = [duration_range]
            duration_step = [duration_step]
            additional_durations = [additional_durations]

        base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method = self._check_bucket_args(
            base_size, step, align, mode, preset, aspect_ratios, num_buckets, added_method,
        )

        for (
                duration_range_i, duration_step_i,
                base_size_i, step_i, align_i, mode_i, preset_i, aspect_ratios_i, num_buckets_i,
                additional_durations_i,
        ) in zip(
            duration_range, duration_step, base_size, step, align, mode, preset, aspect_ratios, num_buckets,
            additional_durations,
        ):
            for bucket in self.buckets:
                bucket.set_duration_and_resolution_buckets(
                    duration_range=duration_range_i,
                    duration_step=duration_step_i,
                    base_size=base_size_i,
                    step=step_i,
                    align=align_i,
                    mode=mode_i,
                    preset=preset_i,
                    aspect_ratios=aspect_ratios_i,
                    num_buckets=num_buckets_i,
                    additional_durations=additional_durations_i,
                )

    def get_target_size(self, ind, **kwargs):
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_target_size(ind - bias, **kwargs)

    def register_get_size_fn(self, fn: Callable[['ArrowIndexV2', int], Any]):
        for bucket in self.buckets:
            bucket.register_get_size_fn(fn)

    def get_video_target_size(self, ind, **kwargs):
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_video_target_size(ind - bias, **kwargs)

    def random_dindex(self, ref_ind, seed=None, intra_bucket=True):
        """
        Get a random in dataset index. `dindex` stands for dataset index.

        If intra_bucket is True, the random index will be in the same bucket as ref_index.

        Notice: The returned index is not shuffled. If one use MultiResolutionBucketIndexV2, make sure to
                set `use_shuffle=False` when calling get_*(new_index, use_shuffle=False) to get the data.
        """
        if seed is not None:
            old_state = random.getstate()
            random.seed(seed)

        if intra_bucket:
            ref_ind = self.ind_mapper[ref_ind]
            i = bisect.bisect_right(self.cum_length, ref_ind)
            bias = self.cum_length[i - 1] if i > 0 else 0
            # Get a random dataset index in the same bucket (before shuffle)
            new_index = bias + self.buckets[i].random_dindex(ref_ind - bias, intra_bucket=intra_bucket)
        else:
            new_index = int(random.random() * len(self))

        if seed is not None:
            random.setstate(old_state)

        return new_index


class MultiResolutionBucketIndexV2(MultiIndexV2):
    """
    Multi-resolution bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Parameters
    ----------
    index_file: str
        The index file of the bucket index.
    batch_size: int
        The batch size of each GPU.
    world_size: int
        The number of GPUs.
    ceph_base: dict
        A dict mapping from ceph path to ceph shortname. If None, the ceph_base inside index_file will be used.
        If not None, the ceph_base inside index_file will be overwritten.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    shadows: list of str
        A list of shadow names to be loaded. If None, no shadow will be loaded.
    ceph_base_inv: dict
        A dict mapping from ceph shortname to ceph path.
    """
    buckets: List[Bucket]

    def __init__(self,
                 index_file: str,
                 batch_size: int,
                 world_size: int,
                 ceph_base: Optional[Dict[str, str]] = None,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 shadows: Optional[List[str]] = None,
                 ceph_base_inv: Optional[Dict[str, str]] = None,
                 verbose=0,
                 ):
        self.verbose = verbose
        if self.verbose <= 0:
            self.logger = EmptyLogger()
        else:
            self.logger = logger

        self.index_file = index_file
        self.batch_size = batch_size
        self.world_size = world_size
        self.shadow_file_fn = shadow_file_fn

        align = batch_size * world_size
        if align <= 0:
            raise ValueError(f'Align size must be positive, but got {align} = {batch_size} x {world_size}')
        self.align_size = align

        # 每个桶的 index 都要跟 index 对齐, 便于取 batch
        self.buckets, self._resolutions = Bucket.from_bucket_index(index_file,
                                                                   align=align,
                                                                   ceph_base=ceph_base,
                                                                   shadow_file_fn=shadow_file_fn,
                                                                   shadows=shadows,
                                                                   ceph_base_inv=ceph_base_inv,
                                                                   verbose=verbose - 1,
                                                                   )
        # Define a bucket map for easy access
        self.buckets_map = {f'{bucket.height}x{bucket.width}': bucket for bucket in self.buckets}

        self.ceph_base = self.buckets[0].ceph_base
        self.ceph_base_inv = self.buckets[0].ceph_base_inv
        self.arrow_files = self.buckets[0].arrow_files
        self.config_file = self.buckets[0].config_file
        self._base_size = self._resolutions.base_size
        self._step = self._resolutions.step

        self.buckets = sorted(self.buckets, key=lambda x: x.ratio)
        self.cum_length = self.calc_cum_length()

        self.total_length = sum([len(bucket) for bucket in self.buckets])
        assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'

        self.ind_mapper = _arange(self.total_length)

        # For multi-index v2
        self.probability = 1

    def __repr__(self):
        res_str = f"MultiResolutionBucketIndexV2(batch_size={self.batch_size}, world_size={self.world_size}, " \
                  f"total_length={self.total_length})"
        return res_str

    @property
    def step(self):
        return self._step

    @property
    def base_size(self):
        return self._base_size

    @property
    def resolutions(self):
        return self._resolutions

    def shuffle(self, seed=None, fast=False, use_cache=False, save_cache=False, info=None):

        info_repr = f"[{self.__class__.__name__}.shuffle] " if info is None else f"[{info}] [{self.__class__.__name__}.shuffle] "

        # Check cache for shuffled indices
        if use_cache or save_cache:
            if seed is None:
                raise ValueError('seed must be provided when cache is True.')
            py_version = platform.python_version()
            index_file_signature = hashlib.md5(Path(self.index_file).read_bytes()).hexdigest()[:8]
            suffix = f"_py{py_version}_seed{seed}_bs{self.batch_size}_ws{self.world_size}_" + \
                     (f"fast" if fast else "nofast") + \
                     f"_{index_file_signature}"
            cache_path = Path(self.index_file).parent / f'{Path(self.index_file).stem}{suffix}.index.npz'
        else:
            cache_path = None

        # Shuffle indexes
        #   先排序, 再 shuffle, 这样才能被seed控制住
        self.buckets = sorted(self.buckets, key=lambda x: x.ratio)
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            self.logger.info(f"{info_repr}shuffle buckets")
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        cache_data = {}
        has_cache = False
        if use_cache:
            if cache_path.exists():
                print(f'Loading cache from {cache_path}')
                cache_data = np.load(cache_path)
                has_cache = True
            else:
                print(f"Cache not found at {cache_path}")

        if has_cache:
            desired_keys = set(list(self.buckets_map.keys()))
            key2seed = {f'{bucket.height}x{bucket.width}': seed + i for i, bucket in enumerate(self.buckets)}
            for k_seed, v in cache_data.items():
                k, *_seed = k_seed.split('_')
                cache_bucket_seed = int(_seed[0]) if len(_seed) > 0 else None
                if k not in desired_keys:
                    print(f"Redundant key {k} found in cache. This bucket is empty.")
                else:
                    desired_seed = key2seed[k]
                    if cache_bucket_seed != desired_seed:
                        print(f"Warning: seed mismatch for bucket {k}. Desired seed: {desired_seed}, "
                              f"cache seed: {cache_bucket_seed}")
                    self.buckets_map[k].indices = v
        else:
            # Shuffle indices in each bucket
            self.logger.info(f"{info_repr} shuffle each bucket")
            for i, bucket in enumerate(self.buckets):
                bucket.shuffle(seed + i, fast=fast, info=info_repr.strip())

            if save_cache:
                cache_data = {
                    f'{bucket.height}x{bucket.width}_{seed + i}': bucket.indices
                    for i, bucket in enumerate(self.buckets)
                }
                print(f'Saving cache to {cache_path}')
                np.savez_compressed(cache_path, **cache_data)
                print(f'Cache saved.')

        if torch is not None:
            # 使用 torch op 缓解申请不到内存的问题. 虽然不知道为什么比 numpy 好用
            self.logger.info(f"{info_repr} create batch_ind_mapper (batch_size={self.batch_size})")
            if seed is not None:
                _kwargs = {"generator": torch.Generator().manual_seed(seed)}
            else:
                _kwargs = {}
            self.logger.info(f"{info_repr}sampler.shuffle batch_ind_mapper {seed=}")
            if self.batch_size > 1:
                batch_ind_mapper = torch.randperm(self.total_length // self.batch_size, **_kwargs) * self.batch_size
                self.logger.info(f"{info_repr}flat batch_ind_mapper to ind_mapper")
                ind_mapper = torch.stack([batch_ind_mapper + i for i in range(self.batch_size)], dim=1).view(-1)
            else:
                ind_mapper = torch.randperm(self.total_length, **_kwargs)
            self.ind_mapper = ind_mapper.numpy()
            self.logger.info(f"{info_repr}shuffle done")
        else:
            self.logger.info(f"{info_repr} create batch_ind_mapper (batch_size={self.batch_size})")
            if self.batch_size > 1:
                batch_ind_mapper = _arange(self.total_length // self.batch_size) * self.batch_size
            else:
                batch_ind_mapper = _arange(self.total_length)
            if seed is not None:
                sampler = np.random.RandomState(seed)
                self.logger.info(f"{info_repr}sampler.shuffle batch_ind_mapper {seed=}")
                sampler.shuffle(batch_ind_mapper)
            else:
                np.random.shuffle(batch_ind_mapper)
            if self.batch_size > 1:
                self.logger.info(f"{info_repr}flat batch_ind_mapper to ind_mapper")
                ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
            else:
                ind_mapper = batch_ind_mapper
            self.ind_mapper = ind_mapper
            self.logger.info(f"{info_repr}shuffle done")

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        width, height = self.get_image(ind, **kwargs).size
        return width, height, height / width

    def get_target_size(self, ind, **kwargs):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        return self.buckets[i].width, self.buckets[i].height

    def scale_distribution(self, save_file=None):
        if save_file is not None:
            scale_dict = np.load(save_file)
            for bucket in self.buckets:
                bucket.scale_dist = scale_dict[f'{bucket.height}x{bucket.width}']
        else:
            for bucket in tqdm(self.buckets):
                for index in tqdm(bucket.indices, leave=False):
                    scale = bucket.get_scale_by_index(index)
                    bucket.scale_dist.append(scale)
            scale_dict = {f'{bucket.height}x{bucket.width}': bucket.scale_dist for bucket in self.buckets}

            if save_file is not None:
                save_file = Path(save_file)
                save_file.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(save_file, **scale_dict)

        return self


class MultiMultiResolutionBucketIndexV2(MultiIndexV2):
    buckets: List[MultiResolutionBucketIndexV2]

    def __repr__(self):
        sep = '\n            '
        buckets_reprs = []
        for index_file, bucket in zip(self.index_files, self.buckets):
            buckets_reprs.append(f"({bucket.probability}x, {int(len(bucket) * bucket.probability):,}) {Path(index_file).absolute()}")
        res_str = f"MultiMultiResolutionBucketIndexV2(batch_size={self.batch_size}, world_size={self.world_size}, " \
                  f"total_length={self.total_length}, sample_strategy={self.sample_strategy}, \n" \
                  f"index_files={sep.join(buckets_reprs)})"
        return res_str

    @property
    def step(self):
        return [b.step for b in self.buckets]

    @property
    def base_size(self):
        return [b.base_size for b in self.buckets]

    @property
    def resolutions(self):
        return [b.resolutions for b in self.buckets]

    def load_buckets(self, index_files, ceph_base, **kwargs):
        if self.batch_size is None or self.world_size is None:
            raise ValueError("`batch_size` and `world_size` must be provided when using "
                             "`MultiMultiResolutionBucketIndexV2`.")
        buckets = [
            MultiResolutionBucketIndexV2(index_file,
                                         self.batch_size,
                                         self.world_size,
                                         ceph_base=ceph_base,
                                         shadow_file_fn=kwargs.get('shadow_file_fn', None),
                                         shadows=kwargs.get('shadows', None),
                                         ceph_base_inv=kwargs.get('ceph_base_inv', None),
                                         verbose=self.verbose - 1,
                                         )
            for index_file in index_files
        ]
        return buckets

    def sample_indices_with_probability(self, return_batch_indices=False, seed=None, return_list=False, info_repr=None,):
        if torch is not None:
            return self.sample_indices_with_probability_torch(
                return_batch_indices=return_batch_indices, seed=seed, return_list=return_list, info_repr=info_repr
            ).numpy()
        if info_repr is None:
            info_repr = ""
        # Batch matters. Be careful.
        sampler = np.random.RandomState(seed) if seed is not None else self.sampler
        bs = self.batch_size
        ws = self.world_size
        align = bs * ws
        ind_mapper_list = []
        accu = 0
        for i, bucket in enumerate(self.buckets):
            bucket_repr = f"{info_repr}[Bucket {i}, {len(bucket):,}, x{bucket.probability}] "
            self.logger.info(f"{bucket_repr}Start prepare indices")
            assert len(bucket) % align == 0, \
                f'Length of bucket {len(bucket)} is not divisible by batch size {align}({bs}x{ws})'

            p = bucket.probability
            bucket_batch_length = len(bucket) // bs
            if p == 1:
                # Just use all indices
                self.logger.info(f"{bucket_repr}arange")
                batch_indices = _arange(bucket_batch_length)
                if bs > 1:
                    batch_indices *= bs
                if accu > 0:
                    batch_indices += accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                remain_batch_num = int(round(bucket_batch_length * (p - repeat_times)))
                # We must align remain_batch_num to the world size, so that the indices in each replica
                # are the times of (bs x ws).
                remain_batch_num = remain_batch_num // ws * ws
                self.logger.info(f"{bucket_repr}arange")
                if repeat_times == 0:
                    indices_part1 = _arange(0)
                else:
                    indices_part1 = _arange(bucket_batch_length)
                if repeat_times > 1:
                    self.logger.info(f"{bucket_repr}repeat")
                    indices_part1 = indices_part1.repeat(repeat_times)
                if bs > 1:
                    indices_part1 *= bs
                if remain_batch_num == 0:
                    batch_indices = indices_part1
                    if accu > 0:
                        batch_indices += accu
                else:
                    self.logger.info(f"{bucket_repr}sampler.choice {remain_batch_num=}")
                    indices_part2 = sampler.choice(bucket_batch_length, remain_batch_num, replace=False) * bs
                    # +accu is to make sure the indices between different buckets are not overlapped
                    self.logger.info(f"{bucket_repr}concat and sort")
                    batch_indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu

            if return_batch_indices:
                indices = batch_indices
            else:
                indices = np.stack([batch_indices + i for i in range(bs)], axis=1).reshape(-1)
            ind_mapper_list.append(indices)
            accu += len(bucket)

        if return_list:
            return ind_mapper_list
        self.logger.info(f"{info_repr}[Bucket All] final concat")
        ind_mapper = np.concatenate(ind_mapper_list)
        self.logger.info(f"{info_repr}[Bucket All] final concat done")
        return ind_mapper

    def sample_indices_with_probability_torch(self, return_batch_indices=False, seed=None, return_list=False, info_repr=None):
        if info_repr is None:
            info_repr = ""
        # Batch matters. Be careful.
        g = torch.Generator().manual_seed(seed) if seed is not None else self.g
        bs = self.batch_size
        ws = self.world_size
        align = bs * ws
        ind_mapper_list = []
        accu = 0
        for i, bucket in enumerate(self.buckets):
            bucket_repr = f"{info_repr}[Bucket {i}, {len(bucket):,}, x{bucket.probability}] "
            self.logger.info(f"{bucket_repr}Start prepare indices")
            assert len(bucket) % align == 0, \
                f'Length of bucket {len(bucket)} is not divisible by batch size {align}({bs}x{ws})'

            p = bucket.probability
            bucket_batch_length = len(bucket) // bs
            if p == 1:
                # Just use all indices
                self.logger.info(f"{bucket_repr}arange")
                batch_indices = _arange(bucket_batch_length, as_tensor=True)
                if bs > 1:
                    batch_indices *= bs
                if accu > 0:
                    batch_indices += accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                remain_batch_num = int(round(bucket_batch_length * (p - repeat_times)))
                # We must align remain_batch_num to the world size, so that the indices in each replica
                # are the times of (bs x ws).
                remain_batch_num = remain_batch_num // ws * ws
                self.logger.info(f"{bucket_repr}arange")
                if repeat_times == 0:
                    indices_part1 = _arange(0, as_tensor=True)
                else:
                    indices_part1 = _arange(bucket_batch_length, as_tensor=True)
                if repeat_times > 1:
                    self.logger.info(f"{bucket_repr}repeat")
                    indices_part1 = indices_part1.repeat_interleave(repeat_times)
                if bs > 1:
                    indices_part1 *= bs
                if remain_batch_num == 0:
                    batch_indices = indices_part1
                    if accu > 0:
                        batch_indices += accu
                else:
                    self.logger.info(f"{bucket_repr}sampler.choice {remain_batch_num=}")
                    _kwargs = {"generator": g} if seed is not None else {}
                    indices_part2 = torch.randperm(bucket_batch_length, **_kwargs)[:remain_batch_num] * bs
                    # +accu is to make sure the indices between different buckets are not overlapped
                    self.logger.info(f"{bucket_repr}concat and sort")
                    batch_indices = torch.sort(torch.cat((indices_part1, indices_part2), dim=0))[0] + accu

            if return_batch_indices:
                indices = batch_indices
            else:
                indices = torch.stack([batch_indices + i for i in range(bs)], dim=1).view(-1)
            ind_mapper_list.append(indices)
            accu += len(bucket)

        if return_list:
            return ind_mapper_list
        self.logger.info(f"{info_repr}[Bucket All] final concat")
        ind_mapper = torch.cat(ind_mapper_list)
        self.logger.info(f"{info_repr}[Bucket All] final concat done")
        return ind_mapper

    def shuffle(self, seed=None, fast=False, use_cache=False, save_cache=False, seed_multiplier=10000,
                shuffle_ind_mapper=True, info=None):

        info_repr = f"[{self.__class__.__name__}.shuffle] " if info is None else f"[{info}] [{self.__class__.__name__}.shuffle] "

        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indices in each index
        self.logger.info(f"{info_repr} shuffle each bucket")
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i * seed_multiplier, fast=fast, use_cache=use_cache, save_cache=save_cache, info=info_repr.strip())

        if shuffle_ind_mapper:
            self.logger.info(f"{info_repr} create batch_ind_mapper (batch_size={self.batch_size})")
            # Shuffle ind_mapper in batch level
            if self.sample_strategy == 'uniform':
                self.logger.info(f"{info_repr} Uniform arange")
                batch_ind_mapper = _arange(self.total_length // self.batch_size) * self.batch_size
            elif self.sample_strategy == 'probability':
                if all([bucket.probability == 1 for bucket in self.buckets]):
                    # If all buckets have probability 1, we can just use uniform sampling
                    self.logger.info(f"{info_repr} Probability arange")
                    batch_ind_mapper = _arange(self.total_length // self.batch_size) * self.batch_size
                else:
                    self.logger.info(f"{info_repr} Probability sample_indices_with_probability")
                    batch_ind_mapper = self.sample_indices_with_probability(
                        return_batch_indices=True, seed=seed, info_repr=info_repr)
            else:
                raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
            if seed is not None:
                sampler = np.random.RandomState(seed)
                self.logger.info(f"{info_repr}sampler.shuffle batch_ind_mapper {seed=}")
                sampler.shuffle(batch_ind_mapper)
            else:
                np.random.shuffle(batch_ind_mapper)
            if self.batch_size > 1:
                self.logger.info(f"{info_repr}flat batch_ind_mapper to ind_mapper")
                if torch is not None:
                    batch_ind_mapper = torch.from_numpy(batch_ind_mapper)
                    self.ind_mapper = torch.stack([batch_ind_mapper + i for i in range(self.batch_size)], dim=1).view(-1).numpy()
                else:
                    self.ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
            else:
                self.ind_mapper = batch_ind_mapper
            # Make sure the self.total_length is always the same with self.ind_mapper
            self.total_length = len(self.ind_mapper)
        else:
            # Actually, without shuffle ind_mapper, MultiMultiResolutionBucketIndexV2 will become a
            # ConcatMultiResolutionBucketIndexV2, which means the MultiResolutionBucketIndexV2 will be
            # consumed one by one and shuffle only happens in each MultiResolutionBucketIndexV2.
            # Note: When using this mode, one should use DistributedSamplerWithStartIndex instead of
            # BlockDistributionSampler.
            pass
        self.logger.info(f"{info_repr}shuffle done")

    def shuffle_v2(self, seed=None, fast=False, use_cache=False, save_cache=False, seed_multiplier=10000):
        """
        Shuffle v2 will consider world_size and batch_size. It is designed to used with BlockDistributedSampler.
        With shuffle v2, all the data parallel ranks will have exact the same number of batches.
        """
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i * seed_multiplier, fast=fast, use_cache=use_cache, save_cache=save_cache)

        # Shuffle ind_mapper in batch level
        if self.sample_strategy == 'uniform':
            batch_ind_mapper_list = []
            accu = 0
            for bucket in self.buckets:
                batch_ind_mapper = _arange(len(bucket) // self.batch_size) * self.batch_size + accu
                batch_ind_mapper_list.append(batch_ind_mapper)
                accu += len(bucket)
        elif self.sample_strategy == 'probability':
            batch_ind_mapper_list = self.sample_indices_with_probability(
                return_batch_indices=True, seed=seed, return_list=True)
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
        else:
            sampler = np.random

        for batch_ind_mapper_ in batch_ind_mapper_list:
            sampler.shuffle(batch_ind_mapper_)

        batch_ind_mapper_dps = [[] for _ in range(self.world_size)]
        for i, batch_ind_mapper in enumerate(batch_ind_mapper_list):
            # split into world_size parts
            batch_ind_mapper_dp = np.split(batch_ind_mapper, self.world_size)
            for j in range(self.world_size):
                batch_ind_mapper_dps[j].append(batch_ind_mapper_dp[j])
        batch_ind_mapper_dps_concat = []
        for batch_ind_mapper_dp in batch_ind_mapper_dps:
            concat = np.concatenate(batch_ind_mapper_dp)
            sampler.shuffle(concat)
            batch_ind_mapper_dps_concat.append(concat)
        batch_ind_mapper = np.concatenate(batch_ind_mapper_dps_concat)

        if self.batch_size > 1:
            self.ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
        else:
            self.ind_mapper = batch_ind_mapper
        # Make sure the self.total_length is always the same with self.ind_mapper
        self.total_length = len(self.ind_mapper)

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_ratio(ind - bias, **kwargs)

    def get_target_size(self, ind, **kwargs):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        if kwargs.get('use_shuffle', True):
            ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_target_size(ind - bias, **kwargs)
