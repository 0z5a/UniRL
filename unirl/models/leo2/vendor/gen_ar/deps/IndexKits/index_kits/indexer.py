import bisect
import hashlib
import io
import json
import platform
import random
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pyarrow as pa
from PIL import Image, ImageOps
from tqdm import tqdm
from loguru import logger

from .arrow_tools import get_columns, ColumnStatus
from .base import IndexBase
from .builder.base import IndexV2Builder, Indices
from .resolution import ResolutionGroup, DurationAndResolutionGroup
from .utils import format_ceph_base, EmptyLogger, arrow_mapper


def identity_fn(x):
    return x


class ArrowIndexV2(IndexBase):
    """
    ArrowIndexV2 is a new version of ArrowIndex.

    Parameters
    ----------
    index_file: str or pathlib.Path
        The path of index file. Either index_file or res_dict should be provided.
    corrector: callable
        A callable function to correct the path of ceph_base.
    res_dict: dict
        The index dict. Either index_file or res_dict should be provided.
    align: int
        Align the length of indices to be a multiple of align. Generally align should be the batch size * world_size.
    ceph_base: dict
        A dict mapping from ceph path to ceph shortname. If None, the ceph_base inside index_file will be used.
        If not None, the ceph_base inside index_file will be overwritten.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    ceph_base_inv: dict
        A dict mapping from ceph shortname to ceph path. If None, the ceph_base_inv inside index_file will be
        used. If not None, the ceph_base_inv inside index_file will be overwritten. ceph_base_inv has a higher
        priority than ceph_base. If both are provided, ceph_base_inv will be used.

    Examples
    --------
    >>> index_file = 'data.json'
    >>> indexObj = ArrowIndexV2(index_file)
    >>> pil_image = indexObj.get_image(0)
    >>> text = indexObj.get_attribute(0, column='text_zh')

    """
    def __init__(self, index_file=None, corrector=None, res_dict=None, align=1, ceph_base=None,
                 shadow_file_fn=None, ceph_base_inv=None, verbose=0, shadows=None, **kwargs):
        self.verbose = verbose
        if self.verbose <= 0:
            self.logger = EmptyLogger()
        else:
            self.logger = logger

        if index_file is not None:
            with open(index_file, 'r') as f:
                res_dict = json.load(f)
        elif res_dict is not None:
            pass
        else:
            raise ValueError(f'Either index_file or res_dict should be provided.')

        self.shadow_file_fn = {}
        if shadows is not None and shadow_file_fn is not None:
            raise ValueError('shadows and shadow_file_fn cannot be provided at the same time.')
        if shadows is not None:
            if not isinstance(shadows, (list, tuple)):
                raise ValueError(f'shadows should be a list of str, got {shadows}')
            self.shadow_file_fn.update({
                shadow: partial(arrow_mapper, suffix=shadow if shadow.startswith("_") else f"_{shadow}")
                for shadow in shadows
            })
        elif shadow_file_fn is not None:
            if not callable(shadow_file_fn) and not isinstance(shadow_file_fn, dict):
                raise ValueError('shadow_file_fn should be a callable function or a dict.')
            if callable(shadow_file_fn):
                self.shadow_file_fn['default'] = shadow_file_fn
            else:
                for k, v in shadow_file_fn.items():
                    if not callable(v):
                        raise ValueError(f'{k} should be a callable function.')
                    self.shadow_file_fn[k] = v

        self.index_file = index_file
        self._data = res_dict
        self.data_type = res_dict['data_type']
        self._ceph_base_inv = res_dict.get('ceph_base_inv') if ceph_base_inv is None else ceph_base_inv
        if self._ceph_base_inv is None:
            self.ceph_base = format_ceph_base(res_dict['ceph_base'] if ceph_base is None else ceph_base)
        else:
            self.ceph_base = format_ceph_base({v: k for k, v in self._ceph_base_inv.items()})
        self._arrow_files = res_dict['arrow_files']
        self.decoded_arrow_files = None
        self.cum_length = res_dict['cum_length']
        self.example_indices = res_dict.get('example_indices', [])

        self.group_length = res_dict['group_length']
        error_msg = f'Expected group_length type list, got {type(self.group_length)}.'
        if isinstance(self.group_length, dict):
            raise ValueError(f'{error_msg}')
        elif not isinstance(self.group_length, list):
            raise ValueError(error_msg)

        # Load indices
        self.use_shard_indices = kwargs.get('use_shard_indices', None)
        self.indices = res_dict['indices']
        if len(self.indices) > 0:
            self.use_shard_indices = False
        elif 'shard_cum_length' in res_dict:
            self.use_shard_indices = True

        if self.use_shard_indices:
            assert 'indices_file' in res_dict and 'shard_cum_length' in res_dict, \
                'use_shard_indices is True, but indices_file or shard_cum_length is not found in index_dict.'
            self.indices = Indices(
                root=Path(index_file).parent,
                index_files=res_dict['indices_file'],
                shard_cum_length=res_dict['shard_cum_length'],
            )
            self.indices_file = res_dict['indices_file']
        elif 'indices_file' in res_dict:
            self.indices_file = res_dict['indices_file']
            if self.indices_file != '':
                indices_file = Path(index_file).parent / self.indices_file
                if Path(indices_file).exists():
                    self.indices = np.load(indices_file)['x']
                else:
                    raise ValueError(f'This Index file contains an extra file {indices_file} which is missed.')
        else:
            self.indices_file = ''

        self.config_file = res_dict.get('config_file', '')

        if not isinstance(self.indices, (list, np.ndarray, Indices)):
            raise ValueError(f'Expected indices type list or np.ndarray or Indices, got {type(self.indices)}.')

        if align > 1:
            assert not isinstance(self.indices, Indices), f'align is not supported for shard indices.'
            self.align(align)

        if isinstance(self.indices, list):
            self.indices = np.asarray(self.indices, int)

        if len(self._arrow_files) != len(self.cum_length):
            raise ValueError(f'Length of arrow_files and cum_length does not match. {len(self._arrow_files)} != {len(self.cum_length)}')
        if len(self._arrow_files) != len(self.group_length):
            raise ValueError(f'Length of arrow_files and group_length does not match. {len(self._arrow_files)} != {len(self.group_length)}')
        if len(self.indices) == 0:
            raise ValueError(f'No indices found in index_dict.')
        if isinstance(self.indices, (np.ndarray, Indices)) and self.indices[-1] > self.cum_length[-1] - 1:
            raise ValueError(f'Indices exceed cum_length.')

        # TODO: 确保 indices 是一个递增的数组, 这里暂时不做检查, 数据量上亿的话检查起来太慢了

        self.bias = self.cum_length

        if corrector is None:
            corrector = identity_fn

        if self._ceph_base_inv is None:
            self._ceph_base_inv = {}
            for k, v in self.ceph_base.items():
                k = corrector(k)
                self._ceph_base_inv[v] = k

        self._cur_arrow_file = None
        self._cur_table_map = None
        self._cur_table = None
        self._index_bias = 0
        self.last_index = -1

        self._shadow_cur_arrow_file = {}
        self._shadow_cur_table_map = {}
        self._shadow_cur_table = {}
        self._shadow_index_bias = {}
        self.shadow_last_index = {}
        for k in self.shadow_file_fn.keys():
            self._shadow_cur_arrow_file[k] = None
            self._shadow_cur_table_map[k] = None
            self._shadow_cur_table[k] = None
            self._shadow_index_bias[k] = 0
            self.shadow_last_index[k] = -1

        # For multi-index v2
        self.probability = 1
        # For online multi-resolution
        self.resolutions: Optional[ResolutionGroup] = None  # deprecated
        self.durations_and_resolutions: Optional[DurationAndResolutionGroup] = None
        self.get_size: Optional[Callable[[int], Any]] = None

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"""
        ArrowIndexV2(
            data_type        {self.data_type}
            shard_indices    {self.use_shard_indices}
            indices_file     {self.indices_file if not self.use_shard_indices else Path(self.indices_file[0]).parent}
            config_file      {self.config_file}
            ceph_base        {self.ceph_base}
            ceph_base_inv    {self._ceph_base_inv}
            arrow_files      Count={len(self._arrow_files):,}  ({self.arrow_files[0]}, ...)
            cum_length       Count={len(self.cum_length):,}  ({self.cum_length[0]}, ...)
            group_length     Count={len(self.group_length):,}  ({self.group_length[0]}, ...)
            indices          Count={len(self.indices):,}
            example_indices  Count={len(self.example_indices):,}
        )
        """

    @staticmethod
    def from_arrow(src, ceph_base, **kwargs):
        data_dict = IndexV2Builder(ceph_base, src).encode()
        return ArrowIndexV2(res_dict=data_dict, **kwargs)

    @property
    def arrow_files(self):
        if self.decoded_arrow_files is None:
            self.decoded_arrow_files = [self.decode_arrow_file(arrow_file) for arrow_file in self._arrow_files]
        return self.decoded_arrow_files

    @property
    def ceph_base_inv(self):
        return self._ceph_base_inv

    def check_exists(self):
        for arrow_file in tqdm(self._arrow_files):
            if not Path(self.decode_arrow_file(arrow_file)).exists():
                print(arrow_file)

    def align(self, align):
        """
        Repeat the index so that the length is a multiple of batch_size * world_size.
        """
        if len(self) % align == 0:
            return

        repeat_num = align - len(self) % align
        # 计算末尾 n 个值各重复多少次
        if repeat_num >= len(self):
            repeat_n = repeat_num // len(self)
            repeat_times = [repeat_n + 1 for _ in self.indices]
            group_length_new = [ll * (repeat_n + 1) for ll in self.group_length]
            repeat_num -= repeat_n * len(self)
        else:
            repeat_times = [1 for _ in range(repeat_num)]
            group_length_new = [ll for ll in self.group_length]

        for i in range(repeat_num):
            repeat_times[-i - 1] += 1

        repeat_start_idx = len(self) - len(repeat_times)

        group_id = -1
        # 找到最后一个不为 0 的 group
        while group_length_new[group_id] == 0:
            group_id -= 1
        # 分配剩余需要重复的 index, 同时统计查看了几个 index, 如果达到了 group_length, 则切换到下一个 group
        # 同时关注 group_length 的原因是, 重复 index 时, group_length 也需要同步更新.
        group_acc = 0
        for i in range(repeat_num):
            group_length_new[group_id] += 1
            group_acc += 1
            if group_acc == self.group_length[group_id]:
                group_id -= 1
                while group_length_new[group_id] == 0:
                    group_id -= 1
                group_acc = 0

        temp = []
        for i, value in enumerate(self.indices[repeat_start_idx:]):
            temp.extend([value] * repeat_times[i])

        self.indices = np.concatenate([self.indices[:repeat_start_idx], temp])
        # 更新 group_length
        self.group_length = group_length_new

    def shuffle(self, seed=None, fast=False, use_cache=False, save_cache=False, info=None):
        assert not self.use_shard_indices, f"Shuffling shard indices is not supported."

        info_repr = f"[{self.__class__.__name__}.shuffle] " if info is None else f"[{info}] [{self.__class__.__name__}.shuffle] "

        # Check cache for shuffled indices
        if use_cache or save_cache:
            if self.index_file is None:
                raise ValueError('index_file must be provided when cache is True.')
            if seed is None:
                raise ValueError('seed must be provided when cache is True.')
            py_version = platform.python_version()
            index_file_signature = hashlib.md5(Path(self.index_file).read_bytes()).hexdigest()[:8]
            suffix = f"_py{py_version}_seed{seed}_" + (f"fast" if fast else "nofast") + f"_{index_file_signature}"
            cache_path = Path(self.index_file).parent / f'{Path(self.index_file).stem}{suffix}.index.npz'
        else:
            cache_path = None

        cache_data = None
        has_cache = False
        if use_cache:
            if cache_path.exists():
                print(f'Loading cache from {cache_path}')
                cache_data = np.load(cache_path)
                has_cache = True
            else:
                print(f"Cache not found at {cache_path}")

        if has_cache:
            self.indices = cache_data['indices']
            self.group_length = cache_data['group_length'].tolist()
        else:
            if fast:
                self.shuffle_fast(seed, info_repr=info_repr)
            else:
                indices = self.indices.tolist()

                if seed is not None:
                    state = random.getstate()
                    random.seed(seed)

                indices_group_list = []
                group_cum_len = 0
                for group_len in self.group_length:
                    indices_group = indices[group_cum_len:group_cum_len + group_len]
                    random.shuffle(indices_group)
                    indices_group_list.append((indices_group, group_len))
                    group_cum_len += group_len
                random.shuffle(indices_group_list)
                self.group_length = [x[1] for x in indices_group_list]
                self.indices = np.asarray(list(chain.from_iterable([x[0] for x in indices_group_list])))

                if seed is not None:
                    random.setstate(state)  # noqa

            if save_cache:
                cache_data = {'indices': self.indices, 'group_length': np.asarray(self.group_length)}
                print(f'Saving cache to {cache_path}')
                np.savez(cache_path, **cache_data)
                print(f'Cache saved.')
        self.logger.info(f"{info_repr}shuffle done")

    def shuffle_fast(self, seed=None, info_repr=None):
        assert not self.use_shard_indices, f"Shuffling shard indices is not supported."

        if info_repr is None:
            info_repr = ""
        # 先排序, 再 shuffle, 这样能保证每次使用相同的 seed 能获得相同的 shuffle 结果
        self.logger.info(f"{info_repr}sorted")
        self.indices.sort()
        if seed is not None:
            sampler = np.random.RandomState(seed)
            self.logger.info(f"{info_repr}sampler.shuffle {seed=}")
            sampler.shuffle(self.indices)
        else:
            np.random.shuffle(self.indices)

    def decode_arrow_file(self, arrow_file, shadow=None):
        if '@' not in arrow_file:
            full_path = arrow_file
        else:
            ceph_base_short, arrow_file = arrow_file.split('@')
            ceph_base = self._ceph_base_inv[ceph_base_short]
            full_path = str(Path(ceph_base, arrow_file.lstrip('/')))
        if shadow is not None:
            full_path = self.shadow_file_fn[shadow](full_path)
        return full_path

    def get_table(self, arrow_file, shadow=None):
        """
        Read an arrow file and return an arrow table.
        """
        if shadow is None:
            if self._cur_table is not None:
                if self._cur_arrow_file == arrow_file:
                    # This is the same arrow file. Return the cached table.
                    return self._cur_table
                else:
                    # This is a different arrow file. Clear the cache.
                    self._cur_table_map.close()
                    self._cur_table = None

            self._cur_arrow_file = arrow_file
            self._cur_table_map = pa.memory_map(f"{arrow_file}", "r")
            self._cur_table = pa.ipc.RecordBatchFileReader(self._cur_table_map).read_all()
            return self._cur_table
        else:
            if self._shadow_cur_table[shadow] is not None:
                if self._shadow_cur_arrow_file[shadow] == arrow_file:
                    return self._shadow_cur_table[shadow]
                else:
                    self._shadow_cur_table_map[shadow].close()
                    self._shadow_cur_table[shadow] = None

            self._shadow_cur_arrow_file[shadow] = arrow_file
            self._shadow_cur_table_map[shadow] = pa.memory_map(f"{arrow_file}", "r")
            self._shadow_cur_table[shadow] = pa.ipc.RecordBatchFileReader(self._shadow_cur_table_map[shadow]).read_all()
            return self._shadow_cur_table[shadow]

    def get_arrow_file_by_index(self, index, return_index_bias=False, shadow=None):
        i = bisect.bisect_right(self.cum_length, index)
        arrow_file = self.decode_arrow_file(self._arrow_files[i], shadow=shadow)

        if return_index_bias:
            if i == 0:
                index_bias = 0
            else:
                index_bias = self.cum_length[i - 1]

            return arrow_file, index_bias

        return arrow_file

    def get_arrow_file(self, ind, shadow=None, use_shuffle=None):   # noqa
        """
        Get arrow file by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        shadow: str
            The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.
        use_shuffle: bool
            A placeholder for compatibility with MultiBucket. Not used in ArrowIndexV2.

        Returns
        -------
        arrow_file: str
            The arrow file path.
        """
        index = self.indices[ind]
        return self.get_arrow_file_by_index(index, shadow=shadow)

    def load_table_by_index(self, index, shadow=None):
        if shadow is None:
            if index == self.last_index:
                return self._cur_table
            arrow_file, self._index_bias = \
                self.get_arrow_file_by_index(index, return_index_bias=True)
            self._cur_table = self.get_table(arrow_file)
            self.last_index = index
            return self._cur_table
        else:
            if index == self.shadow_last_index[shadow]:
                return self._shadow_cur_table[shadow]
            shadow_arrow_file, _shadow_index_bias = \
                self.get_arrow_file_by_index(index, return_index_bias=True, shadow=shadow)
            self._shadow_index_bias[shadow] = _shadow_index_bias
            self._shadow_cur_table[shadow] = self.get_table(shadow_arrow_file, shadow=shadow)
            self.shadow_last_index[shadow] = index
            return self._shadow_cur_table[shadow]

    def get_data_by_index(self, index, columns=None, allow_missing=False, return_meta=True, shadow=None,
                          return_table=False):
        table = self.load_table_by_index(index, shadow=shadow)
        if isinstance(columns, str):
            columns = [columns]
        if columns is None:
            columns = list(table.column_names)

        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        in_arrow_index = index - index_bias
        if return_meta:
            cur_arrow_file = self._cur_arrow_file if shadow is None else self._shadow_cur_arrow_file[shadow]
            data = {
                'index': index,
                'in_arrow_index': in_arrow_index,
                'arrow_name': cur_arrow_file,
                'shadow': shadow,
            }
        else:
            data = {}
        if return_table:
            data['table'] = table

        if allow_missing:
            for col in columns:
                if col in table.column_names:
                    data[col] = table[col][in_arrow_index].as_py()
        else:
            for col in columns:
                data[col] = table[col][in_arrow_index].as_py()
        return data

    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, shadow=None,
                 return_table=False, use_shuffle=None):     # noqa
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
        shadow: str
            The shadow name. If None, return the main data. If not None, return the shadow data.
        return_table: bool
            If True, the resulting dict will contain the arrow table.
        use_shuffle: bool
            A placeholder for compatibility with MultiBucket. Not used in ArrowIndexV2.

        Returns
        -------
        data: dict
            A dict containing the data.
        """
        index = self.indices[ind]
        return self.get_data_by_index(index, columns, allow_missing=allow_missing, return_meta=return_meta,
                                      shadow=shadow, return_table=return_table)

    def get_attribute_by_index(self, index, column, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        return table[column][index - index_bias].as_py()

    def get_attribute(self, ind, column, shadow=None, use_shuffle=None):    # noqa
        """
        Get single attribute by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            The column name.
        shadow: str
            The shadow name. If None, return the main data. If not None, return the shadow data.
        use_shuffle: bool
            A placeholder for compatibility with MultiBucket. Not used in ArrowIndexV2.

        Returns
        -------
        data: can be any type
        """
        index = self.indices[ind]
        return self.get_attribute_by_index(index, column, shadow=shadow)

    @staticmethod
    def bin_to_image(binary):
        image_bytes = io.BytesIO(binary)
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image_by_index(self, index, column=None, ret_type='pil', max_size=-1, shadow=None,
                           convert_mode="RGB", apply_exif=False):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]

        col = column or ('image' if 'image' in table.column_names else 'binary')
        temp = table[col][index - index_bias].as_py()
        image_bytes = io.BytesIO(temp)
        image_bytes.seek(0)
        try:
            # convert(RGB) has two purposes:
            # 1. Convert the image to RGB mode. Some images are in grayscale/RGBA mode, which will cause channel
            #    inconsistency in following processing.
            # 2. Convert the image to RGB mode. Some images are in P mode, which will be forced to use NEAREST resample
            #    method in resize (even if you specify LANCZOS), which will cause blurry images.
            pil_image = Image.open(image_bytes)
            if apply_exif:
                pil_image = ImageOps.exif_transpose(pil_image)
            if convert_mode is not None:
                pil_image = pil_image.convert(convert_mode)
        except Exception as e:
            print(f'get_image_by_index | Error: {e} ({self.get_arrow_file_by_index(index), index - index_bias})')
            raise e

        if max_size > 0:
            # Resize the image to max_size. max_size is the size of long edge
            w, h = pil_image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            pil_image = pil_image.resize((new_w, new_h))

        if ret_type == 'numpy':
            return np.array(pil_image)

        return pil_image

    def get_image(self, ind, column=None, ret_type='pil', max_size=-1, shadow=None, use_shuffle=None,
                  convert_mode="RGB", apply_exif=False):  # noqa
        """
        Get image by in-dataset index.

        Args:
        ind (int): The in-dataset index.
        column (str): The column name of the image. Default to 'image' or 'binary'.
        ret_type (str): The return type. Can be 'pil' or 'numpy'. Default to 'pil'.
        max_size (int): If not -1, resize the image to max_size. max_size is the size of long edge.
        shadow (str): The shadow name. If None, return the main image. If not None, return the shadow image.
        use_shuffle (bool): A placeholder for compatibility with MultiBucket. Not used in ArrowIndexV2.
        convert_mode (str): The mode to convert the image to. Default to "RGB". If None, do not convert the image.
        apply_exif (bool): If True, apply exif orientation to the image. Default to False.

        Returns
        -------
        image: PIL.Image.Image or np.ndarray
        """
        index = self.indices[ind]
        return self.get_image_by_index(index, column, ret_type, max_size, shadow=shadow, convert_mode=convert_mode,
                                       apply_exif=apply_exif)

    def get_md5_by_index(self, index, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        return table['md5'][index - index_bias].as_py()

    def get_md5(self, ind, shadow=None, use_shuffle=None):  # noqa
        index = self.indices[ind]
        return self.get_md5_by_index(index, shadow=shadow)

    def get_columns_by_index(self, index, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        return table.column_names

    def get_columns(self, ind, shadow=None, use_shuffle=None):  # noqa
        index = self.indices[ind]
        return self.get_columns_by_index(index, shadow=shadow)

    def source_distribution(self, save_path=None, shadow=None):
        sources = defaultdict(int)
        for index in tqdm(self.indices):
            source = self.get_attribute_by_index(index, 'source', shadow=shadow)
            sources[source] += 1
        # 输出 source 的分布, 从大到小排列
        sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        for k, v in sources:
            print(f'{k:20s} {v:10d}')
        if save_path is not None:
            Path(save_path).write_text(
                '\n'.join([f'{k:20s} {v:10d}' for k, v in sources]))

    def save(self, save_path, compress=True, shard_size=None):
        """
        Save the index to a json file.

        Parameters
        ----------
        save_path: str or pathlib.Path
            The path to save the index file.
        compress: bool
            If True, the index file will be compressed. Default to True.
        shard_size: int or None
            If not None, the index file will be sharded into multiple files for stream loading.
        """
        assert not self.use_shard_indices, "Shard indices are not supported for saving."
        builder = IndexV2Builder(data_type=self.data_type,
                                 ceph_base=self.ceph_base,
                                 arrow_files=self.arrow_files,
                                 cum_length=self.cum_length,
                                 group_length=self.group_length,
                                 indices=self.indices,
                                 )
        builder.build(save_path, compress=compress, shard_size=shard_size)

    def sample_batch_indices(self, n):
        if isinstance(self.indices, Indices):
            return np.random.choice(self.indices.shard_indices, n)
        else:
            return np.random.choice(self.indices, n)

    def sample_batch(self, n, columns, progress=True, shadow=None):
        if isinstance(n, int):
            indices = self.sample_batch_indices(n)
        else:
            indices = n

        if progress:
            pbar = tqdm(indices)
        else:
            pbar = indices

        batch_data = []
        for i in pbar:
            batch_data.append(self.get_data_by_index(i, columns, shadow=shadow))
        return batch_data

    @staticmethod
    def resize_and_crop(image, target_size, resample=Image.Resampling.LANCZOS, crop_type='center', crop_coords=None):
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
            Supported values include ('center', 'random', 'fixed', 'resize'). Default to 'resize'.
            - If 'center', crop the center part of the image.
            - If 'random', crop a random part of the image.
            - If 'fixed', crop the part specified by crop_coords.
            - If 'resize', crop will be disabled. The image will be resized to target_size directly,
                leaning to the change of aspect ratio.
        crop_coords: tuple
            The left top coordinates of the crop. (crop_left, crop_top)
        Returns
        -------
        image: PIL.Image.Image
            The resized and cropped image.
        crop_pos: tuple
            The position of the cropped part. (crop_left, crop_top)
        """
        tw, th = target_size
        w, h = image.size

        tr = th / tw
        r = h / w

        if crop_type == "resize":
            resize_width = tw
            resize_height = th
            crop_top = 0
            crop_left = 0
            image = image.resize((resize_width, resize_height), resample=resample)
        else:
            # maintain the aspect ratio
            if r < tr:
                resize_height = th
                resize_width = int(round(th / h * w))
            else:
                resize_width = tw
                resize_height = int(round(tw / w * h))

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

            image = image.resize((resize_width, resize_height), resample=resample)
            image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

        return image, (crop_left, crop_top)

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
        tw, th = target_size
        w, h = image.size

        tr = th / tw
        r = h / w

        # resize
        if r < tr:
            resize_width = tw
            resize_height = int(round(tw / w * h))
        else:
            resize_height = th
            resize_width = int(round(th / h * w))

        image = image.resize((resize_width, resize_height), resample=resample)

        # pad
        pad_width = tw - resize_width
        pad_height = th - resize_height
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)

        return image, (pad_left, pad_top)

    def set_resolution_buckets(self, base_size, step=None, align=1, mode=None, preset=None,
                               aspect_ratios=None, num_buckets=None, added_method="insert"):
        """ For image online bucketing """
        self.resolutions = ResolutionGroup(
            base_size=base_size, step=step, align=align, mode=mode, preset=preset,
            aspect_ratios=aspect_ratios, num_buckets=num_buckets, added_method=added_method,
        )

    def set_duration_and_resolution_buckets(self, duration_range, duration_step, base_size, step=None,
                                            align=1, mode=None, preset=None, aspect_ratios=None, num_buckets=None,
                                            added_method="insert", additional_durations=None):
        """
        For video online bucketing. We assume video bucketing and image bucketing are two different systems,
        which means they can be used in one index simultaneously.
        """
        self.durations_and_resolutions = DurationAndResolutionGroup(
            duration_range=duration_range, duration_step=duration_step, base_size=base_size, step=step,
            align=align, mode=mode, preset=preset, aspect_ratios=aspect_ratios, num_buckets=num_buckets,
            added_method=added_method, additional_durations=additional_durations,
        )

    def get_target_size(self, ind, use_shuffle=None):
        _ = use_shuffle
        if self.resolutions is None:
            if hasattr(self, "durations_and_resolutions"):
                return self.get_video_target_size(ind, use_shuffle=use_shuffle)
            raise ValueError('Please call self.set_resolution_buckets(*args, **kwargs) before using get_target_size.')
        try:
            if self.get_size is None:
                data = self.get_data(ind, columns=['height', 'width'], return_meta=False, allow_missing=True)
                if 'height' not in data or 'width' not in data:
                    # Try to get the hw from shadow arrow `_hw`
                    data = self.get_data(ind, columns=['height', 'width'], return_meta=False, shadow='hw')
                sizes = int(data['width']), int(data['height'])
            else:
                sizes = self.get_size(ind)
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            sizes = -1, -1
        return self.resolutions.get_target_size(*sizes)

    def register_get_size_fn(self, fn: Callable[['ArrowIndexV2', int], Any]):
        self.get_size = fn.__get__(self)

    def get_video_target_size(self, ind, use_shuffle=None):
        _ = use_shuffle
        assert self.durations_and_resolutions is not None, \
            'Please call self.set_duration_and_resolution_buckets(*args, **kwargs) before using get_video_target_size.'
        assert self.get_size is not None, \
            'Please call self.register_get_size_fn(fn) before using get_video_target_size.'
        try:
            sizes = self.get_size(ind)
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            sizes = -1, -1, -1
        return self.durations_and_resolutions.get_target_size(*sizes)

    def random_dindex(self, ref_ind, seed=None, intra_bucket=None):     # noqa
        """
        Get a random in dataset index. `dindex` stands for `dataset index`.
        """
        if seed is not None:
            old_state = random.getstate()
            random.seed(seed)

        new_index = int(random.random() * len(self))

        if seed is not None:
            random.setstate(old_state)  # noqa

        return new_index

    def iter_arrows(
            self,
            columns=None,
            arrow_indices=None,
            return_type="dict",
            skip_empty=True,
            strict=True,
            with_index=False,
            as_py=False,
            verbose=0,
            in_arrow_indices_fn=None,
    ):
        # Make sure arrow_indices are in the valid range
        if arrow_indices is None:
            arrow_indices = list(range(len(self._arrow_files)))
        else:
            for arrow_index in arrow_indices:
                if arrow_index < 0 or arrow_index >= len(self._arrow_files):
                    raise ValueError(f'arrow_index {arrow_index} is out of range [0, {len(self._arrow_files) - 1}]')
        # Make sure shadows are valid. We cannot check if columns are valid here,
        # because some columns may not exist in some arrow files. Invalid columns for some arrows will be raised.
        if columns is not None:
            for col in columns:
                if '@' in col:
                    col, shadow = col.split('@')
                    if shadow not in self.shadow_file_fn:
                        raise ValueError(f'shadow {shadow} is not defined in shadow_file_fn.')

        assert return_type in ('list', 'dict'), f'return_type must be "list" or "dict", but got {return_type}'

        # Calculating group cumulated length
        group_cum_lengths = np.insert(np.cumsum(self.group_length), 0, 0)

        for arrow_index in arrow_indices:
            # Get arrow file
            arrow_name = self.decode_arrow_file(self.arrow_files[arrow_index])
            # Get indices in this arrow
            in_dataset_index_start = group_cum_lengths[arrow_index]
            in_dataset_index_end = group_cum_lengths[arrow_index + 1]
            in_json_indices = self.indices[in_dataset_index_start:in_dataset_index_end]  # np.ndarray
            if skip_empty and len(in_json_indices) == 0:
                continue

            # Shift in_json_index to in_arrow_index
            in_json_index_offset = 0 if arrow_index == 0 else self.cum_length[arrow_index - 1]
            in_arrow_indices = in_json_indices - in_json_index_offset   # np.ndarray
            assert len(in_arrow_indices) == in_dataset_index_end - in_dataset_index_start, \
                f"Length mismatch: {len(in_arrow_indices)} != {in_dataset_index_end - in_dataset_index_start}"

            if len(in_arrow_indices) == 0:
                yield None, arrow_name
                continue

            if with_index:
                index_column = list(range(in_dataset_index_start, in_dataset_index_end))
            else:
                index_column = None

            if in_arrow_indices_fn is not None:
                in_arrow_indices, index_column = in_arrow_indices_fn(in_arrow_indices, index_column)
                if index_column is not None:
                    assert len(in_arrow_indices) == len(index_column), \
                        f"Length mismatch after `in_arrow_indices_fn`: {len(in_arrow_indices)} != {len(index_column)}"
                if len(in_arrow_indices) == 0:
                    yield None, arrow_name
                    continue

            if columns is None:
                columns = self.get_columns_by_index(in_json_indices[0])

            # Load data
            list_of_columns = get_columns(
                arrow_name, columns, shadow_fn_dict=self.shadow_file_fn, as_py=as_py, verbose=verbose
            )
            # Extract subset according to in_arrow_indices
            if return_type == 'dict':
                column_data = {}
            else:
                column_data = []
            for i, (values, status, msg) in enumerate(list_of_columns):
                if verbose > 0:
                    print(f"Processing columns | {arrow_name}:{columns[i]} | {status}")
                if status != ColumnStatus.SUCCESS:
                    if strict:
                        raise ValueError(f"Error in {arrow_name} {columns[i]}: {msg}")
                    else:
                        # Fill with None
                        extracted_values = [None] * len(in_arrow_indices)
                        if not as_py:
                            extracted_values = pa.array(extracted_values)
                else:
                    if as_py:
                        extracted_values = [values[j] for j in in_arrow_indices]
                    else:
                        try:
                            extracted_values = values.take(in_arrow_indices)
                        except pa.lib.ArrowInvalid as e:
                            if "offset overflow while concatenating arrays" in str(e):
                                # cast to large_string
                                if verbose > 0:
                                    print(f"    Cast column {columns[i]} to large_string due to offset overflow.")
                                values = values.cast(pa.large_string())
                                extracted_values = values.take(in_arrow_indices)
                            else:
                                raise e
                if return_type == 'dict':
                    column_data[columns[i]] = extracted_values
                else:
                    column_data.append(extracted_values)

            if with_index:
                if not as_py:
                    index_column = pa.array(index_column)
                if return_type == 'dict':
                    column_data['_in_dataset_index_'] = index_column
                else:
                    column_data.append(index_column)

            yield column_data, arrow_name
