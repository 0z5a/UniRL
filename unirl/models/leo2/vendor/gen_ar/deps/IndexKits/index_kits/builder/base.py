import bisect
import json
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..arrow_tools import get_table
from ..utils import format_ceph_base
from ..version import __version__


def assert_type(data, dtype, msg=''):
    if not isinstance(data, dtype):
        raise ValueError(f'Expected {msg} type {dtype}, got {type(data)}.')


def name2short(ceph_base, name):
    for k, v in ceph_base.items():
        # 如果 name 以 <k>/ 开头, 则认为 name 是 k 的子目录
        if name.startswith(k) and name[len(k)] == '/':
            return f"{v}@{name[len(k) + 1:]}"
    return None


def ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, dict):
        data = {k: ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # Assert that all elements in data are python integer, not numpy integer.
        # Because numpy integer cannot be serialized to json.
        data = [int(x) for x in data]
    else:
        raise ValueError(f'Expected data type list, tuple, dict or np.ndarray, got {type(data)}.')
    return data


def list_to_ndarray(data):
    if isinstance(data, (list, tuple)):
        data = np.asarray(data)
    elif isinstance(data, dict):
        data = {k: list_to_ndarray(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError(f'Expected data type list, tuple, dict or np.ndarray, got {type(data)}.')
    return data


def assert_increasing(lst):
    for i in tqdm(range(1, min(len(lst), 1_000_000))):
        assert lst[i] >= lst[i - 1], (
            f'Indices should be an increasing list, but got {lst[i - 1]} <= {lst[i]}.'
        )


class IndexSaver(object):
    def __init__(self, index_dict, indices_to_save, save_path, shard_size=None):
        self.index_dict = index_dict
        self.indices_to_save = indices_to_save
        self.save_path = Path(save_path)
        self.shard_size = shard_size

    def save(self):
        if self.shard_size is None:
            self.save_single()
        else:
            self.save_shards()

    def save_single(self):
        index_dict = {k: v for k, v in self.index_dict.items()}

        if self.indices_to_save is not None:
            indices_file = self.save_path.parent / f'{self.save_path.stem}.index'
            print(f"Saving indices to the separate file: {indices_file}.npz")
            indices_dict = {k: np.asarray(v) for k, v in self.indices_to_save.items()}
            np.savez(indices_file, **indices_dict)
            index_dict['indices_file'] = indices_file.name + '.npz'

        with self.save_path.open('w') as f:
            json.dump(index_dict, f, indent=4, ensure_ascii=False)

    def save_shards(self):
        assert len(self.indices_to_save) == 1, f"Sharding is only supported for Base Index."
        key = list(self.indices_to_save.keys())[0]
        indices = self.indices_to_save[key]
        num_shards = (len(indices) + self.shard_size - 1) // self.shard_size

        indices_dir = self.save_path.parent / f'{self.save_path.stem}.index'
        indices_dir.mkdir(exist_ok=True)
        print(f"Saving sharded indices to {indices_dir}")

        shard_files = []
        cum_length = [0]
        pbar = tqdm(range(num_shards))
        for i in pbar:
            shard_indices = {key: indices[i * self.shard_size: (i + 1) * self.shard_size]}
            shard_file = indices_dir / f'{i:05d}.npz'
            np.savez(shard_file, **shard_indices)
            shard_files.append(f"{shard_file.parent.name}/{shard_file.name}")
            cum_length.append(cum_length[-1] + len(shard_indices[key]))
            pbar.set_description(f"Saving shard {i + 1}/{num_shards}, cumulative length: {cum_length[-1]:,}")
        self.index_dict['indices_file'] = shard_files
        self.index_dict['shard_cum_length'] = cum_length[1:]

        with self.save_path.open('w') as f:
            json.dump(self.index_dict, f, indent=4, ensure_ascii=False)


class IndexV2Builder(object):
    def __init__(self,
                 ceph_base,
                 arrow_files,
                 indices=None,
                 cum_length=None,
                 group_length=None,
                 data_type=None,
                 max_indices=5_000_000,
                 example_num=1000,
                 config_file=None,
                 ):
        """
        Build index v2 from an index dict.

        Args:
            ceph_base (dict or None): A dict mapping from ceph path to ceph shortname.
            arrow_files (list): A list of arrow files.
            indices (list or np.ndarray or dict): A list of indices or a dict of indices.
                If not provided, it will be specified as range(cum_length[-1]).
            cum_length (list or np.ndarray): A list of cumulative length of arrow files.
                If not provided, it will be calculated from arrow files.
            group_length (list or np.ndarray): A list of group length or a dict of group length for each arrow file.
                If not provided, it will be calculated.
            data_type (str or list): Some custom information of this index.
            max_indices (int): If the number of indices is larger than max_indices, the indices will be saved in a
                separate file. Default to 5_000_000.
            example_num (int): The number of examples to be saved in the index file. Default to 1000.
            config_file (str or list): The path of config file.

        Examples:
            >>> builder = IndexV2Builder(
            >>>     data_type='gold',
            >>>     ceph_base={
            >>>         '/apdcephfs_cq5/share_300167803': 'cq5',
            >>>     },
            >>>     arrow_files=arrow_files,
            >>>     cum_length=cum_length,
            >>>     indices=indices,
            >>> )
            >>> save_path = "base.json"
            >>> builder.build(save_path, compress=False)

        """
        self.ceph_base = format_ceph_base(ceph_base)
        self.arrow_files = arrow_files
        self.indices = indices
        self.cum_length = cum_length
        self.group_length = group_length
        self.data_type = data_type
        self.max_indices = max_indices
        self.example_num = example_num
        self.config_file = config_file

        if isinstance(arrow_files, str):
            if '*' in arrow_files or '?' in arrow_files:
                self.arrow_files = list(glob(arrow_files))
            else:
                self.arrow_files = [arrow_files]
        elif isinstance(self.arrow_files, tuple):
            self.arrow_files = list(self.arrow_files)
        if not isinstance(self.arrow_files, list):
            raise ValueError(f'Expected arrow_files to be a list, got {type(self.arrow_files)}.')

        if self.cum_length is None:
            continuous = False
            if self.indices is None:
                self.group_length = []
                continuous = True

            print(f"Calculating cum_length...")
            self.cum_length = []
            cur_cum_length = 0
            pbar = tqdm(self.arrow_files)
            for arrow_file in pbar:
                table_length = len(get_table(arrow_file))
                cur_cum_length += table_length
                self.cum_length.append(cur_cum_length)
                pbar.set_description(f"{self.cum_length[-1]:>12d}")

                if continuous:
                    self.group_length.append(table_length)

        if self.indices is None:
            self.indices = list(range(self.cum_length[-1]))
        elif self.group_length is None:
            # Assert indices is a increasing list/dict
            # We only check the first 1_000_000 indices for performance.
            print(f"Checking indices being increasing (only check the first 1_000_000 indices)...")
            if isinstance(self.indices, (list, np.ndarray)):
                if len(self.indices) > 1:
                    assert_increasing(self.indices)
            elif isinstance(self.indices, dict):
                for k, v in self.indices.items():
                    if len(v) > 1:
                        assert_increasing(v)
            else:
                raise ValueError(f'Expected indices type list or dict, got {type(self.indices)}.')

        if self.group_length is None:
            self.group_length = []

        if self.data_type is None:
            self.data_type = ['Made by IndexV2Builder']
        elif isinstance(self.data_type, str):
            self.data_type = [self.data_type]

        assert_type(self.data_type, list, 'data_type')
        assert_type(self.ceph_base, (dict, type(None)), 'ceph_base')
        assert_type(self.cum_length, (list, np.ndarray), 'cum_length')
        assert_type(self.group_length, (list, dict, np.ndarray), 'group_length')
        assert_type(self.indices, (list, dict, np.ndarray), 'indices')
        print("Convert cum_length and group_length to list...")
        self.cum_length = ndarray_to_list(self.cum_length)
        self.group_length = ndarray_to_list(self.group_length)
        if isinstance(self.indices, dict) or (isinstance(self.indices, (list, np.ndarray)) and len(self.indices) > self.max_indices):
            print("Convert indices to ndarray...")
            self.indices = list_to_ndarray(self.indices)
        elif isinstance(self.indices, (list, np.ndarray)):
            self.indices = ndarray_to_list(self.indices)
        else:
            raise ValueError(f'Expected indices type list, dict or ndarray, got {type(self.indices)}.')

        if isinstance(self.indices, dict):
            for k, v in self.indices.items():
                assert_type(v, (list, np.ndarray), f'indices[{k}]')

        if len(self.arrow_files) != len(self.cum_length):
            raise ValueError(f'Length of arrow_files and cum_length does not match. {len(self.arrow_files)} != {len(self.cum_length)}')
        if len(self.indices) == 0:
            raise ValueError(f'The indices list is empty.')
        if isinstance(self.indices, (list, np.ndarray)) and self.indices[-1] > self.cum_length[-1] - 1:
            raise ValueError(f'Indices exceed cum_length. {self.indices[-1]} > {self.cum_length[-1] - 1}')
        if len(self.group_length) > 0:
            if len(self.arrow_files) != len(self.group_length):
                raise ValueError(f'Length of arrow_files and group_length does not match. {len(self.arrow_files)} != {len(self.group_length)}')
            if sum(self.group_length) != len(self.indices):
                raise ValueError(f'Sum of group_length does not match length of indices. {sum(self.group_length)} != {len(self.indices)}')

    def encode(self):
        # 编码 arrow files
        print("Encoding arrow files...")
        arrow_files = []
        for arrow_file in tqdm(self.arrow_files):
            if self.ceph_base:
                if '@' in arrow_file:
                    shortname = arrow_file
                else:
                    shortname = name2short(self.ceph_base, arrow_file)
                if shortname is None:
                    ret = '\n'
                    raise ValueError(f"""Cannot find short name for {arrow_file}. Current ceph_base is:
                    {f'{ret}    '.join([f'{k}: {v}' for k, v in self.ceph_base.items()])}
                    """)
            else:
                shortname = arrow_file
            arrow_files.append(shortname)
        self.arrow_files = arrow_files

        # 计算 group_length
        print("Calculating group length...")
        if isinstance(self.indices, (list, np.ndarray)):
            if len(self.group_length) == 0:
                self.group_length = self.calc_group_length(self.indices, self.cum_length)
            else:
                print("Group length already calculated, skip.")
        elif isinstance(self.indices, dict):
            if not isinstance(self.group_length, dict):
                self.group_length = {}
            for k, v in self.indices.items():
                print(f"Calculating group length for {k}...")
                if k not in self.group_length or len(self.group_length[k]) == 0:
                    self.group_length[k] = self.calc_group_length(v, self.cum_length)
                else:
                    print("Group length already calculated, skip.")
        else:
            raise ValueError(f'Expected indices type list, np.ndarray or dict, got {type(self.indices)}.')

        # Get config file absolute path
        if self.config_file is not None:
            if isinstance(self.config_file, (list, tuple)):
                config_file = [str(Path(p).absolute()) for p in self.config_file]
            else:
                config_file = str(Path(self.config_file).absolute())
        else:
            config_file = ''

        # indices stats
        if isinstance(self.indices, dict):
            indices_length = {k: len(v) for k, v in self.indices.items()}
            leading_indices = []
            tail_indices = []
        else:
            indices_length = len(self.indices)
            if len(self.indices) > 8:
                leading_indices = [int(x) for x in self.indices[:4]]
                tail_indices = [int(x) for x in self.indices[-4:]]
            else:
                leading_indices = [int(x) for x in self.indices]
                tail_indices = []

        return {
            'data_type': self.data_type,
            'config_file': config_file,
            'indices_file': '',
            'ceph_base': self.ceph_base,
            'arrow_files': self.arrow_files,
            'cum_length': self.cum_length,
            'group_length': self.group_length,
            'indices': self.indices,
            'example_indices': [],
            'indices_length': indices_length,
            'leading_indices': leading_indices,
            'tail_indices': tail_indices,
            'version': __version__,
        }

    def build(self, save_path, compress=True, shard_size=None):
        return self.save(save_path, compress=compress, shard_size=shard_size)

    def save(self, save_path, compress=True, shard_size=None):
        """
        Make index v2 from an index dict.

        Parameters
        ----------
        save_path: str or pathlib.Path
            The path to save the index file.
        compress: bool
            If True, save the index file in compressed format. Default to True.
        shard_size: int or None
            If not None, the index file will be sharded into multiple files for stream loading.
        """
        index_dict = self.encode()
        # Ensure the indices either a list or a dict.

        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)

        if isinstance(index_dict['indices'], (list, np.ndarray)) and len(index_dict['indices']) > self.max_indices:
            # self.example_indices = index_dict['indices'][:self.example_num]
            indices_to_save = {'x': index_dict['indices']}
            index_dict['indices'] = []
        elif isinstance(index_dict['indices'], dict):
            indices_to_save = index_dict['indices']
            index_dict['indices'] = {}
            # num_keys = len(indices_to_save)
            # example_num_per_key = max(self.example_num // num_keys, 10)
            # index_dict['example_indices'] = {k: v[:example_num_per_key] for k, v in index_dict['indices'].items()}
            index_dict['example_indices'] = {}
        else:
            indices_to_save = None

        # 单独保存 indices
        saver = IndexSaver(
            index_dict=index_dict,
            indices_to_save=indices_to_save,
            save_path=save_path,
            shard_size=shard_size,
        )
        saver.save()

    @staticmethod
    def calc_group_length(indices, cum_length):
        group_lengths = []
        cum_ind = 0
        count = 0
        for index in tqdm(indices):
            if index < cum_length[cum_ind]:
                # index 还在当前 group 中
                count += 1
            else:
                # index 超出了当前 group, 需要切换到下一个 group
                group_lengths.append(count)
                cum_ind += 1
                # index 如果超出下一个 group, 继续切换到下一个 group
                while index >= cum_length[cum_ind]:
                    group_lengths.append(0)
                    cum_ind += 1
                count = 1
        # indices 数组消耗完毕, 最后一个包含 index 的 group 也要加上
        group_lengths.append(count)
        assert len(group_lengths) <= len(cum_length), (len(group_lengths), len(cum_length))
        # 检查如果 group 数量小于 cum_length 的数量, 则最后 n 个 group 都是空的, 需要补0
        if len(group_lengths) < len(cum_length):
            group_lengths.extend([0] * (len(cum_length) - len(group_lengths)))

        return group_lengths


class Indices(object):
    """
    A class to handle the sharded indices. It can wrap a list of sharded indices and load them on demand.
    """
    def __init__(self, root, index_files, shard_cum_length):
        self.root = Path(root)
        self.index_files = index_files
        self.shard_cum_length = shard_cum_length
        assert len(self.index_files) == len(self.shard_cum_length), \
            (f"Length of index_files and shard_cum_length does not match. "
             f"{len(self.index_files)} != {len(self.shard_cum_length)}")

        self.last_user_index = None
        self.last_inner_index = None
        self._cur_shard_file_i = None
        self._cur_shard_indices = None

    @property
    def shard_indices(self):
        return self._cur_shard_indices

    def __len__(self):
        return self.shard_cum_length[-1]

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return self.slice_indices(index.start, index.stop, index.step)
        if index < 0:
            index = self.shard_cum_length[-1] + index
        if index < 0 or index >= self.shard_cum_length[-1]:
            raise IndexError(f"Index out of range. {index} not in [0, {self.shard_cum_length[-1]})")
        if index == self.last_user_index:
            return self.last_inner_index

        # Find the shard file that contains the index
        i = bisect.bisect_right(self.shard_cum_length, index)

        # If the index is in the current shard file, use the cached shard file
        if i != self._cur_shard_file_i:
            # If the index is in a new shard file, load the new shard file
            self._cur_shard_file_i = i
            self._cur_shard_indices = np.load(self.root / self.index_files[i])['x']

        if i == 0:
            index_bias = 0
        else:
            index_bias = self.shard_cum_length[i - 1]
        rel_user_index = index - index_bias
        rel_inner_index = self._cur_shard_indices[rel_user_index]
        self.last_user_index = index
        self.last_inner_index = rel_inner_index
        return rel_inner_index

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def slice_indices(self, start, end, step) -> np.ndarray:
        assert step == 1 or step is None, f"Step must be 1 for now, got {step}"
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        assert start < end, "Start must be less than end."
        # Determine the start and end shard files
        start_i = bisect.bisect_right(self.shard_cum_length, start)
        end_i = bisect.bisect_right(self.shard_cum_length, end - 1)

        if start_i == end_i:
            if start_i == 0:
                index_bias = 0
            else:
                index_bias = self.shard_cum_length[start_i - 1]
            rel_start = start - index_bias
            rel_end = end - index_bias
            if start_i == self._cur_shard_file_i:
                return self._cur_shard_indices[rel_start: rel_end]
            else:
                shard_indices = np.load(self.root / self.index_files[start_i])['x']
                return shard_indices[rel_start: rel_end]

        else:
            indices = []
            # Split [start, end] to sections represented by cum_length
            for i in range(start_i, end_i + 1):
                if i == 0:
                    index_bias = 0
                else:
                    index_bias = self.shard_cum_length[i - 1]

                if i == start_i:
                    sli_start, sli_end = start - index_bias, self.shard_cum_length[i] - index_bias
                elif i == end_i:
                    sli_start, sli_end = self.shard_cum_length[i - 1] - index_bias, end - index_bias
                else:
                    sli_start, sli_end = self.shard_cum_length[i - 1] - index_bias, self.shard_cum_length[i] - index_bias

                # Load the shard file
                shard_indices = np.load(self.root / self.index_files[i])['x']
                indices.append(shard_indices[sli_start:sli_end])
            return np.concatenate(indices)
