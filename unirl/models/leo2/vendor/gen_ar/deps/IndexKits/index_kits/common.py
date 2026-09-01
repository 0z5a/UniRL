import json
from functools import partial
from glob import glob
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .arrow_tools import get_table
from .builder.base import IndexV2Builder
from .indexer import ArrowIndexV2
from .resolution import ResolutionGroup
from .utils import LoadIndexError


def load_index(
        src: Union[str, List[str]],
        ceph_base: Optional[Dict[str, str]] = None,
        multireso: bool = False,
        batch_size: int = 1,
        world_size: int = 1,
        sample_strategy: str = 'uniform',
        probability: Optional[List[float]] = None,
        shadow_file_fn: Optional[Dict[str, Callable]] = None,
        seed: Optional[int] = None,
        ceph_base_inv: Optional[Dict[str, str]] = None,
        verbose=0,
        use_shard_indices=None,
):
    if isinstance(src, str):
        src = [src]
    if isinstance(ceph_base, str):
        ceph_base = json.loads(ceph_base)
    if isinstance(ceph_base_inv, str):
        ceph_base_inv = json.loads(ceph_base_inv)
    if src[0].endswith('.arrow'):
        if multireso:
            raise ValueError('Arrow file does not support multiresolution. Please make base index V2 first and then'
                             'build multiresolution index.')
        if ceph_base is None:
            ceph_base = {}
        idx = ArrowIndexV2.from_arrow(src, ceph_base, use_shard_indices=use_shard_indices)
    elif src[0].endswith('.json'):
        try:
            if multireso:
                from .bucket import MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2

                assert not use_shard_indices, 'Use shard indices is not supported for multi-resolution index.'

                if len(src) == 1:
                    idx = MultiResolutionBucketIndexV2(src[0], ceph_base=ceph_base, batch_size=batch_size,
                                                       world_size=world_size,
                                                       shadow_file_fn=shadow_file_fn, ceph_base_inv=ceph_base_inv,
                                                       verbose=verbose,
                                                       )
                else:
                    idx = MultiMultiResolutionBucketIndexV2(src, ceph_base=ceph_base, batch_size=batch_size,
                                                            world_size=world_size,
                                                            sample_strategy=sample_strategy, probability=probability,
                                                            shadow_file_fn=shadow_file_fn, seed=seed,
                                                            ceph_base_inv=ceph_base_inv,
                                                            verbose=verbose,
                                                            )
            else:
                if len(src) == 1:
                    idx = ArrowIndexV2(src[0], ceph_base=ceph_base,
                                       shadow_file_fn=shadow_file_fn, ceph_base_inv=ceph_base_inv,
                                       verbose=verbose, use_shard_indices=use_shard_indices,
                                       )
                else:
                    from .bucket import MultiIndexV2
                    idx = MultiIndexV2(src, ceph_base=ceph_base,
                                       sample_strategy=sample_strategy, probability=probability,
                                       shadow_file_fn=shadow_file_fn, seed=seed, ceph_base_inv=ceph_base_inv,
                                       verbose=verbose, use_shard_indices=use_shard_indices,
                                       )
        except ValueError as e:
            if "Expected group_length type list" in str(e):
                raise LoadIndexError("The index file is a multi-resolution index file. Please set `multireso=True`.")
            if "Expected group_length type dict" in str(e):
                raise LoadIndexError("The index file is a base index file. Please set `multireso=False`.")
            raise e
        except Exception as e:
            raise e
    else:
        raise ValueError(f'Unknown file type: {src[0]}')
    return idx


def get_attribute(data, attr_list):
    ret_data = {}
    for attr in attr_list:
        ret_data[attr] = data.get(attr, None)
        if ret_data[attr] is None:
            raise ValueError(f'Missing key `{attr}` in data. {data.keys()}')
    return ret_data


def get_optional_attribute(data, attr_list):
    ret_data = {}
    for attr in attr_list:
        ret_data[attr] = data.get(attr, None)
    return ret_data


def detect_index_type(data):
    if isinstance(data['group_length'], dict):
        return 'multireso'
    else:
        return 'base'


def decode_arrow_file(arrow_file, ceph_base_inv=None):
    if '@' not in arrow_file:
        return arrow_file

    ceph_base_short, arrow_file_path = arrow_file.split('@')
    if ceph_base_short in ceph_base_inv:
        ceph_base = ceph_base_inv[ceph_base_short]
    else:
        ceph_base = {v: k for k, v in ceph_base_inv.items()}
        raise KeyError(f"When decoding `{arrow_file}`, short name `{ceph_base_short}` is missing in "
                       f"ceph_base dict: {ceph_base}")
    full_path = str(Path(ceph_base, arrow_file_path.lstrip('/')))
    return full_path


def show_arrow_info(src, show_all=False, rows=5, start=0, full_width=False):
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', max(rows, 25))
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', None if full_width else 200)

    def table_to_dataframe(tb, rows, cols=None):
        data = []
        if cols is None:
            cols = tb.column_names
        max_rows = len(tb)
        for i in rows:
            if i >= max_rows:
                break
            item = {col: tb[col][i].as_py() for col in cols}
            data.append(item)
        return pd.DataFrame(data)

    table = get_table(src)
    print(f"File: {Path(src).absolute()}")
    print(f"Total rows: {len(table)}")
    if not show_all:
        bytes_columns = []
        non_bytes_columns = []
        for col in table.column_names:
            value = table[col][0].as_py()
            if value is not None and isinstance(value, bytes):
                bytes_columns.append(col)
            else:
                non_bytes_columns.append(col)
        bytes_info = ''
        if bytes_columns:
            bytes_info = f"(Omit bytes columns: {', '.join(bytes_columns)})"
        print(f"Arrow file information{bytes_info}:")
        if rows == 1:
            print(table_to_dataframe(table, [start], non_bytes_columns))
        else:
            print(table_to_dataframe(table, range(start, start + rows), non_bytes_columns))
    else:
        print("Arrow file information:")
        if rows == 1:
            print(table_to_dataframe(table, [start]))
        else:
            print(table_to_dataframe(table, range(start, start + rows)))


def show_index_info(src, only_arrow_files=False, depth=1, show_all=False, rows=5, start=0, full_width=False):
    """
    Show arrow or base/multireso index information.
    If the file suffix is `.arrow`, it will show the arrow file information.
    If the file suffix is `.json`, it will show the index file information.

    Parameters
    ----------
    src : str
        The path to the index file.
    only_arrow_files : bool
        If True, only show the arrow files. Default is False.
    depth : int
        The depth of the arrow file path. Default is 1.
    show_all : bool
        Only for arrow files. If True, show all columns of the arrow file. Default is False.
    rows : int
        Only for arrow files. The number of rows to show. Default is 5.
    start : int
        Only for arrow files. The start index of the rows to show. Default is 0.
    full_width : bool
        If True, show the full width of the columns. Default is False.
    """
    if not Path(src).exists():
        raise ValueError(f'{src} does not exist.')
    if Path(src).suffix == '.arrow':
        show_arrow_info(src, show_all, rows, start, full_width)
        return None

    print(f"Loading index file {src} ...")
    with open(src, 'r') as f:
        src_data = json.load(f)
    print(f"Loaded.")
    data = get_attribute(src_data, ['data_type', 'indices_file', 'ceph_base', 'arrow_files', 'cum_length',
                                    'group_length', 'indices', 'example_indices'])
    for key in ['indices_length', 'leading_indices', 'tail_indices']:
        if key in src_data:
            data[key] = src_data[key]
    opt_data = get_optional_attribute(src_data, ['config_file', 'shard_cum_length'])

    _ceph_base_inv = {v: k for k, v in data['ceph_base'].items()}
    _decode = partial(decode_arrow_file, ceph_base_inv=_ceph_base_inv)

    # Format arrow_files examples
    arrow_files = data['arrow_files']
    if only_arrow_files:
        existed = set()
        arrow_files_output_list = []
        for arrow_file in arrow_files:
            if depth == 0:
                if arrow_file not in existed:
                    arrow_files_output_list.append(_decode(arrow_file))
                    existed.add(arrow_file)
            elif depth > 0:
                parts = Path(arrow_file).parts
                if depth >= len(parts):
                    continue
                else:
                    arrow_file_part = '/'.join(parts[:-depth])
                    if arrow_file_part not in existed:
                        arrow_files_output_list.append(_decode(arrow_file_part))
                        existed.add(arrow_file_part)
            else:
                raise ValueError(f'Depth {depth} has exceeded the limit of arrow file {arrow_file}.')
        arrow_files_repr = '\n'.join(arrow_files_output_list)
        print(arrow_files_repr)
        return None

    return_space = '\n' + ' ' * 25

    if len(arrow_files) <= 4:
        arrow_files_repr = return_space.join([_decode(arrow_file) for arrow_file in arrow_files])
    else:
        arrow_files_repr = return_space.join([_decode(_) for _ in arrow_files[:2]] + ['...']
                                             + [_decode(_) for _ in arrow_files[-2:]])
    arrow_files_length = len(arrow_files)

    # Format data_type
    data_type = data['data_type']
    if isinstance(data_type, str):
        data_type = [data_type]
    data_type_common = []
    src_files = []
    found_src_files = False
    for data_type_item in data_type:
        if not found_src_files and data_type_item.strip() != 'src_files=':
            data_type_common.append(data_type_item.strip())
            continue
        found_src_files = True
        if data_type_item.endswith('.json'):
            src_files.append(data_type_item.strip())
        else:
            data_type_common.append(data_type_item.strip())
    data_type_part2_with_ids = []
    max_id_len = len(str(len(src_files)))
    for sid, data_type_item in enumerate(src_files, start=1):
        data_type_part2_with_ids.append(f'{str(sid).rjust(max_id_len)}. {data_type_item}')
    data_type = data_type_common + data_type_part2_with_ids
    data_repr = return_space.join(data_type)

    # Format ceph_base
    ceph_base = data['ceph_base']
    if len(ceph_base) > 0:
        ceph_base_keys_max_length = max([len(k) for k in ceph_base.keys()])
        ceph_base_list = [f'{k:<{ceph_base_keys_max_length}}: {v}' for k, v in ceph_base.items()]
    else:
        ceph_base_list = []
    ceph_base_repr = return_space.join(ceph_base_list)

    # Format cum_length examples
    cum_length = data['cum_length']
    if len(cum_length) <= 8:
        cum_length_repr = ', '.join([str(i) for i in cum_length])
    else:
        cum_length_repr = ', '.join([str(i) for i in cum_length[:4]] + ['...'] + [str(i) for i in cum_length[-4:]])
    cum_length_length = len(cum_length)

    if detect_index_type(data) == 'base':
        # Format group_length examples
        group_length = data['group_length']
        if len(group_length) <= 8:
            group_length_repr = ', '.join([str(i) for i in group_length])
        else:
            group_length_repr = ', '.join([str(i) for i in group_length[:4]] + ['...'] + [str(i) for i in group_length[-4:]])
        group_length_length = len(group_length)

        # Format indices file examples
        if isinstance(data['indices_file'], str):
            indices_file_repr = f"[{data['indices_file']}]"
        else:
            if len(data['indices_file']) > 4:
                indices_file_repr = '[' + ', '.join([str(i) for i in data['indices_file'][:2]] + ['...'] + [str(i) for i in data['indices_file'][-2:]]) + ']'
            else:
                indices_file_repr = '[' + ', '.join([str(i) for i in data['indices_file']]) + ']'

        # Format indices examples
        if 'indices_length' in data:
            indices_length = data['indices_length']
            if len(data.get('tail_indices', [])) == 0:
                indices_repr = ', '.join([str(i) for i in data.get('leading_indices', [])])
            else:
                indices_repr = ', '.join([str(i) for i in data.get('leading_indices', [])] + ['...'] + [str(i) for i in data.get('tail_indices', [])])
        else:
            indices = data['indices']
            if len(indices) == 0 and data['indices_file'] != '':
                indices_file = Path(src).parent / data['indices_file']
                if Path(indices_file).exists():
                    print(f"Loading indices from {indices_file} ...")
                    indices = np.load(indices_file)['x']
                    print(f"Loaded.")
                else:
                    raise ValueError(f'This Index file contains an extra file {indices_file} which is missed.')
            if len(indices) <= 8:
                indices_repr = ', '.join([str(i) for i in indices])
            else:
                indices_repr = ', '.join([str(i) for i in indices[:4]] + ['...'] + [str(i) for i in indices[-4:]])

            # Calculate indices total length
            indices_length = len(indices)

        print_str = f"""File: {Path(src).absolute()}
        
ArrowIndexV2(
          \033[4mdata_type:\033[0m {data_repr}"""

        # Process optional data
        if opt_data['config_file'] is not None:
            print_str += f"""
        \033[4mconfig_file:\033[0m {opt_data['config_file']}"""

        # Add common data
        print_str += f"""
       \033[4mindices_file:\033[0m {indices_file_repr}"""

        if opt_data['shard_cum_length'] is not None:
            print_str += f"""
         \033[4mshard_size:\033[0m {opt_data['shard_cum_length'][0]:,}"""

        print_str += f"""
          \033[4mceph_base:\033[0m {ceph_base_repr}
        \033[4marrow_files: Count = {arrow_files_length:,}\033[0m
                     Examples:
                         {arrow_files_repr}
         \033[4mcum_length: Count = {cum_length_length:,}\033[0m
                     Examples: {cum_length_repr}
       \033[4mgroup_length: Count = {group_length_length:,}\033[0m
                     Examples: {group_length_repr}
            \033[4mindices: Count = {indices_length:,}\033[0m
                     Examples: {indices_repr}"""

    else:
        base_size = None
        step = None
        for item in data_type:
            if 'base_size=' in item:
                base_size = item[len('base_size='):]
            if 'reso_step=' in item:
                step = item[len('reso_step='):]
        if base_size and base_size != 'None':
            base_size = int(base_size)
        if step and step != 'None':
            step = int(step)

        indices_lengths = []
        keys = []
        if 'indices_length' in data:
            for key, length in data['indices_length'].items():
                if length > 0:
                    indices_lengths.append(length)
                    keys.append(key)
        else:
            indices_file = Path(src).parent / data['indices_file']
            assert Path(indices_file).exists(), f'indices_file {indices_file} not found'
            print(f"Loading indices from {indices_file} ...")
            indices_data = np.load(indices_file)
            print(f"Loaded.")
            for key, indices in indices_data.items():
                length = len(indices)
                if length > 0:
                    indices_lengths.append(length)
                    keys.append(key)
        indices_length = sum(indices_lengths)
        resolutions = ResolutionGroup.from_list_of_hxw(keys, base_size=base_size, step=step)
        resolutions.attr = [f'{length:>,d}' for length in indices_lengths]
        resolutions.prefix_space = 25

        print_str = f"""File: {Path(src).absolute()}

MultiResolutionBucketIndexV2(
          \033[4mdata_type:\033[0m {data_repr}"""

        # Process optional data
        if opt_data['config_file'] is not None:
            print_str += f"""
        \033[4mconfig_file:\033[0m {opt_data['config_file']}"""

        # Process config files of base index files
        config_files = []
        for src_file in src_files:
            src_file = Path(src_file)
            if src_file.exists():
                with src_file.open() as f:
                    base_data = json.load(f)
                if 'config_file' in base_data:
                    config_files.append(base_data['config_file'])
                else:
                    config_files.append('Unknown')
            else:
                config_files.append('Missing the src file')
        if config_files:
            config_file_str = return_space.join([f'{str(sid).rjust(max_id_len)}. {config_file}'
                                                 for sid, config_file in enumerate(config_files, start=1)])
            print_str += f"""
  \033[4mbase config files:\033[0m {config_file_str}"""

        # Add common data
        print_str += f"""
       \033[4mindices_file:\033[0m {data['indices_file']}
          \033[4mceph_base:\033[0m {ceph_base_repr}
        \033[4marrow_files: Count = {arrow_files_length:,}\033[0m
                     Examples: {arrow_files_repr}
         \033[4mcum_length: Count = {cum_length_length:,}\033[0m
                     Examples: {cum_length_repr}
            \033[4mindices: Count = {indices_length:,}\033[0m
            \033[4mbuckets: Count = {len(keys)}\033[0m
                     {resolutions}"""

    print(print_str + '\n)\n')


def shuffle_index(srcs, multireso, seed, fast=False, batch_size=1, world_size=1, seed_multiplier=10000):
    if multireso:
        kwargs = dict(multireso=True, batch_size=batch_size, world_size=world_size)
    else:
        kwargs = {}

    for i, src in enumerate(srcs):
        print(f"===============================================================================================")
        print(f"Loading index file:\n    {src}")
        print(f"-------------------------------------------------------------")
        index_manager = load_index(src, **kwargs)
        index_manager.shuffle(seed + i * seed_multiplier, fast=fast, save_cache=True)


def to_list(x):
    if isinstance(x, str):
        return [x]
    return x

def merge_base(src_files, tgt_file):
    if len(src_files) == 0:
        raise ValueError('src_files is empty.')
    if len(src_files) == 1:
        raise ValueError('Only one src file is provided.')
    if Path(tgt_file).exists():
        raise ValueError(f'tgt_file {tgt_file} already exists.')
    print(f"Reading {len(src_files)} index files ...")
    index_list = [load_index(src_file) for src_file in src_files]
    print("Start merging.")

    print("Merging data_type")
    data_type = []
    for i, (index, src_file) in enumerate(zip(index_list, src_files)):
        absolute_src_file = str(Path(src_file).absolute())
        data_type += [f'-- {i} -- {absolute_src_file}'] + to_list(index.data_type)

    print("Merging ceph_base")
    ceph_base = {}
    ceph_base_inv = {}
    for index in index_list:
        # 检查 ceph_base 是否有冲突, key 和 value 必须是一一对应的
        for k, v in index.ceph_base.items():
            if k in ceph_base:
                if ceph_base[k] != v:
                    raise ValueError(f'ceph_base conflict: {k} -> {v} and {ceph_base[k]}')
            if v in ceph_base_inv:
                if ceph_base_inv[v] != k:
                    raise ValueError(f'ceph_base_inv conflict: {v} -> {k} and {ceph_base_inv[v]}')
            ceph_base[k] = v

    print("Merging arrow_files")
    arrow_files = []
    for i, index in enumerate(index_list):
        arrow_files += index.arrow_files

    print("Merging cum_length and indices")
    count_length = 0
    cum_length = []
    indices = []
    for i, index in enumerate(index_list):
        cum_length.extend([count_length + c for c in index.cum_length])
        indices.extend([count_length + c for c in index.indices])
        count_length += index.cum_length[-1]

    print("Merging group_length")
    group_length = []
    for i, index in enumerate(index_list):
        group_length.extend(index.group_length)

    print("Merging config_file")
    config_file = []
    for i, index in enumerate(index_list):
        config_file.extend([index.config_file])

    print("Building index file ...")
    builder = IndexV2Builder(ceph_base=ceph_base,
                             data_type=data_type,
                             arrow_files=arrow_files,
                             cum_length=cum_length,
                             group_length=group_length,
                             indices=indices,
                             config_file=config_file,
                             )
    builder.save(tgt_file)


def counter(args):
    src_files = args.src
    arrow_files = []
    for src in src_files:
        if '*' in src or '?' in src:
            arrow_files.extend(glob(src))
        elif Path(src).is_dir():
            arrow_files.extend(list(Path(src).rglob('*.arrow')))
        else:
            arrow_files.append(src)

    if len(arrow_files) == 0:
        raise ValueError('No arrow files found.')

    acc = 0
    for arrow_file in tqdm(arrow_files):
        table = get_table(arrow_file)
        length = len(table)
        acc += length
    print(f"\nNumber of arrow files: {len(arrow_files):,}")
    print(f"Total number of rows: {acc:,}\n")


def shard_base_index(source, target, shard_size, **kwargs):
    logger.info(f"Loading index file from {source} ...")
    idx = load_index(source, **kwargs)
    logger.info(f"Loaded.")

    logger.info(f"Saving sharded index file to {target} ...")
    idx.save(target, shard_size=shard_size)
    logger.info(f"Saved.")
