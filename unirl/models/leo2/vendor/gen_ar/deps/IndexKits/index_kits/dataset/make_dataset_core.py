import gc
import json
import pickle
from collections import defaultdict
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from index_kits.arrow_tools import get_table
from index_kits.builder.base import IndexV2Builder
from index_kits.builder.multireso import build_multi_resolution_bucket, build_multi_resolution_bucket_fast
from index_kits.dataset.config_parse import DatasetConfig
from index_kits.utils import block_for_rank


def get_indices(arrow_file, repeat_times, filter_fn, repeat_fn, callback=None):
    """
    Get valid indices from a single arrow_file.

    Parameters
    ----------
    arrow_file: str
    repeat_times: int
        Repeat remain indices multiple times.
    filter_fn
    callback

    Returns
    -------

    """
    try:
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, 'r')).read_all()
    except Exception as e:
        print(arrow_file, e)
        raise e
    length = len(table)

    if len(table) == 0:
        print(f"Warning: Empty table: {arrow_file}")
        indices = []
        stats = {}

    else:
        # Apply filter_fn if available
        if filter_fn is not None:
            mask, stats, md5s = filter_fn(arrow_file, table)
        else:
            mask = pd.Series([True] * length)
            stats = {}
            md5s = None

        # Apply callback function if available
        if callback is not None:
            mask, stats = callback(arrow_file, table, mask, stats, md5s)

        # Get indices
        if mask is not None:
            indices = np.where(mask)[0].tolist()
        else:
            indices = list(range(length))

        # Apply indices repeat
        if repeat_fn is not None:
            indices, repeat_stats = repeat_fn(arrow_file, table, indices, repeat_times, md5s)
            stats.update(repeat_stats)

    # manual memory management to avoid memory issue. Pyarrow does not release memory immediately.
    del table
    gc.collect()

    return arrow_file, length, indices, stats


def load_md5_files(files, name=None):
    if isinstance(files, str):
        files = [files]
    md5s = set()
    for file in files:
        md5s.update(Path(file).read_text().splitlines())
    print(f"    {name} md5s: {len(md5s):,}")

    return md5s

def load_md52cls_files(files, name=None):
    if isinstance(files, str):
        files = [files]
    md52cls = {}
    for file in files:
        with Path(file).open() as f:
            md52cls.update(json.load(f))
    print(f"    {name} md52cls: {len(md52cls):,}")

    return md52cls


def merge_and_build_index(data_type, src, dconfig, save_path, md5_col="md5", compress=True, shard_size=None):
    if isinstance(src, str):
        files = list(sorted(glob(src)))
    else:
        files = list(sorted(src))
    print(f"Found {len(files):,} temp pickle files.")
    for fname in files:
        print(f"    {fname}")
    arrow_files = []
    table_lengths = []
    indices_list = []
    bad_stats_total = defaultdict(int)
    total_indices = 0
    total_processed_length = 0
    for file_name in tqdm(files):
        with Path(file_name).open('rb') as f:
            data = pickle.load(f)
        for arrow_file, table_length, indices, *args in tqdm(data, leave=False):
            arrow_files.append(arrow_file)
            table_lengths.append(table_length)
            total_processed_length += table_length
            indices_list.append(indices)
            total_indices += len(indices)
            if len(args) > 0 and args[0]:
                bad_stats = args[0]
                for k, v in bad_stats.items():
                    bad_stats_total[k] += v

    if len(bad_stats_total):
        stats_save_dir = Path(save_path).parent
        stats_save_dir.mkdir(parents=True, exist_ok=True)
        stats_save_path = stats_save_dir / (Path(save_path).stem + '_stats.txt')
        stats_save_path.write_text('\n'.join([f'{k:>50s} {v}' for k, v in bad_stats_total.items()]) + '\n')
        print(f"Save stats to {stats_save_path}")

    print(f'Arrow files: {len(arrow_files):,}')
    print(f'Processed indices: {total_processed_length:,}')
    print(f'Valid indices: {total_indices:,}')

    cum_length = 0
    total_indices = []
    cum_lengths = []
    group_lengths = []
    existed = set()
    print(f"Accumulating indices...")
    pbar = tqdm(zip(arrow_files, table_lengths, indices_list), total=len(arrow_files), mininterval=1)
    _count = 0
    for arrow_file, table_length, indices in pbar:
        if len(indices) > 0 and dconfig.remove_md5_dup:
            new_indices = []
            table = get_table(arrow_file)
            if md5_col not in table.column_names:
                raise ValueError(f"Column '{md5_col}' not found in {arrow_file}. "
                                 f"When `remove_md5_dup: true` is set, {md5_col} column is required.")
            md5s = table[md5_col].to_pandas()
            for i in indices:
                md5 = md5s[i]
                if md5 in existed:
                    continue
                existed.add(md5)
                new_indices.append(i)
            indices = new_indices

        total_indices.extend([int(i + cum_length) for i in indices])
        cum_length += table_length
        cum_lengths.append(cum_length)
        group_lengths.append(len(indices))

        _count += 1

        if _count % 100 == 0:
            pbar.set_description(f'Indices: {len(total_indices):,}')

    try:
        builder = IndexV2Builder(data_type=data_type,
                                 ceph_base=dconfig.ceph_base,
                                 arrow_files=arrow_files,
                                 cum_length=cum_lengths,
                                 group_length=group_lengths,
                                 indices=total_indices,
                                 config_file=dconfig.config_file,
                                 )
    except ValueError as e:
        if "The indices list is empty" in str(e):
            raise ValueError(f"Empty indices list.")
        else:
            raise e
    builder.build(save_path, compress=compress, shard_size=shard_size)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_path).absolute()}\n'
          f'    Number of indices: {len(total_indices):,}\n'
          f'Number of arrow files: {len(arrow_files):,}\n'
          )
    return Path(save_path)


def worker_startup(rank, world_size, dconfig, prefix, work_dir, callback=None):
    # Prepare names for this worker
    arrow_names = block_for_rank(dconfig.names, rank, world_size)
    print(f'Rank {rank} has {len(arrow_names):,} names.')

    # Run get indices
    print(f"Start getting indices...")
    indices = []
    for arrow_name, repeat_times in tqdm(arrow_names, position=rank, desc=f"#{rank}: ", leave=False):
        indices.append(get_indices(arrow_name, repeat_times, dconfig.filter, dconfig.repeater, callback))

    # Save to a temp file
    temp_save_path = work_dir / f'data/temp_pickles/{prefix}-{rank + 1:04d}_of_{world_size:04d}.pkl'
    temp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_save_path.open('wb') as f:
        pickle.dump(indices, f)
    print(f'Rank {rank} finished. Write temporary data to {temp_save_path}')

    return temp_save_path


def startup(config_file,
            save,
            world_size=1,
            work_dir='.',
            callback=None,
            use_cache=False,
            compress=True,
            shard_size=None,
            ):
    work_dir = Path(work_dir)
    save_path = Path(save)
    if save_path.suffix != '.json':
        save_path = save_path.parent / (save_path.name + '.json')
    print(f"Using save_path: {save_path}")
    prefix = f"{save_path.stem}"

    # Parse dataset config and build the data_type list
    dconfig = DatasetConfig(work_dir, config_file)
    data_type = []
    for k, v in dconfig.data_type.items():
        data_type.extend(v)
        print(f"{k}:")
        for x in v:
            print(f'    {x}')
    if dconfig.remove_md5_dup:
        data_type.append('Remove md5 duplicates.')
    else:
        data_type.append('Keep md5 duplicates.')

    # Start processing
    if not use_cache:
        temp_pickles = []
        if world_size == 1:
            print(f"\nRunning in single process mode...")
            temp_pickles.append(worker_startup(rank=0,
                                               world_size=1,
                                               dconfig=dconfig,
                                               prefix=prefix,
                                               work_dir=work_dir,
                                               callback=callback,
                                               ))
        else:
            print(f"\nRunning in multi-process mode (world_size={world_size})...")
            p = Pool(world_size)
            temp_pickles_ = []
            for i in range(world_size):
                temp_pickles_.append(p.apply_async(worker_startup, args=(i, world_size, dconfig, prefix, work_dir, callback)))

            for res in temp_pickles_:
                temp_pickles.append(res.get())
            # close
            p.close()
            p.join()
    else:
        temp_pickles = glob(f'{work_dir}/data/temp_pickles/{prefix}-*_of_{world_size:04d}.pkl')

    # Merge temp pickles and build index
    saved = merge_and_build_index(data_type,
                                  temp_pickles,
                                  dconfig,
                                  save_path,
                                  md5_col=dconfig.md5_col,
                                  compress=compress,
                                  shard_size=shard_size,
                                  )
    # If saved, remove temp pickles
    if saved.exists() and saved.stat().st_size > 0:
        for temp_pickle in temp_pickles:
            Path(temp_pickle).unlink()


make_base = startup


def make_multireso(target,
                   config_file=None,
                   src=None,
                   base_size=None,
                   reso_step=None,
                   align=None,
                   mode=None,
                   preset=None,
                   aspect_ratios=None,
                   num_buckets=None,
                   min_size=0,
                   min_area=0,
                   max_size=16384,
                   max_area=268_435_456,
                   md5_file=None,
                   legacy=False,
                   world_size=1,
                   height_col='height',
                   width_col='width',
                   image_path_col=None,
                   binary_col=None,
                   compress=True,
                   ):
    if config_file is not None:
        with Path(config_file).open() as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    src = config.get('src', src)
    base_size = config.get('base_size', base_size)
    reso_step = config.get('reso_step', reso_step)
    align = config.get('align', align)
    mode = config.get('mode', mode)
    preset = config.get('preset', preset)
    aspect_ratios = config.get('aspect_ratios', aspect_ratios)
    num_buckets = config.get('num_buckets', num_buckets)
    min_size = config.get('min_size', min_size)
    min_area = config.get('min_area', min_area)
    max_size = config.get('max_size', max_size)
    max_area = config.get('max_area', max_area)
    md5_file = config.get('md5_file', md5_file)
    height_col = config.get('height_col', height_col)
    width_col = config.get('width_col', width_col)
    image_path_col = config.get('image_path_col', image_path_col)
    binary_col = config.get('binary_col', binary_col)

    valid_arguments = [
        "src", "base_size", "reso_step", "align", "mode", "preset", "aspect_ratios", "num_buckets",
        "min_size", "min_area", "max_size", "max_area", "md5_file",
        "height_col", "width_col", "image_path_col", "binary_col",
    ]
    # Check invalid arguments
    invalid_arguments = []
    for k in config.keys():
        if k not in valid_arguments:
            invalid_arguments.append(k)
    if len(invalid_arguments) > 0:
        raise ValueError(f'Invalid arguments in config file: {invalid_arguments}')

    if src is None:
        raise ValueError('src must be provided in either config file or command line.')
    if base_size is None:
        raise ValueError('base_size must be provided.')

    if md5_file is not None:
        with open(md5_file, 'rb') as f:
            md5_hw = pickle.load(f)
        print(f'Md5 to height and width: {len(md5_hw):,}')
    else:
        md5_hw = None

    if not legacy:
        if md5_hw is not None:
            raise ValueError('fast mode does not support md5_hw.')
        build_multi_resolution_bucket_fast(
            config_file=config_file,
            base_size=base_size,
            reso_step=reso_step,
            align=align,
            mode=mode,
            preset=preset,
            aspect_ratios=aspect_ratios,
            num_buckets=num_buckets,
            min_size=min_size,
            min_area=min_area,
            max_size=max_size,
            max_area=max_area,
            src_index_files=src,
            save_file=target,
            world_size=world_size,
            height_col=height_col,
            width_col=width_col,
            image_path_col=image_path_col,
            binary_col=binary_col or 'image',
            compress=compress,
        )
    else:
        if world_size > 1:
            raise ValueError('For backward compatibility, world_size must be 1 in slow mode. If you want to use '
                             'multi-process mode, please remove --legacy flag.')
        if image_path_col is not None:
            raise ValueError('image_path_col is not supported in slow mode.')
        build_multi_resolution_bucket(
            config_file=config_file,
            base_size=base_size,
            reso_step=reso_step,
            align=align,
            mode=mode,
            preset=preset,
            aspect_ratios=aspect_ratios,
            num_buckets=num_buckets,
            min_size=min_size,
            min_area=min_area,
            max_size=max_size,
            max_area=max_area,
            src_index_files=src,
            save_file=target,
            md5_hw=md5_hw,
            height_col=height_col,
            width_col=width_col,
            binary_col=binary_col,
            compress=compress,
        )


# =============================================================================
# Automatically create base index and multireso index from arrow files
# =============================================================================

def create_base_yaml(arrow_dir: Path, base_name: str):
    if not str(arrow_dir).startswith('/'):
        raise ValueError("Only support absolute path for arrow_dir.")

    data: dict[str, Any] = dict(
        source=[f"{arrow_dir}/*.arrow"],
    )

    arrow_dir_parts = arrow_dir.parts
    if len(arrow_dir_parts) >= 3:
        ceph_prefix = '/' + '/'.join(arrow_dir_parts[1:3])
        data['ceph_base'] = {ceph_prefix: "prefix"}

    save_file = arrow_dir.parent / f"{base_name}.yaml"
    with save_file.open('w') as f:
        yaml.dump(data, f)
    return save_file


def create_multireso_yaml(base_index_file: Path, args, multireso_name: str):
    if not base_index_file.exists():
        raise ValueError(f"Base index file {base_index_file} does not exist.")

    data: dict[str, Any] = dict(
        src=[str(base_index_file)],
        base_size=args.base_size,
    )
    keys = [
        "reso_step", "align", "mode", "preset", "aspect_ratios", "num_buckets",
        "min_size", "min_area", "max_size", "max_area",
    ]
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            data[key] = value

    if "reso_step" not in data:
        data["reso_step"] = args.base_size // 16
    if "min_size" not in data:
        data["min_size"] = int(args.base_size * 0.75)
    if "min_area" not in data:
        data["min_area"] = args.base_size * args.base_size

    save_file = base_index_file.parent / f"{multireso_name}{args.base_size}.yaml"
    with save_file.open('w') as f:
        yaml.dump(data, f)
    return save_file


def build_index(args):
    arrow_dir = Path(args.arrow_dir).expanduser().absolute()
    if not arrow_dir.exists():
        raise ValueError(f"Arrow directory {arrow_dir} does not exist.")
    arrow_files = list(arrow_dir.rglob('*.arrow'))
    if len(arrow_files) == 0:
        raise ValueError(f"No arrow files found in {arrow_dir}.")

    config_file = create_base_yaml(arrow_dir, args.base_name)
    base_index_file = arrow_dir.parent / f"{args.base_name}.json"
    make_base(config_file,
              base_index_file,
              world_size=args.world_size,
              work_dir='.',
              compress=False,
              shard_size=args.shard_size,
              )

    if args.multireso:
        config_file = create_multireso_yaml(base_index_file, args)
        mr_index_file = arrow_dir.parent / f"{args.multireso_name}{args.base_size}.json"
        make_multireso(mr_index_file,
                       config_file=config_file,
                       world_size=args.world_size,
                       compress=False,
                       )
