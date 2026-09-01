import json

from tqdm import tqdm
import numpy as np
from typing import List
from multiprocessing import Pool
from pathlib import Path
from functools import partial
import pandas as pd

from ..arrow_tools import get_columns, get_table, ColumnStatus
from ..common import load_index
from ..utils import LoadIndexError, arrow_mapper


def worker_dump_data(rank: int,
                     world_size: int,
                     src: str,
                     target: str,
                     index_type: str,
                     columns: List[str],
                     with_index: bool = False,
                     with_header: bool = False,
                     strict: bool = False,
                     verbose: int = 0,
                     chunk_size: int = 5_000_000,
                     ceph_base_inv: dict = None,
                     ):
    idx = load_index(src, multireso=index_type == "multireso", ceph_base_inv=ceph_base_inv)
    # Prepare names for this worker
    arrow_name_indices = list(range(len(idx.arrow_files)))[rank::world_size]
    print(f'Rank {rank} has {len(arrow_name_indices):,} names.')

    if any([('@' in col) for col in columns]):
        shadow_fn_dict = {}
        for col in columns:
            _, shadow = col.split('@')
            shadow_fn_dict[shadow] = partial(arrow_mapper, suffix=f'_{shadow}')
    else:
        shadow_fn_dict = None

    # Creating target directory and filename
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    def _dump_and_clear(data_buffer, shard_index, key=None, arrow_name=None):
        key = "" if key is None else f"{key}_"
        dump_file = target / f'part_{rank:03d}_of_{world_size:03d}_{key}{shard_index}.csv'
        try:
            df = pd.DataFrame(data_buffer)
            df.to_csv(dump_file, index=False, header=with_header)
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}\nCurrent arrow: {arrow_name}")
            raise e
        data_buffer.clear()
        shard_index += 1
        return shard_index

    if index_type == 'base':
        # Calculating group cumulated length
        group_lengths = idx.group_length
        group_cum_lengths = np.cumsum(group_lengths)
        group_cum_lengths = np.insert(group_cum_lengths, 0, 0)

        total_count = 0
        shard_index = 0
        data_buffer = []
        pbar = tqdm(arrow_name_indices, position=rank, desc=f"#{rank}: ", leave=False)

        # Traversing arrows
        for arrow_name_index in pbar:
            # Get indices in this arrow
            in_dataset_index_start = group_cum_lengths[arrow_name_index]
            in_dataset_index_end = group_cum_lengths[arrow_name_index + 1]
            in_json_indices = idx.indices[in_dataset_index_start:in_dataset_index_end]  # np.ndarray
            if len(in_json_indices) == 0:
                if verbose > 0:
                    print("Skip 0 length group:", idx.decode_arrow_file(idx.arrow_files[arrow_name_index]), "Arrow index:", arrow_name_index)
                continue

            # Shift in_json_index to in_arrow_index
            in_json_index_offset = 0 if arrow_name_index == 0 else idx.cum_length[arrow_name_index - 1]
            in_arrow_indices = in_json_indices - in_json_index_offset   # np.ndarray
            assert len(in_arrow_indices) == in_dataset_index_end - in_dataset_index_start, \
                f"Length mismatch: {len(in_arrow_indices)} != {in_dataset_index_end - in_dataset_index_start}"

            arrow_name = idx.decode_arrow_file(idx.arrow_files[arrow_name_index])
            list_of_columns = get_columns(arrow_name, columns, shadow_fn_dict=shadow_fn_dict)
            if strict:
                for i, (values, status, msg) in enumerate(list_of_columns):
                    if status != ColumnStatus.SUCCESS:
                        print(f"Error: {msg}")
                        print(f"  {arrow_name} {columns[i]}")
                        raise ValueError(f"Error in {arrow_name} {columns[i]}")
            else:
                for i, (values, status, msg) in enumerate(list_of_columns):
                    if status != ColumnStatus.SUCCESS:
                        print(f"Warning: {msg}")
                        print(f"  {arrow_name} {columns[i]}")
                        continue
            if with_index:
                index_column = list(range(in_dataset_index_start, in_dataset_index_end))
            for i, ai in enumerate(in_arrow_indices):
                item = {"_in_dataset_index_": index_column[i]} if with_index else {}
                for j in range(len(columns)):
                    item[columns[j]] = list_of_columns[j][0][ai]
                data_buffer.append(item)
                total_count += 1

                if len(data_buffer) == chunk_size:
                    shard_index = _dump_and_clear(data_buffer, shard_index, arrow_name=arrow_name)
            pbar.set_description(
                f"#{rank}: total_count = {total_count:,}, dumped shards = {shard_index:,} (x{chunk_size:,})")

        if len(data_buffer) > 0:
            shard_index = _dump_and_clear(data_buffer, shard_index)
        total_shards = shard_index

    else:
        # Calculating group cumulated length
        bucket_group_lengths = {key: bucket.group_length for key, bucket in idx.buckets_map.items()}
        keys = list(bucket_group_lengths.keys())
        bucket_group_cum_lengths = {}
        for key in keys:
            bucket_group_cum_lengths[key] = np.insert(np.cumsum(bucket_group_lengths[key]), 0, 0)

        total_count = {key: 0 for key in keys}
        shard_index = {key: 0 for key in keys}
        data_buffer = {key: [] for key in keys}
        pbar = tqdm(arrow_name_indices, position=rank, desc=f"#{rank}: ", leave=False)

        # Traversing arrows
        for arrow_name_index in pbar:
            key2indices = {}
            for key in keys:
                # Get indices in this arrow for this bucket
                in_dataset_index_start = bucket_group_cum_lengths[key][arrow_name_index]
                in_dataset_index_end = bucket_group_cum_lengths[key][arrow_name_index + 1]
                bucket = idx.buckets_map[key]
                in_json_indices = bucket.indices[in_dataset_index_start:in_dataset_index_end]  # np.ndarray
                if len(in_json_indices) == 0:
                    if verbose > 0:
                        print("Skip 0 length group:", idx.decode_arrow_file(idx.arrow_files[arrow_name_index]), "Arrow index:", arrow_name_index)
                    continue

                # Shift in_json_index to in_arrow_index
                in_json_index_offset = 0 if arrow_name_index == 0 else bucket.cum_length[arrow_name_index - 1]
                in_arrow_indices = in_json_indices - in_json_index_offset  # np.ndarray
                assert len(in_arrow_indices) == in_dataset_index_end - in_dataset_index_start, \
                    f"Length mismatch: {len(in_arrow_indices)} != {in_dataset_index_end - in_dataset_index_start}"
                key2indices[key] = (in_arrow_indices, in_dataset_index_start, in_dataset_index_end)

            arrow_name = idx.buckets[0].decode_arrow_file(idx.buckets[0].arrow_files[arrow_name_index])
            list_of_columns = get_columns(arrow_name, columns, shadow_fn_dict=shadow_fn_dict)
            if strict:
                for i, (values, status, msg) in enumerate(list_of_columns):
                    if status != ColumnStatus.SUCCESS:
                        print(f"Error: {msg}")
                        print(f"  {arrow_name} {columns[i]}")
                        raise ValueError(f"Error in {arrow_name} {columns[i]}")
            else:
                for i, (values, status, msg) in enumerate(list_of_columns):
                    if status != ColumnStatus.SUCCESS:
                        print(f"Warning: {msg}")
                        print(f"  {arrow_name} {columns[i]}")
                        continue

            for key in key2indices:
                in_arrow_indices, in_dataset_index_start, in_dataset_index_end = key2indices[key]
                if with_index:
                    index_column = list(range(in_dataset_index_start, in_dataset_index_end))
                for i, ai in enumerate(in_arrow_indices):
                    item = {"_in_dataset_index_": index_column[i]} if with_index else {}
                    for j in range(len(columns)):
                        item[columns[j]] = list_of_columns[j][0][ai]
                    data_buffer[key].append(item)
                    total_count[key] += 1

                    if len(data_buffer[key]) == chunk_size:
                        shard_index[key] = _dump_and_clear(data_buffer[key], shard_index[key], key, arrow_name=arrow_name)
            pbar.set_description(
                f"#{rank}: "
                f"total_count = {sum(total_count.values()):,}, "
                f"dumped shards = {sum(shard_index.values()):,} (x{chunk_size:,})"
            )

        for key in keys:
            if len(data_buffer[key]) > 0:
                shard_index[key] = _dump_and_clear(data_buffer[key], shard_index[key], key)
        total_count = sum(total_count.values())
        total_shards = sum(shard_index.values())

    print(f"#{rank}: Finished dumping. "
          f"Total count = {total_count:,}, "
          f"Total shards = {total_shards:,} (x{chunk_size:,})")
    return total_count


def dump_data(src,
              columns,
              target,
              with_index=False,
              with_header=False,
              num_proc=1,
              strict=False,
              verbose=0,
              chunk_size=5_000_000,
              arrow_prefix=None,
              ):
    if not isinstance(src, str):
        raise LoadIndexError(f'Expected `src` type str, got {type(src)}.')
    if not src.endswith('.arrow') and not src.endswith('.json'):
        raise LoadIndexError(f'Expected `src` to be an arrow file or an index file ending with .json, got {src}.')

    # Detect base or multireso index
    if src.endswith('.arrow'):
        index_type = "base"
        ceph_base_inv = None
    else:
        with open(src) as f:
            data = json.load(f)
        if isinstance(data['group_length'], list):
            index_type = "base"
        elif isinstance(data['group_length'], dict):
            index_type = "multireso"
        else:
            raise LoadIndexError(f'Expected `group_length` type list or dict, got {type(data["group_length"])}.')
        if arrow_prefix is not None and len(data["ceph_base"]) > 1:
            raise LoadIndexError(
                f"When `arrow_prefix` is provided, the index file should have no more than one ceph_base, "
                f"got {len(data['ceph_base'])}."
            )
        if arrow_prefix is not None and len(data["ceph_base"]) == 1:
            ceph_base_inv = {v: arrow_prefix for k, v in data["ceph_base"].items()}
        else:
            ceph_base_inv = None

    if num_proc == 1:
        counts = [
            worker_dump_data(
                0, 1, src, target, index_type, columns, with_header, with_index, strict, verbose, chunk_size,
                ceph_base_inv,
            )
        ]
    else:
        # Dump data
        p = Pool(num_proc)
        results = []
        for i in range(num_proc):
            result = p.apply_async(worker_dump_data, args=(
                i, num_proc, src, target, index_type, columns, with_header, with_index, strict, verbose, chunk_size,
                ceph_base_inv,
            ))
            results.append(result)

        counts = []
        for res in results:
            counts.append(res.get())
        p.close()

    print("")
    print(f"Total data: {sum(counts):,}\nDumped to: {target}")
    print("")


def align_check_single_arrow(args):
    arrow_file, target_srcs, allow_missing, cols = args
    arrow_parent = arrow_file.parent.name
    arrow_name = arrow_file.name
    ref_table = get_table(arrow_file)
    cols_data = {}
    if cols is not None:
        for col in cols:
            cols_data[col] = ref_table[col].to_pylist()
    for target in target_srcs:
        target_file = target / arrow_parent / arrow_name
        if not allow_missing and not target_file.exists():
            print(f"Missing: {target_file}")
        else:
            target_table = get_table(target_file)
            if len(ref_table) != len(target_table):
                print(f"Length mismatch: {len(ref_table)} != {len(target_table)}. {arrow_file} vs {target_file}")
            elif cols is not None:
                for col in cols:
                    target_col = target_table[col].to_pylist()
                    if not all([a == b for a, b in zip(cols_data[col], target_col)]):
                        print(f"Data mismatch(col={col}): {arrow_file} vs {target_file}")


def align_check(srcs, subdir, allow_missing, cols=None, world_size=1):
    assert len(srcs) > 1, f"Need at least 2 sources to align check."
    srcs = [Path(src) for src in srcs]

    refs = list(srcs[0].glob(f'{subdir}/*.arrow'))
    print(f"Found {len(refs):,} arrows in {srcs[0]}")

    if world_size == 1:
        for ref in tqdm(refs):
            align_check_single_arrow((ref, srcs[1:], allow_missing, cols))
    else:
        with Pool(processes=world_size) as p:
            _ = list(tqdm(
                p.imap(align_check_single_arrow, [(ref, srcs[1:], allow_missing, cols) for ref in refs]),
                total=len(refs)
            ))
