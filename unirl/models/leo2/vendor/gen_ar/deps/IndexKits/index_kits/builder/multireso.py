import io
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import numpy as np
from tqdm import tqdm
from PIL import Image

from .base import IndexV2Builder
from ..arrow_tools import get_columns, ColumnStatus
from ..indexer import ArrowIndexV2
from ..resolution import ResolutionGroup
from ..utils import arrow_mapper as default_arrow_mapper
from ..utils import block_for_rank


def build_multi_resolution_bucket(config_file,
                                  base_size,
                                  src_index_files,
                                  save_file,
                                  reso_step=None,
                                  align=16,
                                  mode=None,
                                  preset=None,
                                  aspect_ratios=None,
                                  num_buckets=None,
                                  min_size=0,
                                  min_area=0,
                                  max_size=16384,
                                  max_area=268_435_456,
                                  md5_hw=None,
                                  ceph_base=None,
                                  height_col='height',
                                  width_col='width',
                                  binary_col=None,
                                  compress=True,
                                  ):
    # Compute base size
    resolutions = ResolutionGroup(base_size, step=reso_step, align=align, mode=mode, preset=preset,
                                  aspect_ratios=aspect_ratios, num_buckets=num_buckets)
    print(resolutions)

    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(src_index_files, str):
        src_index_files = [src_index_files]
    src_indexes = []
    merged_ceph_base = {}
    print(f'Loading indexes:')
    for src_index_file in src_index_files:
        src_indexes.append(ArrowIndexV2(src_index_file, ceph_base=ceph_base))
        print(f'    {src_index_file} | cum_length: {src_indexes[-1].cum_length[-1]} | indices: {len(src_indexes[-1])}')
        for k, v in src_indexes[-1].ceph_base.items():
            if k in merged_ceph_base and merged_ceph_base[k] != v:
                raise ValueError(f'ceph_base {k} is not consistent: k={k}, v1={merged_ceph_base[k]}, v2={v}')
            else:
                merged_ceph_base[k] = v

    if md5_hw is None:
        md5_hw = {}

    arrow_files = src_indexes[0].arrow_files[:]     # !!!important!!!, copy the list
    for src_index in src_indexes[1:]:
        arrow_files.extend(src_index.arrow_files[:])

    cum_length = src_indexes[0].cum_length[:]
    for src_index in src_indexes[1:]:
        cum_length.extend([x + cum_length[-1] for x in src_index.cum_length])
    print(f'cum_length: {cum_length[-1]}')

    group_length_list = src_indexes[0].group_length[:]
    for src_index in src_indexes[1:]:
        group_length_list.extend(src_index.group_length[:])

    total_indices = sum([len(src_index) for src_index in src_indexes])
    total_group_length = sum(group_length_list)
    assert total_indices == total_group_length, f'Total indices {total_indices} != Total group length {total_group_length}'

    buckets = [[] for _ in range(len(resolutions))]
    cum_length_tmp = 0
    total_index_count = 0
    for src_index, src_index_file in zip(src_indexes, src_index_files):
        index_count = 0
        pbar = tqdm(src_index.indices.tolist())
        for i in pbar:
            try:
                height = int(src_index.get_attribute_by_index(i, height_col))
                width = int(src_index.get_attribute_by_index(i, width_col))
            except Exception as e1:
                try:
                    md5 = src_index.get_attribute_by_index(i, 'md5')
                    height, width = md5_hw[md5]
                except Exception as e2:
                    try:
                        width, height = src_index.get_image_by_index(i, column=binary_col).size
                    except Exception as e3:
                        print(f'Error: {e1} --> {e2} --> {e3}. We will skip this image.')
                        continue

            if height < min_size or width < min_size or height * width < min_area:
                continue
            if height > max_size or width > max_size or height * width > max_area:
                continue

            ratio = height / width
            idx = np.argmin(np.abs(resolutions.ratio - ratio))
            buckets[idx].append(i + cum_length_tmp)
            index_count += 1
        print(f"Valid indices {index_count} in {src_index_file}.")
        cum_length_tmp += src_index.cum_length[-1]
        total_index_count += index_count

    print(f'Total indices: {total_index_count}')

    print(f'Making bucket index.')
    indices = {}
    for i, bucket in tqdm(enumerate(buckets)):
        if len(bucket) == 0:
            continue
        reso = f'{resolutions[i]}'
        resolutions.attr[i] = f'{len(bucket):>6d}'
        indices[reso] = bucket

    builder = IndexV2Builder(data_type=['multi-resolution-bucket-v2',
                                        f'base_size={base_size}',
                                        f'reso_step={reso_step}',
                                        f'align={align}',
                                        f'mode={mode}',
                                        f'preset={preset}',
                                        f'aspect_ratios={aspect_ratios}',
                                        f'num_buckets={num_buckets}',
                                        f'min_size={min_size}',
                                        f'min_area={min_area}',
                                        f'src_files='] +
                                       [f'{src_index_file}' for src_index_file in src_index_files],
                             ceph_base=ceph_base if ceph_base is not None else merged_ceph_base,
                             arrow_files=arrow_files,
                             cum_length=cum_length,
                             indices=indices,
                             config_file=config_file,
                             )
    builder.build(save_file, compress=compress)
    print(resolutions)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_file).absolute()}\n'
          f'    Number of indices: {sum([len(v) for k, v in indices.items()]):,}\n'
          f'Number of arrow files: {len(arrow_files):,}\n'
          )


def get_image_size_from_binary(binary):
    image_bytes = io.BytesIO(binary)
    image_bytes.seek(0)
    return Image.open(image_bytes).size


def process_single_arrow(pbar, rank, arrow_file, in_arrow_indices, ratios, min_size=0, min_area=0,
                         max_size=16384, max_area=268_435_456,
                         height_col='height', width_col='width', image_path_col=None, binary_col='image'):
    try:
        # (col_values, status, message)
        h_res, w_res = get_columns(arrow_file, [height_col, width_col])
        if h_res[1] != ColumnStatus.SUCCESS or w_res[1] != ColumnStatus.SUCCESS:
            h_res, w_res = get_columns(arrow_file, [height_col, width_col],
                                       shadows=['hw', 'hw'],
                                       shadow_fn_dict={'hw': partial(default_arrow_mapper, suffix='_hw')})
            if h_res[1] != ColumnStatus.SUCCESS or w_res[1] != ColumnStatus.SUCCESS:
                raise ValueError(f'Failed to get h/w.')
        # heights 和 widths 中可能存在无效值. 这里不做检查, 需要使用者在做 base index 时自行剔除无效值.
        heights = h_res[0]
        widths = w_res[0]
        # 可以从文件中直接读取 height 和 width, 所以标记 heights 和 widths 是整个 arrow 的 sizes
        whole_arrow_sizes = True
    except Exception as e:
        print(f'#{rank}: {e.__class__.__name__}: {e}. {arrow_file}. Online calculating h/w.')
        if image_path_col is not None:
            image_paths, status, message = get_columns(arrow_file, image_path_col)
            assert status == ColumnStatus.SUCCESS, message
            sizes = [Image.open(image_paths[i]).size for i in in_arrow_indices]
        else:
            images, status, message = get_columns(arrow_file, binary_col)
            assert status == ColumnStatus.SUCCESS, message
            sizes = [get_image_size_from_binary(images[i]) for i in in_arrow_indices]
        widths, heights = zip(*sizes)
        # 从图片中读取 height 和 width, 为了减少 IO，只读取 in_arrow_indices 中的图片
        whole_arrow_sizes = False

    bid_indices = []
    try:
        for ei, i in enumerate(in_arrow_indices):
            if whole_arrow_sizes:
                height, width = heights[i], widths[i]
            else:
                height, width = heights[ei], widths[ei]
            if height < min_size or width < min_size or height * width < min_area:
                continue
            if height > max_size or width > max_size or height * width > max_area:
                continue

            ratio = height / width
            idx = np.argmin(np.abs(ratios - ratio))
            bid_indices.append((idx, i))
    except IndexError as e:
        print(f'#{rank}: IndexError encountered while processing arrow file {arrow_file}.')
        print(f'#{rank}: in_arrow_indices: {in_arrow_indices[:30]} ... {in_arrow_indices[-30:]}')
        raise e

    # Return bucket id and in-arrow index
    return bid_indices


def worker_process_resolutions(rank: int,
                               world_size: int,
                               src: str,
                               base_size: int,
                               reso_step: int,
                               align=1,
                               mode=None,
                               preset=None,
                               aspect_ratios=None,
                               num_buckets=None,
                               min_size=0,
                               min_area=0,
                               max_size=16384,
                               max_area=268_435_456,
                               idx_offset=0,      # for multiple index files
                               ceph_base=None,
                               height_col='height',
                               width_col='width',
                               image_path_col=None,
                               binary_col='image',
                               ):
    # Compute base size
    resolutions = ResolutionGroup(base_size, step=reso_step, align=align, mode=mode, preset=preset,
                                  aspect_ratios=aspect_ratios, num_buckets=num_buckets)

    idx = ArrowIndexV2(src, ceph_base=ceph_base)
    # Calculating group cumulated length
    group_lengths = idx.group_length
    group_cum_lengths = np.cumsum(group_lengths)
    group_cum_lengths = np.insert(group_cum_lengths, 0, 0)

    all_arrow_name_indices = list(range(len(idx.arrow_files)))
    arrow_name_indices = block_for_rank(all_arrow_name_indices, rank, world_size)

    buckets = [[] for _ in range(len(resolutions))]

    # Run getting data
    indices_count = 0
    pbar = tqdm(arrow_name_indices, position=rank, desc=f"#{rank}: ", leave=False)
    for arrow_name_index in pbar:
        group_len = group_lengths[arrow_name_index]
        if group_len == 0:
            continue
        in_dataset_index_start = group_cum_lengths[arrow_name_index]
        in_dataset_index_end = in_dataset_index_start + group_len
        in_json_indices = idx.indices[in_dataset_index_start:in_dataset_index_end]
        offset = 0 if arrow_name_index == 0 else idx.cum_length[arrow_name_index - 1]
        in_arrow_indices = in_json_indices - offset
        assert len(in_arrow_indices) == in_dataset_index_end - in_dataset_index_start, \
            f"Length mismatch: {len(in_arrow_indices)} != {in_dataset_index_end - in_dataset_index_start}"

        arrow_name = idx.decode_arrow_file(idx.arrow_files[arrow_name_index])
        bid_indices = process_single_arrow(
            pbar, rank, arrow_name, in_arrow_indices, resolutions.ratio, min_size, min_area, max_size, max_area,
            height_col, width_col, image_path_col, binary_col,
        )
        for bid, in_arrow_index in bid_indices:
            buckets[bid].append(in_arrow_index + offset + idx_offset)
        indices_count += len(bid_indices)
        pbar.set_description(f"#{rank}: count = {indices_count:,}")

    return rank, buckets, indices_count


def build_multi_resolution_bucket_fast(config_file,
                                       base_size,
                                       src_index_files,
                                       save_file,
                                       reso_step=None,
                                       align=16,
                                       mode=None,
                                       preset=None,
                                       aspect_ratios=None,
                                       num_buckets=None,
                                       min_size=0,
                                       min_area=0,
                                       max_size=16384,
                                       max_area=268_435_456,
                                       ceph_base=None,
                                       world_size=1,
                                       height_col='height',
                                       width_col='width',
                                       image_path_col=None,
                                       binary_col='image',
                                       compress=True,
                                       ):
    # Compute base size
    resolutions = ResolutionGroup(base_size, step=reso_step, align=align, mode=mode, preset=preset,
                                  aspect_ratios=aspect_ratios, num_buckets=num_buckets)
    print(resolutions)

    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(src_index_files, str):
        src_index_files = [src_index_files]
    src_indexes = []
    merged_ceph_base = {}
    print(f'Loading indexes:')
    for src_index_file in src_index_files:
        src_indexes.append(ArrowIndexV2(src_index_file, ceph_base=ceph_base))
        print(f'    {src_index_file} | cum_length: {src_indexes[-1].cum_length[-1]} | indices: {len(src_indexes[-1])}')
        for k, v in src_indexes[-1].ceph_base.items():
            if k in merged_ceph_base and merged_ceph_base[k] != v:
                raise ValueError(f'ceph_base {k} is not consistent: k={k}, v1={merged_ceph_base[k]}, v2={v}')
            else:
                merged_ceph_base[k] = v

    arrow_files = src_indexes[0].arrow_files[:]     # !!!important!!!, copy the list
    for src_index in src_indexes[1:]:
        arrow_files.extend(src_index.arrow_files[:])

    cum_length = src_indexes[0].cum_length[:]
    for src_index in src_indexes[1:]:
        cum_length.extend([x + cum_length[-1] for x in src_index.cum_length])
    print(f'cum_length: {cum_length[-1]}')

    group_length_list = src_indexes[0].group_length[:]
    for src_index in src_indexes[1:]:
        group_length_list.extend(src_index.group_length[:])

    total_indices = sum([len(src_index) for src_index in src_indexes])
    total_group_length = sum(group_length_list)
    assert total_indices == total_group_length, f'Total indices {total_indices} != Total group length {total_group_length}'

    buckets = [[] for _ in range(len(resolutions))]
    idx_offset = 0
    total_index_count = 0
    for src_index, src_index_file in zip(src_indexes, src_index_files):
        outputs = []
        if world_size == 1:
            print(f"\nRunning in single process mode {src_index_file}...")
            outputs.append(
                worker_process_resolutions(
                    rank=0, world_size=1, src=src_index_file, base_size=base_size, reso_step=reso_step, align=align,
                    mode=mode, preset=preset, aspect_ratios=aspect_ratios, num_buckets=num_buckets,
                    min_size=min_size, min_area=min_area,
                    max_size=max_size, max_area=max_area,
                    idx_offset=idx_offset, ceph_base=ceph_base, height_col=height_col, width_col=width_col,
                    image_path_col=image_path_col, binary_col=binary_col,
                )
            )
        else:
            print(f"\nRunning in multi-process mode (world_size={world_size}) {src_index_file}...")
            p = Pool(world_size)
            outputs_ = []
            for i in range(world_size):
                outputs_.append(p.apply_async(
                    worker_process_resolutions,
                    args=(i, world_size, src_index_file, base_size, reso_step, align,
                          mode, preset, aspect_ratios, num_buckets,
                          min_size, min_area, max_size, max_area, idx_offset, ceph_base,
                          height_col, width_col, image_path_col, binary_col),
                ))
            for res in tqdm(outputs_, desc=f"Collecting results from {src_index_file} ..."):
                outputs.append(res.get())
            # close
            print(f"Closing multiprocessing pool...")
            p.close()
            p.join()

        indices_count = sum([count for _, _, count in outputs])
        print(f"Valid indices {indices_count} in {src_index_file}.")
        # Sort
        print(f"Sorting outputs...")
        outputs = sorted(outputs, key=lambda x: x[0])
        # Accumulate
        print(f"Accumulating indices...")
        for _, worker_buckets, _ in outputs:
            for i, bucket in enumerate(worker_buckets):
                buckets[i].extend(bucket)
        # Update idx_offset
        idx_offset += src_index.cum_length[-1]
        total_index_count += indices_count

    print(f'Total indices: {total_index_count}')

    print(f'Making bucket index.')
    indices = {}
    for i, bucket in tqdm(enumerate(buckets), total=len(buckets)):
        if len(bucket) == 0:
            continue
        reso = f'{resolutions[i]}'
        resolutions.attr[i] = f'{len(bucket):>6d}'
        indices[reso] = bucket

    builder = IndexV2Builder(data_type=['multi-resolution-bucket-v2',
                                        f'base_size={base_size}',
                                        f'reso_step={reso_step}',
                                        f'align={align}',
                                        f'mode={mode}',
                                        f'preset={preset}',
                                        f'aspect_ratios={aspect_ratios}',
                                        f'num_buckets={num_buckets}',
                                        f'min_size={min_size}',
                                        f'min_area={min_area}',
                                        f'max_size={max_size}',
                                        f'max_area={max_area}',
                                        f'src_files='] +
                                       [f'{src_index_file}' for src_index_file in src_index_files],
                             ceph_base=ceph_base if ceph_base is not None else merged_ceph_base,
                             arrow_files=arrow_files,
                             cum_length=cum_length,
                             indices=indices,
                             config_file=config_file,
                             )
    builder.build(save_file, compress=compress)
    print(resolutions)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_file).absolute()}\n'
          f'    Number of indices: {sum([len(v) for k, v in indices.items()]):,}\n'
          f'Number of arrow files: {len(arrow_files):,}\n'
          )
