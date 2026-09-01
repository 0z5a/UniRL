import ast
import io
import os
import pickle
import psutil
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List, Callable, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm
from PIL import Image

from processors.image_kits import md5sum_binary, download_data


# ===============================================================================
# Custom arrow mappers
# ===============================================================================
def arrow_mapper_adt2ceph(arrow_file, suffix, ceph):
    # adtfs 满了, 把 adtfs 的 shadow 文件映射到 nj10
    parts = list(Path(arrow_file).parts)
    # Add shadow_suffix to third-to-last part
    parts[-3] += suffix
    shadow_arrow_file = str(Path(*parts))
    shadow_arrow_file = shadow_arrow_file.replace('/mnt/adtfs3', f'{ceph}/0_public_datasets/adtfs3')
    shadow_arrow_file = shadow_arrow_file.replace('/mnt/adtfs2', f'{ceph}/0_public_datasets/adtfs2')
    return shadow_arrow_file


# ===============================================================================
# Read and write arrow files
# ===============================================================================
def get_table(arrow_file):
    """
    Read an arrow file and return an arrow table.
    """
    return pa.ipc.RecordBatchFileReader(pa.memory_map(f"{arrow_file}", "r")).read_all()


def to_table(table, arrow_path, allow_overwrite=False):
    # 为了安全, 我们不允许覆写
    if not allow_overwrite and Path(arrow_path).exists():
        raise FileExistsError(f"Arrow file {arrow_path} already exists. Please remove it first or use a different path.")
    # 确保目标文件夹存在
    Path(arrow_path).parent.mkdir(parents=True, exist_ok=True)
    arrow_path_tmp = str(arrow_path) + '.tmp'

    # 保存到 tmp 文件中, 然后再重命名, 避免中断时出现问题
    with pa.OSFile(arrow_path_tmp, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    # 如果目标文件存在, 则删除
    if Path(arrow_path).exists():
        Path(arrow_path).unlink()
    # 临时文件重命名为目标文件
    Path(arrow_path_tmp).rename(arrow_path)


def dataframe_to_table(dataframe, save_path):
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    to_table(table, save_path)


def pydict_to_table(pydict, save_path):
    table = pa.Table.from_pydict(pydict)
    to_table(table, save_path)


def _binary_reader(image_path):
    with open(image_path, 'rb') as f:
        return f.read()


def _binary_reader_with_md5_height_width(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    md5 = md5sum_binary(data)
    buffer = io.BytesIO(data)
    buffer.seek(0)
    width, height = Image.open(buffer).size
    return data, md5, height, width


def _url_reader_with_md5_height_width(url):
    data = download_data(url, ret_binary=True)
    md5 = md5sum_binary(data)
    buffer = io.BytesIO(data)
    buffer.seek(0)
    width, height = Image.open(buffer).size
    return data, md5, height, width


def make_arrows(data_provider,
                save_dir,
                image_source_col=None,
                image_target_col=None,
                reader=None,
                remove_source_col=False,
                num_slice=5000,
                num_processors=16,
                force=False,
                start_idx=0,
                allow_col_missing=False,
                ):
    """
    Make arrows from a data table.

    Make sure the save directory is empty. If the saving directory exists,
    the process will exit directly.

    Parameters
    ----------
    data_provider: Iterator
    save_dir: str, pathlib.Path
        The directory to save the arrows.
    image_source_col: str, List[str]
        The column name that read image data from. This column can be either
        url(start with 'http') or local path.
    image_target_col: str, List[str], List[List[str]]
        The column name that save image data to.
    reader: Optional[Union[Callable, str]]
        Read a image and dump to data. If str, should be one of the following:
        - 'binary': read image as binary data.
        - 'binary_md5_height_width': read image as binary data, and return md5, height, width.
    remove_source_col: bool
        If True, remove the source column in the arrow.
    num_slice: int or None
        The data is saved to a bunch of arrows, with each arrow contains `num_slice` records.
        If None, save all data to one arrow.
    num_processors: int
        Number of processors to parse image data.
    force: bool
        If True, force to overwrite the existing arrows.
    start_idx: int
        The starting index of the arrow files.
    allow_col_missing: bool
        If True, allow the column missing in the data.

    """
    # Check exists.
    arrow_dir = Path(save_dir)
    if arrow_dir.exists() and len(list(arrow_dir.glob("*.arrow"))) > 0 and not force:
        logger.info(f"Directory {arrow_dir} exists and is not empty.")
        return
    arrow_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(image_source_col, str):
        image_source_cols = [image_source_col]
        image_target_cols = [image_target_col]
    else:
        image_source_cols = image_source_col
        image_target_cols = image_target_col
        if image_source_cols is not None:
            assert len(image_source_cols) == len(image_target_cols), (
                f"Length of image_source_col should be equal to the length of image_target_col, "
                f"got {len(image_source_cols)} vs {len(image_target_cols)}"
            )

    if reader == "binary":
        reader = _binary_reader
        assert all([len(target_cols) == 1 for target_cols in image_target_cols]), (
            f"When reader is 'binary', image_target_col should be a str, got {image_target_cols}"
        )
    elif reader == "binary_md5_height_width":
        reader = _binary_reader_with_md5_height_width
        assert all([len(target_cols) == 4 for target_cols in image_target_cols]), (
            f"When reader is 'binary_md5_height_width', image_target_col should be a list of 4 str "
            f"as the columns of (binary, md5, height, width), got {image_target_cols}"
        )
    elif reader == "url_md5_height_width":
        reader = _url_reader_with_md5_height_width
        assert all([len(target_cols) == 4 for target_cols in image_target_cols]), (
            f"When reader is 'url_md5_height_width', image_target_col should be a list of 4 str "
            f"as the columns of (binary, md5, height, width), got {image_target_cols}"
        )
    elif reader is not None:
        assert callable(reader), f"reader should be a callable, got {type(reader)}"

    def process_batch(dataframe, pbar, idx):
        while True:
            arrow_path = arrow_dir / f'{idx:05d}.arrow'
            if not arrow_path.exists():
                break
            idx += 1

        pbar.set_description(f"[{idx}] Making arrow {arrow_path} ({len(dataframe)}) ...")

        if reader:
            with Pool(processes=num_processors) as pool:
                for source_col, target_col in tqdm(zip(image_source_cols, image_target_cols),
                                                   total=len(image_source_cols), leave=False):
                    if source_col not in dataframe.columns:
                        if allow_col_missing:
                            continue
                        raise ValueError(f"Column {source_col} not found in the dataframe.")
                    source_list = dataframe[source_col].to_list()
                    # We use multiprocess to accelerate reading image binary data.
                    binary_data_list = list(tqdm(pool.imap(reader, source_list), total=len(source_list), leave=False))
                    if isinstance(target_col, str):
                        dataframe[target_col] = binary_data_list
                        dataframe = dataframe[dataframe[target_col].notna()]
                    else:
                        assert isinstance(target_col, (list, tuple)), (
                            f"image_target_col should be a str of list of str, got {type(target_col)}"
                        )
                        assert isinstance(binary_data_list[0], (list, tuple)), (
                            f"When image_target_col is a list/tuple, `reader` should return an list/tuple"
                        )
                        assert len(target_col) == len(binary_data_list[0]), (
                            f"Length of image_target_col should be equal to the length of the returned list/tuple"
                            f"from `reader`, got {len(target_col)} vs {len(binary_data_list[0])}"
                        )
                        for i, col in enumerate(target_col):
                            dataframe[col] = [x[i] for x in binary_data_list]
                        dataframe = dataframe[dataframe[target_col[0]].notna()]
                    if remove_source_col:
                        dataframe.drop(columns=[source_col], inplace=True)

        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        to_table(table, arrow_path)
        pbar.update()
        return idx

    pbar = tqdm(position=0)
    idx = start_idx
    current_data = None
    for data in data_provider:
        # data is a pandas.Dataframe
        if num_slice is None:
            idx = process_batch(data, pbar, idx)
            idx += 1
        else:
            if current_data is None:
                current_data = data
            else:
                current_data = pd.concat([current_data, data], ignore_index=True)
            if len(current_data) < num_slice:
                continue

            while len(current_data) >= num_slice:
                dataframe = current_data[:num_slice].copy()
                current_data = current_data[num_slice:]
                idx = process_batch(dataframe, pbar, idx)
                idx += 1
    # Process the remaining data
    if current_data is not None and len(current_data) > 0:
        dataframe = current_data.copy()
        process_batch(dataframe, pbar, idx)


def token_reader(token):
    try:
        list_of_list = ast.literal_eval(token)
        array = np.array(list_of_list)
        h, w = array.shape
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        return np.nan, 0, 0
    return array.reshape(-1), h, w


def csv_token_to_arrow(csv_files):
    csv_files = sorted(glob(csv_files))

    def data_provider():
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            yield data

    section = 10000
    groups = []
    for i in range(0, len(csv_files), section):
        groups.append(csv_files[i:i + section])
    print(f"Total {len(csv_files)} csv files, {len(groups)} groups.")

    for i, group in enumerate(groups, start=1):
        print(f"Group: {i}, {len(groups)} csv files")
        make_arrows(data_provider(),
                    save_dir=f'/apdcephfs_nj10/share_301739632/0_public_datasets/text2image_ar_std300m/images_token/241025_{i}',
                    image_source_col='image_code',
                    image_target_col='hy_magvitv2_241024',
                    reader=token_reader,
                    remove_source_col=True,
                    )


class ConstValueDict(object):
    def __init__(self, value):
        self.value = value

    def __len__(self):
        return 1

    def __getitem__(self, key):
        return self.value

    def __contains__(self, key):
        return True

    def items(self):
        return [(None, self.value)]


def _csv_column_to_arrow(src_files: List,
                         arrow_list: List,
                         src_csv_col: str,
                         dst_arrow_col: str,
                         empty_value: Union[str, int, float],
                         shadow_fn: Callable,
                         progress_bar_pos: int = 0,
                         run: bool = False,
                         value_processor=None,
                         verbose=0,
                         batch_value_processor=None,
                         md5_col='md5',
                         allow_overwrite=False,
                         ):
    """
    Flush one column in csv files to arrow files.

    Parameters
    ----------
    src_files: list
        src file list
    arrow_list: list
        arrow file list
    src_csv_col: str
        Data column name in src file
    dst_arrow_col: str
        Column name to write to arrow
    empty_value: str, int, float
        Default value if the value is empty
    progress_bar_pos: int
        tqdm position
    shadow_fn: callable
        Function to generate new arrow file name
    run: bool
        If True, run without asking
    value_processor: callable
        Process value before writing to arrow
    """
    if shadow_fn is None:
        raise ValueError("shadow_fn is None")
    if not isinstance(src_files, (list, ConstValueDict, dict)):
        raise ValueError("src_files is not a list or ConstValueDict or dict")
    if not isinstance(arrow_list, list):
        raise ValueError("arrow_list is not a list")

    pid = os.getpid()

    md5_to_value = {}
    if src_files is None:
        # 创建一个空字典. 这是为了补全所有的 shadow arrows
        pass
    elif isinstance(src_files, ConstValueDict):
        md5_to_value = src_files
    elif isinstance(src_files, dict):
        md5_to_value = src_files
    else:
        src_files = list(src_files)
        src_type = Path(src_files[0]).suffix
        logger.info(f'[{progress_bar_pos}] Reading {len(src_files)} src files...')
        pbar_kwargs = dict(position=progress_bar_pos, total=len(src_files), desc=f'[{progress_bar_pos}]')
        if src_type == '.csv':
            # 从 csv 中读取需要的列, 需要md5和源列
            for cid, csv_file in tqdm(enumerate(src_files, start=1), **pbar_kwargs):
                try:
                    if verbose:
                        logger.info(f"[{progress_bar_pos}] {cid}. {csv_file}")
                    md5_to_value.update(pd.read_csv(csv_file, header=0).set_index(md5_col)[src_csv_col].to_dict())
                except KeyboardInterrupt as e:
                    raise e
                except pd.errors.EmptyDataError:
                    logger.info(f"[{progress_bar_pos}] EmptyDataError: {csv_file}")
        elif src_type == '.txt':
            for cid, txt_file in enumerate(src_files, start=1):
                logger.info(f"[{progress_bar_pos}] {cid}. {txt_file}")
                try:
                    with Path(txt_file).open('r') as f:
                        for key in tqdm(f, **pbar_kwargs):
                            # strip 去掉换行符
                            md5_to_value[key.strip()] = 1
                except KeyboardInterrupt as e:
                    raise e
                except pd.errors.EmptyDataError:
                    logger.info(f"[{progress_bar_pos}] EmptyDataError: {txt_file}")
        elif src_type == '.arrow':
            # 从 arrow 中读取需要的列, 需要md5和源列
            pbar = tqdm(enumerate(src_files, start=1), **pbar_kwargs)
            memory = psutil.Process(pid).memory_info().rss / (1024 ** 3)
            for cid, arrow_file in pbar:
                try:
                    if verbose:
                        logger.info(f"[{progress_bar_pos}] {cid}. {arrow_file}")
                    table = get_table(arrow_file)
                    md5_to_value.update(table.select([md5_col, src_csv_col]).to_pandas().set_index(md5_col)[src_csv_col].to_dict())
                    mem_diff = psutil.Process(pid).memory_info().rss / (1024 ** 3) - memory
                    mem_required = (len(src_files) - cid) * (mem_diff / cid)
                    pbar.set_description(f"[{progress_bar_pos}] [{cid}/{len(src_files)}] Memory consumption: {mem_diff:.1f} GB, memory required: {mem_required:.1f} GB")
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    logger.error(f"[{progress_bar_pos}] Error: {e}. Arrow file: {arrow_file}")
        elif src_type == '.pkl':
            # 从 pkl 文件中读取一个字典, key为md5, value为源的值
            for pkl_file in tqdm(src_files, **pbar_kwargs):
                with Path(pkl_file).open('rb') as f:
                    md5_to_value.update(pickle.load(f))
        elif src_type == '.parquet':
            # 从 parquet 文件中读取需要的列, 需要md5和源列
            for parquet_file in tqdm(src_files, **pbar_kwargs):
                table = pq.read_table(parquet_file)
                section = 5000
                logger.info(f"{parquet_file} has total length: {len(table)}, "
                            f"split into {(len(table) + section - 1) // section} sections.")
                for i in tqdm(range(0, len(table), section)):
                    sub_table = table.slice(offset=i, length=min(section, len(table) - i))
                    md5_to_value.update(sub_table.to_pandas().set_index(md5_col)[src_csv_col].to_dict())
        else:
            raise ValueError(f"source type `{src_type}` not supported")
    logger.info(f'[{progress_bar_pos}] Reading done! Total values: {len(md5_to_value):,}. '
                f'Write to {len(arrow_list)} arrows in column `{dst_arrow_col}`:')

    # Show some examples
    for i, (k, v) in enumerate(md5_to_value.items()):
        if i < 5:
            print(f"[{progress_bar_pos}] {k}: {value_processor(v) if value_processor is not None else v}")
        else:
            break

    if run:
        dry_run = False
    else:
        dry_run = input("dry run? [Y/n] ")
        if dry_run in ['y', 'Y', '', None]:
            dry_run = True
        else:
            dry_run = False

    total_count = 0
    pbar = tqdm(arrow_list, position=progress_bar_pos)
    error_names = []
    for ai, arrow_name in enumerate(pbar):
        try:
            # 获取新 arrow 的名称
            new_arrow_name = shadow_fn(arrow_name)

            # 读取 table
            table = get_table(arrow_name)

            md5s = table[md5_col].to_pylist()
            count = 0
            new_column = []
            for md5 in md5s:
                if md5 in md5_to_value:
                    val = md5_to_value[md5]
                    count += 1
                    total_count += 1
                else:
                    val = empty_value
                if val != empty_value and value_processor is not None:
                    val = value_processor(val)
                new_column.append(val)

            # 如果 shadow arrow 存在且没有更新数据, 则跳过以节省写入时间
            if count == 0 and Path(new_arrow_name).exists():
                continue

            pbar.set_description(f'[{progress_bar_pos:>2d}] Count: {count:5d}, '
                                 f'Total count: {total_count:,}, {new_arrow_name[-80:]}')

            if batch_value_processor is not None:
                new_column = batch_value_processor(new_column)

            # 查看目标 arrow 是否存在
            new_table = None
            if Path(new_arrow_name).exists():
                new_table = get_table(new_arrow_name)
                if len(new_table) != len(table):
                    logger.info(f'[{progress_bar_pos}] Warning: Length not equal: {len(new_table)} vs {len(table)}')
                    new_table = None

            if new_table is None:
                # 创建一个空表, 包含 md5 列
                new_table = pa.Table.from_pylist([{
                    md5_col: md5,
                } for md5 in md5s])

            # 如果 col 列存在, 则取出来, 并把新的值更新进去
            if dst_arrow_col in new_table.column_names:
                ci = new_table.column_names.index(dst_arrow_col)
                values = new_table[dst_arrow_col].to_pylist()
                # 用 new_column 中的值替换原来的值, 如果 new_column 中的值为空, 则不替换
                new_column = [new_v if new_v != empty_value else old_v for old_v, new_v in zip(values, new_column)]
                # 删除原来的列
                new_table = new_table.remove_column(ci)

            # 添加新列
            new_table = new_table.append_column(dst_arrow_col, pa.array(new_column))
            # 保存
            if not dry_run:
                to_table(new_table, new_arrow_name, allow_overwrite=allow_overwrite)

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            error_names.append(arrow_name)
            logger.info(f"[{progress_bar_pos}] {type(e)}: {e}. Arrow name: {arrow_name}")

    if error_names:
        now = datetime.today().strftime("%y%m%d_%H%M%S")
        with Path(f'{now}_csv_to_arrow_error_names_{progress_bar_pos}.txt').open('w') as f:
            f.write('\n'.join(error_names))

    return error_names


def error_handler(e):
    logger.error(f"{type(e).__name__}: {e}")


def p_csv_column_to_arrow(csv_files, arrow_list, kwds, num_proc=1):
    logger.info(f"Shadow fn: {arrow_list[0]} --> {kwds['shadow_fn'](arrow_list[0])}")
    if num_proc == 1:
        _csv_column_to_arrow(csv_files, arrow_list, **kwds)
    else:
        p = Pool(num_proc)
        results = []
        for i in range(num_proc):
            worker_arrow_list = arrow_list[i::num_proc]
            worker_args = (csv_files, worker_arrow_list)
            worker_kwds = kwds.copy()
            worker_kwds['progress_bar_pos'] = i
            print(f"Worker {i}: {len(worker_arrow_list)} arrows")
            results.append(
                p.apply_async(_csv_column_to_arrow,
                              args=worker_args,
                              kwds=worker_kwds,
                              error_callback=error_handler,
                              )
            )
        p.close()
        p.join()


if __name__ == "__main__":
    csv_token_to_arrow("/apdcephfs_nj10/share_301739632/ckczzjzhang/code_data/rank_*_iter_*.csv")
