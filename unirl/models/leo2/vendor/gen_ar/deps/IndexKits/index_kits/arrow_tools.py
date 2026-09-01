import enum
from pathlib import Path

import pyarrow as pa

# =========================
#  Read Arrow Tables
# =========================

def get_table(arrow_file):
    """
    Read an arrow file and return an arrow table.
    """
    return pa.ipc.RecordBatchFileReader(pa.memory_map(str(arrow_file), "r")).read_all()


class ColumnStatus(enum.Enum):
    SUCCESS = 0
    ERROR = 1


def get_columns(arrow_file, columns, shadows=None, shadow_fn_dict=None, as_py=True, verbose=0):
    data = []
    cache = {}
    input_str = isinstance(columns, str)
    if input_str:
        columns = [columns]
    if shadows is None:
        shadows = [col.split('@')[1] if '@' in col else None for col in columns]
    elif isinstance(shadows, str):
        shadows = [shadows]
    else:
        assert len(shadows) == len(columns), (
            f"Length of shadows should be the same as columns, but got {len(shadows)} and {len(columns)}.")
    columns = [col.split('@')[0] for col in columns]

    for col, shadow in zip(columns, shadows):
        if shadow is None:
            cur_arrow_file = arrow_file
        else:
            cur_arrow_file = shadow_fn_dict[shadow](arrow_file)
        if cur_arrow_file not in cache:
            cache[cur_arrow_file] = get_table(cur_arrow_file)
        table = cache[cur_arrow_file]
        try:
            values = table[col]
            if as_py:
                if verbose > 0:
                    print(f"Convert to pylist | {arrow_file}:{col} | Total {len(values)} samples.")
                values = values.to_pylist()
            status = ColumnStatus.SUCCESS
            msg = ""
        except Exception as e:
            values = [None] * len(table)
            status = ColumnStatus.ERROR
            msg = f"{e.__class__.__name__}: {e}\ncur_arrow_file: {cur_arrow_file}, shadow: {shadow}, column: {col}"
        data.append((values, status, msg))

    if input_str:
        return data[0]
    return data


# =========================
#  Write Arrow Tables
# =========================

def to_table(table, arrow_path, force=False):
    # We do not want to overwrite existing files
    if Path(arrow_path).exists() and not force:
        raise FileExistsError(f"Arrow file {arrow_path} already exists. Please remove it first or use a different path.")
    # Make sure the parent directory exists
    Path(arrow_path).parent.mkdir(parents=True, exist_ok=True)
    arrow_path_tmp = str(arrow_path) + '.tmp'

    # Save to a temporary file first, then rename it to the target file
    with pa.OSFile(arrow_path_tmp, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    # If the target file exists, remove it
    if Path(arrow_path).exists():
        Path(arrow_path).unlink()
    # Rename the temporary file to the target file
    Path(arrow_path_tmp).rename(arrow_path)


def dataframe_to_table(dataframe, save_path, force=False):
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    to_table(table, save_path, force=force)


df_to_table = dataframe_to_table    # alias


def pydict_to_table(pydict, save_path, force=False):
    table = pa.Table.from_pydict(pydict)
    to_table(table, save_path, force=force)
